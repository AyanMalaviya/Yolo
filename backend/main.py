import threading, time, cv2, json, logging
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image as PILImage
from datetime import datetime
from detector import (
    load_yolo, state, state_lock,
    WEAPON_CLASSES, ProximityTracker,
    push_alert, pad_crop,
    VLM_COOLDOWN_SEC, RED_HOLD_SEC, RED_CONFIDENCE, LINE_POSITION
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Stream settings ───────────────────────────────────────────────────────────
STREAM_WIDTH        = 640
STREAM_HEIGHT       = 480
STREAM_FPS          = 30
PROCESS_EVERY       = 2      # run YOLO every Nth frame

# ── VLM timing ────────────────────────────────────────────────────────────────
SCENE_INTERVAL          = 8.0    # passive scene every N seconds
NEW_PERSON_COOLDOWN     = 15.0   # re-describe same person after N seconds
VLM_COOLDOWN            = 4.0
COUNT_CHANGE_DELAY      = 2.0    # wait N seconds after count change before VLM

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Extend shared state ───────────────────────────────────────────────────────
state["scene_description"] = "Waiting for first analysis..."
state["detection_summary"] = ""
state["person_log"]        = []
state["person_count"]      = 0   # live tracked person count

# ── Load models ───────────────────────────────────────────────────────────────
yolo_model    = load_yolo()
vlm_model     = None
vlm_processor = None

def load_vlm():
    global vlm_model, vlm_processor
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        log.info("Loading SmolVLM2-2.2B (4-bit)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        vlm_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            quantization_config=bnb,
            device_map="cuda",
            _attn_implementation="eager"
        )
        vlm_model.eval()
        log.info("SmolVLM2-2.2B ready.")
        return vlm_model, vlm_processor
    except Exception as e:
        log.warning(f"VLM load failed: {e}. Running YOLO-only mode.")
        return None, None

vlm_model, vlm_processor = load_vlm()

# ── Engine state ──────────────────────────────────────────────────────────────
engine = {
    "running":    False,
    "source":     None,
    "thread":     None,
    "frame":      None,
    "frame_lock": threading.Lock(),
}

# ── Prompts ───────────────────────────────────────────────────────────────────
PERSON_PROMPT = """Describe this person in one sentence for identification.
Include: approximate age range, gender presentation, clothing colors and type, any items they are carrying or wearing.
Example: Adult male, 20s, red hoodie, blue jeans, carrying a black backpack.
Reply with ONLY the description sentence — no extra text."""

SCENE_PROMPT = """Describe this surveillance footage in one sentence.
Include number of people, what they are doing, and any notable objects.
Example: Two people are walking near a door, one is holding a bag.
Reply with ONLY the description sentence — no extra text."""

COUNT_CHANGE_PROMPT = """The number of people in this surveillance area just changed.
Describe the current scene in one sentence.
Include: how many people are visible, what they are doing, anything notable.
Example: Three people are now visible, one just entered from the left carrying a bag.
Reply with ONLY the description sentence — no extra text."""

THREAT_PROMPT = """Look at this image carefully.
Is there a violent attack, weapon being used aggressively, or forced break-in happening?
Reply with ONLY this format:
THREAT: yes or no. One sentence reason.
Example: THREAT: no. Two people are talking normally.
Example: THREAT: yes. A person is swinging an axe at another person."""

# ── VLM helpers ───────────────────────────────────────────────────────────────
def resize_for_vlm(crop_bgr, max_px=512):
    h, w  = crop_bgr.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        return cv2.resize(crop_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return crop_bgr

def call_vlm_text(crop_bgr, prompt):
    if vlm_model is None or vlm_processor is None:
        return ""
    try:
        crop  = resize_for_vlm(crop_bgr)
        image = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},                  # placeholder only
                {"type": "text", "text": prompt}
            ]
        }]

        # Step 1 — build text prompt from template (no tokenize here)
        text_prompt = vlm_processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Step 2 — pass image separately to processor
        inputs = vlm_processor(
            text=text_prompt,
            images=[image],                         # ← image passed here
            return_tensors="pt"
        ).to(vlm_model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=80)

        result = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return lines[-1] if lines else result
    except Exception as e:
        log.warning(f"VLM error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()

def call_vlm_threat(crop_bgr):
    try:
        raw       = call_vlm_text(crop_bgr, THREAT_PROMPT)
        is_threat = "threat: yes" in raw.lower()
        desc      = raw.split(".", 1)[-1].strip() if "." in raw else raw
        return {
            "threat":      is_threat,
            "type":        "assault" if is_threat else "none",
            "confidence":  "high" if is_threat else "low",
            "description": desc or raw
        }
    except Exception as e:
        return {"threat": False, "type": "none", "confidence": "low", "description": str(e)}

# ── Detection engine ──────────────────────────────────────────────────────────
def run_engine(source):
    prox           = ProximityTracker()
    vlm_thread     = None
    last_vlm_time  = 0.0
    described_ids  = {}
    frame_count    = 0
    last_annotated = None

    # Person count tracking
    prev_person_count  = 0
    count_changed_at   = None   # timestamp when count last changed

    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)

    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        engine["running"] = False
        return

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info(f"Stream opened → {source} | {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

    # ── VLM thread targets ────────────────────────────────────────────────────
    def threat_vlm(crop, reason_prefix):
        result     = call_vlm_threat(crop)
        is_threat  = result.get("threat", False)
        confidence = result.get("confidence", "low")
        if is_threat and confidence in RED_CONFIDENCE:
            push_alert("RED",    f"{reason_prefix} — confirmed: {result.get('type')}", result)
        else:
            push_alert("YELLOW", f"{reason_prefix} — not confirmed", result)

    def person_vlm(track_id, crop):
        desc = call_vlm_text(crop, PERSON_PROMPT)
        if not desc:
            return
        entry = {
            "time":        datetime.now().strftime("%H:%M:%S"),
            "track_id":    track_id,
            "description": desc,
        }
        with state_lock:
            state["person_log"].append(entry)
            state["person_log"] = state["person_log"][-50:]
        log.info(f"[PERSON] ID#{track_id}: {desc}")

    def scene_vlm(crop, prompt=None):
        desc = call_vlm_text(crop, prompt or SCENE_PROMPT)
        if desc:
            with state_lock:
                state["scene_description"] = desc
            log.info(f"[SCENE] {desc}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while engine["running"]:
        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1
        now = time.time()

        # Skip YOLO on non-processed frames
        if frame_count % PROCESS_EVERY != 0:
            if last_annotated is not None:
                _, buf = cv2.imencode(".jpg", last_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with engine["frame_lock"]:
                    engine["frame"] = buf.tobytes()
            continue

        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                   conf=0.4, imgsz=640, verbose=False)

        people_boxes = {}
        yolo_trigger = None
        class_counts = {}

        for box in results[0].boxes:
            if box.id is None:
                continue
            cls_id   = int(box.cls)
            track_id = int(box.id)
            xyxy     = box.xyxy[0].cpu().numpy()
            conf     = float(box.conf)
            name     = yolo_model.names[cls_id]
            class_counts[name] = class_counts.get(name, 0) + 1

            if cls_id == 0:
                people_boxes[track_id] = xyxy

                # New person VLM description
                last_described = described_ids.get(track_id, 0)
                if (vlm_model is not None and
                        now - last_described > NEW_PERSON_COOLDOWN and
                        (vlm_thread is None or not vlm_thread.is_alive())):
                    described_ids[track_id] = now
                    person_crop = pad_crop(frame, xyxy, pad=40)
                    if person_crop.size > 0:
                        vlm_thread = threading.Thread(
                            target=person_vlm,
                            args=(track_id, person_crop.copy()),
                            daemon=True
                        )
                        vlm_thread.start()

            if cls_id in WEAPON_CLASSES and yolo_trigger is None:
                yolo_trigger = (f"Weapon: {WEAPON_CLASSES[cls_id]}", pad_crop(frame, xyxy))

        # ── Person count change detection ─────────────────────────────────────
        current_count = len(people_boxes)
        with state_lock:
            state["person_count"] = current_count

        if current_count != prev_person_count:
            direction = "entered" if current_count > prev_person_count else "left"
            log.info(f"[COUNT] {prev_person_count} → {current_count} person(s) | someone {direction}")
            count_changed_at  = now
            prev_person_count = current_count

        # Fire VLM 2 seconds after count change stabilizes
        with state_lock:
            last_vlm = state["last_vlm_time"]

        if (count_changed_at is not None and
                now - count_changed_at >= COUNT_CHANGE_DELAY and
                now - last_vlm >= VLM_COOLDOWN and
                vlm_model is not None and
                (vlm_thread is None or not vlm_thread.is_alive())):
            with state_lock:
                state["last_vlm_time"] = now
            count_changed_at = None   # reset so it doesn't fire again
            vlm_thread = threading.Thread(
                target=scene_vlm,
                args=(frame.copy(), COUNT_CHANGE_PROMPT),
                daemon=True
            )
            vlm_thread.start()
            log.info(f"[COUNT VLM] Triggered — {current_count} person(s) in frame")

        # Proximity check
        if yolo_trigger is None:
            prox_result = prox.update(people_boxes)
            if prox_result:
                pair_ids, mb = prox_result
                yolo_trigger = (f"Sustained contact — IDs {pair_ids}", pad_crop(frame, mb))

        with state_lock:
            cur_alert = state["alert"]
            last_red  = state["last_red_time"]
            last_vlm  = state["last_vlm_time"]

        # Threat VLM dispatch
        if yolo_trigger:
            reason, crop = yolo_trigger
            if (vlm_model is not None and
                    now - last_vlm >= VLM_COOLDOWN and
                    (vlm_thread is None or not vlm_thread.is_alive())):
                with state_lock:
                    state["last_vlm_time"] = now
                vlm_thread = threading.Thread(
                    target=threat_vlm, args=(crop.copy(), reason), daemon=True)
                vlm_thread.start()
            if cur_alert == "CLEAR":
                push_alert("YELLOW", reason)
        else:
            if (cur_alert != "CLEAR" and
                    now - last_red >= RED_HOLD_SEC and
                    (vlm_thread is None or not vlm_thread.is_alive())):
                push_alert("CLEAR", "")

        # Passive scene description (fallback if no count change for a while)
        if (vlm_model is not None and
                now - last_vlm >= SCENE_INTERVAL and
                count_changed_at is None and
                not yolo_trigger and
                (vlm_thread is None or not vlm_thread.is_alive())):
            with state_lock:
                state["last_vlm_time"] = now
            vlm_thread = threading.Thread(
                target=scene_vlm, args=(frame.copy(),), daemon=True)
            vlm_thread.start()

        # Detection summary
        summary = ", ".join(f"{v}× {k}" for k, v in class_counts.items()) or "Nothing detected"
        with state_lock:
            state["detection_summary"] = summary

        # ── Overlay ───────────────────────────────────────────────────────────
        annotated = results[0].plot()
        h, w      = annotated.shape[:2]

        with state_lock:
            alert_now  = state["alert"]
            reason_now = state["reason"]
            desc_now   = state["scene_description"]
            count_now  = state["person_count"]

        color = {"CLEAR":(30,160,30), "YELLOW":(0,180,255), "RED":(0,0,210)}.get(alert_now, (60,60,60))
        cv2.rectangle(annotated, (0,0), (w, 52), color, -1)
        cv2.putText(annotated, f"[{alert_now}]  {reason_now[:72]}",
                    (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,0,0), 2)

        # Person count badge top-right
        cv2.rectangle(annotated, (w-160, 0), (w, 52), (30,30,30), -1)
        cv2.putText(annotated, f"People: {count_now}",
                    (w-150, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)

        # Scene description bottom
        if desc_now:
            cv2.putText(annotated, f"Scene: {desc_now[:90]}",
                        (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200,255,200), 1)

        last_annotated = annotated.copy()
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with engine["frame_lock"]:
            engine["frame"] = buf.tobytes()

    cap.release()
    engine["running"] = False
    push_alert("CLEAR", "")

# ── MJPEG stream ──────────────────────────────────────────────────────────────
def mjpeg_generator():
    while True:
        with engine["frame_lock"]:
            frame = engine["frame"]
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)

# ── Engine helpers ────────────────────────────────────────────────────────────
def _stop_engine():
    engine["running"] = False
    if engine["thread"] and engine["thread"].is_alive():
        engine["thread"].join(timeout=3)
    with state_lock:
        state["alert"]           = "CLEAR"
        state["reason"]          = ""
        state["vlm_description"] = ""
        state["person_count"]    = 0
    with engine["frame_lock"]:
        engine["frame"] = None

def _start_engine(source):
    _stop_engine()
    engine["source"]  = source
    engine["running"] = True
    engine["thread"]  = threading.Thread(target=run_engine, args=(source,), daemon=True)
    engine["thread"].start()

# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start/camera")
def start_camera(index: int = 0):
    _start_engine(index)
    return {"status": "started", "source": f"camera:{index}"}

@app.post("/start/video")
async def start_video(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    _start_engine(str(save_path))
    return {"status": "started", "source": file.filename}

@app.post("/start/path")
def start_path(path: str):
    _start_engine(path)
    return {"status": "started", "source": path}

@app.post("/stop")
def stop():
    _stop_engine()
    return {"status": "stopped"}

@app.get("/status")
def get_status():
    with state_lock:
        return {
            "running":           engine["running"],
            "source":            str(engine["source"]),
            "alert":             state["alert"],
            "reason":            state["reason"],
            "description":       state["vlm_description"],
            "threat_type":       state["threat_type"],
            "scene_description": state["scene_description"],
            "detection_summary": state["detection_summary"],
            "person_count":      state["person_count"],
        }

@app.get("/alerts")
def get_alerts():
    with state_lock:
        return state["alert_log"]

@app.get("/persons")
def get_persons():
    with state_lock:
        return state["person_log"]

@app.get("/logs")
def get_logs():
    log_path = Path("logs/people_log.json")
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return []

@app.get("/vram")
def get_vram():
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "total_gb":     round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb":  round(reserved, 2),
            "free_gb":      round(total - reserved, 2),
            "usage_pct":    round((reserved / total) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}
