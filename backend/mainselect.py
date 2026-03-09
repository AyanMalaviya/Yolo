import threading
import time
import cv2
import json
import logging
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image as PILImage
from datetime import datetime
from detector import (
    load_yolo, load_weapon_model, load_vlm,
    run_vlm, run_weapon_inference, draw_weapon_boxes,
    state, state_lock,
    WEAPON_CLASSES, ProximityTracker,
    push_alert, pad_crop,
    VLM_COOLDOWN_SEC, RED_HOLD_SEC, RED_CONFIDENCE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Stream settings ───────────────────────────────────────────────────────────
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
STREAM_FPS    = 30

# ── VLM timing defaults ───────────────────────────────────────────────────────
NEW_PERSON_COOLDOWN = 30.0
VLM_COOLDOWN        = 8.0
COUNT_CHANGE_DELAY  = 2.0
VLM_THREAD_TIMEOUT  = 20.0   # abandon hung VLM threads after 20s

# ── Detection modes ───────────────────────────────────────────────────────────
VALID_MODES = {"both", "yolo_only", "vlm_only"}

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
state["person_count"]      = 0
state["weapon_detections"] = []
state["source_fps"]        = 0.0
state["vlm_enabled"]       = True
state["detection_mode"]    = "both"
state["vlm_interval"]      = 15.0

# ── Load models ───────────────────────────────────────────────────────────────
yolo_model            = load_yolo()
weapon_model          = load_weapon_model()
vlm_model, vlm_processor = load_vlm()

# ── GPU optimizations ────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85)
    yolo_model.to("cuda")
    weapon_model.to("cuda")
    yolo_model.model.half()
    weapon_model.model.half()
    log.info(
        f"[GPU] {torch.cuda.get_device_name(0)} | "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total"
    )

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


# ── VLM helper ────────────────────────────────────────────────────────────────
def call_vlm_text(crop_bgr: np.ndarray, prompt: str) -> str:
    if vlm_model is None or vlm_processor is None:
        return ""
    try:
        image = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        h, w  = crop_bgr.shape[:2]
        scale = 512 / max(h, w)
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)))

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]
        text_prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs      = vlm_processor(
            text=text_prompt, images=[image], return_tensors="pt"
        ).to(vlm_model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            out = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=60)

        result = vlm_processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return lines[-1] if lines else result
    except Exception as e:
        log.warning(f"VLM text error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()


# ── VLM thread helper — stamps start time for watchdog ───────────────────────
def _start_vlm_thread(target, args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t._start_time = time.time()
    t.start()
    return t


# ── Detection engine ──────────────────────────────────────────────────────────
def run_engine(source):
    prox              = ProximityTracker()
    vlm_thread        = None
    described_ids     = {}
    prev_person_count = 0
    count_changed_at  = None
    frame_count       = 0

    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          STREAM_FPS)

    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        engine["running"] = False
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0 or actual_fps > 120:
        actual_fps = 30.0
    frame_delay = 1.0 / actual_fps

    with state_lock:
        state["source_fps"] = round(actual_fps, 2)

    log.info(
        f"Stream opened → {source} | "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{actual_fps:.1f}fps"
    )

    # ── VLM thread targets ────────────────────────────────────────────────────
    def threat_vlm(crop, reason_prefix):
        result     = run_vlm(crop, vlm_model, vlm_processor)
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
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1
        now = time.time()

        # ── Read all config flags once per frame ──────────────────────────────
        with state_lock:
            vlm_on   = state["vlm_enabled"]
            mode     = state["detection_mode"]
            vlm_ivl  = state["vlm_interval"]
            last_vlm = state["last_vlm_time"]

        use_vlm    = vlm_on and mode in ("both", "vlm_only")
        use_weapon = mode in ("both", "yolo_only")

        # ── VLM thread watchdog — abandon if hung > 20s ───────────────────────
        if (vlm_thread is not None and
                vlm_thread.is_alive() and
                hasattr(vlm_thread, "_start_time") and
                now - vlm_thread._start_time > VLM_THREAD_TIMEOUT):
            log.warning("[VLM] Thread hung >20s — abandoning + clearing VRAM")
            vlm_thread = None
            torch.cuda.empty_cache()

        # ── If switched to yolo_only mid-session, abandon stale VLM thread ───
        if not use_vlm and vlm_thread is not None and vlm_thread.is_alive():
            vlm_thread.join(timeout=0.05)
            if not vlm_thread.is_alive():
                vlm_thread = None

        # ── YOLO26n tracking (always runs) ────────────────────────────────────
        track_results = yolo_model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.4, imgsz=640, verbose=False
        )

        # ── Weapons model (skipped in vlm_only) ───────────────────────────────
        weapon_detections = []
        weapon_trigger    = None

        if use_weapon:
            weapon_detections, weapon_trigger, _ = run_weapon_inference(weapon_model, frame)

        with state_lock:
            state["weapon_detections"] = weapon_detections

        people_boxes = {}
        yolo_trigger = weapon_trigger if mode == "both" else None
        class_counts = {}

        # ── Process YOLO26n boxes ─────────────────────────────────────────────
        for box in track_results[0].boxes:
            if box.id is None:
                continue
            cls_id   = int(box.cls)
            track_id = int(box.id)
            xyxy     = box.xyxy[0].cpu().numpy()
            conf     = float(box.conf)
            name     = yolo_model.names[cls_id]
            class_counts[name] = class_counts.get(name, 0) + 1

            if cls_id == 0:   # person
                people_boxes[track_id] = xyxy

                # Person VLM description
                last_described = described_ids.get(track_id, 0)
                if (use_vlm and
                        vlm_model is not None and
                        now - last_described > NEW_PERSON_COOLDOWN and
                        (vlm_thread is None or not vlm_thread.is_alive())):
                    described_ids[track_id] = now
                    person_crop = pad_crop(frame, xyxy, pad=40)
                    if person_crop.size > 0:
                        vlm_thread = _start_vlm_thread(
                            person_vlm, (track_id, person_crop.copy()))

        # ── Person count change ───────────────────────────────────────────────
        current_count = len(people_boxes)
        with state_lock:
            state["person_count"] = current_count

        if current_count != prev_person_count:
            log.info(f"[COUNT] {prev_person_count} → {current_count}")
            count_changed_at  = now
            prev_person_count = current_count

        # Count change VLM
        if (use_vlm and
                count_changed_at is not None and
                now - count_changed_at >= COUNT_CHANGE_DELAY and
                now - last_vlm >= VLM_COOLDOWN and
                vlm_model is not None and
                (vlm_thread is None or not vlm_thread.is_alive())):
            with state_lock:
                state["last_vlm_time"] = now
            count_changed_at = None
            vlm_thread = _start_vlm_thread(
                scene_vlm, (frame.copy(), COUNT_CHANGE_PROMPT))
            log.info(f"[COUNT VLM] triggered — {current_count} person(s)")

        # ── Proximity check (both mode only) ──────────────────────────────────
        if mode == "both" and yolo_trigger is None:
            prox_result = prox.update(people_boxes)
            if prox_result:
                pair_ids, mb = prox_result
                yolo_trigger = (f"Sustained contact — IDs {pair_ids}", pad_crop(frame, mb))

        with state_lock:
            cur_alert = state["alert"]
            last_red  = state["last_red_time"]
            last_vlm  = state["last_vlm_time"]

        # ── Alert logic ───────────────────────────────────────────────────────
        if mode == "both" and yolo_trigger:
            reason, crop = yolo_trigger
            if (use_vlm and
                    vlm_model is not None and
                    now - last_vlm >= VLM_COOLDOWN and
                    (vlm_thread is None or not vlm_thread.is_alive())):
                with state_lock:
                    state["last_vlm_time"] = now
                vlm_thread = _start_vlm_thread(threat_vlm, (crop.copy(), reason))
            if cur_alert == "CLEAR":
                push_alert("YELLOW", reason)

        elif mode == "yolo_only" and weapon_trigger:
            # Raise YELLOW from weapon without VLM confirm
            reason, _ = weapon_trigger
            if cur_alert == "CLEAR":
                push_alert("YELLOW", reason)

        else:
            # No trigger — clear alert after hold time expires
            if (cur_alert != "CLEAR" and
                    now - last_red >= RED_HOLD_SEC and
                    (vlm_thread is None or not vlm_thread.is_alive())):
                push_alert("CLEAR", "")

        # ── Passive scene VLM (configurable interval) ─────────────────────────
        if (use_vlm and
                vlm_model is not None and
                now - last_vlm >= vlm_ivl and
                count_changed_at is None and
                not yolo_trigger and
                (vlm_thread is None or not vlm_thread.is_alive())):
            with state_lock:
                state["last_vlm_time"] = now
            vlm_thread = _start_vlm_thread(scene_vlm, (frame.copy(),))

        # ── Detection summary ─────────────────────────────────────────────────
        weapon_names = [d["label"] for d in weapon_detections]
        person_str   = f"{len(people_boxes)} person(s)" if people_boxes else ""
        weapon_str   = f"⚠️ {', '.join(weapon_names)}" if weapon_names else ""
        other_str    = ", ".join(
            f"{v}× {k}" for k, v in class_counts.items()
            if k != "person" and k not in weapon_names
        )
        summary = " | ".join(p for p in [person_str, weapon_str, other_str] if p) or "Nothing detected"
        with state_lock:
            state["detection_summary"] = summary

        # ── Overlay ───────────────────────────────────────────────────────────
        annotated = track_results[0].plot()
        if weapon_detections:
            annotated = draw_weapon_boxes(annotated, weapon_detections)
        h_f, w_f = annotated.shape[:2]

        with state_lock:
            alert_now  = state["alert"]
            reason_now = state["reason"]
            desc_now   = state["scene_description"]
            count_now  = state["person_count"]
            mode_now   = state["detection_mode"]
            vlm_on_now = state["vlm_enabled"]

        # Alert banner
        color = {"CLEAR": (30,160,30), "YELLOW": (0,180,255), "RED": (0,0,210)}.get(alert_now, (60,60,60))
        cv2.rectangle(annotated, (0, 0), (w_f, 52), color, -1)
        cv2.putText(annotated, f"[{alert_now}]  {reason_now[:72]}",
                    (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)

        # Top-right badge
        badge = f"People:{count_now}  [{mode_now.upper()}]"
        cv2.rectangle(annotated, (w_f-260, 0), (w_f, 52), (30, 30, 30), -1)
        cv2.putText(annotated, badge,
                    (w_f-250, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        # Bottom-right indicators
        cv2.putText(annotated, f"{actual_fps:.0f}fps",
                    (w_f-55, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        vlm_active = vlm_on_now and mode_now != "yolo_only"
        vlm_label  = "VLM:ON" if vlm_active else "VLM:OFF"
        vlm_color  = (160, 100, 255) if vlm_active else (80, 80, 80)
        cv2.putText(annotated, vlm_label,
                    (w_f-75, h_f-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, vlm_color, 1)

        # Bottom-left scene description
        if desc_now and use_vlm:
            cv2.putText(annotated, f"Scene: {desc_now[:90]}",
                        (8, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1)
        elif not use_vlm:
            cv2.putText(annotated, f"Mode: {mode_now} — VLM inactive",
                        (8, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with engine["frame_lock"]:
            engine["frame"] = buf.tobytes()

        # FPS pacing
        elapsed = time.time() - loop_start
        if frame_delay - elapsed > 0:
            time.sleep(frame_delay - elapsed)

    cap.release()
    engine["running"] = False
    with state_lock:
        state["weapon_detections"] = []
        state["source_fps"]        = 0.0
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
        state["alert"]             = "CLEAR"
        state["reason"]            = ""
        state["person_count"]      = 0
        state["weapon_detections"] = []
        state["source_fps"]        = 0.0
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
            "weapon_detections": state["weapon_detections"],
            "source_fps":        state["source_fps"],
            "vlm_enabled":       state["vlm_enabled"],
            "detection_mode":    state["detection_mode"],
            "vlm_interval":      state["vlm_interval"],
        }


@app.get("/alerts")
def get_alerts():
    with state_lock:
        return state["alert_log"]


@app.get("/persons")
def get_persons():
    with state_lock:
        return state["person_log"]


@app.get("/vram")
def get_vram():
    try:
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return {
            "total_gb":     round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb":  round(reserved, 2),
            "free_gb":      round(total - reserved, 2),
            "usage_pct":    round((reserved / total) * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/weapon_classes")
def get_weapon_classes():
    return {"classes": list(WEAPON_CLASSES.values())}


@app.post("/vlm/enable")
def vlm_enable():
    with state_lock:
        state["vlm_enabled"] = True
    log.info("[VLM] Enabled")
    return {"vlm_enabled": True}


@app.post("/vlm/disable")
def vlm_disable():
    with state_lock:
        state["vlm_enabled"] = False
    log.info("[VLM] Disabled")
    return {"vlm_enabled": False}


@app.post("/mode/{mode}")
def set_mode(mode: str):
    if mode not in VALID_MODES:
        return {"error": f"Invalid mode. Choose from: {VALID_MODES}"}
    with state_lock:
        state["detection_mode"] = mode
    log.info(f"[MODE] → {mode}")
    return {"detection_mode": mode}


@app.post("/vlm/interval")
def set_vlm_interval(seconds: float):
    seconds = max(5.0, min(seconds, 120.0))
    with state_lock:
        state["vlm_interval"] = seconds
    log.info(f"[VLM] Interval → {seconds}s")
    return {"vlm_interval": seconds}
