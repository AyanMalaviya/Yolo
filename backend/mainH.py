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
    run_vlm, vlm_infer, run_weapon_inference, draw_weapon_boxes,
    state, state_lock,
    WEAPON_CLASSES, ProximityTracker,
    push_alert, pad_crop, get_vram_usage_pct,
    VLM_COOLDOWN_SEC, RED_HOLD_SEC, RED_CONFIDENCE,
    GPU_YOLO, GPU_VLM,
    _vlm_is_qwen,
)
import detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STREAM_WIDTH        = 1280     # upgraded from 640 — A4000 handles 1080p easily
STREAM_HEIGHT       = 720
STREAM_FPS          = 30
NEW_PERSON_COOLDOWN = 30.0
VLM_COOLDOWN        = 8.0
COUNT_CHANGE_DELAY  = 2.0
VLM_THREAD_TIMEOUT  = 20.0
WEAPON_MIN_FRAMES   = 3

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
state["mode_switching"]    = False

# ── Load models ───────────────────────────────────────────────────────────────
yolo_model               = load_yolo()
weapon_model             = load_weapon_model()
vlm_model, vlm_processor = load_vlm()

# ── After load_vlm() call ─────────────────────────────────────────────────────
if vlm_model is not None:
    try:
        model_type = getattr(vlm_model.config, "model_type", "")
        if "qwen3_vl" in model_type or "qwen2_5_vl" in model_type or "qwen2_vl" in model_type:
            detector._vlm_is_qwen = True
            log.info(f"[VLM] Qwen inference path active — model_type: {model_type}")
    except Exception:
        pass



# ── Log GPU layout ────────────────────────────────────────────────────────────
if torch.cuda.device_count() >= 2:
    for i in range(2):
        props = torch.cuda.get_device_properties(i)
        log.info(f"[GPU {i}] {props.name} | {props.total_memory/1024**3:.1f}GB")
    log.info(f"[GPU] YOLO+Weapon → {GPU_YOLO} | VLM → {GPU_VLM}")
else:
    log.warning("[GPU] Only 1 GPU found — running single-GPU mode")

# ── Engine state ──────────────────────────────────────────────────────────────
engine = {
    "running":    False,
    "source":     None,
    "thread":     None,
    "frame":      None,
    "frame_lock": threading.Lock(),
}

# ── Prompts ───────────────────────────────────────────────────────────────────
PERSON_PROMPT = """Look at this person in the surveillance image. If they are near door, Start with entered or exit depending on which side facing.

Write 2 short sentences:
1. What are they doing right now?
2. What is the most noticeable thing about them that would help identify them?

Incase of any lethal object in hand, start with alert."""

SCENE_PROMPT = """Describe this surveillance footage in one sentence.
Include number of people, what they are doing, and any notable objects.
Incase of any lethal object in hand, start with alert."""

COUNT_CHANGE_PROMPT = """The number of people in this surveillance area just changed.
Describe the current scene in one sentence. Incase of any lethal object in hand, start with alert."""



# ── VLM thread safety ─────────────────────────────────────────────────────────
def safe_vlm_call(target_fn, args: tuple):
    """
    No longer needs VRAM gating — VLM is on its own dedicated GPU 1.
    Still wraps for OOM safety just in case.
    """
    try:
        torch.cuda.empty_cache()
        target_fn(*args)
    except torch.cuda.OutOfMemoryError:
        log.error(f"[VLM] OOM on {GPU_VLM} — clearing cache")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(GPU_VLM)
    except Exception as e:
        log.warning(f"[VLM] safe_vlm_call error: {e}")


def _start_vlm_thread(target, args):
    def wrapped():
        safe_vlm_call(target, args)

    t = threading.Thread(target=wrapped, daemon=True)
    t._start_time = time.time()
    t.start()
    return t


# ── VLM text helper ───────────────────────────────────────────────────────────
def call_vlm_text(crop_bgr: np.ndarray, prompt: str) -> str:
    return vlm_infer(crop_bgr, prompt, vlm_model, vlm_processor, max_tokens=80)


# ── VLM offload/reload (mode switching) ──────────────────────────────────────
def offload_vlm_to_cpu():
    global vlm_model
    if vlm_model is None:
        return
    try:
        log.info("[MODE] Offloading VLM to CPU...")
        vlm_model = vlm_model.to("cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(GPU_VLM)
        log.info(f"[MODE] VLM offloaded. {GPU_VLM} free: "
                 f"{torch.cuda.memory_reserved(1)/1024**3:.1f}GB")
    except Exception as e:
        log.warning(f"[MODE] VLM offload failed: {e}")


def reload_vlm_to_gpu():
    global vlm_model
    if vlm_model is None:
        return
    try:
        log.info(f"[MODE] Reloading VLM to {GPU_VLM}...")
        vlm_model = vlm_model.to(GPU_VLM)
        torch.cuda.synchronize(GPU_VLM)
        log.info(f"[MODE] VLM reloaded. {GPU_VLM} used: "
                 f"{torch.cuda.memory_reserved(1)/1024**3:.1f}GB")
    except torch.cuda.OutOfMemoryError:
        log.error(f"[MODE] OOM reloading VLM to {GPU_VLM}")
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"[MODE] VLM reload failed: {e}")


# ── Detection engine ──────────────────────────────────────────────────────────
def run_engine(source):
    prox               = ProximityTracker()
    vlm_thread         = None
    described_ids      = {}
    prev_person_count  = 0
    count_changed_at   = None
    frame_count        = 0
    weapon_consecutive = 0

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
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {actual_fps:.1f}fps"
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

        try:
            frame_count += 1
            now = time.time()

            with state_lock:
                vlm_on       = state["vlm_enabled"]
                mode         = state["detection_mode"]
                vlm_ivl      = state["vlm_interval"]
                last_vlm     = state["last_vlm_time"]
                is_switching = state["mode_switching"]

            if is_switching:
                elapsed = time.time() - loop_start
                if frame_delay - elapsed > 0:
                    time.sleep(frame_delay - elapsed)
                continue

            use_vlm    = vlm_on and mode in ("both", "vlm_only")
            use_weapon = mode in ("both", "yolo_only")

            # ── Watchdog ───────────────────────────────────────────────────────
            if (vlm_thread is not None and
                    vlm_thread.is_alive() and
                    hasattr(vlm_thread, "_start_time") and
                    now - vlm_thread._start_time > VLM_THREAD_TIMEOUT):
                log.warning("[VLM] Thread hung >20s — abandoning")
                vlm_thread = None
                torch.cuda.empty_cache()

            if not use_vlm and vlm_thread is not None and vlm_thread.is_alive():
                vlm_thread.join(timeout=0.05)
                if not vlm_thread.is_alive():
                    vlm_thread = None

            # ── YOLO11l tracking on GPU 0 ──────────────────────────────────────
            track_results = yolo_model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=0.4, imgsz=1280,   # upgraded to 1280 — A4000 handles it
                verbose=False
            )

            # ── Weapons model on GPU 0 ─────────────────────────────────────────
            weapon_detections = []
            weapon_trigger    = None

            if use_weapon:
                weapon_detections, weapon_trigger, _ = run_weapon_inference(weapon_model, frame)

            if weapon_detections:
                weapon_consecutive += 1
            else:
                weapon_consecutive = 0

            confirmed_weapon = weapon_consecutive >= WEAPON_MIN_FRAMES

            with state_lock:
                state["weapon_detections"] = weapon_detections

            people_boxes      = {}
            yolo_trigger      = (weapon_trigger if (mode == "both"      and confirmed_weapon) else None)
            yolo_only_trigger = (weapon_trigger if (mode == "yolo_only" and confirmed_weapon) else None)
            class_counts      = {}

            # ── Process YOLO boxes ─────────────────────────────────────────────
            for box in track_results[0].boxes:
                if box.id is None:
                    continue
                cls_id   = int(box.cls)
                track_id = int(box.id)
                xyxy     = box.xyxy[0].cpu().numpy()
                name     = yolo_model.names[cls_id]
                class_counts[name] = class_counts.get(name, 0) + 1

                if cls_id == 0:
                    people_boxes[track_id] = xyxy

                    cx       = int((xyxy[0] + xyxy[2]) / 2 / 80)
                    cy_grid  = int((xyxy[1] + xyxy[3]) / 2 / 80)
                    pos_hash = cx * 1000 + cy_grid

                    last_entry      = described_ids.get(track_id, {"time": 0, "pos_hash": -1})
                    first_time      = last_entry["time"] == 0
                    time_ok         = now - last_entry["time"] > NEW_PERSON_COOLDOWN
                    moved_far       = last_entry["pos_hash"] != pos_hash
                    should_describe = first_time or (time_ok and moved_far)

                    if (use_vlm and vlm_model is not None and
                            should_describe and
                            (vlm_thread is None or not vlm_thread.is_alive())):
                        described_ids[track_id] = {"time": now, "pos_hash": pos_hash}
                        person_crop = pad_crop(frame, xyxy, pad=40)
                        if person_crop.size > 0:
                            vlm_thread = _start_vlm_thread(
                                person_vlm, (track_id, person_crop.copy()))

            # ── Person count change ────────────────────────────────────────────
            current_count = len(people_boxes)
            with state_lock:
                state["person_count"] = current_count

            if current_count != prev_person_count:
                log.info(f"[COUNT] {prev_person_count} → {current_count}")
                count_changed_at  = now
                prev_person_count = current_count

            if (use_vlm and count_changed_at is not None and
                    now - count_changed_at >= COUNT_CHANGE_DELAY and
                    now - last_vlm >= VLM_COOLDOWN and vlm_model is not None and
                    (vlm_thread is None or not vlm_thread.is_alive())):
                with state_lock:
                    state["last_vlm_time"] = now
                count_changed_at = None
                vlm_thread = _start_vlm_thread(
                    scene_vlm, (frame.copy(), COUNT_CHANGE_PROMPT))

            # ── Proximity check ────────────────────────────────────────────────
            if mode == "both" and yolo_trigger is None:
                prox_result = prox.update(people_boxes)
                if prox_result:
                    pair_ids, mb = prox_result
                    yolo_trigger = (f"Sustained contact — IDs {pair_ids}", pad_crop(frame, mb))

            with state_lock:
                cur_alert = state["alert"]
                last_red  = state["last_red_time"]
                last_vlm  = state["last_vlm_time"]

            # ── Alert logic ────────────────────────────────────────────────────
            if mode == "both" and yolo_trigger:
                reason, crop = yolo_trigger
                # No VRAM gate needed — VLM is on a dedicated GPU
                if (use_vlm and vlm_model is not None and
                        now - last_vlm >= VLM_COOLDOWN and
                        (vlm_thread is None or not vlm_thread.is_alive())):
                    with state_lock:
                        state["last_vlm_time"] = now
                    vlm_thread = _start_vlm_thread(threat_vlm, (crop.copy(), reason))
                if cur_alert == "CLEAR":
                    push_alert("YELLOW", reason)

            elif mode == "yolo_only" and yolo_only_trigger:
                reason, _ = yolo_only_trigger
                if cur_alert == "CLEAR":
                    push_alert("YELLOW", reason)

            else:
                if (cur_alert != "CLEAR" and
                        now - last_red >= RED_HOLD_SEC and
                        (vlm_thread is None or not vlm_thread.is_alive())):
                    push_alert("CLEAR", "")

            # ── Passive scene VLM ──────────────────────────────────────────────
            if (use_vlm and vlm_model is not None and
                    now - last_vlm >= vlm_ivl and
                    count_changed_at is None and not yolo_trigger and
                    (vlm_thread is None or not vlm_thread.is_alive())):
                with state_lock:
                    state["last_vlm_time"] = now
                vlm_thread = _start_vlm_thread(scene_vlm, (frame.copy(),))

            # ── Detection summary ──────────────────────────────────────────────
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

            # ── Overlay ────────────────────────────────────────────────────────
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

            color = {"CLEAR": (30,160,30), "YELLOW": (0,180,255), "RED": (0,0,210)}.get(alert_now, (60,60,60))
            cv2.rectangle(annotated, (0, 0), (w_f, 52), color, -1)
            cv2.putText(annotated, f"[{alert_now}]  {reason_now[:80]}",
                        (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)

            badge = f"People:{count_now}  [{mode_now.upper()}]"
            cv2.rectangle(annotated, (w_f-280, 0), (w_f, 52), (30, 30, 30), -1)
            cv2.putText(annotated, badge,
                        (w_f-270, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

            cv2.putText(annotated, f"{actual_fps:.0f}fps",
                        (w_f-55, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            vlm_active = vlm_on_now and mode_now != "yolo_only"
            vlm_color  = (160, 100, 255) if vlm_active else (80, 80, 80)
            cv2.putText(annotated, "VLM:ON" if vlm_active else "VLM:OFF",
                        (w_f-80, h_f-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, vlm_color, 1)

            if desc_now and use_vlm:
                cv2.putText(annotated, f"Scene: {desc_now[:100]}",
                            (8, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 255, 200), 1)
            elif not use_vlm:
                cv2.putText(annotated, f"Mode: {mode_now} — VLM inactive",
                            (8, h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 80), 1)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with engine["frame_lock"]:
                engine["frame"] = buf.tobytes()

        except torch.cuda.OutOfMemoryError:
            log.error(f"[ENGINE] OOM on frame #{frame_count} — clearing VRAM, continuing")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.5)
        except Exception as e:
            log.error(f"[ENGINE] Frame #{frame_count} error: {e} — skipping")

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
            "mode_switching":    state["mode_switching"],
            "vram_gpu0_pct":     round(get_vram_usage_pct(0), 1),
            "vram_gpu1_pct":     round(get_vram_usage_pct(1), 1),
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
    result = {}
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            result[f"gpu{i}"] = {
                "name":         props.name,
                "total_gb":     round(props.total_memory / 1024**3, 2),
                "allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                "reserved_gb":  round(torch.cuda.memory_reserved(i) / 1024**3, 2),
                "free_gb":      round((props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3, 2),
                "usage_pct":    round(get_vram_usage_pct(i), 1),
            }
        except Exception as e:
            result[f"gpu{i}"] = {"error": str(e)}
    return result


@app.get("/weapon_classes")
def get_weapon_classes():
    return {"classes": list(WEAPON_CLASSES.values())}


@app.post("/vlm/enable")
def vlm_enable():
    with state_lock:
        state["vlm_enabled"] = True
    return {"vlm_enabled": True}


@app.post("/vlm/disable")
def vlm_disable():
    with state_lock:
        state["vlm_enabled"] = False
    return {"vlm_enabled": False}


@app.post("/mode/{mode}")
def set_mode(mode: str):
    if mode not in VALID_MODES:
        return {"error": f"Invalid mode. Choose from: {VALID_MODES}"}

    with state_lock:
        current_mode = state["detection_mode"]
        if current_mode == mode:
            return {"detection_mode": mode, "mode_switching": False}
        state["mode_switching"] = True

    def do_switch():
        try:
            was_vlm  = current_mode in ("both", "vlm_only")
            will_vlm = mode in ("both", "vlm_only")
            if was_vlm and not will_vlm:
                offload_vlm_to_cpu()
            elif not was_vlm and will_vlm:
                reload_vlm_to_gpu()
            else:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            with state_lock:
                state["detection_mode"]    = mode
                state["alert"]             = "CLEAR"
                state["reason"]            = ""
                state["weapon_detections"] = []
        except Exception as e:
            log.error(f"[MODE] Switch error: {e}")
        finally:
            with state_lock:
                state["mode_switching"] = False
            log.info(f"[MODE] Switch complete → {mode}")

    threading.Thread(target=do_switch, daemon=True).start()
    return {"detection_mode": mode, "mode_switching": True}


@app.post("/vlm/interval")
def set_vlm_interval(seconds: float):
    seconds = max(5.0, min(seconds, 120.0))
    with state_lock:
        state["vlm_interval"] = seconds
    return {"vlm_interval": seconds}
