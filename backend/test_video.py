"""
Standalone video test: person tracking + weapon detection + VLM scene description
Run: python test_video.py --source "C:/Videos/clip.mp4"
     python test_video.py --source "C:/Videos/clip.mp4" --no-vlm
"""

import argparse, cv2, json, time, logging, threading
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH  = "yolo12s.pt"
VLM_MODEL_ID     = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR       = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# COCO weapon classes YOLO knows
WEAPON_CLASSES   = {49: "knife", 76: "scissors"}
PROXIMITY_THRESH = 0.65   # fraction of avg height
PROXIMITY_SECS   = 1.5    # faster trigger for video testing
VLM_COOLDOWN     = 4.0    # seconds between VLM calls

# ── VLM Prompt ────────────────────────────────────────────────────────────────
VLM_PROMPT = """You are a surveillance AI analyzing CCTV footage.

Identify if any of these are clearly visible:
- Axe, hatchet, machete, baseball bat, or any blunt/bladed weapon
- Someone swinging or threatening with an object
- Forced entry or breaking behavior
- One person physically attacking another with harmful intent

Do NOT flag:
- Normal object carrying
- Ambiguous or unclear scenes
- Playful or sporting activity

Reply ONLY with valid JSON, nothing else:
{"threat": false, "weapon": "none", "action": "normal", "confidence": "low", "description": "No threat detected"}
{"threat": true, "weapon": "axe|knife|bat|other", "action": "assault|intrusion|threatening", "confidence": "low|medium|high", "description": "One concise sentence max"}"""

# ── Load Models ───────────────────────────────────────────────────────────────
def load_yolo():
    log.info("Loading YOLO12s...")
    m = YOLO(YOLO_MODEL_PATH)
    log.info("YOLO12s ready.")
    return m

def load_vlm():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        log.info("Loading Qwen2.5-VL-3B in 4-bit...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, quantization_config=bnb, device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        log.info("VLM ready.")
        return model, processor
    except Exception as e:
        log.warning(f"VLM not loaded: {e}")
        return None, None

# ── Utilities ─────────────────────────────────────────────────────────────────
def people_are_close(b1, b2):
    cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
    cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
    dist      = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    avg_h     = ((b1[3]-b1[1]) + (b2[3]-b2[1])) / 2
    return dist < avg_h * PROXIMITY_THRESH

def merge_boxes(b1, b2, pad=60, w=640, h=480):
    x1 = max(0, min(b1[0],b2[0]) - pad)
    y1 = max(0, min(b1[1],b2[1]) - pad)
    x2 = min(w, max(b1[2],b2[2]) + pad)
    y2 = min(h, max(b1[3],b2[3]) + pad)
    return int(x1), int(y1), int(x2), int(y2)

def crop_box(frame, xyxy, pad=50):
    h, w = frame.shape[:2]
    x1 = max(0, int(xyxy[0])-pad)
    y1 = max(0, int(xyxy[1])-pad)
    x2 = min(w, int(xyxy[2])+pad)
    y2 = min(h, int(xyxy[3])+pad)
    return frame[y1:y2, x1:x2]

def run_vlm(crop_bgr, vlm_model, processor):
    try:
        from qwen_vl_utils import process_vision_info
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  VLM_PROMPT}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_new_tokens=90, do_sample=False)
        decoded  = processor.decode(out[0], skip_special_tokens=True)
        json_str = decoded[decoded.rfind("{"):decoded.rfind("}")+1]
        return json.loads(json_str)
    except Exception as e:
        log.warning(f"VLM error: {e}")
        return {"threat": False, "weapon": "none", "confidence": "low",
                "description": "VLM parse error"}

# ── Main Processing ───────────────────────────────────────────────────────────
def process_video(source, use_vlm, yolo_model, vlm_model, vlm_processor):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open: {source}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = OUTPUT_DIR / f"result_{Path(source).stem}_{datetime.now().strftime('%H%M%S')}.mp4"
    writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    log.info(f"Processing: {source}  |  {width}x{height} @ {fps:.1f}fps  |  {total} frames")
    log.info(f"Output → {out_path}")

    # State
    alert_state   = "CLEAR"
    alert_reason  = ""
    vlm_desc      = ""
    last_vlm_time = 0.0
    vlm_thread    = None
    pair_since    = {}         # (id1,id2) → timestamp
    alert_log     = []

    # Thread-safe VLM result container
    vlm_result_box = [None]
    vlm_lock       = threading.Lock()

    def async_vlm(crop, reason):
        result = run_vlm(crop, vlm_model, vlm_processor)
        with vlm_lock:
            vlm_result_box[0] = (reason, result)
        log.info(f"VLM → {result}")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        now = time.time()

        # Collect VLM result if thread finished
        with vlm_lock:
            if vlm_result_box[0] is not None:
                reason, result = vlm_result_box[0]
                vlm_result_box[0] = None
                vlm_desc = result.get("description", "")
                is_threat   = result.get("threat", False)
                confidence  = result.get("confidence", "low")
                weapon_seen = result.get("weapon", "none")
                if is_threat and confidence in {"medium", "high"}:
                    alert_state  = "RED"
                    alert_reason = f"{reason} | weapon:{weapon_seen}"
                    alert_log.append({
                        "frame": frame_num,
                        "time":  round(frame_num/fps, 1),
                        "alert": "RED",
                        "reason": alert_reason,
                        "vlm": vlm_desc
                    })
                    log.warning(f"🔴 RED ALERT @ frame {frame_num} ({frame_num/fps:.1f}s) — {alert_reason}")
                else:
                    if alert_state != "RED":
                        alert_state  = "YELLOW"
                        alert_reason = f"{reason} | not confirmed"

        results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml",
                                   conf=0.35, verbose=False)

        people_boxes = {}
        trigger_crop = None
        trigger_reason = None

        for box in results[0].boxes:
            if box.id is None:
                continue
            cls_id   = int(box.cls)
            track_id = int(box.id)
            xyxy     = box.xyxy[0].cpu().numpy()

            if cls_id == 0:
                people_boxes[track_id] = xyxy

            # Known COCO weapon
            if cls_id in WEAPON_CLASSES and trigger_crop is None:
                trigger_reason = f"COCO weapon detected: {WEAPON_CLASSES[cls_id]}"
                trigger_crop   = crop_box(frame, xyxy)
                if alert_state == "CLEAR":
                    alert_state  = "YELLOW"
                    alert_reason = trigger_reason

        # Proximity detection between people pairs
        if trigger_crop is None:
            ids = list(people_boxes.keys())
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    id1, id2 = ids[i], ids[j]
                    pair = (min(id1,id2), max(id1,id2))
                    if people_are_close(people_boxes[id1], people_boxes[id2]):
                        pair_since.setdefault(pair, now)
                        if now - pair_since[pair] >= PROXIMITY_SECS:
                            x1, y1, x2, y2 = merge_boxes(
                                people_boxes[id1], people_boxes[id2], w=width, h=height)
                            trigger_crop   = frame[y1:y2, x1:x2]
                            trigger_reason = f"Close contact — IDs {pair}"
                            if alert_state == "CLEAR":
                                alert_state  = "YELLOW"
                                alert_reason = trigger_reason
                    else:
                        pair_since.pop(pair, None)

        # Dispatch VLM
        if (trigger_crop is not None and use_vlm and vlm_model is not None and
                now - last_vlm_time >= VLM_COOLDOWN and
                (vlm_thread is None or not vlm_thread.is_alive())):
            last_vlm_time = now
            vlm_thread = threading.Thread(
                target=async_vlm,
                args=(trigger_crop.copy(), trigger_reason),
                daemon=True
            )
            vlm_thread.start()

        # Slowly decay alert (only in video mode — after 5 seconds no trigger)
        if trigger_crop is None and alert_state == "YELLOW":
            alert_state  = "CLEAR"
            alert_reason = ""

        # ── Overlay ───────────────────────────────────────────────────────────
        annotated = results[0].plot()
        color = {"CLEAR": (30,160,30), "YELLOW": (0,180,255), "RED": (0,0,210)}.get(alert_state, (80,80,80))

        # Alert banner
        cv2.rectangle(annotated, (0, 0), (width, 56), color, -1)
        cv2.putText(annotated, f"[{alert_state}]  {alert_reason[:75]}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,0,0), 2)

        # Frame counter
        cv2.putText(annotated, f"Frame {frame_num}/{total}  |  {frame_num/fps:.1f}s",
                    (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        # VLM description
        if vlm_desc:
            cv2.putText(annotated, f"VLM: {vlm_desc[:95]}",
                        (10, height-14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,255,200), 1)

        writer.write(annotated)
        cv2.imshow("Test Video — YOLO + VLM", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log.info("Quit early.")
            break

        if frame_num % 100 == 0:
            log.info(f"Progress: {frame_num}/{total} frames ({100*frame_num/total:.1f}%)")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save alert log JSON
    log_path = OUTPUT_DIR / f"alerts_{Path(source).stem}.json"
    with open(log_path, "w") as f:
        json.dump(alert_log, f, indent=2)

    log.info(f"✅ Done. Output video → {out_path}")
    log.info(f"📋 Alert log    → {log_path}")
    log.info(f"🔴 RED alerts:  {sum(1 for a in alert_log if a['alert'] == 'RED')}")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Path to video file, e.g. C:/Videos/robbery.mp4")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip VLM — YOLO detection only")
    args = parser.parse_args()

    yolo = load_yolo()
    vlm_model, vlm_proc = (None, None) if args.no_vlm else load_vlm()

    process_video(
        source       = args.source,
        use_vlm      = not args.no_vlm,
        yolo_model   = yolo,
        vlm_model    = vlm_model,
        vlm_processor= vlm_proc,
    )
