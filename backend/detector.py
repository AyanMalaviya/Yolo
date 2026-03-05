import cv2, time, json, threading, logging, csv
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = "yolo12s.pt"
VLM_MODEL_ID       = "Qwen/Qwen2.5-VL-3B-Instruct"
WEAPON_CLASSES     = {49: "knife", 76: "scissors"}
PROXIMITY_DURATION = 2.5
VLM_COOLDOWN_SEC   = 3.0
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}
LINE_POSITION      = 0.5

LOG_DIR        = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_LOG_PATH   = LOG_DIR / "people_log.csv"
JSON_LOG_PATH  = LOG_DIR / "people_log.json"

# ── Shared State ──────────────────────────────────────────────────────────────
state = {
    "alert":            "CLEAR",
    "reason":           "",
    "vlm_description":  "",
    "threat_type":      "none",
    "last_vlm_time":    0.0,
    "last_red_time":    0.0,
    "alert_log":        [],
    "scene_description": "Waiting for first analysis...",  
    "detection_summary": "",                               
}

state_lock = threading.Lock()

# ── Model Loaders ─────────────────────────────────────────────────────────────
def load_yolo():
    log.info("Loading YOLO12s on GPU...")
    model = YOLO(YOLO_MODEL_PATH)
    log.info("YOLO12s ready.")
    return model

def load_vlm():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        log.info("Loading Qwen2.5-VL-3B (4-bit)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, quantization_config=bnb, device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        log.info("VLM ready.")
        return model, processor
    except Exception as e:
        log.warning(f"VLM load failed ({e}). Running YOLO-only mode.")
        return None, None

# ── VLM Prompt ────────────────────────────────────────────────────────────────
VLM_PROMPT = """You are a surveillance AI. Analyze this image for genuine physical threats only.

Flag as a threat ONLY if you clearly see:
- A weapon (knife, bat, gun) being used aggressively against a person
- Forced entry: someone breaking a door, window, or barrier
- One person violently attacking another with clear harmful intent (not playful)
- A person visibly being restrained against their will or showing signs of injury

Do NOT flag as a threat:
- Friends play-fighting, sparring, or rough-housing
- People standing, sitting, or walking close together
- Normal physical contact (hugs, handshakes, shoulder grabs)
- Ambiguous or unclear situations

Respond ONLY with valid JSON and nothing else:
{"threat": false, "type": "none", "confidence": "low", "description": "No threat detected"}
{"threat": true, "type": "fight|weapon|intrusion|assault", "confidence": "low|medium|high", "description": "One concise sentence"}"""

# ── Utilities ─────────────────────────────────────────────────────────────────
def are_people_close(b1, b2):
    c1 = ((b1[0]+b1[2])/2, (b1[1]+b1[3])/2)
    c2 = ((b2[0]+b2[2])/2, (b2[1]+b2[3])/2)
    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    avg_height = ((b1[3]-b1[1]) + (b2[3]-b2[1])) / 2
    return dist < avg_height * 0.7

def pad_crop(frame, box, pad=60):
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0])-pad)
    y1 = max(0, int(box[1])-pad)
    x2 = min(w, int(box[2])+pad)
    y2 = min(h, int(box[3])+pad)
    return frame[y1:y2, x1:x2]

def merged_bbox(b1, b2):
    return [min(b1[0],b2[0]), min(b1[1],b2[1]), max(b1[2],b2[2]), max(b1[3],b2[3])]

def run_vlm(crop, vlm_model, processor):
    try:
        from qwen_vl_utils import process_vision_info
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  VLM_PROMPT}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=img_inputs, videos=vid_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_new_tokens=80, do_sample=False)
        decoded = processor.decode(out[0], skip_special_tokens=True)
        js = decoded[decoded.rfind("{"):decoded.rfind("}")+1]
        return json.loads(js)
    except Exception as e:
        log.warning(f"VLM error: {e}")
        return {"threat": False, "type": "none", "confidence": "low", "description": "VLM error"}

def push_alert(alert, reason, vlm_result=None):
    with state_lock:
        state["alert"]  = alert
        state["reason"] = reason
        if vlm_result:
            state["vlm_description"] = vlm_result.get("description", "")
            state["threat_type"]     = vlm_result.get("type", "none")
        if alert == "RED":
            state["last_red_time"] = time.time()
        # ── Only log YELLOW and RED, never CLEAR ──
        if alert != "CLEAR":
            entry = {"time": datetime.now().strftime("%H:%M:%S"), "alert": alert, "reason": reason}
            if vlm_result:
                entry["vlm"] = vlm_result.get("description", "")
            state["alert_log"].append(entry)
            state["alert_log"] = state["alert_log"][-100:]
    log.info(f"[{alert}] {reason}")

# ── Proximity Tracker ─────────────────────────────────────────────────────────
class ProximityTracker:
    def __init__(self):
        self.pair_since = {}

    def update(self, people_boxes):
        ids = list(people_boxes.keys())
        now = time.time()
        active_pairs = set()
        result = None
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair = (min(id1,id2), max(id1,id2))
                if are_people_close(people_boxes[id1], people_boxes[id2]):
                    active_pairs.add(pair)
                    self.pair_since.setdefault(pair, now)
                    if now - self.pair_since[pair] >= PROXIMITY_DURATION:
                        mb = merged_bbox(people_boxes[id1], people_boxes[id2])
                        result = (pair, mb)
        for p in list(self.pair_since):
            if p not in active_pairs:
                del self.pair_since[p]
        return result

# ── People Logger ─────────────────────────────────────────────────────────────
class PeopleLogger:
    def __init__(self):
        self.prev_positions = {}
        self.crossed_ids    = set()
        self.counts         = {"enter": 0, "exit": 0}
        self._init_csv()

    def _init_csv(self):
        if not CSV_LOG_PATH.exists():
            with open(CSV_LOG_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "track_id", "event", "confidence", "alert_state"])

    def _write_csv(self, entry):
        with open(CSV_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([entry["timestamp"], entry["track_id"],
                             entry["event"], entry["confidence"], entry["alert_state"]])

    def _write_json(self, entry):
        data = []
        if JSON_LOG_PATH.exists():
            try:
                with open(JSON_LOG_PATH, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        data.append(entry)
        with open(JSON_LOG_PATH, "w") as f:
            json.dump(data[-500:], f, indent=2)

    def update(self, track_id, center_y, frame_height, conf, alert_state):
        line_y = frame_height * LINE_POSITION
        if track_id in self.prev_positions:
            prev_y = self.prev_positions[track_id]
            crossed = (
                (prev_y < line_y <= center_y) or
                (prev_y > line_y >= center_y)
            )
            if crossed and track_id not in self.crossed_ids:
                event = "EXIT" if center_y >= line_y else "ENTER"
                self.crossed_ids.add(track_id)
                self.counts[event.lower()] += 1
                entry = {
                    "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "track_id":    track_id,
                    "event":       event,
                    "confidence":  round(conf, 2),
                    "alert_state": alert_state,
                }
                self._write_csv(entry)
                self._write_json(entry)
                log.info(f"[LOG] ID#{track_id} → {event} | IN:{self.counts['enter']} OUT:{self.counts['exit']}")
        self.prev_positions[track_id] = center_y

    def get_counts(self):
        return self.counts
