import cv2, time, json, threading, logging, csv
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = "yolo26n.pt"
WEAPON_MODEL_REPO  = "Subh775/Threat-Detection-YOLOv8n"
VLM_MODEL_ID       = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
WEAPON_CLASSES     = {}   # populated by load_weapon_model()
PROXIMITY_DURATION = 2.5
VLM_COOLDOWN_SEC   = 3.0
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}
LINE_POSITION      = 0.5

LOG_DIR       = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CSV_LOG_PATH  = LOG_DIR / "people_log.csv"
JSON_LOG_PATH = LOG_DIR / "people_log.json"


# ── Shared State ──────────────────────────────────────────────────────────────
state = {
    "alert":             "CLEAR",
    "reason":            "",
    "vlm_description":   "",
    "threat_type":       "none",
    "last_vlm_time":     0.0,
    "last_red_time":     0.0,
    "alert_log":         [],
    "scene_description": "Waiting for first analysis...",
    "detection_summary": "",
    "weapon_detections": [],
    "source_fps":        0.0,
}

state_lock = threading.Lock()


# ── Model Loaders ─────────────────────────────────────────────────────────────
def load_yolo():
    log.info("Loading YOLO26n for tracking...")
    model = YOLO(YOLO_MODEL_PATH)
    log.info(f"YOLO26n ready. {len(model.names)} classes.")
    return model


def load_weapon_model():
    global WEAPON_CLASSES

    candidates = [
        "Subh775/Threat-Detection-YOLOv8n",
        "Subh775/Firearm_Detection_Yolov8n",
        "Hadi959/weapon-detection-yolov8",
    ]

    for repo_id in candidates:
        try:
            files    = list(list_repo_files(repo_id))
            pt_files = [f for f in files if f.endswith(".pt")]
            if not pt_files:
                log.warning(f"[WEAPON] No .pt in {repo_id}, skipping...")
                continue
            filename   = pt_files[0]
            log.info(f"[WEAPON] Trying {repo_id} / {filename} ...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model      = YOLO(model_path)
            WEAPON_CLASSES = {i: name for i, name in model.names.items()}
            log.info(f"[WEAPON] Loaded {repo_id}. Classes: {list(model.names.values())}")
            return model
        except Exception as e:
            log.warning(f"[WEAPON] {repo_id} failed: {e}. Trying next...")

    # COCO fallback — knife and scissors only
    log.warning("[WEAPON] All HF models failed. Falling back to YOLOv8n COCO (knife/scissors).")
    model = YOLO("yolov8n.pt")
    WEAPON_CLASSES = {49: "knife", 76: "scissors"}
    return model


def load_vlm():
    """Load SmolVLM2-2.2B-Instruct in 4-bit quantization."""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        log.info("Loading SmolVLM2-2.2B-Instruct (4-bit)...")
        bnb       = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        model     = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_ID,
            quantization_config=bnb,
            device_map="cuda",
            _attn_implementation="eager"
        )
        model.eval()
        log.info("SmolVLM2-2.2B ready.")
        return model, processor
    except Exception as e:
        log.warning(f"VLM load failed ({e}). Running YOLO-only mode.")
        return None, None


# ── VLM Prompts ───────────────────────────────────────────────────────────────
VLM_THREAT_PROMPT = """You are a surveillance AI. Analyze this image for genuine physical threats only.

Flag as a threat ONLY if you clearly see:
- A weapon (knife, axe, bat, gun, sword) being used aggressively against a person
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


# ── Core VLM Inference (SmolVLM2) ─────────────────────────────────────────────
def _smolvlm_infer(crop_bgr: np.ndarray, prompt: str, vlm_model, processor, max_tokens: int = 80) -> str:
    """
    Single inference call — takes BGR crop, returns raw string output.
    All other VLM helpers call this.
    """
    if vlm_model is None or processor is None:
        return ""
    try:
        # Resize to max 512px on longest side
        h, w    = crop_bgr.shape[:2]
        scale   = 512 / max(h, w)
        if scale < 1.0:
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)

        image    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]

        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs      = processor(
            text=text_prompt,
            images=[image],
            return_tensors="pt"
        ).to(vlm_model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            out = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)

        result = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

        # Strip assistant prefix SmolVLM2 sometimes adds
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()

        # Return last non-empty line
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return lines[-1] if lines else result

    except Exception as e:
        log.warning(f"[VLM] Inference error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()


def run_vlm(crop_bgr: np.ndarray, vlm_model, processor) -> dict:
    """Threat analysis — returns parsed JSON dict."""
    if vlm_model is None or processor is None:
        return {"threat": False, "type": "none", "confidence": "low", "description": "VLM not loaded"}

    raw = _smolvlm_infer(crop_bgr, VLM_THREAT_PROMPT, vlm_model, processor, max_tokens=80)

    # Try to parse JSON
    try:
        js = raw[raw.rfind("{") : raw.rfind("}") + 1]
        return json.loads(js)
    except (json.JSONDecodeError, ValueError):
        # SmolVLM2 sometimes returns plain text — parse manually
        is_threat = "threat: yes" in raw.lower() or '"threat": true' in raw.lower()
        return {
            "threat":      is_threat,
            "type":        "assault" if is_threat else "none",
            "confidence":  "medium" if is_threat else "low",
            "description": raw[:120]
        }


def run_vlm_text(crop_bgr: np.ndarray, prompt: str, vlm_model, processor) -> str:
    """Free-text VLM response — for scene/person descriptions."""
    return _smolvlm_infer(crop_bgr, prompt, vlm_model, processor, max_tokens=80)


# ── Weapon Inference + Drawing ────────────────────────────────────────────────
def run_weapon_inference(weapon_model, frame: np.ndarray) -> tuple:
    """
    Run weapons model on frame.
    Returns: (detections, trigger_reason, trigger_crop)
      detections    — list of {label, confidence, bbox}
      trigger_reason — string if weapon found, else None
      trigger_crop  — padded crop around first weapon, else None
    """
    results        = weapon_model(frame, conf=0.40, imgsz=640, verbose=False)
    detections     = []
    trigger_reason = None
    trigger_crop   = None

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = WEAPON_CLASSES.get(cls_id, "weapon")
        xyxy   = box.xyxy[0].cpu().numpy()

        detections.append({
            "label":      name,
            "confidence": round(conf, 2),
            "bbox":       list(map(int, xyxy.tolist())),
        })

        if trigger_reason is None:
            trigger_reason = f"Weapon detected: {name} ({int(conf * 100)}%)"
            trigger_crop   = pad_crop(frame, xyxy)

        log.info(f"[WEAPON] {name} @ {int(conf * 100)}% conf")

    return detections, trigger_reason, trigger_crop


def draw_weapon_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw red labeled boxes for weapon detections on top of YOLO26 overlay."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label           = f"{det['label']} {int(det['confidence'] * 100)}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ── Utilities ─────────────────────────────────────────────────────────────────
def are_people_close(b1, b2) -> bool:
    c1         = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
    c2         = ((b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2)
    dist       = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    avg_height = ((b1[3] - b1[1]) + (b2[3] - b2[1])) / 2
    return dist < avg_height * 0.7


def pad_crop(frame: np.ndarray, box, pad: int = 60) -> np.ndarray:
    h, w = frame.shape[:2]
    x1   = max(0, int(box[0]) - pad)
    y1   = max(0, int(box[1]) - pad)
    x2   = min(w, int(box[2]) + pad)
    y2   = min(h, int(box[3]) + pad)
    return frame[y1:y2, x1:x2]


def merged_bbox(b1, b2) -> list:
    return [min(b1[0], b2[0]), min(b1[1], b2[1]),
            max(b1[2], b2[2]), max(b1[3], b2[3])]


def push_alert(alert: str, reason: str, vlm_result: dict = None):
    with state_lock:
        state["alert"]  = alert
        state["reason"] = reason
        if vlm_result:
            state["vlm_description"] = vlm_result.get("description", "")
            state["threat_type"]     = vlm_result.get("type", "none")
        if alert == "RED":
            state["last_red_time"] = time.time()
        if alert != "CLEAR":
            entry = {
                "time":   datetime.now().strftime("%H:%M:%S"),
                "alert":  alert,
                "reason": reason
            }
            if vlm_result:
                entry["vlm"] = vlm_result.get("description", "")
            state["alert_log"].append(entry)
            state["alert_log"] = state["alert_log"][-100:]
    log.info(f"[{alert}] {reason}")


# ── Proximity Tracker ─────────────────────────────────────────────────────────
class ProximityTracker:
    def __init__(self):
        self.pair_since: dict = {}

    def update(self, people_boxes: dict):
        ids          = list(people_boxes.keys())
        now          = time.time()
        active_pairs: set = set()
        result       = None

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                pair     = (min(id1, id2), max(id1, id2))
                if are_people_close(people_boxes[id1], people_boxes[id2]):
                    active_pairs.add(pair)
                    self.pair_since.setdefault(pair, now)
                    if now - self.pair_since[pair] >= PROXIMITY_DURATION:
                        mb     = merged_bbox(people_boxes[id1], people_boxes[id2])
                        result = (pair, mb)

        for p in list(self.pair_since):
            if p not in active_pairs:
                del self.pair_since[p]

        return result


# ── People Logger ─────────────────────────────────────────────────────────────
class PeopleLogger:
    def __init__(self):
        self.prev_positions: dict = {}
        self.crossed_ids: set     = set()
        self.counts               = {"enter": 0, "exit": 0}
        self._init_csv()

    def _init_csv(self):
        if not CSV_LOG_PATH.exists():
            with open(CSV_LOG_PATH, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "track_id", "event", "confidence", "alert_state"]
                )

    def _write_csv(self, entry: dict):
        with open(CSV_LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                entry["timestamp"], entry["track_id"],
                entry["event"],     entry["confidence"],
                entry["alert_state"]
            ])

    def _write_json(self, entry: dict):
        data = []
        if JSON_LOG_PATH.exists():
            try:
                with open(JSON_LOG_PATH) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
        data.append(entry)
        with open(JSON_LOG_PATH, "w") as f:
            json.dump(data[-500:], f, indent=2)

    def update(self, track_id: int, center_y: float,
               frame_height: int, conf: float, alert_state: str):
        line_y = frame_height * LINE_POSITION
        if track_id in self.prev_positions:
            prev_y  = self.prev_positions[track_id]
            crossed = (prev_y < line_y <= center_y) or (prev_y > line_y >= center_y)
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
                log.info(
                    f"[LOG] ID#{track_id} → {event} | "
                    f"IN:{self.counts['enter']} OUT:{self.counts['exit']}"
                )
        self.prev_positions[track_id] = center_y

    def get_counts(self) -> dict:
        return self.counts
