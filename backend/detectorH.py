import cv2, time, json, threading, logging
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
YOLO_MODEL_PATH    = "yolo11l.pt"          # upgraded from yolo26n → YOLO11l
WEAPON_MODEL_PATH  = "yolov8m.pt"          # upgraded from n → m for better accuracy
VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
WEAPON_CLASSES     = {}
PROXIMITY_DURATION = 2.5
VLM_COOLDOWN_SEC   = 3.0
RED_HOLD_SEC       = 6.0
RED_CONFIDENCE     = {"medium", "high"}

GPU_YOLO = "cuda:0"    # all YOLO / video processing
GPU_VLM  = "cuda:1"    # VLM exclusively — never shares with YOLO

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

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
    "person_log":        [],
    "person_count":      0,
    "vlm_enabled":       True,
    "detection_mode":    "both",
    "vlm_interval":      15.0,
    "mode_switching":    False,
}

state_lock = threading.Lock()


# ── Model Loaders ─────────────────────────────────────────────────────────────
def load_yolo():
    log.info(f"Loading YOLO11l on {GPU_YOLO}...")
    model = YOLO(YOLO_MODEL_PATH)
    model.to(GPU_YOLO)
    model.model.half()
    # torch.compile for ~20-30% faster inference on A4000
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
        log.info("[YOLO] torch.compile applied ✅")
    except Exception as e:
        log.warning(f"[YOLO] torch.compile skipped: {e}")
    log.info(f"YOLO11l ready on {GPU_YOLO} — {len(model.names)} classes.")
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
                continue
            filename   = pt_files[0]
            log.info(f"[WEAPON] Trying {repo_id} / {filename} ...")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model      = YOLO(model_path)
            model.to(GPU_YOLO)    # ← weapon model always on GPU 0
            model.model.half()
            WEAPON_CLASSES = {i: name for i, name in model.names.items()}
            log.info(f"[WEAPON] Loaded on {GPU_YOLO}. Classes: {list(model.names.values())}")
            return model
        except Exception as e:
            log.warning(f"[WEAPON] {repo_id} failed: {e}. Trying next...")

    log.warning("[WEAPON] Falling back to YOLOv8n COCO.")
    model = YOLO("yolov8n.pt")
    model.to(GPU_YOLO)
    WEAPON_CLASSES = {49: "knife"}
    return model


def load_vlm():
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from qwen_vl_utils import process_vision_info   # must be installed separately

        log.info(f"Loading Qwen3-VL-8B on {GPU_VLM} (8-bit)...")

        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
        model     = AutoModelForImageTextToText.from_pretrained(
            VLM_MODEL_ID,
            quantization_config=bnb,
            device_map={"": GPU_VLM},
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        log.info(f"Qwen3-VL-8B ready on {GPU_VLM}.")
        return model, processor
    except Exception as e:
        log.warning(f"Qwen3-VL-8B failed ({e}). Falling back to SmolVLM2...")
        return _load_smolvlm_fallback()



def _load_smolvlm_fallback():
    """SmolVLM2-2.2B at full BF16 — no quantization needed on 16GB."""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        log.info(f"Loading SmolVLM2-2.2B (BF16, no quantization) on {GPU_VLM}...")
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        model     = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            torch_dtype=torch.bfloat16,              # full BF16 — no quantization needed
            device_map={"": GPU_VLM},
            _attn_implementation="flash_attention_2",
        )
        model.eval()
        log.info("SmolVLM2-2.2B BF16 fallback ready.")
        return model, processor
    except Exception as e:
        log.warning(f"SmolVLM2 fallback failed ({e}). VLM disabled.")
        return None, None


# ── VRAM helpers (per GPU) ────────────────────────────────────────────────────
def get_vram_usage_pct(device_index: int = 0) -> float:
    try:
        reserved = torch.cuda.memory_reserved(device_index)
        total    = torch.cuda.get_device_properties(device_index).total_memory
        return (reserved / total) * 100
    except Exception:
        return 0.0


# ── VLM Prompts ───────────────────────────────────────────────────────────────
VLM_THREAT_PROMPT = """You are a surveillance AI. Analyze this image for potential threats. Focus on people, their actions, and any objects they hold."""

# ── Core VLM Inference ────────────────────────────────────────────────────────
def _qwen_infer(crop_bgr: np.ndarray, prompt: str, vlm_model, processor,
                max_tokens: int = 80) -> str:
    from qwen_vl_utils import process_vision_info
    try:
        h, w  = crop_bgr.shape[:2]
        scale = 768 / max(h, w)
        if scale < 1.0:
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ]
        }]

        # Qwen3-VL — apply_chat_template with tokenize=True + return_dict=True
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(GPU_VLM)

        with torch.no_grad():
            out = vlm_model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                max_new_tokens=max_tokens,
            )

        # Strip input tokens from output — Qwen3-VL standard pattern
        input_len = inputs["input_ids"].shape[1]
        result    = processor.batch_decode(
            out[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return result

    except torch.cuda.OutOfMemoryError:
        log.error(f"[VLM] OOM on {GPU_VLM}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(GPU_VLM)
        return ""
    except Exception as e:
        log.warning(f"[VLM] Qwen3 inference error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()


def _smolvlm_infer(crop_bgr: np.ndarray, prompt: str, vlm_model, processor,
                   max_tokens: int = 80) -> str:
    """Inference path for SmolVLM2 fallback."""
    try:
        h, w  = crop_bgr.shape[:2]
        scale = 512 / max(h, w)
        if scale < 1.0:
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)

        image    = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}]
        }]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs      = processor(
            text=text_prompt, images=[image], return_tensors="pt"
        ).to(GPU_VLM, dtype=torch.bfloat16)

        with torch.no_grad():
            out = vlm_model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                max_new_tokens=max_tokens,
            )

        result = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        return lines[-1] if lines else result

    except torch.cuda.OutOfMemoryError:
        log.error(f"[VLM] OOM on {GPU_VLM}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize(GPU_VLM)
        return ""
    except Exception as e:
        log.warning(f"[VLM] SmolVLM inference error: {e}")
        return ""
    finally:
        torch.cuda.empty_cache()


# ── Unified infer dispatcher ──────────────────────────────────────────────────
_vlm_is_qwen = False   # set True after load if Qwen loaded successfully

def vlm_infer(crop_bgr: np.ndarray, prompt: str, vlm_model, processor,
              max_tokens: int = 80) -> str:
    if vlm_model is None or processor is None:
        return ""
    if _vlm_is_qwen:
        return _qwen_infer(crop_bgr, prompt, vlm_model, processor, max_tokens)
    return _smolvlm_infer(crop_bgr, prompt, vlm_model, processor, max_tokens)


def run_vlm(crop_bgr: np.ndarray, vlm_model, processor) -> dict:
    if vlm_model is None:
        return {"threat": False, "type": "none", "confidence": "low",
                "description": "VLM not loaded"}
    raw = vlm_infer(crop_bgr, VLM_THREAT_PROMPT, vlm_model, processor, max_tokens=100)
    try:
        js = raw[raw.rfind("{") : raw.rfind("}") + 1]
        return json.loads(js)
    except (json.JSONDecodeError, ValueError):
        is_threat = '"threat": true' in raw.lower()
        return {
            "threat":      is_threat,
            "type":        "assault" if is_threat else "none",
            "confidence":  "medium" if is_threat else "low",
            "description": raw[:120],
        }


# ── Weapon Inference ──────────────────────────────────────────────────────────
def run_weapon_inference(weapon_model, frame: np.ndarray) -> tuple:
    results        = weapon_model(frame, conf=0.65, imgsz=640, verbose=False)
    detections     = []
    trigger_reason = None
    trigger_crop   = None
    fh, fw         = frame.shape[:2]
    frame_area     = fh * fw

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = WEAPON_CLASSES.get(cls_id, "weapon")
        xyxy   = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = map(int, xyxy)
        box_w  = x2 - x1
        box_h  = y2 - y1
        area   = box_w * box_h
        aspect = box_w / max(box_h, 1)

        if area < frame_area * 0.02:
            log.debug(f"[WEAPON] {name} skipped — too small ({area}px²)")
            continue
        if not (1.2 < aspect < 7.0):
            log.debug(f"[WEAPON] {name} skipped — aspect {aspect:.2f}")
            continue
        if box_w > fw * 0.85 or box_h > fh * 0.85:
            log.debug(f"[WEAPON] {name} skipped — bbox too large")
            continue

        detections.append({
            "label":      name,
            "confidence": round(conf, 2),
            "bbox":       [x1, y1, x2, y2],
        })
        if trigger_reason is None:
            trigger_reason = f"Weapon detected: {name} ({int(conf * 100)}%)"
            trigger_crop   = pad_crop(frame, xyxy)
            log.info(f"[WEAPON] ✅ {name} @ {int(conf*100)}% | area={area} aspect={aspect:.2f}")

    return detections, trigger_reason, trigger_crop


def draw_weapon_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
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
                "reason": reason,
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
