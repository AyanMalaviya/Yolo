import torch
from ultralytics import YOLO

def print_vram(label):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    print(f"[{label}]  Allocated: {allocated:.2f} GB  |  Reserved: {reserved:.2f} GB")

torch.cuda.reset_peak_memory_stats()
print_vram("Baseline (before any model)")

# YOLO
model = YOLO("yolo12s.pt")
model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)  # warm up
print_vram("After YOLO12s loaded + warmup")
yolo_mem = torch.cuda.memory_allocated()

# VLM (only if you have it downloaded)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb,
        device_map="cuda"
    )
    vlm_mem = torch.cuda.memory_allocated() - yolo_mem
    print_vram("After Qwen VLM loaded (4-bit)")
    print(f"\n  YOLO12s alone:   {yolo_mem / 1024**3:.2f} GB")
    print(f"  Qwen VLM alone:  {vlm_mem  / 1024**3:.2f} GB")
    print(f"  Combined total:  {(yolo_mem + vlm_mem) / 1024**3:.2f} GB")
    print(f"  Free remaining:  {(6.0 - (yolo_mem + vlm_mem) / 1024**3):.2f} GB")
except Exception as e:
    print(f"VLM not loaded: {e}")
