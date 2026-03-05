import argparse
from detector import load_yolo, load_vlm, detection_loop

parser = argparse.ArgumentParser(description="YOLO + VLM Surveillance Engine")
parser.add_argument("--source", type=str, default="0",
                    help="Camera index (0,1,2) or full path to video file")
parser.add_argument("--no-vlm", action="store_true",
                    help="Skip VLM — YOLO-only mode (faster startup)")
parser.add_argument("--no-window", action="store_true",
                    help="Disable OpenCV preview window")
args = parser.parse_args()

yolo = load_yolo()
vlm_model, vlm_proc = (None, None) if args.no_vlm else load_vlm()

detection_loop(
    source      = args.source,
    show_window = not args.no_window,
    yolo_model  = yolo,
    vlm_model   = vlm_model,
    vlm_processor = vlm_proc,
)
