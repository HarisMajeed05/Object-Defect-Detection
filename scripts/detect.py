"""
scripts/detect.py
===================
Runs a trained YOLO model on an image, a folder of images, a video file, or a
live webcam, draws bounding boxes with class name and confidence, and shows
FPS and object count overlaid, same as the reference article's script, with
CPU/GPU device selection added.

USAGE
-----
Webcam (device index 0):
    python scripts/detect.py --model runs/detect/train/weights/best.pt --source usb0

Single image:
    python scripts/detect.py --model runs/detect/train/weights/best.pt --source test.jpg

Folder of images:
    python scripts/detect.py --model runs/detect/train/weights/best.pt --source img_dir/

Video file:
    python scripts/detect.py --model runs/detect/train/weights/best.pt --source test_vid.mp4

Force GPU inference (falls back to CPU automatically if none is available):
    python scripts/detect.py --model best.pt --source usb0 --gpu

Press 'q' to quit, works the same for all source types.
"""

import argparse
import glob
import os
import time

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_device(want_gpu: bool) -> str:
    if not want_gpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU requested and available: {torch.cuda.get_device_name(0)}")
            return "0"
    except ImportError:
        pass
    print("GPU requested but not available, falling back to CPU.")
    return "cpu"


def parse_resolution(res_str):
    if not res_str:
        return None
    w, h = res_str.lower().split("x")
    return int(w), int(h)


def classify_source(source):
    if source.startswith("usb"):
        return "webcam", int(source.replace("usb", ""))
    if os.path.isdir(source):
        return "folder", source
    ext = os.path.splitext(source)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image", source
    if ext in VIDEO_EXTENSIONS:
        return "video", source
    raise SystemExit(f"Could not determine source type for: {source}")


def draw_results(frame, result, fps=None):
    boxes = result.boxes
    count = 0
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{result.names[cls_id]}: {int(conf * 100)}%"

            color = _color_for_class(cls_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            count += 1

    y = 30
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        y += 30
    cv2.putText(frame, f"Number of objects: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame, count


def _color_for_class(cls_id):
    # Deterministic per-class color, same class always gets the same color
    # across frames, without needing a hand maintained color table.
    rng = (cls_id * 47) % 255
    return (int((rng * 3) % 255), int((rng * 7) % 255), int((rng * 13) % 255))


def run_on_stream(model, cap, window_name, conf, device):
    prev_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, device=device, verbose=False)[0]
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now

        frame, _ = draw_results(frame, results, fps=fps)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained weights, e.g. runs/detect/train/weights/best.pt")
    ap.add_argument("--source", required=True,
                     help="Image file, folder, video file, or 'usbN' for webcam index N")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--resolution", default=None, help="Optional WxH override, e.g. 1280x720")
    ap.add_argument("--gpu", action="store_true", help="Run inference on GPU instead of CPU")
    args = ap.parse_args()

    device = resolve_device(args.gpu)
    model = YOLO(args.model)
    res = parse_resolution(args.resolution)

    kind, value = classify_source(args.source)
    window_name = "YOLO Detection - press q to quit"

    if kind == "webcam":
        cap = cv2.VideoCapture(value)
        if not cap.isOpened():
            raise SystemExit(f"Could not open webcam index {value}")
        if res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        run_on_stream(model, cap, window_name, args.conf, device)
        cap.release()

    elif kind == "video":
        cap = cv2.VideoCapture(value)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {value}")
        run_on_stream(model, cap, window_name, args.conf, device)
        cap.release()

    elif kind == "image":
        frame = cv2.imread(value)
        if frame is None:
            raise SystemExit(f"Could not read image: {value}")
        results = model(frame, conf=args.conf, device=device, verbose=False)[0]
        frame, count = draw_results(frame, results)
        print(f"{value}: {count} object(s) detected")
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)

    elif kind == "folder":
        paths = sorted(
            p for p in glob.glob(os.path.join(value, "*"))
            if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise SystemExit(f"No images found in {value}")
        for path in paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
            results = model(frame, conf=args.conf, device=device, verbose=False)[0]
            frame, count = draw_results(frame, results)
            print(f"{path}: {count} object(s) detected")
            cv2.imshow(window_name, frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
