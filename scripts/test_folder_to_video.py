"""
scripts/test_folder_to_video.py
================================
Runs a trained YOLO model on every image inside a folder (e.g. your val/images
folder, when you don't have a dedicated test set) and compiles the annotated
results into a single output video, one image held for a configurable number
of seconds, so you can review the whole validation set like a slideshow.

Reuses the same box drawing style as scripts/detect.py, with CPU/GPU device
selection, and prints a short summary report at the end (total images,
total detections, per-class counts, average confidence) since that doubles
as a quick performance check.

USAGE
-----
Basic run on the val folder, each image shown for 1.5 seconds:
    python scripts/test_folder_to_video.py --model runs/detect/train/weights/best.pt --source data/validation/images --output val_test.mp4

Custom duration per image and output fps:
    python scripts/test_folder_to_video.py --model best.pt --source data/validation/images --output val_test.mp4 --seconds 2 --fps 30

Force GPU inference:
    python scripts/test_folder_to_video.py --model best.pt --source data/validation/images --output val_test.mp4 --gpu

Only include images the model actually detected something in:
    python scripts/test_folder_to_video.py --model best.pt --source data/validation/images --output val_test.mp4 --skip-empty
"""

import argparse
import glob
import os
import time
import numpy as np

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Curated, visually distinct palette (BGR for OpenCV) instead of a hash formula,
# so class colors always look intentional rather than randomly muddy.
PALETTE = [
    (244, 133, 66),   # blue
    (83, 168, 52),    # green
    (5, 188, 251),    # amber
    (53, 67, 234),    # red
    (255, 89, 154),   # purple
    (222, 188, 26),   # cyan
    (182, 109, 255),  # pink
    (0, 152, 255),    # orange
]
_class_color_cache = {}

ACCENT = (255, 196, 0)          # cyan-blue accent used for header/footer chrome
PANEL_BG = (38, 32, 28)         # dark navy-charcoal panel background
TEXT_LIGHT = (245, 245, 245)


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


def _color_for_class(class_name):
    # Same class always gets the same palette color, assigned in first-seen
    # order, so colors stay consistent across the whole video.
    if class_name not in _class_color_cache:
        _class_color_cache[class_name] = PALETTE[len(_class_color_cache) % len(PALETTE)]
    return _class_color_cache[class_name]


def draw_rounded_rect(img, pt1, pt2, color, radius=8):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius <= 0:
        cv2.rectangle(img, pt1, pt2, color, -1, cv2.LINE_AA)
        return
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    for cx, cy, a1, a2 in [
        (x1 + radius, y1 + radius, 180, 270),
        (x2 - radius, y1 + radius, 270, 360),
        (x1 + radius, y2 - radius, 90, 180),
        (x2 - radius, y2 - radius, 0, 90),
    ]:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, a1, a2, color, -1, cv2.LINE_AA)


def draw_corner_box(img, x1, y1, x2, y2, color, thickness=3, corner_ratio=0.22):
    w, h = x2 - x1, y2 - y1
    corner_len = max(14, int(min(w, h) * corner_ratio))

    # faint full outline for context, then bold corner accents on top
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * corner_len, py), color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * corner_len), color, thickness, cv2.LINE_AA)


def draw_label(img, x, y, text, color, anchor="above"):
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.55, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 10, 6
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2

    if anchor == "above":
        y1, y2 = y - box_h - 4, y - 4
    else:
        y1, y2 = y + 4, y + box_h + 4
    x1, x2 = x, x + box_w

    draw_rounded_rect(img, (x1, y1), (x2, y2), color, radius=6)
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y), font, scale, TEXT_LIGHT, thick, cv2.LINE_AA)


HEADER_H = 52
FOOTER_H = 6


def draw_results(frame, boxes_data, image_name, index, total, header_h=HEADER_H, footer_h=FOOTER_H):
    """
    boxes_data: list of (x1, y1, x2, y2, class_name, conf) tuples, already in
    the coordinate space of `frame`.

    Assumes the caller has already reserved `header_h` pixels of solid
    background at the top and `footer_h` pixels at the bottom of `frame`.
    """
    frame_h, frame_w = frame.shape[:2]

    # --- Draw header/footer chrome FIRST, so any box/label drawn near the
    # top or bottom of the content area sits on TOP of the chrome instead of
    # being painted over by it. Previously this was drawn last, which is
    # what was clipping labels near the top edge (e.g. "Snickers 91%"). ---
    cv2.rectangle(frame, (0, 0), (frame_w, header_h), PANEL_BG, -1)
    cv2.line(frame, (0, header_h), (frame_w, header_h), ACCENT, 2, cv2.LINE_AA)

    cv2.putText(frame, image_name, (16, 33), cv2.FONT_HERSHEY_DUPLEX, 0.62, TEXT_LIGHT, 1, cv2.LINE_AA)
    right_text = f"{index} / {total}"
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
    cv2.putText(frame, right_text, (frame_w - rw - 16, 33), cv2.FONT_HERSHEY_DUPLEX, 0.62, ACCENT, 1, cv2.LINE_AA)

    bar_y = frame_h - footer_h
    cv2.rectangle(frame, (0, bar_y), (frame_w, frame_h), (25, 22, 20), -1)

    # --- Now draw boxes and labels on top of everything ---
    count = 0
    class_counts = {}
    confidences = []

    for x1, y1, x2, y2, class_name, conf in boxes_data:
        color = _color_for_class(class_name)

        # Clamp label position so it never gets drawn INTO the header strip
        # itself (e.g. a box whose top edge sits right at the header line
        # would otherwise put its label half inside the header text area).
        label_y = max(y1, header_h + 24)

        draw_corner_box(frame, x1, y1, x2, y2, color)
        draw_label(frame, x1, label_y, f"{class_name}  {int(conf * 100)}%", color, anchor="above")

        count += 1
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(conf)

    # Object count text + footer progress bar drawn last, on top, since
    # they're simple overlays that don't conflict with box labels.
    count_text = f"{count} object{'s' if count != 1 else ''} detected"
    cv2.putText(frame, count_text, (16, header_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    progress_w = int(frame_w * (index / total))
    cv2.rectangle(frame, (0, bar_y), (progress_w, frame_h), ACCENT, -1)

    return frame, count, class_counts, confidences

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained weights, e.g. runs_candy/detect/train/weights/best.pt")
    ap.add_argument("--source", required=True, help="Folder of images to test, e.g. data_candy/validation/images")
    ap.add_argument("--output", default="test_output.mp4", help="Path for the output video file")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second for the output video")
    ap.add_argument("--seconds", type=float, default=1.5, help="How many seconds each image is held on screen")
    ap.add_argument("--width", type=int, default=1280, help="Output video width, images are resized to fit")
    ap.add_argument("--height", type=int, default=720, help="Output video height, images are resized to fit")
    ap.add_argument("--gpu", action="store_true", help="Run inference on GPU instead of CPU")
    ap.add_argument("--skip-empty", action="store_true", help="Exclude images with zero detections from the video")
    args = ap.parse_args()

    if not os.path.isdir(args.source):
        raise SystemExit(f"Source folder not found: {args.source}")

    image_paths = sorted(
        p for p in glob.glob(os.path.join(args.source, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No images found in {args.source}")

    device = resolve_device(args.gpu)
    model = YOLO(args.model)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    frames_per_image = max(1, int(args.fps * args.seconds))

    total_images = len(image_paths)
    included_images = 0
    total_detections = 0
    total_class_counts = {}
    all_confidences = []

    print(f"Found {total_images} images in {args.source}")
    print(f"Each image will be held for {args.seconds}s ({frames_per_image} frames) at {args.fps} fps")
    print(f"Writing output video to {args.output}\n")

    start_time = time.time()

    # The header/footer chrome now lives in reserved strips, not overlaid on
    # the image, so the actual photo only gets the remaining vertical space.
    content_h = args.height - HEADER_H - FOOTER_H

    for idx, path in enumerate(image_paths, start=1):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  [skip] Could not read {path}")
            continue

        h, w = frame.shape[:2]
        scale = min(args.width / w, content_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_left = (args.width - new_w) // 2
        pad_top = HEADER_H + (content_h - new_h) // 2  # offset below the header strip

        resized = cv2.resize(frame, (new_w, new_h))

        # Build the full canvas as solid black, then paste the resized image
        # into its designated content area, leaving header/footer strips
        # untouched (pure black) for the chrome to be drawn into.
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Run inference on the ORIGINAL full-res frame for best detection
        # accuracy, then remap box coordinates into canvas space using the
        # same scale + padding offsets used to place the image.
        result = model(frame, conf=args.conf, device=device, verbose=False)[0]
        image_name = os.path.basename(path)

        boxes_data = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]

                sx1 = x1 * scale + pad_left
                sy1 = y1 * scale + pad_top
                sx2 = x2 * scale + pad_left
                sy2 = y2 * scale + pad_top

                boxes_data.append((int(sx1), int(sy1), int(sx2), int(sy2), class_name, conf))

        annotated, count, class_counts, confidences = draw_results(
            canvas, boxes_data, image_name, idx, total_images
        )

        if args.skip_empty and count == 0:
            print(f"  [{idx}/{total_images}] {image_name}: 0 objects, skipped (--skip-empty)")
            continue

        for _ in range(frames_per_image):
            writer.write(annotated)

        included_images += 1
        total_detections += count
        all_confidences.extend(confidences)
        for cls_name, c in class_counts.items():
            total_class_counts[cls_name] = total_class_counts.get(cls_name, 0) + c

        print(f"  [{idx}/{total_images}] {image_name}: {count} object(s) detected")

    writer.release()
    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("TEST RUN SUMMARY")
    print("=" * 50)
    print(f"Images processed:     {total_images}")
    print(f"Images in video:      {included_images}")
    print(f"Total detections:     {total_detections}")
    avg_per_image = total_detections / included_images if included_images else 0
    print(f"Avg detections/image: {avg_per_image:.2f}")
    if all_confidences:
        avg_conf = sum(all_confidences) / len(all_confidences)
        print(f"Avg confidence:       {avg_conf:.2f}")
    if total_class_counts:
        print("Detections by class:")
        for cls_name, c in sorted(total_class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls_name}: {c}")
    print(f"Processing time:      {elapsed:.1f}s")
    print(f"Output video saved:   {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
    
    
# python scripts/test_folder_to_video.py --model runs_candy/detect/train/weights/best.pt --source data_candy/val/images --output val_test.mp4 --seconds 2 --fps 60