"""
scripts/train.py
==================
Trains a YOLO11 (or YOLOv8/YOLOv5) object detection model, with three things
layered on top of a bare `yolo detect train` command:

1. Runs on CPU by default, switches to GPU with a single --gpu flag, and
   falls back to CPU with a clear warning if you ask for GPU but none is
   available, rather than crashing.

2. Saves periodic checkpoint snapshots during training (--checkpoint-interval),
   not just the standard last.pt/best.pt every epoch, so a crash partway
   through a long run doesn't cost you more progress than one epoch. Only
   keeps the most recent few of these (--keep-checkpoints, default 3),
   older numbered snapshots are deleted automatically as new ones are saved,
   so this doesn't quietly fill up your disk over a long training run.

3. Supports --resume to continue training exactly from where a previous run
   left off, same optimizer state, same epoch count, not starting over.

HONEST LIMITATION, worth knowing up front: checkpointing here is per-epoch,
not per-instant. Ultralytics writes last.pt at the end of each completed
epoch. If you press Ctrl+C mid-epoch, that partial epoch's progress is not
saved, training resumes from the end of the last FULLY completed epoch. This
is how the underlying training loop actually works, there's no meaningful way
to checkpoint mid-batch without corrupting the optimizer state, so this
script doesn't pretend otherwise.

USAGE
-----
Fresh training run, CPU (default):
    python scripts/train.py --data data.yaml --model yolo11s.pt --epochs 60

Fresh training run, GPU:
    python scripts/train.py --data data.yaml --model yolo11s.pt --epochs 60 --gpu

Resume the most recent interrupted run:
    python scripts/train.py --data data.yaml --resume

Stop early any time with Ctrl+C, the last completed epoch's checkpoint is
already saved on disk, just rerun with --resume when you're ready to continue.
"""

import argparse
import glob
import os
import re
import sys

from ultralytics import YOLO


def prune_old_checkpoints(weights_dir: str, keep: int):
    """Keeps only the most recent N numbered checkpoint snapshots (epoch5.pt,
    epoch10.pt, etc), deleting older ones as new ones appear. Only touches
    files matching that epochN.pt pattern, last.pt and best.pt are never
    affected by this, they're a different thing and always kept."""
    pattern = os.path.join(weights_dir, "epoch*.pt")
    files = glob.glob(pattern)

    def epoch_number(path):
        match = re.search(r"epoch(\d+)\.pt$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    files.sort(key=epoch_number)
    while len(files) > keep:
        oldest = files.pop(0)
        try:
            os.remove(oldest)
            print(f"Pruned old checkpoint: {os.path.basename(oldest)}")
        except OSError as e:
            print(f"Could not remove {oldest}: {e}")


def resolve_device(want_gpu: bool) -> str:
    if not want_gpu:
        return "cpu"

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed at all, falling back to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU requested and available: {name}")
        return "0"

    print(
        "GPU was requested with --gpu, but no CUDA-capable GPU was detected "
        "(or the GPU build of PyTorch isn't installed). Falling back to CPU. "
        "See the README for the command to install the GPU build of PyTorch."
    )
    return "cpu"


def find_last_checkpoint(project: str, name: str) -> str:
    expected = os.path.join(project, name, "weights", "last.pt")
    if os.path.exists(expected):
        return expected

    # Fall back to searching anywhere under the project root, this is what
    # saves you if a previous run wrote to a slightly different path than
    # what you're resuming with (e.g. the classic nested runs/detect/runs/detect
    # problem caused by a relative --project resolving differently between
    # two terminal sessions with different working directories).
    candidates = []
    for root, _, files in os.walk(project):
        if "last.pt" in files:
            candidates.append(os.path.join(root, "last.pt"))

    if not candidates:
        raise SystemExit(
            f"--resume was given but no checkpoint was found at {expected}, "
            f"and no last.pt was found anywhere under {os.path.abspath(project)} either. "
            f"Check --project/--name match the run you're trying to resume, "
            f"or omit --resume to start a fresh run."
        )

    if len(candidates) == 1:
        print(f"Note: expected checkpoint not found at {expected}, "
              f"but found one nearby and using it instead: {candidates[0]}")
        return candidates[0]

    print(f"Note: expected checkpoint not found at {expected}. "
          f"Found multiple last.pt files under {os.path.abspath(project)}, pick one:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c}")
    choice = input("Enter the number to resume from: ").strip()
    return candidates[int(choice)]


def main():
    ap = argparse.ArgumentParser(description="Train a YOLO detection model, CPU or GPU")
    ap.add_argument("--data", default="data.yaml", help="Path to the dataset YAML config")
    ap.add_argument("--model", default="yolo11s.pt",
                     help="Base model to start from, e.g. yolo11n.pt / yolo11s.pt / yolo11m.pt "
                          "(also accepts yolov8*.pt or yolov5*.pt). Ignored if --resume is set.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1,
                     help="Batch size, -1 lets Ultralytics auto-select based on available memory")
    ap.add_argument("--gpu", action="store_true", help="Train on GPU instead of CPU")
    ap.add_argument("--checkpoint-interval", type=int, default=5,
                     help="Save a numbered checkpoint every N epochs, in addition to last.pt/best.pt "
                          "which are always saved every epoch (default: every 5 epochs)")
    ap.add_argument("--keep-checkpoints", type=int, default=3,
                     help="Only keep this many most recent numbered checkpoints, older ones are "
                          "deleted automatically as new ones are saved (default: 3). "
                          "last.pt and best.pt are never affected by this.")
    ap.add_argument("--project", default="runs/detect", help="Where training runs are saved")
    ap.add_argument("--name", default="train", help="Name of this training run")
    ap.add_argument("--resume", action="store_true",
                     help="Resume the run at --project/--name from its last checkpoint")
    args = ap.parse_args()

    # Pin this to one exact absolute location right away. A relative
    # --project path resolves differently depending on the working
    # directory the command happens to be run from, which is exactly what
    # causes the classic nested runs/detect/runs/detect folder bug if a
    # training run and its later --resume are launched from different
    # terminal sessions.
    args.project = os.path.abspath(args.project)

    device = resolve_device(args.gpu)

    if args.resume:
        checkpoint = find_last_checkpoint(args.project, args.name)
        print(f"Resuming from: {checkpoint}")
        model = YOLO(checkpoint)
    else:
        print(f"Starting fresh training from base model: {args.model}")
        model = YOLO(args.model)

    print(f"Device: {'GPU' if device != 'cpu' else 'CPU'} | Epochs: {args.epochs} | "
          f"Image size: {args.imgsz} | Checkpoint every {args.checkpoint_interval} epoch(s), "
          f"keeping the last {args.keep_checkpoints}")
    print(f"Saving to: {os.path.join(args.project, args.name)}")
    print("(this is now always an absolute path, so --resume finds it the same way every time)")

    def on_model_save(trainer):
        weights_dir = str(trainer.save_dir / "weights")
        prune_old_checkpoints(weights_dir, keep=args.keep_checkpoints)

    model.add_callback("on_model_save", on_model_save)

    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            save=True,                              # always save last.pt / best.pt each epoch
            save_period=args.checkpoint_interval,    # additionally save epochN.pt snapshots
            resume=args.resume,
        )
    except KeyboardInterrupt:
        weights_dir = os.path.join(args.project, args.name, "weights")
        print()
        print("Training stopped early (Ctrl+C).")
        print(f"Your last completed epoch is saved at: {os.path.join(weights_dir, 'last.pt')}")
        print("To continue training from exactly this point, run:")
        print(f"    python scripts/train.py --data {args.data} --resume "
              f"--project {args.project} --name {args.name} --epochs {args.epochs}")
        sys.exit(0)

    print()
    print("Training complete.")
    print(f"Best weights: {os.path.join(args.project, args.name, 'weights', 'best.pt')}")
    print(f"Last weights: {os.path.join(args.project, args.name, 'weights', 'last.pt')}")


if __name__ == "__main__":
    main()
