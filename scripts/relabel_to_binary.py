"""
scripts/relabel_to_binary.py
==============================
Most real defect datasets (Roboflow Universe included) aren't already
binary, they use specific class names like "Broken Cap", "Good Cap",
"Loose Cap", "aged", "fresh", "diseased". This script rewrites YOLO label
files so every class becomes either "normal" (class 0) or "defect" (class 1),
based on a simple keyword match against each original class name.

Anything that looks like a "good" state (matches one of --normal-keywords)
becomes normal. Everything else becomes defect. This is a blunt instrument,
review the printed mapping before trusting it, a class name that doesn't
clearly signal good/bad either way will default to defect, better to
mislabel conservatively than to quietly call something normal that wasn't.

Works directly on the train/val folder structure created by
train_val_split.py. Original label files are backed up before anything is
overwritten, nothing is lost if the mapping needs adjusting and rerunning.

USAGE
-----
    python scripts/relabel_to_binary.py --data-root data --classnames-file classes.txt

    python scripts/relabel_to_binary.py --data-root data --classnames-file classes.txt \
        --normal-keywords good,normal,fresh,ok,intact,no defect
"""

import argparse
import os
import shutil


DEFAULT_NORMAL_KEYWORDS = ["good", "normal", "fresh", "ok", "intact", "no defect", "no_defect"]


def load_classnames(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_mapping(classnames, normal_keywords):
    """Returns {old_index: new_index} where new_index is 0 (normal) or 1 (defect)."""
    mapping = {}
    for i, name in enumerate(classnames):
        lowered = name.lower()
        is_normal = any(kw.lower() in lowered for kw in normal_keywords)
        mapping[i] = 0 if is_normal else 1
    return mapping


def print_mapping(classnames, mapping):
    print("Class mapping (review this before trusting the result):")
    for i, name in enumerate(classnames):
        new_label = "normal" if mapping[i] == 0 else "defect"
        print(f"  [{i}] {name:<30} -> {new_label}")
    print()


def relabel_folder(labels_dir, mapping):
    backup_dir = labels_dir.rstrip("/\\") + "_original_backup"
    if os.path.exists(backup_dir):
        print(f"Backup already exists at {backup_dir}, skipping backup (using existing original).")
    else:
        shutil.copytree(labels_dir, backup_dir)
        print(f"Backed up original labels to: {backup_dir}")

    changed = 0
    for filename in os.listdir(backup_dir):
        if not filename.endswith(".txt"):
            continue
        src_path = os.path.join(backup_dir, filename)
        dst_path = os.path.join(labels_dir, filename)

        new_lines = []
        with open(src_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = mapping.get(old_cls)
                if new_cls is None:
                    print(f"  Warning: class id {old_cls} in {filename} has no mapping, skipping this box")
                    continue
                new_lines.append(" ".join([str(new_cls)] + parts[1:]))

        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        changed += 1

    print(f"Rewrote {changed} label file(s) in {labels_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Folder containing train/ and val/ subfolders")
    ap.add_argument("--classnames-file", required=True,
                     help="Path to the dataset's classes.txt (one class name per line, in index order)")
    ap.add_argument("--normal-keywords", default=",".join(DEFAULT_NORMAL_KEYWORDS),
                     help="Comma separated keywords, any class name containing one of these "
                          "(case-insensitive) becomes 'normal', everything else becomes 'defect'")
    args = ap.parse_args()

    classnames = load_classnames(args.classnames_file)
    normal_keywords = [k.strip() for k in args.normal_keywords.split(",") if k.strip()]
    mapping = build_mapping(classnames, normal_keywords)
    print_mapping(classnames, mapping)

    found_any = False
    for split in ("train", "val"):
        labels_dir = os.path.join(args.data_root, split, "labels")
        if os.path.isdir(labels_dir):
            found_any = True
            relabel_folder(labels_dir, mapping)

    if not found_any:
        raise SystemExit(
            f"No train/labels or val/labels found under {args.data_root}. "
            f"Run scripts/train_val_split.py first."
        )

    print()
    print("Done. Your data.yaml should now use:")
    print('  nc: 2')
    print('  names: ["normal", "defect"]')


if __name__ == "__main__":
    main()
