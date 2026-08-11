"""
scripts/merge_datasets.py
============================
Combines multiple downloaded Roboflow datasets, each with its own class
naming scheme, into one unified train/val dataset with a single binary
normal/defect labeling. This is what makes it possible to train one model
across the fruit, bottle, parcel, glass, aluminium, and welding datasets
listed in the README at once, instead of training separate models per
object type.

Each source dataset is expected to already be downloaded in Roboflow's
standard YOLO export layout:

    some_dataset_folder/
    ├── data.yaml          (defines nc and names for THIS dataset)
    ├── train/images, train/labels
    ├── valid/images, valid/labels   (Roboflow calls it "valid", we call it "val")
    └── test/images, test/labels     (optional, folded into val if present)

For each source, every class name is first matched against --normal-keywords
(word/phrase boundary matching, not substring), anything that looks like a
good state becomes class 0 (normal), everything else becomes class 1
(defect). The mapping is printed per dataset before anything is copied,
review it.

KEYWORD MATCHING HAS A CEILING. It only works when a dataset's class names
literally describe a good/bad STATE ("fresh" vs "damaged"). When a dataset's
classes describe object PARTS instead (e.g. "cap", "label", "workpiece",
"welding line"), no keyword list can get it right on its own — a part name
isn't inherently normal or defective. For those, use --class-override to
hand-correct specific classes per dataset, e.g.:

    --class-override bottle_defect:not-crumbled=normal \
                      welding:Workpiece=normal \
                      welding:"Welding Line"=normal

Overrides are applied AFTER keyword matching and always win. Matching for
both keywords and overrides is case-insensitive and ignores how the class
name separates words (space, "-", or "_" are treated the same).

Images are copied (not moved) into the unified output, renamed with a
per-dataset prefix to avoid filename collisions between datasets that
happen to both have "image_001.jpg".

USAGE
-----
    python scripts/merge_datasets.py \
        --sources fruit_defect=path/to/fruit-defect-detection \
                  bottle_caps=path/to/bottle-defects-o4gnx \
                  bottle_crack=path/to/bottle-crack-detection \
                  welding=path/to/welding-defect-detection \
        --class-override bottle_defect:not-crumbled=normal \
        --out data
"""

import argparse
import os
import re
import shutil

import yaml

# NOTE: keep these as whole words/phrases, not fragments. Matching is done
# on word/phrase boundaries (see build_mapping), so a keyword like "ok" only
# matches the standalone word "ok", never a fragment inside another word.
# Even so, keyword matching only works for STATE class names (good/bad).
# For PART class names (cap, label, workpiece, ...), use --class-override.
DEFAULT_NORMAL_KEYWORDS = ["good", "normal", "fresh", "ok", "intact", "no defect", "no_defect", "no-defect"]


def load_source_classnames(source_dir):
    data_yaml_path = os.path.join(source_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        raise SystemExit(f"No data.yaml found in {source_dir}, expected Roboflow's standard export layout.")
    with open(data_yaml_path) as f:
        config = yaml.safe_load(f)
    names = config.get("names")
    if isinstance(names, dict):
        # Some exports use {0: "name0", 1: "name1"} instead of a plain list
        names = [names[i] for i in sorted(names)]
    return names


def _normalize(text):
    """Lowercase and collapse separators (-, _, whitespace) to single spaces
    so 'no-defect', 'no_defect', and 'no defect' all compare the same way,
    and so word-boundary matching works consistently regardless of how the
    dataset author separated words in the class name."""
    return re.sub(r"[\s\-_]+", " ", text.lower()).strip()


def build_mapping(classnames, normal_keywords, overrides=None):
    """
    overrides: optional dict of normalized-classname -> 0/1, applied after
    keyword matching. Takes precedence over keyword-based classification.
    """
    overrides = overrides or {}
    mapping = {}
    normalized_keywords = [_normalize(kw) for kw in normal_keywords]
    for i, name in enumerate(classnames):
        normalized_name = _normalize(name)

        if normalized_name in overrides:
            mapping[i] = overrides[normalized_name]
            continue

        is_normal = any(
            re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", normalized_name)
            for kw in normalized_keywords
        )
        mapping[i] = 0 if is_normal else 1
    return mapping


def parse_overrides(raw_overrides):
    """
    Parses entries like 'tag:classname=label' (label is 'normal' or 'defect')
    into {tag: {normalized_classname: 0_or_1}}.
    """
    overrides = {}
    if not raw_overrides:
        return overrides

    for item in raw_overrides:
        if ":" not in item or "=" not in item:
            raise SystemExit(
                f"--class-override entries must be tag:classname=normal|defect, got: {item}"
            )
        tag_part, rest = item.split(":", 1)
        classname, label = rest.rsplit("=", 1)
        tag = tag_part.strip()
        classname = classname.strip().strip('"').strip("'")
        label = label.strip().lower()

        if label not in ("normal", "defect"):
            raise SystemExit(f"--class-override label must be 'normal' or 'defect', got: {item}")

        overrides.setdefault(tag, {})[_normalize(classname)] = 0 if label == "normal" else 1

    return overrides


def find_split_dir(source_dir, split_names):
    for name in split_names:
        candidate = os.path.join(source_dir, name)
        if os.path.isdir(os.path.join(candidate, "images")):
            return candidate
    return None


def merge_split(source_dir, split_names, out_split_dir, tag, mapping):
    split_dir = find_split_dir(source_dir, split_names)
    if not split_dir:
        return 0

    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    out_images_dir = os.path.join(out_split_dir, "images")
    out_labels_dir = os.path.join(out_split_dir, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(images_dir):
        stem, ext = os.path.splitext(filename)
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            continue

        new_stem = f"{tag}_{stem}"
        shutil.copy2(os.path.join(images_dir, filename), os.path.join(out_images_dir, f"{new_stem}{ext}"))

        new_lines = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = mapping.get(old_cls, 1)  # unrecognized class ids default to defect, not silently dropped
                new_lines.append(" ".join([str(new_cls)] + parts[1:]))

        with open(os.path.join(out_labels_dir, f"{new_stem}.txt"), "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        count += 1

    return count


def parse_sources(raw_sources):
    """Parses tag=path pairs from the command line into a dict."""
    sources = {}
    for item in raw_sources:
        if "=" not in item:
            raise SystemExit(f"--sources entries must be tag=path, got: {item}")
        tag, path = item.split("=", 1)
        tag = tag.strip()
        path = path.strip()
        if not os.path.isdir(path):
            raise SystemExit(f"Source path does not exist: {path}")
        sources[tag] = path
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                     help="One or more tag=path pairs, e.g. fruit=path/to/dataset bottle=path/to/other")
    ap.add_argument("--out", default="data", help="Unified output folder (default: ./data)")
    ap.add_argument("--normal-keywords", default=",".join(DEFAULT_NORMAL_KEYWORDS),
                     help="Comma separated keywords used to detect 'normal' classes across all sources")
    ap.add_argument("--class-override", nargs="*", default=[],
                     help="Hand-correct specific classes that keyword matching can't get right "
                          "(e.g. part names like 'cap' or 'workpiece'). Format: "
                          "tag:classname=normal|defect, space separated, one per class. "
                          "Example: bottle_defect:not-crumbled=normal welding:Workpiece=normal")
    args = ap.parse_args()

    sources = parse_sources(args.sources)
    normal_keywords = [k.strip() for k in args.normal_keywords.split(",") if k.strip()]
    overrides_by_tag = parse_overrides(args.class_override)

    total_train = 0
    total_val = 0

    for tag, source_dir in sources.items():
        print(f"\n=== {tag} ({source_dir}) ===")
        classnames = load_source_classnames(source_dir)
        tag_overrides = overrides_by_tag.get(tag, {})
        mapping = build_mapping(classnames, normal_keywords, tag_overrides)

        # Warn about overrides that were specified but never matched a real
        # class name in this dataset — usually a typo in --class-override.
        matched_names = {_normalize(n) for n in classnames}
        for override_name in tag_overrides:
            if override_name not in matched_names:
                print(f"  WARNING: --class-override for '{override_name}' did not match any class in {tag}")

        for i, name in enumerate(classnames):
            label = "normal" if mapping[i] == 0 else "defect"
            note = " (override)" if _normalize(name) in tag_overrides else ""
            print(f"  [{i}] {name:<25} -> {label}{note}")

        n_train = merge_split(source_dir, ["train"], os.path.join(args.out, "train"), tag, mapping)
        n_val = merge_split(source_dir, ["valid", "val"], os.path.join(args.out, "val"), tag, mapping)
        n_test = merge_split(source_dir, ["test"], os.path.join(args.out, "val"), tag, mapping)  # fold test into val

        print(f"  Copied: {n_train} train, {n_val + n_test} val")
        total_train += n_train
        total_val += n_val + n_test

    print(f"\nMerged {len(sources)} dataset(s) into {os.path.abspath(args.out)}")
    print(f"Total: {total_train} train images, {total_val} val images")
    print()
    print("Set your data.yaml to:")
    print(f"  path: {os.path.abspath(args.out)}")
    print("  train: train/images")
    print("  val: val/images")
    print("  nc: 2")
    print('  names: ["normal", "defect"]')


if __name__ == "__main__":
    main()


'''
python scripts/merge_datasets.py --sources fruit=downloads/fruit_defect bottle_caps=downloads/bottle_caps bottle_crack=downloads/bottle_crack bottle_defect=downloads/bottle_defect parcel1=downloads/defect_parcel parcel2=downloads/parcels_with_defect glass=downloads/glass_defect aluminium=downloads/aluminium_defect welding=downloads/welding_defect --class-override bottle_defect:not-crumbled=normal welding:Workpiece=normal welding:"Welding Line"=normal --out data
'''


'''
python scripts/merge_datasets.py --sources fruit=downloads/fruit_defect bottle_caps=downloads/bottle_caps bottle_crack=downloads/bottle_crack bottle_defect=downloads/bottle_defect parcel1=downloads/defect_parcel parcel2=downloads/parcels_with_defect glass=downloads/glass_defect aluminium=downloads/aluminium_defect welding=downloads/welding_defect --out data
'''
