# Defect Detection, Training and Deployment

Detects normal vs defective objects (fruit, bottles, or anything else with a
visible good/bad distinction) using a custom trained YOLO model. Ctrl+C safe resume, GPU toggle, plus a script to collapse any multi-class
defect dataset down to a simple binary normal/defect scheme.

## 1. Setup

```bash
conda env create -f environment.yml
conda activate defect-detection-env
```

CPU by default. To add GPU support later without recreating the environment:
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 2. Getting datasets

This is set up to train on multiple datasets merged into one, since a single
model that has seen fruit, bottles, parcels, glass, aluminium, and welding
defects generalizes better to "defect detection in general" than nine
separate single-object models would.

### Datasets used

**Fruit:**
- Fruit Defect Detection (aged / damaged / diseased / fresh, 77 images):
  https://universe.roboflow.com/manohar-singh-rajawat/fruit-defect-detection

**Bottles:**
- Bottle Defects, cap condition (Broken Cap / Broken Ring / Good Cap / Loose Cap / No Cap, 623 images):
  https://universe.roboflow.com/product-defect-detection/bottle-defects-o4gnx
- Bottle Crack Detection (305 images):
  https://universe.roboflow.com/daff/bottle-crack-detection
- Bottle Defect Detection, cap/crumble condition (262 images):
  https://universe.roboflow.com/spark-intelligence-scqhh/bottle-defect-detection

**Other objects:**
- Defect Parcel (328 images):
  https://universe.roboflow.com/ip-ass-fqxdg/defect-parcel
- Parcels with Defect (225 images):
  https://universe.roboflow.com/john-koleosho-3hd0h/parcels-with-defect
- Glass Defect Detection (684 images):
  https://universe.roboflow.com/tongji-university-lhe7b/glass-defect-detection
- Aluminium Defect Detection, instance segmentation (480 images):
  https://universe.roboflow.com/lethfinv/alumunium-defect-detection

**Added as the highest-usage defect dataset found:** Welding Defect Detection
by Final Year Project, 408 images, 553 downloads/uses at the time this was
written, the highest of any defect detection dataset found across a search
of Roboflow Universe's defect category.
https://universe.roboflow.com/final-year-project-kswbt/welding-defect-detection

### Downloading each one

```bash
pip install roboflow
python -c "
import roboflow
# roboflow.login()
roboflow.download_dataset(
    dataset_url='https://universe.roboflow.com/final-year-project-kswbt/welding-defect-detection/2',
    model_format='yolov11',
    location='downloads/welding_defect'
)
"
```
Repeat for each dataset, changing `dataset_url` and `location`. Authenticate
once with a free Roboflow account. Each download lands already split into
train/valid/test, in Roboflow's standard export layout.

### Merging them into one dataset

```bash
python scripts/merge_datasets.py \
    --sources fruit=downloads/fruit_defect \
              bottle_caps=downloads/bottle_caps \
              bottle_crack=downloads/bottle_crack \
              bottle_defect=downloads/bottle_defect \
              parcel1=downloads/defect_parcel \
              parcel2=downloads/parcels_with_defect \
              glass=downloads/glass_defect \
              aluminium=downloads/aluminium_defect \
              welding=downloads/welding_defect \
    --out data
```

This prints the class mapping it's about to apply for every single dataset
before copying anything, review each one, a class name that doesn't clearly
signal good or bad defaults to "defect" (safer to over-flag than to
silently call something normal that wasn't). Images are copied with a
per-dataset filename prefix so nothing collides, then combined into one
`data/train` and `data/val`.

Then copy `data.yaml.example` to `data.yaml`, it's already set to `nc: 2`,
`names: ["normal", "defect"]`, matching what the merge script outputs, just
update the `path:` line to your merged `data/` folder's absolute path.

### If you only want one dataset, not all of them

Skip `merge_datasets.py` entirely, use `scripts/relabel_to_binary.py` on a
single downloaded dataset instead, exactly as in the coin/single-dataset
workflow. Both scripts share the same keyword-matching logic, so the results
are consistent whichever path you take.

## 3. Train

```bash
# CPU (default)
python scripts/train.py --data data.yaml --model yolo11s.pt --epochs 60

# GPU
python scripts/train.py --data data.yaml --model yolo11s.pt --epochs 60 --gpu

# Resume after Ctrl+C or a crash
python scripts/train.py --data data.yaml --resume
```

Same checkpointing behavior as before: `last.pt`/`best.pt` every epoch,
periodic numbered snapshots (`--checkpoint-interval`, default every 5
epochs), only the most recent 3 kept (`--keep-checkpoints`), older ones
pruned automatically. `--project` is resolved to an absolute path
immediately so `--resume` always finds the right run regardless of which
directory you launch it from.

## 4. Run the trained model

```bash
python scripts/detect.py --model runs/detect/train/weights/best.pt --source usb0
python scripts/detect.py --model runs/detect/train/weights/best.pt --source test.jpg
python scripts/detect.py --model runs/detect/train/weights/best.pt --source test_vid.mp4
```

Same overlay as before, bounding box, class label, confidence, FPS, object
count, matching the visual style in the reference screenshots (green for
normal, a distinct color for defect, driven automatically by class id).


## Push to GitHub with LFS

Checkpoints and weights go
through Git LFS (`.gitattributes` already configured for `*.pt`), the raw
dataset stays out of git via `.gitignore`.

```bash
git init
git lfs install
git add .
git commit -m "Initial commit: defect detection training and deployment scripts"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/defect-detection.git
git push -u origin main
```
