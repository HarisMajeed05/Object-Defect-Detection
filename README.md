# Object Defect Detection

Trains a YOLO model to tell normal objects from defective ones (fruit, bottles, parcels, glass, aluminium, welds). Training can be stopped with Ctrl+C and resumed, and there is a switch for GPU.

## Setup

```bash
conda env create -f environment.yml
conda activate defect-detection-env
```

The environment uses the CPU build of PyTorch. To add GPU support later without recreating it:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Datasets

All datasets come from Roboflow Universe. Training on several of them together gives one model that covers many kinds of defects.

| Object | Dataset | Images |
|---|---|---|
| Fruit | [Fruit Defect Detection](https://universe.roboflow.com/manohar-singh-rajawat/fruit-defect-detection) | 77 |
| Bottles | [Bottle Defects](https://universe.roboflow.com/product-defect-detection/bottle-defects-o4gnx) | 623 |
| Bottles | [Bottle Crack Detection](https://universe.roboflow.com/daff/bottle-crack-detection) | 305 |
| Bottles | [Bottle Defect Detection](https://universe.roboflow.com/spark-intelligence-scqhh/bottle-defect-detection) | 262 |
| Parcels | [Defect Parcel](https://universe.roboflow.com/ip-ass-fqxdg/defect-parcel) | 328 |
| Parcels | [Parcels with Defect](https://universe.roboflow.com/john-koleosho-3hd0h/parcels-with-defect) | 225 |
| Glass | [Glass Defect Detection](https://universe.roboflow.com/tongji-university-lhe7b/glass-defect-detection) | 684 |
| Aluminium | [Aluminium Defect Detection](https://universe.roboflow.com/lethfinv/alumunium-defect-detection) | 480 |
| Welding | [Welding Defect Detection](https://universe.roboflow.com/final-year-project-kswbt/welding-defect-detection) | 408 |

### Download

Create a free Roboflow account and log in once. Then download each dataset in YOLOv11 format:

```bash
pip install roboflow
python -c "
import roboflow
roboflow.download_dataset(
    dataset_url='https://universe.roboflow.com/final-year-project-kswbt/welding-defect-detection/2',
    model_format='yolov11',
    location='downloads/welding_defect'
)
"
```

Repeat for each dataset, changing `dataset_url` and `location`.

### Merge into one dataset

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

Before copying anything, the script prints the class mapping for each dataset. Check each one. A class name that does not clearly mean good or bad becomes `defect`. Images get a per dataset prefix so names do not clash, and everything ends up in `data/train` and `data/val`.

Then create `data.yaml` in the project root:

```yaml
path: <absolute path to the data folder>
train: train/images
val: val/images
nc: 2
names: ["normal", "defect"]
```

### Using a single dataset

Skip the merge step and run `scripts/relabel_to_binary.py` on the one dataset you downloaded. It uses the same class matching as the merge script.

## Train

```bash
# CPU
python scripts/train.py --data data.yaml --model yolo11s.pt --epochs 60

# GPU
python scripts/train.py --data data.yaml --model yolo11s.pt --epochs 60 --gpu

# Resume after Ctrl+C or a crash
python scripts/train.py --data data.yaml --resume
```

`last.pt` and `best.pt` are saved every epoch. Numbered snapshots are also saved every 5 epochs (`--checkpoint-interval`), and only the latest 3 are kept (`--keep-checkpoints`). `--project` is turned into an absolute path, so `--resume` works from any folder.

## Run the trained model

```bash
python scripts/detect.py --model runs/detect/train/weights/best.pt --source usb0
python scripts/detect.py --model runs/detect/train/weights/best.pt --source test.jpg
python scripts/detect.py --model runs/detect/train/weights/best.pt --source test_vid.mp4
```

The output shows the box, class, confidence, FPS and object count. Normal objects are drawn in green and defects in another color.
