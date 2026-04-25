# Baseline Training First Run

Start with **YOLOv8n** first. Once that works on Kaggle, reuse the same flow for the other three models.

## First Model

- model: `YOLOv8n`
- run name: `baseline_yolov8n`
- dataset yaml: `preprocessing/source_yolo/cityscapes_detection.yaml`

## What To Upload To Kaggle

1. the processed dataset folder:
   - `preprocessing/source_yolo`
2. the training script:
   - `src/training/train_baseline_ultralytics.py`
3. the training plan reference:
   - `src/config/training_plan.py`

## Kaggle Setup

Install Ultralytics if needed:

```bash
pip install ultralytics
```

## First Run Command

```bash
python src/training/train_baseline_ultralytics.py --model yolov8n --data /kaggle/working/source_yolo/cityscapes_detection.yaml --device 0
```

## What To Save After This Run

- `best.pt`
- `last.pt`
- training curves
- validation metrics
- sample prediction images

## After YOLOv8n Works

Repeat for:

- `yolov11n`
- `yolov9t`
- `yolov5s`
