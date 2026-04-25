"""Train one baseline YOLO model using Ultralytics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.config.training_plan import BASELINE_MODELS, DATASET_YAML_PATH


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""

    parser = argparse.ArgumentParser(description="Train one baseline YOLO model.")
    parser.add_argument(
        "--model",
        required=True,
        choices=[config.model_name.lower() for config in BASELINE_MODELS],
        help="Model name to train: yolov8n, yolov11n, yolov9t, or yolov5s.",
    )
    parser.add_argument(
        "--data",
        default=DATASET_YAML_PATH,
        help="Path to the Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--project",
        default="runs/source_baselines",
        help="Output project directory for Ultralytics runs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, e.g. 0 on Kaggle GPU or cpu locally.",
    )
    return parser.parse_args()


def find_model_config(model_name: str):
    """Return the baseline config for one requested model."""

    lookup = {config.model_name.lower(): config for config in BASELINE_MODELS}
    return lookup[model_name]


def main() -> None:
    """Train the requested baseline model."""

    args = parse_args()
    config = find_model_config(args.model)

    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(PROJECT_ROOT / ".workspace" / "ultralytics"),
    )

    print(f"Training {config.model_name}")
    print(f"weights: {config.weights_name}")
    print(f"data: {args.data}")
    print(f"run name: {config.run_name}")

    model = YOLO(config.weights_name)
    train_kwargs = {
        "data": args.data,
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "patience": config.patience,
        "workers": config.workers,
        "project": args.project,
        "name": config.run_name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
