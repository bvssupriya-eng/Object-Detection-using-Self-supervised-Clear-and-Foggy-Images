"""Baseline training plan for the four source-only models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BaselineModelConfig:
    """Stores one model's baseline training settings."""

    model_name: str
    weights_name: str
    run_name: str
    epochs: int
    image_size: int
    batch_size: int
    patience: int
    workers: int


DATASET_YAML_PATH = "C:/MachineVision/preprocessing/source_yolo/cityscapes_detection.yaml"

BASELINE_MODELS = [
    BaselineModelConfig(
        model_name="YOLOv8n",
        weights_name="yolov8n.pt",
        run_name="baseline_yolov8n",
        epochs=25,
        image_size=640,
        batch_size=16,
        patience=5,
        workers=4,
    ),
    BaselineModelConfig(
        model_name="YOLOv11n",
        weights_name="yolo11n.pt",
        run_name="baseline_yolov11n",
        epochs=25,
        image_size=640,
        batch_size=16,
        patience=5,
        workers=4,
    ),
    BaselineModelConfig(
        model_name="YOLOv9t",
        weights_name="yolov9t.pt",
        run_name="baseline_yolov9t",
        epochs=20,
        image_size=640,
        batch_size=16,
        patience=5,
        workers=4,
    ),
    BaselineModelConfig(
        model_name="YOLOv5s",
        weights_name="yolov5s.pt",
        run_name="baseline_yolov5s",
        epochs=20,
        image_size=640,
        batch_size=16,
        patience=5,
        workers=4,
    ),
]
