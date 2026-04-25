"""Quality checks and summary reports for the preprocessing step."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.classes import CITYSCAPES_DETECTION_CLASSES


PREPROCESSING_ROOT = PROJECT_ROOT / "preprocessing"
SOURCE_YOLO_ROOT = PREPROCESSING_ROOT / "source_yolo"


def build_label_path(image_path: Path, split: str) -> Path:
    """Map a processed image path to its YOLO label path."""

    relative_path = image_path.relative_to(SOURCE_YOLO_ROOT / "images" / split)
    label_name = relative_path.name.replace("_leftImg8bit.png", ".txt")
    return SOURCE_YOLO_ROOT / "labels" / split / relative_path.parent / label_name


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO label rows from one file."""

    if not label_path.exists():
        return []

    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        class_id_text, x_center_text, y_center_text, width_text, height_text = stripped.split()
        rows.append(
            (
                int(class_id_text),
                float(x_center_text),
                float(y_center_text),
                float(width_text),
                float(height_text),
            )
        )
    return rows


def create_preprocessing_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build image-level and object-level preprocessing summaries."""

    image_rows: list[dict[str, int | str]] = []
    object_rows: list[dict[str, float | str]] = []

    for split in ("train", "val", "test"):
        image_paths = sorted((SOURCE_YOLO_ROOT / "images" / split).rglob("*.png"))
        for image_path in image_paths:
            label_path = build_label_path(image_path, split)
            label_rows = parse_label_file(label_path)
            class_names = [CITYSCAPES_DETECTION_CLASSES[row[0]] for row in label_rows]
            image_rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "object_count": len(label_rows),
                    "is_empty": int(len(label_rows) == 0),
                    "classes_present": ", ".join(sorted(set(class_names))),
                }
            )

            for class_id, x_center, y_center, width, height in label_rows:
                object_rows.append(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "class_name": CITYSCAPES_DETECTION_CLASSES[class_id],
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "area_ratio": width * height,
                    }
                )

    return pd.DataFrame(image_rows), pd.DataFrame(object_rows)


def save_empty_label_summary(image_df: pd.DataFrame) -> None:
    """Save summary of empty vs labeled images."""

    summary_rows = []
    for split_name, split_df in image_df.groupby("split"):
        total_images = len(split_df)
        empty_images = int(split_df["is_empty"].sum())
        labeled_images = total_images - empty_images
        summary_rows.append(
            {
                "split": split_name,
                "total_images": total_images,
                "labeled_images": labeled_images,
                "empty_images": empty_images,
                "empty_image_ratio": round(empty_images / total_images, 6),
            }
        )

    pd.DataFrame(summary_rows).to_csv(PREPROCESSING_ROOT / "empty_label_summary.csv", index=False)


def save_bbox_quality_summary(object_df: pd.DataFrame) -> None:
    """Save summary of possible YOLO box quality issues."""

    invalid_boxes = (
        (object_df["width"] <= 0.0)
        | (object_df["height"] <= 0.0)
        | (object_df["x_center"] < 0.0)
        | (object_df["x_center"] > 1.0)
        | (object_df["y_center"] < 0.0)
        | (object_df["y_center"] > 1.0)
        | (object_df["width"] > 1.0)
        | (object_df["height"] > 1.0)
    )
    tiny_boxes = (object_df["width"] < 0.005) | (object_df["height"] < 0.005)

    summary = pd.DataFrame(
        [
            {
                "total_boxes": int(len(object_df)),
                "invalid_boxes": int(invalid_boxes.sum()),
                "tiny_boxes": int(tiny_boxes.sum()),
                "min_width": float(object_df["width"].min()) if not object_df.empty else 0.0,
                "min_height": float(object_df["height"].min()) if not object_df.empty else 0.0,
                "max_width": float(object_df["width"].max()) if not object_df.empty else 0.0,
                "max_height": float(object_df["height"].max()) if not object_df.empty else 0.0,
            }
        ]
    )
    summary.to_csv(PREPROCESSING_ROOT / "bbox_quality_summary.csv", index=False)


def save_preprocessing_report(image_df: pd.DataFrame, object_df: pd.DataFrame) -> None:
    """Save main preprocessing reports."""

    image_df.to_csv(PREPROCESSING_ROOT / "preprocessing_report.csv", index=False)
    object_df.to_csv(PREPROCESSING_ROOT / "preprocessing_objects_report.csv", index=False)


def save_foggy_target_manifest() -> None:
    """Save a clean manifest of target foggy images for later pseudo-labeling."""

    foggy_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"
    rows: list[dict[str, str]] = []

    for split in ("train", "val", "test"):
        for image_path in sorted((foggy_root / split).rglob("*.png")):
            fog_level = image_path.stem.split("_beta_")[1] if "_beta_" in image_path.stem else "unknown"
            rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "fog_beta": fog_level,
                    "city": image_path.parent.name,
                }
            )

    pd.DataFrame(rows).to_csv(PREPROCESSING_ROOT / "foggy_target_manifest.csv", index=False)


def save_class_presence_summary(image_df: pd.DataFrame, object_df: pd.DataFrame) -> None:
    """Save selected class presence summary for train and val splits."""

    counter: Counter = Counter(object_df["class_name"].tolist())
    rows = [
        {
            "class_name": class_name,
            "object_count": counter.get(class_name, 0),
        }
        for class_name in CITYSCAPES_DETECTION_CLASSES
    ]
    pd.DataFrame(rows).to_csv(PREPROCESSING_ROOT / "class_presence_summary.csv", index=False)


def main() -> None:
    """Run all preprocessing checks and save reports."""

    PREPROCESSING_ROOT.mkdir(parents=True, exist_ok=True)
    image_df, object_df = create_preprocessing_report()
    save_preprocessing_report(image_df, object_df)
    save_empty_label_summary(image_df)
    save_bbox_quality_summary(object_df)
    save_foggy_target_manifest()
    save_class_presence_summary(image_df, object_df)
    print(f"[done] preprocessing reports saved to {PREPROCESSING_ROOT}")


if __name__ == "__main__":
    main()
