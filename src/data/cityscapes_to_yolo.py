"""Convert Cityscapes polygon annotations into YOLO detection labels."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.classes import CITYSCAPES_CLASS_TO_ID, CITYSCAPES_DETECTION_CLASSES


@dataclass(slots=True)
class ConversionStats:
    """Tracks how much data was converted during a run."""

    image_count: int = 0
    label_file_count: int = 0
    object_count: int = 0
    skipped_object_count: int = 0


def polygon_to_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    """Converts a polygon point list into an enclosing bounding box."""

    x_coordinates = [point[0] for point in polygon]
    y_coordinates = [point[1] for point in polygon]
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)


def bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Converts absolute xyxy box coordinates into YOLO normalized format."""

    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height


def load_annotation(annotation_path: Path) -> dict:
    """Reads one Cityscapes polygon annotation file."""

    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_image_path(image_root: Path, split: str, city: str, annotation_name: str) -> Path:
    """Maps one gtFine annotation file to its clear Cityscapes image path."""

    image_name = annotation_name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
    return image_root / split / city / image_name


def build_label_name(annotation_name: str) -> str:
    """Maps one gtFine annotation file to the YOLO label filename expected by Ultralytics."""

    return annotation_name.replace("_gtFine_polygons.json", "_leftImg8bit.txt")


def convert_annotation_to_yolo_lines(annotation: dict) -> tuple[list[str], int, int]:
    """
    Converts supported objects in a Cityscapes annotation into YOLO label lines.

    Returns label lines plus counts for kept and skipped objects.
    """

    image_width = annotation["imgWidth"]
    image_height = annotation["imgHeight"]
    label_lines: list[str] = []
    kept_objects = 0
    skipped_objects = 0

    for obj in annotation["objects"]:
        label_name = obj["label"]
        if label_name not in CITYSCAPES_CLASS_TO_ID:
            skipped_objects += 1
            continue

        bbox = polygon_to_bbox(obj["polygon"])
        x_center, y_center, width, height = bbox_to_yolo(bbox, image_width, image_height)
        class_id = CITYSCAPES_CLASS_TO_ID[label_name]
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
        kept_objects += 1

    return label_lines, kept_objects, skipped_objects


def ensure_split_dirs(output_root: Path, split: str) -> tuple[Path, Path]:
    """Creates output image and label directories for one split."""

    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def convert_split(
    annotation_root: Path,
    image_root: Path,
    output_root: Path,
    split: str,
) -> ConversionStats:
    """Converts one Cityscapes split into YOLO images and label files."""

    split_annotation_root = annotation_root / split
    image_output_root, label_output_root = ensure_split_dirs(output_root, split)
    stats = ConversionStats()

    for annotation_path in sorted(split_annotation_root.rglob("*_gtFine_polygons.json")):
        city = annotation_path.parent.name
        image_path = build_image_path(image_root, split, city, annotation_path.name)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing matching image for annotation: {annotation_path}")

        annotation = load_annotation(annotation_path)
        label_lines, kept_objects, skipped_objects = convert_annotation_to_yolo_lines(annotation)
        output_image_dir = image_output_root / city
        output_label_dir = label_output_root / city
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        output_image_path = output_image_dir / image_path.name
        output_label_path = output_label_dir / build_label_name(annotation_path.name)

        shutil.copy2(image_path, output_image_path)
        output_label_path.write_text("\n".join(label_lines), encoding="utf-8")

        stats.image_count += 1
        stats.label_file_count += 1
        stats.object_count += kept_objects
        stats.skipped_object_count += skipped_objects

    return stats


def write_dataset_yaml(output_root: Path) -> None:
    """Writes the YOLO dataset config file for clear Cityscapes training."""

    yaml_lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for class_id, class_name in enumerate(CITYSCAPES_DETECTION_CLASSES):
        yaml_lines.append(f"  {class_id}: {class_name}")
    (output_root / "cityscapes_detection.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Runs conversion for train, val, and test splits."""

    project_root = PROJECT_ROOT
    annotation_root = project_root / "datasets" / "raw" / "gtFine_trainvaltest" / "gtFine"
    image_root = project_root / "datasets" / "raw" / "leftImg8bit_trainvaltest" / "leftImg8bit"
    output_root = project_root / "preprocessing" / "source_yolo"

    print("Selected classes:")
    for class_id, class_name in enumerate(CITYSCAPES_DETECTION_CLASSES):
        print(f"  {class_id}: {class_name}")

    aggregate = ConversionStats()
    for split in ("train", "val", "test"):
        stats = convert_split(annotation_root, image_root, output_root, split)
        aggregate.image_count += stats.image_count
        aggregate.label_file_count += stats.label_file_count
        aggregate.object_count += stats.object_count
        aggregate.skipped_object_count += stats.skipped_object_count
        print(
            f"[{split}] images={stats.image_count}, "
            f"labels={stats.label_file_count}, "
            f"kept_objects={stats.object_count}, "
            f"skipped_objects={stats.skipped_object_count}"
        )

    write_dataset_yaml(output_root)
    print(
        f"[done] images={aggregate.image_count}, "
        f"label_files={aggregate.label_file_count}, "
        f"kept_objects={aggregate.object_count}, "
        f"skipped_objects={aggregate.skipped_object_count}"
    )
    print(f"YOLO dataset config written to: {output_root / 'cityscapes_detection.yaml'}")


if __name__ == "__main__":
    main()
