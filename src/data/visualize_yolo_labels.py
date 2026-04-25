"""Visualize YOLO labels on sample images to validate annotation conversion."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.classes import CITYSCAPES_DETECTION_CLASSES


CLASS_COLORS = {
    "person": (255, 99, 71),
    "rider": (255, 165, 0),
    "car": (0, 191, 255),
    "truck": (60, 179, 113),
    "bus": (255, 215, 0),
    "motorcycle": (186, 85, 211),
    "bicycle": (72, 209, 204),
}


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Converts normalized YOLO coordinates to pixel xyxy box coordinates."""

    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    width_px = width * image_width
    height_px = height * image_height

    x_min = int(round(x_center_px - width_px / 2.0))
    y_min = int(round(y_center_px - height_px / 2.0))
    x_max = int(round(x_center_px + width_px / 2.0))
    y_max = int(round(y_center_px + height_px / 2.0))
    return x_min, y_min, x_max, y_max


def draw_labels(image_path: Path, label_path: Path, output_path: Path) -> int:
    """Draws all YOLO labels for one image and saves the preview."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    image_height, image_width = image.shape[:2]
    label_count = 0

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            class_id_text, x_center_text, y_center_text, width_text, height_text = stripped.split()
            class_id = int(class_id_text)
            class_name = CITYSCAPES_DETECTION_CLASSES[class_id]
            color = CLASS_COLORS[class_name]
            x_min, y_min, x_max, y_max = yolo_to_xyxy(
                float(x_center_text),
                float(y_center_text),
                float(width_text),
                float(height_text),
                image_width,
                image_height,
            )

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                image,
                class_name,
                (x_min, max(20, y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            label_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return label_count


def collect_sample_images(split_dir: Path, sample_count: int, seed: int) -> list[Path]:
    """Returns a stable random sample of images from one split."""

    image_paths = sorted(split_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {split_dir}")

    rng = random.Random(seed)
    count = min(sample_count, len(image_paths))
    return rng.sample(image_paths, count)


def build_label_path(dataset_root: Path, image_path: Path, split: str) -> Path:
    """Maps one processed image path to its label file path."""

    relative_path = image_path.relative_to(dataset_root / "images" / split)
    label_name = relative_path.name.replace(".png", ".txt")
    return dataset_root / "labels" / split / relative_path.parent / label_name


def collect_labeled_sample_images(
    dataset_root: Path,
    split: str,
    sample_count: int,
    seed: int,
) -> list[Path]:
    """Prefers sample images that contain at least one YOLO label."""

    split_dir = dataset_root / "images" / split
    image_paths = sorted(split_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {split_dir}")

    labeled_images: list[Path] = []
    unlabeled_images: list[Path] = []
    for image_path in image_paths:
        label_path = build_label_path(dataset_root, image_path, split)
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            labeled_images.append(image_path)
        else:
            unlabeled_images.append(image_path)

    rng = random.Random(seed)
    rng.shuffle(labeled_images)
    rng.shuffle(unlabeled_images)

    selected = labeled_images[:sample_count]
    if len(selected) < sample_count:
        selected.extend(unlabeled_images[: sample_count - len(selected)])
    return selected


def visualize_split(
    dataset_root: Path,
    split: str,
    sample_count: int,
    seed: int,
) -> None:
    """Creates preview images for a given split."""

    output_dir = PROJECT_ROOT / "preprocessing" / "label_validation" / split
    sample_images = collect_labeled_sample_images(
        dataset_root,
        split=split,
        sample_count=sample_count,
        seed=seed,
    )

    print(f"[{split}] selected {len(sample_images)} images for validation")
    for image_path in sample_images:
        label_path = build_label_path(dataset_root, image_path, split)
        output_path = output_dir / image_path.name
        label_count = draw_labels(image_path, label_path, output_path)
        print(f"  saved {output_path.name} with {label_count} labels")


def main() -> None:
    """Generates qualitative previews for train and val splits."""

    dataset_root = PROJECT_ROOT / "preprocessing" / "source_yolo"
    visualize_split(dataset_root, split="train", sample_count=5, seed=11)
    visualize_split(dataset_root, split="val", sample_count=5, seed=17)
    print("[done] label previews saved under preprocessing/label_validation")


if __name__ == "__main__":
    main()
