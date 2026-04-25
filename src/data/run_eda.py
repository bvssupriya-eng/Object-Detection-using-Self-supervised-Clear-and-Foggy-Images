"""Run exploratory data analysis for Cityscapes and Foggy Cityscapes."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")

from src.config.classes import CITYSCAPES_DETECTION_CLASSES

EDA_OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"
EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def size_image(path:Path):
    imht, imwt = cv2.imgsize(str(path)).shape[:2]
    print(f"{path.name}: {imwt}x{imht}")

def load_json(path: Path) -> dict:
    """Load one JSON annotation file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_polygon_annotations(annotation_root: Path, split: str) -> list[Path]:
    """Return all polygon annotation files for a split."""

    return sorted((annotation_root / split).rglob("*_gtFine_polygons.json"))


def count_selected_classes(annotation_paths: list[Path]) -> Counter:
    """Count only the selected detection classes across annotations."""

    counts: Counter = Counter()
    for annotation_path in annotation_paths:
        annotation = load_json(annotation_path)
        for obj in annotation["objects"]:
            label_name = obj["label"]
            if label_name in CITYSCAPES_DETECTION_CLASSES:
                counts[label_name] += 1
    return counts


def polygon_to_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert a polygon to an enclosing bounding box."""

    x_coordinates = [point[0] for point in polygon]
    y_coordinates = [point[1] for point in polygon]
    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)


def collect_object_statistics(annotation_paths: list[Path]) -> tuple[list[dict[str, float | str]], list[dict[str, int | str]]]:
    """Collect bounding box and per-image object statistics for selected classes."""

    bbox_rows: list[dict[str, float | str]] = []
    image_rows: list[dict[str, int | str]] = []

    for annotation_path in annotation_paths:
        annotation = load_json(annotation_path)
        image_width = annotation["imgWidth"]
        image_height = annotation["imgHeight"]
        selected_count = 0

        for obj in annotation["objects"]:
            label_name = obj["label"]
            if label_name not in CITYSCAPES_DETECTION_CLASSES:
                continue

            x_min, y_min, x_max, y_max = polygon_to_bbox(obj["polygon"])
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            area_ratio = area / float(image_width * image_height)
            bbox_rows.append(
                {
                    "annotation_file": annotation_path.name,
                    "class_name": label_name,
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": area,
                    "bbox_area_ratio": area_ratio,
                }
            )
            selected_count += 1

        image_rows.append(
            {
                "annotation_file": annotation_path.name,
                "selected_object_count": selected_count,
            }
        )

    return bbox_rows, image_rows


def save_class_frequency_outputs(train_counts: Counter, val_counts: Counter) -> None:
    """Save class frequency table and bar chart."""

    rows: list[dict[str, int | str]] = []
    for class_name in CITYSCAPES_DETECTION_CLASSES:
        train_count = train_counts.get(class_name, 0)
        val_count = val_counts.get(class_name, 0)
        rows.append(
            {
                "class_name": class_name,
                "train_count": train_count,
                "val_count": val_count,
                "total_count": train_count + val_count,
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(EDA_OUTPUT_DIR / "class_frequency.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(dataframe["class_name"], dataframe["train_count"], label="train", color="#2563eb")
    plt.bar(dataframe["class_name"], dataframe["val_count"], bottom=dataframe["train_count"], label="val", color="#f59e0b")
    plt.xticks(rotation=20)
    plt.ylabel("Object Count")
    plt.title("Cityscapes Selected Class Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "class_frequency.png", dpi=200)
    plt.close()


def save_split_summary() -> None:
    """Save split-level image count summary for source and foggy domains."""

    source_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest" / "leftImg8bit"
    foggy_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"

    rows: list[dict[str, int | str]] = []
    for split in ("train", "val", "test"):
        source_count = len(list((source_root / split).rglob("*.png")))
        foggy_count = len(list((foggy_root / split).rglob("*.png")))
        rows.append(
            {
                "split": split,
                "source_clear_images": source_count,
                "target_foggy_images": foggy_count,
            }
        )

    pd.DataFrame(rows).to_csv(EDA_OUTPUT_DIR / "split_summary.csv", index=False)


def save_object_density_outputs(train_image_stats: list[dict[str, int | str]], val_image_stats: list[dict[str, int | str]]) -> None:
    """Save per-image object count statistics and plots."""

    train_df = pd.DataFrame(train_image_stats)
    train_df["split"] = "train"
    val_df = pd.DataFrame(val_image_stats)
    val_df["split"] = "val"
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(EDA_OUTPUT_DIR / "objects_per_image.csv", index=False)

    summary_rows = []
    for split_name, split_df in combined.groupby("split"):
        summary_rows.append(
            {
                "split": split_name,
                "mean_objects_per_image": round(split_df["selected_object_count"].mean(), 3),
                "min_objects_per_image": int(split_df["selected_object_count"].min()),
                "max_objects_per_image": int(split_df["selected_object_count"].max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(EDA_OUTPUT_DIR / "objects_per_image_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(train_df["selected_object_count"], bins=30, alpha=0.7, label="train", color="#2563eb")
    plt.hist(val_df["selected_object_count"], bins=30, alpha=0.7, label="val", color="#f59e0b")
    plt.xlabel("Objects per Image")
    plt.ylabel("Image Count")
    plt.title("Selected Objects per Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "objects_per_image_histogram.png", dpi=200)
    plt.close()


def save_bbox_analysis_outputs(train_bbox_rows: list[dict[str, float | str]], val_bbox_rows: list[dict[str, float | str]]) -> None:
    """Save bounding box statistics and plots."""

    train_df = pd.DataFrame(train_bbox_rows)
    train_df["split"] = "train"
    val_df = pd.DataFrame(val_bbox_rows)
    val_df["split"] = "val"
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_csv(EDA_OUTPUT_DIR / "bbox_statistics.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(combined["bbox_area_ratio"], bins=40, color="#0f766e")
    plt.xlabel("Bounding Box Area Ratio")
    plt.ylabel("Object Count")
    plt.title("Bounding Box Size Distribution")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "bbox_area_ratio_histogram.png", dpi=200)
    plt.close()

    class_summary = (
        combined.groupby("class_name")[["bbox_width", "bbox_height", "bbox_area_ratio"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "bbox_width": "mean_bbox_width",
                "bbox_height": "mean_bbox_height",
                "bbox_area_ratio": "mean_bbox_area_ratio",
            }
        )
    )
    class_summary.to_csv(EDA_OUTPUT_DIR / "bbox_class_summary.csv", index=False)


def save_fog_level_summary() -> None:
    """Summarize the count of foggy images for each beta level."""

    foggy_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"

    beta_counter: Counter = Counter()
    for image_path in foggy_root.rglob("*.png"):
        parts = image_path.stem.split("_beta_")
        if len(parts) == 2:
            beta_counter[parts[1]] += 1

    fog_df = pd.DataFrame(
        [{"fog_beta": fog_beta, "image_count": count} for fog_beta, count in sorted(beta_counter.items())]
    )
    fog_df.to_csv(EDA_OUTPUT_DIR / "fog_level_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.bar(fog_df["fog_beta"], fog_df["image_count"], color="#7c3aed")
    plt.xlabel("Fog Beta Level")
    plt.ylabel("Image Count")
    plt.title("Foggy Cityscapes Fog Level Distribution")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "fog_level_distribution.png", dpi=200)
    plt.close()


def compute_mean_brightness(image_path: Path) -> float:
    """Compute the average grayscale intensity of one image."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return float(image.mean())


def save_brightness_comparison(sample_count: int = 120, seed: int = 29) -> None:
    """Compare source and target image brightness distributions."""

    clear_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
    foggy_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy" / "val"

    clear_images = sample_images(sorted(clear_root.rglob("*.png")), sample_count=sample_count, seed=seed)
    foggy_images = sample_images(sorted(foggy_root.rglob("*.png")), sample_count=sample_count, seed=seed + 1)

    clear_brightness = [compute_mean_brightness(path) for path in clear_images]
    foggy_brightness = [compute_mean_brightness(path) for path in foggy_images]

    pd.DataFrame(
        {
            "domain": ["clear"] * len(clear_brightness) + ["foggy"] * len(foggy_brightness),
            "mean_brightness": clear_brightness + foggy_brightness,
        }
    ).to_csv(EDA_OUTPUT_DIR / "brightness_comparison.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {"domain": "clear", "mean_brightness": round(mean(clear_brightness), 3)},
            {"domain": "foggy", "mean_brightness": round(mean(foggy_brightness), 3)},
        ]
    )
    summary_df.to_csv(EDA_OUTPUT_DIR / "brightness_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(clear_brightness, bins=20, alpha=0.7, label="clear", color="#22c55e")
    plt.hist(foggy_brightness, bins=20, alpha=0.7, label="foggy", color="#64748b")
    plt.xlabel("Mean Grayscale Brightness")
    plt.ylabel("Image Count")
    plt.title("Source vs Foggy Brightness Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "brightness_distribution.png", dpi=200)
    plt.close()


def sample_images(image_paths: list[Path], sample_count: int, seed: int) -> list[Path]:
    """Choose a stable sample of images."""

    rng = random.Random(seed)
    return rng.sample(image_paths, min(sample_count, len(image_paths)))


def save_image_grid(image_paths: list[Path], output_path: Path, title: str) -> None:
    """Save a simple image grid for qualitative EDA."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        raise ValueError("No image paths provided for grid generation.")

    columns = 2
    rows = math.ceil(len(image_paths) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, image_path in zip(axes_list, image_paths, strict=False):
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axis.imshow(image_rgb)
        axis.set_title(image_path.name, fontsize=9)
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def find_matching_foggy_image(clear_image_path: Path) -> Path:
    """Map a clear Cityscapes image to one foggy counterpart."""

    city = clear_image_path.parent.name
    split = clear_image_path.parent.parent.name
    foggy_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"
    stem = clear_image_path.stem.replace("_leftImg8bit", "")
    foggy_candidates = sorted((foggy_root / split / city).glob(f"{stem}_leftImg8bit_foggy_beta_*.png"))
    if not foggy_candidates:
        raise FileNotFoundError(f"No foggy match found for {clear_image_path.name}")
    return foggy_candidates[0]


def save_clear_vs_foggy_comparison(sample_count: int = 4, seed: int = 21) -> None:
    """Save side-by-side clear and foggy comparison images."""

    clear_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest" / "leftImg8bit" / "val"
    clear_images = sorted(clear_root.rglob("*.png"))
    sampled_clear_images = sample_images(clear_images, sample_count=sample_count, seed=seed)

    for clear_image_path in sampled_clear_images:
        foggy_image_path = find_matching_foggy_image(clear_image_path)
        clear_image = cv2.imread(str(clear_image_path))
        foggy_image = cv2.imread(str(foggy_image_path))

        clear_resized = cv2.resize(clear_image, (960, 480))
        foggy_resized = cv2.resize(foggy_image, (960, 480))
        comparison = cv2.hconcat([clear_resized, foggy_resized])

        cv2.putText(
            comparison,
            "Clear Cityscapes",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            comparison,
            "Foggy Cityscapes",
            (990, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = EDA_OUTPUT_DIR / f"comparison_{clear_image_path.stem}.png"
        cv2.imwrite(str(output_path), comparison)


def main() -> None:
    """Run the complete EDA pipeline for selected classes and image samples."""

    annotation_root = PROJECT_ROOT / "datasets" / "raw" / "gtFine_trainvaltest" / "gtFine"
    clear_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest" / "leftImg8bit"

    train_annotations = find_polygon_annotations(annotation_root, split="train")
    val_annotations = find_polygon_annotations(annotation_root, split="val")
    train_counts = count_selected_classes(train_annotations)
    val_counts = count_selected_classes(val_annotations)
    train_bbox_rows, train_image_stats = collect_object_statistics(train_annotations)
    val_bbox_rows, val_image_stats = collect_object_statistics(val_annotations)

    save_class_frequency_outputs(train_counts, val_counts)
    save_split_summary()
    save_object_density_outputs(train_image_stats, val_image_stats)
    save_bbox_analysis_outputs(train_bbox_rows, val_bbox_rows)
    save_fog_level_summary()
    save_brightness_comparison()

    train_clear_samples = sample_images(sorted((clear_root / "train").rglob("*.png")), sample_count=6, seed=13)
    val_clear_samples = sample_images(sorted((clear_root / "val").rglob("*.png")), sample_count=6, seed=19)
    foggy_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"
    val_foggy_samples = sample_images(sorted((foggy_root / "val").rglob("*.png")), sample_count=6, seed=23)

    save_image_grid(
        train_clear_samples,
        EDA_OUTPUT_DIR / "train_clear_samples.png",
        title="Cityscapes Train Sample Images",
    )
    save_image_grid(
        val_clear_samples,
        EDA_OUTPUT_DIR / "val_clear_samples.png",
        title="Cityscapes Validation Sample Images",
    )
    save_image_grid(
        val_foggy_samples,
        EDA_OUTPUT_DIR / "val_foggy_samples.png",
        title="Foggy Cityscapes Validation Sample Images",
    )
    save_clear_vs_foggy_comparison()

    print(f"[done] EDA outputs saved to {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
