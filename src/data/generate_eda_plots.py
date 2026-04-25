"""Generate EDA summaries and SVG plots for Cityscapes-style datasets.

This script is intentionally dependency-light so it can run in a plain Python
environment without matplotlib, PIL, or OpenCV.

Example:
    python src/data/generate_eda_plots.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.classes import CITYSCAPES_DETECTION_CLASSES


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def histogram(values: list[float], bin_count: int) -> tuple[list[str], list[int]]:
    if not values:
        return [], []
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return [f"{minimum:.4f}"], [len(values)]

    bin_width = (maximum - minimum) / bin_count
    counts = [0] * bin_count
    for value in values:
        index = min(int((value - minimum) / bin_width), bin_count - 1)
        counts[index] += 1

    labels = []
    for index in range(bin_count):
        start = minimum + index * bin_width
        end = start + bin_width
        labels.append(f"{start:.3f}-{end:.3f}")
    return labels, counts


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white" />',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="24" font-family="Arial" fill="#111827">{escape(title)}</text>',
    ]


def write_bar_chart_svg(
    output_path: Path,
    title: str,
    categories: list[str],
    series: list[dict],
    y_axis_label: str,
) -> None:
    width = 1200
    height = 700
    margin_left = 95
    margin_right = 40
    margin_top = 70
    margin_bottom = 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_value = 0.0
    for item in series:
        if item["type"] == "stacked":
            max_value = max(max_value, max(item["values"][i] + item["bottom"][i] for i in range(len(categories))))
        else:
            max_value = max(max_value, max(item["values"]))
    max_value = max(max_value, 1.0)

    lines = svg_header(width, height, title)
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>')

    tick_count = 5
    for tick in range(tick_count + 1):
        y_value = max_value * tick / tick_count
        y = margin_top + plot_height - (plot_height * tick / tick_count)
        lines.append(f'<line x1="{margin_left - 6}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{y + 5}" text-anchor="end" font-size="12" font-family="Arial" fill="#4b5563">{y_value:.0f}</text>')

    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2}" transform="rotate(-90 24 {margin_top + plot_height / 2})" '
        f'text-anchor="middle" font-size="16" font-family="Arial" fill="#111827">{escape(y_axis_label)}</text>'
    )

    bar_group_count = sum(1 for item in series if item["type"] == "grouped")
    group_slot_width = plot_width / max(len(categories), 1)
    bar_width = min(48, group_slot_width / max(bar_group_count + 1, 1))

    for category_index, category in enumerate(categories):
        category_center = margin_left + group_slot_width * (category_index + 0.5)
        grouped_offset = -(bar_group_count - 1) * bar_width / 2
        grouped_seen = 0

        for item in series:
            value = item["values"][category_index]
            if item["type"] == "stacked":
                bottom = item["bottom"][category_index]
                y0 = margin_top + plot_height - (bottom / max_value) * plot_height
                y1 = margin_top + plot_height - ((bottom + value) / max_value) * plot_height
                x = category_center - bar_width / 2
                height_value = max(y0 - y1, 1)
                lines.append(
                    f'<rect x="{x}" y="{y1}" width="{bar_width}" height="{height_value}" fill="{item["color"]}" />'
                )
            else:
                x = category_center + grouped_offset + grouped_seen * bar_width
                y = margin_top + plot_height - (value / max_value) * plot_height
                height_value = max((value / max_value) * plot_height, 1)
                lines.append(
                    f'<rect x="{x}" y="{y}" width="{bar_width}" height="{height_value}" fill="{item["color"]}" />'
                )
                grouped_seen += 1

        lines.append(
            f'<text x="{category_center}" y="{margin_top + plot_height + 28}" text-anchor="end" '
            f'transform="rotate(-20 {category_center} {margin_top + plot_height + 28})" '
            f'font-size="13" font-family="Arial" fill="#111827">{escape(category)}</text>'
        )

    legend_x = width - 220
    legend_y = 90
    for index, item in enumerate(series):
        box_y = legend_y + index * 26
        lines.append(f'<rect x="{legend_x}" y="{box_y - 12}" width="16" height="16" fill="{item["color"]}" />')
        lines.append(
            f'<text x="{legend_x + 24}" y="{box_y}" font-size="13" font-family="Arial" fill="#111827">{escape(item["label"])}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_histogram_svg(
    output_path: Path,
    title: str,
    labels: list[str],
    counts: list[int],
    y_axis_label: str,
) -> None:
    width = 1200
    height = 700
    margin_left = 95
    margin_right = 40
    margin_top = 70
    margin_bottom = 170
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(counts) if counts else 1

    lines = svg_header(width, height, title)
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151" stroke-width="2"/>')

    tick_count = 5
    for tick in range(tick_count + 1):
        y_value = max_count * tick / tick_count
        y = margin_top + plot_height - (plot_height * tick / tick_count)
        lines.append(f'<line x1="{margin_left - 6}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{y + 5}" text-anchor="end" font-size="12" font-family="Arial" fill="#4b5563">{y_value:.0f}</text>')

    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2}" transform="rotate(-90 24 {margin_top + plot_height / 2})" '
        f'text-anchor="middle" font-size="16" font-family="Arial" fill="#111827">{escape(y_axis_label)}</text>'
    )

    slot_width = plot_width / max(len(labels), 1)
    bar_width = max(slot_width - 6, 1)
    for index, label in enumerate(labels):
        x = margin_left + index * slot_width + 3
        bar_height = (counts[index] / max_count) * plot_height if max_count else 0
        y = margin_top + plot_height - bar_height
        lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{max(bar_height, 1)}" fill="#0f766e" />')
        text_x = x + bar_width / 2
        lines.append(
            f'<text x="{text_x}" y="{margin_top + plot_height + 34}" text-anchor="end" '
            f'transform="rotate(-35 {text_x} {margin_top + plot_height + 34})" '
            f'font-size="11" font-family="Arial" fill="#111827">{escape(label)}</text>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


class EDAPlotGenerator:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.annotation_root = PROJECT_ROOT / "datasets" / "raw" / "gtFine_trainvaltest" / "gtFine"
        self.clear_root = PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest" / "leftImg8bit"
        self.foggy_root = (
            PROJECT_ROOT / "datasets" / "raw" / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"
        )

    def run(self) -> None:
        self._validate_paths()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_annotations = self._find_polygon_annotations("train")
        val_annotations = self._find_polygon_annotations("val")

        train_counts = self._count_selected_classes(train_annotations)
        val_counts = self._count_selected_classes(val_annotations)
        train_bbox_rows, train_image_rows = self._collect_object_statistics(train_annotations)
        val_bbox_rows, val_image_rows = self._collect_object_statistics(val_annotations)

        self._save_class_frequency_outputs(train_counts, val_counts)
        self._save_split_summary()
        self._save_object_density_outputs(train_image_rows, val_image_rows)
        self._save_bbox_analysis_outputs(train_bbox_rows, val_bbox_rows)
        self._save_fog_level_summary()

        print(f"[done] EDA outputs saved to {self.output_dir}")

    def _validate_paths(self) -> None:
        required_paths = [self.annotation_root, self.clear_root, self.foggy_root]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required dataset paths:\n" + "\n".join(missing))

    def _find_polygon_annotations(self, split: str) -> list[Path]:
        return sorted((self.annotation_root / split).rglob("*_gtFine_polygons.json"))

    def _count_selected_classes(self, annotation_paths: list[Path]) -> Counter:
        counts: Counter = Counter()
        for annotation_path in annotation_paths:
            annotation = load_json(annotation_path)
            for obj in annotation.get("objects", []):
                label = obj.get("label")
                if label in CITYSCAPES_DETECTION_CLASSES:
                    counts[label] += 1
        return counts

    def _polygon_to_bbox(self, polygon: list[list[float]]) -> tuple[float, float, float, float]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def _collect_object_statistics(
        self,
        annotation_paths: list[Path],
    ) -> tuple[list[dict[str, float | str]], list[dict[str, int | str]]]:
        bbox_rows: list[dict[str, float | str]] = []
        image_rows: list[dict[str, int | str]] = []

        for annotation_path in annotation_paths:
            annotation = load_json(annotation_path)
            img_width = annotation["imgWidth"]
            img_height = annotation["imgHeight"]
            selected_count = 0

            for obj in annotation.get("objects", []):
                label = obj.get("label")
                if label not in CITYSCAPES_DETECTION_CLASSES:
                    continue

                x_min, y_min, x_max, y_max = self._polygon_to_bbox(obj["polygon"])
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                area_ratio = area / float(img_width * img_height)
                bbox_rows.append(
                    {
                        "annotation_file": annotation_path.name,
                        "class_name": label,
                        "bbox_width": round(width, 3),
                        "bbox_height": round(height, 3),
                        "bbox_area": round(area, 3),
                        "bbox_area_ratio": round(area_ratio, 6),
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

    def _save_class_frequency_outputs(self, train_counts: Counter, val_counts: Counter) -> None:
        rows = []
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

        write_csv(
            self.output_dir / "class_frequency.csv",
            ["class_name", "train_count", "val_count", "total_count"],
            rows,
        )

        categories = [row["class_name"] for row in rows]
        train_values = [row["train_count"] for row in rows]
        val_values = [row["val_count"] for row in rows]
        write_bar_chart_svg(
            self.output_dir / "class_frequency.svg",
            "Cityscapes Selected Class Frequency",
            categories,
            [
                {
                    "label": "train",
                    "values": train_values,
                    "bottom": [0] * len(train_values),
                    "color": "#2563eb",
                    "type": "stacked",
                },
                {
                    "label": "val",
                    "values": val_values,
                    "bottom": train_values,
                    "color": "#f59e0b",
                    "type": "stacked",
                },
            ],
            "Object Count",
        )

    def _save_split_summary(self) -> None:
        rows = []
        for split in ("train", "val", "test"):
            clear_count = len(list((self.clear_root / split).rglob("*.png")))
            foggy_count = len(list((self.foggy_root / split).rglob("*.png")))
            rows.append(
                {
                    "split": split,
                    "source_clear_images": clear_count,
                    "target_foggy_images": foggy_count,
                }
            )

        write_csv(
            self.output_dir / "split_summary.csv",
            ["split", "source_clear_images", "target_foggy_images"],
            rows,
        )

        write_bar_chart_svg(
            self.output_dir / "split_summary.svg",
            "Source vs Target Split Counts",
            [row["split"] for row in rows],
            [
                {
                    "label": "clear",
                    "values": [row["source_clear_images"] for row in rows],
                    "color": "#22c55e",
                    "type": "grouped",
                },
                {
                    "label": "foggy",
                    "values": [row["target_foggy_images"] for row in rows],
                    "color": "#64748b",
                    "type": "grouped",
                },
            ],
            "Image Count",
        )

    def _save_object_density_outputs(
        self,
        train_image_rows: list[dict[str, int | str]],
        val_image_rows: list[dict[str, int | str]],
    ) -> None:
        rows = []
        for row in train_image_rows:
            rows.append(
                {
                    "annotation_file": row["annotation_file"],
                    "selected_object_count": row["selected_object_count"],
                    "split": "train",
                }
            )
        for row in val_image_rows:
            rows.append(
                {
                    "annotation_file": row["annotation_file"],
                    "selected_object_count": row["selected_object_count"],
                    "split": "val",
                }
            )

        write_csv(
            self.output_dir / "objects_per_image.csv",
            ["annotation_file", "selected_object_count", "split"],
            rows,
        )

        summary_rows = []
        for split in ("train", "val"):
            split_values = [int(row["selected_object_count"]) for row in rows if row["split"] == split]
            summary_rows.append(
                {
                    "split": split,
                    "mean_objects_per_image": round(mean(split_values), 3),
                    "min_objects_per_image": min(split_values),
                    "max_objects_per_image": max(split_values),
                }
            )

        write_csv(
            self.output_dir / "objects_per_image_summary.csv",
            [
                "split",
                "mean_objects_per_image",
                "min_objects_per_image",
                "max_objects_per_image",
            ],
            summary_rows,
        )

        histogram_labels, histogram_counts = histogram(
            [float(row["selected_object_count"]) for row in rows],
            bin_count=25,
        )
        write_histogram_svg(
            self.output_dir / "objects_per_image_histogram.svg",
            "Selected Objects per Image",
            histogram_labels,
            histogram_counts,
            "Image Count",
        )

    def _save_bbox_analysis_outputs(
        self,
        train_bbox_rows: list[dict[str, float | str]],
        val_bbox_rows: list[dict[str, float | str]],
    ) -> None:
        rows = []
        for row in train_bbox_rows:
            row_with_split = dict(row)
            row_with_split["split"] = "train"
            rows.append(row_with_split)
        for row in val_bbox_rows:
            row_with_split = dict(row)
            row_with_split["split"] = "val"
            rows.append(row_with_split)

        write_csv(
            self.output_dir / "bbox_statistics.csv",
            [
                "annotation_file",
                "class_name",
                "bbox_width",
                "bbox_height",
                "bbox_area",
                "bbox_area_ratio",
                "split",
            ],
            rows,
        )

        area_ratios = [float(row["bbox_area_ratio"]) for row in rows]
        histogram_labels, histogram_counts = histogram(area_ratios, bin_count=30)
        write_histogram_svg(
            self.output_dir / "bbox_area_ratio_histogram.svg",
            "Bounding Box Size Distribution",
            histogram_labels,
            histogram_counts,
            "Object Count",
        )

        class_summary_rows = []
        for class_name in CITYSCAPES_DETECTION_CLASSES:
            class_rows = [row for row in rows if row["class_name"] == class_name]
            if not class_rows:
                continue
            class_summary_rows.append(
                {
                    "class_name": class_name,
                    "mean_bbox_width": round(mean(float(row["bbox_width"]) for row in class_rows), 3),
                    "mean_bbox_height": round(mean(float(row["bbox_height"]) for row in class_rows), 3),
                    "mean_bbox_area_ratio": round(
                        mean(float(row["bbox_area_ratio"]) for row in class_rows),
                        6,
                    ),
                }
            )

        write_csv(
            self.output_dir / "bbox_class_summary.csv",
            ["class_name", "mean_bbox_width", "mean_bbox_height", "mean_bbox_area_ratio"],
            class_summary_rows,
        )

        write_bar_chart_svg(
            self.output_dir / "bbox_class_summary.svg",
            "Average Bounding Box Area Ratio by Class",
            [row["class_name"] for row in class_summary_rows],
            [
                {
                    "label": "mean area ratio",
                    "values": [row["mean_bbox_area_ratio"] for row in class_summary_rows],
                    "color": "#0f766e",
                    "type": "grouped",
                }
            ],
            "Mean Area Ratio",
        )

    def _save_fog_level_summary(self) -> None:
        beta_counter: Counter = Counter()
        for image_path in self.foggy_root.rglob("*.png"):
            parts = image_path.stem.split("_beta_")
            if len(parts) == 2:
                beta_counter[parts[1]] += 1

        rows = [
            {"fog_beta": fog_beta, "image_count": image_count}
            for fog_beta, image_count in sorted(beta_counter.items())
        ]
        write_csv(self.output_dir / "fog_level_summary.csv", ["fog_beta", "image_count"], rows)

        write_bar_chart_svg(
            self.output_dir / "fog_level_distribution.svg",
            "Foggy Cityscapes Fog Level Distribution",
            [row["fog_beta"] for row in rows],
            [
                {
                    "label": "foggy image count",
                    "values": [row["image_count"] for row in rows],
                    "color": "#7c3aed",
                    "type": "grouped",
                }
            ],
            "Image Count",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dependency-light EDA plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "eda",
        help="Directory where output CSV and SVG files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EDAPlotGenerator(args.output_dir).run()


if __name__ == "__main__":
    main()
