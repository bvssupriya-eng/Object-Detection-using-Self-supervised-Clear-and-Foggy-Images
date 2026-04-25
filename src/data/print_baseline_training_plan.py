"""Print Kaggle-ready baseline training commands."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.training_plan import BASELINE_MODELS, DATASET_YAML_PATH


def main() -> None:
    """Print the command plan for all baseline runs."""

    print("Dataset YAML:")
    print(f"  {DATASET_YAML_PATH}")
    print()
    print("Baseline training runs:")
    for config in BASELINE_MODELS:
        print(f"- {config.model_name}")
        print(
            "  yolo detect train "
            f"model={config.weights_name} "
            f"data=\"{DATASET_YAML_PATH}\" "
            f"epochs={config.epochs} imgsz={config.image_size} batch={config.batch_size} "
            f"patience={config.patience} project=runs/source_baselines name={config.run_name}"
        )


if __name__ == "__main__":
    main()
