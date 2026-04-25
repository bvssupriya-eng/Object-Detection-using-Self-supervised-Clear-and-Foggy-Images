# Results Index

Use this folder as the single place for project outputs.

## Main files

- `final_report.html`: one-file summary report with tables, plots, and sample predictions

## Folders

- `source_baselines/`: best baseline weights and per-model training CSVs
- `adaptation/`: adapted-model weights and adaptation training CSVs
- `comparison/`: final comparison tables and comparison plots
- `plots/`: per-model training result graphs
- `prediction_samples/`: selected foggy prediction images for report/demo use
- `eda/`: exploratory data analysis tables, plots, and sample images

## Best files to open first

- `final_report.html`
- `comparison/baseline_comparison.csv`
- `comparison/yolov5s_baseline_vs_adapted.csv`
- `comparison/baseline_comparison_combined.png`

## Current experimental summary

- Best baseline model: `YOLOv5s`
- Compared baseline models: `YOLOv8n`, `YOLOv11n`, `YOLOv5s`
- Pilot adaptation completed for: `YOLOv5s` on `frankfurt`
- Pilot adaptation did not improve over baseline in this setup
