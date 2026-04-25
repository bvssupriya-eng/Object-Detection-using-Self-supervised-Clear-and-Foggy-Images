# Object Detection Using Self-Supervised Learning on Clear and Foggy Images

This project studies object detection under domain shift from clear urban scenes to foggy urban scenes. The workflow uses Cityscapes as the source domain and Foggy Cityscapes as the target domain, with a focus on preprocessing, exploratory data analysis, YOLO-format conversion, baseline training, pilot adaptation, and result presentation through a Streamlit interface.

## Project Goals

- audit the Cityscapes and Foggy Cityscapes datasets
- convert selected Cityscapes detection labels into YOLO format
- validate preprocessing quality with visual checks and summary reports
- compare compact YOLO baselines on clear-weather street scenes
- study how baseline models behave on foggy images
- support pilot self-training and qualitative comparison for adaptation

## Datasets

The project is organized around two datasets:

- `Cityscapes`
  source-domain urban street scenes in clear weather
- `Foggy Cityscapes`
  target-domain urban street scenes with synthetic fog at multiple beta levels

Raw datasets are expected under:

```text
datasets/raw/
```

These large files are intentionally excluded from Git.

## Repository Structure

```text
MachineVision/
├── notebooks/                  # markdown notes from audit, EDA, and experiment tracking
├── preprocessing/              # preprocessing summaries and lightweight docs
├── results/                    # generated outputs such as EDA plots, comparisons, and reports
├── src/
│   ├── config/                 # class definitions and training plan helpers
│   ├── data/                   # dataset audit, EDA, conversion, and validation scripts
│   ├── reporting/              # report and presentation builders
│   └── training/               # model training scripts
├── streamlit_app.py            # GUI for prediction and comparison
└── requirements.txt            # project dependencies
```

## Main Components

### 1. Data Preparation

The preprocessing pipeline converts selected Cityscapes polygon annotations into YOLO detection labels and performs validation checks to reduce image-label mismatches.

Relevant scripts:

- `src/data/cityscapes_to_yolo.py`
- `src/data/preprocessing_checks.py`
- `src/data/visualize_yolo_labels.py`

### 2. Exploratory Data Analysis

EDA is used to inspect class distribution, object density, fog severity distribution, and bounding-box statistics before training.

Relevant scripts:

- `src/data/run_eda.py`
- `src/data/generate_eda_plots.py`

Typical outputs go to:

```text
results/eda/
```

### 3. Training

The project includes support for compact YOLO baseline experiments and a training plan for the next stages of the pipeline.

Relevant scripts:

- `src/training/train_baseline_ultralytics.py`
- `src/data/print_baseline_training_plan.py`
- `src/config/training_plan.py`

### 4. Reporting and Presentation

Utilities are included for building project artifacts beyond raw model outputs.

Relevant scripts:

- `src/reporting/build_word_report.py`
- `src/reporting/build_presentation.py`

### 5. Streamlit GUI

The repository also includes a local interface for visual prediction comparison.

Features:

- upload a single image
- run a baseline model and an adapted model
- compare original, baseline, and adapted outputs side by side
- review a lightweight detection summary

Launch with:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

## Setup

### Local Environment

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Recommended workspace-local config:

```powershell
$env:YOLO_CONFIG_DIR="C:\MachineVision\.workspace\ultralytics"
$env:JUPYTER_DATA_DIR="C:\MachineVision\.workspace\jupyter"
.\.venv\Scripts\python -m ipykernel install --prefix .venv --name machinevision --display-name "Python (.venv) MachineVision"
```

## Suggested Workflow

1. Place raw Cityscapes and Foggy Cityscapes data in `datasets/raw/`.
2. Run preprocessing and annotation conversion scripts.
3. Generate EDA summaries and plots.
4. Validate label quality visually.
5. Train baseline YOLO models.
6. Compare clear-domain and foggy-domain behavior.
7. Use the GUI and reporting utilities for presentation.

## Current Progress

Completed:

- environment setup
- raw dataset placement
- dataset audit
- EDA
- annotation conversion to YOLO format
- visual label validation
- preprocessing summary
- baseline training plan

Planned next:

- first baseline training run with YOLOv8n
- remaining baseline experiments
- broader baseline comparison and adaptation workflow

## Notes

- Large datasets, model weights, generated results, and intermediate preprocessing artifacts are excluded through `.gitignore`.
- This repository is meant to keep source code, lightweight notes, and reproducible workflow logic under version control.

## Authoring Focus

This project is structured as a practical machine vision workflow rather than a single notebook experiment. The emphasis is on reproducibility, clean preprocessing, interpretable analysis, and clear reporting for clear-to-foggy domain adaptation experiments.
