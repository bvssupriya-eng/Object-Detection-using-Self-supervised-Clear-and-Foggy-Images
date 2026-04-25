# MachineVision Project

Current focus is only on the early project stages:

1. dataset audit
2. EDA
3. annotation conversion
4. label validation

## Use These Folders

- `datasets/raw`
  original Cityscapes and Foggy Cityscapes data

- `datasets/processed/source_yolo`
  converted YOLO-format source dataset

- `src/config`
  class definitions

- `src/data`
  data scripts

- `notebooks`
  audit and EDA notes

- `results/qualitative/label_validation`
  saved label-check images

## Ignore For Now

- model training
- Kaggle workflow
- GUI
- final report packaging

Those will be added back later when we actually reach those phases.

## Current Scripts

- `src/data/cityscapes_to_yolo.py`
- `src/data/visualize_yolo_labels.py`

## Current Next Step

Complete proper EDA on the Cityscapes and Foggy Cityscapes datasets.
