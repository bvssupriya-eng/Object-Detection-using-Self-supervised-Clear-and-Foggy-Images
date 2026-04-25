# Environment Setup

## Local Environment

1. create virtual environment in `.venv`
2. install packages from `requirements.txt`
3. use workspace-local config directories:
   - `YOLO_CONFIG_DIR=.workspace/ultralytics`
   - `JUPYTER_DATA_DIR=.workspace/jupyter`
4. register local Jupyter kernel

## Kaggle Environment

Use `requirements-kaggle.txt` as the reference package list for notebooks.

## Why two files

- `requirements.txt` is for local development, notebooks, preprocessing, evaluation, and GUI
- `requirements-kaggle.txt` is a lighter training-focused dependency list

## Recommended Commands

```powershell
$env:YOLO_CONFIG_DIR="C:\MachineVision\.workspace\ultralytics"
$env:JUPYTER_DATA_DIR="C:\MachineVision\.workspace\jupyter"
.\.venv\Scripts\python -m ipykernel install --prefix .venv --name machinevision --display-name "Python (.venv) MachineVision"
```
