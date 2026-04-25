# Streamlit GUI

Run the GUI locally from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

## What it does

- Upload one image
- Run one baseline model
- Run one adapted model
- Show original, baseline prediction, and adapted prediction on one page
- Show a small detection summary table for both outputs

## Model paths used

- Baseline weights: `results/source_baselines/*_best.pt`
- Adapted weights: `results/adaptation/*_best.pt`

If you add more `.pt` files to those folders, they will appear automatically in the app.
