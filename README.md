# Road-Accident-Risk

Accident risk prediction (regression) using a preprocessing pipeline + `HistGradientBoostingRegressor`.

## Project structure

```
Road-Accident-Risk/
  artifacts/                 # trained model + scaler + preprocessor (pkl)
  data/
    raw/                     # train.csv, test.csv, sample_submission.csv, ...
    processed/               # generated/derived datasets (ignored)
  notebooks/                 # EDA / experiments
  outputs/                   # predictions (ignored)
  scripts/
    train.py                 # train + save artifacts
    predict.py               # batch predict CSV -> outputs/results.csv
  src/
    road_accident_risk/      # importable python package
      preprocessing.py
      predictor.py
      training.py
      paths.py
  requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

This reads `data/raw/train.csv` and writes artifacts to `artifacts/`.

```bash
PYTHONPATH=src python scripts/train.py
```

## Predict (batch)

This reads `data/raw/test.csv` and writes `outputs/results.csv`.

```bash
PYTHONPATH=src python scripts/predict.py
```

To choose custom paths:

```bash
PYTHONPATH=src python scripts/predict.py --input path/to/test.csv --output outputs/my_preds.csv
```

## Notes

- **Artifacts**: `artifacts/final_model.pkl`, `artifacts/scaler.pkl`, `artifacts/preprocessor.pkl`, `artifacts/feature_names.pkl`
- **Legacy files**: `src/train_model.py`, `src/Client_predictor.py`, `src/preprocessing_module.py` remain as thin legacy entrypoints; use `scripts/` + `src/road_accident_risk/` going forward.