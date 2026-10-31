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
  requirements.txt           # complete training and development environment
  requirements-docker.txt    # minimal API/inference runtime
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the Docker image, install only the runtime dependencies:

```bash
pip install -r requirements-docker.txt
```

`requirements-docker.txt` contains the dependencies needed by the prediction
API and batch inference. The full `requirements.txt` adds notebook, EDA, and
model-experimentation packages such as Jupyter, plotting libraries, SciPy, and
XGBoost.

## Dockerization

The prediction API is dockerized using the minimal runtime dependencies and
the trained artifacts. The Docker Hub image is available at:

[mu3mar/road-accident-risk-prediction](https://hub.docker.com/r/mu3mar/road-accident-risk-prediction)

Build the image locally:

```bash
docker build -t mu3mar/road-accident-risk-prediction:latest .
```

Run the API container:

```bash
docker run -d \
  --name road-accident-api \
  -p 8000:8000 \
  mu3mar/road-accident-risk-prediction:latest
```

Check that the API is ready:

```bash
curl http://127.0.0.1:8000/
```

To publish a locally built image to Docker Hub:

```bash
docker login
docker push mu3mar/road-accident-risk-prediction:latest
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

## Web interface

The `frontend/` directory is independent from the Python backend. It can be opened
directly as `frontend/index.html` or served with any static file server. If the
backend is not available, the interface uses a clearly labelled local heuristic so
the form and result experience still work. When the API is running, the trained
regression model is used automatically.

Start the model API:

```bash
PYTHONPATH=src python scripts/server.py
```

In another terminal, optionally serve the frontend:

```bash
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500`. The API defaults to `http://127.0.0.1:8000`;
override it with `?api=http://host:port` or the `roadguard-api` local-storage
value. The API accepts the same fields as the training data:
`road_type`, `num_lanes`, `curvature`, `speed_limit`, `lighting`, `weather`,
`road_signs_present`, `public_road`, `time_of_day`, `holiday`, `school_season`,
and `num_reported_accidents`.

## Notes

- **Artifacts**: `artifacts/final_model.pkl`, `artifacts/scaler.pkl`, `artifacts/preprocessor.pkl`, `artifacts/feature_names.pkl`
- **Legacy files**: `src/train_model.py`, `src/Client_predictor.py`, `src/preprocessing_module.py` remain as thin legacy entrypoints; use `scripts/` + `src/road_accident_risk/` going forward.