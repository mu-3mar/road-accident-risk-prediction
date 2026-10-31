# RoadGuard: Road Accident Risk Prediction

An end-to-end machine-learning project that estimates a continuous
`accident_risk` score from road, traffic, weather, lighting, and time-related
conditions. The project includes exploratory analysis, preprocessing,
model selection, a saved inference pipeline, a batch prediction CLI, a
lightweight HTTP API, a browser interface, and Docker support.

> **Important:** This is a regression project, not a classification project.
> The model returns a numeric risk score rather than a class label or a
> calibrated probability. The score is intended for educational and planning
> purposes only; it is not a guarantee of road safety.

## Results at a glance

The final model was selected after comparing several regressors and tuning
`HistGradientBoostingRegressor`. The reported hold-out evaluation uses a
random 5% test split with `random_state=15`.

| Model | Test RMSE | Test R² |
|---|---:|---:|
| Linear Regression | 0.074 | 0.804 |
| Gradient Boosting | 0.057 | 0.883 |
| HistGradientBoosting | 0.056 | 0.886 |
| AdaBoost | 0.087 | 0.725 |
| XGBoost | 0.056 | 0.887 |

### Final saved model

The production artifact is a `HistGradientBoostingRegressor` trained with the
selected configuration below:

```text
min_samples_leaf = 50
max_leaf_nodes   = 63
max_iter         = 300
learning_rate    = 0.05
early_stopping   = True
validation_fraction = 0.1
random_state     = 0
```

Final hold-out metrics recorded during modeling:

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Train | 0.0558 | 0.0434 | 0.8874 |
| Test | **0.0561** | **0.0433** | **0.8868** |

The hyperparameter search used `RandomizedSearchCV` with 12 candidates and
5-fold cross-validation. The best cross-validation R² recorded for the search
was `0.8862`.

## Dataset

This project uses the
[Predicting Road Accident Risk dataset](https://www.kaggle.com/competitions/playground-series-s5e10/overview)
from the Kaggle Playground Series, Season 5, Episode 10.

The dataset is not included in this repository. Download it from Kaggle,
extract the files, and place them in `data/raw/`:

```text
data/raw/
  train.csv
  test.csv
  sample_submission.csv
```

The training data recorded in the EDA notebook contains **517,754 rows and
14 columns**. The target column is `accident_risk`; Kaggle evaluation uses
RMSE.

### Input features

The prediction pipeline consumes these 12 input indicators:

- `road_type`: `highway`, `rural`, or `urban`
- `num_lanes`: number of lanes
- `curvature`: normalized road curvature from 0 to 1
- `speed_limit`: speed limit in km/h
- `lighting`: `daylight`, `dim`, or `night`
- `weather`: `clear`, `foggy`, or `rainy`
- `road_signs_present`: whether road signs are present
- `public_road`: whether the segment is a public road
- `time_of_day`: `morning`, `afternoon`, or `evening`
- `holiday`: whether the record is during a holiday
- `school_season`: whether school is in session
- `num_reported_accidents`: reported accidents for the road segment

`id` is preserved for batch output but excluded from model features.

## Machine-learning workflow

1. **EDA:** reviewed dataset shape, data quality, numerical and categorical
   distributions, correlations, and relationships with the target.
2. **Preprocessing:**
   - converts boolean indicators to integers;
   - one-hot encodes `road_type`, `lighting`, `weather`, and `time_of_day`;
   - enforces a stable 20-feature schema for training and inference;
   - standardizes `speed_limit` with a fitted `StandardScaler`.
3. **Model comparison:** evaluated Linear Regression, Gradient Boosting,
   HistGradientBoosting, AdaBoost, and XGBoost using RMSE and R².
4. **Tuning:** used 5-fold randomized search over learning rate, number of
   iterations, leaf count, and minimum leaf size.
5. **Artifact export:** saves the trained model, scaler, preprocessor, and
   feature names so inference uses the same transformations as training.

## Repository structure

```text
.
├── artifacts/                     # serialized model and preprocessing artifacts
│   ├── final_model.pkl
│   ├── scaler.pkl
│   ├── preprocessor.pkl
│   └── feature_names.pkl
├── data/
│   └── data.md                    # dataset source and download instructions
├── frontend/                      # standalone RoadGuard browser interface
├── notebooks/
│   ├── 01-EDA.ipynb
│   ├── 02-Preprocessing.ipynb
│   └── 03-Modeling.ipynb
├── outputs/
│   └── results.csv                # generated batch predictions
├── scripts/
│   ├── train.py                   # train and save artifacts
│   ├── predict.py                 # batch prediction
│   └── server.py                  # HTTP prediction API
├── src/road_accident_risk/
│   ├── preprocessing.py           # feature engineering and validation
│   ├── predictor.py               # artifact loading and inference
│   ├── training.py                # training and evaluation
│   └── paths.py                   # project paths
├── Dockerfile
├── requirements.txt
└── requirements-docker.txt
```

## Installation

Python 3.11 is used by the Docker image. Create a virtual environment and
install the complete training/development requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runtime-only environment uses `requirements-docker.txt`, which contains
the dependencies needed for the API and batch inference.

## Train the model

After placing `train.csv` in `data/raw/`:

```bash
PYTHONPATH=src python scripts/train.py
```

The command writes these files to `artifacts/` and prints the hold-out
`test_rmse` and `test_r2`:

```text
final_model.pkl
scaler.pkl
preprocessor.pkl
feature_names.pkl
```

The training implementation defaults to `test_size=0.05` and
`random_state=15`.

## Batch prediction

After placing `test.csv` in `data/raw/`:

```bash
PYTHONPATH=src python scripts/predict.py
```

The default output is `outputs/results.csv` with this schema:

```text
id,accident_risk
```

Custom input and output paths are supported:

```bash
PYTHONPATH=src python scripts/predict.py \
  --input path/to/test.csv \
  --output outputs/my_predictions.csv
```

## HTTP API

Start the local API:

```bash
PYTHONPATH=src python scripts/server.py
```

The server listens on `http://127.0.0.1:8000`.

### Health check

```bash
curl http://127.0.0.1:8000/
```

Expected response:

```json
{"service": "road-accident-risk", "status": "ready"}
```

### Prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "road_type": "urban",
    "num_lanes": 2,
    "curvature": 0.2,
    "speed_limit": 60,
    "lighting": "daylight",
    "weather": "clear",
    "road_signs_present": 1,
    "public_road": 1,
    "time_of_day": "morning",
    "holiday": 0,
    "school_season": 1,
    "num_reported_accidents": 2
  }'
```

Successful responses contain the model-generated score:

```json
{"accident_risk": 0.2935, "source": "model"}
```

The API validates required fields, categorical values, and numeric ranges.
Invalid input returns HTTP `422`; malformed JSON returns HTTP `400`.

## Web interface

The `frontend/` directory contains the RoadGuard interface. Serve it with a
static file server while the API is running:

```bash
python3 -m http.server 5500 --directory frontend
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500). The interface:

- checks whether the model API is available;
- sends all 12 indicators to `/predict` when connected;
- displays the returned score and its source;
- provides reset and randomize controls;
- uses a clearly labelled local heuristic only when the API cannot be
  reached, so offline UI behavior is not confused with model evaluation.

You can point the interface to another API with:

```text
http://127.0.0.1:5500/?api=http://host:port
```

## Docker

The Docker image contains the minimal runtime dependencies, source code,
prediction scripts, and trained artifacts.

Build and run:

```bash
docker build -t mu3mar/road-accident-risk-prediction:latest .
docker run --rm -p 8000:8000 \
  mu3mar/road-accident-risk-prediction:latest
```

Verify the container:

```bash
curl http://127.0.0.1:8000/
```

The published image is available on
[Docker Hub](https://hub.docker.com/r/mu3mar/road-accident-risk-prediction).
To publish a new local build:

```bash
docker login
docker push mu3mar/road-accident-risk-prediction:latest
```

## Reproducibility and limitations

- The hold-out metrics are based on one shuffled 5% split; they should not be
  interpreted as a guarantee of performance on every road or region.
- The target is a continuous risk score. It should not be interpreted as a
  probability unless separately calibrated and validated for that purpose.
- The dataset must be downloaded from Kaggle and is intentionally not
  committed to the repository.
- Pickle artifacts should only be loaded from trusted sources.
- The frontend offline estimate is a UI fallback and is not included in the
  reported model metrics.

## Legacy entry points

`src/train_model.py`, `src/Client_predictor.py`, and
`src/preprocessing_module.py` remain for compatibility. New workflows should
use `scripts/` together with the importable package under
`src/road_accident_risk/`.
