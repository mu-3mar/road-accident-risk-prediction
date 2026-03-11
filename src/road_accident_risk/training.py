from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .paths import artifacts_dir, raw_data_dir
from .preprocessing import DataPreprocessor


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    scaler_path: Path
    preprocessor_path: Path
    feature_names_path: Path
    test_rmse: float
    test_r2: float


def train_and_save(
    train_csv: Path | None = None,
    out_dir: Path | None = None,
    test_size: float = 0.05,
    random_state: int = 15,
) -> TrainResult:
    if train_csv is None:
        train_csv = raw_data_dir() / "train.csv"
    if out_dir is None:
        out_dir = artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = DataPreprocessor()
    df = pd.read_csv(train_csv)
    df = preprocessor.preprocess(df, has_target=True)

    X = df.drop("accident_risk", axis=1)
    y = df["accident_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    cols_to_scale = ["speed_limit"]
    scaler = StandardScaler()
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    best_params = {
        "min_samples_leaf": 50,
        "max_leaf_nodes": 63,
        "max_iter": 300,
        "learning_rate": 0.05,
        "random_state": 0,
        "early_stopping": True,
        "validation_fraction": 0.1,
    }

    model = HistGradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    # (Optional) extra metrics for logs
    _ = mean_absolute_error(y_test, test_pred)
    _ = mean_squared_error(y_test, test_pred)

    model_path = out_dir / "final_model.pkl"
    scaler_path = out_dir / "scaler.pkl"
    preprocessor_path = out_dir / "preprocessor.pkl"
    feature_names_path = out_dir / "feature_names.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    with open(feature_names_path, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    return TrainResult(
        model_path=model_path,
        scaler_path=scaler_path,
        preprocessor_path=preprocessor_path,
        feature_names_path=feature_names_path,
        test_rmse=float(test_rmse),
        test_r2=float(test_r2),
    )

