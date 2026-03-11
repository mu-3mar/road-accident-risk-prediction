from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .paths import artifacts_dir


class AccidentRiskPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
    ) -> None:
        base = artifacts_dir()
        model_path = Path(model_path) if model_path is not None else (base / "final_model.pkl")
        scaler_path = Path(scaler_path) if scaler_path is not None else (base / "scaler.pkl")
        preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path is not None else (base / "preprocessor.pkl")
        )

        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.cols_to_scale = ["speed_limit"]

    def predict(self, input_data: dict | pd.DataFrame):
        is_single = isinstance(input_data, dict)

        X = (
            self.preprocessor.preprocess_for_prediction(input_data)
            if is_single
            else self.preprocessor.preprocess(input_data, has_target=False)
        )

        X.loc[:, self.cols_to_scale] = self.scaler.transform(X[self.cols_to_scale])
        preds = self.model.predict(X)
        return float(preds[0]) if is_single else preds

    def predict_with_validation(self, input_data: dict | pd.DataFrame):
        is_valid, message = self.preprocessor.validate_input(input_data)
        if not is_valid:
            return None, False, message
        return self.predict(input_data), True, "Success"

