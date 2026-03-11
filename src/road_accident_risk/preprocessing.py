from __future__ import annotations

import pandas as pd


class DataPreprocessor:
    def __init__(self) -> None:
        self.expected_features = [
            "num_lanes",
            "curvature",
            "speed_limit",
            "road_signs_present",
            "public_road",
            "holiday",
            "school_season",
            "num_reported_accidents",
            "road_type_highway",
            "road_type_rural",
            "road_type_urban",
            "lighting_daylight",
            "lighting_dim",
            "lighting_night",
            "weather_clear",
            "weather_foggy",
            "weather_rainy",
            "time_of_day_afternoon",
            "time_of_day_evening",
            "time_of_day_morning",
        ]

        self.categorical_columns = {
            "road_type": ["highway", "rural", "urban"],
            "lighting": ["daylight", "dim", "night"],
            "weather": ["clear", "foggy", "rainy"],
            "time_of_day": ["afternoon", "evening", "morning"],
        }

        self.boolean_columns = [
            "road_signs_present",
            "public_road",
            "holiday",
            "school_season",
        ]

    def preprocess(self, data: pd.DataFrame | dict, has_target: bool = True) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        if "id" in df.columns:
            df = df.drop("id", axis=1)

        target = None
        if has_target and "accident_risk" in df.columns:
            target = df["accident_risk"].copy()
            df = df.drop("accident_risk", axis=1)

        for col in self.boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)

        df = pd.get_dummies(df, columns=list(self.categorical_columns.keys()), dtype=int)

        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0

        df = df[self.expected_features]

        if target is not None:
            df["accident_risk"] = target

        return df

    def preprocess_for_prediction(self, input_data: dict) -> pd.DataFrame:
        return self.preprocess(input_data, has_target=False)

    def validate_input(self, input_data: dict | pd.DataFrame) -> tuple[bool, str]:
        if isinstance(input_data, dict):
            data_dict = input_data
        elif isinstance(input_data, pd.DataFrame):
            if len(input_data) == 0:
                return False, "DataFrame is empty"
            data_dict = input_data.iloc[0].to_dict()
        else:
            return False, "Input must be dict or DataFrame"

        required_fields = [
            "road_type",
            "num_lanes",
            "curvature",
            "speed_limit",
            "lighting",
            "weather",
            "road_signs_present",
            "public_road",
            "time_of_day",
            "holiday",
            "school_season",
            "num_reported_accidents",
        ]

        missing_fields = [field for field in required_fields if field not in data_dict]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        for col, valid_values in self.categorical_columns.items():
            if col in data_dict and data_dict[col] not in valid_values:
                return (
                    False,
                    f"Invalid value for {col}: '{data_dict[col]}'. Must be one of {valid_values}",
                )

        if not (1 <= data_dict.get("num_lanes", 0) <= 10):
            return False, "num_lanes must be between 1 and 10"

        if not (0 <= data_dict.get("curvature", -1) <= 1):
            return False, "curvature must be between 0 and 1"

        if not (0 <= data_dict.get("speed_limit", -1) <= 120):
            return False, "speed_limit must be between 0 and 120"

        if not (0 <= data_dict.get("num_reported_accidents", -1)):
            return False, "num_reported_accidents must be non-negative"

        return True, "Valid"


def preprocess_data(data: pd.DataFrame | dict, has_target: bool = True) -> pd.DataFrame:
    return DataPreprocessor().preprocess(data, has_target=has_target)

