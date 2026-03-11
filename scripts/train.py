from __future__ import annotations

from road_accident_risk.training import train_and_save


def main() -> None:
    result = train_and_save()
    print("Training finished.")
    print(f"- model:        {result.model_path}")
    print(f"- scaler:       {result.scaler_path}")
    print(f"- preprocessor: {result.preprocessor_path}")
    print(f"- features:     {result.feature_names_path}")
    print(f"- test_rmse:    {result.test_rmse:.4f}")
    print(f"- test_r2:      {result.test_r2:.4f}")


if __name__ == "__main__":
    main()

