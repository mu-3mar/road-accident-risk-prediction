from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from road_accident_risk.paths import outputs_dir, raw_data_dir
from road_accident_risk.predictor import AccidentRiskPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict accident risk (batch CSV).")
    p.add_argument(
        "--input",
        type=Path,
        default=raw_data_dir() / "test.csv",
        help="Path to input CSV (default: data/raw/test.csv)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=outputs_dir() / "results.csv",
        help="Where to write predictions (default: outputs/results.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = AccidentRiskPredictor()

    df = pd.read_csv(args.input)
    preds = predictor.predict(df)
    out = pd.DataFrame({"id": df["id"], "accident_risk": preds})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

