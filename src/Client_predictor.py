# -*- coding: utf-8 -*-
"""Legacy entrypoint. Prefer running `python scripts/predict.py`."""

from road_accident_risk.predictor import AccidentRiskPredictor  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(
        "Use `python scripts/predict.py` for batch predictions "
        "(or import AccidentRiskPredictor from road_accident_risk)."
    )
