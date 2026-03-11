# -*- coding: utf-8 -*-
"""Legacy module. Prefer `road_accident_risk.preprocessing`."""

import pandas as pd


from road_accident_risk.preprocessing import DataPreprocessor, preprocess_data  # noqa: F401
    
__all__ = ["DataPreprocessor", "preprocess_data"]