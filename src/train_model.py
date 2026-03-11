# -*- coding: utf-8 -*-
"""Legacy entrypoint. Prefer running `python -m scripts.train`."""

from road_accident_risk.training import train_and_save


if __name__ == "__main__":
    train_and_save()