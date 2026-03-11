from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # scripts/ run from repo root in README; fall back to current file parents
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_data_dir() -> Path:
    return data_dir() / "raw"


def processed_data_dir() -> Path:
    return data_dir() / "processed"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def outputs_dir() -> Path:
    return repo_root() / "outputs"

