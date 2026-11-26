"""Shared fixtures: the baseline parameter file."""

from pathlib import Path

import pytest

from casoria.params import Config, load_config

BASELINE_YAML = Path(__file__).resolve().parents[1] / "config" / "baseline.yaml"


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config(BASELINE_YAML)
