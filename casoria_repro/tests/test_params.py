"""The parameter file is validated and overrides are re-validated."""

import pytest
from pydantic import ValidationError

from casoria.model import SOC_INPUT_NAMES
from casoria.params import UNCERTAIN_INPUTS, Config, apply_overrides


def test_baseline_loads(config):
    assert list(config.archetypes) == ["API-High", "API-Mid", "API-Low", "SH-High", "SH-Mid", "SH-Low"]
    assert config.role_split.as_array().shape == (3, 3)


def test_shares_must_sum_to_one(config):
    with pytest.raises(ValidationError, match="severity_share"):
        apply_overrides(config, {"severity_share.low": 0.5})


def test_accuracy_must_be_a_probability(config):
    with pytest.raises(ValidationError):
        apply_overrides(config, {"archetypes.API-High.accuracy": 1.2})


def test_unknown_keys_are_rejected(config):
    data = config.model_dump()
    data["alerts_per_day"] = 1000
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_override_paths_must_exist(config):
    with pytest.raises(KeyError, match="speed_ration"):
        apply_overrides(config, {"speed_ration": 0.5})


def test_scenarios_apply(config):
    assert config.scenario("no_rework_penalty").rework_penalty == 0.0
    assert config.scenario("full_automation").speed_ratio == 0.0
    assert config.scenario("conservative") == config


def test_uncertain_inputs_match_the_model(config):
    """Every shared uncertain input is a field of model.SocInputs (plus per-archetype accuracy)."""
    assert set(UNCERTAIN_INPUTS) == set(SOC_INPUT_NAMES) | {"accuracy"}
