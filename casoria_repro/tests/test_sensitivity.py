"""One-at-a-time tornado analysis."""

import pytest

from casoria.params import apply_overrides
from casoria.sensitivity import scaled, tornado


@pytest.fixture(scope="module")
def table(config):
    return tornado(config)


def test_one_row_per_parameter_and_archetype(config, table):
    assert len(table) == len(config.archetypes) * len(config.sensitivity.parameters)


def test_ranks_follow_swing(table):
    for _, part in table.groupby("archetype"):
        assert list(part["rank"]) == list(range(1, len(part) + 1))
        assert part["swing"].is_monotonic_decreasing


def test_gpu_cost_does_not_move_api_models(table):
    gpu = table[(table["parameter"] == "gpu.cost_per_hour") & table["archetype"].str.startswith("API")]
    assert (gpu["swing"] == 0).all()


def test_alert_volume_only_matters_for_self_hosted(table):
    """Paper: N only matters for self-hosted models (savings and capex both scale with N)."""
    n = table[table["parameter"] == "alerts_per_window"].set_index("archetype")["swing"]
    assert n[["API-High", "API-Mid", "API-Low"]].max() == pytest.approx(0, abs=1e-9)
    assert (n[["SH-High", "SH-Mid", "SH-Low"]] > 0).all()


def test_speed_ratio_dominates_for_api_high(table):
    api_high = table[table["archetype"] == "API-High"]
    assert api_high.iloc[0]["parameter"] == "speed_ratio"


def test_bounds_clip_accuracy(table):
    row = table[(table["archetype"] == "API-High") & (table["parameter"].str.endswith("accuracy"))].iloc[0]
    assert row["high_value"] == 1.0


def test_new_parameter_needs_no_code_change(config):
    entries = [{"path": "cost_of_missed_incident"}, {"path": "bytes_per_param"}]
    extended = apply_overrides(config, {"sensitivity.parameters": entries})
    result = tornado(extended)
    assert set(result["parameter"]) == {"cost_of_missed_incident", "bytes_per_param"}


def test_scaled_handles_tables_and_bounds():
    assert scaled({"low": 10, "high": 60}, 0.5, None) == {"low": 5, "high": 30}
    assert scaled(0.92, 1.25, (0.0, 1.0)) == 1.0
