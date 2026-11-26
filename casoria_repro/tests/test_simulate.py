"""Monte Carlo runner: shapes, reproducibility, agreement with the deterministic model."""

import warnings

import numpy as np
import pytest

from casoria.metrics import roi
from casoria.model import evaluate_central
from casoria.params import apply_overrides
from casoria.simulate import draw_soc_inputs, generator, largest_confident_theta, run_monte_carlo


@pytest.fixture(scope="module")
def small_config(config):
    """Fewer draws and CV levels so the tests run fast."""
    return apply_overrides(
        config,
        {
            "monte_carlo.draws": 2000,
            "monte_carlo.cv_levels": [0.01, 0.25],
            "scenarios": {"conservative": {}},
        },
    )


@pytest.fixture(scope="module")
def result(small_config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_monte_carlo(small_config)


def test_draw_shapes(small_config):
    inputs = draw_soc_inputs(small_config, 0.25, generator(small_config, 0, 0, 0))
    assert inputs.alerts_per_window.shape == (2000,)
    assert inputs.effort_minutes.shape == (2000, 3)
    assert inputs.tokens_in.shape == (2000, 3)


def test_summary_covers_every_configuration(small_config, result):
    n_thetas = small_config.capex.theta_grid.num
    expected_rows = len(small_config.archetypes) * len(small_config.monte_carlo.cv_levels) * n_thetas
    assert len(result.summary) == expected_rows


def test_runs_are_reproducible(small_config, result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = run_monte_carlo(small_config)
    assert again.summary.equals(result.summary)


def test_tiny_cv_matches_the_deterministic_model(small_config, result):
    """At CV = 1% the mean ROI is close to the deterministic ROI."""
    theta = small_config.capex.baseline_theta
    for name in small_config.archetypes:
        outcome = evaluate_central(small_config, name)
        expected = roi(outcome.savings, outcome.baseline_cost, theta)[0]
        row = result.summary[
            (result.summary["archetype"] == name)
            & (result.summary["cv"] == 0.01)
            & np.isclose(result.summary["theta"], theta)
        ].iloc[0]
        assert row["mean_roi"] == pytest.approx(expected, abs=0.05)


def test_p_profit_falls_as_theta_rises(result):
    for _, group in result.summary.dropna().groupby(["archetype", "cv"]):
        assert np.all(np.diff(group.sort_values("theta")["p_profit"]) <= 0)


def test_largest_confident_theta(result):
    table = largest_confident_theta(result.summary, 0.95)
    api_high = table[(table["archetype"] == "API-High") & (table["cv"] == 0.01)].iloc[0]
    sh_low = table[(table["archetype"] == "SH-Low") & (table["cv"] == 0.01)].iloc[0]
    assert api_high["max_confident_theta"] > 0.1     # deterministic share is 14.7%
    assert np.isnan(sh_low["max_confident_theta"])


def test_infeasible_beta_is_skipped_with_a_warning(config):
    small = apply_overrides(
        config,
        {"monte_carlo.draws": 100, "monte_carlo.cv_levels": [0.35], "scenarios": {"conservative": {}}},
    )
    with pytest.warns(UserWarning, match="API-High"):
        out = run_monte_carlo(small)
    api_high = out.summary[out.summary["archetype"] == "API-High"]
    assert api_high["p_profit"].isna().all()
    assert api_high["note"].str.contains("Beta").all()
    assert out.summary[out.summary["archetype"] == "API-Mid"]["p_profit"].notna().all()
