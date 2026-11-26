"""Decision metrics: Proposition 1, the missed-incident bound, VaR and CVaR."""

import numpy as np
import pytest

from casoria.metrics import (
    conditional_value_at_risk,
    max_missed_incident_rate,
    max_theta,
    probability_of_profit,
    roi,
    total_capex,
    value_at_risk,
)

N_CASES = 100_000


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_proposition_1_holds_on_random_inputs(rng):
    """Eq. 16: ROI > y  <=>  theta < Delta C / ((1 + y) C_noLLM)."""
    savings = rng.normal(0, 2000, N_CASES)
    baseline_cost = rng.uniform(100, 10_000, N_CASES)
    theta = rng.uniform(0.001, 1.0, N_CASES)
    target = rng.uniform(-0.9, 3.0, N_CASES)

    left = roi(savings, baseline_cost, theta) > target
    right = theta < max_theta(savings, baseline_cost, target)

    # Skip cases that sit on the boundary up to rounding error.
    clear = np.abs(theta - max_theta(savings, baseline_cost, target)) > 1e-9
    assert np.array_equal(left[clear], right[clear])


def test_missed_incident_bound_on_random_inputs(rng):
    """Eq. 17: subtracting lambda * C_miss from the savings keeps ROI > 0 iff lambda < lambda_max."""
    savings = rng.normal(0, 2000, N_CASES)
    baseline_cost = rng.uniform(100, 10_000, N_CASES)
    theta = 0.1
    c_miss = 1e6
    rate = rng.uniform(0, 0.005, N_CASES)

    profitable = roi(savings - rate * c_miss, baseline_cost, theta) > 0
    bound = max_missed_incident_rate(savings, baseline_cost, theta, c_miss)
    clear = np.abs(rate - bound) > 1e-12
    assert np.array_equal(profitable[clear], (rate < bound)[clear])


def test_roi_example():
    # Savings 708, baseline 4818, theta 0.1 -> capex 481.8 -> ROI 0.4695
    assert roi(np.array(708.0), np.array(4818.0), 0.1) == pytest.approx(708 / 481.8 - 1)


def test_probability_of_profit():
    assert probability_of_profit(np.array([-1.0, 0.0, 0.5, 2.0])) == pytest.approx(0.5)


def test_var_and_cvar_on_known_draws():
    draws = np.arange(-50, 50, dtype=float)   # 100 draws: -50 ... 49
    # 10% quantile with linear interpolation: -50 + 0.1 * 99 = -40.1
    assert value_at_risk(draws, 0.10) == pytest.approx(40.1)
    # The worst 10 draws are -50 ... -41, mean -45.5
    assert conditional_value_at_risk(draws, 0.10) == pytest.approx(45.5)


def test_cvar_is_at_least_var(rng):
    draws = rng.lognormal(0, 1, (10_000, 5)) - 2
    assert np.all(conditional_value_at_risk(draws, 0.1) >= value_at_risk(draws, 0.1))


def test_metrics_work_along_the_draw_axis(rng):
    draws = rng.normal(size=(1000, 7))
    assert probability_of_profit(draws).shape == (7,)
    assert value_at_risk(draws, 0.1).shape == (7,)
    assert conditional_value_at_risk(draws, 0.1).shape == (7,)


def test_total_capex():
    # theta 0.1 of $4,818/day, 365 days, 3 years
    assert total_capex(0.1, 4818.0, 24, 3) == pytest.approx(0.1 * 4818 * 365 * 3)
