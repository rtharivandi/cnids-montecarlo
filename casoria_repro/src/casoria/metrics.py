"""Decision metrics of Casoria et al. (2026), Eq. 13-20, plus CVaR.

Capex convention
----------------
theta = C_cap / C^noLLM (Eq. 14) is the decision variable. In a Monte Carlo
run we apply theta to each draw's own baseline cost, i.e.
C_cap = theta * C^noLLM per draw.
ASSUMPTION: the paper does not say which convention it uses. This one matches
its finding that the alert volume N only matters for self-hosted models:
for API models both savings and C_cap scale with N, so ROI does not depend on N.
"""

from __future__ import annotations

import numpy as np

HOURS_PER_YEAR = 365 * 24


def capex_per_window(theta: float | np.ndarray, baseline_cost: np.ndarray) -> np.ndarray:
    """Eq. 14 solved for C_cap: the amortized capex per window, theta * C^noLLM."""
    return theta * baseline_cost


def roi(
    savings: np.ndarray,              # Delta C^m, per window
    baseline_cost: np.ndarray,        # C^noLLM, per window
    theta: float | np.ndarray,        # C_cap / C^noLLM
) -> np.ndarray:
    """Eq. 15: ROI^m = (Delta C^m - C_cap) / C_cap."""
    capex = capex_per_window(theta, baseline_cost)
    return (savings - capex) / capex


def max_theta(
    savings: np.ndarray,
    baseline_cost: np.ndarray,
    target_roi: float = 0.0,          # y
) -> np.ndarray:
    """Eq. 16 (Proposition 1): ROI^m > y  <=>  theta < Delta C^m / ((1 + y) C^noLLM).

    Returns the right-hand side, the largest capex ratio that still beats y.
    """
    return savings / ((1 + target_roi) * baseline_cost)


def max_missed_incident_rate(
    savings: np.ndarray,
    baseline_cost: np.ndarray,
    theta: float,
    cost_of_missed_incident: float,   # C_miss
) -> np.ndarray:
    """Eq. 17: ROI^m > 0  <=>  lambda < (Delta C^m - theta C^noLLM) / C_miss.

    lambda is the rate (per window) of undetected wrong answers that turn
    into an incident costing C_miss.
    """
    return (savings - capex_per_window(theta, baseline_cost)) / cost_of_missed_incident


def total_capex(
    theta: float,
    baseline_cost: float,
    window_hours: float,
    amortization_years: float,
) -> float:
    """The one-off capex implied by theta, before amortization (for reporting)."""
    windows_per_year = HOURS_PER_YEAR / window_hours
    return theta * baseline_cost * windows_per_year * amortization_years


# ---------------------------------------------------------------------------
# Over Monte Carlo draws (Eq. 19-20). `roi_draws` has the draws on axis 0.
# ---------------------------------------------------------------------------
def probability_of_profit(roi_draws: np.ndarray) -> np.ndarray:
    """Eq. 19: P(ROI^m > 0 | CV), the share of draws with positive ROI."""
    return np.mean(roi_draws > 0, axis=0)


def value_at_risk(roi_draws: np.ndarray, level: float) -> np.ndarray:
    """Eq. 20: VaR_z(ROI) = -(z-quantile of ROI).

    A VaR of 0.5 at z = 10% means: in the worst 10% of draws the investment
    loses at least half the committed capex.
    """
    return -np.quantile(roi_draws, level, axis=0)


def conditional_value_at_risk(roi_draws: np.ndarray, level: float) -> np.ndarray:
    """CVaR_z(ROI): minus the mean ROI over the worst z share of draws.

    Not in the paper; added because our seminar paper is about tail risk.
    """
    worst_count = int(np.ceil(level * roi_draws.shape[0]))
    worst = np.sort(roi_draws, axis=0)[:worst_count]
    return -worst.mean(axis=0)
