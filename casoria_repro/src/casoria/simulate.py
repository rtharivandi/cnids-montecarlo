"""Monte Carlo runner: P(profit), VaR and CVaR of the ROI (Eq. 19-20).

For every scenario and CV level we draw the shared inputs (alert volume,
effort, eta, beta, requests, tokens) once and reuse them for every archetype,
so archetypes are compared on the same random workload. Each archetype then
gets its own accuracy draws. Each of these streams has its own seed, built
from (seed, scenario, CV level, stream), so results are reproducible and do not
shift when one configuration is skipped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from casoria.distributions import InfeasibleBetaError, from_mean_cv
from casoria.metrics import conditional_value_at_risk, probability_of_profit, roi, value_at_risk
from casoria.model import SOC_INPUT_NAMES, SocInputs, central_value, evaluate
from casoria.params import Config

SHARED_STREAM = 0   # stream id of the shared inputs; archetype i uses stream i + 1


def generator(config: Config, scenario_index: int, cv_index: int, stream: int) -> np.random.Generator:
    """A reproducible random generator for one stream of one configuration."""
    return np.random.default_rng([config.monte_carlo.seed, scenario_index, cv_index, stream])


def draw_soc_inputs(config: Config, cv: float, rng: np.random.Generator) -> SocInputs:
    """Draw the shared inputs. Inputs not listed in monte_carlo.distributions keep their YAML value."""
    n_draws = config.monte_carlo.draws
    values = {}
    for name in SOC_INPUT_NAMES:
        mean = central_value(config, name)               # shape () or (a,)
        size = (n_draws, *mean.shape)
        family = config.monte_carlo.distributions.get(name)
        if family is None:
            values[name] = np.broadcast_to(mean, size).astype(float)
        else:
            values[name] = from_mean_cv(family, mean, cv).rvs(size=size, random_state=rng)
    return SocInputs(**values)


def draw_accuracy(config: Config, archetype_name: str, cv: float, rng: np.random.Generator) -> np.ndarray:
    """Draw E_m for one archetype, shape (n,)."""
    mean = config.archetypes[archetype_name].accuracy
    size = (config.monte_carlo.draws,)
    family = config.monte_carlo.distributions.get("accuracy")
    if family is None:
        return np.full(size, mean)
    return from_mean_cv(family, mean, cv).rvs(size=size, random_state=rng)


@dataclass(frozen=True)
class Draws:
    """Monte Carlo draws for one (scenario, archetype, CV) configuration."""

    scenario: str
    archetype: str
    cv: float
    savings: np.ndarray          # Delta C^m after risk adjustments, (n,)
    baseline_cost: np.ndarray    # C^noLLM, (n,)

    def roi(self, theta: float | np.ndarray) -> np.ndarray:
        """ROI per draw (Eq. 15); for an array of thetas, shape (n, len(theta))."""
        theta = np.asarray(theta)
        if theta.ndim == 0:
            return roi(self.savings, self.baseline_cost, theta)
        return roi(self.savings[:, None], self.baseline_cost[:, None], theta[None, :])


def summarize(draws: Draws, thetas: np.ndarray, tail_level: float) -> pd.DataFrame:
    """P(profit), mean ROI, VaR and CVaR of the ROI for every theta (Eq. 19-20)."""
    roi_draws = draws.roi(thetas)                        # (n, theta)
    return pd.DataFrame(
        {
            "scenario": draws.scenario,
            "archetype": draws.archetype,
            "cv": draws.cv,
            "theta": thetas,
            "p_profit": probability_of_profit(roi_draws),
            "mean_roi": roi_draws.mean(axis=0),
            "var_roi": value_at_risk(roi_draws, tail_level),
            "cvar_roi": conditional_value_at_risk(roi_draws, tail_level),
            "note": "",
        }
    )


def infeasible_rows(scenario: str, archetype: str, cv: float, thetas: np.ndarray, reason: str) -> pd.DataFrame:
    """Placeholder rows (NaN metrics) for a configuration that cannot be simulated."""
    return pd.DataFrame(
        {
            "scenario": scenario,
            "archetype": archetype,
            "cv": cv,
            "theta": thetas,
            "p_profit": np.nan,
            "mean_roi": np.nan,
            "var_roi": np.nan,
            "cvar_roi": np.nan,
            "note": reason,
        }
    )


@dataclass(frozen=True)
class MonteCarloResult:
    summary: pd.DataFrame        # one row per scenario x archetype x CV x theta
    draws: list[Draws]           # raw draws, for plots


def run_monte_carlo(config: Config) -> MonteCarloResult:
    """Run every scenario x CV level x archetype in the config."""
    thetas = config.capex.theta_grid.values()
    tail_level = config.monte_carlo.tail_level
    tables: list[pd.DataFrame] = []
    all_draws: list[Draws] = []

    for scenario_index, scenario in enumerate(config.scenarios):
        scenario_config = config.scenario(scenario)
        for cv_index, cv in enumerate(config.monte_carlo.cv_levels):
            try:
                shared = draw_soc_inputs(
                    scenario_config, cv, generator(config, scenario_index, cv_index, SHARED_STREAM)
                )
            except InfeasibleBetaError as error:
                _warn_skip(scenario, "all archetypes", cv, error)
                for archetype in config.archetypes:
                    tables.append(infeasible_rows(scenario, archetype, cv, thetas, str(error)))
                continue

            for archetype_index, archetype in enumerate(config.archetypes):
                rng = generator(config, scenario_index, cv_index, archetype_index + 1)
                try:
                    accuracy = draw_accuracy(scenario_config, archetype, cv, rng)
                except InfeasibleBetaError as error:
                    _warn_skip(scenario, archetype, cv, error)
                    tables.append(infeasible_rows(scenario, archetype, cv, thetas, str(error)))
                    continue
                outcome = evaluate(scenario_config, shared, archetype, accuracy)
                draws = Draws(scenario, archetype, cv, outcome.savings_after_risk, outcome.baseline_cost)
                all_draws.append(draws)
                tables.append(summarize(draws, thetas, tail_level))

    return MonteCarloResult(summary=pd.concat(tables, ignore_index=True), draws=all_draws)


def _warn_skip(scenario: str, archetype: str, cv: float, error: Exception) -> None:
    warnings.warn(f"skipped {scenario} / {archetype} / CV={cv}: {error}", stacklevel=3)


def largest_confident_theta(summary: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    """For each scenario, archetype and CV: the largest theta with P(profit) >= confidence_level.

    NaN means no theta on the grid reaches that confidence.
    """
    confident = summary[summary["p_profit"] >= confidence_level]
    best = confident.groupby(["scenario", "archetype", "cv"])["theta"].max()
    every = summary[["scenario", "archetype", "cv"]].drop_duplicates().set_index(["scenario", "archetype", "cv"])
    return every.join(best.rename("max_confident_theta")).reset_index()
