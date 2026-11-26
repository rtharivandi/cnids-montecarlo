"""Turn a mean and a coefficient of variation (CV) into a scipy distribution.

The paper names the families (Beta for eta and accuracy, Gamma for alert
volume, effort, beta, tokens and requests) and uses one common CV for all
uncertain inputs, so each input has standard deviation  sd = mean * CV.

ASSUMPTION: the paper does not state how mean and CV map to the distribution
parameters. We use the method of moments: the distribution has exactly the
given mean and standard deviation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


class InfeasibleBetaError(ValueError):
    """No Beta distribution on [0, 1] has the requested mean and spread."""


@dataclass(frozen=True)
class Constant:
    """A degenerate distribution that always returns its value.

    Used when the mean is 0 (e.g. eta = 0 in the full-automation scenario,
    beta = 0 without rework penalty): no spread is possible around 0.
    """

    value: np.ndarray

    def rvs(self, size: tuple[int, ...], random_state: np.random.Generator) -> np.ndarray:
        return np.broadcast_to(self.value, size).astype(float)

    def mean(self) -> np.ndarray:
        return self.value

    def std(self) -> np.ndarray:
        return np.zeros_like(self.value)


def gamma_from_mean_cv(mean: np.ndarray | float, cv: float) -> Any:
    """Gamma with the given mean and CV: shape = 1 / CV^2, scale = mean * CV^2."""
    mean = np.asarray(mean, dtype=float)
    if np.all(mean == 0):
        return Constant(mean)
    if np.any(mean <= 0):
        raise ValueError(f"Gamma needs a positive mean, got {mean}")
    return stats.gamma(a=1 / cv**2, scale=mean * cv**2)


def beta_from_mean_cv(mean: np.ndarray | float, cv: float) -> Any:
    """Beta on [0, 1] with the given mean and CV (method of moments).

    sd = mean * CV,  k = mean (1 - mean) / sd^2 - 1,  a = mean k,  b = (1 - mean) k.
    Needs sd^2 < mean (1 - mean); otherwise no Beta distribution fits.
    """
    mean = np.asarray(mean, dtype=float)
    if np.all(mean == 0):
        return Constant(mean)
    if np.any((mean <= 0) | (mean >= 1)):
        raise ValueError(f"Beta needs a mean strictly between 0 and 1, got {mean}")
    variance = (mean * cv) ** 2
    max_variance = mean * (1 - mean)
    if np.any(variance >= max_variance):
        raise InfeasibleBetaError(
            f"no Beta distribution has mean {mean} and CV {cv}: "
            f"sd^2 = {variance} must be below mean(1 - mean) = {max_variance}"
        )
    k = max_variance / variance - 1
    return stats.beta(a=mean * k, b=(1 - mean) * k)


FAMILIES: dict[str, Callable[[np.ndarray | float, float], Any]] = {
    "beta": beta_from_mean_cv,
    "gamma": gamma_from_mean_cv,
}


def from_mean_cv(family: str, mean: np.ndarray | float, cv: float) -> Any:
    """A frozen distribution of `family` ("beta" or "gamma") with this mean and CV."""
    return FAMILIES[family](mean, cv)
