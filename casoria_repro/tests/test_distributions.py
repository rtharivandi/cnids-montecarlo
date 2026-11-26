"""Mean + CV -> distribution (method of moments)."""

import numpy as np
import pytest

from casoria.distributions import Constant, InfeasibleBetaError, beta_from_mean_cv, gamma_from_mean_cv


@pytest.mark.parametrize("mean", [0.75, 1000.0, 8000.0])
@pytest.mark.parametrize("cv", [0.05, 0.25, 0.35])
def test_gamma_has_requested_mean_and_cv(mean, cv):
    dist = gamma_from_mean_cv(mean, cv)
    assert dist.mean() == pytest.approx(mean)
    assert dist.std() / dist.mean() == pytest.approx(cv)


@pytest.mark.parametrize("mean", [0.45, 0.75, 0.92])
@pytest.mark.parametrize("cv", [0.05, 0.15, 0.25])
def test_beta_has_requested_mean_and_cv(mean, cv):
    dist = beta_from_mean_cv(mean, cv)
    assert dist.mean() == pytest.approx(mean)
    assert dist.std() / dist.mean() == pytest.approx(cv)


def test_vector_means_give_one_distribution_per_entry():
    dist = gamma_from_mean_cv(np.array([10.0, 30.0, 60.0]), 0.25)
    draws = dist.rvs(size=(200_000, 3), random_state=np.random.default_rng(1))
    assert draws.mean(axis=0) == pytest.approx([10, 30, 60], rel=0.01)


def test_infeasible_beta_raises_a_clear_error():
    # API-High accuracy 0.92 at CV 35%: sd^2 = 0.104 > 0.92 * 0.08 = 0.074
    with pytest.raises(InfeasibleBetaError, match="mean\\(1 - mean\\)"):
        beta_from_mean_cv(0.92, 0.35)


@pytest.mark.parametrize("make", [beta_from_mean_cv, gamma_from_mean_cv])
def test_zero_mean_is_a_constant(make):
    """eta = 0 (full automation) and beta = 0 (no rework) cannot vary."""
    dist = make(0.0, 0.25)
    assert isinstance(dist, Constant)
    assert np.all(dist.rvs(size=(5,), random_state=np.random.default_rng(0)) == 0)


def test_beta_rejects_means_outside_zero_one():
    with pytest.raises(ValueError):
        beta_from_mean_cv(1.0, 0.05)
