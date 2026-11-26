"""Deterministic central values must match the paper (and our recomputation)."""

import pytest

from casoria.model import break_even_accuracy, evaluate_central
from casoria.params import apply_overrides

# Savings per day ($) and savings / C_noLLM at central values, before capex.
EXPECTED_SAVINGS = {
    "API-High": (708, 0.147),
    "API-Mid": (573, 0.119),
    "API-Low": (-930, -0.193),
    "SH-High": (-334, -0.069),
    "SH-Mid": (-192, -0.040),
    "SH-Low": (-1637, -0.340),
}
SELF_HOSTED = ["SH-High", "SH-Mid", "SH-Low"]


def test_baseline_analyst_hours(config):
    outcome = evaluate_central(config, "API-High")
    l1, l2, l3 = outcome.baseline_hours[0]
    assert l1 == pytest.approx(34.75)
    assert l2 == pytest.approx(29.5)
    assert l3 == pytest.approx(4.5)
    assert outcome.baseline_hours.sum() == pytest.approx(68.75)


def test_baseline_cost(config):
    outcome = evaluate_central(config, "API-High")
    assert outcome.baseline_cost[0] == pytest.approx(4818, abs=1)


def test_token_demand(config):
    outcome = evaluate_central(config, "API-High")
    assert outcome.token_demand[0] == pytest.approx(20.6e6, rel=0.005)


@pytest.mark.parametrize("name", SELF_HOSTED)
def test_every_self_hosted_model_needs_one_replica(config, name):
    assert evaluate_central(config, name).replicas[0] == 1


def test_sh_high_needs_five_gpus(config):
    outcome = evaluate_central(config, "SH-High")
    assert outcome.gpus[0] == 5
    assert outcome.infra_cost[0] == pytest.approx(960)


@pytest.mark.parametrize("name", list(EXPECTED_SAVINGS))
def test_savings_per_day(config, name):
    savings, share = EXPECTED_SAVINGS[name]
    outcome = evaluate_central(config, name)
    assert outcome.savings[0] == pytest.approx(savings, abs=1)
    assert outcome.savings_share[0] == pytest.approx(share, abs=0.0005)


def test_api_models_pay_no_gpus_and_self_hosted_pay_no_tokens(config):
    for name in config.archetypes:
        outcome = evaluate_central(config, name)
        if name in SELF_HOSTED:
            assert outcome.token_cost[0] == 0
        else:
            assert outcome.infra_cost[0] == 0


def test_break_even_accuracy_at_baseline(config):
    assert break_even_accuracy(config.speed_ratio, config.rework_penalty) == pytest.approx(0.75)


def test_break_even_accuracy_leaves_analyst_time_unchanged(config):
    """At E = beta / (1 + beta - eta), Eq. 4 gives exactly the baseline hours."""
    threshold = break_even_accuracy(config.speed_ratio, config.rework_penalty)
    at_threshold = apply_overrides(config, {"archetypes.API-High.accuracy": threshold})
    outcome = evaluate_central(at_threshold, "API-High")
    assert outcome.llm_hours.sum() == pytest.approx(outcome.baseline_hours.sum())


def test_full_automation_api_high(config):
    """eta = 0: the paper reports theta up to ~80-90% stays profitable."""
    outcome = evaluate_central(config.scenario("full_automation"), "API-High")
    assert outcome.savings_share[0] == pytest.approx(0.84, abs=0.01)


def test_without_risk_adjustments_both_savings_agree(config):
    outcome = evaluate_central(config, "API-Mid")
    assert outcome.risk_cost[0] == 0
    assert outcome.savings_after_risk[0] == outcome.savings[0]


def test_risk_adjustment_is_subtracted(config):
    """The extension hook: a constant extra cost lowers the savings by that amount."""

    def flat_cost(config, inputs, archetype_name, accuracy):
        return 100.0 + 0 * accuracy

    plain = evaluate_central(config, "API-High")
    adjusted = evaluate_central(config, "API-High", risk_adjustments=[flat_cost])
    assert adjusted.savings_after_risk[0] == pytest.approx(plain.savings[0] - 100)
    assert adjusted.savings[0] == pytest.approx(plain.savings[0])
