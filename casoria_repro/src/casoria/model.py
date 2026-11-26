"""The deterministic cost model of Casoria et al. (2026), Eq. 1-13.

Every cost is per analysis window (dT = 24 h).

Array conventions
-----------------
Every input that can be uncertain carries a leading *draw* axis n, so the same
functions serve one deterministic evaluation (n = 1) and a Monte Carlo run
(n = 10,000). Axes always come in this order:

    draw n, severity a, task k, role r

For example `role_task_effort` has shape (n, a, k, r). Fixed tables such as the
role split alpha (a, r) or hourly costs (r,) have no draw axis.

Extension point
---------------
`evaluate` accepts a list of `RiskAdjustment` functions. Each returns an extra
expected cost per window that the LLM brings (missed incidents, prompt
injection, data exposure, ...). Their sum is subtracted from the savings to
give `Outcome.savings_after_risk`. The base model passes none, so the two
savings figures are equal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import numpy as np

from casoria.params import ApiArchetype, Config, SelfHostedArchetype

MINUTES_PER_HOUR = 60
SECONDS_PER_HOUR = 3600
TOKENS_PER_MILLION = 1e6
BYTES_PER_GB = 1e9      # see the ASSUMPTION on gpu.memory_gb in the YAML
PARAMS_PER_BILLION = 1e9


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SocInputs:
    """Inputs shared by all archetypes. Each one may be drawn at random."""

    alerts_per_window: np.ndarray   # N,          shape (n,)
    effort_minutes: np.ndarray      # tau_a,      shape (n, a)
    speed_ratio: np.ndarray         # eta,        shape (n,)
    rework_penalty: np.ndarray      # beta,       shape (n,)
    llm_requests: np.ndarray        # req_a,      shape (n, a)
    tokens_in: np.ndarray           # tok_in_a,   shape (n, a)
    tokens_out: np.ndarray          # tok_out_a,  shape (n, a)


SOC_INPUT_NAMES = tuple(field.name for field in fields(SocInputs))


def central_value(config: Config, name: str) -> np.ndarray:
    """The YAML value of a shared input as an array: shape () or (a,)."""
    value = getattr(config, name)
    if hasattr(value, "as_array"):
        return value.as_array()
    return np.asarray(float(value))


def central_inputs(config: Config) -> SocInputs:
    """The YAML (central) values as a single draw, i.e. with n = 1."""
    return SocInputs(**{name: central_value(config, name)[None, ...] for name in SOC_INPUT_NAMES})


def central_accuracy(config: Config, archetype_name: str) -> np.ndarray:
    """E_m from the YAML as a single draw, shape (1,)."""
    return np.array([config.archetypes[archetype_name].accuracy])


# ---------------------------------------------------------------------------
# Workload and analyst time (Eq. 1-4)
# ---------------------------------------------------------------------------
def alerts_needing_analyst(
    alerts_per_window: np.ndarray,   # N,   (n,)
    severity_share: np.ndarray,      # p_a, (a,)
    analyst_share: np.ndarray,       # h_a, (a,)
) -> np.ndarray:
    """N * p_a * h_a, the alerts per window that reach an analyst (Eq. 1). Shape (n, a)."""
    return alerts_per_window[:, None] * severity_share * analyst_share


def task_effort_hours(
    effort_minutes: np.ndarray,      # tau_a, (n, a)
    task_split: np.ndarray,          # (k,)
) -> np.ndarray:
    """tau_{k,a}: hours of task k per alert of severity a. Shape (n, a, k)."""
    return (effort_minutes / MINUTES_PER_HOUR)[:, :, None] * task_split


def role_task_effort_hours(
    task_effort: np.ndarray,         # tau_{k,a},   (n, a, k)
    role_split: np.ndarray,          # alpha_{a,r}, (a, r)
) -> np.ndarray:
    """Eq. 3: tau_{k,r,a} = tau_{k,a} * alpha_{a,r}. Shape (n, a, k, r)."""
    return task_effort[:, :, :, None] * role_split[None, :, None, :]


def baseline_hours_per_role(
    analyst_alerts: np.ndarray,      # N p_a h_a,    (n, a)
    role_task_effort: np.ndarray,    # tau_{k,r,a},  (n, a, k, r)
) -> np.ndarray:
    """Eq. 1: tau_r^noLLM = N sum_a h_a p_a sum_k tau_{k,r,a}. Shape (n, r)."""
    return np.einsum("na,nakr->nr", analyst_alerts, role_task_effort)


def llm_hours_per_role(
    analyst_alerts: np.ndarray,      # N p_a h_a,    (n, a)
    role_task_effort: np.ndarray,    # tau_{k,r,a},  (n, a, k, r)
    accuracy: np.ndarray,            # E_m,  (n,)
    speed_ratio: np.ndarray,         # eta,  (n,)
    rework_penalty: np.ndarray,      # beta, (n,)
) -> np.ndarray:
    """Eq. 4: analyst hours per role when LLM m assists. Shape (n, r).

    With probability E_m the LLM is right and the analyst needs only the
    share eta of the usual time (prompting and checking). Otherwise the
    analyst redoes the task and spends (1 + beta) times the usual time.
    """
    # Give the per-draw scalars the shape (n, 1, 1, 1) so they broadcast.
    right = accuracy[:, None, None, None]
    eta = speed_ratio[:, None, None, None]
    beta = rework_penalty[:, None, None, None]

    hours_when_right = right * eta * role_task_effort
    hours_when_wrong = (1 - right) * (1 + beta) * role_task_effort
    return np.einsum("na,nakr->nr", analyst_alerts, hours_when_right + hours_when_wrong)


def personnel_cost(
    hours_per_role: np.ndarray,      # (n, r)
    cost_per_hour: np.ndarray,       # c_r, (r,)
) -> np.ndarray:
    """sum_r c_r * tau_r: Eq. 2 without the LLM, first term of Eq. 5 with it. Shape (n,).

    The fixed overhead C_inv is left out: it is the same with and without the
    LLM and cancels in the savings (Eq. 13).
    """
    return hours_per_role @ cost_per_hour


def break_even_accuracy(speed_ratio: float, rework_penalty: float) -> float:
    """Accuracy above which the LLM saves analyst time: beta / (1 + beta - eta).

    Follows from Eq. 4: E*eta + (1 - E)*(1 + beta) < 1.
    """
    return rework_penalty / (1 + rework_penalty - speed_ratio)


# ---------------------------------------------------------------------------
# Technology cost (Eq. 6-12)
# ---------------------------------------------------------------------------
def token_cost(
    analyst_alerts: np.ndarray,      # N p_a h_a,  (n, a)
    llm_requests: np.ndarray,        # req_a,      (n, a)
    tokens_in: np.ndarray,           # tok_in_a,   (n, a)
    tokens_out: np.ndarray,          # tok_out_a,  (n, a)
    price_in_per_million: float,     # gamma_in_m
    price_out_per_million: float,    # gamma_out_m
) -> np.ndarray:
    """Eq. 7-8: C_tok = N sum_a h_a p_a req_a (tok_in gamma_in + tok_out gamma_out) / 1e6. Shape (n,)."""
    price_per_request = (
        tokens_in * price_in_per_million + tokens_out * price_out_per_million
    ) / TOKENS_PER_MILLION
    return (analyst_alerts * llm_requests * price_per_request).sum(axis=1)


def token_demand(
    analyst_alerts: np.ndarray,      # N p_a h_a,  (n, a)
    llm_requests: np.ndarray,        # req_a,      (n, a)
    tokens_in: np.ndarray,           # tok_in_a,   (n, a)
    tokens_out: np.ndarray,          # tok_out_a,  (n, a)
) -> np.ndarray:
    """Eq. 11: D = N sum_a h_a p_a req_a (tok_in + tok_out), tokens per window. Shape (n,)."""
    return (analyst_alerts * llm_requests * (tokens_in + tokens_out)).sum(axis=1)


def gpus_per_replica(
    total_params_billion: float,     # delta_tot_m
    bytes_per_param: float,          # 2 for FP16
    gpu_memory_gb: float,            # V_gpu
) -> int:
    """Eq. 10, first factor: n_mem = ceil(2 delta_tot / V_gpu), GPUs to hold one copy of the weights."""
    weight_bytes = total_params_billion * PARAMS_PER_BILLION * bytes_per_param
    return int(np.ceil(weight_bytes / (gpu_memory_gb * BYTES_PER_GB)))


def replica_throughput(
    reference_throughput: float,     # phi_ref, tokens/s
    reference_active_billion: float, # delta_ref
    active_params_billion: float,    # delta_act_m
    gpus_in_replica: int,            # n_mem_m
) -> float:
    """Eq. 12: phi_m = phi_ref * (delta_ref / delta_act_m) * n_mem_m, tokens/s of one replica."""
    return reference_throughput * (reference_active_billion / active_params_billion) * gpus_in_replica


def replicas_needed(
    demand: np.ndarray,              # D, tokens per window, (n,)
    throughput: float,               # phi_m, tokens/s
    window_hours: float,             # dT
) -> np.ndarray:
    """Eq. 10, second factor: ceil(D / (phi_m dT 3600)). Shape (n,)."""
    return np.ceil(demand / (throughput * window_hours * SECONDS_PER_HOUR))


def infra_cost(
    gpus: np.ndarray,                # n_gpu_m, (n,)
    gpu_cost_per_hour: float,        # c_gpu
    window_hours: float,             # dT
) -> np.ndarray:
    """Eq. 9: C_infra = n_gpu c_gpu dT. Shape (n,)."""
    return gpus * gpu_cost_per_hour * window_hours


# ---------------------------------------------------------------------------
# Putting it together (Eq. 5, 6, 13)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Outcome:
    """Costs for one archetype. Every field has shape (n,) unless noted."""

    baseline_hours: np.ndarray            # tau_r^noLLM, shape (n, r)
    llm_hours: np.ndarray                 # tau_r^m,     shape (n, r)
    baseline_cost: np.ndarray             # C^noLLM (Eq. 2)
    personnel_cost_with_llm: np.ndarray   # sum_r c_r tau_r^m (Eq. 5, first term)
    token_cost: np.ndarray                # C_tok   (Eq. 7-8), 0 when self-hosted
    infra_cost: np.ndarray                # C_infra (Eq. 9), 0 for API models
    token_demand: np.ndarray              # D (Eq. 11), tokens per window
    gpus: np.ndarray                      # n_gpu (Eq. 10), 0 for API models
    replicas: np.ndarray                  # replicas (Eq. 10), 0 for API models
    risk_cost: np.ndarray                 # sum of RiskAdjustments, 0 in the base model

    @property
    def tech_cost(self) -> np.ndarray:
        """Eq. 6: C_tech = C_tok + C_infra."""
        return self.token_cost + self.infra_cost

    @property
    def llm_cost(self) -> np.ndarray:
        """Eq. 5: C^m = sum_r c_r tau_r^m + C_tech (C_inv cancels and is left out)."""
        return self.personnel_cost_with_llm + self.tech_cost

    @property
    def savings(self) -> np.ndarray:
        """Eq. 13: Delta C^m = C^noLLM - C^m, per window."""
        return self.baseline_cost - self.llm_cost

    @property
    def savings_after_risk(self) -> np.ndarray:
        """Delta C^m minus the extra risk costs. Equals `savings` in the base model."""
        return self.savings - self.risk_cost

    @property
    def savings_share(self) -> np.ndarray:
        """Delta C^m / C^noLLM; by Proposition 1 the largest theta with ROI > 0."""
        return self.savings_after_risk / self.baseline_cost


RiskAdjustment = Callable[[Config, SocInputs, str, np.ndarray], np.ndarray]
"""Extra expected cost per window that LLM m adds, shape (n,).

Called as `adjustment(config, inputs, archetype_name, accuracy)`. Planned:
per-stage missed incidents, prompt-injection misses, cloud data exposure.
"""


def evaluate(
    config: Config,
    inputs: SocInputs,
    archetype_name: str,
    accuracy: np.ndarray,                                   # E_m, (n,)
    risk_adjustments: Sequence[RiskAdjustment] = (),
) -> Outcome:
    """Evaluate Eq. 1-13 for one archetype, for every draw in `inputs`."""
    archetype = config.archetypes[archetype_name]
    window_hours = config.analysis_window_hours
    n_draws = len(inputs.alerts_per_window)

    # Workload and analyst time (Eq. 1-5)
    analyst_alerts = alerts_needing_analyst(
        inputs.alerts_per_window,
        config.severity_share.as_array(),
        config.analyst_share.as_array(),
    )
    effort = role_task_effort_hours(
        task_effort_hours(inputs.effort_minutes, config.task_split.as_array()),
        config.role_split.as_array(),
    )
    hours_without_llm = baseline_hours_per_role(analyst_alerts, effort)
    hours_with_llm = llm_hours_per_role(
        analyst_alerts, effort, accuracy, inputs.speed_ratio, inputs.rework_penalty
    )
    cost_per_hour = config.personnel_cost_per_hour.as_array()

    # Technology cost (Eq. 6-12)
    demand = token_demand(analyst_alerts, inputs.llm_requests, inputs.tokens_in, inputs.tokens_out)
    no_cost = np.zeros(n_draws)
    if isinstance(archetype, ApiArchetype):
        tokens = token_cost(
            analyst_alerts,
            inputs.llm_requests,
            inputs.tokens_in,
            inputs.tokens_out,
            archetype.price_in_per_million,
            archetype.price_out_per_million,
        )
        gpus, replicas, infra = no_cost, no_cost, no_cost
    elif isinstance(archetype, SelfHostedArchetype):
        gpus_in_replica = gpus_per_replica(
            archetype.total_params_billion, config.bytes_per_param, config.gpu.memory_gb
        )
        throughput = replica_throughput(
            config.reference_model.throughput_tokens_per_s,
            config.reference_model.active_params_billion,
            archetype.active_params_billion,
            gpus_in_replica,
        )
        replicas = replicas_needed(demand, throughput, window_hours)
        gpus = gpus_in_replica * replicas
        infra = infra_cost(gpus, config.gpu.cost_per_hour, window_hours)
        tokens = no_cost
    else:  # pragma: no cover - pydantic only allows the two types above
        raise TypeError(f"unknown archetype type {type(archetype).__name__}")

    risk = sum(
        (adjust(config, inputs, archetype_name, accuracy) for adjust in risk_adjustments),
        start=no_cost,
    )

    return Outcome(
        baseline_hours=hours_without_llm,
        llm_hours=hours_with_llm,
        baseline_cost=personnel_cost(hours_without_llm, cost_per_hour),
        personnel_cost_with_llm=personnel_cost(hours_with_llm, cost_per_hour),
        token_cost=tokens,
        infra_cost=infra,
        token_demand=demand,
        gpus=gpus,
        replicas=replicas,
        risk_cost=risk,
    )


def evaluate_central(
    config: Config,
    archetype_name: str,
    risk_adjustments: Sequence[RiskAdjustment] = (),
) -> Outcome:
    """Evaluate one archetype at the YAML values, without uncertainty (n = 1)."""
    return evaluate(
        config,
        central_inputs(config),
        archetype_name,
        central_accuracy(config, archetype_name),
        risk_adjustments,
    )
