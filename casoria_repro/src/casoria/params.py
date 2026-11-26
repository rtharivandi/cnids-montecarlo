"""Load and validate the parameter file (config/baseline.yaml).

All numbers live in the YAML file. This module only checks that the file is
complete and consistent (shares sum to one, probabilities lie in [0, 1], no
unknown keys) and turns the per-severity / per-role tables into numpy arrays
in one fixed order, so the model code never has to look up names.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Fixed index order used by every array in the model.
SEVERITIES = ("low", "medium", "high")           # index a
TASKS = ("triage", "analysis", "documentation")  # index k
ROLES = ("L1", "L2", "L3")                       # index r

# Inputs that the Monte Carlo run may draw at random. `accuracy` belongs to
# each archetype; the others are shared by all archetypes (see model.SocInputs).
UNCERTAIN_INPUTS = (
    "alerts_per_window",
    "effort_minutes",
    "speed_ratio",
    "rework_penalty",
    "llm_requests",
    "tokens_in",
    "tokens_out",
    "accuracy",
)

# Floating-point slack when checking that shares sum to one.
SUM_TOLERANCE = 1e-9

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]
Positive = Annotated[float, Field(gt=0.0)]


class _Strict(BaseModel):
    """Base class: unknown keys are an error and objects are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Small tables
# ---------------------------------------------------------------------------
class PerSeverity(_Strict):
    """One value per alert severity a."""

    low: NonNegative
    medium: NonNegative
    high: NonNegative

    def as_array(self) -> np.ndarray:
        """Values in SEVERITIES order, shape (a,)."""
        return np.array([getattr(self, a) for a in SEVERITIES], dtype=float)


class PerTask(_Strict):
    """One value per SOC task k."""

    triage: NonNegative
    analysis: NonNegative
    documentation: NonNegative

    def as_array(self) -> np.ndarray:
        """Values in TASKS order, shape (k,)."""
        return np.array([getattr(self, k) for k in TASKS], dtype=float)


class PerRole(_Strict):
    """One value per analyst role r."""

    L1: NonNegative
    L2: NonNegative
    L3: NonNegative

    def as_array(self) -> np.ndarray:
        """Values in ROLES order, shape (r,)."""
        return np.array([getattr(self, r) for r in ROLES], dtype=float)


class RoleSplit(_Strict):
    """alpha_{a,r}: share of the work on a severity-a alert done by role r."""

    low: PerRole
    medium: PerRole
    high: PerRole

    def as_array(self) -> np.ndarray:
        """Shape (a, r)."""
        return np.stack([getattr(self, a).as_array() for a in SEVERITIES])


# ---------------------------------------------------------------------------
# Model archetypes
# ---------------------------------------------------------------------------
class ApiArchetype(_Strict):
    """A model used through a cloud API: pays per token, no GPUs (Eq. 6)."""

    deployment: Literal["api"]
    model_name: str
    accuracy: Fraction                       # E_m
    price_in_per_million: NonNegative        # gamma_in_m, $ per 1M input tokens
    price_out_per_million: NonNegative       # gamma_out_m, $ per 1M output tokens


class SelfHostedArchetype(_Strict):
    """A self-hosted open-weight model: pays GPU rent, no tokens (Eq. 6)."""

    deployment: Literal["self_hosted"]
    model_name: str
    accuracy: Fraction                       # E_m
    total_params_billion: Positive           # delta_tot_m
    active_params_billion: Positive          # delta_act_m


Archetype = Annotated[ApiArchetype | SelfHostedArchetype, Field(discriminator="deployment")]


# ---------------------------------------------------------------------------
# Infrastructure, capex, Monte Carlo, sensitivity
# ---------------------------------------------------------------------------
class GpuConfig(_Strict):
    name: str
    memory_gb: Positive                      # V_gpu
    cost_per_hour: NonNegative               # c_gpu


class ReferenceModel(_Strict):
    name: str
    throughput_tokens_per_s: Positive        # phi_ref
    active_params_billion: Positive          # delta_ref


class ThetaGrid(_Strict):
    start: Positive
    stop: Positive
    num: Annotated[int, Field(ge=2)]

    def values(self) -> np.ndarray:
        """The theta values to sweep, shape (num,)."""
        return np.linspace(self.start, self.stop, self.num)


class CapexConfig(_Strict):
    amortization_years: Positive
    baseline_theta: Positive                 # theta = C_cap / C_noLLM
    theta_grid: ThetaGrid


class MonteCarloConfig(_Strict):
    draws: Annotated[int, Field(gt=0)]
    seed: int
    cv_levels: Annotated[list[Positive], Field(min_length=1)]
    tail_level: Annotated[float, Field(gt=0.0, lt=1.0)]        # z in VaR_z, CVaR_z
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    distributions: dict[str, Literal["beta", "gamma"]]

    @model_validator(mode="after")
    def _known_inputs(self) -> MonteCarloConfig:
        unknown = set(self.distributions) - set(UNCERTAIN_INPUTS)
        if unknown:
            raise ValueError(
                f"monte_carlo.distributions names unknown inputs {sorted(unknown)}; "
                f"allowed: {list(UNCERTAIN_INPUTS)}"
            )
        return self


class SensitivityEntry(_Strict):
    path: str                                          # dotted path into the YAML
    bounds: tuple[float, float] | None = None          # clip low/high values to this range


class SensitivityConfig(_Strict):
    relative_change: Annotated[float, Field(gt=0.0, lt=1.0)]
    parameters: list[SensitivityEntry]


# ---------------------------------------------------------------------------
# The whole file
# ---------------------------------------------------------------------------
class Config(_Strict):
    """Everything in config/baseline.yaml, validated."""

    analysis_window_hours: Positive                    # dT
    personnel_cost_per_hour: PerRole                   # c_r
    alerts_per_window: NonNegative                     # N
    severity_share: PerSeverity                        # p_a
    analyst_share: PerSeverity                         # h_a
    effort_minutes: PerSeverity                        # tau_a
    task_split: PerTask                                # tau_{k,a} = tau_a * task_split_k
    role_split: RoleSplit                              # alpha_{a,r}
    speed_ratio: Fraction                              # eta
    rework_penalty: NonNegative                        # beta
    llm_requests: PerSeverity                          # req_a
    tokens_in: PerSeverity                             # tok_in_a
    tokens_out: PerSeverity                            # tok_out_a
    gpu: GpuConfig
    bytes_per_param: Positive
    reference_model: ReferenceModel
    archetypes: Annotated[dict[str, Archetype], Field(min_length=1)]
    cost_of_missed_incident: NonNegative               # C_miss
    capex: CapexConfig
    monte_carlo: MonteCarloConfig
    scenarios: dict[str, dict[str, Any]]
    sensitivity: SensitivityConfig

    @model_validator(mode="after")
    def _shares_are_consistent(self) -> Config:
        _require_sum_one("severity_share", self.severity_share.as_array())
        _require_sum_one("task_split", self.task_split.as_array())
        for severity, row in zip(SEVERITIES, self.role_split.as_array()):
            _require_sum_one(f"role_split.{severity}", row)
        if np.any(self.analyst_share.as_array() > 1.0):
            raise ValueError("analyst_share values must lie in [0, 1]")
        return self

    def scenario(self, name: str) -> Config:
        """This config with the overrides of scenario `name` applied."""
        if name not in self.scenarios:
            raise KeyError(f"unknown scenario {name!r}; known: {list(self.scenarios)}")
        return apply_overrides(self, self.scenarios[name])


def _require_sum_one(name: str, values: np.ndarray) -> None:
    if abs(values.sum() - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1, got {values.sum():.6f}")


# ---------------------------------------------------------------------------
# Loading and overriding
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> Config:
    """Read a YAML parameter file and validate it."""
    with open(path, encoding="utf-8") as file:
        return Config.model_validate(yaml.safe_load(file))


def get_path(data: Mapping[str, Any], path: str) -> Any:
    """Look up a dotted path such as "gpu.cost_per_hour" in nested dicts."""
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            raise KeyError(f"no parameter at path {path!r} (missing {key!r})")
        node = node[key]
    return node


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set an existing dotted path in nested dicts (typos raise KeyError)."""
    *parents, leaf = path.split(".")
    node = get_path(data, ".".join(parents)) if parents else data
    if leaf not in node:
        raise KeyError(f"no parameter at path {path!r} (missing {leaf!r})")
    node[leaf] = value


def apply_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """Return a new, re-validated Config with dotted-path overrides applied."""
    data = copy.deepcopy(config.model_dump())
    for path, value in overrides.items():
        set_path(data, path, value)
    return Config.model_validate(data)
