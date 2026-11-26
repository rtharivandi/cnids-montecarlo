"""One-at-a-time (tornado) sensitivity of the ROI at the baseline theta.

Each entry in `sensitivity.parameters` of the YAML is a dotted path. The
parameter is moved to value x (1 - change) and value x (1 + change), all other
inputs stay at their central values, and we record the ROI (Eq. 15) at
`capex.baseline_theta`. "{archetype}" in a path is replaced by each
archetype's name, so `archetypes.{archetype}.accuracy` moves each model's own
accuracy. Tables (per-severity or per-role values) are scaled entry by entry.
Adding a line to the YAML adds a bar; no code change is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from casoria.metrics import roi
from casoria.model import evaluate_central
from casoria.params import Config, SensitivityEntry, apply_overrides, get_path


def scaled(value: Any, factor: float, bounds: tuple[float, float] | None) -> Any:
    """Multiply a number, or every number in a (nested) mapping, by factor; clip to bounds."""
    if isinstance(value, Mapping):
        return {key: scaled(item, factor, bounds) for key, item in value.items()}
    new_value = value * factor
    if bounds is not None:
        new_value = float(np.clip(new_value, *bounds))
    return new_value


def central_roi(config: Config, archetype_name: str, theta: float) -> float:
    """ROI (Eq. 15) at central values for one archetype."""
    outcome = evaluate_central(config, archetype_name)
    return float(roi(outcome.savings_after_risk, outcome.baseline_cost, theta)[0])


def tornado(config: Config) -> pd.DataFrame:
    """ROI at low and high settings of every listed parameter, for every archetype.

    Rows are sorted by archetype, then by swing |ROI_high - ROI_low| (largest
    first); `rank` 1 is the most influential parameter for that archetype.
    """
    tables = []
    for archetype in config.archetypes:
        rows = [_tornado_row(config, archetype, entry) for entry in config.sensitivity.parameters]
        table = pd.DataFrame(rows).sort_values("swing", ascending=False, kind="stable")
        table["rank"] = np.arange(1, len(table) + 1)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _tornado_row(config: Config, archetype: str, entry: SensitivityEntry) -> dict[str, Any]:
    """Move one parameter down and up and record the ROI for one archetype."""
    theta = config.capex.baseline_theta
    change = config.sensitivity.relative_change
    path = entry.path.format(archetype=archetype)
    value = get_path(config.model_dump(), path)
    low = scaled(value, 1 - change, entry.bounds)
    high = scaled(value, 1 + change, entry.bounds)
    roi_low = central_roi(apply_overrides(config, {path: low}), archetype, theta)
    roi_high = central_roi(apply_overrides(config, {path: high}), archetype, theta)
    is_number = not isinstance(value, Mapping)
    return {
        "archetype": archetype,
        "parameter": entry.path,
        "central_value": value if is_number else np.nan,   # tables are scaled entry by entry
        "low_value": low if is_number else np.nan,
        "high_value": high if is_number else np.nan,
        "relative_change": change,
        "roi_base": central_roi(config, archetype, theta),
        "roi_low": roi_low,
        "roi_high": roi_high,
        "swing": abs(roi_high - roi_low),
    }
