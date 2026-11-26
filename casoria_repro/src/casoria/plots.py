"""Figures: P(profit) vs theta, ROI distributions, tornado chart.

Colours follow the archetype (fixed order, never by rank); line style adds a
second cue: solid for API models, dashed for self-hosted ones.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from casoria.params import ApiArchetype, Config
from casoria.simulate import Draws

# Categorical palette, slots 1-8 in fixed order (validated for colour-vision deficiency).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#2b2a27"
TEXT_MUTED = "#6b6a64"
GRID = "#e4e3de"
LOW_COLOR, HIGH_COLOR = SERIES_COLORS[0], SERIES_COLORS[1]

plt.rcParams.update(
    {
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def archetype_style(config: Config) -> dict[str, dict[str, str]]:
    """Colour and line style per archetype, in YAML order."""
    return {
        name: {
            "color": SERIES_COLORS[i % len(SERIES_COLORS)],
            "linestyle": "-" if isinstance(archetype, ApiArchetype) else "--",
        }
        for i, (name, archetype) in enumerate(config.archetypes.items())
    }


def plot_profit_probability(summary: pd.DataFrame, config: Config, scenario: str) -> Figure:
    """Eq. 19: P(ROI > 0) against theta, one panel per CV level."""
    styles = archetype_style(config)
    cv_levels = config.monte_carlo.cv_levels
    fig, axes = plt.subplots(1, len(cv_levels), figsize=(3.2 * len(cv_levels), 3.2), sharey=True)
    for ax, cv in zip(np.atleast_1d(axes), cv_levels):
        panel = summary[(summary["scenario"] == scenario) & (summary["cv"] == cv)]
        for name, style in styles.items():
            line = panel[panel["archetype"] == name]
            ax.plot(line["theta"], line["p_profit"], linewidth=2, label=name, **style)
        ax.axhline(config.monte_carlo.confidence_level, color=TEXT_MUTED, linewidth=1, linestyle=":")
        ax.set_title(f"CV = {cv:.0%}")
        ax.set_xlabel("theta = capex / baseline cost")
        ax.set_ylim(-0.02, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("P(ROI > 0)")
    _figure_legend(fig, styles)
    fig.suptitle(f"Probability of profit, {scenario.replace('_', ' ')} scenario", color=TEXT, y=1.08)
    return fig


def plot_roi_distributions(draws: Sequence[Draws], config: Config, scenario: str, theta: float) -> Figure:
    """ROI per draw at one theta: box = quartiles, whiskers = 5th-95th percentile."""
    styles = archetype_style(config)
    cv_levels = config.monte_carlo.cv_levels
    fig, axes = plt.subplots(1, len(cv_levels), figsize=(3.2 * len(cv_levels), 3.4), sharey=True)
    for ax, cv in zip(np.atleast_1d(axes), cv_levels):
        by_name = {d.archetype: d for d in draws if d.scenario == scenario and d.cv == cv}
        names = [name for name in styles if name in by_name]   # skipped configurations are missing
        boxes = ax.boxplot(
            [by_name[name].roi(theta) for name in names],
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            widths=0.6,
            medianprops={"color": TEXT, "linewidth": 1.5},
        )
        for patch, name in zip(boxes["boxes"], names):
            patch.set_facecolor(styles[name]["color"])
            patch.set_edgecolor(styles[name]["color"])
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=45, ha="right")
        ax.axhline(0, color=TEXT_MUTED, linewidth=1)
        ax.set_title(f"CV = {cv:.0%}")
    np.atleast_1d(axes)[0].set_ylabel(f"ROI at theta = {theta:.0%}")
    fig.suptitle(f"ROI distribution, {scenario.replace('_', ' ')} scenario", color=TEXT, y=1.02)
    return fig


def plot_tornado(table: pd.DataFrame, config: Config) -> Figure:
    """One-at-a-time sensitivity: ROI when each parameter moves down (blue) or up (orange)."""
    names = list(config.archetypes)
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows))
    change = config.sensitivity.relative_change
    for ax, name in zip(axes.flat, names):
        part = table[table["archetype"] == name].sort_values("swing")   # largest bar on top
        base = part["roi_base"].iloc[0]
        positions = np.arange(len(part))
        ax.barh(positions, part["roi_low"] - base, left=base, height=0.7, color=LOW_COLOR, label=f"-{change:.0%}")
        ax.barh(positions, part["roi_high"] - base, left=base, height=0.7, color=HIGH_COLOR, label=f"+{change:.0%}")
        ax.axvline(base, color=TEXT, linewidth=1)
        ax.set_yticks(positions, [_short_label(p) for p in part["parameter"]])
        ax.set_title(f"{name} (ROI {base:+.2f})")
        ax.grid(axis="y", visible=False)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        f"Sensitivity of ROI at theta = {config.capex.baseline_theta:.0%} (conservative scenario)",
        color=TEXT,
        y=1.07,
    )
    fig.supxlabel("ROI", color=TEXT)
    fig.tight_layout()
    return fig


def _figure_legend(fig: Figure, styles: dict[str, dict[str, str]]) -> None:
    handles = [plt.Line2D([], [], linewidth=2, **style) for style in styles.values()]
    fig.legend(handles, list(styles), loc="upper center", ncol=len(styles), bbox_to_anchor=(0.5, 1.02))


def _short_label(path: str) -> str:
    """'archetypes.{archetype}.accuracy' -> 'accuracy'; 'gpu.cost_per_hour' -> 'gpu.cost_per_hour'."""
    return path.removeprefix("archetypes.{archetype}.")
