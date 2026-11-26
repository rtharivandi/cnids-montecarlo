"""Run the whole reproduction and write CSVs and figures to results/.

Usage:  python run_all.py [path/to/config.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never open windows

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from casoria import plots  # noqa: E402
from casoria.metrics import max_missed_incident_rate, roi, total_capex  # noqa: E402
from casoria.model import break_even_accuracy, evaluate_central  # noqa: E402
from casoria.params import ROLES, Config, load_config  # noqa: E402
from casoria.sensitivity import tornado  # noqa: E402
from casoria.simulate import largest_confident_theta, run_monte_carlo  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "baseline.yaml"
RESULTS = HERE / "results"
DAYS_PER_YEAR = 365


def central_values(config: Config) -> pd.DataFrame:
    """Deterministic results at the YAML values, for every scenario and archetype."""
    theta = config.capex.baseline_theta
    rows = []
    for scenario in config.scenarios:
        scenario_config = config.scenario(scenario)
        for archetype in config.archetypes:
            outcome = evaluate_central(scenario_config, archetype)
            hours = dict(zip(ROLES, outcome.baseline_hours[0]))
            savings, baseline = outcome.savings_after_risk, outcome.baseline_cost
            missed_rate = max_missed_incident_rate(savings, baseline, theta, config.cost_of_missed_incident)[0]
            rows.append(
                {
                    "scenario": scenario,
                    "archetype": archetype,
                    "model": config.archetypes[archetype].model_name,
                    **{f"baseline_hours_{role}": h for role, h in hours.items()},
                    "baseline_cost": baseline[0],
                    "personnel_cost_with_llm": outcome.personnel_cost_with_llm[0],
                    "token_cost": outcome.token_cost[0],
                    "infra_cost": outcome.infra_cost[0],
                    "gpus": outcome.gpus[0],
                    "replicas": outcome.replicas[0],
                    "token_demand": outcome.token_demand[0],
                    "savings": savings[0],
                    "savings_share": outcome.savings_share[0],      # = max theta for ROI > 0 (Prop. 1)
                    "break_even_accuracy": break_even_accuracy(
                        scenario_config.speed_ratio, scenario_config.rework_penalty
                    ),
                    "roi_at_baseline_theta": roi(savings, baseline, theta)[0],
                    "max_missed_rate_per_window": missed_rate,          # Eq. 17
                    "years_between_misses_at_break_even": (
                        1 / (missed_rate * DAYS_PER_YEAR * 24 / config.analysis_window_hours)
                        if missed_rate > 0
                        else float("nan")
                    ),
                    "capex_total_at_baseline_theta": total_capex(
                        theta, baseline[0], config.analysis_window_hours, config.capex.amortization_years
                    ),
                }
            )
    return pd.DataFrame(rows)


def qualitative_checks(
    config: Config, central: pd.DataFrame, confident: pd.DataFrame, sensitivity: pd.DataFrame
) -> list[str]:
    """Compare against the paper's qualitative statements. Reports, never fails."""
    level = config.monte_carlo.confidence_level
    lines = [f'"Confidently profitable" = P(profit) >= {level:.0%} (see monte_carlo.confidence_level).', ""]

    conservative = confident[confident["scenario"] == "conservative"]
    lowest_cv = min(config.monte_carlo.cv_levels)
    lines.append(f"Paper: at CV = 5%, only high-accuracy API archetypes are confidently profitable, for theta < ~10%.")
    for _, row in conservative[conservative["cv"] == lowest_cv].iterrows():
        theta = row["max_confident_theta"]
        lines.append(f"  {row['archetype']:9s} CV {lowest_cv:.0%}: " + (f"up to theta = {theta:.2f}" if pd.notna(theta) else "never"))

    lines.append("")
    lines.append("Paper: at CV >= 25%, no archetype is confidently profitable.")
    high_cv = conservative[conservative["cv"] >= 0.25]
    reaching = high_cv[high_cv["max_confident_theta"].notna()]
    if reaching.empty:
        lines.append("  Reproduced: none reaches the confidence level at any theta on the grid.")
    else:
        for _, row in reaching.iterrows():
            lines.append(f"  NOT reproduced: {row['archetype']} at CV {row['cv']:.0%} up to theta = {row['max_confident_theta']:.2f}")

    lines.append("")
    lines.append("Paper: SH-High loses money because GPU cost absorbs the efficiency gain.")
    sh_high = central[(central["scenario"] == "conservative") & (central["archetype"] == "SH-High")].iloc[0]
    time_saved = sh_high["baseline_cost"] - sh_high["personnel_cost_with_llm"]
    lines.append(
        f"  Analyst cost saved ${time_saved:,.0f}/day vs GPU cost ${sh_high['infra_cost']:,.0f}/day "
        f"-> savings ${sh_high['savings']:,.0f}/day"
    )

    lines.append("")
    lines.append("Paper: with full automation, API-High stays profitable only if a missed incident occurs less than once every ~7 years.")
    full = central[(central["scenario"] == "full_automation") & (central["archetype"] == "API-High")].iloc[0]
    lines.append(f"  At theta = {config.capex.baseline_theta:.0%}: once every {full['years_between_misses_at_break_even']:.1f} years")

    lines.append("")
    lines.append("Paper (tornado): eta dominates for accurate models; beta matters as much for weak ones;")
    lines.append("alert volume N only matters for self-hosted models.")
    for archetype, part in sensitivity.groupby("archetype", sort=False):
        top = ", ".join(part.sort_values("rank")["parameter"].head(3))
        n_swing = part.loc[part["parameter"] == "alerts_per_window", "swing"]
        n_text = f"{n_swing.iloc[0]:.3f}" if not n_swing.empty else "n/a"
        lines.append(f"  {archetype:9s} top 3: {top}; swing of N: {n_text}")
    return lines


def main(config_path: Path) -> None:
    config = load_config(config_path)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    central = central_values(config)
    central.to_csv(RESULTS / "central_values.csv", index=False)

    monte_carlo = run_monte_carlo(config)
    monte_carlo.summary.to_csv(RESULTS / "mc_summary.csv", index=False)
    confident = largest_confident_theta(monte_carlo.summary, config.monte_carlo.confidence_level)
    confident.to_csv(RESULTS / "max_confident_theta.csv", index=False)

    sensitivity = tornado(config.scenario("conservative"))
    sensitivity.to_csv(RESULTS / "sensitivity.csv", index=False)

    for scenario in config.scenarios:
        _save(plots.plot_profit_probability(monte_carlo.summary, config, scenario), f"p_profit_{scenario}.png")
        _save(
            plots.plot_roi_distributions(monte_carlo.draws, config, scenario, config.capex.baseline_theta),
            f"roi_distribution_{scenario}.png",
        )
    _save(plots.plot_tornado(sensitivity, config), "tornado.png")

    checks = qualitative_checks(config, central, confident, sensitivity)
    (RESULTS / "qualitative_checks.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("\n".join(checks))
    print(f"\nWrote results to {RESULTS}")


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(RESULTS / "figures" / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG)
