# Casoria et al. (2026) — reproduction

A small, readable reproduction of the cost model and Monte Carlo analysis in

> Casoria, Araldo, Patanè & Romano (2026). *From Hype to ROI: An Economic Model
> for LLM Adoption in Information Systems and Security Operations.*
> arXiv:2610.33352.

It is the core model for our seminar paper; the extra risk terms (missed
incidents, prompt injection, data exposure) will be added later through the
hook described at the end.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest            # 72 tests, < 2 s
.venv/bin/python run_all.py # ~25 s, writes results/
```

`run_all.py` takes an optional path to another YAML file.

## The model in one page

Everything is a cost per 24-hour window. Indices: role *r* ∈ {L1, L2, L3},
severity *a* ∈ {low, medium, high}, task *k* ∈ {triage, analysis,
documentation}, model archetype *m*.

| Step | Equation | Code (`src/casoria/`) |
|---|---|---|
| Analyst hours without an LLM | τ_r^noLLM = N Σ_a h_a p_a Σ_k τ_{k,r,a}, with τ_{k,r,a} = τ_{k,a} α_{a,r} (Eq. 1, 3) | `model.baseline_hours_per_role` |
| Baseline cost | C^noLLM = Σ_r c_r τ_r^noLLM (Eq. 2; C_inv cancels) | `model.personnel_cost` |
| Analyst hours with LLM *m* | right with probability E_m → share η of the time; wrong → (1 + β) times the time (Eq. 4) | `model.llm_hours_per_role` |
| Token cost (API) | N Σ_a h_a p_a req_a (tok_in γ_in + tok_out γ_out) / 10⁶ (Eq. 7–8) | `model.token_cost` |
| GPU cost (self-hosted) | GPUs per replica = ⌈2 δ_tot / V_gpu⌉, replicas = ⌈D / (φ dT 3600)⌉, cost = n_gpu c_gpu dT (Eq. 9–12) | `model.gpus_per_replica`, `replicas_needed`, `infra_cost` |
| Savings | ΔC = C^noLLM − C^m (Eq. 13) | `model.Outcome.savings` |
| ROI | θ = C_cap / C^noLLM, ROI = (ΔC − C_cap) / C_cap (Eq. 14–15) | `metrics.roi` |
| Proposition 1 | ROI > y ⇔ θ < ΔC / ((1 + y) C^noLLM) (Eq. 16) | `metrics.max_theta` |
| Missed incidents | ROI > 0 ⇔ λ < (ΔC − θ C^noLLM) / C_miss (Eq. 17) | `metrics.max_missed_incident_rate` |
| Uncertainty | P(ROI > 0 \| CV), VaR₁₀%, plus CVaR₁₀% (Eq. 19–20) | `metrics.probability_of_profit`, `value_at_risk`, `conditional_value_at_risk` |

A useful consequence of Eq. 4: the LLM saves analyst time only if its
accuracy exceeds β / (1 + β − η), which is **75%** at the baseline η = β = 0.75
(`model.break_even_accuracy`).

### Array conventions

Every input that can be uncertain has a leading **draw** axis, so the same
functions do one deterministic evaluation (1 draw) and a Monte Carlo run
(10,000 draws). Axis order is always draw *n*, severity *a*, task *k*, role *r*;
e.g. `role_task_effort` has shape (n, a, k, r).

## Files

```
config/baseline.yaml        every number, each with its source; our choices say "# ASSUMPTION"
src/casoria/params.py       pydantic models: loads + validates the YAML, dotted-path overrides
src/casoria/model.py        Eq. 1-13 as small pure functions; evaluate() puts them together
src/casoria/metrics.py      ROI, Proposition 1, Eq. 17, P(profit), VaR, CVaR
src/casoria/distributions.py mean + CV -> scipy Beta / Gamma (method of moments)
src/casoria/simulate.py     Monte Carlo runner -> summary table per scenario x archetype x CV x theta
src/casoria/sensitivity.py  one-at-a-time tornado, parameters listed in the YAML
src/casoria/plots.py        figures
run_all.py                  writes results/
tests/                      the validation targets below
```

## Outputs (`results/`)

| File | Content |
|---|---|
| `central_values.csv` | deterministic costs, savings, ROI at θ = 10%, Eq. 17 bound, per scenario and archetype |
| `mc_summary.csv` | P(profit), mean ROI, VaR₁₀%, CVaR₁₀% per scenario × archetype × CV × θ |
| `max_confident_theta.csv` | the largest θ with P(profit) ≥ 95% |
| `sensitivity.csv` | tornado table, ranked per archetype |
| `qualitative_checks.txt` | our results next to the paper's qualitative statements |
| `figures/p_profit_<scenario>.png` | P(profit) against θ, one panel per CV |
| `figures/roi_distribution_<scenario>.png` | ROI at θ = 10%, boxes = quartiles, whiskers = 5th–95th percentile |
| `figures/tornado.png` | sensitivity of ROI at θ = 10% |

## Validation

The tests check these deterministic central values (no uncertainty, before capex):

| | Paper / our recomputation | Code |
|---|---|---|
| Baseline analyst hours/day | L1 34.75, L2 29.5, L3 4.5 | ✓ |
| C^noLLM | $4,818/day | $4,817.85 |
| Token demand D | 20.6M tokens/day, 1 replica per self-hosted model | 20.625M, 1 |
| API-High savings | +$708 (14.7%) | +$707.7 |
| API-Mid | +$573 (11.9%) | +$573.2 |
| API-Low | −$930 (−19.3%) | −$930.4 |
| SH-High | −$334 (−6.9%), 5 GPUs = $960/day | −$333.7 |
| SH-Mid | −$192 (−4.0%) | −$192.0 |
| SH-Low | −$1,637 (−34.0%) | −$1,637.4 |
| Break-even accuracy | 0.75 | 0.75 |
| Full automation, API-High | ΔC / C^noLLM ≈ 0.84 | 0.837 |

Proposition 1 and Eq. 17 are property-tested on 100,000 random inputs.

### Against the paper's qualitative findings (reported, not enforced)

| Paper says | We get |
|---|---|
| At CV = 5%, only high-accuracy API models are confidently profitable, for θ < ~10% | Only API-High and API-Mid reach P(profit) ≥ 95%, but only up to θ = 0.04 and 0.02. At θ = 0.10, P(profit) is 0.81 for API-High |
| At CV ≥ 25%, no model is | ✓ none reaches 95% anywhere on the θ grid |
| VaR₁₀%: the worst decile loses more than half the capex almost everywhere | ✓ VaR₁₀% > 0.5 on 99% of the conservative grid; the exceptions are API-High (θ ≤ 0.14) and API-Mid (θ ≤ 0.09) at CV = 5% |
| η dominates for accurate models; β matters for weak ones, most for SH-Low; N only for self-hosted | ✓ all three (see `tornado.png`) |
| SH-High loses money because GPU rent absorbs the time saved | ✓ $626/day of analyst time saved vs $960/day GPU rent |
| Full automation: API-High stays profitable only if a missed incident occurs less than once every ~7 years | ✓ 7.9 years at θ = 10% |

The gap in the first row most likely comes from how finely the paper draws
its inputs. We draw η, β and E once per draw; if the paper draws them
independently per task, role or alert, the noise averages out and the
spread of the ROI shrinks. We have not tuned anything to close the gap.

## Assumptions (all also marked in the YAML or docstrings)

1. **High-severity output tokens = 4,000**, from the paper's stated 4:1 ratio.
2. **Request and token counts are per alert that needs an analyst**, not per
   task and role. This reproduces D = 20.6M tokens/day.
3. **Mean + CV → distribution by method of moments.** Gamma: shape = 1/CV²,
   scale = mean·CV². Beta: sd = mean·CV, k = mean(1 − mean)/sd² − 1,
   a = mean·k, b = (1 − mean)·k.
4. **Infeasible Beta configurations are skipped.** API-High's accuracy
   0.92 at CV = 35% needs sd² = 0.104 > 0.92·0.08 = 0.074, so no Beta
   distribution fits. The run warns and writes NaN rows with a `note`.
5. **θ is applied per draw**: C_cap = θ · C^noLLM of that draw. This is what
   makes N irrelevant for API models, as the paper reports.
6. **Shared draws across archetypes** (common random numbers): all six
   models are evaluated on the same random workload; only accuracy is drawn
   per model. Every stream has its own seed.
7. **Requests and tokens are drawn as continuous Gamma values**, not integers.
8. **"Confidently profitable" means P(profit) ≥ 95%**, a value we chose.
9. **1 GB = 10⁹ bytes**; this changes no GPU count here.

## Extending the model (planned risk terms)

`model.evaluate()` takes a list of **risk adjustments**: functions

```python
def my_risk(config, inputs, archetype_name, accuracy) -> np.ndarray:  # $ per window, shape (n,)
    ...
```

Their sum is subtracted from the savings:
`Outcome.savings_after_risk = ΔC − Σ risk`. The Monte Carlo runner and the
tornado already use `savings_after_risk`, so ROI, P(profit), VaR and CVaR
pick the new terms up without other changes. Planned terms:

- per-stage missed incidents (recall ρ, spot-check share κ, detection rate d, breach probability q)
- prompt-injection misses (attack rate A, success rate s_inj)
- a data-exposure cost for cloud models only

New parameters go into the YAML (and a pydantic model in `params.py`); a new
uncertain input becomes a field of `model.SocInputs` and a line under
`monte_carlo.distributions`; a new tornado bar is one line under
`sensitivity.parameters`.
