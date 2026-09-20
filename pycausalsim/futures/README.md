# pycausalsim.futures

PyCausalSim is retrodictive by design. It learns a graph from existing data and
estimates interventions within that data's support. Forecasting inverts both
halves: the graph is specified rather than discovered, and every intervention
of interest lies far outside observed support.

This module supplies what that inversion needs. The simulation and validation
machinery of the core library still applies. The discovery layer does not.

| Component | Gap it closes |
|---|---|
| `TemporalSCM` | Graphs are cross-sectional. Futures are trajectories, and the dynamics live in lags and accumulation. |
| `MechanismPrior` | Fitted surfaces extrapolate badly and silently. Declared mechanism extrapolates loudly. |
| `StructuralIntervention` | `do()` sets a value. It does not delete an edge. Most futures questions are structural. |
| `Scenario` | Scenarios should be legible to people who do not write Python. |
| `rank_drivers` | Which lever matters, rather than which lever fits a persuasive narrative. |
| `ClaimSet` | Futurism has no accountability mechanism. This one emits dated claims and scores them. |
| `backtest` | An engine that cannot retrodict has no standing to forecast. |

## Quick start

```python
import numpy as np
from pycausalsim.futures import TemporalSCM, Scenario, compare, rank_drivers

scm = TemporalSCM(horizon=range(2026, 2042), draws=4000, seed=7)

scm.parameter("attrition",  "normal", loc=0.045, scale=0.008)
scm.parameter("conversion", "normal", loc=0.085, scale=0.012)
scm.lever("apprenticeship", default=0.05)

scm.exogenous("token_price", lambda c: 0.97 * 10.0 ** (-0.52 * c.t))
scm.add_equation("automation", f_automation, parents=["token_price"])
scm.add_equation("junior", f_junior,
                 parents=["automation", "apprenticeship"],
                 pre_history=[100.0] * 11 + [88.0])

scm.state("senior_stock",
          lambda c: (c.lag("senior_stock", 1) * (1 - c.p.attrition)
                     + c.p.conversion * c.lag("junior", 10)),
          parents=["junior:lag(10)", "senior_stock:lag(1)"],
          init=lambda p: p.conversion / p.attrition * 100.0)

drift = Scenario("Drift")
funded = Scenario("Funded apprenticeship")
funded.at(2027).set("apprenticeship", ramp_to=0.72, over=4)

result = compare(scm, [drift, funded], target="senior_stock")
print(result.table())
```

## Time as a first-class citizen

A node is declared with `add_equation` if it is memoryless and with `state` if
it carries memory. Parents are written `"name"` for the current step and
`"name:lag(k)"` for k steps back. Lagged parents impose no ordering constraint,
which is what keeps the unrolled graph acyclic. A cycle at lag zero is an error
with a message that says to break it with a lag.

Equations receive a `Context`:

| Call | Meaning |
|---|---|
| `c["automation"]` | current-step value, after any structural masking |
| `c.lag("junior", 10)` | value ten steps back |
| `c.p.attrition` | sampled parameter array, shape `(draws,)` |
| `c.t`, `c.year` | step index and calendar year |
| `c["apprenticeship"]` | lever value on this scenario's path |

Values before the horizon come from `pre_history` on the node being lagged, or
from `init` for state variables. Ten years of lag needs ten years of history,
and the engine says so by name rather than returning zeros.

## Declared extrapolation

```python
scm.mechanism("automation ~ token_price",
              form="logistic", saturates_at=1.0,
              valid_range=(0.0005, 100.0), outside="flag")
```

`outside` is `flag`, `clip` or `error`. Every step where a parent leaves its
range is recorded, so a run reports which conclusions rest on observed data and
which rest on assumed mechanism:

```
Support: edges extrapolated outside their declared range
  automation ~ token_price     2030-2041  up to 100% of draws  (flag)
  Conclusions downstream of these edges rest on declared mechanism,
  not on observed data.
```

That paragraph belongs next to any forecast this library produces.

## Structural interventions

```python
from pycausalsim.futures import SI

scm.structural_rule(
    when="automation > 0.55",
    apply=SI.remove_edge("junior", "senior_stock", on_removed=0.0),
    note="training stops being a free byproduct of junior output")
```

Rules fire per draw. If the condition holds in 40 percent of draws at a step,
the edge is severed in exactly those draws and the population splits into
worlds with different graphs. `when` takes a simple comparison string or a
callable over the context. `SI` provides `remove_edge`, `add_edge` and
`rewire`.

This is the piece no other causal library has. Automation does not lower the
value of apprenticeship. It severs the mechanism by which apprenticeship used
to happen for free, and that is an edge deletion caused by a value.

## Scenarios

```python
kc = Scenario("Ownership and institutions")
kc.at(2027).set("apprenticeship", ramp_to=0.72, over=4)
kc.at(2031).set("income_floor", to=0.60)

kc.describe()          # human-readable
kc.to_json("kc.json")  # round-trips through Scenario.from_dict
kc.without("ownership")
kc | other             # merge
```

## Comparing and ranking

```python
cmp = compare(scm, [drift, transfers, full], target="wellbeing", params=p)
cmp.table()
cmp.diff("full", "Drift")
cmp.share_of_gap("transfers", "full")     # 0.27
rank_drivers(scm, drift, target="wellbeing", intensity=0.75)
```

All scenarios run on the same parameter draws, so differences are paired and
the intervals are tighter than independent sampling would give.
`share_of_gap` is the number that separates a policy which helps from one
which substitutes for the thing that would actually work.

## Falsifiable claims

```python
scm.claim("entry_gap_25pct",
          "entry-level gap in exposed occupations exceeds 25 percent",
          test=lambda r: r["junior"] <= 75.0,
          resolve_by=2029,
          source="Stanford Digital Economy Lab payroll series")

claims = cmp.falsifiable_claims(scenario="Drift", threshold=0.70)
claims.export("claims_2026.json")

# years later
ClaimSet.load("claims_2026.json").score({"entry_gap_25pct": True})
# {'n': 1, 'brier': 0.036, 'accuracy': 1.0, ...}
```

Brier score is 0 for perfect and 0.25 for a coin flip. Unresolved keys are
skipped, so a claim set can be scored progressively as the years arrive.

## Backtesting

```python
bt = backtest(scm, observations={
    "junior": {2026: 88.0},
    "capital_share": {2026: 0.38},
})
print(bt.summary())
```

Reports per-anchor error, MAPE, and how many anchors fell inside the stated 90
percent interval. Coverage far above 90 percent means the priors are not doing
any work; far below means the model is more confident than its record
justifies. Backtesting a specified model scores whether the declared mechanism
reproduces observed anchors, not whether refitted coefficients do.

## Worked example

`examples/postlabor_futures.py` is the full model behind *Simulating the Note
from the Future*: a fifteen-year post-labor transition with four policy levers,
a ten-year apprenticeship lag, declared extrapolation on the cost of cognition,
and an optional structural rule that severs the training byproduct channel.

```bash
python examples/postlabor_futures.py
```

A regression test asserts that it reproduces the published figures exactly.

## Design notes

**Why the unrolled graph stays acyclic.** Feedback in a dynamic system is
feedback across time. Writing it as a lag makes the unrolled graph a DAG, so
identification results for static SCMs carry over without modification. The
engine enforces this rather than trusting the modeller.

**Why structural rules fire per draw.** A threshold crossing is uncertain like
anything else. Collapsing it to a population average would hide the bimodality
that matters most, namely that some futures sever the edge and some do not.

**Why parameters are shared across scenarios.** Paired comparison. Running
each scenario on independent draws would widen every interval for no reason
and would make small policy differences unmeasurable.

**What this module deliberately does not do.** It does not discover structure.
Every edge is a stated judgment, and the Monte Carlo quantifies uncertainty
over coefficients rather than over structure. Structural uncertainty is almost
certainly larger, and the honest way to explore it is to publish the graph and
let readers rewire it.
