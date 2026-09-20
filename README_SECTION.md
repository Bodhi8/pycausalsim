# Insert into the main README

Two edits. Replace the opening description, and add the Futures section after
"Built-in Validation" and before "Installation". The Project Structure and
Roadmap blocks at the end also need updating.

---

## 1. Replace the opening paragraph

**PyCausalSim** is a Python framework for causal discovery and inference
through simulation. It answers two kinds of question with the same machinery.

**Backward:** what caused the outcome already in your data? Discover the graph,
estimate interventions, validate against unmeasured confounding.

**Forward:** what happens under a policy nobody has tried yet? Specify a graph,
unroll it over a horizon, and simulate scenarios far outside observed support,
with the extrapolation declared rather than silent.

Most causal libraries do only the first. The second needs time, declared
mechanism, interventions on structure rather than value, and a way to score
predictions after the fact. That is `pycausalsim.futures`.

---

## 2. New section: Futures and Scenario Simulation

### 8. Futures and Scenario Simulation

Causal discovery is retrodictive. It learns a graph from existing data and
estimates interventions within that data's support. Forecasting inverts both
halves, and `pycausalsim.futures` supplies what the inversion needs.

```python
from pycausalsim.futures import TemporalSCM, Scenario, SI, compare, rank_drivers

scm = TemporalSCM(horizon=range(2026, 2042), draws=4000, seed=7)

scm.parameter("attrition", "normal", loc=0.045, scale=0.008)
scm.lever("apprenticeship", default=0.05)

# lags make the unrolled graph acyclic, so identification carries over
scm.state("senior_stock",
          lambda c: (c.lag("senior_stock", 1) * (1 - c.p.attrition)
                     + c.p.conversion * c.lag("junior", 10)),
          parents=["junior:lag(10)", "senior_stock:lag(1)"],
          init=lambda p: p.conversion / p.attrition * 100.0)

# extrapolation is declared, never silent
scm.mechanism("automation ~ token_price", form="logistic",
              valid_range=(0.0005, 100.0), outside="flag")

# interventions on structure, not only on value
scm.structural_rule(when="automation > 0.55",
                    apply=SI.remove_edge("junior", "senior_stock"),
                    note="training stops being a free byproduct")

funded = Scenario("Funded apprenticeship")
funded.at(2027).set("apprenticeship", ramp_to=0.72, over=4)

cmp = compare(scm, [Scenario("Drift"), funded], target="wellbeing")
print(cmp.table())
print(cmp.share_of_gap("Drift", "Funded apprenticeship"))
rank_drivers(scm, Scenario("Drift"), target="wellbeing")
```

**What it adds**

- **`TemporalSCM`** unrolls a graph over a horizon. Parents are written
  `"name"` or `"name:lag(k)"`. State variables carry memory. A cycle at lag
  zero is an error, because feedback in a dynamic system is feedback across
  time.
- **`MechanismPrior`** requires a declared functional form and validity range
  on any edge that will be driven outside observed data. Every excursion is
  recorded, so results report which conclusions rest on data and which rest on
  assumed mechanism.
- **`StructuralIntervention`** adds, removes and rewires edges, and rules fire
  per draw. `do()` sets a value; it does not delete an edge, and the most
  consequential futures questions are structural.
- **`Scenario`** is a declarative bundle of timed interventions, readable and
  editable by people who do not write Python.
- **`ClaimSet`** emits dated, checkable claims with probabilities and a named
  source, exports them to JSON, and scores them later with a Brier score.
- **`backtest`** scores the model against observed anchors and reports whether
  the stated intervals are calibrated.

Full documentation: [`pycausalsim/futures/README.md`](pycausalsim/futures/README.md).
Worked example: [`examples/postlabor_futures.py`](examples/postlabor_futures.py),
a fifteen-year post-labor transition model with four policy levers and a
ten-year apprenticeship lag.

---

## 3. Add to Use Cases

6. **Policy Scenario Analysis**: compare interventions over a horizon, rank
   levers by causal effect, and publish claims that can be scored later
7. **Strategic Forecasting**: simulate futures outside observed support with
   the extrapolation declared rather than hidden

---

## 4. Replace the Project Structure block

```
pycausalsim/
├── pycausalsim/
│   ├── __init__.py         # Package exports
│   ├── simulator.py        # Main CausalSimulator class
│   ├── attribution.py      # Marketing Attribution
│   ├── experiment.py       # A/B Test Analysis
│   ├── results.py          # Result classes
│   ├── models/             # Structural Causal Models
│   ├── discovery/          # Causal discovery algorithms
│   ├── uplift/             # Uplift modeling
│   ├── agents/             # Agent-based simulation
│   ├── validation/         # Sensitivity analysis
│   ├── adapters/           # DoWhy/EconML integration
│   ├── futures/            # Temporal SCMs and scenario forecasting
│   │   ├── temporal.py     #   TemporalSCM, Context, RunResult
│   │   ├── mechanism.py    #   MechanismPrior, SupportReport
│   │   ├── structural.py   #   StructuralIntervention, StructuralRule
│   │   ├── scenario.py     #   Scenario language
│   │   ├── results.py      #   Comparison, rank_drivers, ClaimSet
│   │   └── backtest.py     #   Retrodiction scoring
│   ├── utils/              # Utilities
│   └── visualization/      # Plotting
├── tests/                  # Test suite
├── examples/               # Example scripts
├── pyproject.toml          # Package configuration
└── README.md
```

---

## 5. Replace the Roadmap

### Roadmap

**Shipped in 0.2**

- Temporal SCMs with lags, state variables and per-draw structural rules
- Declared mechanism priors with out-of-support reporting
- Scenario language, paired comparison, driver ranking
- Falsifiable claim sets with Brier scoring
- Backtesting against observed anchors

**Next**

- **Agent layer for futures.** `agents/` exists but does not yet plug into
  `TemporalSCM`. Aggregate models report population means, which cannot
  distinguish two people with identical material conditions and
  non-comparable lives. Heterogeneous institution access, geography and cohort
  would recover that.
- **Geography.** No spatial dimension today, which is the largest omission in
  the post-labor example. Regional bottleneck prices and migration between
  regions are the first two edges to add.
- **Structural uncertainty.** The Monte Carlo covers coefficients, not
  structure. Sampling over a distribution of graphs is the honest version and
  is considerably harder.
- **Scenario DSL in YAML**, so scenarios can be authored and reviewed without
  touching Python at all.
- **A public claim ledger.** Exported claim sets are only useful if someone
  scores them. A registry of dated predictions with resolution sources would
  give forecasting the accountability mechanism it has never had.
