---
title: 'PyCausalSim: A Python framework for causal discovery and effect estimation through simulation'
tags:
  - Python
  - causal inference
  - structural causal models
  - simulation
  - causal discovery
  - marketing analytics
authors:
  - name: Brian Curry
    orcid: 0009-0002-8555-7475
    affiliation: 1
affiliations:
  - name: Vector1 Research, Kansas City, MO, United States
    index: 1
date: 3 August 2026
bibliography: paper.bib
---

# Summary

`PyCausalSim` is a Python framework for causal discovery and causal effect
estimation built around a simulation-first workflow. Rather than treating
causal discovery, effect estimation, and robustness checking as separate
exercises in separate tools, `PyCausalSim` unifies them around structural
causal models (SCMs) [@pearl2009] that can be discovered from data, specified
by hand, simulated under interventions, and stress-tested against
misspecification.

The core object, `CausalSimulator`, accepts observational data, discovers a
candidate causal graph using one of several discovery algorithms — spanning
constraint-based (PC, FCI), score-based (GES, FGES), functional (LiNGAM
[@shimizu2006]), and continuous-optimization (NOTEARS [@zheng2018])
families — and exposes `simulate_intervention()`, a do-operator
[@pearl2009] over the fitted structural model that returns interventional
effect estimates with uncertainty intervals. Companion methods rank all
causal drivers of a target (`rank_drivers()`) and search for intervention
policies under constraints (`optimize_policy()`). A
`StructuralCausalModel` class supports counterfactual generation with
evidence conditioning. A validation module implements sensitivity analysis
with confounding bounds, placebo tests, and refutation tests in the spirit
of @sharma2020. Applied modules provide experiment analysis with
doubly-robust effect estimation [@bang2005] and heterogeneity analysis,
uplift modeling [@gutierrez2017] with effect-based user segmentation, and
marketing attribution using causal Shapley values with constrained budget
optimization.

`PyCausalSim` is implemented on the scientific Python stack (`numpy`,
`pandas`, `scipy`, `scikit-learn` [@pedregosa2011]), with optional adapter
integrations to the wider causal ecosystem including DoWhy [@sharma2020]
and EconML [@econml2019]. It is released under the MIT license.

# Statement of need

Applied analysts routinely need to answer interventional questions — "what
happens to the outcome if we change this variable?" — from observational
data. The dominant workflow answers correlational questions instead, because
the causal toolchain is fragmented: graph discovery lives in one library,
identification and estimation in another, sensitivity analysis in a third,
and simulation of candidate structural models is typically hand-rolled.
This fragmentation has practical consequences: analysts skip the robustness
steps that don't fit the pipeline, and competing causal structures that are
observationally indistinguishable — a mediator versus a confounder, for
example — go untested even when they imply opposite decisions.

`PyCausalSim` addresses this gap with a single coherent API in which the
simulation of structural models is the connective tissue: discovered or
hypothesized graphs become generative objects that can be simulated under
known interventions, compared against experimental or quasi-experimental
evidence, and probed for the sensitivity of their conclusions. This
simulation-first design also makes the framework suitable as a synthetic
data-generating engine for downstream research — for example, generating
training distributions of known causal environments for simulation-based
inference [@cranmer2020] — a use case the author is actively developing for
marketing measurement, where structural parameters such as advertising
carryover and saturation must be inferred from short, confounded panels.

The intended audience is applied data scientists and computational
researchers — particularly in marketing science, business analytics, and
economics — who need decision-grade causal answers without assembling a
bespoke toolchain, as well as researchers who need a programmable SCM
simulation engine. Existing libraries each cover part of this surface:
DoWhy [@sharma2020] provides an identification-and-refutation workflow,
EconML [@econml2019] and CausalML [@chen2020] provide estimators for
heterogeneous effects, and causal-learn provides discovery algorithms.
`PyCausalSim` is complementary: it interoperates with these libraries
through adapters while contributing the simulation layer — interventional
simulation over discovered or specified SCMs, structure comparison under
intervention, driver ranking, and policy search — as a first-class, unified
workflow.

# Functionality

The framework is organized around five capabilities:

- **Discovery.** Six causal discovery algorithms behind a common interface —
  PC and FCI (constraint-based), GES and FGES (score-based), LiNGAM
  (functional, for non-Gaussian data), and NOTEARS (continuous
  optimization) — returning candidate graphs that can be constrained with
  domain knowledge and visualized.
- **Simulation and intervention.** Fitted or user-specified SCMs are
  generative: `simulate_intervention(variable, value)` implements the
  do-operator with Monte Carlo simulation and returns effect estimates
  with confidence intervals; `rank_drivers()` orders all variables by
  causal effect on the target; `optimize_policy()` searches intervention
  settings under user constraints; counterfactuals can be generated
  conditional on observed evidence.
- **Validation.** `validate()` produces confounding bounds at varying
  assumed strengths, placebo tests, and multiple refutation methods,
  summarizing how fragile each conclusion is to violations of the causal
  assumptions.
- **Experimentation.** `ExperimentAnalysis` estimates treatment effects
  from randomized or quasi-experimental data, including a doubly-robust
  estimator, and analyzes effect heterogeneity across covariates.
- **Applied modules.** Uplift modeling with segmentation into persuadable,
  sure-thing, lost-cause, and sleeping-dog cohorts; marketing attribution
  via causal Shapley values with constrained budget optimization.

# Acknowledgements

The author thanks the maintainers of the open-source causal inference
ecosystem — in particular DoWhy, EconML, causal-learn, and CausalML — whose
work this framework builds upon and interoperates with.

# References
