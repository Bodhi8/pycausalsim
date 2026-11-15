# Changelog

## Unreleased

### Changed: marketing attribution rewritten and tested against known channel effects
- `shapley` valued a set of channels by the raw conversion rate of users who saw
  them, so baseline conversions and the effect of co-occurring channels were
  credited to whichever channel was present. A channel with no effect received
  about a fifth of the credit in a four-channel test. Shapley values are now
  computed on a fitted model of conversion, exactly for up to 8 channels, and
  baseline conversions are credited to no channel.
- `logistic` and `linear` used the absolute value of coefficients, so a channel
  that lowered conversion was credited for it. They now report signed
  contributions; a harmful channel gets a zero share and a warning.
- `uplift` and `get_incremental_effects()` compared exposed with unexposed users
  without adjustment. They now adjust for the other channels and for
  `covariate_cols`.
- `markov` started journeys uniformly across channels and redistributed a removed
  channel's traffic, and returned equal shares on ordinary data. It is now the
  standard first-order removal effect.
- `optimize_budget` could return an allocation that did not sum to the budget when
  constraints applied. It now always sums to the total and rejects limits that
  cannot be met. It is documented as a proportional rule of thumb, not an optimum.
- `compare_methods` no longer hides failures or changes the fitted state.
- The global NumPy random state is no longer reseeded.
- New: `covariate_cols` (adjust for measured confounders), `path_col` (ordered
  journeys for the path heuristics), `interactions=True`, `get_contributions()`,
  `get_baseline()`, `incremental_effects_table()` with bootstrap intervals.
- When no channel has a positive contribution the shares are all zero, with a
  warning, instead of an equal split.
- Touchpoint detection now takes only 0/1 columns.
- `first_touch`, `last_touch` and `markov` warn when no `path_col` is given, since
  they then take column order as the order of contact.

### Changed: causal discovery rewritten and tested against known graphs
- **PC** oriented every unshielded triple as a collider and guessed remaining
  directions from variances, so chains and forks came back as colliders. It is now
  PC-stable with separating sets, collider orientation and Meek's rules.
- **GES** returned every edge in both directions, with spurious edges. It is now
  greedy equivalence search (Chickering, 2002) with the BIC score.
- **LiNGAM** followed column order. It is now DirectLiNGAM, and warns when the data
  look Gaussian, where direction is not identified.
- **NOTEARS** returned a complete graph with self-loops. It is now the linear
  augmented-Lagrangian method with an acyclic result.
- **Hybrid** is defined as the GES graph without the edges PC's tests reject.
- PC, GES and hybrid return the equivalence class: an edge whose direction the data
  cannot determine is listed in both directions. `undirected_edges(graph)` lists
  them and `is_dag(graph)` checks a graph.
- New `tiers` argument to `CausalDiscovery.discover`: ordering knowledge as
  `{variable: tier}`. No edge runs from a later tier to an earlier one; GES uses the
  ordering during the search. `target` is shorthand for a last tier of its own.
- `CausalSimulator.discover_graph` passes declared confounders and the target as
  tiers, so a learned graph can no longer make the target a cause of a treatment,
  and warns, naming the edges, whenever it orients one by assumption.
- The `correlation` method is documented as a screening tool, not causal discovery.
- The module no longer claims FCI, which is not implemented.


### Fixed
- `CausalSimulator.simulate_intervention` now counts effects that travel through
  mediators. The default `monte_carlo` method predicted the target from its direct
  parents only, so for `x -> m -> y` it reported an effect of zero.
- The `backdoor` method adjusted for every other variable in the data, mediators
  included, which removed mediated effects. It now adjusts for the parents of the
  treatment (or the confounders you supply) and never for variables downstream of it.
- P-values divided the bootstrap spread by the square root of the number of replicates,
  so they shrank toward zero as `n_simulations` grew. With no true effect, 95% of
  simulated datasets were reported significant at 0.05. The bootstrap standard deviation
  is now used as the standard error.
- Confidence intervals now come from refitting the model on each bootstrap resample.
  The `scm` method's interval previously reflected only injected noise and was far too
  narrow.
- A non-DataFrame `data` argument raises `TypeError` as documented.
- `validate()` reported every effect as fully robust: its "bounds" were the estimate plus
  or minus a fixed percentage of itself, so they never crossed zero and the robustness
  value was always 1.0. It now computes the robustness value and confounding bounds of
  Cinelli and Hazlett (2020) from a linear outcome model. The placebo and stability
  checks compared correlations with an effect in the target's units; they now re-estimate
  the effect itself.
- `optimize_policy` evaluated settings from the target's direct parents only and searched
  over variables that do not cause the target. It now propagates joint interventions
  through the graph and searches over causes of the target.
- `rank_drivers` silently accepted `method='shap'` and `'gradient'` and ignored them, and
  ranked a variable whose effect could not be estimated as zero. Unknown methods now
  raise, and such a variable is left out with a warning.
- `set_graph` and `discover_graph` accepted graphs with cycles and self-loops. Both now
  check that the graph is acyclic and names only columns of the data. Edges that
  discovery could not orient are oriented from the declared confounder and treatment
  roles, or by column order with a warning.
- `CausalEffect.summary()` printed the effect multiplied by 100 and called it the
  relative change. It now reports the effect as a share of the baseline outcome.
- README examples for `validate()` called methods that do not exist, and
  `examples/complete_example.py` generated data in which conversion was clipped to its
  floor for almost every row, so there was no effect to find.

### Known issues
- `discover_graph(method='notears')` can return a graph with a self-loop. This is now
  reported as an error instead of being used.

### Changed
- `simulate_intervention` takes `n_bootstrap` (default `min(n_simulations, 100)`), the
  number of bootstrap refits. `n_simulations` on the returned effect is the number used.
  Refitting is slower than the old resampling.
- `monte_carlo` and `scm` are now the same estimator.
- The point estimate comes from the full data; the bootstrap supplies the interval.
- Bootstrap draws use the simulator's own random generator, so a fixed `random_state`
  gives the same result regardless of other use of NumPy's global state.
- A warning is raised when the intervention value is outside the variable's observed
  range, since tree-based mechanisms cannot extrapolate.

### Added
- `tests/test_known_truth.py`: intervention effects, driver ranking, sensitivity analysis,
  policy optimization and discovery-to-simulation are checked against simulated data with
  known answers.
- `CausalEffect.baseline_outcome`; `n_bootstrap` on `rank_drivers` and `validate`;
  `grid_points` on `optimize_policy`.

## 0.6.0 (unreleased)

### Added
- `futures.simulate_loop` and `futures.compare_loops`: run a measurement loop
  (`papilon.Loop`, Papilon 1.0) against a simulated brand quarter after quarter. The world
  responds to the spend the loop chose; recommended lift tests run in the world and come
  back with the noise of a matched-market design. Each cycle is graded on the error and
  coverage of the estimated returns, the true gain of the plan and realized sales. Compares
  a model fitted once, one refit each quarter, and the full loop.
- `media_plan_scm` accepts any carryover kernel and logistic saturation from Papilon 1.0's
  `posterior_curves()`.

## 0.5.0 (2026-11-11)

### Changed
- License changed from MIT to the Apache License, Version 2.0, the same license as
  Papilon. It adds an explicit patent grant from contributors, requires derivative
  works to keep the NOTICE file, and does not license the project names. Versions
  0.4.0 and earlier remain available under the MIT License. Every source file carries
  the license header; a test checks it.

### Added
- `futures.worlds_from_graph(graph, ...)`: simulated brands whose structure is a
  declared knowledge graph (a `papilon.MarketingGraph` or its `to_dict()`). Drivers
  with `sets_spend` and `drives` edges become confounders, observed or hidden;
  `mediates` makes one channel's spend follow another's advertising, and the answer
  key reports each channel's direct and total return. `hide=[...]` withholds a driver
  from the data to show what not measuring it costs.
- `futures.lift_test(worlds, i, channel, ...)`: a simulated geo holdout or scale-up on
  any world, replayed through the causal model so the true incremental return is
  exact, with a standard error from the design (weeks, geo share, pre-period fit).
  The result drops into `papilon.MMM(roi_priors=...)`.
- `futures.grade(estimator, worlds)` and `futures.leaderboard({...}, worlds)`: score
  any marketing mix model, from any library, on the same worlds: interval coverage,
  ROAS error and bias, share of the best possible budget gain captured and share of
  brands where its plan lost money. Ships `papilon_estimator` and `ols_estimator`.
- `marketing_worlds(response="s_curve")`: S-shaped response curves, the functional
  form a concave model gets wrong. `marketing_worlds(competitor=True)`: unobserved
  competitor campaigns that depress demand while display spend rises in answer.
- `media_plan_scm` plays plans forward on S-shaped curves when the posterior curves
  carry `shape` (Papilon 0.5).
- `MarketingWorlds.channels`, `.controls`, `.target`, `.l_max`; `truth()` reports
  `shape`.

### Unchanged
- Worlds made with the 0.4 arguments are identical to 0.4.0, value for value; the new
  options draw from a separate random stream. A regression test pins this.

## 0.4.0 (2026-11-05)

### Added
- `pycausalsim.futures.marketing`: marketing mix models as temporal causal models.
  - `marketing_worlds(n_worlds, weeks, hard, seed)`: many simulated brands at once,
    one per draw, from a declared graph. Seasonal demand drives both paid search
    spend and sales; online video peaks one to three weeks after it airs; with
    `hard=True`, unobserved product launches raise spend and demand together and
    social creative wears out. Each world carries its answer key (true ROAS and
    marginal ROAS from a 1% spend bump), plus `best_plan()` and `steady_sales()`
    for grading budget decisions.
  - `media_plan_scm(curves, current, weeks)`: play a budget plan forward week by
    week from posterior response curves (for example
    `papilon.MMM.posterior_curves()`), with carryover, the ramp from the current
    mix and joint posterior uncertainty. `sustained_max` declares a mechanism
    prior on each response curve, so spend beyond what the brand has sustained is
    flagged as extrapolation.
  - `compare_plans(scm, plans)`: quarter totals, gain against the first plan with
    a 90% interval, probability of beating it and extrapolation flags.

## 0.3.0 (2026-10-25)

### Added
- `TemporalSCM.empirical_path(name, paths, weights=None)`: resample whole
  historical episodes, such as past winters, keeping the correlation across
  steps. The pick is shared across scenarios, and weights let outside
  probabilities steer the draws.
- `parameter(..., "choice", values=[...], p=[...])` for finite priors.
- `Scenario.path(lever, {step: value})` for step-by-step lever values.
- `RunResult.window(node, start, end, agg)` for per-draw aggregates over a
  span, also available inside claim tests.
- `rolling_backtest(cases)` to pool backtests from many forecast origins,
  with per-origin results in `.cases`.
- `Context.step` and `TemporalSCM.steps` as step-size-neutral names for
  `year` and `years`. Monthly horizons with `YYYYMM` labels work throughout.
- `examples/henry_hub_winter.py`: Henry Hub natural gas winter forecast with
  bundled EIA data, scenarios, a hedging decision and a leave-one-winter-out
  backtest, plus a regression test.
- CI runs every example on every push.

### Fixed
- Uplift AUUC and Qini no longer crash on NumPy 2, which removed `np.trapz`.

### Changed
- Python 3.10 or later. Python 3.8 and 3.9 are past end of life and were never
  tested in CI.
- One version source: `pycausalsim.__version__`, read by `pyproject.toml`.
  `setup.py` is removed.
- The README now lists the discovery methods that exist: FCI is not
  implemented, and `"fges"` is an alias for GES.

## 0.2.0
- `pycausalsim.futures`: temporal SCMs, declared mechanism priors, structural
  interventions, scenarios, driver ranking, falsifiable claims, backtesting.

## 0.1.1
- Initial public release.
