# Changelog

## 0.5.0 (unreleased)

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
