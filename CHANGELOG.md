# Changelog

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
