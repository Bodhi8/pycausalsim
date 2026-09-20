"""
Backtesting.

A futures engine unable to retrodict the preceding years has no standing to
forecast the next fifteen, and its backtest score belongs next to every
forecast it produces.

Backtesting a specified model is not the same as backtesting a fitted one.
There are no coefficients to refit, so what is being scored is whether the
declared mechanism reproduces observed anchors, and whether the stated
intervals are calibrated rather than merely wide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnchorScore:
    anchor: str
    year: int
    observed: float
    predicted: float
    error: float
    pct_error: float
    inside_90: bool
    z: float

    def __repr__(self):
        flag = "in " if self.inside_90 else "OUT"
        return (f"  {self.anchor:22s} {self.year}  obs {self.observed:9.3f}  "
                f"pred {self.predicted:9.3f}  {self.pct_error:+7.1%}  [{flag}]")


class BacktestResult:
    def __init__(self, scores, window):
        self.scores = scores
        self.window = window

    @property
    def coverage(self):
        """Share of anchors inside the stated 90 percent interval."""
        if not self.scores:
            return float("nan")
        return float(np.mean([s.inside_90 for s in self.scores]))

    @property
    def mape(self):
        if not self.scores:
            return float("nan")
        return float(np.mean([abs(s.pct_error) for s in self.scores]))

    def calibration_note(self):
        c = self.coverage
        if np.isnan(c):
            return "no anchors scored"
        if len(self.scores) < 5:
            return (f"Only {len(self.scores)} anchors scored. Too few to say "
                    f"anything about calibration; add more observations.")
        if c > 0.97:
            return ("Intervals are too wide. Coverage above the nominal 90 "
                    "percent means the priors are not doing any work.")
        if c < 0.75:
            return ("Intervals are too narrow. The model is more confident "
                    "than its record justifies.")
        return "Intervals are roughly calibrated against this window."

    def summary(self):
        lines = [f"Backtest {self.window[0]} to {self.window[1]}",
                 f"  {'anchor':22s} {'year':>4s}  {'observed':>13s}  "
                 f"{'predicted':>14s}  {'error':>7s}"]
        lines += [repr(s) for s in self.scores]
        lines += ["",
                  f"  MAPE:     {self.mape:.1%}",
                  f"  Coverage: {self.coverage:.0%} of anchors inside the 90% interval",
                  f"  {self.calibration_note()}"]
        return "\n".join(lines)

    def to_dict(self):
        return dict(window=list(self.window), mape=self.mape,
                    coverage=self.coverage,
                    anchors=[s.__dict__ for s in self.scores])


def backtest(scm, observations: dict, scenario=None, draws=None,
             params=None):
    """
    observations: {anchor_name: {year: observed_value}}

    Every anchor must be a node in the graph. Years outside the horizon are
    skipped rather than silently dropped from the denominator, and the count
    of skipped anchors is reported.
    """
    run = scm.run(scenario, draws=draws, params=params)
    scores, skipped = [], 0
    for anchor, series in observations.items():
        if anchor not in run.series:
            raise KeyError(f"{anchor!r} is not a node in this model")
        for year, obs in series.items():
            year = int(year)
            if year not in scm.years:
                skipped += 1
                continue
            v = run.at(anchor, year)
            pred = float(v.mean())
            lo, hi = np.percentile(v, 5), np.percentile(v, 95)
            sd = float(v.std()) or 1e-9
            scores.append(AnchorScore(
                anchor=anchor, year=year, observed=float(obs),
                predicted=pred, error=pred - obs,
                pct_error=(pred - obs) / obs if obs else float("nan"),
                inside_90=bool(lo <= obs <= hi),
                z=(obs - pred) / sd))
    if skipped:
        print(f"backtest: {skipped} observation(s) outside the horizon, skipped")
    window = (scm.years[0], scm.years[-1])
    return BacktestResult(scores, window)
