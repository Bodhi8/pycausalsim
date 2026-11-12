#   Copyright 2025 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Simulate a measurement loop on brands with known returns.

``simulate_loop`` runs ``papilon.Loop`` against one world from
:func:`marketing_worlds`, quarter after quarter: the loop plans, the world responds
to the spend the loop chose (carryover, saturation, noise and all), the
recommended lift test runs and comes back measured with the noise of a geo design,
and the loop refits. Because the world's truth is known at every step, each cycle
is graded: how far the estimated returns are from the truth, whether the ranges
hold it, and how much the plan made or lost against the true response curves.

Three policies can be compared on the same worlds, with the same noise:

    "static"   fit once on the history and keep that plan
    "refresh"  refit every cycle on all the data, no tests, no exploration
    "loop"     refresh, plus the recommended lift test and a small exploration pulse

``compare_loops`` runs them side by side. This is how to find out whether an
always-on model converges on the truth or locks in its own error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["simulate_loop", "compare_loops"]


def _run(worlds, Q):
    return worlds._sim(Q)


def simulate_loop(worlds, i: int, policy: str = "loop", history: int = 104, cycles: int = 6, horizon: int = 13,
                  explore: float = 0.10, test_weeks: int = 8, geo_share: float = 0.5, r2: float = 0.9,
                  model_kwargs: dict | None = None, plan_kwargs: dict | None = None,
                  loop_kwargs: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Run one policy on world ``i`` for ``cycles`` planning periods. One row per cycle.

    ``worlds`` must have at least ``history + cycles * horizon`` weeks. ``loop_kwargs`` go to
    ``papilon.Loop`` (for example ``payback_periods``). Columns:
    ``abs_error`` and ``coverage_90`` of the estimated returns against the truth over
    the fitted window, ``plan_gain`` (true steady-state gain per week of the plan over
    the historical mix), ``best_gain`` (the most any plan could gain under the same
    limits), ``realized`` (true sales over the cycle's weeks), and ``tested`` (the
    channel tested in the cycle, if any).
    """
    from papilon import Loop
    if policy not in ("static", "refresh", "loop"):
        raise ValueError('policy must be "static", "refresh" or "loop"')
    T = len(worlds.dates)
    if history + cycles * horizon > T:
        raise ValueError("the worlds are too short for this many cycles")
    ch = list(worlds.channels)
    Q = dict(worlds.params)
    for c in ch:
        Q[f"x_{c}"] = Q[f"x_{c}"].copy()
    run = _run(worlds, Q)
    frame = lambda r, upto: _frame(worlds, r, i, upto)
    hist_mean = frame(run, history)[ch].mean().to_numpy()
    lims = [(0.0, 3.0 * h) for h in hist_mean]
    best = worlds.steady_sales(i, worlds.best_plan(i, hist_mean.sum(), lims)) - worlds.steady_sales(i, hist_mean)
    lp = Loop(frame(run, history), worlds.target, ch, "date", controls=list(worlds.controls), horizon=horizon,
              budget=float(hist_mean.sum()),
              explore=explore if policy == "loop" else 0.0, recommend_tests=policy == "loop",
              test_weeks=test_weeks, test_share=geo_share, model_kwargs=model_kwargs, plan_kwargs=plan_kwargs,
              seed=seed, **(loop_kwargs or {}))
    rng = np.random.default_rng([seed, i, 11])
    rows, tests, static_schedule = [], [], None
    for k in range(cycles):
        t0 = history + k * horizon
        new = frame(run, t0).iloc[t0 - horizon:] if k else None
        if policy == "static" and k:
            e = lp.log[0]
        else:
            e = lp.cycle(new_data=new, tests=tests)
        tests = []
        m = e["model"]
        # grade the estimates against the truth over the window the model saw
        seen = slice(0, t0)
        roi = m.get_roi()
        true = {c: run[f"effect_{c}"][seen, i].sum() / max(run[f"spend_{c}"][seen, i].sum(), 1e-12) for c in ch}
        err = np.mean([abs(roi.loc[c, "roas_mean"] - true[c]) for c in ch])
        cov = np.mean([roi.loc[c, "roas_p05"] <= true[c] <= roi.loc[c, "roas_p95"] for c in ch])
        steady = e["plan"]["optimal_spend"].to_numpy()
        gain = worlds.steady_sales(i, steady) - worlds.steady_sales(i, hist_mean)
        # apply the schedule (and the experiment, if any) to the world's future weeks
        sched = e["schedule"][ch].to_numpy()
        rows_ = slice(t0, t0 + horizon)
        for j, c in enumerate(ch):
            Q[f"x_{c}"][i, rows_] = sched[:, j]
        tested = None
        if policy == "loop" and e["experiment"] is not None:
            c = e["experiment"]["channel"]
            w = slice(t0, t0 + min(test_weeks, horizon))
            planned = _run(worlds, Q)
            Q[f"x_{c}"][i, w] *= (1 - geo_share)                       # treated regions go dark
            after = _run(worlds, Q)
            span = slice(t0, min(T, t0 + min(test_weeks, horizon) + worlds.l_max - 1))
            lift = float((planned["sales"][span, i] - after["sales"][span, i]).sum())
            sigma = float(np.std(worlds.params["noise"][i]))
            n_span = span.stop - span.start
            se = sigma * np.sqrt((1 - r2) * n_span * geo_share / (1 - geo_share))
            dates = worlds.dates
            tests = [{"channel": c, "start": dates[w.start], "end": dates[w.stop - 1],
                      "lift": lift + rng.normal(0, se), "se": se, "factor": 1 / (1 - geo_share),
                      "readout": worlds.l_max - 1}]
            tested = c
        run = _run(worlds, Q)
        rows.append({"world": i, "policy": policy, "cycle": k, "weeks_seen": t0, "abs_error": float(err),
                     "coverage_90": float(cov), "plan_gain": float(gain), "best_gain": float(best),
                     "realized": float(run["sales"][rows_, i].sum()), "tested": tested,
                     "explore": " / ".join(e["explore"]), "gate": e["gate"]})
    return pd.DataFrame(rows)


def _frame(worlds, run, i, upto):
    r = {"date": worlds.dates[:upto]}
    for c in worlds.channels:
        r[c] = np.round(run[f"spend_{c}"][:upto, i], 2)
    for name, dec in worlds.controls.items():
        v = run[name][:upto, i]
        r[name] = v.astype(int) if dec is None else np.round(v, dec)
    r[worlds.target] = np.round(run["sales"][:upto, i], 2)
    return pd.DataFrame(r)


def compare_loops(worlds, ids, policies=("static", "refresh", "loop"), **kwargs) -> pd.DataFrame:
    """Run each policy on each world. Returns every cycle row; ``attrs["summary"]`` has, per
    policy and cycle, the mean ROAS error, coverage and share of the best gain captured."""
    out = pd.concat([simulate_loop(worlds, i, p, **kwargs) for i in ids for p in policies], ignore_index=True)
    g = out.groupby(["policy", "cycle"])
    out.attrs["summary"] = pd.DataFrame({"abs_error": g.abs_error.mean(), "coverage_90": g.coverage_90.mean(),
                                         "capture": g.plan_gain.sum() / g.best_gain.sum(),
                                         "realized": g.realized.mean()}).reset_index()
    return out
