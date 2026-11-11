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
Grade any marketing mix model against simulated brands with known returns.

An estimator is a function ``estimator(df, info) -> dict`` with

    "roas"  {channel: (estimate, p05, p95)}     the model's return on spend and 90% range
    "plan"  spend per week by channel, or None   the budget it recommends at the same total

``info`` gives it what a modeler would know: ``media``, ``target``, ``date_col``,
``controls``, ``current`` (historical weekly spend by channel) and ``total``.
``grade`` runs it on each world and scores the answers against the answer key;
``leaderboard`` does that for several estimators on the same worlds. Because the
contract is a plain function, any library can be entered: wrap its fit and its
optimizer in twenty lines and it is graded on the same brands as everything else.

Two estimators ship here: ``papilon_estimator`` (needs ``papilon``) and
``ols_estimator``, the regression-with-fixed-adstock baseline that a careful
model ought to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["grade", "leaderboard", "papilon_estimator", "ols_estimator"]


def grade(estimator, worlds, ids=None, bound_mult: float = 3.0) -> pd.DataFrame:
    """
    Run ``estimator`` on worlds ``ids`` (default: all) and score it.

    Returns one row per world and channel. ``attrs["summary"]`` holds, across
    worlds: ``coverage_90`` (share of 90% ranges that hold the true return; 0.9
    is calibrated), ``abs_error`` and ``bias`` on ROAS, ``plan_capture`` (true
    gain of the recommended plans as a share of the best possible gain, each
    channel free between 0 and ``bound_mult`` times its history) and
    ``plan_lost_money`` (share of worlds where the plan did worse than the
    current mix). Plan scores are NaN if the estimator returns no plan.
    """
    ch = list(worlds.channels)
    rows, gains = [], []
    for i in (range(worlds.n) if ids is None else ids):
        df, truth = worlds.frame(i), worlds.truth(i)["channels"]
        hist = df[ch].mean().to_numpy()
        info = {"media": ch, "target": worlds.target, "date_col": "date", "controls": list(worlds.controls),
                "current": hist.copy(), "total": float(hist.sum()), "world": int(i)}
        res = estimator(df, info)
        gain = best = np.nan
        if res.get("plan") is not None:
            x = np.asarray(res["plan"], dtype=float)
            if abs(x.sum() - hist.sum()) > 1e-4 * hist.sum() or np.any(x < -1e-9):
                raise ValueError("a plan must be non-negative and keep the total budget")
            base = worlds.steady_sales(i, hist)
            gain = worlds.steady_sales(i, x) - base
            best = worlds.steady_sales(i, worlds.best_plan(i, hist.sum(), [(0.0, bound_mult * h) for h in hist])) - base
            gains.append((gain, best))
        for c in ch:
            est, lo, hi = (float(v) for v in res["roas"][c])
            rows.append({"world": int(i), "channel": c, "true_roas": truth[c]["roas"], "roas": est,
                         "p05": lo, "p95": hi, "covered": lo <= truth[c]["roas"] <= hi,
                         "error": est - truth[c]["roas"], "plan_gain": gain, "best_gain": best})
    out = pd.DataFrame(rows)
    g = np.array(gains) if gains else np.full((1, 2), np.nan)
    out.attrs["summary"] = {
        "worlds": int(out.world.nunique()), "coverage_90": float(out.covered.mean()),
        "abs_error": float(out.error.abs().mean()), "bias": float(out.error.mean()),
        "plan_capture": float(g[:, 0].sum() / g[:, 1].sum()) if gains else np.nan,
        "plan_lost_money": float(np.mean(g[:, 0] < 0)) if gains else np.nan,
        "coverage_by_channel": out.groupby("channel", sort=False).covered.mean().to_dict(),
        "bias_by_channel": out.groupby("channel", sort=False).error.mean().to_dict(),
    }
    return out


def leaderboard(estimators: dict, worlds, ids=None, bound_mult: float = 3.0) -> pd.DataFrame:
    """Grade several estimators on the same worlds. One row each, best plan capture first."""
    keep = ["worlds", "coverage_90", "abs_error", "bias", "plan_capture", "plan_lost_money"]
    rows = {name: {k: grade(est, worlds, ids, bound_mult).attrs["summary"][k] for k in keep}
            for name, est in estimators.items()}
    return pd.DataFrame(rows).T.sort_values("plan_capture", ascending=False)


def papilon_estimator(carryover: dict | None = None, plan: dict | None = None, extra=None, **mmm_kwargs):
    """
    Papilon as an estimator. ``carryover`` is passed to ``MMM.compare_carryover``
    (None fits a single family), ``plan`` to ``optimize_budget``, the rest to
    ``papilon.MMM``. ``extra(df, info)`` may return more ``MMM`` keyword
    arguments per world, for example lift-test priors from :func:`lift_test`.
    """
    def estimator(df, info):
        import papilon as pp
        kw = dict(mmm_kwargs) | (extra(df, info) if extra else {})
        kw.setdefault("controls", info["controls"])
        kw.setdefault("seed", 11)
        m = pp.MMM(df, target=info["target"], media=info["media"], date_col=info["date_col"], **kw)
        if carryover:
            m.compare_carryover({k: [c for c in v if c in info["media"]] for k, v in carryover.items()})
        else:
            m.fit()
        roi = m.get_roi()
        x = m.optimize_budget(info["total"], **(plan or {}))["optimal_spend"].to_numpy()
        return {"roas": {c: (roi.loc[c, "roas_mean"], roi.loc[c, "roas_p05"], roi.loc[c, "roas_p95"])
                         for c in info["media"]}, "plan": x}
    return estimator


def ols_estimator(alpha: float = 0.5, l_max: int = 8, max_change: float = 0.5, harmonics: int = 2):
    """
    The baseline to beat: least squares on geometrically adstocked spend (fixed
    retention ``alpha``), with trend, yearly Fourier terms and controls. Returns
    are the coefficients, ranges are normal intervals, and the plan moves money
    from the lowest-return channels to the highest within ``max_change`` of
    current spend. No saturation, so it always wants to move as far as allowed.
    """
    def estimator(df, info):
        ch, n = info["media"], len(df)
        w = alpha ** np.arange(l_max); w /= w.sum()
        ad = np.column_stack([np.convolve(df[c].to_numpy(float), w)[:n] for c in ch])
        t = np.arange(n) / 52.18
        cols = [np.ones(n), t] + [f(2 * np.pi * k * t) for k in range(1, harmonics + 1) for f in (np.sin, np.cos)]
        cols += [df[c].to_numpy(float) for c in info["controls"]]
        X = np.column_stack([ad] + cols)
        y = df[info["target"]].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        cov = resid @ resid / max(n - X.shape[1], 1) * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))[: len(ch)]
        cur = info["current"]
        x = cur.copy()
        order = np.argsort(-beta[: len(ch)])
        lo_i, hi_i = len(ch) - 1, 0
        room_up, room_dn = cur * max_change, cur * max_change
        while hi_i < lo_i:  # move budget from the worst channel to the best until a cap binds
            a, b = order[hi_i], order[lo_i]
            mv = min(room_up[a], room_dn[b])
            x[a] += mv; x[b] -= mv; room_up[a] -= mv; room_dn[b] -= mv
            if room_up[a] <= 1e-12:
                hi_i += 1
            if room_dn[b] <= 1e-12:
                lo_i -= 1
        return {"roas": {c: (beta[j], beta[j] - 1.645 * se[j], beta[j] + 1.645 * se[j]) for j, c in enumerate(ch)},
                "plan": x}
    return estimator
