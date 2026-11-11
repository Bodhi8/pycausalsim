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
Marketing worlds and media plans as temporal causal models.

Two jobs, both built on :class:`TemporalSCM`:

``marketing_worlds``
    Simulate many brands at once, one per draw, from a declared causal graph:
    seasonal demand drives both paid search spend and sales, online video peaks
    weeks after it airs, and (with ``hard=True``) unobserved product launches
    raise spend and demand together while social creative wears out. Every
    world comes with its answer key, so a marketing mix model can be graded on
    hundreds of brands instead of one.

``media_plan_scm`` / ``compare_plans``
    Take posterior response curves from a fitted marketing mix model (for
    example ``papilon.MMM.posterior_curves()``) and play a budget plan forward
    week by week, with carryover, the ramp from the current mix and the full
    posterior uncertainty, instead of assuming the new mix holds from day one.

The graph, per brand and week t::

    season ──► search_spend ──► adstock_search ──► effect_search ─┐
       │        launch ──► search_spend, social_spend, baseline    │
       └──────────────────────────────────────► baseline ──────────┼──► sales
    social_spend, video_spend, display_spend ──► adstock ──► effect┘
    price_index, promo ──► baseline

``launch`` is not exported in the data a model sees: it is the unobserved
confounder.

Since 0.5 the worlds can also be made harder in the two ways the standard ones
are kind: ``response="s_curve"`` gives every channel an S-shaped response (the
functional form a concave model gets wrong), and ``competitor=True`` adds
unobserved competitor campaigns that depress demand while the brand answers
with display. ``lift_test`` simulates a geo holdout experiment on any world, and
``worlds_from_graph`` builds worlds from a declared knowledge graph instead of
the fixed one above.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .scenario import Scenario
from .temporal import TemporalSCM, _Params

__all__ = ["CHANNELS", "MarketingWorlds", "marketing_worlds", "media_plan_scm", "compare_plans",
           "lift_test", "worlds_from_graph"]

CHANNELS = ["paid_search", "paid_social", "online_video", "display"]
_PERIOD = 52.18


def _kernel(alpha, peak, l_max):
    """Carryover weights over lags 0..l_max-1, per draw, normalized to sum to one.

    Geometric where peak is NaN, delayed (alpha ** (l - peak) ** 2) otherwise:
    the same kernels as Jin et al. (2017) and papilon.mmm.transforms.
    """
    lags = np.arange(l_max)[None, :]
    a = np.asarray(alpha, dtype=float)[:, None]
    pk = np.asarray(peak, dtype=float)[:, None]
    geo = a ** lags
    dly = a ** ((lags - np.nan_to_num(pk)) ** 2)
    w = np.where(np.isnan(pk), geo, dly)
    return w / w.sum(axis=1, keepdims=True)


@dataclass
class MarketingWorlds:
    """Output of :func:`marketing_worlds`: one simulated brand per draw."""

    run: object               # RunResult or dict, series of shape (weeks, worlds)
    params: dict
    dates: pd.DatetimeIndex
    hard: bool
    channels: list = field(default_factory=lambda: list(CHANNELS))
    controls: dict = field(default_factory=lambda: {"price_index": 4, "promo": None})  # column -> decimals (None = int)
    l_max: int = 8
    target: str = "sales"

    @property
    def n(self):
        return self.run["sales"].shape[1]

    def frame(self, i: int) -> pd.DataFrame:
        """The data a modeler would see for world i: dates, spend, controls, sales."""
        r = self.run
        cols = {"date": self.dates}
        for c in self.channels:
            cols[c] = np.round(r[f"spend_{c}"][:, i], 2)
        for name, dec in self.controls.items():
            cols[name] = r[name][:, i].astype(int) if dec is None else np.round(r[name][:, i], dec)
        cols[self.target] = np.round(r["sales"][:, i], 2)
        return pd.DataFrame(cols)

    def _shape(self, c, i):
        s = self.params.get(f"shape_{c}")
        return 1.0 if s is None else float(s[i])

    def truth(self, i: int) -> dict:
        """Answer key for world i: generating parameters, ROAS and marginal ROAS.

        ``roas`` is the channel's own (direct) effect per dollar. In worlds built
        from a graph with ``mediates`` relations, ``total_roas`` also counts sales
        that reach the outcome through the channels it feeds.
        """
        p = self.params
        out = {"channels": {}}
        if "launch" in p:
            out["launch_weeks"] = int((self.run["launch"][:, i] > 0).sum())
        for c in self.channels:
            out["channels"][c] = {
                "beta_final": float(p[f"beta_{c}"][i] * p[f"wear_{c}"][i]),
                "half_sat": float(p[f"hs_{c}"][i]),
                "alpha": float(p[f"alpha_{c}"][i]),
                "peak": None if np.isnan(p[f"peak_{c}"][i]) else float(p[f"peak_{c}"][i]),
                "shape": self._shape(c, i),
                "roas": float(self._roas[c][i]),
                "marginal_roas": float(self._mroas[c][i]),
            }
            if getattr(self, "_total_roas", None):
                out["channels"][c]["total_roas"] = float(self._total_roas[c][i])
        return out

    def steady_sales(self, i: int, spend) -> float:
        """True steady-state weekly media sales at constant spend, with end-of-window effects."""
        x = np.asarray(spend, dtype=float)
        t = self.truth(i)["channels"]
        return float(sum(t[c]["beta_final"] * x[j] ** t[c]["shape"]
                         / (x[j] ** t[c]["shape"] + t[c]["half_sat"] ** t[c]["shape"])
                         for j, c in enumerate(self.channels)))

    def best_plan(self, i: int, total: float, bounds) -> np.ndarray:
        """Spend that maximizes true steady-state sales under the same bounds."""
        from scipy import optimize
        k = len(self.channels)
        lo, hi = np.array(bounds, dtype=float).T
        starts = [np.clip(np.full(k, total / k), lo, hi)]
        if any(self._shape(c, i) != 1.0 for c in self.channels):
            # S-curves are not concave: also start from each channel pushed to its cap
            for j in range(k):
                s = lo.copy(); s[j] = min(hi[j], total - (lo.sum() - lo[j]))
                rest = total - s.sum()
                room = hi - s; room[j] = 0
                if rest > 0 and room.sum() > 0:
                    s = s + room / room.sum() * min(rest, room.sum())
                starts.append(s)
        best = None
        for x0 in starts:
            r = optimize.minimize(lambda x: -self.steady_sales(i, x), x0, method="SLSQP", bounds=bounds,
                                  constraints=[{"type": "eq", "fun": lambda x: x.sum() - total}],
                                  options={"ftol": 1e-12, "maxiter": 500})
            if best is None or (r.fun < best.fun and abs(r.x.sum() - total) < 1e-6 * total):
                best = r
        return best.x


def marketing_worlds(n_worlds: int = 100, weeks: int = 156, hard: bool = False,
                     seed: int = 0, l_max: int = 8, response: str = "hill",
                     competitor: bool = False) -> MarketingWorlds:
    """
    Simulate ``n_worlds`` brands with four digital channels and known returns.

    Parameters are drawn per brand: true ROAS between 0.5 and 2.2 per channel,
    carryover, half-saturation near typical spend, seasonal confounding of paid
    search, video flights that peak one to three weeks after airing.

    hard=True adds two things real data has and the standard worlds don't:
    two to four unobserved product launches that lift search and social spend
    and baseline demand together (confounding the model cannot control for),
    and social creative that loses 20 to 40% of its effect over the window.

    response="s_curve" gives each channel a Hill curve with shape between 1.4
    and 2.4, so the first dollars do little and returns peak near typical spend:
    the functional form a concave model gets wrong. competitor=True adds two to
    four unobserved competitor campaigns that depress demand for several weeks
    while the brand raises display spend in answer, a confounder with the
    opposite sign to launches. Both are drawn from a separate random stream, so
    worlds made with the defaults are identical to 0.4.
    """
    if response not in ("hill", "s_curve"):
        raise ValueError('response must be "hill" or "s_curve"')
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng([int(seed), 5])   # 0.5 options; leaves the 0.4 stream untouched
    n, T = int(n_worlds), int(weeks)
    t = np.arange(T)
    season = np.sin(2 * np.pi * t / _PERIOD) + 0.5 * np.sin(4 * np.pi * t / _PERIOD)

    P = {}
    shape = {c: (rng2.uniform(1.4, 2.4, n) if response == "s_curve" else np.ones(n)) for c in CHANNELS}
    comp = np.zeros((n, T))
    if competitor:
        for i in range(n):
            for s in rng2.choice(np.arange(8, T - 10), rng2.integers(2, 5), replace=False):
                comp[i, s:s + rng2.integers(4, 10)] = 1.0
    mean_spend = {"paid_search": rng.uniform(20, 40, n), "paid_social": rng.uniform(30, 60, n),
                  "online_video": rng.uniform(30, 60, n), "display": rng.uniform(12, 30, n)}
    alpha_rng = {"paid_search": (0.10, 0.40), "paid_social": (0.30, 0.60),
                 "online_video": (0.40, 0.70), "display": (0.20, 0.50)}
    for c in CHANNELS:
        m = mean_spend[c]
        P[f"mean_{c}"] = m
        P[f"hs_{c}"] = m * rng.uniform(0.8, 1.6, n)
        roas = rng.uniform(0.5, 2.2, n)
        P[f"shape_{c}"] = shape[c]
        # steady ROAS at mean spend ~ roas (for shape 1 this is roas * (m + hs), as in 0.4)
        P[f"beta_{c}"] = (roas * (m + P[f"hs_{c}"]) if response == "hill"
                          else roas * m * (m ** shape[c] + P[f"hs_{c}"] ** shape[c]) / m ** shape[c])
        P[f"alpha_{c}"] = rng.uniform(*alpha_rng[c], n)
        P[f"peak_{c}"] = rng.uniform(1.0, 3.0, n) if c == "online_video" else np.full(n, np.nan)
        P[f"wear_{c}"] = np.ones(n)
    if hard:
        P["wear_paid_social"] = 1 - rng.uniform(0.2, 0.4, n)

    # exogenous paths, per brand and week (drawn here so reruns are exact)
    P["season"] = np.tile(season, (n, 1))
    conf = rng.uniform(0.15, 0.35, n)[:, None]
    launch = np.zeros((n, T))
    if hard:
        for i in range(n):
            for s in rng.choice(np.arange(8, T - 8), rng.integers(2, 5), replace=False):
                launch[i, s:s + rng.integers(4, 9)] = 1.0
    P["launch"] = launch
    ms = {c: mean_spend[c][:, None] for c in CHANNELS}
    P["x_paid_search"] = np.clip(ms["paid_search"] * (1 + conf * season + 0.4 * launch
                                                     + rng.normal(0, 0.2, (n, T))), 1, None)
    P["x_paid_social"] = rng.gamma(4.0, 1.0, (n, T)) / 4.0 * ms["paid_social"] * (1 + 0.3 * launch)
    period = rng.choice([8, 10, 13], n)[:, None]
    on = (t[None, :] % period) < np.round(0.4 * period)
    P["x_online_video"] = np.where(on, rng.uniform(1.5, 3.5, (n, T)), rng.uniform(0, 0.2, (n, T))) * ms["online_video"] / 1.0
    P["x_online_video"] *= ms["online_video"] / P["x_online_video"].mean(axis=1, keepdims=True)
    P["x_display"] = np.where(rng.random((n, T)) < 0.8, rng.uniform(0.4, 1.6, (n, T)), 0.0) * ms["display"]
    if competitor:
        P["x_display"] = np.maximum(P["x_display"], 0.6 * ms["display"] * comp) * (1 + 0.5 * comp)
    P["competitor"] = comp
    P["competitor_hit"] = rng2.uniform(40, 90, n)
    P["price_index"] = 1 + 0.04 * np.sin(2 * np.pi * t / 26)[None, :] + rng.normal(0, 0.015, (n, T))
    P["promo"] = (rng.random((n, T)) < 0.12).astype(float)
    P["noise"] = rng.normal(0, 1, (n, T)) * rng.uniform(25, 35, n)[:, None]
    P["intercept"] = rng.uniform(500, 700, n)
    P["season_amp"] = rng.uniform(60, 120, n)
    P["launch_lift"] = rng.uniform(40, 90, n)
    P["wear_t"] = np.tile(t / max(T - 1, 1), (n, 1))

    run = _simulate(P, n, T, l_max)
    dates = pd.date_range("2023-01-02", periods=T, freq="W-MON")
    worlds = MarketingWorlds(run=run, params=P, dates=dates, hard=hard, l_max=l_max)
    worlds._sim = lambda Q: _simulate(Q, n, T, l_max)
    _answer_key(worlds, P, worlds._sim)
    return worlds


def _build(n, T, l_max, P):
    scm = TemporalSCM(horizon=range(T), step="week", draws=n)
    zero = [0.0] * (l_max - 1)
    w = {c: _kernel(P[f"alpha_{c}"], P[f"peak_{c}"], l_max) for c in CHANNELS}
    scm.exogenous("season", lambda c: c.p.season[:, c.t])
    scm.exogenous("launch", lambda c: c.p.launch[:, c.t])
    scm.exogenous("competitor", lambda c: c.p.competitor[:, c.t])
    scm.exogenous("price_index", lambda c: c.p.price_index[:, c.t])
    scm.exogenous("promo", lambda c: c.p.promo[:, c.t])
    for ch in CHANNELS:
        parents = (["season", "launch"] if ch in ("paid_search", "paid_social")
                   else ["competitor"] if ch == "display" else [])
        scm.add_equation(f"spend_{ch}", lambda c, ch=ch: c.p[f"x_{ch}"][:, c.t], parents=parents,
                         pre_history=zero)
        lag_parents = [f"spend_{ch}"] + [f"spend_{ch}:lag({k})" for k in range(1, l_max)]

        def adstock(c, ch=ch, wk=w[ch]):
            return sum(wk[:, k] * (c[f"spend_{ch}"] if k == 0 else c.lag(f"spend_{ch}", k))
                       for k in range(l_max))
        scm.add_equation(f"adstock_{ch}", adstock, parents=lag_parents)

        def effect(c, ch=ch):
            beta = c.p[f"beta_{ch}"] * (1 - (1 - c.p[f"wear_{ch}"]) * c.p.wear_t[:, c.t])
            a, s = c[f"adstock_{ch}"], c.p[f"shape_{ch}"]
            if np.all(s == 1.0):
                return beta * a / (a + c.p[f"hs_{ch}"])
            return beta * a ** s / (a ** s + c.p[f"hs_{ch}"] ** s)
        scm.add_equation(f"effect_{ch}", effect, parents=[f"adstock_{ch}"])
    scm.add_equation("baseline", lambda c: (c.p.intercept + 40.0 * c.t / _PERIOD
                                            + c.p.season_amp * c["season"]
                                            - 350.0 * (c["price_index"] - 1) + 120.0 * c["promo"]
                                            + c.p.launch_lift * c["launch"]
                                            - c.p.competitor_hit * c["competitor"]),
                     parents=["season", "launch", "competitor", "price_index", "promo"])
    scm.add_equation("sales", lambda c: c["baseline"] + sum(c[f"effect_{ch}"] for ch in CHANNELS)
                     + c.p.noise[:, c.t], parents=["baseline"] + [f"effect_{ch}" for ch in CHANNELS])
    return scm


def _simulate(P, n, T, l_max):
    return _build(n, T, l_max, P).run(params=_Params(P))


def _answer_key(worlds, P, sim):
    """True ROAS over the window and marginal ROAS (1% more spend every week)."""
    run = worlds.run
    worlds._roas, worlds._mroas = {}, {}
    for c in worlds.channels:
        spend = run[f"spend_{c}"].sum(axis=0)
        eff = run[f"effect_{c}"].sum(axis=0)
        worlds._roas[c] = eff / spend
        bumped = dict(P)
        bumped[f"x_{c}"] = P[f"x_{c}"] * 1.01
        eff_b = sim(bumped)[f"effect_{c}"].sum(axis=0)
        worlds._mroas[c] = (eff_b - eff) / (0.01 * spend)


def lift_test(worlds: MarketingWorlds, i: int, channel: str, start: int | None = None, weeks: int = 8,
              change: float = -1.0, geo_share: float = 0.5, r2: float = 0.9, seed: int = 0) -> dict:
    """
    Simulate a geo lift test on world ``i`` and return what it would measure.

    Spend on ``channel`` is changed by ``change`` (-1.0 is a full holdout, +0.5 is
    50% more) for ``weeks`` weeks from ``start`` (default: the last ``weeks``
    weeks that leave room for carryover). The world is replayed through its
    causal model with every other draw held fixed, so the true incremental sales,
    carryover included, are known exactly.

    The measurement adds the noise a matched-market design would have: treated
    geos carry ``geo_share`` of the market, the rest act as control, and weekly
    sales noise is split between them. ``r2`` is the share of that noise the
    design's pre-period model removes (0.9 is typical of a synthetic-control
    design with a long pre-period; 0 is a raw difference between regions). That
    gives a standard error on the return, so a short test on a small channel
    comes back honestly vague.

    Returns ``measured`` and ``se`` (ready for ``roi_priors={channel: (measured,
    se)}``), ``true`` (the incremental return of this intervention) and the
    design. A holdout measures the average return of the spend removed, which is
    what a ROAS prior describes; a small increase measures the marginal return,
    which is lower on a saturating curve.
    """
    if not 0 < geo_share < 1:
        raise ValueError("geo_share must be between 0 and 1")
    P, T = worlds.params, len(worlds.dates)
    start = T - weeks - worlds.l_max if start is None else int(start)
    if start < 0 or start + weeks > T:
        raise ValueError("test window is outside the data")
    Q = dict(P)
    x = P[f"x_{channel}"].copy()
    x[i, start:start + weeks] *= (1 + change)
    Q[f"x_{channel}"] = x
    alt = worlds._sim(Q)
    d_sales = float((alt["sales"][:, i] - worlds.run["sales"][:, i]).sum())
    d_spend = float((alt[f"spend_{channel}"][:, i] - worlds.run[f"spend_{channel}"][:, i]).sum())
    if abs(d_spend) < 1e-9:
        raise ValueError("the channel has no spend in the test window")
    true = d_sales / d_spend
    sigma = float(np.std(P["noise"][i]))
    span = weeks + worlds.l_max - 1                       # read-out window includes carryover
    se = sigma * np.sqrt((1 - r2) * span * geo_share / (1 - geo_share)) / (geo_share * abs(d_spend))
    measured = true + np.random.default_rng([int(seed), int(i)]).normal(0, se)
    return {"channel": channel, "measured": float(measured), "se": float(se), "true": float(true),
            "start": start, "weeks": weeks, "change": change, "geo_share": geo_share, "r2": r2,
            "spend_changed": d_spend}


# ---------------------------------------------------------------- media plans
def media_plan_scm(curves: dict, current, weeks: int = 13, draws: int | None = None,
                   seed: int = 0, sustained_max=None) -> TemporalSCM:
    """
    A TemporalSCM that plays a media plan forward with posterior uncertainty.

    ``curves`` is the dict from ``papilon.MMM.posterior_curves()``: arrays of
    shape (draws, channels) for beta, half_sat, alpha and peak, plus media
    names. Each Monte Carlo draw uses one joint posterior draw, so correlation
    between channels survives. Spend before week 0 is ``current``, so the plan
    ramps in through carryover. Levers are named ``spend_<channel>``.

    ``sustained_max`` (by channel) declares a mechanism prior on each response
    curve: weeks where adstocked spend exceeds what the brand has sustained are
    recorded as extrapolation, never silent.
    """
    media = list(curves["media"])
    l_max = int(curves.get("l_max", 8))
    nd = curves["beta"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(nd, draws or nd, replace=(draws or nd) > nd)
    n = len(idx)
    cur = np.asarray(current, dtype=float)
    scm = TemporalSCM(horizon=range(weeks), step="week", draws=n, seed=seed)
    P = {}
    for j, c in enumerate(media):
        P[f"beta_{c}"] = curves["beta"][idx, j]
        P[f"hs_{c}"] = curves["half_sat"][idx, j]
        P[f"shape_{c}"] = curves["shape"][idx, j] if "shape" in curves else np.ones(n)
        w = _kernel(curves["alpha"][idx, j], curves["peak"][idx, j], l_max)
        scm.lever(f"spend_{c}", default=float(cur[j]))
        scm.add_equation(f"x_{c}", lambda c_, c=c: np.full(n, c_.lever(f"spend_{c}")),
                         pre_history=[float(cur[j])] * (l_max - 1))

        def adstock(c_, c=c, wk=w):
            return sum(wk[:, k] * (c_[f"x_{c}"] if k == 0 else c_.lag(f"x_{c}", k)) for k in range(l_max))
        scm.add_equation(f"adstock_{c}", adstock,
                         parents=[f"x_{c}"] + [f"x_{c}:lag({k})" for k in range(1, l_max)])
        scm.add_equation(f"effect_{c}", lambda c_, c=c: c_.p[f"beta_{c}"] * c_[f"adstock_{c}"] ** c_.p[f"shape_{c}"]
                         / (c_[f"adstock_{c}"] ** c_.p[f"shape_{c}"] + c_.p[f"hs_{c}"] ** c_.p[f"shape_{c}"]),
                         parents=[f"adstock_{c}"])
        if sustained_max is not None:
            scm.mechanism(f"effect_{c} ~ adstock_{c}", form="saturating",
                          valid_range=(0.0, float(np.asarray(sustained_max)[j])), outside="flag",
                          note="response curve beyond spend the brand has sustained")
    scm.add_equation("media_sales", lambda c_: sum(c_[f"effect_{c}"] for c in media),
                     parents=[f"effect_{c}" for c in media])
    scm._plan_params = _Params(P)
    scm._media = media
    return scm


def compare_plans(scm: TemporalSCM, plans: dict) -> pd.DataFrame:
    """
    Run named plans (dict name -> spend by channel, in the scm's media order)
    through the same posterior draws and return quarter totals.

    Columns: expected gain in media-driven sales over the horizon versus the
    first plan, its 90% interval, probability of beating the first plan and
    the share of weeks flagged as extrapolation.
    """
    names = list(plans)
    runs = {}
    for name in names:
        sc = Scenario(name)
        for j, c in enumerate(scm._media):
            sc.set(f"spend_{c}", float(np.asarray(plans[name], dtype=float)[j]))
        runs[name] = scm.run(sc, params=scm._plan_params)
    base = runs[names[0]]["media_sales"].sum(axis=0)
    rows = []
    for name in names:
        tot = runs[name]["media_sales"].sum(axis=0)
        gain = tot - base
        flags = runs[name].support.by_edge() if hasattr(runs[name].support, "by_edge") else {}
        rows.append({"plan": name, "media_sales": float(tot.mean()), "gain_vs_first": float(gain.mean()),
                     "gain_p05": float(np.quantile(gain, 0.05)), "gain_p95": float(np.quantile(gain, 0.95)),
                     "prob_better": float(np.mean(gain > 0)) if name != names[0] else np.nan,
                     "extrapolated_edges": len(flags)})
    return pd.DataFrame(rows).set_index("plan")


# ------------------------------------------------------- worlds from a graph
def _graph_sim(G, P, T, l_max):
    """Vectorized structural equations for worlds_from_graph. Arrays are (worlds, weeks) inside, (weeks, worlds) out."""
    run, ad = {}, {}
    for c in G["order"]:
        x = P[f"x_{c}"]
        for src, key in G["feeds"].get(c, []):
            x = x * (1 + P[key][:, None] * ad[src] / P[f"mean_{src}"][:, None])
        w = _kernel(P[f"alpha_{c}"], P[f"peak_{c}"], l_max)
        a = np.zeros_like(x)
        for k in range(l_max):
            a[:, k:] += w[:, k:k + 1] * x[:, :T - k]
        ad[c] = a
        s, hs = P[f"shape_{c}"][:, None], P[f"hs_{c}"][:, None]
        run[f"spend_{c}"] = x.T
        run[f"effect_{c}"] = (P[f"beta_{c}"][:, None] * a ** s / (a ** s + hs ** s)).T
    run["sales"] = P["baseline"].T + sum(run[f"effect_{c}"] for c in G["order"]) + P["noise"].T
    for d, col in G["exported"].items():
        run[col] = P[f"drv_{d}"].T
    return run


def worlds_from_graph(graph, n_worlds: int = 100, weeks: int = 156, seed: int = 0, l_max: int = 8,
                      response: str = "hill", hide: list | None = None) -> MarketingWorlds:
    """
    Simulate brands whose structure is a declared knowledge graph.

    ``graph`` is a ``papilon.MarketingGraph`` or its ``to_dict()``: channels with
    a funnel stage, drivers and events with the channels and outcome they affect,
    and ``mediates`` relations between channels. Each world draws its own spend
    levels, returns (true ROAS 0.5 to 2.2), carryover and saturation, then:

    * a driver with a ``sets_spend`` edge moves that channel's spend, and with a
      ``drives`` edge moves baseline demand: both together make it a confounder;
    * a driver with a data column is exported in ``frame()``; one without (or
      named in ``hide``) is left out, so the model never sees it;
    * ``A mediates B`` makes B's spend rise with A's recent advertising, so part
      of A's return arrives through B. ``truth()`` reports A's direct ``roas`` and
      its ``total_roas`` from removing A and replaying the world;
    * upper-funnel channels are flighted and peak one to three weeks after they
      air; mid-funnel spend is bursty; lower-funnel spend is steady.

    The result has the same interface as :func:`marketing_worlds`, so the same
    grading code runs on it. ``steady_sales`` and ``best_plan`` use each
    channel's direct curve.
    """
    spec = graph.to_dict() if hasattr(graph, "to_dict") else graph
    rng = np.random.default_rng(seed)
    n, T = int(n_worlds), int(weeks)
    t = np.arange(T)
    nodes = {nd["id"]: nd for nd in spec["nodes"]}
    edges = [e for e in spec["edges"] if e.get("status", "confirmed") == "confirmed"]
    outcome = spec["outcome"]
    chans = [k for k, nd in nodes.items() if nd["type"] == "Channel"]
    col = {c: nodes[c].get("column") or c for c in chans}
    drivers = [k for k, nd in nodes.items() if nd["type"] == "Driver"]
    hide = set(hide or [])
    if not chans:
        raise ValueError("the graph has no channels")

    med = [(e["source"], e["target"]) for e in edges if e["relation"] == "mediates"]
    order, left = [], list(chans)
    while left:  # sources before the channels they feed
        ready = [c for c in left if all(a not in left for a, b in med if b == c)]
        if not ready:
            raise ValueError("mediates relations form a cycle")
        order += ready; left = [c for c in left if c not in ready]

    P, season = {}, np.sin(2 * np.pi * t / _PERIOD) + 0.5 * np.sin(4 * np.pi * t / _PERIOD)
    for d in drivers:
        nd = nodes[d]
        if nd.get("modeled") or "season" in d.lower():
            series = np.tile(season, (n, 1))
        elif nd.get("kind") == "event":
            series = np.zeros((n, T))
            for i in range(n):
                for s in rng.choice(np.arange(8, T - 8), rng.integers(2, 5), replace=False):
                    series[i, s:s + rng.integers(4, 9)] = 1.0
        else:
            e = rng.normal(0, 1, (n, T))
            for k in range(1, T):
                e[:, k] = 0.8 * e[:, k - 1] + 0.6 * e[:, k]
            series = (e - e.mean(axis=1, keepdims=True)) / e.std(axis=1, keepdims=True)
        P[f"drv_{d}"] = series

    alpha_rng = {"lower": (0.10, 0.40), "mid": (0.30, 0.60), "upper": (0.40, 0.70)}
    for c in order:
        stage = nodes[c].get("stage", "lower")
        m = rng.uniform(15, 60, n)
        P[f"mean_{c}"] = m
        P[f"hs_{c}"] = m * rng.uniform(0.8, 1.6, n)
        roas = rng.uniform(0.5, 2.2, n)
        sh = rng.uniform(1.4, 2.4, n) if response == "s_curve" else np.ones(n)
        P[f"shape_{c}"] = sh
        P[f"beta_{c}"] = roas * m * (m ** sh + P[f"hs_{c}"] ** sh) / m ** sh
        P[f"alpha_{c}"] = rng.uniform(*alpha_rng[stage], n)
        P[f"peak_{c}"] = rng.uniform(1.0, 3.0, n) if stage == "upper" else np.full(n, np.nan)
        P[f"wear_{c}"] = np.ones(n)
        if stage == "upper":
            period = rng.choice([8, 10, 13], n)[:, None]
            x = np.where((t[None, :] % period) < np.round(0.4 * period), rng.uniform(1.5, 3.5, (n, T)),
                         rng.uniform(0, 0.2, (n, T)))
        elif stage == "mid":
            x = rng.gamma(4.0, 1.0, (n, T)) / 4.0
        else:
            x = 1 + rng.normal(0, 0.2, (n, T))
        mult = np.ones((n, T))
        for e in edges:
            if e["relation"] == "sets_spend" and e["target"] == c:
                mult += rng.uniform(0.15, 0.40, n)[:, None] * P[f"drv_{e['source']}"]
        x = np.clip(x * mult, 0.02, None)
        P[f"x_{c}"] = x * (m / x.mean(axis=1))[:, None]

    base = rng.uniform(500, 700, n)[:, None] + 40.0 * t[None, :] / _PERIOD
    for e in edges:
        if e["relation"] == "drives" and e["source"] in drivers and e["target"] == outcome:
            d, nd = e["source"], nodes[e["source"]]
            if nd.get("modeled") or "season" in d.lower():
                coef = rng.uniform(60, 120, n)
            elif nd.get("kind") == "event":
                coef = rng.uniform(40, 90, n)
            else:
                coef = rng.uniform(20, 60, n) * rng.choice([-1.0, 1.0], n)
            P[f"coef_{d}"] = coef
            base = base + coef[:, None] * P[f"drv_{d}"]
    P["baseline"] = base
    P["noise"] = rng.normal(0, 1, (n, T)) * rng.uniform(25, 35, n)[:, None]
    feeds = {}
    for a, b in med:
        key = f"med_{a}__{b}"
        P[key] = rng.uniform(0.10, 0.30, n)
        feeds.setdefault(b, []).append((a, key))

    exported, controls = {}, {}
    for d in drivers:
        c_ = nodes[d].get("column")
        if c_ and d not in hide and c_ not in hide:
            exported[d] = c_
            controls[c_] = None if nodes[d].get("kind") == "event" else 3
    G = {"order": order, "feeds": feeds, "exported": exported}
    sim = lambda Q: _graph_sim(G, Q, T, l_max)
    run = sim(P)
    names = {c: col[c] for c in order}
    # expose channels under their data column names
    run_named = dict(run)
    for c in order:
        run_named[f"spend_{names[c]}"], run_named[f"effect_{names[c]}"] = run[f"spend_{c}"], run[f"effect_{c}"]
    Pn = dict(P)
    for c in order:
        for k in ("mean", "hs", "shape", "beta", "alpha", "peak", "wear", "x"):
            Pn[f"{k}_{names[c]}"] = P[f"{k}_{c}"]

    def sim_named(Q):
        Q = dict(Q)
        for c in order:
            Q[f"x_{c}"] = Q[f"x_{names[c]}"]
        r = sim(Q)
        for c in order:
            r[f"spend_{names[c]}"], r[f"effect_{names[c]}"] = r[f"spend_{c}"], r[f"effect_{c}"]
        return r

    worlds = MarketingWorlds(run=run_named, params=Pn, dates=pd.date_range("2023-01-02", periods=T, freq="W-MON"),
                             hard=bool(set(drivers) - set(exported)), channels=[names[c] for c in chans],
                             controls=controls, l_max=l_max, target=outcome)
    worlds._sim = sim_named
    worlds.hidden = [d for d in drivers if d not in exported and not (nodes[d].get("modeled") or "season" in d.lower())]
    _answer_key(worlds, Pn, sim_named)
    worlds._total_roas = {}
    for c in chans:
        Q = dict(Pn); Q[f"x_{names[c]}"] = np.zeros((n, T))
        without = sim_named(Q)["sales"].sum(axis=0)
        worlds._total_roas[names[c]] = (run["sales"].sum(axis=0) - without) / run[f"spend_{c}"].sum(axis=0)
    return worlds
