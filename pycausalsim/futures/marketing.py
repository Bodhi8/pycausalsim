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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scenario import Scenario
from .temporal import TemporalSCM, _Params

__all__ = ["CHANNELS", "MarketingWorlds", "marketing_worlds", "media_plan_scm", "compare_plans"]

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

    run: object               # RunResult, series of shape (weeks, worlds)
    params: dict
    dates: pd.DatetimeIndex
    hard: bool

    @property
    def n(self):
        return self.run["sales"].shape[1]

    def frame(self, i: int) -> pd.DataFrame:
        """The data a modeler would see for world i: dates, spend, controls, sales."""
        r = self.run
        cols = {"date": self.dates}
        for c in CHANNELS:
            cols[c] = np.round(r[f"spend_{c}"][:, i], 2)
        cols["price_index"] = np.round(r["price_index"][:, i], 4)
        cols["promo"] = r["promo"][:, i].astype(int)
        cols["sales"] = np.round(r["sales"][:, i], 2)
        return pd.DataFrame(cols)

    def truth(self, i: int) -> dict:
        """Answer key for world i: generating parameters, ROAS and marginal ROAS."""
        p = self.params
        out = {"channels": {}, "launch_weeks": int((self.run["launch"][:, i] > 0).sum())}
        for c in CHANNELS:
            out["channels"][c] = {
                "beta_final": float(p[f"beta_{c}"][i] * p[f"wear_{c}"][i]),
                "half_sat": float(p[f"hs_{c}"][i]),
                "alpha": float(p[f"alpha_{c}"][i]),
                "peak": None if np.isnan(p[f"peak_{c}"][i]) else float(p[f"peak_{c}"][i]),
                "roas": float(self._roas[c][i]),
                "marginal_roas": float(self._mroas[c][i]),
            }
        return out

    def steady_sales(self, i: int, spend) -> float:
        """True steady-state weekly media sales at constant spend, with end-of-window effects."""
        x = np.asarray(spend, dtype=float)
        t = self.truth(i)["channels"]
        return float(sum(t[c]["beta_final"] * x[j] / (x[j] + t[c]["half_sat"])
                         for j, c in enumerate(CHANNELS)))

    def best_plan(self, i: int, total: float, bounds) -> np.ndarray:
        """Spend that maximizes true steady-state sales under the same bounds."""
        from scipy import optimize
        x0 = np.clip(np.full(len(CHANNELS), total / len(CHANNELS)), *np.array(bounds).T)
        r = optimize.minimize(lambda x: -self.steady_sales(i, x), x0, method="SLSQP", bounds=bounds,
                              constraints=[{"type": "eq", "fun": lambda x: x.sum() - total}],
                              options={"ftol": 1e-12, "maxiter": 500})
        return r.x


def marketing_worlds(n_worlds: int = 100, weeks: int = 156, hard: bool = False,
                     seed: int = 0, l_max: int = 8) -> MarketingWorlds:
    """
    Simulate ``n_worlds`` brands with four digital channels and known returns.

    Parameters are drawn per brand: true ROAS between 0.5 and 2.2 per channel,
    carryover, half-saturation near typical spend, seasonal confounding of paid
    search, video flights that peak one to three weeks after airing.

    hard=True adds two things real data has and the standard worlds don't:
    two to four unobserved product launches that lift search and social spend
    and baseline demand together (confounding the model cannot control for),
    and social creative that loses 20 to 40% of its effect over the window.
    """
    rng = np.random.default_rng(seed)
    n, T = int(n_worlds), int(weeks)
    t = np.arange(T)
    season = np.sin(2 * np.pi * t / _PERIOD) + 0.5 * np.sin(4 * np.pi * t / _PERIOD)

    P = {}
    mean_spend = {"paid_search": rng.uniform(20, 40, n), "paid_social": rng.uniform(30, 60, n),
                  "online_video": rng.uniform(30, 60, n), "display": rng.uniform(12, 30, n)}
    alpha_rng = {"paid_search": (0.10, 0.40), "paid_social": (0.30, 0.60),
                 "online_video": (0.40, 0.70), "display": (0.20, 0.50)}
    for c in CHANNELS:
        m = mean_spend[c]
        P[f"mean_{c}"] = m
        P[f"hs_{c}"] = m * rng.uniform(0.8, 1.6, n)
        roas = rng.uniform(0.5, 2.2, n)
        P[f"beta_{c}"] = roas * (m + P[f"hs_{c}"])        # steady ROAS at mean spend ~ roas
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
    P["price_index"] = 1 + 0.04 * np.sin(2 * np.pi * t / 26)[None, :] + rng.normal(0, 0.015, (n, T))
    P["promo"] = (rng.random((n, T)) < 0.12).astype(float)
    P["noise"] = rng.normal(0, 1, (n, T)) * rng.uniform(25, 35, n)[:, None]
    P["intercept"] = rng.uniform(500, 700, n)
    P["season_amp"] = rng.uniform(60, 120, n)
    P["launch_lift"] = rng.uniform(40, 90, n)
    P["wear_t"] = np.tile(t / max(T - 1, 1), (n, 1))

    run = _simulate(P, n, T, l_max)
    dates = pd.date_range("2023-01-02", periods=T, freq="W-MON")
    worlds = MarketingWorlds(run=run, params=P, dates=dates, hard=hard)
    _answer_key(worlds, P, n, T, l_max)
    return worlds


def _build(n, T, l_max, P):
    scm = TemporalSCM(horizon=range(T), step="week", draws=n)
    zero = [0.0] * (l_max - 1)
    w = {c: _kernel(P[f"alpha_{c}"], P[f"peak_{c}"], l_max) for c in CHANNELS}
    scm.exogenous("season", lambda c: c.p.season[:, c.t])
    scm.exogenous("launch", lambda c: c.p.launch[:, c.t])
    scm.exogenous("price_index", lambda c: c.p.price_index[:, c.t])
    scm.exogenous("promo", lambda c: c.p.promo[:, c.t])
    for ch in CHANNELS:
        parents = ["season", "launch"] if ch in ("paid_search", "paid_social") else []
        scm.add_equation(f"spend_{ch}", lambda c, ch=ch: c.p[f"x_{ch}"][:, c.t], parents=parents,
                         pre_history=zero)
        lag_parents = [f"spend_{ch}"] + [f"spend_{ch}:lag({k})" for k in range(1, l_max)]

        def adstock(c, ch=ch, wk=w[ch]):
            return sum(wk[:, k] * (c[f"spend_{ch}"] if k == 0 else c.lag(f"spend_{ch}", k))
                       for k in range(l_max))
        scm.add_equation(f"adstock_{ch}", adstock, parents=lag_parents)

        def effect(c, ch=ch):
            beta = c.p[f"beta_{ch}"] * (1 - (1 - c.p[f"wear_{ch}"]) * c.p.wear_t[:, c.t])
            a = c[f"adstock_{ch}"]
            return beta * a / (a + c.p[f"hs_{ch}"])
        scm.add_equation(f"effect_{ch}", effect, parents=[f"adstock_{ch}"])
    scm.add_equation("baseline", lambda c: (c.p.intercept + 40.0 * c.t / _PERIOD
                                            + c.p.season_amp * c["season"]
                                            - 350.0 * (c["price_index"] - 1) + 120.0 * c["promo"]
                                            + c.p.launch_lift * c["launch"]),
                     parents=["season", "launch", "price_index", "promo"])
    scm.add_equation("sales", lambda c: c["baseline"] + sum(c[f"effect_{ch}"] for ch in CHANNELS)
                     + c.p.noise[:, c.t], parents=["baseline"] + [f"effect_{ch}" for ch in CHANNELS])
    return scm


def _simulate(P, n, T, l_max):
    return _build(n, T, l_max, P).run(params=_Params(P))


def _answer_key(worlds, P, n, T, l_max):
    """True ROAS over the window and marginal ROAS (1% more spend every week)."""
    run = worlds.run
    worlds._roas, worlds._mroas = {}, {}
    for c in CHANNELS:
        spend = run[f"spend_{c}"].sum(axis=0)
        eff = run[f"effect_{c}"].sum(axis=0)
        worlds._roas[c] = eff / spend
        bumped = dict(P)
        bumped[f"x_{c}"] = P[f"x_{c}"] * 1.01
        eff_b = _simulate(bumped, n, T, l_max)[f"effect_{c}"].sum(axis=0)
        worlds._mroas[c] = (eff_b - eff) / (0.01 * spend)


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
        w = _kernel(curves["alpha"][idx, j], curves["peak"][idx, j], l_max)
        scm.lever(f"spend_{c}", default=float(cur[j]))
        scm.add_equation(f"x_{c}", lambda c_, c=c: np.full(n, c_.lever(f"spend_{c}")),
                         pre_history=[float(cur[j])] * (l_max - 1))

        def adstock(c_, c=c, wk=w):
            return sum(wk[:, k] * (c_[f"x_{c}"] if k == 0 else c_.lag(f"x_{c}", k)) for k in range(l_max))
        scm.add_equation(f"adstock_{c}", adstock,
                         parents=[f"x_{c}"] + [f"x_{c}:lag({k})" for k in range(1, l_max)])
        scm.add_equation(f"effect_{c}", lambda c_, c=c: c_.p[f"beta_{c}"] * c_[f"adstock_{c}"]
                         / (c_[f"adstock_{c}"] + c_.p[f"hs_{c}"]), parents=[f"adstock_{c}"])
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
