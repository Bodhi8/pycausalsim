"""
Henry Hub natural gas, winter 2026-27: the futures curve as the center,
storage surprises as the cause.

Companion to "How to Use PyCausalSim to Improve LNG and Natural Gas
Forecasting". The model:

    weather, LNG feedgas, freeze-offs  ->  storage surprise (Bcf vs normal)
    storage surprise                   ->  cumulative surprise (state)
    cumulative surprise + futures curve ->  monthly Henry Hub average

The price response to storage is estimated from EIA data (winters 2014-15 to
2023-24) as the futures curve's miss regressed on the cumulative storage
surprise. Weather is resampled whole-winter from the last twelve winters with
`empirical_path`, so a cold December and an expensive February arrive together.

Data (U.S. Energy Information Administration, public domain) ship in
examples/data/henry_hub as year-by-month CSVs:
    spot.csv       Henry Hub spot price, monthly average (RNGWHHDm), $/MMBtu
    c1-c4.csv      NYMEX futures Contracts 1 to 4, monthly average (RNGC1m-RNGC4m)
    storage.csv    Lower 48 working gas in underground storage, MMcf

    python examples/henry_hub_winter.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pycausalsim.futures import (
    TemporalSCM, Scenario, compare, rolling_backtest)

DATA = os.path.join(os.path.dirname(__file__), "data", "henry_hub")
ABBR = ["jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"]
WINTER = (11, 12, 1, 2, 3)
DAYS = {11: 30, 12: 31, 1: 31, 2: 28, 3: 31}

# NYMEX settles on October 23, 2026, November 2026 to March 2027.
MONTHS = [202611, 202612, 202701, 202702, 202703]
CURVE = [2.640, 3.050, 3.506, 3.170, 2.504]


# --------------------------------------------------------------------- data
def load(name):
    w = pd.read_csv(os.path.join(DATA, f"{name}.csv"))
    w = w.melt(id_vars="year", value_vars=ABBR, var_name="m", value_name="v")
    w["t"] = pd.PeriodIndex.from_fields(
        year=w["year"], month=w["m"].map(ABBR.index) + 1, freq="M")
    return w.dropna(subset=["v"]).set_index("t")["v"].sort_index()


def storage_surprise():
    """Monthly storage change minus the same month's average change over the
    prior five years (at least three). Negative means tighter than normal."""
    dw = (load("storage") / 1000.0).diff()
    out = pd.Series(np.nan, index=dw.index)
    for t in dw.index:
        prior = [dw[t - 12 * k] for k in range(1, 6)
                 if (t - 12 * k) in dw.index and not np.isnan(dw[t - 12 * k])]
        if len(prior) >= 3:
            out[t] = dw[t] - np.mean(prior)
    return out


def calibrate(exclude_winter=None, horizons=4):
    """Per-horizon premium, storage slope and residual sd for winter months,
    plus the observed range of the cumulative surprise in Bcf.
    `exclude_winter` (e.g. 2021 for 2021-22) supports leave-one-out tests."""
    spot, surp = load("spot"), storage_surprise()
    idx = spot.index
    calib, lo, hi = {}, np.inf, -np.inf
    for h in range(1, horizons + 1):
        fut = load(f"c{h}")
        miss = (np.log(spot.reindex(idx + h).to_numpy())
                - np.log(fut.reindex(idx).to_numpy()))
        cum = sum(surp.reindex(idx + k).to_numpy() for k in range(1, h + 1))
        tgt = idx + h
        winter_of = np.where(tgt.month >= 11, tgt.year, tgt.year - 1)
        ok = np.isin(tgt.month, WINTER) & ~np.isnan(miss) & ~np.isnan(cum)
        if exclude_winter is not None:
            ok &= winter_of != exclude_winter
        X = np.column_stack([np.ones(ok.sum()), cum[ok] / 100.0])
        b, *_ = np.linalg.lstsq(X, miss[ok], rcond=None)
        e = miss[ok] - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * e.var(ddof=2)))
        calib[h] = dict(const=b[0], const_se=se[0], beta=-b[1], beta_se=se[1],
                        sd=float(e.std(ddof=2)))
        lo, hi = min(lo, cum[ok].min()), max(hi, cum[ok].max())
    # One step past the estimated horizons (March from an October curve).
    r3, r4 = calib[horizons - 1], calib[horizons]
    calib[horizons + 1] = dict(
        const=2 * r4["const"] - r3["const"], const_se=1.2 * r4["const_se"],
        beta=r4["beta"], beta_se=1.2 * r4["beta_se"], sd=2 * r4["sd"] - r3["sd"])
    return calib, (float(lo), float(hi))


def winter_paths(exclude_winter=None, n_steps=5):
    surp = storage_surprise()
    rows = []
    for y in range(2012, 2030):
        if y == exclude_winter:
            continue
        idx = ([pd.Period(f"{y}-{m:02d}", "M") for m in (11, 12)]
               + [pd.Period(f"{y + 1}-{m:02d}", "M") for m in (1, 2, 3)])
        vals = [surp.get(i, np.nan) for i in idx]
        if not np.any(np.isnan(vals)):
            rows.append(vals[:n_steps])
    return np.array(rows)


# -------------------------------------------------------------------- model
def build(months=MONTHS, curve=CURVE, calib=None, paths=None,
          draws=20000, seed=2026, weather_weights=None):
    calib, cum_range = calib or calibrate()
    paths = winter_paths(n_steps=len(months)) if paths is None else paths
    n = len(months)

    scm = TemporalSCM(horizon=months, step="month", draws=draws, seed=seed)
    for h in range(1, n + 1):
        r = calib[h]
        scm.parameter(f"const_{h}", "normal", loc=r["const"], scale=r["const_se"])
        scm.parameter(f"beta_{h}", "normal", loc=r["beta"], scale=r["beta_se"])
        scm.parameter(f"z_{h}", "normal", loc=0.0, scale=1.0)

    scm.lever("weather", default=0.0)         # Bcf vs normal, negative = colder
    scm.lever("random_weather", default=1.0)  # 1 = resample past winters
    scm.lever("lng_extra", default=0.0)       # Bcf/d feedgas above expectations
    scm.lever("freeze_off", default=0.0)      # Bcf/d production lost

    # Whole past winters, optionally weighted (e.g. by a seasonal outlook).
    scm.empirical_path("weather_draw", paths, weights=weather_weights)

    scm.add_equation(
        "storage_surprise",
        lambda c: (c["weather"] + c["random_weather"] * c["weather_draw"]
                   - (c["lng_extra"] + c["freeze_off"]) * DAYS[c.step % 100]),
        parents=["weather", "random_weather", "weather_draw",
                 "lng_extra", "freeze_off"])
    scm.state("cum_surprise",
              lambda c: c.lag("cum_surprise", 1) + c["storage_surprise"],
              parents=["cum_surprise:lag(1)", "storage_surprise"],
              init=lambda p: 0.0)

    sds = [calib[h]["sd"] for h in range(1, n + 1)]

    def resid(c):
        step_sd = np.sqrt(sds[c.t] ** 2 - (sds[c.t - 1] ** 2 if c.t else 0.0))
        return c.lag("resid", 1) + step_sd * c.p[f"z_{c.t + 1}"]
    scm.state("resid", resid, parents=["resid:lag(1)"], init=lambda p: 0.0)

    def price(c):
        h = c.t + 1
        tight = -c["cum_surprise"] / 100.0
        miss = c.p[f"const_{h}"] + c.p[f"beta_{h}"] * tight + c["resid"]
        return curve[c.t] * np.exp(miss)
    scm.add_equation("price", price, parents=["cum_surprise", "resid"])

    scm.mechanism("price ~ cum_surprise", form="log_linear",
                  valid_range=cum_range,
                  outside="flag",
                  note="storage slope estimated on winters 2014-15 to 2023-24")

    strip = float(np.mean(curve))
    scm.claim("jan_above_400", "January 2027 Henry Hub spot averages above $4.00",
              test=lambda r: r["price"] > 4.00, resolve_by=202701,
              source="EIA RNGWHHDm")
    scm.claim("winter_below_strip",
              f"Nov 2026 to Mar 2027 spot average settles below the ${strip:.2f} strip",
              test=lambda r: r.window("price", months[0], months[-1]) < strip,
              resolve_by=months[-1], source="EIA RNGWHHDm")
    return scm


def replay(name, path):
    """A past winter's storage surprises replayed onto this winter's curve."""
    s = Scenario(name)
    s.set("random_weather", 0.0)
    return s.path("weather", dict(zip(MONTHS, path)))


MARKET = Scenario("Market view")
POLAR_VORTEX = replay("Replay 2013-14 polar vortex", [-165, -330, -328, -192, -166])
LNG_UPSIDE = Scenario("LNG feedgas +1.5 Bcf/d").set("lng_extra", 1.5)


def hedge_for_budget(run, curve=CURVE, cap=0.10):
    """Smallest share locked at the strip that keeps the simulated 95th
    percentile winter cost within (1 + cap) x strip."""
    sims = np.stack([run.at("price", m) for m in MONTHS])
    c = np.asarray(curve)
    for k in np.round(np.arange(0, 1.0001, 0.05), 2):
        cost = (k * c[:, None] + (1 - k) * sims).mean(axis=0)
        if np.percentile(cost, 95) <= (1 + cap) * c.mean():
            return float(k), float(cost.mean()), float(np.percentile(cost, 95))
    return 1.0, float(c.mean()), float(c.mean())


def winter_backtest(first=2014, last=2023, draws=6000):
    """Leave-one-winter-out, from each October's futures curve, Nov to Feb."""
    spot = load("spot")
    cases = []
    for y in range(first, last + 1):
        mp = [pd.Period(f"{y}-11", "M"), pd.Period(f"{y}-12", "M"),
              pd.Period(f"{y + 1}-01", "M"), pd.Period(f"{y + 1}-02", "M")]
        months = [int(m.strftime("%Y%m")) for m in mp]
        curve = [load(f"c{h}")[pd.Period(f"{y}-10", "M")] for h in (1, 2, 3, 4)]
        scm = build(months, curve, calib=calibrate(exclude_winter=y),
                    paths=winter_paths(exclude_winter=y, n_steps=4),
                    draws=draws, seed=500 + y)
        cases.append((f"{y}-{str(y + 1)[2:]}", scm,
                      {"price": {m: float(spot[p]) for m, p in zip(months, mp)}}))
    return rolling_backtest(cases)


if __name__ == "__main__":
    scm = build()
    cmp = compare(scm, [MARKET, POLAR_VORTEX, LNG_UPSIDE], target="price")
    print(cmp.table(year=202701))
    for name in cmp.runs:
        w = cmp[name].window("price", MONTHS[0], MONTHS[-1])
        print(f"  {name:32s} winter average {w.mean():.2f} "
              f"[{np.percentile(w, 5):.2f}, {np.percentile(w, 95):.2f}]")
    print()
    print(cmp[POLAR_VORTEX.name].support.summary())
    print()
    k, mean, p95 = hedge_for_budget(cmp[MARKET.name])
    print(f"Hedge to a strip+10% budget: lock {k:.0%}, expected ${mean:.2f}, "
          f"95th percentile ${p95:.2f}")
    print()
    print(cmp.falsifiable_claims(scenario=MARKET.name))
    print()
    print(winter_backtest().summary())
