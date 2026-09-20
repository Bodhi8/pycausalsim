"""
The post-labor transition model, expressed in pycausalsim.futures.

Companion to "Simulating the Note from the Future". Reproduces the scenario
table, the driver ranking and the verification notch, and additionally emits a
claim set and a backtest that the standalone version could not produce.

    python examples/postlabor_futures.py
"""
import numpy as np

from pycausalsim.futures import (
    TemporalSCM, Scenario, SI, rank_drivers, compare, backtest)

HORIZON = range(2026, 2042)


def build(draws=4000, seed=20260919, sever_byproduct=False):
    scm = TemporalSCM(horizon=HORIZON, step="year", draws=draws, seed=seed)

    # ---- priors. Wide where the evidence is thin. ----------------------
    scm.parameter("decay",    "normal", loc=0.52,  scale=0.10)
    scm.parameter("a1",       "normal", loc=0.62,  scale=0.10)
    scm.parameter("lam_j",    "normal", loc=1.05,  scale=0.18)
    scm.parameter("attrition","normal", loc=0.045, scale=0.008)
    scm.parameter("conversion","normal",loc=0.085, scale=0.012)
    scm.parameter("gamma",    "normal", loc=1.45,  scale=0.25)
    scm.parameter("nu",       "normal", loc=0.55,  scale=0.15)
    scm.parameter("b_land",   "normal", loc=0.42,  scale=0.10)
    scm.parameter("b_energy", "normal", loc=0.30,  scale=0.09)
    scm.parameter("b_care",   "normal", loc=0.28,  scale=0.08)
    scm.parameter("k_auto",   "normal", loc=0.21,  scale=0.05)
    scm.parameter("u_max",    "normal", loc=0.38,  scale=0.06)
    scm.parameter("w_latent", "normal", loc=0.62,  scale=0.07)

    # ---- policy levers --------------------------------------------------
    scm.lever("income_floor",   default=0.25)
    scm.lever("institutions",   default=0.10)
    scm.lever("apprenticeship", default=0.05)
    scm.lever("ownership",      default=0.05)

    # ---- exogenous driver ------------------------------------------------
    scm.exogenous("token_price",
                  lambda c: 0.97 * 10.0 ** (-c.p.decay * c.t))

    # ---- automation depth, the only mechanical link ----------------------
    def automation(c):
        cheap = np.clip(np.log10(0.97 / c["token_price"]), 0, None)
        a = 1.0 / (1.0 + np.exp(-(-1.35 + c.p.a1 * cheap)))
        a0 = 1.0 / (1.0 + np.exp(1.35))
        return np.clip((a - a0) / (1 - a0), 0, 1)

    scm.add_equation("automation", automation, parents=["token_price"])
    scm.mechanism("automation ~ token_price", form="logistic",
                  saturates_at=1.0, valid_range=(0.0005, 100.0),
                  outside="flag",
                  note="no observation exists below a thousandth of 2026 cost")

    # ---- the severed ladder ---------------------------------------------
    scm.add_equation(
        "junior",
        lambda c: np.clip(88.0 * (1 - c.p.lam_j * c["automation"] * 0.62)
                          + 46.0 * c["apprenticeship"], 5, 130),
        parents=["automation", "apprenticeship"],
        pre_history=[100.0] * 11 + [88.0])

    # training used to be a free byproduct of junior production. Automation
    # does not lower its value, it severs the mechanism.
    # Structural variant, off by default so this example reproduces the
    # numbers published in the article. Switch it on to see what happens when
    # the byproduct channel is severed outright rather than merely thinned.
    if sever_byproduct:
        scm.structural_rule(
            when="automation > 0.55",
            apply=SI.remove_edge("junior", "senior_stock", on_removed=0.0),
            note="training stops being a free byproduct of junior output")

    scm.state(
        "senior_stock",
        lambda c: (c.lag("senior_stock", 1) * (1 - c.p.attrition)
                   + c.p.conversion * c.lag("junior", 10)),
        parents=["junior:lag(10)", "senior_stock:lag(1)"],
        init=lambda p: p.conversion / p.attrition * 100.0)

    # ---- verification: demand, capacity, gap -----------------------------
    scm.add_equation("verify_demand",
                     lambda c: 100.0 * (1 + c.p.gamma * c["automation"]),
                     parents=["automation"])

    scm.add_equation(
        "verify_capacity",
        lambda c: (c["senior_stock"] / (c.p.conversion / c.p.attrition * 100.0)
                   * 100.0 * (1 + c.p.nu * c["automation"])),
        parents=["senior_stock", "automation"])

    scm.add_equation("verify_gap",
                     lambda c: c["verify_demand"] / np.maximum(c["verify_capacity"], 1e-6),
                     parents=["verify_demand", "verify_capacity"])

    scm.add_equation(
        "output",
        lambda c: (100.0 * (1 + 1.9 * c["automation"])
                   / (1 + 2.2 * np.clip(c["verify_gap"] - 1, 0, None))),
        parents=["automation", "verify_gap"])

    # ---- prices -----------------------------------------------------------
    scm.add_equation("cognitive_price",
                     lambda c: 100.0 * np.exp(-3.25 * c["automation"]),
                     parents=["automation"])

    scm.add_equation(
        "bottleneck_price",
        lambda c: 100.0 * (1 + c.p.b_land * c["automation"] * (1 - 0.55 * c["ownership"])
                           + c.p.b_energy * 0.55 * c["automation"]
                           + c.p.b_care * 0.40 * (c.t / (len(scm.years) - 1))),
        parents=["automation", "ownership"])

    def cpi(c):
        w = 0.46 * np.exp(-2.4 * c.t / (len(scm.years) - 1))
        return w * c["cognitive_price"] + (1 - w) * c["bottleneck_price"]

    scm.add_equation("cpi", cpi, parents=["cognitive_price", "bottleneck_price"])

    # ---- distribution ------------------------------------------------------
    scm.add_equation(
        "capital_share",
        lambda c: 0.38 + c.p.k_auto * c["automation"] - 0.55 * c["ownership"] * c["automation"],
        parents=["automation", "ownership"])

    scm.add_equation("outside_employment",
                     lambda c: c.p.u_max * c["automation"],
                     parents=["automation"])

    # ---- wellbeing, on Jahoda's latent deprivation structure ---------------
    def wellbeing(c):
        M = np.clip((c["income_floor"] * 145.0 + c["ownership"] * 95.0)
                    / np.maximum(c["cpi"], 1e-6), 0, 1.3)
        material = 1 / (1 + np.exp(-4.0 * (M - 0.72)))
        # apprenticeship is itself an institution: it supplies time structure,
        # social contact, status, collective purpose and effortful activity
        inst = np.clip(c["institutions"] + 0.45 * c["apprenticeship"], 0, 1)
        latent = np.clip(inst * (0.55 + 0.45 * inst), 0, 1)
        w_out = (1 - c.p.w_latent) * material + c.p.w_latent * latent
        w_emp = 0.78 - 0.30 * np.clip(c["verify_gap"] - 1, 0, None)
        u = c["outside_employment"]
        return u * w_out + (1 - u) * w_emp

    scm.add_equation("wellbeing", wellbeing,
                     parents=["cpi", "income_floor", "ownership", "institutions",
                              "apprenticeship", "verify_gap", "outside_employment"])

    scm.add_equation(
        "legitimacy",
        lambda c: np.clip(0.80 - 0.85 * (c["capital_share"] - 0.38)
                          - 0.55 * c["outside_employment"] * (1 - c["institutions"])
                          + 0.30 * c["ownership"], 0, 1),
        parents=["capital_share", "outside_employment", "institutions", "ownership"])

    # ---- claims: dated, checkable, with a source --------------------------
    scm.claim("entry_gap_25pct",
              "entry-level employment gap in exposed occupations exceeds 25 percent",
              test=lambda r: r["junior"] <= 75.0, resolve_by=2029,
              source="Stanford Digital Economy Lab payroll series")
    scm.claim("verify_gap_opens",
              "verification demand exceeds senior capacity by more than 40 percent",
              test=lambda r: r["verify_gap"] >= 1.40, resolve_by=2036,
              source="BLS occupational employment, audit and assurance categories")
    scm.claim("cpi_bottleneck_dominates",
              "bottleneck prices exceed 140 percent of their 2026 level",
              test=lambda r: r["bottleneck_price"] >= 140.0, resolve_by=2036,
              source="BLS CPI shelter, energy services and medical care series")
    scm.claim("capital_share_50",
              "capital share of income exceeds 50 percent",
              test=lambda r: r["capital_share"] >= 0.50, resolve_by=2038,
              source="BEA national income and product accounts")

    return scm


# ---------------------------------------------------------------- scenarios
DRIFT = Scenario("Drift")

TRANSFERS = Scenario("Transfers only")
TRANSFERS.at(2032).set("income_floor", ramp_to=0.85, over=3)
TRANSFERS.at(2026).set("institutions", to=0.15)
TRANSFERS.at(2026).set("apprenticeship", to=0.10)
TRANSFERS.at(2026).set("ownership", to=0.10)

FULL = Scenario("Ownership and institutions")
FULL.at(2027).set("apprenticeship", ramp_to=0.72, over=4,
                  note="funded as a cost centre, not harvested as a byproduct")
FULL.at(2028).set("institutions", ramp_to=0.78, over=6)
FULL.at(2028).set("ownership", ramp_to=0.52, over=7)
FULL.at(2031).set("income_floor", ramp_to=0.60, over=3)


def main():
    scm = build()
    p = scm.sample_params()

    cmp = compare(scm, [DRIFT, TRANSFERS, FULL], target="wellbeing", params=p)
    print(cmp.table())
    print()

    share = cmp.share_of_gap("Transfers only", "Ownership and institutions")
    print(f"Transfers close {share:.0%} of the distance to the full package.")
    d = cmp.diff("Ownership and institutions", "Drift")
    print(f"Full package vs Drift: {d['mean']:+.3f} "
          f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], "
          f"better in {d['p_positive']:.0%} of draws")
    print()

    print("Driver ranking, single-lever do() from Drift:")
    for e in rank_drivers(scm, DRIFT, target="wellbeing", seed_params=p):
        print("  " + repr(e))
    print()

    drift_run = cmp["Drift"]
    print(drift_run.support.summary())
    if drift_run.fired:
        print("\nStructural rules fired:")
        for note, events in drift_run.fired.items():
            yr, frac = events[0]
            print(f"  {yr}: {note}  ({frac:.0%} of draws)")
    print()

    claims = cmp.falsifiable_claims(scenario="Drift", model_version="postlabor-0.2")
    print("Falsifiable claims, Drift scenario:")
    print(claims)
    claims.export("claims_2026.json")
    print("\nwrote claims_2026.json")

    bt = backtest(scm, observations={
        "junior": {2026: 88.0},
        "cognitive_price": {2026: 100.0},
        "capital_share": {2026: 0.38},
    }, scenario=DRIFT, params=p)
    print()
    print(bt.summary())


if __name__ == "__main__":
    main()
