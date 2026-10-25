import json
import os
import tempfile

import numpy as np
import pytest

from pycausalsim.futures import (
    TemporalSCM, Scenario, SI, rank_drivers, compare, backtest,
    rolling_backtest, ClaimSet, OutOfSupport)


# ----------------------------------------------------------------- fixtures
def simple(draws=300, seed=0):
    scm = TemporalSCM(horizon=range(2000, 2011), draws=draws, seed=seed)
    scm.parameter("g", "normal", loc=0.10, scale=0.01)
    scm.parameter("decay", "fixed", value=0.05)
    scm.lever("policy", default=0.0)
    scm.exogenous("drive", lambda c: np.full(c.p.g.shape, float(c.t)))
    scm.add_equation("flow", lambda c: 10.0 + c.p.g * c["drive"] + 5 * c["policy"],
                     parents=["drive", "policy"], pre_history=[10.0, 10.0])
    scm.state("stock",
              lambda c: c.lag("stock", 1) * (1 - c.p.decay) + c.lag("flow", 2),
              parents=["flow:lag(2)", "stock:lag(1)"],
              init=lambda p: np.full(p.g.shape, 100.0))
    return scm


# -------------------------------------------------------------------- core
def test_runs_and_shapes():
    scm = simple()
    r = scm.run()
    assert r["stock"].shape == (11, 300)
    assert r.at("stock", 2005).shape == (300,)
    assert np.all(np.isfinite(r["stock"]))


def test_lag_uses_pre_history_then_series():
    scm = simple(draws=5)
    r = scm.run()
    # at t=0 and t=1 the flow lag comes from pre_history (10.0)
    expected_t0 = 100.0 * 0.95 + 10.0
    assert np.allclose(r["stock"][0], expected_t0)


def test_state_init_seeds_its_own_lag():
    scm = TemporalSCM(horizon=range(0, 3), draws=4, seed=1)
    scm.state("s", lambda c: c.lag("s", 1) + 1.0,
              parents=["s:lag(1)"], init=lambda p: np.zeros(4))
    r = scm.run()
    assert np.allclose(r["s"][:, 0], [1.0, 2.0, 3.0])


def test_cycle_at_lag_zero_is_rejected():
    scm = TemporalSCM(horizon=range(0, 3), draws=2)
    scm.add_equation("a", lambda c: c["b"], parents=["b"])
    scm.add_equation("b", lambda c: c["a"], parents=["a"])
    with pytest.raises(ValueError, match="cycle"):
        scm.run()


def test_lag_breaks_the_cycle():
    scm = TemporalSCM(horizon=range(0, 4), draws=2)
    scm.add_equation("a", lambda c: c.lag("b", 1) + 1.0, parents=["b:lag(1)"])
    scm.add_equation("b", lambda c: c["a"] * 2.0, parents=["a"],
                     pre_history=[0.0])
    r = scm.run()
    assert r["a"].shape[0] == 4


def test_bad_parent_spec():
    scm = TemporalSCM(horizon=range(0, 2), draws=2)
    with pytest.raises(ValueError):
        scm.add_equation("a", lambda c: 1.0, parents=["b:lagg(2)"])


def test_missing_history_is_a_clear_error():
    scm = TemporalSCM(horizon=range(0, 3), draws=2)
    scm.add_equation("a", lambda c: c.lag("a", 5), parents=["a:lag(5)"])
    with pytest.raises(KeyError, match="history"):
        scm.run()


# ----------------------------------------------------------------- levers
def test_scenario_step_and_ramp():
    scm = simple()
    s = Scenario("push")
    s.at(2005).set("policy", ramp_to=1.0, over=2)
    r = scm.run(s)
    path = r.levers["policy"]
    assert path[4] == 0.0
    assert path[5] == pytest.approx(0.5)
    assert path[6] == pytest.approx(1.0)
    assert path[-1] == pytest.approx(1.0)


def test_scenario_before_horizon_applies_immediately():
    scm = simple()
    s = Scenario("early").set("policy", 0.4)
    assert scm.run(s).levers["policy"][0] == pytest.approx(0.4)


def test_scenario_without_drops_a_lever():
    s = Scenario("x")
    s.at(2005).set("policy", to=1.0)
    s.at(2005).set("other", to=1.0)
    assert s.without("policy").levers() == ["other"]


def test_scenario_roundtrips_through_json():
    s = Scenario("x")
    s.at(2005).set("policy", ramp_to=0.5, over=3, note="hi")
    back = Scenario.from_dict(json.loads(s.to_json()))
    assert back.name == "x"
    assert back.interventions[0].ramp_to == 0.5
    assert back.interventions[0].note == "hi"


# ------------------------------------------------------------- mechanisms
def test_out_of_support_is_flagged_not_silent():
    scm = simple()
    scm.mechanism("flow ~ drive", valid_range=(0, 3), outside="flag")
    r = scm.run()
    assert not r.support.clean
    rep = r.support.by_edge()["flow ~ drive"]
    assert rep["first_year"] == 2004
    assert rep["max_fraction"] == 1.0


def test_out_of_support_can_raise():
    scm = simple()
    scm.mechanism("flow ~ drive", valid_range=(0, 3), outside="error")
    with pytest.raises(OutOfSupport):
        scm.run()


def test_clip_bounds_the_parent():
    scm = simple()
    scm.mechanism("flow ~ drive", valid_range=(0, 3), outside="clip")
    r = scm.run()
    assert r["flow"][-1].mean() == pytest.approx(10.0 + 0.10 * 3, abs=0.02)


def test_clean_support_reports_clean():
    scm = simple()
    scm.mechanism("flow ~ drive", valid_range=(-1e9, 1e9))
    assert scm.run().support.clean


# ------------------------------------------------------- structural change
def test_remove_edge_changes_the_result():
    def mk(rule):
        scm = simple(seed=3)
        if rule:
            scm.structural_rule(when="drive > 4",
                                apply=SI.remove_edge("flow", "stock"),
                                note="channel severed")
        return scm.run()

    intact, severed = mk(False), mk(True)
    assert severed.at("stock", 2010).mean() < intact.at("stock", 2010).mean()
    assert severed.fired


def test_rule_fires_per_draw():
    scm = TemporalSCM(horizon=range(0, 2), draws=1000, seed=5)
    scm.parameter("x", "normal", loc=0.0, scale=1.0)
    scm.exogenous("noise", lambda c: c.p.x)
    scm.add_equation("y", lambda c: c["noise"] + 100.0, parents=["noise"])
    scm.structural_rule(when="noise > 0",
                        apply=SI.remove_edge("noise", "y"), note="half")
    r = scm.run()
    frac = r.fired["half"][0][1]
    assert 0.4 < frac < 0.6           # roughly half the draws


def test_callable_condition_works():
    scm = simple()
    scm.structural_rule(when=lambda c: c["drive"] > 4,
                        apply=SI.remove_edge("flow", "stock"), note="cb")
    assert scm.run().fired


def test_unparseable_condition_is_rejected():
    scm = simple()
    with pytest.raises(ValueError, match="cannot parse"):
        scm.structural_rule(when="drive is large",
                            apply=SI.remove_edge("flow", "stock"))


# --------------------------------------------------------------- compare
def test_compare_is_paired_on_the_same_draws():
    scm = simple()
    base = Scenario("base")
    push = Scenario("push").set("policy", 1.0)
    c = compare(scm, [base, push], target="stock")
    d = c.diff("push", "base")
    assert d["mean"] > 0
    assert d["p_positive"] == 1.0


def test_share_of_gap():
    scm = simple()
    base = Scenario("base")
    half = Scenario("half").set("policy", 0.5)
    full = Scenario("full").set("policy", 1.0)
    c = compare(scm, [base, half, full], target="stock")
    assert c.share_of_gap("half", "full") == pytest.approx(0.5, abs=0.02)


def test_rank_drivers_orders_by_effect():
    scm = simple()
    scm.lever("weak", default=0.0)
    scm.add_equation("flow2", lambda c: c["flow"] + 0.01 * c["weak"],
                     parents=["flow", "weak"])
    eff = rank_drivers(scm, Scenario("base"), target="stock",
                       levers=["policy", "weak"])
    assert [e.lever for e in eff] == ["policy", "weak"]
    assert eff[0].effect > eff[1].effect
    assert eff[0].ci[0] <= eff[0].effect <= eff[0].ci[1]


# ---------------------------------------------------------------- claims
def test_claims_carry_probabilities_and_export():
    scm = simple()
    scm.claim("stock_high", "stock exceeds 150", lambda r: r["stock"] > 150,
              resolve_by=2010, source="ledger")
    c = compare(scm, [Scenario("base")], target="stock")
    claims = c.falsifiable_claims(model_version="t")
    assert len(claims) == 1
    assert 0.0 <= claims.claims[0].p <= 1.0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        claims.export(path)
        back = ClaimSet.load(path)
        assert back.claims[0].key == "stock_high"


def test_threshold_filters_claims():
    scm = simple()
    scm.claim("never", "stock exceeds a million", lambda r: r["stock"] > 1e6,
              resolve_by=2010)
    c = compare(scm, [Scenario("base")], target="stock")
    assert len(c.falsifiable_claims(threshold=0.5)) == 0
    assert len(c.falsifiable_claims(threshold=0.0)) == 1


def test_brier_score_rewards_confident_and_correct():
    from pycausalsim.futures import FalsifiableClaim
    good = ClaimSet([FalsifiableClaim("a", "s", 2030, 0.95, "x")])
    bad = ClaimSet([FalsifiableClaim("a", "s", 2030, 0.05, "x")])
    assert good.score({"a": True})["brier"] < bad.score({"a": True})["brier"]


def test_unresolved_claims_are_skipped():
    from pycausalsim.futures import FalsifiableClaim
    cs = ClaimSet([FalsifiableClaim("a", "s", 2030, 0.5, "x")])
    assert cs.score({})["n"] == 0


# -------------------------------------------------------------- backtest
def test_backtest_scores_anchors():
    scm = simple()
    bt = backtest(scm, {"stock": {2000: 105.0}})
    assert len(bt.scores) == 1
    assert bt.scores[0].observed == 105.0
    assert "Backtest" in bt.summary()


def test_backtest_rejects_unknown_anchor():
    scm = simple()
    with pytest.raises(KeyError):
        backtest(scm, {"nope": {2000: 1.0}})


def test_backtest_skips_years_outside_horizon():
    scm = simple()
    bt = backtest(scm, {"stock": {1990: 1.0, 2000: 105.0}})
    assert len(bt.scores) == 1


# ------------------------------------------------- the published example
def test_postlabor_example_reproduces_published_numbers():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from postlabor_futures import build, DRIFT, TRANSFERS, FULL

    scm = build(draws=4000, seed=20260919)
    p = scm.sample_params()
    c = compare(scm, [DRIFT, TRANSFERS, FULL], target="wellbeing", params=p)

    assert c["Drift"].at("wellbeing", 2041).mean() == pytest.approx(0.409, abs=0.01)
    assert c["Transfers only"].at("wellbeing", 2041).mean() == pytest.approx(0.488, abs=0.01)
    assert c["Ownership and institutions"].at("wellbeing", 2041).mean() == pytest.approx(0.702, abs=0.01)
    assert c.share_of_gap("Transfers only", "Ownership and institutions") == pytest.approx(0.27, abs=0.02)

    eff = {e.lever: e.effect for e in rank_drivers(scm, DRIFT, target="wellbeing",
                                                   seed_params=p)}
    assert eff["institutions"] == pytest.approx(0.136, abs=0.005)
    assert eff["institutions"] / eff["income_floor"] == pytest.approx(2.8, abs=0.15)
    assert eff["institutions"] > eff["apprenticeship"] > eff["ownership"] > eff["income_floor"]


# ------------------------------------------------------------ 0.3 additions
def monthly(draws=500, seed=1, weights=None):
    """Three monthly steps with resampled episodes, YYYYMM step labels."""
    paths = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]])
    scm = TemporalSCM(horizon=[202611, 202612, 202701], step="month",
                      draws=draws, seed=seed)
    scm.lever("shift", default=0.0)
    scm.empirical_path("shock", paths, weights=weights)
    scm.add_equation("y", lambda c: c["shock"] + c["shift"],
                     parents=["shock", "shift"])
    return scm, paths


def test_choice_parameter_draws_only_listed_values():
    scm = TemporalSCM(horizon=range(2020, 2022), draws=4000, seed=3)
    scm.parameter("regime", "choice", values=[0.8, 1.0, 1.3], p=[0.2, 0.5, 0.3])
    p = scm.sample_params()
    assert set(np.unique(p.regime)) <= {0.8, 1.0, 1.3}
    assert np.mean(p.regime == 1.0) == pytest.approx(0.5, abs=0.03)


def test_choice_parameter_needs_values():
    scm = TemporalSCM(horizon=range(2020, 2022))
    with pytest.raises(ValueError, match="values"):
        scm.parameter("regime", "choice")


def test_empirical_path_keeps_each_episode_intact():
    scm, paths = monthly()
    y = scm.run()["y"]                       # (steps, draws)
    rows = {tuple(r) for r in paths}
    assert all(tuple(y[:, d]) in rows for d in range(y.shape[1]))


def test_empirical_path_weights_steer_the_draws():
    scm, paths = monthly(draws=2000, weights=[0.0, 1.0, 0.0])
    assert np.all(scm.run()["y"][0] == 10.0)


def test_empirical_path_is_paired_across_scenarios():
    scm, _ = monthly()
    shifted = Scenario("Shifted").set("shift", 5.0)
    c = compare(scm, [Scenario("Base"), shifted], target="y")
    d = c["Shifted"]["y"] - c["Base"]["y"]
    assert np.allclose(d, 5.0)


def test_empirical_path_rejects_too_few_steps():
    scm = TemporalSCM(horizon=[1, 2, 3])
    with pytest.raises(ValueError, match="steps"):
        scm.empirical_path("shock", np.zeros((4, 2)))


def test_scenario_path_sets_each_step():
    scm, _ = monthly()
    s = Scenario("Path").path("shift", {202612: 1.0, 202701: 4.0})
    lv = scm.run(s).levers["shift"]
    assert list(lv) == [0.0, 1.0, 4.0]


def test_window_aggregates_per_draw():
    scm, _ = monthly(draws=50)
    r = scm.run()
    y = r["y"]
    assert np.allclose(r.window("y"), y.mean(axis=0))
    assert np.allclose(r.window("y", 202612, 202701, agg="sum"), y[1] + y[2])
    assert np.allclose(r.window("y", agg="last"), y[-1])
    with pytest.raises(ValueError, match="agg"):
        r.window("y", agg="median")


def test_claims_can_test_a_window():
    scm, _ = monthly(draws=3000)
    scm.claim("avg_positive", "average shock is positive",
              test=lambda r: r.window("y", 202611, 202701) > 0,
              resolve_by=202701)
    claims = compare(scm, [Scenario("Base")], target="y").falsifiable_claims()
    assert claims.claims[0].p == pytest.approx(2 / 3, abs=0.03)


def test_step_aliases():
    scm, _ = monthly()
    seen = []
    scm.add_equation("probe", lambda c: seen.append(c.step) or np.zeros(scm.draws))
    scm.run()
    assert scm.steps == [202611, 202612, 202701]
    assert seen == [202611, 202612, 202701]


def test_rolling_backtest_pools_every_origin():
    cases = []
    for k, start in enumerate((2000, 2005)):
        scm = TemporalSCM(horizon=range(start, start + 3), draws=500, seed=k)
        scm.parameter("level", "normal", loc=10.0, scale=1.0)
        scm.add_equation("x", lambda c: c.p.level)
        cases.append((f"origin {start}", scm, {"x": {start: 10.2, start + 1: 9.7}}))
    bt = rolling_backtest(cases)
    assert len(bt.scores) == 4
    assert set(bt.cases) == {"origin 2000", "origin 2005"}
    assert bt.window == ("origin 2000", "origin 2005")
    with pytest.raises(ValueError):
        rolling_backtest([])


def test_henry_hub_example_reproduces_published_numbers():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from henry_hub_winter import (build, MARKET, POLAR_VORTEX, LNG_UPSIDE,
                                  MONTHS, hedge_for_budget, winter_backtest)

    c = compare(build(), [MARKET, POLAR_VORTEX, LNG_UPSIDE], target="price")
    winter = c[MARKET.name].window("price", MONTHS[0], MONTHS[-1])
    assert c[MARKET.name].at("price", 202701).mean() == pytest.approx(3.22, abs=0.03)
    assert winter.mean() == pytest.approx(2.70, abs=0.03)
    assert np.mean(winter < 2.974) == pytest.approx(0.70, abs=0.02)
    assert c[LNG_UPSIDE.name].window("price").mean() == pytest.approx(3.01, abs=0.03)
    assert not c[POLAR_VORTEX.name].support.clean
    assert hedge_for_budget(c[MARKET.name])[0] == pytest.approx(0.65)

    bt = winter_backtest()
    assert len(bt.scores) == 40
    assert bt.coverage == pytest.approx(0.875, abs=0.03)
    assert bt.mape == pytest.approx(0.223, abs=0.01)
