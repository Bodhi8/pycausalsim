import numpy as np
import pytest

from pycausalsim.futures import compare_plans, marketing_worlds, media_plan_scm
from pycausalsim.futures.marketing import CHANNELS


@pytest.fixture(scope="module")
def worlds():
    return marketing_worlds(12, seed=3)


def test_frames_have_model_inputs_and_hide_the_confounder(worlds):
    df = worlds.frame(0)
    assert list(df.columns) == ["date", *CHANNELS, "price_index", "promo", "sales"]
    assert len(df) == 156 and (df[CHANNELS] >= 0).all().all()
    assert "launch" not in df.columns


def test_answer_key_matches_simulated_contributions(worlds):
    for i in range(worlds.n):
        t = worlds.truth(i)["channels"]
        for c in CHANNELS:
            eff = worlds.run[f"effect_{c}"][:, i].sum()
            spend = worlds.run[f"spend_{c}"][:, i].sum()
            assert np.isclose(t[c]["roas"], eff / spend)
            # diminishing returns: the next dollar earns less than the average
            assert 0 < t[c]["marginal_roas"] < t[c]["roas"]


def test_search_spend_follows_season(worlds):
    r = [np.corrcoef(worlds.run["spend_paid_search"][:, i], worlds.params["season"][i])[0, 1]
         for i in range(worlds.n)]
    assert np.mean(r) > 0.4


def test_hard_worlds_add_launches_and_wear_out():
    w = marketing_worlds(6, hard=True, seed=4)
    assert all(w.truth(i)["launch_weeks"] > 0 for i in range(w.n))
    assert (w.params["wear_paid_social"] < 1).all()
    assert (w.params["wear_paid_search"] == 1).all()


def test_worlds_are_reproducible():
    a, b = marketing_worlds(3, seed=9), marketing_worlds(3, seed=9)
    assert a.frame(2).equals(b.frame(2))


def test_best_plan_beats_current_mix(worlds):
    df = worlds.frame(1)
    hist = df[CHANNELS].mean().to_numpy()
    best = worlds.best_plan(1, hist.sum(), [(0, 3 * h) for h in hist])
    assert np.isclose(best.sum(), hist.sum())
    assert worlds.steady_sales(1, best) >= worlds.steady_sales(1, hist) - 1e-9


def _curves(beta, hs, alpha, peak):
    return {"media": CHANNELS, "beta": np.array(beta, float), "half_sat": np.array(hs, float),
            "alpha": np.array(alpha, float), "peak": np.array(peak, float), "l_max": 8}


def test_plan_ramps_in_and_converges_to_steady_state():
    beta, hs = [90, 160, 220, 45], [40, 55, 70, 35]
    cur = np.array([30.0, 50.0, 47.0, 22.0])
    plan = np.array([20.0, 50.0, 57.0, 22.0])
    c = _curves([beta], [hs], [[0.2, 0.45, 0.6, 0.3]], [[np.nan, np.nan, 2.0, np.nan]])
    scm = media_plan_scm(c, cur, weeks=20, draws=1)
    run = scm.run(params=scm._plan_params)  # current mix holds: flat at steady state
    steady = lambda x: sum(b * v / (v + h) for b, v, h in zip(beta, x, hs))
    assert np.allclose(run["media_sales"][:, 0], steady(cur))
    res = compare_plans(scm, {"current": cur, "plan": plan})
    # 20 weeks of the plan gain less than 20 weeks at the new steady state (carryover ramp)
    full = 20 * (steady(plan) - steady(cur))
    assert 0 < res.loc["plan", "gain_vs_first"] < full


def test_compare_plans_reports_uncertainty_and_flags_extrapolation():
    rng = np.random.default_rng(0)
    n = 200
    beta = np.column_stack([rng.normal(90, 20, n), rng.normal(160, 30, n), rng.normal(220, 40, n), rng.normal(45, 10, n)])
    hs = np.tile([40, 55, 70, 35], (n, 1))
    alpha = np.tile([0.2, 0.45, 0.6, 0.3], (n, 1))
    peak = np.tile([np.nan, np.nan, 2.0, np.nan], (n, 1))
    cur = np.array([30.0, 50.0, 47.0, 22.0])
    scm = media_plan_scm(_curves(beta, hs, alpha, peak), cur, weeks=13, draws=200,
                         sustained_max=[40, 60, 60, 30])
    res = compare_plans(scm, {"current": cur, "big_video": [15.0, 40.0, 90.0, 4.0]})
    row = res.loc["big_video"]
    assert row.gain_p05 < row.gain_vs_first < row.gain_p95
    assert 0 <= row.prob_better <= 1
    assert row.extrapolated_edges >= 1          # video beyond what was sustained
    assert res.loc["current", "extrapolated_edges"] == 0


# ----------------------------------------------------------------- 0.5
def test_default_worlds_are_unchanged_by_new_options():
    from pycausalsim.futures import marketing_worlds
    a = marketing_worlds(6, weeks=80, hard=True, seed=3)
    b = marketing_worlds(6, weeks=80, hard=True, seed=3, response="hill", competitor=False)
    assert np.array_equal(a.frame(2).to_numpy()[:, 1:].astype(float), b.frame(2).to_numpy()[:, 1:].astype(float))
    assert a.truth(2)["channels"]["display"]["shape"] == 1.0
    # regression anchors: values produced by 0.4.0 with the same seed
    assert a.frame(2)["sales"].iloc[0] == pytest.approx(782.97)
    assert a.truth(2)["channels"]["display"]["roas"] == pytest.approx(1.5713368397)


def test_s_curve_and_competitor_worlds():
    from pycausalsim.futures import marketing_worlds
    w = marketing_worlds(8, weeks=104, seed=2, response="s_curve", competitor=True)
    t = w.truth(0)["channels"]
    assert all(1.4 <= t[c]["shape"] <= 2.4 for c in t)
    assert w.params["competitor"].sum() > 0 and "competitor" not in w.frame(0).columns
    hist = w.frame(0)[list(t)].mean().to_numpy()
    bounds = [(0.0, 3 * h) for h in hist]
    best = w.best_plan(0, hist.sum(), bounds)
    assert best.sum() == pytest.approx(hist.sum()) and w.steady_sales(0, best) >= w.steady_sales(0, hist) - 1e-9
    with pytest.raises(ValueError):
        marketing_worlds(2, response="linear")


def test_lift_test_measures_the_true_intervention():
    from pycausalsim.futures import lift_test, marketing_worlds
    w = marketing_worlds(6, weeks=104, seed=4)
    r = lift_test(w, 1, "paid_search", weeks=8)
    assert r["spend_changed"] < 0 and r["true"] > 0 and r["se"] > 0
    assert abs(r["measured"] - r["true"]) < 5 * r["se"]
    # a holdout measures the average return of the spend removed; a small increase the marginal one
    up = lift_test(w, 1, "paid_search", weeks=8, change=0.1)
    assert up["true"] < r["true"]
    assert lift_test(w, 1, "paid_search", r2=0.0)["se"] > r["se"]
    assert lift_test(w, 1, "paid_search", weeks=16)["se"] < r["se"]


GRAPH = {
    "outcome": "revenue",
    "nodes": [
        {"id": "revenue", "type": "Outcome"},
        {"id": "search", "type": "Channel", "stage": "lower", "column": "search_spend"},
        {"id": "video", "type": "Channel", "stage": "upper", "column": "video_spend"},
        {"id": "seasonality", "type": "Driver", "column": None, "modeled": True, "kind": "driver"},
        {"id": "launch", "type": "Driver", "column": "launch", "modeled": False, "kind": "event"},
    ],
    "edges": [
        {"source": "search", "target": "revenue", "relation": "drives", "status": "confirmed"},
        {"source": "video", "target": "revenue", "relation": "drives", "status": "confirmed"},
        {"source": "seasonality", "target": "search", "relation": "sets_spend", "status": "confirmed"},
        {"source": "seasonality", "target": "revenue", "relation": "drives", "status": "confirmed"},
        {"source": "launch", "target": "search", "relation": "sets_spend", "status": "confirmed"},
        {"source": "launch", "target": "revenue", "relation": "drives", "status": "confirmed"},
        {"source": "video", "target": "search", "relation": "mediates", "status": "confirmed"},
        {"source": "search", "target": "video", "relation": "mediates", "status": "proposed"},
    ],
}


def test_worlds_from_graph_follow_the_declared_structure():
    from pycausalsim.futures import worlds_from_graph
    w = worlds_from_graph(GRAPH, 10, weeks=104, seed=1)
    df = w.frame(0)
    assert list(df.columns) == ["date", "search_spend", "video_spend", "launch", "revenue"]
    assert w.channels == ["search_spend", "video_spend"] and w.hidden == []
    t = w.truth(0)["channels"]
    assert t["video_spend"]["peak"] is not None and t["search_spend"]["peak"] is None
    # video feeds search, so its total return exceeds its direct return; search feeds nothing
    assert all(w.truth(i)["channels"]["video_spend"]["total_roas"] > w.truth(i)["channels"]["video_spend"]["roas"]
               for i in range(10))
    assert t["search_spend"]["total_roas"] == pytest.approx(t["search_spend"]["roas"])
    # launches raise search spend
    on = df["launch"] == 1
    assert df.loc[on, "search_spend"].mean() > df.loc[~on, "search_spend"].mean()
    hidden = worlds_from_graph(GRAPH, 10, weeks=104, seed=1, hide=["launch"])
    assert "launch" not in hidden.frame(0).columns and hidden.hidden == ["launch"]
    assert np.allclose(hidden.frame(0)["revenue"], df["revenue"])
    cyc = {**GRAPH, "edges": GRAPH["edges"] + [
        {"source": "search", "target": "video", "relation": "mediates", "status": "confirmed"}]}
    with pytest.raises(ValueError):
        worlds_from_graph(cyc, 2)


def test_grade_and_leaderboard_score_any_estimator():
    from pycausalsim.futures import grade, leaderboard, marketing_worlds, ols_estimator
    w = marketing_worlds(6, weeks=104, seed=5)

    def oracle(df, info):
        t = w.truth(info["world"])["channels"]
        return {"roas": {c: (t[c]["roas"], t[c]["roas"] - 0.1, t[c]["roas"] + 0.1) for c in info["media"]},
                "plan": w.best_plan(info["world"], info["total"], [(0.0, 3 * h) for h in info["current"]])}
    s = grade(oracle, w).attrs["summary"]
    assert s["coverage_90"] == 1.0 and s["abs_error"] < 1e-9
    assert s["plan_capture"] == pytest.approx(1.0) and s["plan_lost_money"] == 0.0
    lb = leaderboard({"oracle": oracle, "ols": ols_estimator()}, w, ids=range(3))
    assert lb.index[0] == "oracle" and lb.loc["ols", "worlds"] == 3
    with pytest.raises(ValueError):
        grade(lambda df, info: {"roas": {c: (1, 0, 2) for c in info["media"]}, "plan": info["current"] * 2}, w, ids=[0])


def test_media_plan_scm_uses_curve_shape():
    from pycausalsim.futures import compare_plans, media_plan_scm
    base = {"media": ["a", "b"], "beta": np.array([[100.0, 100.0]]), "half_sat": np.array([[10.0, 10.0]]),
            "alpha": np.array([[0.3, 0.3]]), "peak": np.array([[np.nan, np.nan]]), "l_max": 8}
    cur = np.array([4.0, 4.0])                             # well below half-saturation
    flat = compare_plans(media_plan_scm(base, cur, weeks=20), {"current": cur, "tilt": [8.0, 0.0]})
    s = compare_plans(media_plan_scm({**base, "shape": np.array([[3.0, 3.0]])}, cur, weeks=20),
                      {"current": cur, "tilt": [8.0, 0.0]})
    assert flat.loc["tilt", "gain_vs_first"] < 0          # concave: an even split is best
    assert s.loc["tilt", "gain_vs_first"] > 0             # S-shaped: concentrating clears the threshold
