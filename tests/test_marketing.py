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
