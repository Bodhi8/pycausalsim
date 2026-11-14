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
"""Known-truth tests for CausalSimulator.simulate_intervention.

Each test builds data from a structural model whose true effect is known, then
checks that the estimate recovers it. These guard against the estimator
returning a plausible-looking number that is wrong.
"""


import numpy as np
import pandas as pd
import pytest

from pycausalsim import CausalSimulator

N = 1500
BOOT = 40
METHODS = ["monte_carlo", "scm", "backdoor"]


def _estimate(df, graph, variable, shift=1.0, method="monte_carlo", seed=0, **kw):
    sim = CausalSimulator(df, target="y", random_state=seed, **kw)
    sim.set_graph(graph)
    return sim.simulate_intervention(variable, df[variable].mean() + shift,
                                     method=method, n_bootstrap=BOOT)


def _direct(seed=1, n=N):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(0, 1, n)}), {"y": ["x"]}


def _mediated(seed=2, n=N):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = 1.5 * x + rng.normal(0, 1, n)
    y = 2.0 * m + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "m": m, "y": y}), {"m": ["x"], "y": ["m"]}


def _confounded(seed=3, n=N):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, 1, n)
    y = 1.0 * x + 3.0 * z + rng.normal(0, 1, n)
    return pd.DataFrame({"z": z, "x": x, "y": y}), {"x": ["z"], "y": ["x", "z"]}


@pytest.mark.parametrize("method", METHODS)
def test_direct_effect_is_recovered(method):
    df, graph = _direct()
    e = _estimate(df, graph, "x", method=method)
    assert e.point_estimate == pytest.approx(2.0, abs=0.25)
    assert e.ci_lower <= 2.0 <= e.ci_upper


@pytest.mark.parametrize("method", METHODS)
def test_effect_through_a_mediator_is_counted(method):
    """x -> m -> y with no direct edge: the whole effect (1.5 * 2 = 3) is mediated."""
    df, graph = _mediated()
    e = _estimate(df, graph, "x", method=method)
    assert e.point_estimate == pytest.approx(3.0, abs=0.45)
    assert e.ci_lower > 1.5


@pytest.mark.parametrize("method", METHODS)
def test_confounding_is_adjusted_when_the_graph_names_it(method):
    """The unadjusted slope of y on x is 2.5; the causal effect is 1.0."""
    df, graph = _confounded()
    naive = np.polyfit(df["x"], df["y"], 1)[0]
    assert naive > 2.0
    e = _estimate(df, graph, "x", method=method)
    assert e.point_estimate == pytest.approx(1.0, abs=0.25)


@pytest.mark.parametrize("method", ["monte_carlo", "scm"])
def test_variable_with_no_path_to_target_has_zero_effect(method):
    df, graph = _confounded()
    rng = np.random.default_rng(9)
    df["w"] = rng.normal(0, 1, len(df))  # in the data, but causes nothing
    e = _estimate(df, graph, "w", method=method)
    assert e.point_estimate == 0.0
    assert e.p_value == 1.0


def test_backdoor_does_not_adjust_for_mediators():
    df, graph = _mediated()
    sim = CausalSimulator(df, target="y", random_state=0)
    sim.set_graph(graph)
    assert sim._get_adjustment_set("x") == []
    assert sim._get_adjustment_set("m") == ["x"]


def test_user_confounder_downstream_of_treatment_is_dropped_with_a_warning():
    df, graph = _mediated()
    sim = CausalSimulator(df, target="y", confounders=["m"], random_state=0)
    sim.set_graph(graph)
    with pytest.warns(UserWarning, match="downstream"):
        e = sim.simulate_intervention("x", df["x"].mean() + 1, method="backdoor", n_bootstrap=BOOT)
    assert e.point_estimate == pytest.approx(3.0, abs=0.45)


@pytest.mark.parametrize("method", ["monte_carlo", "backdoor"])
def test_null_effect_is_rarely_called_significant(method):
    """With no true effect, p < 0.05 should be the exception. The old p-value
    divided the bootstrap spread by sqrt(replicates) and rejected almost always."""
    rejections = 0
    trials = 12
    for s in range(trials):
        rng = np.random.default_rng(200 + s)
        df = pd.DataFrame({"x": rng.normal(0, 1, 300), "y": rng.normal(0, 1, 300)})
        e = _estimate(df, {"y": ["x"]}, "x", method=method, seed=s)
        rejections += e.p_value < 0.05
    assert rejections <= 3


def test_more_replicates_do_not_shrink_the_p_value():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(0, 1, 300), "y": rng.normal(0, 1, 300)})
    sim = CausalSimulator(df, target="y", random_state=0)
    sim.set_graph({"y": ["x"]})
    v = df["x"].mean() + 1
    few = sim.simulate_intervention("x", v, method="backdoor", n_bootstrap=30)
    many = sim.simulate_intervention("x", v, method="backdoor", n_bootstrap=300)
    assert many.p_value > 0.01
    assert many.p_value == pytest.approx(few.p_value, abs=0.35)


def test_same_random_state_gives_the_same_answer():
    df, graph = _direct(n=400)
    a = _estimate(df, graph, "x", seed=7)
    np.random.seed(123)  # unrelated use of NumPy's global state must not matter
    b = _estimate(df, graph, "x", seed=7)
    assert a.point_estimate == b.point_estimate
    assert (a.ci_lower, a.ci_upper) == (b.ci_lower, b.ci_upper)


def test_interval_reflects_sample_size():
    small, graph = _direct(seed=11, n=200)
    large, _ = _direct(seed=11, n=3000)
    w_small = _estimate(small, graph, "x", method="backdoor")
    w_large = _estimate(large, graph, "x", method="backdoor")
    assert (w_small.ci_upper - w_small.ci_lower) > 2 * (w_large.ci_upper - w_large.ci_lower)


def test_value_outside_observed_range_warns():
    df, graph = _direct(n=400)
    sim = CausalSimulator(df, target="y", random_state=0)
    sim.set_graph(graph)
    with pytest.warns(UserWarning, match="outside its observed range"):
        sim.simulate_intervention("x", df["x"].max() + 5, n_bootstrap=5)


def test_bad_inputs_raise_clear_errors():
    df, graph = _direct(n=200)
    with pytest.raises(TypeError, match="DataFrame"):
        CausalSimulator([[1, 2], [3, 4]], target="y")
    sim = CausalSimulator(df, target="y", random_state=0)
    with pytest.raises(RuntimeError, match="discover_graph"):
        sim.simulate_intervention("x", 0.0)
    sim.set_graph(graph)
    with pytest.raises(ValueError, match="not in data"):
        sim.simulate_intervention("nope", 0.0)
    with pytest.raises(ValueError, match="target variable"):
        sim.simulate_intervention("y", 0.0)
    with pytest.raises(ValueError, match="Unknown method"):
        sim.simulate_intervention("x", 0.0, method="magic")
    with pytest.raises(ValueError, match="n_bootstrap"):
        sim.simulate_intervention("x", 0.0, n_bootstrap=1)


# ---------------------------------------------------------------------------
# rank_drivers, validate, optimize_policy
# ---------------------------------------------------------------------------

def _drivers_data(seed=4, n=N):
    """y = 3a + m, with b -> m (so b's effect of 1 is fully mediated) and c inert."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)
    m = b + rng.normal(0, 0.5, n)
    y = 3.0 * a + m + rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "m": m, "y": y}), {"m": ["b"], "y": ["a", "m"]}


def test_rank_drivers_orders_by_true_effect_and_counts_mediation():
    df, graph = _drivers_data()
    sim = CausalSimulator(df, target="y", random_state=0).set_graph(graph)
    ranking = dict(sim.rank_drivers(n_bootstrap=6))
    assert list(ranking)[0] == "a"
    assert list(ranking)[-1] == "c"
    assert ranking["a"] == pytest.approx(3.0, abs=0.5)
    assert ranking["b"] == pytest.approx(1.0, abs=0.35)   # reaches y only through m
    assert ranking["c"] == 0.0


def test_rank_drivers_rejects_methods_it_does_not_implement():
    df, graph = _direct(n=200)
    sim = CausalSimulator(df, target="y", random_state=0).set_graph(graph)
    with pytest.raises(ValueError, match="Only 'intervention'"):
        sim.rank_drivers(method="shap")


def _partial_r2(y, x, controls):
    """Share of the residual variance of y (after controls) explained by x."""
    def resid(v, cols):
        X = np.column_stack([np.ones(len(v))] + cols)
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    ry, rx = resid(y, controls), resid(x, controls)
    return float(np.corrcoef(ry, rx)[0, 1] ** 2)


def test_omitted_variable_bias_formula_is_exact():
    """The bias from leaving out a confounder z equals
    se * sqrt(df) * sqrt(R2_y~z|d * R2_d~z / (1 - R2_d~z)), the identity the
    confounding bounds and the robustness value are built on."""
    rng = np.random.default_rng(6)
    n = 800
    z = rng.normal(0, 1, n)
    d = 0.6 * z + rng.normal(0, 1, n)
    y = 1.0 * d + 0.8 * z + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y})
    sim = CausalSimulator(df, target="y", random_state=0).set_graph({"y": ["d"]})
    restricted = sim._linear_fit(df, "d", [])
    full = sim._linear_fit(df, "d", [], extra=z)
    r2_dz = _partial_r2(d, z, [])
    r2_yz = _partial_r2(y, z, [d])
    predicted_bias = restricted["se"] * np.sqrt(restricted["df"]) * np.sqrt(r2_yz * r2_dz / (1 - r2_dz))
    assert abs(restricted["coef"] - full["coef"]) == pytest.approx(predicted_bias, rel=1e-6)


def test_confounding_bounds_contain_the_truth_for_a_confounder_of_that_strength():
    """A hidden confounder explains under 30% of both sides; the 0.3 bounds must
    contain the true effect even though the naive estimate is biased."""
    rng = np.random.default_rng(8)
    n = 4000
    u = rng.normal(0, 1, n)                      # never shown to the simulator
    x = 0.5 * u + rng.normal(0, 1, n)
    y = 1.0 * x + 0.6 * u + rng.normal(0, 1, n)
    assert _partial_r2(x, u, []) < 0.3 and _partial_r2(y, u, [x]) < 0.3
    df = pd.DataFrame({"x": x, "y": y})
    sim = CausalSimulator(df, target="y", random_state=0).set_graph({"y": ["x"]})
    result = sim.validate("x", n_bootstrap=6)
    truth = 1.0 * df["x"].std()
    assert result.original_effect.point_estimate > truth * 1.15      # biased upward
    lower, upper = result.confounding_bounds["0.3"]
    assert lower <= truth <= upper
    narrow_lower, _ = result.confounding_bounds["0.1"]
    assert narrow_lower > lower


def test_robustness_value_separates_strong_from_absent_effects():
    df, graph = _drivers_data()
    sim = CausalSimulator(df, target="y", random_state=0).set_graph(graph)
    strong = sim.validate("a", n_bootstrap=6)
    assert 0.6 < strong.robustness_value <= 1.0
    assert all(check["passed"] for check in strong.refutation_results.values())
    assert strong.placebo_results["placebo_treatment"] > 0.05

    rng = np.random.default_rng(10)
    null = pd.DataFrame({"x": rng.normal(0, 1, N), "y": rng.normal(0, 1, N)})
    weak = CausalSimulator(null, target="y", random_state=0).set_graph({"y": ["x"]}).validate(
        "x", n_bootstrap=6)
    assert weak.robustness_value < 0.1
    assert "Robustness Value" in weak.summary()


def test_optimize_policy_finds_a_known_interior_optimum_through_a_mediator():
    """y = -(m - 2)^2 with x -> m: the best x is 2, and x acts only through m."""
    rng = np.random.default_rng(12)
    n = 1500
    x = rng.uniform(0, 4, n)
    m = x + rng.normal(0, 0.2, n)
    y = -(m - 2.0) ** 2 + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x": x, "m": m, "y": y})
    sim = CausalSimulator(df, target="y", treatment_vars=["x"], random_state=0).set_graph(
        {"m": ["x"], "y": ["m"]})
    best = sim.optimize_policy("maximize", grid_points=9)
    assert best.settings["x"] == pytest.approx(2.0, abs=0.3)
    assert best.improvement > 0.8
    worst = sim.optimize_policy("minimize", grid_points=9)
    assert min(worst.settings["x"], 4 - worst.settings["x"]) < 0.3
    assert worst.expected_outcome < best.expected_outcome


def test_optimize_policy_validates_its_arguments():
    df, graph = _direct(n=200)
    sim = CausalSimulator(df, target="y", random_state=0).set_graph(graph)
    with pytest.raises(ValueError, match="'max' or 'min'"):
        sim.optimize_policy("best")
    with pytest.raises(ValueError, match="min above max"):
        sim.optimize_policy("maximize", constraints={"x": (2.0, 1.0)})
    assert list(sim.optimize_policy("maximize").settings) == ["x"]


# ---------------------------------------------------------------------------
# Graph validation and discovery end to end
# ---------------------------------------------------------------------------

def test_set_graph_rejects_cycles_self_loops_and_unknown_names():
    df, _ = _mediated(n=200)
    sim = CausalSimulator(df, target="y", random_state=0)
    with pytest.raises(ValueError, match="cycle"):
        sim.set_graph({"x": ["m"], "m": ["x"], "y": ["m"]})
    with pytest.raises(ValueError, match="cycle"):
        sim.set_graph({"x": ["y"], "m": ["x"], "y": ["m"]})
    with pytest.raises(ValueError, match="self-loop"):
        sim.set_graph({"x": ["x"], "y": ["x"]})
    with pytest.raises(ValueError, match="not a column"):
        sim.set_graph({"y": ["price"]})
    with pytest.raises(TypeError, match="list"):
        sim.set_graph({"y": "x"})
    assert sim.scm is None  # nothing was fitted from a rejected graph


def test_two_way_edges_are_oriented_from_declared_roles():
    df, _ = _confounded(n=300)
    sim = CausalSimulator(df, target="y", treatment_vars=["x"], confounders=["z"], random_state=0)
    oriented = sim._orient_two_way_edges({"z": ["x"], "x": ["z"], "y": ["x", "z"]})
    assert oriented["x"] == ["z"] and oriented["z"] == []
    assert sim._oriented_edges["by_role"] == ["z -> x"]

    plain = CausalSimulator(df, target="y", random_state=0)
    with pytest.warns(UserWarning, match="column order"):
        plain._orient_two_way_edges({"z": ["x"], "x": ["z"], "y": ["x", "z"]})


def _shop(seed=42, n=5000):
    """The data of examples/complete_example.py: load time lowers conversion by
    5 points per second, and traffic source confounds the two."""
    rng = np.random.default_rng(seed)
    source = rng.choice([0, 1], n, p=[0.6, 0.4])
    load = 3.0 + 0.5 * source + rng.exponential(0.5, n)
    price = 20 + rng.normal(0, 5, n)
    device = rng.choice([0, 1, 2], n)
    prob = np.clip(0.40 - 0.05 * load + 0.03 * source - 0.006 * price, 0.01, 0.99)
    return pd.DataFrame({
        "traffic_source": source, "page_load_time": load, "price": price,
        "device_type": device, "conversion_rate": rng.binomial(1, prob) * 100.0,
    })


@pytest.mark.parametrize("method", ["pc", "ges"])
def test_discovery_then_simulation_recovers_direction_and_scale(method):
    df = _shop()
    sim = CausalSimulator(df, target="conversion_rate",
                          treatment_vars=["page_load_time", "price"],
                          confounders=["traffic_source", "device_type"], random_state=0)
    sim.discover_graph(method=method)
    sim._check_graph(sim.graph)                                   # a valid DAG
    assert "page_load_time" in sim.graph["conversion_rate"]
    assert "price" in sim.graph["conversion_rate"]
    assert sim.graph["device_type"] == [] and "device_type" not in sim.graph["conversion_rate"]
    faster = sim.simulate_intervention("page_load_time", 3.2, n_bootstrap=8)
    truth = -5.0 * (3.2 - df["page_load_time"].mean())            # about +2.5 points
    assert 0.3 * truth < faster.point_estimate < 1.7 * truth


def test_summary_reports_change_relative_to_the_baseline():
    df, graph = _direct(n=400)
    df["y"] = df["y"] + 10.0                       # baseline of about 10
    e = _estimate(df, graph, "x", method="backdoor")
    assert e.baseline_outcome == pytest.approx(df["y"].mean())
    share = e.point_estimate / e.baseline_outcome
    assert f"{share:+.2%} of the baseline" in e.summary()
