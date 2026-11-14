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
