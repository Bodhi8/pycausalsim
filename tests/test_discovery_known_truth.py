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
"""Known-truth tests for causal discovery.

Data are generated from a graph that is known, and each method is checked
against what it can in principle recover: the equivalence class for the
constraint- and score-based methods, the full graph for LiNGAM and NOTEARS
under their own assumptions.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pycausalsim.discovery import (
    CausalDiscovery, _PDAG, _cpdag_from_dag, is_dag, undirected_edges,
)

CLASS_METHODS = ["pc", "ges", "hybrid"]


def _noise(rng, n, kind):
    return rng.normal(0, 1, n) if kind == "gauss" else rng.uniform(-1.73, 1.73, n)


def _make(name, n=3000, seed=0, kind="gauss", columns=None):
    rng = np.random.default_rng(seed)
    e = lambda: _noise(rng, n, kind)  # noqa: E731
    if name == "chain":            # a -> b -> c
        a = e(); b = 0.8 * a + e(); c = 0.8 * b + e()                      # noqa: E702
        df, truth = pd.DataFrame({"a": a, "b": b, "c": c}), {("a", "b"), ("b", "c")}
    elif name == "fork":           # b <- a -> c
        a = e(); b = 0.8 * a + e(); c = 0.8 * a + e()                      # noqa: E702
        df, truth = pd.DataFrame({"a": a, "b": b, "c": c}), {("a", "b"), ("a", "c")}
    elif name == "collider":       # a -> c <- b
        a = e(); b = e(); c = 0.8 * a + 0.8 * b + e()                      # noqa: E702
        df, truth = pd.DataFrame({"a": a, "b": b, "c": c}), {("a", "c"), ("b", "c")}
    elif name == "diamond":        # a -> b -> d, a -> c -> d
        a = e(); b = 0.8 * a + e(); c = 0.8 * a + e(); d = 0.7 * b + 0.7 * c + e()   # noqa: E702
        df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
        truth = {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}
    else:                          # three unrelated variables
        df, truth = pd.DataFrame({"a": e(), "b": e(), "c": e()}), set()
    return (df[columns] if columns else df), truth


def _edges(graph):
    return {(p, c) for c, ps in graph.items() for p in ps}


def _skeleton(edges):
    return {frozenset(e) for e in edges}


def _directed(graph):
    e = _edges(graph)
    return {x for x in e if (x[1], x[0]) not in e}


def _discover(df, method, **kw):
    return CausalDiscovery(random_state=0).discover(df, method=method, **kw)


# ---------------------------------------------------------------------------
# Equivalence-class methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", CLASS_METHODS)
@pytest.mark.parametrize("shape", ["chain", "fork"])
def test_chain_and_fork_are_left_undirected(method, shape):
    """A chain and a fork imply the same independencies, so no direction can be
    read from the data. Earlier versions oriented these edges anyway."""
    df, truth = _make(shape)
    graph = _discover(df, method)
    assert _skeleton(_edges(graph)) == _skeleton(truth)
    assert _directed(graph) == set()
    assert len(undirected_edges(graph)) == 2


@pytest.mark.parametrize("method", CLASS_METHODS)
def test_collider_is_oriented(method):
    df, truth = _make("collider")
    graph = _discover(df, method)
    assert _edges(graph) == truth
    assert undirected_edges(graph) == []


@pytest.mark.parametrize("method", CLASS_METHODS)
def test_diamond_orients_only_the_collider(method):
    df, truth = _make("diamond")
    graph = _discover(df, method)
    assert _skeleton(_edges(graph)) == _skeleton(truth)
    assert _directed(graph) == {("b", "d"), ("c", "d")}
    assert undirected_edges(graph) == [("a", "b"), ("a", "c")]


@pytest.mark.parametrize("method", CLASS_METHODS)
def test_unrelated_variables_get_no_edges(method):
    df, _ = _make("empty", seed=3)
    assert _edges(_discover(df, method, alpha=0.001)) == set()


@pytest.mark.parametrize("method", CLASS_METHODS)
def test_target_is_treated_as_an_outcome(method):
    """With target='c', the undetermined edge b - c is pointed into c; a - b
    stays undetermined because nothing settles it."""
    df, _ = _make("chain")
    graph = _discover(df, method, target="c")
    assert ("b", "c") in _directed(graph)
    assert undirected_edges(graph) == [("a", "b")]


@pytest.mark.parametrize("method", ["pc", "ges"])
def test_result_does_not_depend_on_column_order(method):
    df, _ = _make("diamond", seed=5)
    forward = _edges(_discover(df, method))
    backward = _edges(_discover(df[["d", "c", "b", "a"]], method))
    assert forward == backward


def _random_dag(d, p, rng):
    order = rng.permutation(d)
    names = [f"v{i}" for i in range(d)]
    parents = {n: set() for n in names}
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p:
                parents[names[order[b]]].add(names[order[a]])
    return names, order, parents


def _sample(names, order, parents, n, rng):
    X = {}
    for k in order:
        v = names[k]
        X[v] = sum(rng.uniform(0.7, 1.2) * X[p] for p in sorted(parents[v])) + rng.normal(0, 1, n)
    return pd.DataFrame(X)[names]


def _random_cases(count, d=5, p=0.4, n=5000):
    for s in range(count):
        rng = np.random.default_rng(1000 + s)
        names, order, parents = _random_dag(d, p, rng)
        yield names, parents, _sample(names, order, parents, n, rng)


@pytest.mark.parametrize("method,kw", [("pc", {"alpha": 0.01}), ("ges", {}), ("hybrid", {"alpha": 0.01})])
def test_random_graphs_recover_the_true_equivalence_class(method, kw):
    exact, wrong_marks, wrong_arrows, arrows = 0, 0, 0, 0
    cases = list(_random_cases(12))
    for names, parents, df in cases:
        truth_dag = {(p, c) for c, ps in parents.items() for p in ps}
        truth = _edges(_cpdag_from_dag(names, parents).to_graph())
        got_graph = _discover(df, method, **kw)
        got = _edges(got_graph)
        exact += got == truth
        wrong_marks += len(got ^ truth)
        arrows += len(_directed(got_graph))
        wrong_arrows += len({a for a in _directed(got_graph) if (a[1], a[0]) in truth_dag})
    assert exact >= 8                       # most graphs recovered exactly
    assert wrong_marks / len(cases) <= 1.5  # and the rest nearly so
    assert wrong_arrows <= 0.15 * max(arrows, 1)


def test_ges_matches_the_reference_implementation():
    """Cross-check against causal-learn's GES on the same data. Skipped when
    causal-learn is not installed."""
    pytest.importorskip("causallearn")
    import contextlib
    import io
    from causallearn.search.ScoreBased.GES import ges

    for names, _, df in _random_cases(6, d=6, p=0.35):
        with contextlib.redirect_stdout(io.StringIO()):
            A = ges(df.to_numpy(), score_func="local_score_BIC")["G"].graph
        reference = set()
        for i, j in combinations(range(len(names)), 2):
            if A[i, j] == -1 and A[j, i] == 1:
                reference.add((names[i], names[j]))
            elif A[i, j] == 1 and A[j, i] == -1:
                reference.add((names[j], names[i]))
            elif A[i, j] == -1 and A[j, i] == -1:
                reference |= {(names[i], names[j]), (names[j], names[i])}
        assert _edges(_discover(df, "ges")) == reference


# ---------------------------------------------------------------------------
# LiNGAM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape,columns", [
    ("chain", ["c", "b", "a"]),          # columns in the reverse of the causal order
    ("fork", ["c", "a", "b"]),
    ("diamond", ["d", "b", "c", "a"]),
])
def test_lingam_recovers_direction_from_non_gaussian_noise(shape, columns):
    df, truth = _make(shape, kind="uniform", columns=columns)
    assert _edges(_discover(df, "lingam")) == truth


def test_lingam_warns_when_noise_is_gaussian():
    df, _ = _make("chain", kind="gauss")
    with pytest.warns(UserWarning, match="not identified"):
        graph = _discover(df, "lingam")
    assert is_dag(graph)


# ---------------------------------------------------------------------------
# NOTEARS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["chain", "fork", "collider", "diamond", "empty"])
def test_notears_recovers_graphs_with_equal_noise_scales(shape):
    df, truth = _make(shape)
    graph = _discover(df, "notears")
    assert _edges(graph) == truth
    assert is_dag(graph)


def test_notears_never_returns_a_self_loop_or_cycle():
    """The previous implementation returned a complete graph with every
    variable as its own cause."""
    for seed in range(4):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.normal(0, 1, (300, 4)), columns=list("wxyz"))
        df["z"] = df["w"] * rng.uniform(0.5, 2) + df["z"]
        graph = _discover(df, "notears")
        assert all(v not in parents for v, parents in graph.items())
        assert is_dag(graph)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def test_meek_rule_one_orients_away_from_a_known_arrow():
    g = _PDAG(["a", "b", "c"])
    g.add_edge("a", "b"); g.add_edge("b", "c")           # noqa: E702
    g.arrows.add(("a", "b"))
    g.apply_meek_rules()
    assert g.directed("b", "c")          # otherwise b would be a new collider


def test_equivalence_class_of_a_dag():
    parents = {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}}
    graph = _cpdag_from_dag(["a", "b", "c", "d"], parents).to_graph()
    assert _directed(graph) == {("b", "d"), ("c", "d")}
    extension = CausalDiscovery._consistent_extension(_cpdag_from_dag(["a", "b", "c", "d"], parents))
    assert _edges(_cpdag_from_dag(["a", "b", "c", "d"], extension).to_graph()) == _edges(graph)


def test_graph_helpers():
    assert is_dag({"a": [], "b": ["a"]})
    assert not is_dag({"a": ["b"], "b": ["a"]})
    assert not is_dag({"a": ["a"]})
    assert undirected_edges({"a": ["b"], "b": ["a"], "c": ["a"]}) == [("a", "b")]


def test_correlation_method_still_returns_a_dag():
    df, _ = _make("diamond")
    graph = _discover(df, "correlation", target="d")
    assert set(graph) == set(df.columns) and is_dag(graph)


def test_bad_inputs_raise_clear_errors():
    df, _ = _make("chain", n=200)
    cd = CausalDiscovery()
    with pytest.raises(ValueError, match="Unknown method"):
        cd.discover(df, method="fci")
    with pytest.raises(TypeError, match="DataFrame"):
        cd.discover(df.to_numpy(), method="pc")
    with pytest.raises(ValueError, match="not a column"):
        cd.discover(df, method="pc", target="y")
    with pytest.raises(ValueError, match="numeric"):
        cd.discover(df.assign(label="x"), method="pc")
    with pytest.raises(ValueError, match="missing values"):
        cd.discover(df.assign(a=df["a"].where(df.index > 3)), method="pc")
    with pytest.raises(ValueError, match="Constant"):
        cd.discover(df.assign(k=1.0), method="pc")
    with pytest.raises(ValueError, match="Too few rows"):
        cd.discover(df.head(5), method="pc")
    with pytest.raises(ValueError, match="at least two"):
        cd.discover(df[["a"]], method="pc")


# ---------------------------------------------------------------------------
# Background knowledge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", CLASS_METHODS + ["lingam", "notears"])
def test_tiers_settle_what_the_data_cannot(method):
    """A chain cannot be oriented from Gaussian data, but knowing the order can."""
    import warnings
    df, truth = _make("chain")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)      # lingam's Gaussian warning
        graph = _discover(df, method, tiers={"a": 0, "b": 1, "c": 2})
    assert _edges(graph) == truth
    assert undirected_edges(graph) == []


def test_tiers_must_name_columns():
    df, _ = _make("chain", n=200)
    with pytest.raises(ValueError, match="not columns"):
        _discover(df, "pc", tiers={"z": 0})


@pytest.mark.parametrize("method", ["pc", "ges", "hybrid"])
def test_declared_outcome_is_never_made_a_cause(method):
    """When two paths from s to y cancel, s and y look independent and the data
    alone point to the collider s -> m <- y. Declaring y as the outcome rules
    that out."""
    rng = np.random.default_rng(21)
    n = 6000
    s = rng.normal(0, 1, n)
    m = 1.0 * s + rng.normal(0, 1, n)
    y = -1.0 * m + 1.0 * s + rng.normal(0, 1, n)        # s -> m -> y and s -> y cancel
    df = pd.DataFrame({"s": s, "m": m, "y": y})
    assert abs(np.corrcoef(s, y)[0, 1]) < 0.05
    graph = _discover(df, method, target="y")
    assert "y" not in graph["m"] and "y" not in graph["s"]
    assert "m" in graph["y"]
