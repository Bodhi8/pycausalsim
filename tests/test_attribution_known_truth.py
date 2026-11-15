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
Attribution tested against data whose true channel effects are known.

Each test simulates users from a stated model of conversion, so the right answer
is known in advance, and checks that the method recovers it.
"""

import warnings
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
import pytest

from pycausalsim import MarketingAttribution

CHANNELS = ["email", "search", "display", "social"]
TRUE_EFFECT = {"email": 0.10, "search": 0.04, "display": 0.0, "social": -0.03}
INCREMENTAL = ["shapley", "logistic", "linear", "uplift"]


def _additive(n=40000, seed=0):
    """Conversion probability is additive in exposure. `display` does nothing but
    is shown mostly to people who also saw `search`; `social` lowers conversion."""
    rng = np.random.default_rng(seed)
    search = (rng.random(n) < 0.35).astype(int)
    display = (rng.random(n) < 0.15 + 0.5 * search).astype(int)
    email = (rng.random(n) < 0.4).astype(int)
    social = (rng.random(n) < 0.3).astype(int)
    p = 0.05 + 0.10 * email + 0.04 * search + 0.0 * display - 0.03 * social
    return pd.DataFrame({
        "email": email, "search": search, "display": display, "social": social,
        "converted": (rng.random(n) < p).astype(int),
    })


@pytest.fixture(scope="module")
def additive():
    return _additive()


def _fit(df, method, **kwargs):
    attr = MarketingAttribution(df, "converted", CHANNELS, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        attr.fit(method, **kwargs)
    return attr


# ---------------------------------------------------------------------------
# Incremental methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", INCREMENTAL)
def test_channel_with_no_effect_gets_no_credit(additive, method):
    shares = _fit(additive, method).get_attribution()
    assert shares["display"] < 0.04
    assert shares["email"] > shares["search"] > shares["display"]
    assert sum(shares.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("method", INCREMENTAL)
def test_harmful_channel_gets_no_credit_and_a_negative_contribution(additive, method):
    attr = _fit(additive, method)
    assert attr.get_attribution()["social"] == 0.0
    assert attr.get_contributions()["social"] < 0


def test_harmful_channel_is_reported(additive):
    attr = MarketingAttribution(additive, "converted", CHANNELS)
    with pytest.warns(UserWarning, match="negative contribution"):
        attr.fit("shapley")


@pytest.mark.parametrize("method", INCREMENTAL + ["markov", "last_touch"])
def test_incremental_effects_match_the_truth_whatever_the_method(additive, method):
    effects = _fit(additive, method).get_incremental_effects()
    for channel, truth in TRUE_EFFECT.items():
        assert effects[channel] == pytest.approx(truth, abs=0.012)


@pytest.mark.parametrize("method", ["shapley", "logistic", "linear"])
def test_contributions_are_effect_times_reach(additive, method):
    """In an additive model a channel adds (its effect) x (share of users reached)."""
    contributions = _fit(additive, method).get_contributions()
    for channel, truth in TRUE_EFFECT.items():
        expected = truth * additive[channel].mean()
        assert contributions[channel] == pytest.approx(expected, abs=0.006)


def test_baseline_conversions_are_not_credited_to_channels(additive):
    attr = _fit(additive, "shapley")
    assert attr.get_baseline() == pytest.approx(0.05, abs=0.01)
    total = sum(attr.get_contributions().values())
    assert total == pytest.approx(additive["converted"].mean() - attr.get_baseline(), abs=1e-4)


def test_no_marketing_effect_means_nothing_to_attribute():
    rng = np.random.default_rng(3)
    n = 20000
    df = pd.DataFrame({c: (rng.random(n) < 0.4).astype(int) for c in CHANNELS})
    df["converted"] = (rng.random(n) < 0.1).astype(int)
    attr = MarketingAttribution(df, "converted", CHANNELS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        attr.fit("shapley")
    assert all(abs(v) < 0.006 for v in attr.get_contributions().values())
    assert all(abs(v) < 0.012 for v in attr.get_incremental_effects().values())
    assert attr.get_baseline() == pytest.approx(0.1, abs=0.012)


def test_all_negative_contributions_give_zero_shares():
    rng = np.random.default_rng(4)
    n = 20000
    a, b = (rng.random(n) < 0.5).astype(int), (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame({"a": a, "b": b,
                       "converted": (rng.random(n) < 0.3 - 0.1 * a - 0.1 * b).astype(int)})
    attr = MarketingAttribution(df, "converted", ["a", "b"])
    with pytest.warns(UserWarning, match="nothing to attribute"):
        attr.fit("shapley")
    assert attr.get_attribution() == {"a": 0.0, "b": 0.0}


def test_measured_confounder_is_adjusted_for():
    """Search is shown to people with high intent, who convert more anyway. Its
    true effect is 0.03. Ignoring intent overstates it; adjusting recovers it."""
    rng = np.random.default_rng(5)
    n = 60000
    intent = rng.normal(0, 1, n)
    search = (rng.random(n) < 1 / (1 + np.exp(-1.5 * intent))).astype(int)
    email = (rng.random(n) < 0.4).astype(int)
    p = np.clip(0.15 + 0.06 * intent + 0.03 * search + 0.05 * email, 0.01, 0.99)
    df = pd.DataFrame({"search": search, "email": email, "intent": intent,
                       "converted": (rng.random(n) < p).astype(int)})

    naive = MarketingAttribution(df, "converted", ["search", "email"]).fit("linear")
    adjusted = MarketingAttribution(
        df, "converted", ["search", "email"], covariate_cols=["intent"]).fit("linear")
    assert naive.get_incremental_effects()["search"] > 0.07
    assert adjusted.get_incremental_effects()["search"] == pytest.approx(0.03, abs=0.012)
    assert adjusted.get_incremental_effects()["email"] == pytest.approx(0.05, abs=0.012)
    assert adjusted.get_attribution()["search"] < naive.get_attribution()["search"]


def _synergy(n=80000, seed=6):
    rng = np.random.default_rng(seed)
    E = (rng.random((n, 3)) < [0.5, 0.4, 0.3]).astype(float)

    def prob(e):
        z = -2.5 + 0.8 * e[:, 0] + 0.3 * e[:, 1] + 0.0 * e[:, 2] + 1.2 * e[:, 0] * e[:, 1]
        return 1 / (1 + np.exp(-z))
    df = pd.DataFrame(E.astype(int), columns=["a", "b", "c"])
    df["converted"] = (rng.random(n) < prob(E)).astype(int)
    return df, E, prob


def _true_shapley(E, prob):
    k = E.shape[1]

    def v(S):
        mask = np.zeros(k)
        mask[list(S)] = 1
        return prob(E * mask).mean()
    phi = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for size in range(k):
            w = factorial(size) * factorial(k - size - 1) / factorial(k)
            for S in combinations(others, size):
                phi[i] += w * (v(set(S) | {i}) - v(set(S)))
    return phi


def test_shapley_matches_the_true_values_when_channels_reinforce_each_other():
    df, E, prob = _synergy()
    truth = _true_shapley(E, prob)
    attr = MarketingAttribution(df, "converted", ["a", "b", "c"])
    attr.fit("shapley", interactions=True)
    estimate = np.array([attr.get_contributions()[c] for c in "abc"])
    assert estimate == pytest.approx(truth, abs=0.006)
    assert abs(estimate[2]) < 0.004                      # c does nothing
    # the synergy is shared: b earns more than its own small effect would give
    assert estimate[1] > 0.3 * estimate[0]


def test_sampled_shapley_agrees_with_exact(additive):
    exact = _fit(additive, "shapley").get_contributions()
    sampled = _fit(additive, "shapley", exact=False, n_samples=400).get_contributions()
    for channel in CHANNELS:
        assert sampled[channel] == pytest.approx(exact[channel], abs=0.002)


def test_identical_channels_get_identical_credit():
    rng = np.random.default_rng(7)
    n = 30000
    a = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame({"a": a, "b": 1 - a})
    df["converted"] = (rng.random(n) < 0.2).astype(int)
    both = pd.concat([df, df.rename(columns={"a": "b", "b": "a"})], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        contributions = MarketingAttribution(
            both, "converted", ["a", "b"]).fit("shapley").get_contributions()
    assert contributions["a"] == pytest.approx(contributions["b"], abs=1e-6)


def test_revenue_outcome_is_modelled_linearly():
    rng = np.random.default_rng(8)
    n = 20000
    a, b = (rng.random(n) < 0.5).astype(int), (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({"a": a, "b": b, "revenue": 20 + 5 * a + 2 * b + rng.normal(0, 4, n)})
    attr = MarketingAttribution(df, "revenue", ["a", "b"]).fit("shapley")
    assert attr.get_incremental_effects()["a"] == pytest.approx(5, abs=0.25)
    assert attr.get_incremental_effects()["b"] == pytest.approx(2, abs=0.25)
    assert attr.get_baseline() == pytest.approx(20, abs=0.3)


def test_confidence_intervals_cover_the_truth():
    covered = {c: 0 for c in CHANNELS}
    runs = 25
    for seed in range(runs):
        df = _additive(n=3000, seed=100 + seed)
        table = MarketingAttribution(
            df, "converted", CHANNELS, random_state=seed).incremental_effects_table(80)
        for c in CHANNELS:
            covered[c] += table.loc[c, "ci_lower"] <= TRUE_EFFECT[c] <= table.loc[c, "ci_upper"]
    assert sum(covered.values()) / (runs * len(CHANNELS)) > 0.85


# ---------------------------------------------------------------------------
# Path heuristics
# ---------------------------------------------------------------------------

def _simulate_chain(n=60000, seed=9):
    """Journeys from a known chain. From start: A 0.5, B 0.3, C 0.2.
    A -> B 0.4, convert 0.2, leave 0.4.   B -> convert 0.5, leave 0.5.
    C -> A 0.5, leave 0.5 (C never converts directly)."""
    rng = np.random.default_rng(seed)
    step = {
        "start": (["A", "B", "C"], [0.5, 0.3, 0.2]),
        "A": (["B", "conv", "null"], [0.4, 0.2, 0.4]),
        "B": (["conv", "null"], [0.5, 0.5]),
        "C": (["A", "null"], [0.5, 0.5]),
    }
    paths, converted = [], []
    for _ in range(n):
        state, path = "start", []
        while state not in ("conv", "null"):
            options, probs = step[state]
            state = options[rng.choice(len(options), p=probs)]
            if state not in ("conv", "null"):
                path.append(state)
        paths.append(" > ".join(path))
        converted.append(int(state == "conv"))
    return pd.DataFrame({"path": paths, "converted": converted})


def test_markov_removal_effects_match_the_chain_that_generated_the_journeys():
    # Worked by hand from the chain above:
    #   P(convert | B) = 0.5, P(convert | A) = 0.2 + 0.4 * 0.5 = 0.4, P(convert | C) = 0.2
    #   overall 0.5 * 0.4 + 0.3 * 0.5 + 0.2 * 0.2 = 0.39
    #   without A: 0.3 * 0.5 = 0.15            -> removal effect 1 - 0.15 / 0.39
    #   without B: 0.5 * 0.2 + 0.2 * 0.5 * 0.2 -> removal effect 1 - 0.12 / 0.39
    #   without C: 0.5 * 0.4 + 0.3 * 0.5       -> removal effect 1 - 0.35 / 0.39
    truth = {"A": 1 - 0.15 / 0.39, "B": 1 - 0.12 / 0.39, "C": 1 - 0.35 / 0.39}
    df = _simulate_chain()
    attr = MarketingAttribution(df, "converted", path_col="path").fit("markov")
    assert attr.touchpoint_cols == ["A", "B", "C"]
    assert attr._markov_conversion_rate == pytest.approx(0.39, abs=0.01)
    for channel, value in truth.items():
        assert attr._removal_effects[channel] == pytest.approx(value, abs=0.02)
    shares = attr.get_attribution()
    total = sum(truth.values())
    for channel, value in truth.items():
        assert shares[channel] == pytest.approx(value / total, abs=0.02)


def test_first_and_last_touch_follow_the_recorded_order():
    df = pd.DataFrame({
        "path": ["A > B", "A > B", "B > A", "C", ["A", "C"], "B"],
        "converted": [1, 1, 1, 0, 1, 0],
    })
    attr = MarketingAttribution(df, "converted", path_col="path")
    assert attr.fit("first_touch").get_attribution() == {"A": 0.75, "B": 0.25, "C": 0.0}
    assert attr.fit("last_touch").get_attribution() == {"A": 0.25, "B": 0.5, "C": 0.25}


def test_paths_supply_the_exposure_flags_for_the_incremental_methods():
    df = _simulate_chain(n=5000)
    attr = MarketingAttribution(df, "converted", path_col="path")
    assert set(attr.data["A"].unique()) == {0, 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        attr.fit("shapley")
    assert set(attr.get_contributions()) == {"A", "B", "C"}


@pytest.mark.parametrize("method", ["markov", "first_touch", "last_touch"])
def test_path_heuristics_say_when_they_assume_an_order(additive, method):
    attr = MarketingAttribution(additive.head(2000), "converted", CHANNELS)
    with pytest.warns(UserWarning, match="order of contact"):
        attr.fit(method)
    with pytest.raises(RuntimeError, match="path heuristic"):
        attr.get_contributions()
    assert sum(attr.get_attribution().values()) == pytest.approx(1.0)


def test_unknown_channel_in_a_path_is_rejected():
    df = pd.DataFrame({"a": [1, 0], "path": ["a", "zzz"], "converted": [1, 0]})
    with pytest.raises(ValueError, match="not touchpoints"):
        MarketingAttribution(df, "converted", ["a"], path_col="path")


# ---------------------------------------------------------------------------
# Budget, comparison, housekeeping
# ---------------------------------------------------------------------------

def test_budget_always_sums_to_the_total(additive):
    attr = _fit(additive, "shapley")
    plain = attr.optimize_budget(100)
    assert sum(plain.values()) == pytest.approx(100)
    assert plain["social"] == 0.0 and plain["email"] > plain["search"]

    capped = attr.optimize_budget(100, constraints={"email": (0, 20), "social": (10, 15)})
    assert sum(capped.values()) == pytest.approx(100)
    assert capped["email"] == pytest.approx(20) and capped["social"] == pytest.approx(10)

    costly = attr.optimize_budget(100, cost_per_touchpoint={"email": 10.0})
    assert costly["email"] < plain["email"]


def test_budget_rejects_impossible_constraints(additive):
    attr = _fit(additive, "shapley")
    with pytest.raises(ValueError, match="cannot be met"):
        attr.optimize_budget(100, constraints={"email": (0, 10), "search": (0, 10),
                                               "display": (0, 10), "social": (0, 10)})
    with pytest.raises(ValueError, match="cannot be met"):
        attr.optimize_budget(100, constraints={"email": (80, 90), "search": (30, 40)})
    with pytest.raises(ValueError, match="Unknown channels"):
        attr.optimize_budget(100, constraints={"tv": (0, 10)})


def test_compare_methods_leaves_the_fit_alone(additive):
    attr = _fit(additive.head(5000), "shapley")
    before = attr.get_attribution()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        table = attr.compare_methods()
    assert list(table.columns) == INCREMENTAL + ["markov", "last_touch", "first_touch"]
    assert attr._method == "shapley" and attr.get_attribution() == before


def test_results_are_reproducible_and_global_random_state_is_untouched(additive):
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    df = additive.head(4000)
    one = MarketingAttribution(df, "converted", CHANNELS, random_state=1)
    two = MarketingAttribution(df, "converted", CHANNELS, random_state=1)
    pd.testing.assert_frame_equal(one.incremental_effects_table(20),
                                  two.incremental_effects_table(20))
    assert np.random.random() == expected


def test_input_checks():
    df = pd.DataFrame({"a": [1, 0, 1, 0], "label": list("wxyz"), "converted": [1, 0, 1, 0]})
    with pytest.raises(ValueError, match="not found"):
        MarketingAttribution(df, "missing", ["a"])
    with pytest.raises(ValueError, match="not found"):
        MarketingAttribution(df, "converted", ["b"])
    with pytest.raises(ValueError, match="must be numeric"):
        MarketingAttribution(df, "converted", ["a"], covariate_cols=["label"])
    with pytest.raises(TypeError):
        MarketingAttribution([[1, 0]], "converted")
    bad = df.assign(converted=[1, None, 0, 1])
    with pytest.raises(ValueError, match="missing values"):
        MarketingAttribution(bad, "converted", ["a"])
    attr = MarketingAttribution(df, "converted", ["a"])
    with pytest.raises(RuntimeError, match="fit"):
        attr.get_attribution()
    with pytest.raises(ValueError, match="Unknown method"):
        attr.fit("magic")
    assert MarketingAttribution(df, "converted").touchpoint_cols == ["a"]
