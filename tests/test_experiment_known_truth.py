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
Experiment analysis tested against data whose true effect is known.

Besides checking that estimates land near the truth, these tests check the
error rates: how often a method declares an effect when there is none, and how
often its confidence interval contains the true effect.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pycausalsim import ExperimentAnalysis

METHODS = ["difference", "ols", "ipw", "dr", "matching"]
ADJUSTED = ["ols", "ipw", "dr", "matching"]


def _randomised(seed, n=1000, effect=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    y = effect * t + 0.8 * x + rng.normal(size=n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def _confounded(seed, n=2000):
    """Units with high x are treated more often and have higher outcomes. The
    effect is 1 + x, so the average effect is 1 and the effect on the treated
    is larger."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-1.2 * x))).astype(int)
    y = (1 + x) * t + 2 * x + rng.normal(size=n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def _nonlinear(seed, n=3000):
    """The outcome is not linear in x, so a linear outcome model is wrong; the
    treatment probability is logistic in x, so the propensity model is right."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    y = 1.0 * t + 1.5 * np.exp(x / 1.5) + rng.normal(size=n)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def _run(make, method, runs, truth, **kwargs):
    estimates, covered, significant = [], 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for seed in range(runs):
            effect = ExperimentAnalysis(make(seed, **kwargs), "t", "y", ["x"]).estimate_effect(
                method)
            estimates.append(effect.estimate)
            covered += effect.ci_lower <= truth <= effect.ci_upper
            significant += effect.p_value < 0.05
    return np.mean(estimates), covered / runs, significant / runs


# ---------------------------------------------------------------------------
# Error rates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_no_effect_is_rarely_declared_significant(method):
    """With no true effect, about 5% of experiments should come out significant."""
    _, _, false_positive_rate = _run(_randomised, method, runs=300, truth=0.0)
    assert false_positive_rate < 0.09


@pytest.mark.parametrize("method", METHODS)
def test_randomised_experiment_recovers_the_effect(method):
    mean, coverage, power = _run(_randomised, method, runs=150, truth=0.3, effect=0.3)
    assert mean == pytest.approx(0.3, abs=0.02)
    assert coverage > 0.90
    assert power > 0.9


@pytest.mark.parametrize("method", ADJUSTED)
def test_confounded_data_recovers_the_average_effect(method):
    mean, coverage, _ = _run(_confounded, method, runs=150, truth=1.0)
    assert mean == pytest.approx(1.0, abs=0.04)
    assert coverage > 0.89


@pytest.mark.parametrize("method", ["ipw", "dr", "matching"])
def test_wrong_outcome_model_is_survivable_when_the_propensity_model_is_right(method):
    mean, coverage, _ = _run(_nonlinear, method, runs=120, truth=1.0)
    assert mean == pytest.approx(1.0, abs=0.04)
    assert coverage > 0.88


def test_doubly_robust_survives_a_wrong_propensity_model():
    """Treatment depends on x squared, which a logistic model in x cannot see,
    but the outcome is linear in x, so the outcome model is right."""
    def make(seed, n=3000):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        t = (rng.random(n) < np.clip(0.15 + 0.3 * x ** 2 + 0.1 * x, 0.05, 0.95)).astype(int)
        y = 1.0 * t + 2 * x + rng.normal(size=n)
        return pd.DataFrame({"t": t, "x": x, "y": y})
    mean, coverage, _ = _run(make, "dr", runs=120, truth=1.0)
    assert mean == pytest.approx(1.0, abs=0.03)
    assert coverage > 0.89


def test_difference_in_means_is_biased_without_randomisation_and_says_so():
    df = _confounded(0)
    exp = ExperimentAnalysis(df, "t", "y", ["x"])
    with pytest.warns(UserWarning, match="not\\s+randomised"):
        naive = exp.estimate_effect("difference")
    assert naive.estimate > 2.5                       # truth is 1
    assert abs(exp.balance_table().loc["x", "std_mean_diff"]) > 0.5


def test_adjusting_for_a_predictive_covariate_narrows_the_interval():
    exp = ExperimentAnalysis(_randomised(1, n=4000, effect=0.3), "t", "y", ["x"])
    plain = exp.estimate_effect("difference")
    for method in ("ols", "dr"):
        assert exp.estimate_effect(method).std_error < 0.85 * plain.std_error


def test_methods_fall_back_to_difference_without_covariates():
    exp = ExperimentAnalysis(_randomised(2, effect=0.3), "t", "y")
    plain = exp.estimate_effect("difference")
    for method in ADJUSTED:
        other = exp.estimate_effect(method)
        assert other.estimate == plain.estimate and other.method == "difference_in_means"


def test_results_are_deterministic():
    df = _confounded(3)
    for method in METHODS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            a = ExperimentAnalysis(df, "t", "y", ["x"]).estimate_effect(method)
            b = ExperimentAnalysis(df, "t", "y", ["x"]).estimate_effect(method)
        assert a == b


def test_alpha_sets_the_interval_width():
    exp = ExperimentAnalysis(_randomised(4, effect=0.3), "t", "y", ["x"])
    wide, narrow = exp.estimate_effect("dr", alpha=0.01), exp.estimate_effect("dr", alpha=0.2)
    assert wide.ci_upper - wide.ci_lower > narrow.ci_upper - narrow.ci_lower
    assert "99% CI" in wide.summary()


def test_poor_overlap_is_reported():
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-5 * x))).astype(int)
    df = pd.DataFrame({"t": t, "x": x, "y": t + x + rng.normal(size=n)})
    with pytest.warns(UserWarning, match="almost no comparable"):
        ExperimentAnalysis(df, "t", "y", ["x"]).estimate_effect("ipw")


# ---------------------------------------------------------------------------
# Heterogeneity
# ---------------------------------------------------------------------------

def _segmented(seed, n=8000, gap=1.0):
    rng = np.random.default_rng(seed)
    tenure = rng.uniform(0, 4, n)
    region = rng.choice(["north", "south"], n)
    t = rng.integers(0, 2, n)
    effect = 0.5 + gap * (tenure > 2)
    y = effect * t + 0.3 * tenure + rng.normal(size=n)
    return pd.DataFrame({"t": t, "tenure": tenure, "region": region, "y": y})


def test_segment_effects_match_the_truth():
    exp = ExperimentAnalysis(_segmented(0), "t", "y")
    het = exp.analyze_heterogeneity(["tenure"], n_quantiles=4)[0]
    labels = list(het.segments)
    assert len(labels) == 4 and labels[0].startswith("(-0.") and "data" not in labels[0]
    truth = [0.5, 0.5, 1.5, 1.5]
    for label, expected in zip(labels, truth):
        assert het.segments[label] == pytest.approx(expected, abs=0.15)
        low, high = het.segment_cis[label]
        assert low < het.segments[label] < high
    assert het.interaction_pvalue < 1e-6
    assert sum(het.segment_sizes.values()) == 8000
    assert "_temp_bin" not in exp.data.columns


def test_text_segments_work_and_show_no_difference_when_there_is_none():
    het = ExperimentAnalysis(_segmented(1), "t", "y").analyze_heterogeneity(["region"])[0]
    assert list(het.segments) == ["north", "south"]
    assert het.interaction_pvalue > 0.01


def test_interaction_test_has_the_right_false_positive_rate():
    hits = 0
    runs = 200
    for seed in range(runs):
        df = _segmented(seed, n=1200, gap=0.0)
        het = ExperimentAnalysis(df, "t", "y").analyze_heterogeneity(["tenure"])[0]
        hits += het.interaction_pvalue < 0.05
    assert hits / runs < 0.10


def test_small_segments_are_left_out_with_a_warning():
    df = _segmented(2, n=400)
    df.loc[df.index[:6], "region"] = "east"
    with pytest.warns(UserWarning, match="left out"):
        het = ExperimentAnalysis(df, "t", "y").analyze_heterogeneity(["region"])[0]
    assert "east" not in het.segments


# ---------------------------------------------------------------------------
# Design and projection
# ---------------------------------------------------------------------------

def test_power_analysis_gives_the_power_it_promises():
    pilot = _randomised(0, n=4000, effect=0.2)
    exp = ExperimentAnalysis(pilot, "t", "y")
    n = exp.power_analysis(effect_size=0.2, alpha=0.05, power=0.8)
    # outcome sd within a group is sqrt(0.8^2 + 1) = 1.28
    assert n == pytest.approx(2 * 1.64 * (2.80 / 0.2) ** 2, rel=0.08)

    rng = np.random.default_rng(1)
    detected = 0
    runs = 400
    for _ in range(runs):
        a = 0.2 + 0.8 * rng.normal(size=n) + rng.normal(size=n)
        b = 0.8 * rng.normal(size=n) + rng.normal(size=n)
        df = pd.DataFrame({"t": np.r_[np.ones(n), np.zeros(n)], "y": np.r_[a, b]})
        detected += ExperimentAnalysis(df, "t", "y").estimate_effect().p_value < 0.05
    assert detected / runs == pytest.approx(0.8, abs=0.06)

    assert exp.power_analysis(0.1) > 3.5 * n
    with pytest.raises(ValueError):
        exp.power_analysis(0)


def test_longterm_projection_is_the_measured_effect_under_the_stated_decay():
    exp = ExperimentAnalysis(_randomised(6, effect=0.5), "t", "y")
    measured = exp.estimate_effect()
    out = exp.simulate_longterm(periods=5, decay_rate=0.5)
    expected = measured.estimate * np.exp(-0.5 * np.arange(5))
    assert out["effect"].to_numpy() == pytest.approx(expected)
    assert out["cumulative_effect"].to_numpy() == pytest.approx(np.cumsum(expected))
    assert (out["effect_lower"] < out["effect"]).all()
    assert (out["effect"] < out["effect_upper"]).all()
    pd.testing.assert_frame_equal(out, exp.simulate_longterm(periods=5, decay_rate=0.5))
    flat = exp.simulate_longterm(periods=3, decay_rate=0)
    assert flat["effect"].nunique() == 1
    with pytest.raises(ValueError):
        exp.simulate_longterm(periods=0)


def test_sample_ratio_check():
    balanced = _randomised(7, n=4000)
    assert ExperimentAnalysis(balanced, "t", "y").check_sample_ratio() > 0.001
    skewed = pd.concat([balanced[balanced.t == 1], balanced[balanced.t == 0].head(1500)])
    assert ExperimentAnalysis(skewed, "t", "y").check_sample_ratio() < 0.001


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def test_input_checks():
    df = _randomised(8, n=50)
    with pytest.raises(ValueError, match="not found"):
        ExperimentAnalysis(df, "missing", "y")
    with pytest.raises(ValueError, match="not found"):
        ExperimentAnalysis(df, "t", "y", ["missing"])
    with pytest.raises(ValueError, match="coded 0/1"):
        ExperimentAnalysis(df.assign(t=df.t + 1), "t", "y")
    with pytest.raises(ValueError, match="must be numeric"):
        ExperimentAnalysis(df.assign(y="a"), "t", "y")
    with pytest.raises(ValueError, match="at least two"):
        ExperimentAnalysis(df.assign(t=0), "t", "y")
    with pytest.raises(TypeError):
        ExperimentAnalysis(df.to_numpy(), "t", "y")
    with pytest.raises(ValueError, match="Unknown method"):
        ExperimentAnalysis(df, "t", "y").estimate_effect("magic")

    holes = df.copy()
    holes.loc[holes.index[:5], "y"] = np.nan
    with pytest.warns(UserWarning, match="Dropped 5 of 50"):
        exp = ExperimentAnalysis(holes, "t", "y")
    assert len(exp.data) == 45

    flags = df.assign(t=df.t.astype(bool))
    assert ExperimentAnalysis(flags, "t", "y").estimate_effect().estimate == \
        ExperimentAnalysis(df, "t", "y").estimate_effect().estimate
