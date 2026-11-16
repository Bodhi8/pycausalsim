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
Experiment analysis.

Estimates the average treatment effect (ATE) of a binary treatment, looks for
differences in the effect across segments, and sizes experiments.

Which estimator to use
----------------------
- Randomised experiment: ``difference`` is unbiased. ``ols`` and ``dr`` are also
  unbiased and usually more precise when covariates predict the outcome.
- Treatment not randomised: ``ols``, ``ipw``, ``dr`` and ``matching`` adjust for
  the covariates you pass. They are only as good as the assumption that those
  covariates include everything that drives both treatment and outcome.
  ``difference`` is biased in this case.

Every method reports the ATE with a standard error from a formula (no
resampling), so results are deterministic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExperimentEffect:
    """Results from experiment effect estimation."""
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    std_error: float
    method: str
    n_treated: int
    n_control: int
    alpha: float = 0.05
    estimand: str = "ATE"

    def summary(self) -> str:
        lines = [
            "Treatment Effect Estimate",
            "=" * 40,
            f"Method: {self.method}",
            f"Estimate ({self.estimand}): {self.estimate:.4f}",
            f"{1 - self.alpha:.0%} CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Std Error: {self.std_error:.4f}",
            f"P-value: {self.p_value:.4f}",
            f"Significant at {self.alpha:g}: {'Yes' if self.p_value < self.alpha else 'No'}",
            "",
            f"Treated: {self.n_treated}",
            f"Control: {self.n_control}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ExperimentEffect(estimate={self.estimate:.4f}, p={self.p_value:.4f})"


@dataclass
class HeterogeneousEffects:
    """Treatment effects by segment of one covariate.

    ``segments`` and ``segment_cis`` are keyed by segment label, in order.
    ``interaction_pvalue`` tests whether the effect is the same in every
    segment. ``segment_sizes`` gives the number of rows in each.
    """
    covariate: str
    segments: Dict[str, float]
    segment_cis: Dict[str, Tuple[float, float]]
    interaction_pvalue: float
    segment_sizes: Optional[Dict[str, int]] = None

    def plot(self, ax=None):
        """Plot heterogeneous effects."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        segments = list(self.segments.keys())
        effects = list(self.segments.values())
        errors = [(self.segment_cis[s][1] - self.segment_cis[s][0]) / 2 for s in segments]

        x = range(len(segments))
        ax.bar(x, effects, yerr=errors, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(segments)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel(self.covariate)
        ax.set_ylabel('Treatment Effect')
        ax.set_title(f'Heterogeneous Effects by {self.covariate}')
        return ax


class ExperimentAnalysis:
    """
    Analyze an A/B test or a treated-versus-untreated comparison.

    Parameters
    ----------
    data : pd.DataFrame
        One row per unit.
    treatment : str
        Treatment column, coded 0/1 (or boolean).
    outcome : str
        Numeric outcome column.
    covariates : list of str, optional
        Numeric pre-treatment covariates used for adjustment. Never include a
        variable that the treatment could have affected.
    random_state : int, optional
        Kept for compatibility. Every estimator is deterministic.

    Rows with a missing value in any of these columns are dropped, with a warning.

    Examples
    --------
    >>> exp = ExperimentAnalysis(
    ...     data=ab_test_data,
    ...     treatment='new_feature',
    ...     outcome='engagement',
    ...     covariates=['user_tenure']
    ... )
    >>> effect = exp.estimate_effect(method='dr')
    >>> het = exp.analyze_heterogeneity(covariates=['user_tenure'])
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = list(covariates or [])
        self.random_state = random_state
        self.data = data.copy()
        self._validate_inputs()

        self._treated = self.data[self.data[treatment] == 1]
        self._control = self.data[self.data[treatment] == 0]

    def _validate_inputs(self):
        if self.treatment not in self.data.columns:
            raise ValueError(f"Treatment '{self.treatment}' not found")
        if self.outcome not in self.data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found")
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not found")
            if cov in (self.treatment, self.outcome):
                raise ValueError(f"'{cov}' cannot be both a covariate and the "
                                 "treatment or outcome")

        used = [self.treatment, self.outcome] + self.covariates
        missing = self.data[used].isna().any(axis=1)
        if missing.any():
            warnings.warn(
                f"Dropped {int(missing.sum())} of {len(self.data)} rows with missing "
                "values in the treatment, outcome or covariates.",
                UserWarning, stacklevel=3
            )
            self.data = self.data.loc[~missing].copy()

        values = set(pd.unique(self.data[self.treatment]))
        if not values <= {0, 1, True, False}:
            raise ValueError(
                f"Treatment '{self.treatment}' must be coded 0/1; found values "
                f"{sorted(map(str, values))[:6]}"
            )
        self.data[self.treatment] = self.data[self.treatment].astype(int)
        for col in [self.outcome] + self.covariates:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        counts = self.data[self.treatment].value_counts()
        if counts.get(1, 0) < 2 or counts.get(0, 0) < 2:
            raise ValueError("Need at least two treated and two control rows")

    # ------------------------------------------------------------------
    # Shared pieces
    # ------------------------------------------------------------------

    def _arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.data[self.covariates].to_numpy(dtype=float)
        T = self.data[self.treatment].to_numpy(dtype=float)
        Y = self.data[self.outcome].to_numpy(dtype=float)
        return X, T, Y

    def _result(self, estimate, se, method, alpha, df=None) -> ExperimentEffect:
        se = float(se)
        if se > 0:
            dist = stats.norm if df is None else stats.t(df)
            critical = dist.ppf(1 - alpha / 2)
            p_value = float(2 * dist.sf(abs(estimate) / se))
        else:
            critical, p_value = 0.0, (1.0 if estimate == 0 else 0.0)
        return ExperimentEffect(
            estimate=float(estimate),
            ci_lower=float(estimate - critical * se),
            ci_upper=float(estimate + critical * se),
            p_value=p_value,
            std_error=se,
            method=method,
            n_treated=len(self._treated),
            n_control=len(self._control),
            alpha=alpha,
        )

    def _propensity(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Estimated probability of treatment given the covariates."""
        from sklearn.linear_model import LogisticRegression
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Z = (X - X.mean(axis=0)) / scale
        model = LogisticRegression(C=1e6, max_iter=2000).fit(Z, T.astype(int))
        e = model.predict_proba(Z)[:, 1]
        extreme = float(np.mean((e < 0.01) | (e > 0.99)))
        if extreme > 0:
            warnings.warn(
                f"{extreme:.1%} of rows have an estimated treatment probability "
                "below 1% or above 99%: for them there is almost no comparable "
                "unit in the other group. Their weights are capped, and the "
                "estimate leans on the model. Consider restricting the population.",
                UserWarning, stacklevel=4
            )
        return np.clip(e, 0.01, 0.99)

    @staticmethod
    def _ols_fit(A: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(A, y, rcond=None)[0]

    def _outcome_predictions(self, X, T, Y) -> Tuple[np.ndarray, np.ndarray]:
        """Linear outcome model fitted separately in each group, predicted for all."""
        A = np.column_stack([np.ones(len(X)), X])
        mu1 = A @ self._ols_fit(A[T == 1], Y[T == 1])
        mu0 = A @ self._ols_fit(A[T == 0], Y[T == 0])
        return mu1, mu0

    # ------------------------------------------------------------------
    # Estimators
    # ------------------------------------------------------------------

    def estimate_effect(self, method: str = 'difference', alpha: float = 0.05,
                        **kwargs) -> ExperimentEffect:
        """
        Estimate the average treatment effect.

        Parameters
        ----------
        method : str
            - 'difference': difference in means (Welch). For randomised data.
            - 'ols': regression adjustment with treatment-covariate interactions
              (Lin, 2013) and robust standard errors.
            - 'ipw': inverse probability weighting (normalised weights).
            - 'dr' / 'aipw': doubly robust. Consistent if either the propensity
              model or the outcome model is right.
            - 'matching': nearest-neighbour matching on the propensity score, in
              both directions, bias-corrected, with the Abadie-Imbens standard error.
              ``n_neighbors`` (default 5) sets the number of matches.

            Without covariates every method reduces to 'difference'.
        alpha : float
            The interval has level ``1 - alpha``.

        Returns
        -------
        ExperimentEffect
        """
        methods = {
            'difference': self._difference_in_means,
            'ols': self._ols_estimator,
            'ipw': self._ipw_estimator,
            'dr': self._doubly_robust,
            'aipw': self._doubly_robust,
            'matching': self._matching_estimator,
        }
        key = str(method).lower()
        if key not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {sorted(methods)}")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if key != 'difference' and not self.covariates:
            return self._difference_in_means(alpha=alpha, check_balance=False)
        return methods[key](alpha=alpha, **kwargs)

    def _difference_in_means(self, alpha: float = 0.05, check_balance: bool = True,
                             **kwargs) -> ExperimentEffect:
        y1 = self._treated[self.outcome].to_numpy(dtype=float)
        y0 = self._control[self.outcome].to_numpy(dtype=float)
        v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
        se = np.sqrt(v1 + v0)
        if se > 0:
            df = (v1 + v0) ** 2 / (v1 ** 2 / (len(y1) - 1) + v0 ** 2 / (len(y0) - 1))
        else:
            df = len(y1) + len(y0) - 2

        if check_balance and self.covariates:
            smd = self.balance_table()['std_mean_diff'].abs()
            if smd.max() > 0.25:
                warnings.warn(
                    f"The groups differ on {smd.idxmax()} (standardised difference "
                    f"{smd.max():.2f}), which is unusual for a randomised experiment. "
                    "The difference in means is biased if treatment was not "
                    "randomised; use method='dr' to adjust for the covariates.",
                    UserWarning, stacklevel=3
                )
        return self._result(y1.mean() - y0.mean(), se, 'difference_in_means', alpha, df=df)

    def _ols_estimator(self, alpha: float = 0.05, **kwargs) -> ExperimentEffect:
        X, T, Y = self._arrays()
        n = len(Y)
        Xc = X - X.mean(axis=0)
        A = np.column_stack([np.ones(n), T, Xc, Xc * T[:, None]])
        gram_inv = np.linalg.pinv(A.T @ A)
        beta = gram_inv @ A.T @ Y
        residual = Y - A @ beta
        leverage = np.clip(np.einsum('ij,jk,ik->i', A, gram_inv, A), 0, 1 - 1e-8)
        meat = (A * (residual ** 2 / (1 - leverage))[:, None]).T @ A        # HC2
        variance = (gram_inv @ meat @ gram_inv)[1, 1]
        # The covariate means are estimated too; when the effect varies with the
        # covariates that adds to the uncertainty about the average effect.
        k = X.shape[1]
        slope = beta[2 + k:]
        variance += float(slope @ np.atleast_2d(np.cov(X, rowvar=False)) @ slope) / n
        return self._result(beta[1], np.sqrt(variance), 'ols', alpha)

    def _ipw_estimator(self, alpha: float = 0.05, **kwargs) -> ExperimentEffect:
        X, T, Y = self._arrays()
        n = len(Y)
        e = self._propensity(X, T)
        w1, w0 = T / e, (1 - T) / (1 - e)
        mean1, mean0 = np.sum(w1 * Y) / np.sum(w1), np.sum(w0 * Y) / np.sum(w0)
        influence = w1 * (Y - mean1) / np.mean(w1) - w0 * (Y - mean0) / np.mean(w0)
        # The weights are estimated, which makes the estimate more precise than
        # known weights would. Removing the part of the influence explained by the
        # propensity model's score accounts for that (Lunceford and Davidian, 2004).
        score = (T - e)[:, None] * np.column_stack([np.ones(n), X - X.mean(axis=0)])
        influence = influence - score @ self._ols_fit(score, influence)
        return self._result(mean1 - mean0, influence.std(ddof=1) / np.sqrt(n), 'ipw', alpha)

    def _doubly_robust(self, alpha: float = 0.05, **kwargs) -> ExperimentEffect:
        X, T, Y = self._arrays()
        n = len(Y)
        e = self._propensity(X, T)
        mu1, mu0 = self._outcome_predictions(X, T, Y)
        psi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
        return self._result(psi.mean(), psi.std(ddof=1) / np.sqrt(n), 'doubly_robust', alpha)

    def _matching_estimator(self, n_neighbors: int = 5, alpha: float = 0.05,
                            **kwargs) -> ExperimentEffect:
        from sklearn.neighbors import NearestNeighbors

        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        X, T, Y = self._arrays()
        n = len(Y)
        e = self._propensity(X, T)
        score = np.log(e / (1 - e)).reshape(-1, 1)
        groups = {1: np.where(T == 1)[0], 0: np.where(T == 0)[0]}
        M = min(n_neighbors, len(groups[1]), len(groups[0]))

        # Linear outcome models, used only to correct for the remaining covariate
        # gap between a unit and its matches (Abadie and Imbens, 2011).
        mu = dict(zip((1, 0), self._outcome_predictions(X, T, Y)))

        imputed = {1: np.empty(n), 0: np.empty(n)}     # Y(1) and Y(0) for every unit
        times_used = np.zeros(n)
        sigma2 = np.empty(n)
        for own, other in ((1, 0), (0, 1)):
            idx_own, idx_other = groups[own], groups[other]
            imputed[own][idx_own] = Y[idx_own]
            _, nearest = NearestNeighbors(n_neighbors=M).fit(score[idx_other]).kneighbors(
                score[idx_own])
            matched = idx_other[nearest]
            imputed[other][idx_own] = (
                Y[matched] - mu[other][matched] + mu[other][idx_own][:, None]).mean(axis=1)
            np.add.at(times_used, matched.ravel(), 1.0)
            # Outcome variance at each unit, from its nearest neighbours in its own group
            J = min(M, len(idx_own) - 1)
            _, same = NearestNeighbors(n_neighbors=J + 1).fit(score[idx_own]).kneighbors(
                score[idx_own])
            neighbours = Y[idx_own[same[:, 1:]]]
            sigma2[idx_own] = J / (J + 1) * (Y[idx_own] - neighbours.mean(axis=1)) ** 2

        unit_effect = imputed[1] - imputed[0]
        ate = unit_effect.mean()
        k = times_used / M
        variance = (np.sum((unit_effect - ate) ** 2)
                    + np.sum((k ** 2 + (2 * M - 1) / M * k) * sigma2)) / n ** 2
        return self._result(ate, np.sqrt(variance), 'matching', alpha)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def balance_table(self) -> pd.DataFrame:
        """
        Covariate balance between the groups.

        One row per covariate: the group means and the standardised mean
        difference. In a randomised experiment these differences should be
        small; an absolute value above about 0.1 deserves a look.
        """
        rows = {}
        for cov in self.covariates:
            a = self._treated[cov].to_numpy(dtype=float)
            b = self._control[cov].to_numpy(dtype=float)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            rows[cov] = {
                'mean_treated': a.mean(),
                'mean_control': b.mean(),
                'std_mean_diff': (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0,
            }
        return pd.DataFrame.from_dict(
            rows, orient='index', columns=['mean_treated', 'mean_control', 'std_mean_diff'])

    def check_sample_ratio(self, expected_treated_share: float = 0.5) -> float:
        """
        Sample ratio mismatch check. Returns the p-value of a chi-square test
        that the observed split matches the intended one. A very small value
        (commonly below 0.001) points to a problem in assignment or logging,
        and the experiment's results should not be trusted until it is explained.
        """
        if not 0 < expected_treated_share < 1:
            raise ValueError("expected_treated_share must be between 0 and 1")
        n1, n0 = len(self._treated), len(self._control)
        total = n1 + n0
        expected = [total * expected_treated_share, total * (1 - expected_treated_share)]
        return float(stats.chisquare([n1, n0], expected).pvalue)

    # ------------------------------------------------------------------
    # Heterogeneity
    # ------------------------------------------------------------------

    def analyze_heterogeneity(
        self,
        covariates: Optional[List[str]] = None,
        n_quantiles: int = 4,
        min_segment_size: int = 10,
        alpha: float = 0.05,
    ) -> List[HeterogeneousEffects]:
        """
        Treatment effect by segment.

        A covariate with at most ``n_quantiles`` distinct values is used as is;
        otherwise it is cut into ``n_quantiles`` equal-sized bins. Within each
        segment the effect is the difference in means, which is valid when
        treatment was randomised.

        ``interaction_pvalue`` tests whether the effect is the same in every
        segment shown. Looking at several covariates multiplies the chances of
        a small p-value by luck; treat a single one as a lead, not a finding.

        Parameters
        ----------
        covariates : list of str, optional
            Defaults to the covariates given at construction.
        n_quantiles : int
            Number of bins for a continuous covariate.
        min_segment_size : int
            Segments with fewer rows in either group are left out, with a warning.
        alpha : float
            Level of the segment intervals.
        """
        covariates = list(covariates) if covariates is not None else self.covariates
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")
        results = []
        for cov in covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not found")
            results.append(
                self._analyze_heterogeneity_single(cov, n_quantiles, min_segment_size, alpha))
        return results

    def _analyze_heterogeneity_single(self, covariate, n_quantiles, min_segment_size,
                                      alpha) -> HeterogeneousEffects:
        column = self.data[covariate]
        keep = column.notna()
        column = column[keep]
        T = self.data.loc[keep, self.treatment].to_numpy()
        Y = self.data.loc[keep, self.outcome].to_numpy(dtype=float)

        if column.nunique() <= n_quantiles or not pd.api.types.is_numeric_dtype(column):
            labels = column.astype(str)
            try:
                order = [str(v) for v in sorted(column.unique())]
            except TypeError:
                order = sorted(labels.unique())
        else:
            bins = pd.qcut(column, q=n_quantiles, duplicates='drop')
            labels = bins.astype(str)
            order = [str(c) for c in bins.cat.categories]
        labels = labels.to_numpy()

        effects: Dict[str, float] = {}
        cis: Dict[str, Tuple[float, float]] = {}
        sizes: Dict[str, int] = {}
        skipped = []
        for label in order:
            mask = labels == label
            y1, y0 = Y[mask & (T == 1)], Y[mask & (T == 0)]
            if min(len(y1), len(y0)) < max(2, min_segment_size):
                skipped.append(label)
                continue
            v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
            se = np.sqrt(v1 + v0)
            effect = float(y1.mean() - y0.mean())
            if se > 0:
                df = (v1 + v0) ** 2 / (v1 ** 2 / (len(y1) - 1) + v0 ** 2 / (len(y0) - 1))
                half = stats.t.ppf(1 - alpha / 2, df) * se
            else:
                half = 0.0
            effects[label] = effect
            cis[label] = (float(effect - half), float(effect + half))
            sizes[label] = int(mask.sum())
        if skipped:
            warnings.warn(
                f"Segments of {covariate} left out because a group has fewer than "
                f"{max(2, min_segment_size)} rows: {skipped}",
                UserWarning, stacklevel=4
            )

        return HeterogeneousEffects(
            covariate=covariate,
            segments=effects,
            segment_cis=cis,
            interaction_pvalue=self._test_interaction(labels, T, Y, list(effects)),
            segment_sizes=sizes,
        )

    def _test_interaction(self, labels, T, Y, segments: List[str]) -> float:
        """F-test that the treatment effect is the same in every segment."""
        if len(segments) < 2:
            return float('nan')
        keep = np.isin(labels, segments)
        labels, T, Y = labels[keep], T[keep].astype(float), Y[keep]
        dummies = np.column_stack([(labels == s).astype(float) for s in segments])
        reduced = np.column_stack([dummies, T])
        full = np.column_stack([dummies, dummies * T[:, None]])
        rss_reduced = np.sum((Y - reduced @ self._ols_fit(reduced, Y)) ** 2)
        rss_full = np.sum((Y - full @ self._ols_fit(full, Y)) ** 2)
        df1 = len(segments) - 1
        df2 = len(Y) - full.shape[1]
        if df2 <= 0 or rss_full <= 0:
            return float('nan')
        f_stat = ((rss_reduced - rss_full) / df1) / (rss_full / df2)
        return float(stats.f.sf(f_stat, df1, df2))

    # ------------------------------------------------------------------
    # Projection and design
    # ------------------------------------------------------------------

    def simulate_longterm(
        self,
        periods: int = 90,
        decay_rate: float = 0.1,
        method: str = 'difference',
    ) -> pd.DataFrame:
        """
        Project the measured effect forward under an assumed decay.

        This is a what-if calculation, not an estimate. The experiment measures
        the effect during the test; how it fades afterwards is the
        ``decay_rate`` you supply (the effect in period ``t`` is the measured
        effect times ``exp(-decay_rate * t)``). Nothing in the data informs it.

        Returns
        -------
        pd.DataFrame
            ``period``, ``effect``, ``cumulative_effect``, and lower and upper
            versions of both that carry the confidence interval of the measured
            effect through the same decay.
        """
        if periods < 1:
            raise ValueError("periods must be at least 1")
        if decay_rate < 0:
            raise ValueError("decay_rate must not be negative")
        measured = self.estimate_effect(method=method)
        t = np.arange(periods)
        decay = np.exp(-decay_rate * t)
        out = pd.DataFrame({'period': t})
        for name, value in (('effect', measured.estimate),
                            ('effect_lower', measured.ci_lower),
                            ('effect_upper', measured.ci_upper)):
            out[name] = value * decay
        out['cumulative_effect'] = out['effect'].cumsum()
        out['cumulative_lower'] = out['effect_lower'].cumsum()
        out['cumulative_upper'] = out['effect_upper'].cumsum()
        return out[['period', 'effect', 'cumulative_effect', 'effect_lower',
                    'effect_upper', 'cumulative_lower', 'cumulative_upper']]

    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Sample size per group needed to detect ``effect_size`` (in outcome
        units) with a two-sided test, for two groups of equal size.

        The outcome's spread is taken from this data, within groups, so the
        treatment effect itself does not inflate it.
        """
        if effect_size == 0:
            raise ValueError("effect_size must not be zero")
        if not 0 < alpha < 1 or not 0 < power < 1:
            raise ValueError("alpha and power must be between 0 and 1")
        y1 = self._treated[self.outcome].to_numpy(dtype=float)
        y0 = self._control[self.outcome].to_numpy(dtype=float)
        pooled = ((len(y1) - 1) * y1.var(ddof=1) + (len(y0) - 1) * y0.var(ddof=1)) \
            / (len(y1) + len(y0) - 2)
        if pooled <= 0:
            raise ValueError("The outcome does not vary, so a sample size cannot be computed")
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
        return int(np.ceil(2 * pooled * (z / effect_size) ** 2))

    def summary(self) -> str:
        """Return experiment summary."""
        effect = self.estimate_effect()
        lines = [
            "Experiment Analysis Summary",
            "=" * 50,
            f"Treatment: {self.treatment}",
            f"Outcome: {self.outcome}",
            "",
            f"Treated group: n={len(self._treated)}, "
            f"mean={self._treated[self.outcome].mean():.4f}",
            f"Control group: n={len(self._control)}, "
            f"mean={self._control[self.outcome].mean():.4f}",
            "",
            "Treatment Effect (difference in means):",
            f"  Estimate: {effect.estimate:.4f}",
            f"  95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]",
            f"  P-value: {effect.p_value:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"ExperimentAnalysis(treatment='{self.treatment}', "
                f"n_treated={len(self._treated)}, n_control={len(self._control)})")
