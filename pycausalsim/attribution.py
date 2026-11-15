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
Marketing attribution.

Two kinds of method live here and they answer different questions.

**Incremental methods** (``shapley``, ``logistic``, ``linear``, ``uplift``) fit a
model of conversion given which touchpoints a user was exposed to, and ask what
would have happened without them. Conversions that would have happened with no
marketing at all are a baseline and are credited to no channel. A channel with no
effect earns no credit and a channel that lowers conversion shows a negative
contribution.

**Path heuristics** (``markov``, ``first_touch``, ``last_touch``) describe where
converting journeys went. They need the order of contact and say nothing about
what a channel caused.

Assumptions of the incremental methods
--------------------------------------
- Exposure is read as exposed / not exposed (any value above zero is exposed).
- Whatever drives both exposure and conversion is measured and passed as
  ``covariate_cols``. Without that, a channel shown mainly to people who were
  going to convert anyway is credited with their conversions. Only an experiment
  removes this assumption.
- The outcome model is logistic for a 0/1 conversion column and linear otherwise.
"""

from itertools import combinations
from math import factorial
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

INCREMENTAL_METHODS = ('shapley', 'logistic', 'linear', 'uplift')
PATH_METHODS = ('markov', 'last_touch', 'first_touch')
_EXACT_SHAPLEY_MAX = 8


class MarketingAttribution:
    """
    Marketing attribution from user-level touchpoint data.

    Parameters
    ----------
    data : pd.DataFrame
        One row per user (or journey).
    conversion_col : str
        Outcome column. 0/1 for conversions; any other numeric outcome, such as
        revenue, is modelled linearly.
    touchpoint_cols : list of str, optional
        Exposure columns, one per channel. Any value above zero counts as
        exposed. Detected when omitted (from ``path_col`` if that is given).
    user_col : str, optional
        User identifier column. Not used in estimation.
    random_state : int, optional
        Seed for the sampling steps. The global NumPy state is left alone.
    covariate_cols : list of str, optional
        Numeric user-level variables that drive both exposure and conversion,
        such as prior purchases or intent. The incremental methods adjust for them.
    path_col : str, optional
        Column holding each journey's ordered channels, as a list or as a string
        such as ``"search > email > search"``. Used by the path heuristics.
        Without it they take the order of ``touchpoint_cols`` as the order of
        contact, which is an assumption.

    Examples
    --------
    >>> attr = MarketingAttribution(
    ...     data=touchpoint_data,
    ...     conversion_col='converted',
    ...     touchpoint_cols=['email', 'display', 'search', 'social']
    ... )
    >>> attr.fit(method='shapley')
    >>> attr.get_attribution()            # shares of incremental conversions
    >>> attr.get_incremental_effects()    # effect of an exposure, per channel
    """

    def __init__(
        self,
        data: pd.DataFrame,
        conversion_col: str = 'converted',
        touchpoint_cols: Optional[List[str]] = None,
        user_col: Optional[str] = None,
        random_state: Optional[int] = None,
        covariate_cols: Optional[List[str]] = None,
        path_col: Optional[str] = None,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()
        self.conversion_col = conversion_col
        self.user_col = user_col
        self.covariate_cols = list(covariate_cols or [])
        self.path_col = path_col
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        if conversion_col not in self.data.columns:
            raise ValueError(f"Conversion column '{conversion_col}' not found")
        for col in self.covariate_cols + ([path_col] if path_col else []):
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found")

        self._paths: Optional[List[List[str]]] = None
        if path_col is not None:
            self._paths = [self._parse_path(p) for p in self.data[path_col]]

        if touchpoint_cols is None:
            touchpoint_cols = self._detect_touchpoints()
        self.touchpoint_cols = list(touchpoint_cols)
        self._validate_inputs()

        self._attribution: Optional[Dict[str, float]] = None
        self._contributions: Optional[Dict[str, float]] = None
        self._baseline: Optional[float] = None
        self._method: Optional[str] = None
        self._model = None
        self._interactions = False
        self._incremental_effects: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_path(value) -> List[str]:
        if isinstance(value, str):
            return [p.strip() for p in value.split('>') if p.strip()]
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return []
        return [str(p) for p in value]

    def _detect_touchpoints(self) -> List[str]:
        """Channels named in the paths, or else the 0/1 columns."""
        if self._paths is not None:
            channels = sorted({c for path in self._paths for c in path})
            for c in channels:
                if c not in self.data.columns:
                    self.data[c] = [int(c in path) for path in self._paths]
            return channels
        exclude = {self.conversion_col, self.user_col, self.path_col, *self.covariate_cols}
        found = []
        for col in self.data.columns:
            if col in exclude or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue
            values = set(pd.unique(self.data[col].dropna()))
            if values and values <= {0, 1}:
                found.append(col)
        return found

    def _validate_inputs(self):
        if not self.touchpoint_cols:
            raise ValueError("No touchpoint columns given or detected")
        for col in self.touchpoint_cols:
            if col not in self.data.columns:
                raise ValueError(f"Touchpoint column '{col}' not found")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Touchpoint column '{col}' must be numeric")
        if len(set(self.touchpoint_cols)) != len(self.touchpoint_cols):
            raise ValueError("touchpoint_cols contains duplicates")
        if not pd.api.types.is_numeric_dtype(self.data[self.conversion_col]):
            raise ValueError(f"Conversion column '{self.conversion_col}' must be numeric")
        if self.data[self.conversion_col].isna().any():
            raise ValueError(f"Conversion column '{self.conversion_col}' has missing values")
        for col in self.covariate_cols:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(
                    f"Covariate '{col}' must be numeric. Encode categories first, "
                    "for example with pd.get_dummies."
                )
            if self.data[col].isna().any():
                raise ValueError(f"Covariate '{col}' has missing values")
        if self._paths is not None:
            unknown = sorted({c for p in self._paths for c in p} - set(self.touchpoint_cols))
            if unknown:
                raise ValueError(f"Paths name channels that are not touchpoints: {unknown}")
        if len(self.data) < 2:
            raise ValueError("Need at least two rows")

    def _exposure(self, data: pd.DataFrame) -> np.ndarray:
        return (data[self.touchpoint_cols].to_numpy(dtype=float) > 0).astype(float)

    def _covariates(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.covariate_cols:
            return None
        return data[self.covariate_cols].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Outcome model
    # ------------------------------------------------------------------

    def _design(self, E: np.ndarray, C: Optional[np.ndarray], interactions: bool) -> np.ndarray:
        blocks = [E]
        if interactions and E.shape[1] > 1:
            blocks.append(np.column_stack(
                [E[:, i] * E[:, j] for i, j in combinations(range(E.shape[1]), 2)]
            ))
        if C is not None:
            blocks.append(C)
        return np.hstack(blocks)

    def _fit_outcome_model(
        self,
        data: pd.DataFrame,
        interactions: bool = False,
        force_linear: bool = False,
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """Fit conversion on exposure (and covariates). Returns ``predict(E, C)``."""
        E = self._exposure(data)
        C = self._covariates(data)
        y = data[self.conversion_col].to_numpy(dtype=float)
        if C is not None:
            mu, sd = C.mean(axis=0), C.std(axis=0)
            sd[sd == 0] = 1.0

            def scale(c):
                return (c - mu) / sd
        else:
            def scale(c):
                return None
        X = self._design(E, scale(C), interactions)
        binary = set(np.unique(y)) <= {0.0, 1.0}

        if len(np.unique(y)) == 1:
            constant = float(y[0])
            return lambda e, c: np.full(len(e), constant)

        if binary and not force_linear:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1e6, max_iter=2000)
            model.fit(X, y.astype(int))
            return lambda e, c: model.predict_proba(
                self._design(e, scale(c), interactions))[:, 1]

        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)

        def predict(e, c):
            Z = self._design(e, scale(c), interactions)
            return beta[0] + Z @ beta[1:]
        predict.coef_ = beta[1:1 + E.shape[1]]
        return predict

    def _effects_from(self, predict, E: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        """Average effect of an exposure to each channel, others as observed."""
        out = np.zeros(E.shape[1])
        for i in range(E.shape[1]):
            on, off = E.copy(), E.copy()
            on[:, i], off[:, i] = 1.0, 0.0
            out[i] = float(np.mean(predict(on, C) - predict(off, C)))
        return out

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, method: str = 'shapley', **kwargs) -> 'MarketingAttribution':
        """
        Fit an attribution method.

        Parameters
        ----------
        method : str
            Incremental methods, which credit only what a channel added:

            - 'shapley': Shapley values of the modelled conversion rate. Each
              channel's credit is its average marginal contribution over every
              order in which channels could be switched on.
            - 'logistic': removal effect. The modelled conversions lost if that
              one channel were switched off.
            - 'linear': linear probability model, coefficient times exposure rate.
            - 'uplift': the effect of one exposure, ignoring how many users the
              channel reaches. Ranks channels by effectiveness, not by volume.

            Path heuristics, which describe journeys and are not causal:

            - 'markov': first-order Markov chain removal effect.
            - 'last_touch', 'first_touch'.
        interactions : bool, default False
            Incremental methods only. Add pairwise channel interactions to the
            outcome model, so that channels can reinforce or cannibalise each other.
        n_samples : int, default 1000
            Shapley only. Permutations sampled when there are more than 8
            channels; with 8 or fewer the values are computed exactly.
        exact : bool, optional
            Shapley only. Force exact or sampled computation.

        Returns
        -------
        self
        """
        methods = {
            'shapley': self._fit_shapley,
            'logistic': self._fit_logistic,
            'uplift': self._fit_uplift,
            'markov': self._fit_markov,
            'linear': self._fit_linear,
            'last_touch': self._fit_last_touch,
            'first_touch': self._fit_first_touch,
        }
        key = str(method).lower()
        if key not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {sorted(methods)}")

        self._method = key
        self._contributions = None
        self._baseline = None
        if key in INCREMENTAL_METHODS:
            self._interactions = bool(kwargs.pop('interactions', False))
            self._incremental_effects = None
        methods[key](**kwargs)
        return self

    def _set_incremental(self, contributions: np.ndarray, baseline: float) -> None:
        self._contributions = dict(zip(self.touchpoint_cols, map(float, contributions)))
        self._baseline = float(baseline)
        self._attribution = self._shares(self._contributions)

    def _shares(self, contributions: Dict[str, float]) -> Dict[str, float]:
        """Shares of the positive contributions. Negative ones get no credit."""
        positive = {k: max(0.0, v) for k, v in contributions.items()}
        total = sum(positive.values())
        if total <= 1e-12:
            warnings.warn(
                "No channel shows a positive contribution, so there is nothing to "
                "attribute. All shares are zero.",
                UserWarning, stacklevel=5
            )
            return {k: 0.0 for k in contributions}
        negative = -sum(min(0.0, v) for v in contributions.values())
        if negative > 0.1 * total:
            harmful = sorted(k for k, v in contributions.items() if v < 0)
            warnings.warn(
                f"These channels have a negative contribution and receive no share: "
                f"{harmful}. See get_contributions() for the signed values.",
                UserWarning, stacklevel=5
            )
        return {k: v / total for k, v in positive.items()}

    def _coalition_value(self, predict, E, C) -> Callable[[FrozenSet[int]], float]:
        """v(S): modelled conversion rate if only the channels in S had run."""
        cache: Dict[FrozenSet[int], float] = {}

        def v(S: FrozenSet[int]) -> float:
            if S not in cache:
                mask = np.zeros(E.shape[1])
                mask[list(S)] = 1.0
                cache[S] = float(np.mean(predict(E * mask, C)))
            return cache[S]
        return v

    def _fit_shapley(self, n_samples: int = 1000, exact: Optional[bool] = None, **kwargs):
        E, C = self._exposure(self.data), self._covariates(self.data)
        predict = self._fit_outcome_model(self.data, self._interactions)
        self._model = predict
        v = self._coalition_value(predict, E, C)
        k = E.shape[1]
        if exact is None:
            exact = k <= _EXACT_SHAPLEY_MAX

        phi = np.zeros(k)
        if exact:
            for i in range(k):
                others = [j for j in range(k) if j != i]
                for size in range(k):
                    weight = factorial(size) * factorial(k - size - 1) / factorial(k)
                    for subset in combinations(others, size):
                        S = frozenset(subset)
                        phi[i] += weight * (v(S | {i}) - v(S))
        else:
            if n_samples < 1:
                raise ValueError("n_samples must be at least 1")
            for _ in range(n_samples):
                S: FrozenSet[int] = frozenset()
                previous = v(S)
                for i in self._rng.permutation(k):
                    S = S | {int(i)}
                    current = v(S)
                    phi[i] += (current - previous) / n_samples
                    previous = current
        self._set_incremental(phi, v(frozenset()))

    def _fit_logistic(self, **kwargs):
        E, C = self._exposure(self.data), self._covariates(self.data)
        predict = self._fit_outcome_model(self.data, self._interactions)
        self._model = predict
        v = self._coalition_value(predict, E, C)
        k = E.shape[1]
        everything = frozenset(range(k))
        removal = np.array([v(everything) - v(everything - {i}) for i in range(k)])
        self._set_incremental(removal, v(frozenset()))

    def _fit_linear(self, **kwargs):
        E, C = self._exposure(self.data), self._covariates(self.data)
        predict = self._fit_outcome_model(self.data, self._interactions, force_linear=True)
        self._model = predict
        v = self._coalition_value(predict, E, C)
        k = E.shape[1]
        everything = frozenset(range(k))
        removal = np.array([v(everything) - v(everything - {i}) for i in range(k)])
        self._set_incremental(removal, v(frozenset()))

    def _fit_uplift(self, **kwargs):
        E, C = self._exposure(self.data), self._covariates(self.data)
        predict = self._fit_outcome_model(self.data, self._interactions)
        self._model = predict
        effects = self._effects_from(predict, E, C)
        self._set_incremental(effects, float(np.mean(predict(np.zeros_like(E), C))))

    # ------------------------------------------------------------------
    # Path heuristics
    # ------------------------------------------------------------------

    def _journeys(self) -> List[List[str]]:
        if self._paths is not None:
            return self._paths
        warnings.warn(
            "No path_col was given, so the order of touchpoint_cols "
            f"{self.touchpoint_cols} is taken as the order of contact. Pass "
            "path_col for the real order.",
            UserWarning, stacklevel=4
        )
        E = self._exposure(self.data)
        return [[c for c, on in zip(self.touchpoint_cols, row) if on] for row in E]

    def _set_heuristic(self, credit: Dict[str, float]) -> None:
        total = sum(credit.values())
        if total <= 0:
            warnings.warn("No conversions to attribute. All shares are zero.",
                          UserWarning, stacklevel=4)
            self._attribution = {k: 0.0 for k in credit}
        else:
            self._attribution = {k: v / total for k, v in credit.items()}

    def _fit_markov(self, **kwargs):
        """
        First-order Markov chain attribution.

        Journeys are modelled as a chain ``start -> channels -> conversion or
        no conversion``. A channel's removal effect is the share of conversions
        lost when every visit to it is sent to no conversion instead. Shares are
        the removal effects scaled to sum to one.
        """
        journeys = self._journeys()
        converted = self.data[self.conversion_col].to_numpy(dtype=float)
        k = len(self.touchpoint_cols)
        index = {c: i + 1 for i, c in enumerate(self.touchpoint_cols)}   # 0 is start
        counts = np.zeros((k + 1, k + 1))
        to_conversion = np.zeros(k + 1)
        visits = np.zeros(k + 1)

        for path, value in zip(journeys, converted):
            state = 0
            for channel in path:
                nxt = index[channel]
                counts[state, nxt] += 1
                visits[state] += 1
                state = nxt
            visits[state] += 1
            to_conversion[state] += value

        visits[visits == 0] = 1.0
        Q = counts / visits[:, None]
        R = to_conversion / visits

        def conversion_rate(Q_: np.ndarray, R_: np.ndarray) -> float:
            return float(np.linalg.lstsq(np.eye(k + 1) - Q_, R_, rcond=None)[0][0])

        base = conversion_rate(Q, R)
        self._markov_conversion_rate = base
        removal = {}
        for channel, i in index.items():
            Q_removed, R_removed = Q.copy(), R.copy()
            Q_removed[:, i] = 0.0          # visits to the channel are lost
            Q_removed[i, :] = 0.0
            R_removed[i] = 0.0
            removal[channel] = max(0.0, 1.0 - conversion_rate(Q_removed, R_removed) / base) \
                if base > 0 else 0.0
        self._removal_effects = removal
        self._set_heuristic(removal)

    def _fit_last_touch(self, **kwargs):
        """All credit for a conversion to the last channel of the journey."""
        self._touch(-1)

    def _fit_first_touch(self, **kwargs):
        """All credit for a conversion to the first channel of the journey."""
        self._touch(0)

    def _touch(self, position: int) -> None:
        journeys = self._journeys()
        converted = self.data[self.conversion_col].to_numpy(dtype=float)
        credit = {c: 0.0 for c in self.touchpoint_cols}
        for path, value in zip(journeys, converted):
            if path and value > 0:
                credit[path[position]] += float(value)
        self._set_heuristic(credit)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _require_fit(self) -> None:
        if self._attribution is None:
            raise RuntimeError("Must call fit() first")

    def get_attribution(self) -> Dict[str, float]:
        """
        Attribution shares.

        For the incremental methods these are shares of the conversions that
        marketing added; conversions that would have happened anyway are left
        out, and a channel with a negative contribution gets zero. The shares
        sum to one unless no channel has a positive contribution, in which case
        they are all zero.
        """
        self._require_fit()
        return self._attribution.copy()

    def get_contributions(self) -> Dict[str, float]:
        """
        Signed contribution of each channel, in units of the conversion column
        per user (for a 0/1 column, conversion-rate points). Incremental methods
        only. For 'shapley' these sum to the modelled conversion rate minus the
        baseline. For 'uplift' they are per-exposure effects.
        """
        self._require_fit()
        if self._contributions is None:
            raise RuntimeError(
                f"'{self._method}' is a path heuristic and does not estimate "
                f"contributions. Use one of {INCREMENTAL_METHODS}."
            )
        return self._contributions.copy()

    def get_baseline(self) -> float:
        """Modelled conversion rate with no marketing exposure. Incremental
        methods only."""
        self._require_fit()
        if self._baseline is None:
            raise RuntimeError(
                f"'{self._method}' is a path heuristic and does not estimate a baseline."
            )
        return self._baseline

    def get_incremental_effects(self) -> Dict[str, float]:
        """
        Effect of an exposure to each channel on the conversion column: the
        average difference between exposed and not exposed, with the other
        channels as observed and adjusted for the other channels and the
        covariates. Does not depend on the attribution method chosen.
        """
        self._require_fit()
        if self._incremental_effects is None:
            predict = self._fit_outcome_model(self.data, self._interactions)
            effects = self._effects_from(
                predict, self._exposure(self.data), self._covariates(self.data))
            self._incremental_effects = dict(zip(self.touchpoint_cols, map(float, effects)))
        return self._incremental_effects.copy()

    def incremental_effects_table(
        self,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        interactions: bool = False,
    ) -> pd.DataFrame:
        """
        Incremental effects with bootstrap confidence intervals.

        Returns one row per channel: ``effect``, ``ci_lower``, ``ci_upper``
        (percentile interval at level ``1 - alpha``) and ``exposure_rate``.
        """
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")
        E, C = self._exposure(self.data), self._covariates(self.data)
        point = self._effects_from(self._fit_outcome_model(self.data, interactions), E, C)
        draws = np.empty((n_bootstrap, len(self.touchpoint_cols)))
        n = len(self.data)
        for b in range(n_bootstrap):
            sample = self.data.iloc[self._rng.integers(0, n, n)]
            predict = self._fit_outcome_model(sample, interactions)
            draws[b] = self._effects_from(
                predict, self._exposure(sample), self._covariates(sample))
        lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
        return pd.DataFrame({
            'effect': point,
            'ci_lower': lower,
            'ci_upper': upper,
            'exposure_rate': E.mean(axis=0),
        }, index=self.touchpoint_cols)

    def optimize_budget(
        self,
        total_budget: float,
        cost_per_touchpoint: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Split a budget across channels in proportion to attribution per unit cost.

        This is a rule of thumb, not an optimum. Exposure flags carry no
        information about diminishing returns, so the data cannot say how a
        channel would respond to more spend; a response-curve model is needed
        for that (see ``pycausalsim.futures``).

        Parameters
        ----------
        total_budget : float
            Budget to allocate. The allocation always sums to it.
        cost_per_touchpoint : dict, optional
            Cost of one exposure per channel. Equal costs when omitted.
        constraints : dict, optional
            ``{channel: (min, max)}`` spend limits. Budget freed or taken by a
            limit is redistributed across the remaining channels.

        Returns
        -------
        dict
            Budget per channel.
        """
        self._require_fit()
        if total_budget < 0:
            raise ValueError("total_budget must not be negative")
        channels = self.touchpoint_cols
        costs = dict(cost_per_touchpoint or {})
        limits = dict(constraints or {})
        unknown = sorted((set(costs) | set(limits)) - set(channels))
        if unknown:
            raise ValueError(f"Unknown channels: {unknown}")
        if any(c <= 0 for c in costs.values()):
            raise ValueError("Costs must be positive")
        low = {c: float(limits.get(c, (0.0, np.inf))[0]) for c in channels}
        high = {c: float(limits.get(c, (0.0, np.inf))[1]) for c in channels}
        if any(low[c] > high[c] or low[c] < 0 for c in channels):
            raise ValueError("Each constraint must satisfy 0 <= min <= max")
        if sum(low.values()) > total_budget + 1e-9 or sum(high.values()) < total_budget - 1e-9:
            raise ValueError("The constraints cannot be met with this total budget")

        weight = {c: self._attribution.get(c, 0.0) / costs.get(c, 1.0) for c in channels}
        if sum(weight.values()) <= 0:
            warnings.warn("No channel has a positive share; splitting the budget equally.",
                          UserWarning, stacklevel=2)
            weight = {c: 1.0 for c in channels}

        allocation: Dict[str, float] = {}
        free = list(channels)
        remaining = float(total_budget)
        while free:
            total_weight = sum(weight[c] for c in free)
            if total_weight > 0:
                proposal = {c: remaining * weight[c] / total_weight for c in free}
            else:
                proposal = {c: remaining / len(free) for c in free}
            below = [c for c in free if proposal[c] < low[c] - 1e-12]
            above = [c for c in free if proposal[c] > high[c] + 1e-12]
            if not below and not above:
                allocation.update(proposal)
                break
            # Fix the limits that are violated in the direction of the overall
            # imbalance, then share what is left among the rest.
            fix_low = sum(low[c] - proposal[c] for c in below)
            fix_high = sum(proposal[c] - high[c] for c in above)
            fixed = below if fix_low >= fix_high else above
            for c in fixed:
                allocation[c] = low[c] if fixed is below else high[c]
                remaining -= allocation[c]
                free.remove(c)
        return {c: float(allocation[c]) for c in channels}

    def compare_methods(self, methods: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """
        Attribution shares from several methods side by side.

        The incremental methods and the path heuristics answer different
        questions, so disagreement between the two groups is expected. The
        fitted state of the object is left as it was.
        """
        methods = list(methods or INCREMENTAL_METHODS + PATH_METHODS)
        saved = {name: getattr(self, name) for name in (
            '_attribution', '_contributions', '_baseline', '_method', '_model',
            '_interactions', '_incremental_effects')}
        results = {}
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='No path_col was given')
                for method in methods:
                    self.fit(method=method, **kwargs)
                    results[method] = self._attribution.copy()
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
        if self._paths is None and any(m in PATH_METHODS for m in methods):
            warnings.warn(
                "No path_col was given, so the path heuristics took the order of "
                "touchpoint_cols as the order of contact.",
                UserWarning, stacklevel=2
            )
        return pd.DataFrame(results)

    def plot_attribution(self, ax=None):
        """Plot attribution shares."""
        import matplotlib.pyplot as plt

        self._require_fit()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        touchpoints = list(self._attribution.keys())
        weights = list(self._attribution.values())
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(touchpoints)))
        ax.barh(touchpoints, weights, color=colors)
        ax.set_xlabel('Attribution share')
        ax.set_title(f'Marketing Attribution ({self._method})')
        return ax

    def summary(self) -> str:
        """Return attribution summary."""
        if self._attribution is None:
            return "Attribution not fitted. Call fit() first."

        rate = self.data[self.conversion_col].mean()
        lines = [
            "Marketing Attribution Summary",
            "=" * 50,
            f"Method: {self._method}",
            f"Touchpoints: {len(self.touchpoint_cols)}",
            f"Total records: {len(self.data)}",
            f"Mean of {self.conversion_col}: {rate:.4f}",
        ]
        if self._baseline is not None:
            lines.append(f"Without any touchpoint (modelled): {self._baseline:.4f}")
        else:
            lines.append("Path heuristic: describes journeys, not what a channel caused.")
        lines += ["", "Attribution shares:"]
        effects = self.get_incremental_effects()
        for tp, weight in sorted(self._attribution.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {tp}: {weight:.1%} (effect of an exposure: {effects[tp]:+.4f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._attribution is not None else "unfitted"
        return f"MarketingAttribution(touchpoints={len(self.touchpoint_cols)}, {status})"
