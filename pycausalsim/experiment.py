"""
Experiment Analysis with Causal Inference.

Analyze A/B tests with proper causal inference methods including
heterogeneous treatment effects and long-term impact simulation.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import warnings


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
    
    def summary(self) -> str:
        lines = [
            "Treatment Effect Estimate",
            "=" * 40,
            f"Method: {self.method}",
            f"Estimate: {self.estimate:.4f}",
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"Std Error: {self.std_error:.4f}",
            f"P-value: {self.p_value:.4f}",
            f"Significant: {'Yes' if self.p_value < 0.05 else 'No'}",
            "",
            f"Treated: {self.n_treated}",
            f"Control: {self.n_control}",
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"ExperimentEffect(estimate={self.estimate:.4f}, p={self.p_value:.4f})"


@dataclass
class HeterogeneousEffects:
    """Results from heterogeneous treatment effect analysis."""
    covariate: str
    segments: Dict[str, float]
    segment_cis: Dict[str, Tuple[float, float]]
    interaction_pvalue: float
    
    def plot(self, ax=None):
        """Plot heterogeneous effects."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        segments = list(self.segments.keys())
        effects = list(self.segments.values())
        errors = [(self.segment_cis[s][1] - self.segment_cis[s][0]) / 2 
                  for s in segments]
        
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
    Analyze A/B tests with causal inference methods.
    
    Goes beyond simple t-tests to provide:
    - Doubly robust estimation
    - Heterogeneous treatment effects
    - Long-term impact simulation
    - Covariate adjustment
    
    Parameters
    ----------
    data : pd.DataFrame
        Experiment data
    treatment : str
        Name of treatment assignment column
    outcome : str
        Name of outcome column
    covariates : list of str, optional
        Covariates for adjustment
        
    Examples
    --------
    >>> exp = ExperimentAnalysis(
    ...     data=ab_test_data,
    ...     treatment='new_feature',
    ...     outcome='engagement'
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
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates or []
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self._validate_inputs()
        
        # Compute basic stats
        self._treated = self.data[self.data[treatment] == 1]
        self._control = self.data[self.data[treatment] == 0]
    
    def _validate_inputs(self):
        """Validate inputs."""
        if self.treatment not in self.data.columns:
            raise ValueError(f"Treatment '{self.treatment}' not found")
        
        if self.outcome not in self.data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found")
        
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not found")
    
    def estimate_effect(
        self,
        method: str = 'difference',
        **kwargs
    ) -> ExperimentEffect:
        """
        Estimate average treatment effect.
        
        Parameters
        ----------
        method : str
            Estimation method:
            - 'difference': Simple difference in means
            - 'ols': OLS regression with covariates
            - 'ipw': Inverse probability weighting
            - 'dr': Doubly robust (AIPW)
            - 'matching': Propensity score matching
            
        Returns
        -------
        ExperimentEffect
            Treatment effect estimate with confidence interval
        """
        methods = {
            'difference': self._difference_in_means,
            'ols': self._ols_estimator,
            'ipw': self._ipw_estimator,
            'dr': self._doubly_robust,
            'aipw': self._doubly_robust,  # Alias
            'matching': self._matching_estimator,
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Unknown method: {method}")
        
        return methods[method.lower()](**kwargs)
    
    def _difference_in_means(self, **kwargs) -> ExperimentEffect:
        """Simple difference in means estimator."""
        y1 = self._treated[self.outcome].values
        y0 = self._control[self.outcome].values
        
        ate = y1.mean() - y0.mean()
        
        # Standard error
        var1 = y1.var() / len(y1)
        var0 = y0.var() / len(y0)
        se = np.sqrt(var1 + var0)
        
        # Confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        # P-value (two-sided)
        t_stat = ate / se if se > 0 else 0
        df = len(y1) + len(y0) - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return ExperimentEffect(
            estimate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            std_error=se,
            method='difference_in_means',
            n_treated=len(y1),
            n_control=len(y0)
        )
    
    def _ols_estimator(self, **kwargs) -> ExperimentEffect:
        """OLS regression with covariate adjustment."""
        from sklearn.linear_model import LinearRegression
        
        # Build design matrix
        X_vars = [self.treatment] + self.covariates
        X = self.data[X_vars].values
        y = self.data[self.outcome].values
        
        # Fit OLS
        model = LinearRegression()
        model.fit(X, y)
        
        # Treatment effect is coefficient on treatment
        ate = model.coef_[0]
        
        # Bootstrap for standard error
        n_boot = 500
        boot_effects = []
        n = len(self.data)
        
        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            model_boot = LinearRegression()
            model_boot.fit(X_boot, y_boot)
            boot_effects.append(model_boot.coef_[0])
        
        se = np.std(boot_effects)
        ci_lower, ci_upper = np.percentile(boot_effects, [2.5, 97.5])
        
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return ExperimentEffect(
            estimate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            std_error=se,
            method='ols',
            n_treated=len(self._treated),
            n_control=len(self._control)
        )
    
    def _ipw_estimator(self, **kwargs) -> ExperimentEffect:
        """Inverse probability weighting estimator."""
        from sklearn.linear_model import LogisticRegression
        
        if not self.covariates:
            # Without covariates, IPW = difference in means
            return self._difference_in_means()
        
        X = self.data[self.covariates].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X, T)
        propensity = ps_model.predict_proba(X)[:, 1]
        
        # Clip propensity scores for stability
        propensity = np.clip(propensity, 0.01, 0.99)
        
        # IPW estimator
        weights_treated = T / propensity
        weights_control = (1 - T) / (1 - propensity)
        
        y1_weighted = np.sum(weights_treated * Y) / np.sum(weights_treated)
        y0_weighted = np.sum(weights_control * Y) / np.sum(weights_control)
        
        ate = y1_weighted - y0_weighted
        
        # Bootstrap for inference
        n_boot = 500
        boot_effects = []
        n = len(self.data)
        
        for _ in range(n_boot):
            idx = np.random.choice(n, size=n, replace=True)
            X_b, T_b, Y_b = X[idx], T[idx], Y[idx]
            
            ps_b = LogisticRegression(random_state=self.random_state)
            ps_b.fit(X_b, T_b)
            prop_b = np.clip(ps_b.predict_proba(X_b)[:, 1], 0.01, 0.99)
            
            w1 = T_b / prop_b
            w0 = (1 - T_b) / (1 - prop_b)
            
            y1_w = np.sum(w1 * Y_b) / np.sum(w1)
            y0_w = np.sum(w0 * Y_b) / np.sum(w0)
            boot_effects.append(y1_w - y0_w)
        
        se = np.std(boot_effects)
        ci_lower, ci_upper = np.percentile(boot_effects, [2.5, 97.5])
        p_value = 2 * (1 - stats.norm.cdf(abs(ate / se))) if se > 0 else 1.0
        
        return ExperimentEffect(
            estimate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            std_error=se,
            method='ipw',
            n_treated=len(self._treated),
            n_control=len(self._control)
        )
    
    def _doubly_robust(self, **kwargs) -> ExperimentEffect:
        """
        Doubly robust (AIPW) estimator.
        
        Consistent if either propensity score OR outcome model is correct.
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        
        if not self.covariates:
            return self._difference_in_means()
        
        X = self.data[self.covariates].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        n = len(Y)
        
        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X, T)
        propensity = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
        
        # Estimate outcome models
        # mu1(x) = E[Y | X, T=1]
        # mu0(x) = E[Y | X, T=0]
        
        treated_idx = T == 1
        control_idx = T == 0
        
        mu1_model = Ridge(alpha=1.0)
        mu1_model.fit(X[treated_idx], Y[treated_idx])
        mu1 = mu1_model.predict(X)
        
        mu0_model = Ridge(alpha=1.0)
        mu0_model.fit(X[control_idx], Y[control_idx])
        mu0 = mu0_model.predict(X)
        
        # AIPW estimator
        # tau_AIPW = 1/n * sum[ mu1(X) - mu0(X) + T(Y - mu1(X))/e(X) - (1-T)(Y - mu0(X))/(1-e(X)) ]
        
        term1 = mu1 - mu0
        term2 = T * (Y - mu1) / propensity
        term3 = (1 - T) * (Y - mu0) / (1 - propensity)
        
        psi = term1 + term2 - term3
        ate = np.mean(psi)
        
        # Influence function-based standard error
        se = np.std(psi) / np.sqrt(n)
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return ExperimentEffect(
            estimate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            std_error=se,
            method='doubly_robust',
            n_treated=int(treated_idx.sum()),
            n_control=int(control_idx.sum())
        )
    
    def _matching_estimator(self, n_neighbors: int = 5, **kwargs) -> ExperimentEffect:
        """Propensity score matching estimator."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        if not self.covariates:
            return self._difference_in_means()
        
        X = self.data[self.covariates].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        # Estimate propensity scores
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X, T)
        propensity = ps_model.predict_proba(X)[:, 1].reshape(-1, 1)
        
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        # Match treated to control
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(propensity[control_idx])
        
        matched_effects = []
        for i in treated_idx:
            _, indices = nn.kneighbors(propensity[i].reshape(1, -1))
            matched_control_idx = control_idx[indices[0]]
            
            y_treated = Y[i]
            y_control = Y[matched_control_idx].mean()
            matched_effects.append(y_treated - y_control)
        
        ate = np.mean(matched_effects)
        se = np.std(matched_effects) / np.sqrt(len(matched_effects))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return ExperimentEffect(
            estimate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            std_error=se,
            method='matching',
            n_treated=len(treated_idx),
            n_control=len(control_idx)
        )
    
    def analyze_heterogeneity(
        self,
        covariates: Optional[List[str]] = None,
        n_quantiles: int = 4
    ) -> List[HeterogeneousEffects]:
        """
        Analyze heterogeneous treatment effects.
        
        Parameters
        ----------
        covariates : list of str, optional
            Covariates to analyze heterogeneity on
        n_quantiles : int
            Number of quantiles for continuous covariates
            
        Returns
        -------
        list of HeterogeneousEffects
            Heterogeneous effects for each covariate
        """
        covariates = covariates or self.covariates
        results = []
        
        for cov in covariates:
            het = self._analyze_heterogeneity_single(cov, n_quantiles)
            results.append(het)
        
        return results
    
    def _analyze_heterogeneity_single(
        self,
        covariate: str,
        n_quantiles: int
    ) -> HeterogeneousEffects:
        """Analyze heterogeneity on single covariate."""
        original_covariate = covariate  # Remember original name
        
        # Determine if categorical or continuous
        unique_vals = self.data[covariate].nunique()
        
        if unique_vals <= n_quantiles:
            # Categorical: use actual values
            segments = self.data[covariate].unique()
        else:
            # Continuous: create quantile bins
            self.data['_temp_bin'] = pd.qcut(
                self.data[covariate], 
                q=n_quantiles, 
                labels=False, 
                duplicates='drop'
            )
            segments = self.data['_temp_bin'].unique()
            covariate = '_temp_bin'
        
        segment_effects = {}
        segment_cis = {}
        
        for seg in segments:
            subset = self.data[self.data[covariate] == seg]
            
            if len(subset) < 10:
                continue
            
            treated = subset[subset[self.treatment] == 1][self.outcome]
            control = subset[subset[self.treatment] == 0][self.outcome]
            
            if len(treated) > 0 and len(control) > 0:
                effect = treated.mean() - control.mean()
                se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
                
                segment_effects[str(seg)] = effect
                segment_cis[str(seg)] = (effect - 1.96*se, effect + 1.96*se)
        
        # Clean up temp column
        if '_temp_bin' in self.data.columns:
            self.data.drop('_temp_bin', axis=1, inplace=True)
        
        # Test for interaction
        interaction_pvalue = self._test_interaction(original_covariate)
        
        return HeterogeneousEffects(
            covariate=original_covariate,
            segments=segment_effects,
            segment_cis=segment_cis,
            interaction_pvalue=interaction_pvalue
        )
    
    def _test_interaction(self, covariate: str) -> float:
        """Test for treatment-covariate interaction."""
        from sklearn.linear_model import LinearRegression
        
        # Model with interaction
        data = self.data.copy()
        data['_interaction'] = data[self.treatment] * data[covariate]
        
        X_full = data[[self.treatment, covariate, '_interaction']].values
        X_reduced = data[[self.treatment, covariate]].values
        y = data[self.outcome].values
        
        # F-test for interaction term
        model_full = LinearRegression().fit(X_full, y)
        model_reduced = LinearRegression().fit(X_reduced, y)
        
        rss_full = np.sum((y - model_full.predict(X_full)) ** 2)
        rss_reduced = np.sum((y - model_reduced.predict(X_reduced)) ** 2)
        
        n = len(y)
        p_full = 3
        p_reduced = 2
        
        f_stat = ((rss_reduced - rss_full) / (p_full - p_reduced)) / (rss_full / (n - p_full))
        p_value = 1 - stats.f.cdf(f_stat, p_full - p_reduced, n - p_full)
        
        return p_value
    
    def simulate_longterm(
        self,
        periods: int = 90,
        decay_rate: float = 0.1
    ) -> pd.DataFrame:
        """
        Simulate long-term treatment effects.
        
        Parameters
        ----------
        periods : int
            Number of periods to simulate
        decay_rate : float
            Rate at which treatment effect decays
            
        Returns
        -------
        pd.DataFrame
            Simulated outcomes over time
        """
        # Get initial effect
        effect = self.estimate_effect()
        initial_effect = effect.estimate
        
        # Simulate with decay
        results = []
        for t in range(periods):
            # Effect with exponential decay
            current_effect = initial_effect * np.exp(-decay_rate * t)
            
            # Add noise
            noise = np.random.normal(0, effect.std_error)
            
            results.append({
                'period': t,
                'effect': current_effect + noise,
                'cumulative_effect': sum(r['effect'] for r in results) + current_effect + noise
            })
        
        return pd.DataFrame(results)
    
    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8
    ) -> int:
        """
        Compute required sample size for given effect size.
        
        Parameters
        ----------
        effect_size : float
            Expected treatment effect
        alpha : float
            Significance level
        power : float
            Desired statistical power
            
        Returns
        -------
        int
            Required sample size per group
        """
        # Pooled standard deviation
        y = self.data[self.outcome].values
        sigma = np.std(y)
        
        # Standardized effect size
        d = effect_size / sigma
        
        # Required sample size per group
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / d) ** 2
        
        return int(np.ceil(n))
    
    def summary(self) -> str:
        """Return experiment summary."""
        effect = self.estimate_effect()
        
        lines = [
            "Experiment Analysis Summary",
            "=" * 50,
            f"Treatment: {self.treatment}",
            f"Outcome: {self.outcome}",
            "",
            f"Treated group: n={len(self._treated)}, mean={self._treated[self.outcome].mean():.4f}",
            f"Control group: n={len(self._control)}, mean={self._control[self.outcome].mean():.4f}",
            "",
            "Treatment Effect:",
            f"  Estimate: {effect.estimate:.4f}",
            f"  95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]",
            f"  P-value: {effect.p_value:.4f}",
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (f"ExperimentAnalysis(treatment='{self.treatment}', "
                f"n_treated={len(self._treated)}, n_control={len(self._control)})")
