"""
Utility functions for PyCausalSim.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
import warnings


def simulate_data(
    n_samples: int = 1000,
    n_features: int = 5,
    n_confounders: int = 2,
    treatment_effect: float = 0.5,
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Generate synthetic data with known causal structure.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features (excluding confounders and outcome)
    n_confounders : int
        Number of confounding variables
    treatment_effect : float
        True causal effect of treatment on outcome
    noise_level : float
        Standard deviation of noise
    random_state : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (DataFrame with data, dict with true causal graph)
        
    Examples
    --------
    >>> data, true_graph = simulate_data(n_samples=1000, treatment_effect=0.3)
    >>> simulator = CausalSimulator(data, target='outcome')
    >>> simulator.discover_graph()
    >>> # Compare discovered graph to true_graph
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate confounders
    confounders = {}
    for i in range(n_confounders):
        confounders[f'confounder_{i}'] = np.random.randn(n_samples)
    
    # Generate treatment (affected by confounders)
    treatment = np.zeros(n_samples)
    for c in confounders.values():
        treatment += 0.3 * c
    treatment += np.random.randn(n_samples) * noise_level
    
    # Generate features
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.randn(n_samples)
        # Some features affected by confounders
        if i < n_confounders:
            features[f'feature_{i}'] += 0.2 * confounders[f'confounder_{i}']
    
    # Generate outcome
    outcome = treatment_effect * treatment
    for c in confounders.values():
        outcome += 0.3 * c  # Confounders affect outcome
    for i, f in enumerate(features.values()):
        outcome += 0.1 * f if i < 2 else 0  # First 2 features affect outcome
    outcome += np.random.randn(n_samples) * noise_level
    
    # Create DataFrame
    data = pd.DataFrame(confounders)
    data['treatment'] = treatment
    data.update(pd.DataFrame(features))
    data['outcome'] = outcome
    
    # True causal graph
    true_graph = {
        'outcome': ['treatment'] + list(confounders.keys()) + [f'feature_{i}' for i in range(min(2, n_features))],
        'treatment': list(confounders.keys()),
    }
    for i in range(n_confounders):
        true_graph[f'confounder_{i}'] = []
        if i < n_features:
            true_graph[f'feature_{i}'] = [f'confounder_{i}']
    for i in range(n_confounders, n_features):
        true_graph[f'feature_{i}'] = []
    
    return data, true_graph


def evaluate_graph(
    discovered: Dict[str, List[str]],
    true: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Evaluate discovered graph against ground truth.
    
    Parameters
    ----------
    discovered : dict
        Discovered causal graph
    true : dict
        True causal graph
        
    Returns
    -------
    dict
        Evaluation metrics (precision, recall, F1, SHD)
    """
    # Extract all edges
    discovered_edges = set()
    for child, parents in discovered.items():
        for parent in parents:
            discovered_edges.add((parent, child))
    
    true_edges = set()
    for child, parents in true.items():
        for parent in parents:
            true_edges.add((parent, child))
    
    # Calculate metrics
    true_positives = len(discovered_edges & true_edges)
    false_positives = len(discovered_edges - true_edges)
    false_negatives = len(true_edges - discovered_edges)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural Hamming Distance
    shd = false_positives + false_negatives
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'shd': shd,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def bootstrap_effect(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for treatment effect.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    confounders : list, optional
        Confounding variables
    n_bootstrap : int
        Number of bootstrap samples
    ci_level : float
        Confidence level
        
    Returns
    -------
    tuple
        (point_estimate, ci_lower, ci_upper)
    """
    from sklearn.linear_model import LinearRegression
    
    confounders = confounders or []
    
    effects = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n, size=n, replace=True)
        sample = data.iloc[idx]
        
        # Fit regression
        X_vars = [treatment] + confounders
        X = sample[X_vars].values
        y = sample[outcome].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        effects.append(model.coef_[0])
    
    point_estimate = np.mean(effects)
    alpha = (1 - ci_level) / 2
    ci_lower, ci_upper = np.percentile(effects, [alpha * 100, (1 - alpha) * 100])
    
    return point_estimate, ci_lower, ci_upper


def propensity_score(
    data: pd.DataFrame,
    treatment: str,
    confounders: List[str]
) -> np.ndarray:
    """
    Estimate propensity scores.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    treatment : str
        Treatment variable
    confounders : list
        Confounding variables
        
    Returns
    -------
    np.ndarray
        Propensity scores
    """
    from sklearn.linear_model import LogisticRegression
    
    X = data[confounders].values
    T = data[treatment].values
    
    model = LogisticRegression()
    model.fit(X, T)
    
    return model.predict_proba(X)[:, 1]


def check_balance(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str]
) -> pd.DataFrame:
    """
    Check covariate balance between treatment groups.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    treatment : str
        Treatment variable
    covariates : list
        Covariates to check
        
    Returns
    -------
    pd.DataFrame
        Balance statistics
    """
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]
    
    results = []
    for cov in covariates:
        mean_t = treated[cov].mean()
        mean_c = control[cov].mean()
        std_pooled = np.sqrt((treated[cov].var() + control[cov].var()) / 2)
        
        smd = (mean_t - mean_c) / std_pooled if std_pooled > 0 else 0
        
        results.append({
            'covariate': cov,
            'mean_treated': mean_t,
            'mean_control': mean_c,
            'std_diff': smd,
            'balanced': abs(smd) < 0.1
        })
    
    return pd.DataFrame(results)


def standardize(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Standardize columns to mean 0, std 1.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    columns : list, optional
        Columns to standardize (default: all numeric)
        
    Returns
    -------
    pd.DataFrame
        Standardized data
    """
    result = data.copy()
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        mean = result[col].mean()
        std = result[col].std()
        if std > 0:
            result[col] = (result[col] - mean) / std
    
    return result


def encode_categorical(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    columns : list, optional
        Columns to encode (default: all object/category)
        
    Returns
    -------
    pd.DataFrame
        Encoded data
    """
    result = data.copy()
    
    if columns is None:
        columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in columns:
        dummies = pd.get_dummies(result[col], prefix=col, drop_first=True)
        result = pd.concat([result.drop(col, axis=1), dummies], axis=1)
    
    return result


def detect_confounders(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    threshold: float = 0.1
) -> List[str]:
    """
    Detect potential confounders based on correlation.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
    threshold : float
        Correlation threshold
        
    Returns
    -------
    list
        Potential confounders
    """
    confounders = []
    
    for col in data.columns:
        if col in [treatment, outcome]:
            continue
        
        # Check correlation with both treatment and outcome
        corr_t = abs(np.corrcoef(data[col], data[treatment])[0, 1])
        corr_y = abs(np.corrcoef(data[col], data[outcome])[0, 1])
        
        if corr_t > threshold and corr_y > threshold:
            confounders.append(col)
    
    return confounders


def compute_ate_bounds(
    data: pd.DataFrame,
    treatment: str,
    outcome: str
) -> Tuple[float, float]:
    """
    Compute no-assumption bounds on ATE.
    
    Uses Manski bounds for partial identification.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data
    treatment : str
        Treatment variable
    outcome : str
        Outcome variable
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    y_min = data[outcome].min()
    y_max = data[outcome].max()
    
    treated = data[data[treatment] == 1][outcome]
    control = data[data[treatment] == 0][outcome]
    
    p_treat = len(treated) / len(data)
    p_control = 1 - p_treat
    
    # Lower bound
    lower = (p_treat * treated.mean() + p_control * y_min) - \
            (p_treat * y_max + p_control * control.mean())
    
    # Upper bound  
    upper = (p_treat * treated.mean() + p_control * y_max) - \
            (p_treat * y_min + p_control * control.mean())
    
    return lower, upper


def create_example_data() -> pd.DataFrame:
    """
    Create example e-commerce dataset.
    
    Returns
    -------
    pd.DataFrame
        Example dataset with realistic e-commerce metrics
    """
    np.random.seed(42)
    n = 10000
    
    # User characteristics (confounders)
    traffic_source = np.random.choice(['organic', 'paid', 'social', 'direct'], n, 
                                      p=[0.3, 0.25, 0.2, 0.25])
    device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n,
                                   p=[0.5, 0.35, 0.15])
    user_tenure = np.random.exponential(180, n)  # Days
    
    # Treatment: page load time (seconds)
    page_load_time = 2 + np.random.exponential(1.5, n)
    # Faster on desktop
    page_load_time[device_type == 'desktop'] *= 0.8
    
    # Price
    base_price = np.random.uniform(10, 100, n)
    
    # Outcome: conversion
    # True causal effects:
    # - page_load_time: negative effect (-0.05 per second)
    # - price: negative effect (-0.002 per dollar)
    # - traffic_source affects base rate
    # - device affects base rate
    
    logit = -1.5
    logit += -0.1 * page_load_time  # Slower = less conversion
    logit += -0.01 * base_price  # Higher price = less conversion
    logit += 0.3 * (traffic_source == 'paid')  # Paid traffic converts better
    logit += 0.2 * (device_type == 'desktop')  # Desktop converts better
    logit += 0.001 * np.minimum(user_tenure, 365)  # Tenure helps
    logit += np.random.normal(0, 0.3, n)  # Noise
    
    conversion_prob = 1 / (1 + np.exp(-logit))
    conversion = np.random.binomial(1, conversion_prob)
    
    # Revenue (for converters)
    revenue = conversion * base_price * np.random.uniform(0.8, 1.2, n)
    
    return pd.DataFrame({
        'traffic_source': traffic_source,
        'device_type': device_type,
        'user_tenure': user_tenure,
        'page_load_time': page_load_time,
        'price': base_price,
        'conversion': conversion,
        'revenue': revenue
    })


class CausalEffectEstimator:
    """
    Unified interface for causal effect estimation.
    
    Wraps multiple estimation methods with a common interface.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: Optional[List[str]] = None
    ):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders or []
        
        self._estimate = None
        self._ci = None
        self._method = None
    
    def estimate(
        self,
        method: str = 'ols',
        **kwargs
    ) -> float:
        """
        Estimate causal effect.
        
        Parameters
        ----------
        method : str
            Estimation method: 'ols', 'ipw', 'dr', 'matching'
        **kwargs
            Method-specific arguments
            
        Returns
        -------
        float
            Point estimate
        """
        if method == 'ols':
            return self._ols_estimate()
        elif method == 'ipw':
            return self._ipw_estimate()
        elif method == 'dr':
            return self._dr_estimate()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _ols_estimate(self) -> float:
        """OLS regression estimate."""
        from sklearn.linear_model import LinearRegression
        
        X_vars = [self.treatment] + self.confounders
        X = self.data[X_vars].values
        y = self.data[self.outcome].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        self._estimate = model.coef_[0]
        return self._estimate
    
    def _ipw_estimate(self) -> float:
        """Inverse probability weighting estimate."""
        from sklearn.linear_model import LogisticRegression
        
        if not self.confounders:
            return self._ols_estimate()
        
        # Propensity score
        X = self.data[self.confounders].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        ps = LogisticRegression().fit(X, T).predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)
        
        # IPW estimator
        w1 = T / ps
        w0 = (1 - T) / (1 - ps)
        
        y1 = np.sum(w1 * Y) / np.sum(w1)
        y0 = np.sum(w0 * Y) / np.sum(w0)
        
        self._estimate = y1 - y0
        return self._estimate
    
    def _dr_estimate(self) -> float:
        """Doubly robust estimate."""
        from sklearn.linear_model import LogisticRegression, Ridge
        
        if not self.confounders:
            return self._ols_estimate()
        
        X = self.data[self.confounders].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        # Propensity score
        ps = LogisticRegression().fit(X, T).predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)
        
        # Outcome models
        mu1 = Ridge().fit(X[T == 1], Y[T == 1]).predict(X)
        mu0 = Ridge().fit(X[T == 0], Y[T == 0]).predict(X)
        
        # AIPW
        psi = mu1 - mu0 + T * (Y - mu1) / ps - (1 - T) * (Y - mu0) / (1 - ps)
        
        self._estimate = np.mean(psi)
        return self._estimate
    
    def confidence_interval(self, n_bootstrap: int = 500, ci_level: float = 0.95) -> Tuple[float, float]:
        """
        Get bootstrap confidence interval.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level
            
        Returns
        -------
        tuple
            (lower, upper)
        """
        _, lower, upper = bootstrap_effect(
            self.data, self.treatment, self.outcome,
            self.confounders, n_bootstrap, ci_level
        )
        self._ci = (lower, upper)
        return self._ci
