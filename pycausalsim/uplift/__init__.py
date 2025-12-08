"""
Uplift Modeling for Treatment Effect Heterogeneity.

Identify WHO will respond to treatments by estimating
conditional average treatment effects (CATE).
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class UpliftSegment:
    """A segment of users based on predicted uplift."""
    name: str
    size: int
    predicted_uplift: float
    actual_uplift: float
    conversion_treated: float
    conversion_control: float


class UpliftModeler:
    """
    Uplift modeling to identify treatment-responsive individuals.
    
    Segments users into:
    - Persuadables: Will convert only if treated
    - Sure Things: Will convert regardless
    - Lost Causes: Won't convert regardless
    - Sleeping Dogs: Treatment hurts them
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical data with treatment and outcome
    treatment : str
        Treatment assignment column
    outcome : str
        Outcome column
    features : list of str, optional
        Features for modeling
        
    Examples
    --------
    >>> uplift = UpliftModeler(data, treatment='campaign', outcome='conversion')
    >>> uplift.fit()
    >>> segments = uplift.segment_by_effect()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        features: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.features = features or self._detect_features()
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self._validate_inputs()
        
        # Fitted components
        self._model = None
        self._method = None
        self._uplift_scores = None
        self._is_fitted = False
    
    def _detect_features(self) -> List[str]:
        """Auto-detect feature columns."""
        exclude = {self.treatment, self.outcome}
        return [c for c in self.data.columns if c not in exclude]
    
    def _validate_inputs(self):
        """Validate inputs."""
        if self.treatment not in self.data.columns:
            raise ValueError(f"Treatment '{self.treatment}' not found")
        
        if self.outcome not in self.data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found")
    
    def fit(
        self,
        method: str = 'two_model',
        **kwargs
    ) -> 'UpliftModeler':
        """
        Fit uplift model.
        
        Parameters
        ----------
        method : str
            Uplift method:
            - 'two_model': Separate models for treated/control
            - 'transformed': Transformed outcome approach
            - 'x_learner': X-learner meta-algorithm
            - 'causal_forest': Causal forest (requires econml)
            
        Returns
        -------
        self
        """
        methods = {
            'two_model': self._fit_two_model,
            't_learner': self._fit_two_model,  # Alias
            'transformed': self._fit_transformed,
            'x_learner': self._fit_x_learner,
            'causal_forest': self._fit_causal_forest,
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Unknown method: {method}")
        
        self._method = method.lower()
        methods[self._method](**kwargs)
        
        self._is_fitted = True
        return self
    
    def _fit_two_model(self, **kwargs):
        """
        Two-model (T-learner) approach.
        
        Fits separate models for treatment and control groups.
        CATE = E[Y|X, T=1] - E[Y|X, T=0]
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        
        X = self.data[self.features].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        treated_idx = T == 1
        control_idx = T == 0
        
        # Determine if classification or regression
        unique_y = np.unique(Y)
        is_classification = len(unique_y) == 2
        
        if is_classification:
            model_class = GradientBoostingClassifier
        else:
            model_class = GradientBoostingRegressor
        
        # Fit treatment model
        model_t = model_class(n_estimators=100, random_state=self.random_state)
        model_t.fit(X[treated_idx], Y[treated_idx])
        
        # Fit control model  
        model_c = model_class(n_estimators=100, random_state=self.random_state)
        model_c.fit(X[control_idx], Y[control_idx])
        
        self._model = {'treated': model_t, 'control': model_c}
        
        # Predict uplift
        if is_classification:
            pred_t = model_t.predict_proba(X)[:, 1]
            pred_c = model_c.predict_proba(X)[:, 1]
        else:
            pred_t = model_t.predict(X)
            pred_c = model_c.predict(X)
        
        self._uplift_scores = pred_t - pred_c
    
    def _fit_transformed(self, **kwargs):
        """
        Transformed outcome approach.
        
        Creates a transformed outcome that, when predicted, gives CATE.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = self.data[self.features].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        # Estimate propensity score
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X, T)
        propensity = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
        
        # Transform outcome
        # Z = Y * (T - e(X)) / (e(X) * (1 - e(X)))
        transformed_y = Y * (T - propensity) / (propensity * (1 - propensity))
        
        # Fit model on transformed outcome
        model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(X, transformed_y)
        
        self._model = model
        self._uplift_scores = model.predict(X)
    
    def _fit_x_learner(self, **kwargs):
        """
        X-learner meta-algorithm.
        
        More efficient when treatment/control groups are imbalanced.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = self.data[self.features].values
        T = self.data[self.treatment].values
        Y = self.data[self.outcome].values
        
        treated_idx = T == 1
        control_idx = T == 0
        
        # Stage 1: Fit response functions
        model_t1 = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model_t1.fit(X[treated_idx], Y[treated_idx])
        
        model_c1 = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model_c1.fit(X[control_idx], Y[control_idx])
        
        # Stage 2: Impute counterfactuals
        # For treated: tau_1 = Y - mu_0(X)
        tau_t = Y[treated_idx] - model_c1.predict(X[treated_idx])
        
        # For control: tau_0 = mu_1(X) - Y
        tau_c = model_t1.predict(X[control_idx]) - Y[control_idx]
        
        # Stage 3: Fit CATE models
        model_tau_t = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model_tau_t.fit(X[treated_idx], tau_t)
        
        model_tau_c = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        model_tau_c.fit(X[control_idx], tau_c)
        
        # Propensity-weighted combination
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X, T)
        propensity = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
        
        tau_x_t = model_tau_t.predict(X)
        tau_x_c = model_tau_c.predict(X)
        
        self._uplift_scores = propensity * tau_x_c + (1 - propensity) * tau_x_t
        self._model = {
            'mu_t': model_t1, 'mu_c': model_c1,
            'tau_t': model_tau_t, 'tau_c': model_tau_c,
            'propensity': ps_model
        }
    
    def _fit_causal_forest(self, **kwargs):
        """
        Causal forest using econml.
        """
        try:
            from econml.dml import CausalForestDML
        except ImportError:
            warnings.warn("econml not installed, falling back to two_model")
            return self._fit_two_model(**kwargs)
        
        X = self.data[self.features].values
        T = self.data[self.treatment].values.reshape(-1, 1)
        Y = self.data[self.outcome].values
        
        model = CausalForestDML(random_state=self.random_state)
        model.fit(Y, T, X=X)
        
        self._model = model
        self._uplift_scores = model.effect(X).flatten()
    
    def predict_uplift(
        self,
        X: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predict uplift scores.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Features to predict on. If None, uses training data.
            
        Returns
        -------
        np.ndarray
            Predicted uplift scores
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        if X is None:
            return self._uplift_scores
        
        X_vals = X[self.features].values
        
        if self._method == 'two_model':
            model_t = self._model['treated']
            model_c = self._model['control']
            
            if hasattr(model_t, 'predict_proba'):
                pred_t = model_t.predict_proba(X_vals)[:, 1]
                pred_c = model_c.predict_proba(X_vals)[:, 1]
            else:
                pred_t = model_t.predict(X_vals)
                pred_c = model_c.predict(X_vals)
            
            return pred_t - pred_c
        
        elif self._method == 'transformed':
            return self._model.predict(X_vals)
        
        elif self._method == 'x_learner':
            ps = self._model['propensity'].predict_proba(X_vals)[:, 1]
            tau_t = self._model['tau_t'].predict(X_vals)
            tau_c = self._model['tau_c'].predict(X_vals)
            return ps * tau_c + (1 - ps) * tau_t
        
        elif self._method == 'causal_forest':
            return self._model.effect(X_vals).flatten()
    
    def segment_by_effect(
        self,
        n_segments: int = 4
    ) -> List[UpliftSegment]:
        """
        Segment users by predicted treatment effect.
        
        Parameters
        ----------
        n_segments : int
            Number of segments (default 4 for classic segmentation)
            
        Returns
        -------
        list of UpliftSegment
            User segments with statistics
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        # Add uplift scores to data
        data = self.data.copy()
        data['_uplift'] = self._uplift_scores
        
        # Create quantile-based segments
        data['_segment'] = pd.qcut(
            data['_uplift'], 
            q=n_segments, 
            labels=False, 
            duplicates='drop'
        )
        
        segments = []
        segment_names = ['Sleeping Dogs', 'Lost Causes', 'Sure Things', 'Persuadables']
        
        if n_segments != 4:
            segment_names = [f'Segment {i+1}' for i in range(n_segments)]
        
        for seg_id in sorted(data['_segment'].unique()):
            seg_data = data[data['_segment'] == seg_id]
            
            treated = seg_data[seg_data[self.treatment] == 1]
            control = seg_data[seg_data[self.treatment] == 0]
            
            conv_t = treated[self.outcome].mean() if len(treated) > 0 else 0
            conv_c = control[self.outcome].mean() if len(control) > 0 else 0
            
            name = segment_names[min(seg_id, len(segment_names)-1)]
            
            segments.append(UpliftSegment(
                name=name,
                size=len(seg_data),
                predicted_uplift=seg_data['_uplift'].mean(),
                actual_uplift=conv_t - conv_c,
                conversion_treated=conv_t,
                conversion_control=conv_c
            ))
        
        return segments
    
    def get_top_k(
        self,
        k: int,
        X: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Get top k individuals by predicted uplift.
        
        Parameters
        ----------
        k : int
            Number of individuals to return
        X : pd.DataFrame, optional
            Data to score (default: training data)
            
        Returns
        -------
        pd.DataFrame
            Top k individuals with uplift scores
        """
        if X is None:
            data = self.data.copy()
            data['_uplift'] = self._uplift_scores
        else:
            data = X.copy()
            data['_uplift'] = self.predict_uplift(X)
        
        return data.nlargest(k, '_uplift')
    
    def evaluate(
        self,
        test_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Evaluate uplift model performance.
        
        Parameters
        ----------
        test_data : pd.DataFrame, optional
            Test data for evaluation
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        if test_data is None:
            data = self.data.copy()
            data['_uplift'] = self._uplift_scores
        else:
            data = test_data.copy()
            data['_uplift'] = self.predict_uplift(test_data)
        
        # Qini coefficient
        qini = self._compute_qini(data)
        
        # AUUC (Area Under Uplift Curve)
        auuc = self._compute_auuc(data)
        
        # Actual ATE
        treated = data[data[self.treatment] == 1][self.outcome].mean()
        control = data[data[self.treatment] == 0][self.outcome].mean()
        ate = treated - control
        
        return {
            'qini_coefficient': qini,
            'auuc': auuc,
            'ate': ate,
            'mean_predicted_uplift': data['_uplift'].mean()
        }
    
    def _compute_qini(self, data: pd.DataFrame) -> float:
        """Compute Qini coefficient."""
        # Sort by predicted uplift
        sorted_data = data.sort_values('_uplift', ascending=False)
        
        n = len(sorted_data)
        n_t = (sorted_data[self.treatment] == 1).sum()
        n_c = (sorted_data[self.treatment] == 0).sum()
        
        qini_values = []
        cum_t = 0
        cum_c = 0
        cum_y_t = 0
        cum_y_c = 0
        
        for _, row in sorted_data.iterrows():
            if row[self.treatment] == 1:
                cum_t += 1
                cum_y_t += row[self.outcome]
            else:
                cum_c += 1
                cum_y_c += row[self.outcome]
            
            if cum_t > 0 and cum_c > 0:
                qini = (cum_y_t / cum_t - cum_y_c / cum_c) * (cum_t + cum_c) / n
                qini_values.append(qini)
        
        if not qini_values:
            return 0.0
        
        return np.trapz(qini_values) / len(qini_values)
    
    def _compute_auuc(self, data: pd.DataFrame) -> float:
        """Compute Area Under Uplift Curve."""
        # Sort by predicted uplift
        sorted_data = data.sort_values('_uplift', ascending=False)
        
        n = len(sorted_data)
        fractions = np.linspace(0.1, 1.0, 10)
        
        uplift_curve = []
        for frac in fractions:
            top_n = int(frac * n)
            top_data = sorted_data.head(top_n)
            
            treated = top_data[top_data[self.treatment] == 1][self.outcome]
            control = top_data[top_data[self.treatment] == 0][self.outcome]
            
            if len(treated) > 0 and len(control) > 0:
                uplift = treated.mean() - control.mean()
            else:
                uplift = 0
            
            uplift_curve.append(uplift)
        
        return np.trapz(uplift_curve, fractions)
    
    def plot_segments(self, ax=None):
        """Plot segment analysis."""
        import matplotlib.pyplot as plt
        
        segments = self.segment_by_effect()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        names = [s.name for s in segments]
        predicted = [s.predicted_uplift for s in segments]
        actual = [s.actual_uplift for s in segments]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, predicted, width, label='Predicted Uplift', alpha=0.7)
        ax.bar(x + width/2, actual, width, label='Actual Uplift', alpha=0.7)
        
        ax.set_xlabel('Segment')
        ax.set_ylabel('Uplift')
        ax.set_title('Uplift by Segment')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.axhline(0, color='gray', linestyle='--')
        ax.legend()
        
        return ax
    
    def plot_uplift_curve(self, ax=None):
        """Plot uplift curve."""
        import matplotlib.pyplot as plt
        
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        data = self.data.copy()
        data['_uplift'] = self._uplift_scores
        sorted_data = data.sort_values('_uplift', ascending=False)
        
        n = len(sorted_data)
        fractions = np.linspace(0.01, 1.0, 100)
        
        uplift_curve = []
        for frac in fractions:
            top_n = max(1, int(frac * n))
            top_data = sorted_data.head(top_n)
            
            treated = top_data[top_data[self.treatment] == 1][self.outcome]
            control = top_data[top_data[self.treatment] == 0][self.outcome]
            
            if len(treated) > 0 and len(control) > 0:
                uplift = treated.mean() - control.mean()
            else:
                uplift = 0
            
            uplift_curve.append(uplift)
        
        ax.plot(fractions * 100, uplift_curve, 'b-', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('% of Population (sorted by predicted uplift)')
        ax.set_ylabel('Observed Uplift')
        ax.set_title('Uplift Curve')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def summary(self) -> str:
        """Return model summary."""
        if not self._is_fitted:
            return "Model not fitted. Call fit() first."
        
        metrics = self.evaluate()
        segments = self.segment_by_effect()
        
        lines = [
            "Uplift Model Summary",
            "=" * 50,
            f"Method: {self._method}",
            f"Features: {len(self.features)}",
            f"Observations: {len(self.data)}",
            "",
            "Performance Metrics:",
            f"  Qini Coefficient: {metrics['qini_coefficient']:.4f}",
            f"  AUUC: {metrics['auuc']:.4f}",
            f"  ATE: {metrics['ate']:.4f}",
            "",
            "Segments:",
        ]
        
        for seg in segments:
            lines.append(
                f"  {seg.name}: n={seg.size}, "
                f"predicted={seg.predicted_uplift:.4f}, "
                f"actual={seg.actual_uplift:.4f}"
            )
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"UpliftModeler(features={len(self.features)}, {status})"
