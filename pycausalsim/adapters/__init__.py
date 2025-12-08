"""
Adapters for interoperability with other causal inference libraries.

Integrates with:
- DoWhy (Microsoft)
- EconML (Microsoft)  
- CausalML (Uber)
- Papilon (Complex systems)
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import warnings


def to_dowhy(simulator):
    """
    Convert PyCausalSim simulator to DoWhy CausalModel.
    
    Parameters
    ----------
    simulator : CausalSimulator
        PyCausalSim simulator instance
        
    Returns
    -------
    dowhy.CausalModel
        DoWhy causal model
        
    Examples
    --------
    >>> from pycausalsim.adapters import to_dowhy
    >>> dowhy_model = to_dowhy(simulator)
    >>> effect = dowhy_model.identify_effect()
    """
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        raise ImportError("DoWhy not installed. Run: pip install dowhy")
    
    # Build GML graph string from our graph
    gml = _graph_to_gml(simulator._graph)
    
    # Determine treatment and outcome
    treatment = simulator.treatment_vars[0] if simulator.treatment_vars else simulator._feature_vars[0]
    outcome = simulator.target
    
    model = CausalModel(
        data=simulator.data,
        treatment=treatment,
        outcome=outcome,
        graph=gml
    )
    
    return model


def from_dowhy(dowhy_model, data: Optional[pd.DataFrame] = None):
    """
    Create PyCausalSim simulator from DoWhy CausalModel.
    
    Parameters
    ----------
    dowhy_model : dowhy.CausalModel
        DoWhy model
    data : pd.DataFrame, optional
        Data to use (defaults to DoWhy model's data)
        
    Returns
    -------
    CausalSimulator
        PyCausalSim simulator
    """
    from ..simulator import CausalSimulator
    
    data = data if data is not None else dowhy_model._data
    target = dowhy_model._outcome[0] if isinstance(dowhy_model._outcome, list) else dowhy_model._outcome
    treatment = dowhy_model._treatment[0] if isinstance(dowhy_model._treatment, list) else dowhy_model._treatment
    
    simulator = CausalSimulator(
        data=data,
        target=target,
        treatment_vars=[treatment]
    )
    
    # Extract graph from DoWhy if available
    if hasattr(dowhy_model, '_graph'):
        graph = _gml_to_graph(dowhy_model._graph)
        simulator.set_graph(graph)
    
    return simulator


def to_econml(simulator):
    """
    Convert to EconML-compatible format.
    
    Parameters
    ----------
    simulator : CausalSimulator
        PyCausalSim simulator
        
    Returns
    -------
    dict
        Dictionary with Y, T, X, W arrays for EconML
        
    Examples
    --------
    >>> from pycausalsim.adapters import to_econml
    >>> arrays = to_econml(simulator)
    >>> from econml.dml import LinearDML
    >>> est = LinearDML()
    >>> est.fit(arrays['Y'], arrays['T'], X=arrays['X'], W=arrays['W'])
    """
    treatment = simulator.treatment_vars[0] if simulator.treatment_vars else simulator._feature_vars[0]
    
    Y = simulator.data[simulator.target].values
    T = simulator.data[treatment].values
    
    # Features for heterogeneity
    feature_cols = [c for c in simulator._feature_vars if c != treatment]
    X = simulator.data[feature_cols].values if feature_cols else None
    
    # Confounders
    W = simulator.data[simulator.confounders].values if simulator.confounders else None
    
    return {
        'Y': Y,
        'T': T,
        'X': X,
        'W': W,
        'feature_names': feature_cols,
        'confounder_names': simulator.confounders
    }


def from_econml(econml_model, data: pd.DataFrame, target: str, treatment: str):
    """
    Create simulator from fitted EconML model.
    
    Parameters
    ----------
    econml_model : econml estimator
        Fitted EconML model
    data : pd.DataFrame
        Data used for fitting
    target : str
        Outcome variable name
    treatment : str
        Treatment variable name
        
    Returns
    -------
    CausalSimulator
    """
    from ..simulator import CausalSimulator
    
    simulator = CausalSimulator(
        data=data,
        target=target,
        treatment_vars=[treatment]
    )
    
    # Store EconML model for CATE predictions
    simulator._econml_model = econml_model
    
    return simulator


def to_causalml(simulator):
    """
    Convert to CausalML-compatible format.
    
    Parameters
    ----------
    simulator : CausalSimulator
        PyCausalSim simulator
        
    Returns
    -------
    dict
        Dictionary with arrays for CausalML uplift models
    """
    treatment = simulator.treatment_vars[0] if simulator.treatment_vars else simulator._feature_vars[0]
    
    feature_cols = [c for c in simulator._feature_vars if c != treatment]
    
    return {
        'X': simulator.data[feature_cols],
        'treatment': simulator.data[treatment],
        'y': simulator.data[simulator.target],
        'feature_names': feature_cols
    }


def from_papilon(papilon_results, data: Optional[pd.DataFrame] = None):
    """
    Create simulator from Papilon complex system results.
    
    Parameters
    ----------
    papilon_results : dict or PapilonResult
        Results from Papilon simulation
    data : pd.DataFrame, optional
        Observational data
        
    Returns
    -------
    CausalSimulator
    """
    from ..simulator import CausalSimulator
    
    # Extract data from Papilon results
    if hasattr(papilon_results, 'to_dataframe'):
        data = papilon_results.to_dataframe()
    elif isinstance(papilon_results, dict) and 'data' in papilon_results:
        data = papilon_results['data']
    
    if data is None:
        raise ValueError("No data found in Papilon results")
    
    # Detect target (usually last column or specified)
    if hasattr(papilon_results, 'target'):
        target = papilon_results.target
    else:
        target = data.columns[-1]
    
    simulator = CausalSimulator(
        data=data,
        target=target
    )
    
    # Extract causal structure if available
    if hasattr(papilon_results, 'causal_graph'):
        simulator.set_graph(papilon_results.causal_graph)
    
    return simulator


def _graph_to_gml(graph: Dict[str, List[str]]) -> str:
    """Convert adjacency list to GML format."""
    lines = ["graph [", "  directed 1"]
    
    # Collect all nodes
    nodes = set(graph.keys())
    for parents in graph.values():
        nodes.update(parents)
    
    # Add nodes
    for i, node in enumerate(sorted(nodes)):
        lines.append(f'  node [ id {i} label "{node}" ]')
    
    # Create node index mapping
    node_idx = {node: i for i, node in enumerate(sorted(nodes))}
    
    # Add edges
    for child, parents in graph.items():
        for parent in parents:
            lines.append(f"  edge [ source {node_idx[parent]} target {node_idx[child]} ]")
    
    lines.append("]")
    return "\n".join(lines)


def _gml_to_graph(gml_string: str) -> Dict[str, List[str]]:
    """Convert GML format to adjacency list."""
    graph = {}
    
    # Simple GML parser
    import re
    
    # Extract nodes
    node_pattern = r'node\s*\[\s*id\s+(\d+)\s+label\s+"([^"]+)"\s*\]'
    nodes = {}
    for match in re.finditer(node_pattern, gml_string, re.IGNORECASE):
        node_id, label = match.groups()
        nodes[int(node_id)] = label
        graph[label] = []
    
    # Extract edges
    edge_pattern = r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)\s*\]'
    for match in re.finditer(edge_pattern, gml_string, re.IGNORECASE):
        source, target = match.groups()
        source_node = nodes.get(int(source))
        target_node = nodes.get(int(target))
        if source_node and target_node:
            if target_node not in graph:
                graph[target_node] = []
            graph[target_node].append(source_node)
    
    return graph


class DoWhyAdapter:
    """
    Adapter class for DoWhy integration.
    
    Provides bidirectional conversion between PyCausalSim and DoWhy.
    """
    
    def __init__(self, simulator=None, dowhy_model=None):
        self.simulator = simulator
        self.dowhy_model = dowhy_model
    
    def to_dowhy(self):
        """Convert simulator to DoWhy model."""
        if self.simulator is None:
            raise ValueError("No simulator set")
        return to_dowhy(self.simulator)
    
    def from_dowhy(self):
        """Convert DoWhy model to simulator."""
        if self.dowhy_model is None:
            raise ValueError("No DoWhy model set")
        return from_dowhy(self.dowhy_model)
    
    def identify_effect(self):
        """Use DoWhy for effect identification."""
        model = self.to_dowhy()
        return model.identify_effect()
    
    def estimate_effect(self, method_name: str = "backdoor.linear_regression"):
        """Use DoWhy for effect estimation."""
        model = self.to_dowhy()
        identified = model.identify_effect()
        return model.estimate_effect(identified, method_name=method_name)
    
    def refute_estimate(self, estimate, method_name: str = "random_common_cause"):
        """Use DoWhy refutation methods."""
        model = self.to_dowhy()
        return model.refute_estimate(
            model.identify_effect(),
            estimate,
            method_name=method_name
        )


class EconMLAdapter:
    """
    Adapter class for EconML integration.
    
    Provides access to EconML's heterogeneous treatment effect estimators.
    """
    
    def __init__(self, simulator=None):
        self.simulator = simulator
        self._fitted_model = None
    
    def fit_dml(self, **kwargs):
        """Fit Double Machine Learning estimator."""
        try:
            from econml.dml import LinearDML
        except ImportError:
            raise ImportError("EconML not installed. Run: pip install econml")
        
        arrays = to_econml(self.simulator)
        
        model = LinearDML(**kwargs)
        model.fit(
            arrays['Y'], 
            arrays['T'],
            X=arrays['X'],
            W=arrays['W']
        )
        
        self._fitted_model = model
        return model
    
    def fit_causal_forest(self, **kwargs):
        """Fit Causal Forest estimator."""
        try:
            from econml.dml import CausalForestDML
        except ImportError:
            raise ImportError("EconML not installed. Run: pip install econml")
        
        arrays = to_econml(self.simulator)
        
        model = CausalForestDML(**kwargs)
        model.fit(
            arrays['Y'],
            arrays['T'],
            X=arrays['X'],
            W=arrays['W']
        )
        
        self._fitted_model = model
        return model
    
    def effect(self, X=None):
        """Get treatment effects."""
        if self._fitted_model is None:
            raise RuntimeError("Must fit model first")
        
        if X is None:
            arrays = to_econml(self.simulator)
            X = arrays['X']
        
        return self._fitted_model.effect(X)
    
    def effect_interval(self, X=None, alpha: float = 0.05):
        """Get treatment effect confidence intervals."""
        if self._fitted_model is None:
            raise RuntimeError("Must fit model first")
        
        if X is None:
            arrays = to_econml(self.simulator)
            X = arrays['X']
        
        return self._fitted_model.effect_interval(X, alpha=alpha)
