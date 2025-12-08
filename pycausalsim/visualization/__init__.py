"""
Visualization utilities for PyCausalSim.

Provides consistent, publication-quality plots for:
- Causal graphs
- Effect sizes
- Counterfactuals
- Sensitivity analysis
- Uplift curves
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# Default style settings
STYLE = {
    'figsize': (10, 6),
    'colors': {
        'primary': '#2196F3',
        'secondary': '#FF9800',
        'positive': '#4CAF50',
        'negative': '#F44336',
        'neutral': '#9E9E9E',
        'treatment': '#4CAF50',
        'control': '#F44336',
        'target': '#E91E63',
        'confounder': '#FFC107',
    },
    'alpha': 0.7,
    'font_size': 12,
}


def plot_causal_graph(
    graph: Dict[str, List[str]],
    target: Optional[str] = None,
    treatment_vars: Optional[List[str]] = None,
    confounders: Optional[List[str]] = None,
    ax=None,
    figsize: Tuple[int, int] = (12, 8),
    layout: str = 'spring'
):
    """
    Plot causal graph as directed acyclic graph.
    
    Parameters
    ----------
    graph : dict
        Adjacency list {child: [parents]}
    target : str, optional
        Target variable to highlight
    treatment_vars : list, optional
        Treatment variables to highlight
    confounders : list, optional
        Confounders to highlight
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    layout : str
        Layout algorithm: 'spring', 'circular', 'hierarchical'
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for graph plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create directed graph
    G = nx.DiGraph()
    
    for child, parents in graph.items():
        G.add_node(child)
        for parent in parents:
            G.add_edge(parent, child)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'hierarchical':
        pos = _hierarchical_layout(G, graph)
    else:
        pos = nx.spring_layout(G)
    
    # Color nodes
    node_colors = []
    for node in G.nodes():
        if target and node == target:
            node_colors.append(STYLE['colors']['target'])
        elif treatment_vars and node in treatment_vars:
            node_colors.append(STYLE['colors']['treatment'])
        elif confounders and node in confounders:
            node_colors.append(STYLE['colors']['confounder'])
        else:
            node_colors.append(STYLE['colors']['primary'])
    
    # Draw
    nx.draw(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2000,
        with_labels=True,
        font_size=10,
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        edge_color='gray',
        alpha=0.8
    )
    
    ax.set_title('Causal Graph', fontsize=14)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    if target:
        legend_elements.append(Patch(facecolor=STYLE['colors']['target'], label='Target'))
    if treatment_vars:
        legend_elements.append(Patch(facecolor=STYLE['colors']['treatment'], label='Treatment'))
    if confounders:
        legend_elements.append(Patch(facecolor=STYLE['colors']['confounder'], label='Confounder'))
    legend_elements.append(Patch(facecolor=STYLE['colors']['primary'], label='Other'))
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    return ax


def _hierarchical_layout(G, graph):
    """Create hierarchical layout based on topological order."""
    import networkx as nx
    
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        return nx.spring_layout(G)
    
    layers = {}
    for node in order:
        parents = graph.get(node, [])
        if not parents:
            layers[node] = 0
        else:
            layers[node] = max(layers.get(p, 0) for p in parents) + 1
    
    # Position nodes
    pos = {}
    layer_nodes = {}
    for node, layer in layers.items():
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
    
    for layer, nodes in layer_nodes.items():
        for i, node in enumerate(nodes):
            x = (i + 0.5) / len(nodes)
            y = -layer
            pos[node] = (x, y)
    
    return pos


def plot_effects(
    effects: Dict[str, float],
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    top_n: int = 10,
    title: str = 'Causal Effects'
):
    """
    Plot causal effect sizes with confidence intervals.
    
    Parameters
    ----------
    effects : dict
        {variable: effect} dictionary
    confidence_intervals : dict, optional
        {variable: (lower, upper)} dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    top_n : int
        Number of top effects to show
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute effect
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    vars_, effect_vals = zip(*sorted_effects) if sorted_effects else ([], [])
    
    colors = [STYLE['colors']['positive'] if e > 0 else STYLE['colors']['negative'] 
              for e in effect_vals]
    
    y_pos = np.arange(len(vars_))
    
    # Plot bars
    bars = ax.barh(y_pos, effect_vals, color=colors, alpha=STYLE['alpha'])
    
    # Add confidence intervals
    if confidence_intervals:
        for i, var in enumerate(vars_):
            if var in confidence_intervals:
                lower, upper = confidence_intervals[var]
                ax.plot([lower, upper], [i, i], 'k-', linewidth=2)
                ax.plot([lower], [i], 'k|', markersize=10)
                ax.plot([upper], [i], 'k|', markersize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars_)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Causal Effect')
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    return ax


def plot_counterfactual(
    factual: np.ndarray,
    counterfactual: np.ndarray,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Factual vs Counterfactual'
):
    """
    Plot factual vs counterfactual distributions.
    
    Parameters
    ----------
    factual : np.ndarray
        Observed outcomes
    counterfactual : np.ndarray
        Simulated counterfactual outcomes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Histograms
    ax.hist(factual, bins=50, alpha=0.5, 
            label=f'Factual (mean={np.mean(factual):.4f})',
            color=STYLE['colors']['control'], density=True)
    ax.hist(counterfactual, bins=50, alpha=0.5,
            label=f'Counterfactual (mean={np.mean(counterfactual):.4f})',
            color=STYLE['colors']['treatment'], density=True)
    
    # Mean lines
    ax.axvline(np.mean(factual), color=STYLE['colors']['control'], 
               linestyle='--', linewidth=2)
    ax.axvline(np.mean(counterfactual), color=STYLE['colors']['treatment'],
               linestyle='--', linewidth=2)
    
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Density')
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_sensitivity(
    bounds: Dict[float, Tuple[float, float]],
    original_estimate: float,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Sensitivity to Unobserved Confounding'
):
    """
    Plot sensitivity analysis bounds.
    
    Parameters
    ----------
    bounds : dict
        {gamma: (lower, upper)} dictionary
    original_estimate : float
        Original effect estimate
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    gammas = sorted(bounds.keys())
    lowers = [bounds[g][0] for g in gammas]
    uppers = [bounds[g][1] for g in gammas]
    
    # Shaded region
    ax.fill_between(gammas, lowers, uppers, alpha=0.3, 
                    color=STYLE['colors']['primary'], label='Effect bounds')
    
    # Original estimate
    ax.axhline(original_estimate, color=STYLE['colors']['secondary'],
               linestyle='-', linewidth=2, label='Original estimate')
    
    # Zero line
    ax.axhline(0, color=STYLE['colors']['negative'],
               linestyle='--', linewidth=1, label='Null effect')
    
    ax.set_xlabel('Confounding Strength (Γ)')
    ax.set_ylabel('Causal Effect')
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_uplift_curve(
    predicted_uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    ax=None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot uplift curve.
    
    Parameters
    ----------
    predicted_uplift : np.ndarray
        Predicted uplift scores
    treatment : np.ndarray
        Treatment assignments
    outcome : np.ndarray
        Outcomes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by predicted uplift
    order = np.argsort(predicted_uplift)[::-1]
    
    n = len(predicted_uplift)
    fractions = np.linspace(0.01, 1.0, 100)
    
    uplift_curve = []
    random_curve = []
    
    for frac in fractions:
        top_n = max(1, int(frac * n))
        top_idx = order[:top_n]
        
        t_mask = treatment[top_idx] == 1
        c_mask = treatment[top_idx] == 0
        
        if t_mask.sum() > 0 and c_mask.sum() > 0:
            uplift = outcome[top_idx][t_mask].mean() - outcome[top_idx][c_mask].mean()
        else:
            uplift = 0
        
        uplift_curve.append(uplift)
        
        # Random baseline
        t_all = treatment == 1
        c_all = treatment == 0
        random_uplift = outcome[t_all].mean() - outcome[c_all].mean() if t_all.any() and c_all.any() else 0
        random_curve.append(random_uplift)
    
    ax.plot(fractions * 100, uplift_curve, 'b-', linewidth=2, label='Model')
    ax.plot(fractions * 100, random_curve, 'r--', linewidth=1, label='Random')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('% of Population (sorted by predicted uplift)')
    ax.set_ylabel('Observed Uplift')
    ax.set_title('Uplift Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_heterogeneous_effects(
    segments: Dict[str, float],
    segment_cis: Optional[Dict[str, Tuple[float, float]]] = None,
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Heterogeneous Treatment Effects'
):
    """
    Plot heterogeneous effects across segments.
    
    Parameters
    ----------
    segments : dict
        {segment_name: effect} dictionary
    segment_cis : dict, optional
        {segment_name: (lower, upper)} dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    names = list(segments.keys())
    effects = list(segments.values())
    
    x = np.arange(len(names))
    colors = [STYLE['colors']['positive'] if e > 0 else STYLE['colors']['negative']
              for e in effects]
    
    # Plot bars
    bars = ax.bar(x, effects, color=colors, alpha=STYLE['alpha'])
    
    # Add error bars
    if segment_cis:
        for i, name in enumerate(names):
            if name in segment_cis:
                lower, upper = segment_cis[name]
                ax.errorbar(i, effects[i], 
                           yerr=[[effects[i] - lower], [upper - effects[i]]],
                           fmt='none', color='black', capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Treatment Effect')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return ax


def plot_attribution(
    attribution: Dict[str, float],
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Attribution Weights'
):
    """
    Plot marketing attribution weights.
    
    Parameters
    ----------
    attribution : dict
        {channel: weight} dictionary
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by weight
    sorted_attr = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
    channels, weights = zip(*sorted_attr) if sorted_attr else ([], [])
    
    # Pie chart
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(channels)))
    
    ax.pie(weights, labels=channels, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title(title, fontsize=14)
    
    return ax


def create_dashboard(
    simulator,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Create comprehensive analysis dashboard.
    
    Parameters
    ----------
    simulator : CausalSimulator
        Fitted simulator
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # 1. Causal graph
    if simulator._graph:
        plot_causal_graph(
            simulator._graph,
            target=simulator.target,
            treatment_vars=simulator.treatment_vars,
            confounders=simulator.confounders,
            ax=ax1
        )
    
    # 2. Driver effects
    try:
        drivers = simulator.rank_drivers(n_simulations=100)
        effects = dict(drivers.drivers[:10])
        plot_effects(effects, ax=ax2, title='Top Causal Drivers')
    except Exception:
        ax2.text(0.5, 0.5, 'Effect ranking unavailable', 
                ha='center', va='center')
    
    # 3. Data distribution
    ax3.hist(simulator.data[simulator.target], bins=50, 
             color=STYLE['colors']['primary'], alpha=STYLE['alpha'])
    ax3.set_xlabel(simulator.target)
    ax3.set_ylabel('Count')
    ax3.set_title(f'Distribution of {simulator.target}')
    
    # 4. Summary statistics
    stats_text = [
        f"Observations: {len(simulator.data)}",
        f"Variables: {len(simulator._all_vars)}",
        f"Target: {simulator.target}",
        f"Target mean: {simulator.data[simulator.target].mean():.4f}",
        f"Target std: {simulator.data[simulator.target].std():.4f}",
    ]
    
    if simulator._graph:
        n_edges = sum(len(p) for p in simulator._graph.values())
        stats_text.append(f"Causal edges: {n_edges}")
    
    ax4.text(0.1, 0.5, '\n'.join(stats_text),
            transform=ax4.transAxes, fontsize=12,
            verticalalignment='center', family='monospace')
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    return fig
