"""
Causal Discovery Algorithms.

Implements multiple methods for learning causal structure from data:
- Constraint-based: PC, FCI
- Score-based: GES, FGES  
- Functional: LiNGAM
- Neural: NOTEARS
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats, linalg
from collections import defaultdict
import warnings


class CausalDiscovery:
    """
    Causal structure discovery from observational data.
    
    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def discover(
        self,
        data: pd.DataFrame,
        method: str = 'pc',
        target: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Discover causal structure from data.
        
        Parameters
        ----------
        data : pd.DataFrame
            The observational data
        method : str
            Discovery method: 'pc', 'ges', 'lingam', 'notears', 'hybrid'
        target : str, optional
            Target variable (used for orienting edges)
        **kwargs
            Additional method-specific arguments
            
        Returns
        -------
        dict
            Adjacency list: {child: [parent1, parent2, ...]}
        """
        methods = {
            'pc': self._pc_algorithm,
            'ges': self._ges_algorithm,
            'fges': self._ges_algorithm,  # Alias
            'lingam': self._lingam_algorithm,
            'notears': self._notears_algorithm,
            'hybrid': self._hybrid_algorithm,
            'correlation': self._correlation_graph,
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
        
        return methods[method.lower()](data, target=target, **kwargs)
    
    def _pc_algorithm(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        PC Algorithm for causal discovery.
        
        Based on conditional independence tests to learn causal skeleton
        and orient edges.
        """
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize complete undirected graph
        skeleton = {v: set(variables) - {v} for v in variables}
        
        # Phase 1: Learn skeleton via conditional independence tests
        depth = 0
        max_depth = n_vars - 2
        
        while depth <= max_depth:
            for x in variables:
                neighbors = list(skeleton[x])
                for y in neighbors:
                    if y not in skeleton[x]:
                        continue
                    
                    # Find conditioning sets
                    other_neighbors = [n for n in skeleton[x] if n != y]
                    
                    if len(other_neighbors) >= depth:
                        # Test all subsets of size 'depth'
                        for cond_set in self._subsets(other_neighbors, depth):
                            if self._conditional_independence_test(
                                data, x, y, list(cond_set), alpha
                            ):
                                # Remove edge
                                skeleton[x].discard(y)
                                skeleton[y].discard(x)
                                break
            
            depth += 1
        
        # Phase 2: Orient edges
        dag = self._orient_edges(skeleton, data, target)
        
        # Convert to parent-child format
        graph = defaultdict(list)
        for child in variables:
            graph[child] = []
        
        for parent, children in dag.items():
            for child in children:
                if parent not in graph[child]:
                    graph[child].append(parent)
        
        return dict(graph)
    
    def _conditional_independence_test(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        cond_set: List[str],
        alpha: float
    ) -> bool:
        """Test if X ⊥ Y | cond_set using partial correlation."""
        if len(cond_set) == 0:
            # Simple correlation test
            r, p = stats.pearsonr(data[x], data[y])
            return p > alpha
        
        # Partial correlation
        try:
            partial_corr = self._partial_correlation(data, x, y, cond_set)
            
            # Fisher's z-transform for p-value
            n = len(data)
            k = len(cond_set)
            
            if abs(partial_corr) >= 1:
                return False
            
            z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr + 1e-10))
            se = 1 / np.sqrt(n - k - 3 + 1e-10)
            p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
            
            return p_value > alpha
            
        except Exception:
            return False
    
    def _partial_correlation(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        cond_set: List[str]
    ) -> float:
        """Compute partial correlation of x and y given conditioning set."""
        from sklearn.linear_model import LinearRegression
        
        if not cond_set:
            return np.corrcoef(data[x], data[y])[0, 1]
        
        # Regress out conditioning set
        Z = data[cond_set].values
        
        reg_x = LinearRegression().fit(Z, data[x].values)
        reg_y = LinearRegression().fit(Z, data[y].values)
        
        resid_x = data[x].values - reg_x.predict(Z)
        resid_y = data[y].values - reg_y.predict(Z)
        
        return np.corrcoef(resid_x, resid_y)[0, 1]
    
    def _subsets(self, lst: List, size: int):
        """Generate all subsets of given size."""
        from itertools import combinations
        return combinations(lst, size)
    
    def _orient_edges(
        self,
        skeleton: Dict[str, set],
        data: pd.DataFrame,
        target: Optional[str]
    ) -> Dict[str, set]:
        """Orient undirected edges in skeleton."""
        dag = {v: set() for v in skeleton}
        variables = list(skeleton.keys())
        
        # Find v-structures (colliders)
        for y in variables:
            neighbors = list(skeleton[y])
            for i, x in enumerate(neighbors):
                for z in neighbors[i+1:]:
                    # Check if x - y - z forms a v-structure
                    if z not in skeleton[x]:  # x and z not adjacent
                        # x -> y <- z
                        dag[x].add(y)
                        dag[z].add(y)
        
        # Additional orientation rules
        changed = True
        while changed:
            changed = False
            
            for x in variables:
                for y in skeleton[x]:
                    if y in dag[x] or x in dag[y]:
                        continue
                    
                    # Rule 1: Orient to avoid cycles
                    if self._would_create_cycle(dag, y, x):
                        dag[x].add(y)
                        changed = True
                        continue
                    
                    # Rule 2: Orient away from target
                    if target and y == target:
                        dag[x].add(y)
                        changed = True
        
        # Orient remaining edges toward target
        if target:
            for x in variables:
                for y in skeleton[x]:
                    if y not in dag[x] and x not in dag[y]:
                        if y == target:
                            dag[x].add(y)
                        elif x == target:
                            dag[y].add(x)
        
        # Default: orient by correlation strength
        for x in variables:
            for y in list(skeleton[x]):
                if y not in dag[x] and x not in dag[y]:
                    # Orient based on data characteristics
                    var_x = data[x].var()
                    var_y = data[y].var()
                    if var_x > var_y:
                        dag[x].add(y)
                    else:
                        dag[y].add(x)
        
        return dag
    
    def _would_create_cycle(
        self,
        dag: Dict[str, set],
        source: str,
        target: str
    ) -> bool:
        """Check if adding source -> target would create a cycle."""
        visited = set()
        stack = [target]
        
        while stack:
            node = stack.pop()
            if node == source:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(dag.get(node, set()))
        
        return False
    
    def _ges_algorithm(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Greedy Equivalence Search (GES) algorithm.
        
        Score-based method that searches over equivalence classes of DAGs.
        """
        variables = list(data.columns)
        n = len(data)
        
        # Initialize empty graph
        graph = {v: set() for v in variables}
        
        # Compute correlation matrix for scoring
        corr_matrix = data.corr().values
        var_idx = {v: i for i, v in enumerate(variables)}
        
        def bic_score(parents: List[str], child: str) -> float:
            """Compute BIC score for child given parents."""
            if not parents:
                return -n * np.log(data[child].var() + 1e-10)
            
            from sklearn.linear_model import LinearRegression
            X = data[parents].values
            y = data[child].values
            
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            rss = np.sum(residuals ** 2)
            
            k = len(parents) + 1
            bic = n * np.log(rss / n + 1e-10) + k * np.log(n)
            return -bic  # Negative because we maximize
        
        # Forward phase: add edges
        improved = True
        while improved:
            improved = False
            best_score_increase = 0
            best_edge = None
            
            for x in variables:
                for y in variables:
                    if x == y or x in graph[y]:
                        continue
                    
                    # Try adding x -> y
                    current_score = bic_score(list(graph[y]), y)
                    new_parents = list(graph[y]) + [x]
                    new_score = bic_score(new_parents, y)
                    
                    increase = new_score - current_score
                    if increase > best_score_increase:
                        best_score_increase = increase
                        best_edge = (x, y)
            
            if best_edge and best_score_increase > 0:
                x, y = best_edge
                graph[y].add(x)
                improved = True
        
        # Backward phase: remove edges
        improved = True
        while improved:
            improved = False
            best_score_increase = 0
            best_edge = None
            
            for y in variables:
                for x in list(graph[y]):
                    # Try removing x -> y
                    current_score = bic_score(list(graph[y]), y)
                    new_parents = [p for p in graph[y] if p != x]
                    new_score = bic_score(new_parents, y)
                    
                    increase = new_score - current_score
                    if increase > best_score_increase:
                        best_score_increase = increase
                        best_edge = (x, y)
            
            if best_edge and best_score_increase > 0:
                x, y = best_edge
                graph[y].discard(x)
                improved = True
        
        # Convert to list format
        return {v: list(graph[v]) for v in variables}
    
    def _lingam_algorithm(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        LiNGAM (Linear Non-Gaussian Acyclic Model) algorithm.
        
        Uses independent component analysis to discover causal structure.
        """
        variables = list(data.columns)
        X = data.values
        n, p = X.shape
        
        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Simple ICA-based approach
        try:
            from sklearn.decomposition import FastICA
            
            ica = FastICA(n_components=p, random_state=self.random_state)
            S = ica.fit_transform(X)
            W = ica.mixing_
            
            # Estimate causal order from W
            # Find permutation that makes W close to lower triangular
            order = self._find_causal_order(W)
            ordered_vars = [variables[i] for i in order]
            
        except Exception:
            # Fallback to correlation-based ordering
            ordered_vars = self._correlation_order(data, target)
        
        # Build DAG based on order
        graph = {v: [] for v in variables}
        
        for i, child in enumerate(ordered_vars):
            potential_parents = ordered_vars[:i]
            
            if potential_parents:
                # Regression to find significant parents
                from sklearn.linear_model import LassoCV
                
                X_parents = data[potential_parents].values
                y = data[child].values
                
                try:
                    lasso = LassoCV(cv=5, random_state=self.random_state)
                    lasso.fit(X_parents, y)
                    
                    for j, parent in enumerate(potential_parents):
                        if abs(lasso.coef_[j]) > 0.01:
                            graph[child].append(parent)
                except Exception:
                    # Simple correlation threshold
                    for parent in potential_parents:
                        corr = np.corrcoef(data[parent], data[child])[0, 1]
                        if abs(corr) > 0.1:
                            graph[child].append(parent)
        
        return graph
    
    def _find_causal_order(self, W: np.ndarray) -> List[int]:
        """Find causal ordering from ICA mixing matrix."""
        p = W.shape[0]
        
        # Use row norms as proxy for causal position
        norms = np.linalg.norm(W, axis=1)
        order = np.argsort(norms)
        
        return list(order)
    
    def _correlation_order(
        self,
        data: pd.DataFrame,
        target: Optional[str]
    ) -> List[str]:
        """Order variables by correlation with target."""
        variables = list(data.columns)
        
        if target and target in variables:
            # Order by absolute correlation with target
            correlations = []
            for v in variables:
                if v == target:
                    correlations.append((v, 0))
                else:
                    corr = abs(np.corrcoef(data[v], data[target])[0, 1])
                    correlations.append((v, corr))
            
            # Sort: lowest correlation first (further from target)
            correlations.sort(key=lambda x: x[1])
            return [v for v, _ in correlations]
        
        return variables
    
    def _notears_algorithm(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        lambda1: float = 0.1,
        max_iter: int = 100,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        NOTEARS (Non-combinatorial Optimization via Trace Exponential and 
        Augmented lagRangian for Structure learning) algorithm.
        
        Uses continuous optimization to learn DAG structure.
        """
        X = data.values
        n, d = X.shape
        variables = list(data.columns)
        
        # Standardize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Initialize
        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        h_tol = 1e-8
        rho_max = 1e16
        
        def _loss(W):
            """Least squares loss."""
            M = X @ W
            R = X - M
            return 0.5 / n * np.sum(R ** 2)
        
        def _h(W):
            """DAG constraint: tr(exp(W◦W)) - d = 0."""
            E = linalg.expm(W * W)
            return np.trace(E) - d
        
        def _grad_loss(W):
            """Gradient of loss."""
            M = X @ W
            R = X - M
            return -1.0 / n * X.T @ R
        
        def _grad_h(W):
            """Gradient of DAG constraint."""
            E = linalg.expm(W * W)
            return E.T * W * 2
        
        # Augmented Lagrangian optimization
        for iteration in range(max_iter):
            # Inner optimization (gradient descent)
            for _ in range(10):
                grad = _grad_loss(W) + lambda1 * np.sign(W)
                grad += rho * _h(W) * _grad_h(W) + alpha * _grad_h(W)
                
                W = W - 0.01 * grad
            
            # Update Lagrangian parameters
            h = _h(W)
            alpha = alpha + rho * h
            
            if h > 0.25 * h_tol:
                rho = min(rho * 10, rho_max)
            
            if abs(h) < h_tol:
                break
        
        # Threshold small weights
        W[np.abs(W) < 0.1] = 0
        
        # Convert to graph
        graph = {v: [] for v in variables}
        for i in range(d):
            for j in range(d):
                if W[j, i] != 0:  # j -> i
                    graph[variables[i]].append(variables[j])
        
        return graph
    
    def _hybrid_algorithm(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Hybrid approach combining multiple methods.
        
        Uses consensus of PC, GES, and correlation to build robust graph.
        """
        # Run multiple methods
        graphs = []
        
        try:
            graphs.append(self._pc_algorithm(data, target))
        except Exception:
            pass
        
        try:
            graphs.append(self._ges_algorithm(data, target))
        except Exception:
            pass
        
        graphs.append(self._correlation_graph(data, target))
        
        if not graphs:
            raise RuntimeError("All discovery methods failed")
        
        # Consensus graph
        variables = list(data.columns)
        consensus = {v: [] for v in variables}
        
        for child in variables:
            parent_votes = defaultdict(int)
            
            for g in graphs:
                for parent in g.get(child, []):
                    parent_votes[parent] += 1
            
            # Include if majority vote
            threshold = len(graphs) / 2
            for parent, votes in parent_votes.items():
                if votes >= threshold:
                    consensus[child].append(parent)
        
        return consensus
    
    def _correlation_graph(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        threshold: float = 0.1,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Simple correlation-based graph construction.
        
        Uses correlation > threshold as proxy for causal relationship.
        Orients edges toward target.
        """
        variables = list(data.columns)
        corr_matrix = data.corr()
        
        graph = {v: [] for v in variables}
        
        # Build edges based on correlation
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                corr = corr_matrix.loc[v1, v2]
                
                if abs(corr) > threshold:
                    # Orient edge
                    if target:
                        # Point toward target
                        if v2 == target:
                            graph[v2].append(v1)
                        elif v1 == target:
                            graph[v1].append(v2)
                        else:
                            # Use variance as proxy for causal order
                            if data[v1].var() > data[v2].var():
                                graph[v2].append(v1)
                            else:
                                graph[v1].append(v2)
                    else:
                        # Default: higher variance causes lower variance
                        if data[v1].var() > data[v2].var():
                            graph[v2].append(v1)
                        else:
                            graph[v1].append(v2)
        
        return graph
