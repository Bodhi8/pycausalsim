"""
Agent-Based Simulation for Causal Systems.

Model complex systems with interacting agents to understand
emergent causal effects and network/contagion dynamics.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import warnings


@dataclass
class Agent:
    """
    An agent in the simulation.
    
    Attributes:
        id: Unique identifier
        state: Current state variables
        connections: IDs of connected agents
        history: History of state changes
    """
    id: int
    state: Dict[str, float] = field(default_factory=dict)
    connections: List[int] = field(default_factory=list)
    history: List[Dict[str, float]] = field(default_factory=list)
    
    def update(self, new_state: Dict[str, float]):
        """Update agent state and record history."""
        self.history.append(dict(self.state))
        self.state.update(new_state)


@dataclass
class SimulationStep:
    """Record of a single simulation step."""
    step: int
    aggregate_state: Dict[str, float]
    agent_states: Optional[pd.DataFrame] = None


class AgentBasedSimulator:
    """
    Agent-based simulator for complex causal systems.
    
    Useful for modeling:
    - Network effects (viral spread, social influence)
    - Market dynamics
    - Ecosystem behavior
    - Emergent phenomena
    
    Parameters
    ----------
    n_agents : int
        Number of agents
    causal_model : StructuralCausalModel, optional
        Causal model governing agent behavior
    network_type : str
        Network structure: 'random', 'small_world', 'scale_free', 'grid'
    random_state : int, optional
        Random seed
        
    Examples
    --------
    >>> from pycausalsim.agents import AgentBasedSimulator
    >>> sim = AgentBasedSimulator(n_agents=1000, network_type='small_world')
    >>> sim.initialize_agents(adoption_prob=0.1)
    >>> results = sim.simulate(steps=100)
    >>> contagion = sim.analyze_contagion()
    """
    
    def __init__(
        self,
        n_agents: int = 1000,
        causal_model=None,
        network_type: str = 'random',
        network_density: float = 0.05,
        random_state: Optional[int] = None
    ):
        self.n_agents = n_agents
        self.causal_model = causal_model
        self.network_type = network_type
        self.network_density = network_density
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize agents
        self.agents: List[Agent] = []
        self._network = None
        self._history: List[SimulationStep] = []
        
        # Build network
        self._build_network()
    
    def _build_network(self):
        """Build agent connection network."""
        self.agents = [Agent(id=i) for i in range(self.n_agents)]
        
        if self.network_type == 'random':
            self._build_random_network()
        elif self.network_type == 'small_world':
            self._build_small_world_network()
        elif self.network_type == 'scale_free':
            self._build_scale_free_network()
        elif self.network_type == 'grid':
            self._build_grid_network()
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")
    
    def _build_random_network(self):
        """Build Erdos-Renyi random network."""
        for i, agent in enumerate(self.agents):
            for j in range(self.n_agents):
                if i != j and np.random.random() < self.network_density:
                    agent.connections.append(j)
    
    def _build_small_world_network(self, k: int = 4, p: float = 0.1):
        """Build Watts-Strogatz small-world network."""
        # Start with ring lattice
        for i, agent in enumerate(self.agents):
            for j in range(1, k // 2 + 1):
                agent.connections.append((i + j) % self.n_agents)
                agent.connections.append((i - j) % self.n_agents)
        
        # Rewire with probability p
        for agent in self.agents:
            for idx, neighbor in enumerate(agent.connections.copy()):
                if np.random.random() < p:
                    # Rewire to random node
                    new_neighbor = np.random.randint(self.n_agents)
                    if new_neighbor != agent.id and new_neighbor not in agent.connections:
                        agent.connections[idx] = new_neighbor
    
    def _build_scale_free_network(self, m: int = 2):
        """Build Barabasi-Albert scale-free network."""
        # Start with small complete graph
        for i in range(m + 1):
            for j in range(m + 1):
                if i != j:
                    self.agents[i].connections.append(j)
        
        # Add remaining nodes with preferential attachment
        for i in range(m + 1, self.n_agents):
            # Degree of each existing node
            degrees = [len(self.agents[j].connections) for j in range(i)]
            total_degree = sum(degrees)
            
            if total_degree == 0:
                probs = [1/i] * i
            else:
                probs = [d / total_degree for d in degrees]
            
            # Connect to m nodes with probability proportional to degree
            targets = np.random.choice(i, size=m, replace=False, p=probs)
            for target in targets:
                self.agents[i].connections.append(target)
                self.agents[target].connections.append(i)
    
    def _build_grid_network(self):
        """Build 2D grid network."""
        side = int(np.sqrt(self.n_agents))
        
        for i, agent in enumerate(self.agents):
            row, col = i // side, i % side
            
            # Connect to neighbors
            if row > 0:
                agent.connections.append((row - 1) * side + col)
            if row < side - 1:
                agent.connections.append((row + 1) * side + col)
            if col > 0:
                agent.connections.append(row * side + col - 1)
            if col < side - 1:
                agent.connections.append(row * side + col + 1)
    
    def initialize_agents(
        self,
        initial_state: Optional[Dict[str, float]] = None,
        random_init: bool = True,
        **kwargs
    ):
        """
        Initialize agent states.
        
        Parameters
        ----------
        initial_state : dict, optional
            Base state for all agents
        random_init : bool
            Whether to add random variation
        **kwargs
            Additional state variables
        """
        base_state = initial_state or {}
        base_state.update(kwargs)
        
        for agent in self.agents:
            state = dict(base_state)
            
            if random_init:
                # Add noise to continuous variables
                for key, value in state.items():
                    if isinstance(value, (int, float)) and value != 0:
                        state[key] = value + np.random.normal(0, abs(value) * 0.1)
            
            agent.state = state
    
    def set_treatment(
        self,
        treatment_var: str,
        value: float,
        fraction: float = 0.5,
        selection: str = 'random'
    ):
        """
        Assign treatment to agents.
        
        Parameters
        ----------
        treatment_var : str
            Treatment variable name
        value : float
            Treatment value
        fraction : float
            Fraction of agents to treat
        selection : str
            Selection method: 'random', 'high_degree', 'low_degree'
        """
        n_treated = int(self.n_agents * fraction)
        
        if selection == 'random':
            treated_ids = np.random.choice(self.n_agents, size=n_treated, replace=False)
        elif selection == 'high_degree':
            degrees = [len(a.connections) for a in self.agents]
            treated_ids = np.argsort(degrees)[-n_treated:]
        elif selection == 'low_degree':
            degrees = [len(a.connections) for a in self.agents]
            treated_ids = np.argsort(degrees)[:n_treated]
        else:
            raise ValueError(f"Unknown selection: {selection}")
        
        for agent in self.agents:
            agent.state[treatment_var] = value if agent.id in treated_ids else 0
    
    def simulate(
        self,
        steps: int = 100,
        update_rule: Optional[Callable] = None,
        record_all: bool = False
    ) -> List[SimulationStep]:
        """
        Run agent-based simulation.
        
        Parameters
        ----------
        steps : int
            Number of simulation steps
        update_rule : callable, optional
            Custom update function f(agent, neighbors, step) -> new_state
        record_all : bool
            Whether to record all agent states (memory intensive)
            
        Returns
        -------
        list of SimulationStep
            Simulation history
        """
        self._history = []
        
        for step in range(steps):
            # Update all agents
            new_states = []
            
            for agent in self.agents:
                neighbors = [self.agents[i] for i in agent.connections]
                
                if update_rule:
                    new_state = update_rule(agent, neighbors, step)
                elif self.causal_model:
                    new_state = self._causal_update(agent, neighbors)
                else:
                    new_state = self._default_update(agent, neighbors)
                
                new_states.append(new_state)
            
            # Apply updates
            for agent, new_state in zip(self.agents, new_states):
                agent.update(new_state)
            
            # Record step
            aggregate = self._compute_aggregate()
            agent_states = self._get_all_states() if record_all else None
            
            self._history.append(SimulationStep(
                step=step,
                aggregate_state=aggregate,
                agent_states=agent_states
            ))
        
        return self._history
    
    def _causal_update(
        self,
        agent: Agent,
        neighbors: List[Agent]
    ) -> Dict[str, float]:
        """Update agent using causal model."""
        if not self.causal_model:
            return agent.state
        
        # Prepare input
        input_data = pd.DataFrame([agent.state])
        
        # Add neighborhood influence
        if neighbors:
            neighbor_states = pd.DataFrame([n.state for n in neighbors])
            for col in neighbor_states.columns:
                input_data[f'{col}_neighbor_mean'] = neighbor_states[col].mean()
        
        # Predict new state using causal model
        new_state = dict(agent.state)
        
        for var in self.causal_model.variables:
            try:
                prediction = self.causal_model.predict(input_data, var)
                new_state[var] = prediction[0]
            except Exception:
                pass
        
        return new_state
    
    def _default_update(
        self,
        agent: Agent,
        neighbors: List[Agent]
    ) -> Dict[str, float]:
        """Default update rule with social influence."""
        new_state = dict(agent.state)
        
        if not neighbors:
            return new_state
        
        # Average influence from neighbors
        neighbor_states = [n.state for n in neighbors]
        
        for key in new_state:
            if key in neighbor_states[0]:
                neighbor_values = [ns.get(key, 0) for ns in neighbor_states]
                influence = np.mean(neighbor_values)
                
                # Partial update toward neighbor average
                new_state[key] = 0.9 * new_state[key] + 0.1 * influence
                
                # Add small noise
                new_state[key] += np.random.normal(0, 0.01)
        
        return new_state
    
    def _compute_aggregate(self) -> Dict[str, float]:
        """Compute aggregate statistics across all agents."""
        all_states = [agent.state for agent in self.agents]
        
        aggregate = {}
        if all_states:
            df = pd.DataFrame(all_states)
            for col in df.columns:
                aggregate[f'{col}_mean'] = df[col].mean()
                aggregate[f'{col}_std'] = df[col].std()
                aggregate[f'{col}_sum'] = df[col].sum()
        
        return aggregate
    
    def _get_all_states(self) -> pd.DataFrame:
        """Get all agent states as DataFrame."""
        return pd.DataFrame([agent.state for agent in self.agents])
    
    def analyze_contagion(self) -> Dict[str, Any]:
        """
        Analyze contagion/diffusion dynamics.
        
        Returns
        -------
        dict
            Contagion analysis results
        """
        if not self._history:
            raise RuntimeError("Must run simulation first")
        
        # Track spread over time
        time_series = []
        for step in self._history:
            time_series.append(step.aggregate_state)
        
        df = pd.DataFrame(time_series)
        
        # Find adoption metrics
        results = {
            'time_series': df,
        }
        
        # Identify variables that spread
        for col in df.columns:
            if '_mean' in col:
                var = col.replace('_mean', '')
                
                start_val = df[col].iloc[0]
                end_val = df[col].iloc[-1]
                
                results[f'{var}_adoption_rate'] = (end_val - start_val) / (abs(start_val) + 1e-10)
                results[f'{var}_final_level'] = end_val
        
        return results
    
    def analyze_network_effects(
        self,
        treatment_var: str,
        outcome_var: str
    ) -> Dict[str, float]:
        """
        Analyze network/spillover effects.
        
        Parameters
        ----------
        treatment_var : str
            Treatment variable
        outcome_var : str
            Outcome variable
            
        Returns
        -------
        dict
            Network effect estimates
        """
        treated = []
        control = []
        spillover = []
        
        for agent in self.agents:
            is_treated = agent.state.get(treatment_var, 0) > 0
            outcome = agent.state.get(outcome_var, 0)
            
            # Check if neighbors are treated
            neighbor_treatment = np.mean([
                self.agents[i].state.get(treatment_var, 0)
                for i in agent.connections
            ]) if agent.connections else 0
            
            if is_treated:
                treated.append(outcome)
            elif neighbor_treatment > 0:
                spillover.append(outcome)
            else:
                control.append(outcome)
        
        direct_effect = np.mean(treated) - np.mean(control) if treated and control else 0
        spillover_effect = np.mean(spillover) - np.mean(control) if spillover and control else 0
        
        return {
            'direct_effect': direct_effect,
            'spillover_effect': spillover_effect,
            'total_effect': direct_effect + spillover_effect,
            'n_treated': len(treated),
            'n_spillover': len(spillover),
            'n_control': len(control)
        }
    
    def intervene(
        self,
        variable: str,
        value: float,
        agent_ids: Optional[List[int]] = None
    ):
        """
        Apply intervention to agents.
        
        Parameters
        ----------
        variable : str
            Variable to intervene on
        value : float
            Intervention value
        agent_ids : list of int, optional
            Specific agents to intervene on (default: all)
        """
        if agent_ids is None:
            agent_ids = range(self.n_agents)
        
        for i in agent_ids:
            self.agents[i].state[variable] = value
    
    def get_results(self) -> pd.DataFrame:
        """Get simulation results as DataFrame."""
        records = []
        for step in self._history:
            record = {'step': step.step}
            record.update(step.aggregate_state)
            records.append(record)
        
        return pd.DataFrame(records)
    
    def plot_dynamics(self, variables: Optional[List[str]] = None, ax=None):
        """Plot simulation dynamics over time."""
        import matplotlib.pyplot as plt
        
        if not self._history:
            raise RuntimeError("Must run simulation first")
        
        results = self.get_results()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if variables is None:
            # Plot all mean variables
            variables = [c for c in results.columns if '_mean' in c]
        
        for var in variables:
            if var in results.columns:
                ax.plot(results['step'], results[var], label=var)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Agent-Based Simulation Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_network(self, ax=None, show_state: Optional[str] = None):
        """Plot agent network."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        
        # Simple circular layout
        angles = np.linspace(0, 2 * np.pi, self.n_agents, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Draw edges
        for agent in self.agents:
            for neighbor_id in agent.connections[:5]:  # Limit for clarity
                ax.plot(
                    [positions[agent.id, 0], positions[neighbor_id, 0]],
                    [positions[agent.id, 1], positions[neighbor_id, 1]],
                    'gray', alpha=0.1, linewidth=0.5
                )
        
        # Draw nodes
        if show_state and self.agents[0].state.get(show_state) is not None:
            colors = [agent.state.get(show_state, 0) for agent in self.agents]
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1],
                c=colors, cmap='viridis', s=20, alpha=0.7
            )
            plt.colorbar(scatter, ax=ax, label=show_state)
        else:
            ax.scatter(positions[:, 0], positions[:, 1], s=20, alpha=0.7)
        
        ax.set_title(f'Agent Network ({self.network_type})')
        ax.axis('equal')
        ax.axis('off')
        
        return ax
    
    def summary(self) -> str:
        """Return simulation summary."""
        lines = [
            "Agent-Based Simulator Summary",
            "=" * 50,
            f"Agents: {self.n_agents}",
            f"Network type: {self.network_type}",
            f"Average connections: {np.mean([len(a.connections) for a in self.agents]):.1f}",
            "",
        ]
        
        if self._history:
            lines.append(f"Simulation steps: {len(self._history)}")
            
            final_state = self._history[-1].aggregate_state
            lines.append("\nFinal aggregate state:")
            for key, value in list(final_state.items())[:10]:
                lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append("No simulation run yet")
        
        return "\n".join(lines)
