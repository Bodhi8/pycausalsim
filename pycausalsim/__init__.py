"""
PyCausalSim: Causal Discovery and Inference Through Simulation

A Python framework for discovering and validating causal relationships
using counterfactual simulation, designed for digital metrics optimization.
"""

__version__ = "0.1.0"
__author__ = "Brian Curry"

from .simulator import CausalSimulator
from .attribution import MarketingAttribution
from .experiment import ExperimentAnalysis
from .models import StructuralCausalModel
from .results import CausalEffect, SimulationResult, SensitivityResult

__all__ = [
    "CausalSimulator",
    "MarketingAttribution", 
    "ExperimentAnalysis",
    "StructuralCausalModel",
    "CausalEffect",
    "SimulationResult",
    "SensitivityResult",
]
