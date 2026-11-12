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
PyCausalSim: Causal Discovery and Inference Through Simulation

A Python framework for discovering and validating causal relationships
using counterfactual simulation, designed for digital metrics optimization.
"""

__version__ = "0.6.0"
__author__ = "Brian Curry"

from .simulator import CausalSimulator
from .attribution import MarketingAttribution
from .experiment import ExperimentAnalysis
from .models import StructuralCausalModel
from .results import CausalEffect, SimulationResult, SensitivityResult
from . import futures

__all__ = [
    "CausalSimulator",
    "MarketingAttribution", 
    "ExperimentAnalysis",
    "StructuralCausalModel",
    "CausalEffect",
    "SimulationResult",
    "SensitivityResult",
    "futures",
]
