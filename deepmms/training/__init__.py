"""
Training subpackage: gradient-based and evolutionary training harnesses.

Available trainers
------------------
Experiment           – Adam gradient-based trainer (default)
EvolutionaryTrainer  – OpenES gradient-free trainer (any model)
NEATTrainer          – OpenES + topology growth (NEATAutoencoder)
"""

from .trainer import Experiment
from .evolutionary import EvolutionaryTrainer, NEATTrainer

__all__ = ["Experiment", "EvolutionaryTrainer", "NEATTrainer"]
