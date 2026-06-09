"""
Training subpackage: gradient-based and evolutionary training harnesses.

Available trainers
------------------
Experiment           – Adam gradient-based trainer (default)
EvolutionaryTrainer  – OpenES gradient-free trainer (any model)
NEATTrainer          – OpenES + topology growth (NEATAutoencoder)
BetaVAETrainer       – RMSD + beta-weighted KL loss (BetaVAE)
VQVAETrainer         – RMSD + commitment loss (VQVAE)
HVAETrainer          – RMSD + KL(z1) + KL(z2) (HierarchicalVAE)
FlowTrainer          – NLL = 0.5*||z||^2 - log_det (RealNVPFlow)
MAETrainer           – masked-atom RMSD loss (MaskedAutoencoder)
"""

from .trainer import Experiment
from .evolutionary import EvolutionaryTrainer, NEATTrainer
from .specialized import (
    BetaVAETrainer,
    VQVAETrainer,
    HVAETrainer,
    FlowTrainer,
    MAETrainer,
)

__all__ = [
    "Experiment",
    "EvolutionaryTrainer",
    "NEATTrainer",
    "BetaVAETrainer",
    "VQVAETrainer",
    "HVAETrainer",
    "FlowTrainer",
    "MAETrainer",
]
