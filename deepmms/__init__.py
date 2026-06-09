"""
Deep-MMS: a JAX/Flax package for molecular dynamics trajectory compression
via variational autoencoders.

Top-level exports: the training harness (Experiment), the analysis harness
(Analyzer), the canonical model architecture (BatchNorm_VAE), and the two
primary analysis functions (violin_data, violin_plots).
"""

from .training.trainer import Experiment
from .analysis.reconstruction import Analyzer, violin_data
from .models.vae import BatchNorm_VAE
from .analysis.plotting import violin_plots

__all__ = [
    "Experiment",
    "Analyzer",
    "BatchNorm_VAE",
    "violin_data",
    "violin_plots",
]
