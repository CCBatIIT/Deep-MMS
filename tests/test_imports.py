"""
Import smoke tests — verify every public module and symbol loads without error.
These tests require no data files and should be near-instant.
"""

import importlib
import pytest


PUBLIC_MODULES = [
    "deepmms",
    "deepmms.utils",
    "deepmms.data",
    "deepmms.config",
    "deepmms.dispatch",
    "deepmms.cli",
    "deepmms.models",
    "deepmms.models.base",
    "deepmms.models.vae",
    "deepmms.models.transformer_vae",
    "deepmms.models.neat_vae",
    "deepmms.models.beta_vae",
    "deepmms.models.vq_vae",
    "deepmms.models.equivariant_vae",
    "deepmms.models.perceiver_vae",
    "deepmms.models.hierarchical_vae",
    "deepmms.models.se3_transformer",
    "deepmms.models.mamba_vae",
    "deepmms.models.flow_vae",
    "deepmms.models.mae_vae",
    "deepmms.models.kan_vae",
    "deepmms.training",
    "deepmms.training.loss",
    "deepmms.training.optimizer",
    "deepmms.training.trainer",
    "deepmms.training.evolutionary",
    "deepmms.training.specialized",
    "deepmms.analysis",
    "deepmms.analysis.reconstruction",
    "deepmms.analysis.clustering",
    "deepmms.analysis.perturbation",
    "deepmms.analysis.plotting",
]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_module_imports(module_name):
    """Each public module must import without raising an exception."""
    mod = importlib.import_module(module_name)
    assert mod is not None


TOP_LEVEL_EXPORTS = [
    "Experiment",
    "Analyzer",
    "BatchNorm_VAE",
    "violin_data",
    "violin_plots",
]


@pytest.mark.parametrize("symbol", TOP_LEVEL_EXPORTS)
def test_top_level_exports(symbol):
    """Every symbol listed in deepmms.__all__ must be accessible on the package."""
    import deepmms
    assert hasattr(deepmms, symbol), f"deepmms.{symbol} not found"


MODEL_EXPORTS = [
    "MolecularAutoencoder", "BatchNorm_VAE", "BVEncoder", "BVDecoder",
    "reparameterize", "TransformerVAE", "NEATAutoencoder", "neat_grow",
    "BetaVAE", "VQVAE", "EquivariantVAE", "PerceiverVAE", "HierarchicalVAE",
    "SE3TransformerVAE", "MambaVAE", "RealNVPFlow", "MaskedAutoencoder", "KANVAE",
]


@pytest.mark.parametrize("symbol", MODEL_EXPORTS)
def test_model_exports(symbol):
    """Every symbol in deepmms.models.__all__ must be accessible."""
    import deepmms.models as m
    assert hasattr(m, symbol), f"deepmms.models.{symbol} not found"


TRAINER_EXPORTS = [
    "Experiment", "EvolutionaryTrainer", "NEATTrainer",
    "BetaVAETrainer", "VQVAETrainer", "HVAETrainer", "FlowTrainer", "MAETrainer",
]


@pytest.mark.parametrize("symbol", TRAINER_EXPORTS)
def test_trainer_exports(symbol):
    """Every symbol in deepmms.training.__all__ must be accessible."""
    import deepmms.training as t
    assert hasattr(t, symbol), f"deepmms.training.{symbol} not found"
