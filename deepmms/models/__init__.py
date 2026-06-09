"""
Model subpackage: all molecular autoencoder architectures.

Available models
----------------
BatchNorm_VAE      – symmetric MLP VAE with optional BatchNorm/Dropout (default)
TransformerVAE     – atom-token self-attention VAE; captures long-range interactions
NEATAutoencoder    – tanh MLP that grows topology on loss plateaus (use with NEATTrainer)
"""

from .base import MolecularAutoencoder
from .vae import BatchNorm_VAE, BVEncoder, BVDecoder, reparameterize
from .transformer_vae import TransformerVAE
from .neat_vae import NEATAutoencoder, grow as neat_grow

__all__ = [
    "MolecularAutoencoder",
    "BatchNorm_VAE",
    "BVEncoder",
    "BVDecoder",
    "reparameterize",
    "TransformerVAE",
    "NEATAutoencoder",
    "neat_grow",
]
