"""
Model subpackage: all molecular autoencoder architectures.

Available models
----------------
BatchNorm_VAE      – symmetric MLP VAE with optional BatchNorm/Dropout (default)
TransformerVAE     – atom-token self-attention VAE; captures long-range interactions
NEATAutoencoder    – tanh MLP that grows topology on loss plateaus (use with NEATTrainer)
BetaVAE            – beta-VAE: MLP VAE with tuneable KL weight (use with BetaVAETrainer)
VQVAE              – vector-quantized VAE with discrete codebook latents
EquivariantVAE     – SE(3)-invariant VAE using SchNet-style pairwise distance features
PerceiverVAE       – O(n_atoms) cross-attention Perceiver VAE
HierarchicalVAE    – two-level hierarchical VAE (use with HVAETrainer)
SE3TransformerVAE  – PaiNN-style equivariant message-passing VAE
MambaVAE           – selective SSM (Mamba-inspired) VAE treating atoms as sequence tokens
RealNVPFlow        – RealNVP normalising flow (use with FlowTrainer)
MaskedAutoencoder  – MAE with atom-level masking (use with MAETrainer)
KANVAE             – Kolmogorov-Arnold Network VAE with learnable B-spline activations
"""

from .base import MolecularAutoencoder
from .vae import BatchNorm_VAE, BVEncoder, BVDecoder, reparameterize
from .transformer_vae import TransformerVAE
from .neat_vae import NEATAutoencoder, grow as neat_grow
from .beta_vae import BetaVAE
from .vq_vae import VQVAE
from .equivariant_vae import EquivariantVAE
from .perceiver_vae import PerceiverVAE
from .hierarchical_vae import HierarchicalVAE
from .se3_transformer import SE3TransformerVAE
from .mamba_vae import MambaVAE
from .flow_vae import RealNVPFlow
from .mae_vae import MaskedAutoencoder
from .kan_vae import KANVAE

__all__ = [
    "MolecularAutoencoder",
    "BatchNorm_VAE",
    "BVEncoder",
    "BVDecoder",
    "reparameterize",
    "TransformerVAE",
    "NEATAutoencoder",
    "neat_grow",
    "BetaVAE",
    "VQVAE",
    "EquivariantVAE",
    "PerceiverVAE",
    "HierarchicalVAE",
    "SE3TransformerVAE",
    "MambaVAE",
    "RealNVPFlow",
    "MaskedAutoencoder",
    "KANVAE",
]
