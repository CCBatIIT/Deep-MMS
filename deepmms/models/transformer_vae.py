"""
TransformerVAE: attention-based variational autoencoder for molecular coordinates.

Treats each heavy atom as a sequence token with 3-dimensional features (x, y, z).
Self-attention layers in the encoder allow every atom to attend to every other atom,
naturally capturing long-range inter-atomic relationships without explicitly encoding
molecular topology.  The decoder expands the latent vector back to per-atom features
via cross-attention over a set of learned atom queries.

Drop-in replacement for BatchNorm_VAE: same __call__ / encode / decode interface
and compatible with the Experiment training harness.

JSON config extras (all optional):
    embed_dim   : int   – token embedding width (default: min(256, input_size))
    num_heads   : int   – attention heads per block (default: 4)
    ffn_mult    : float – FFN hidden width as multiple of embed_dim (default: 4.0)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder
from .vae import reparameterize


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → MHA → residual → LayerNorm → FFN → residual."""

    num_heads: int
    ffn_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool):
        # Self-attention branch
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(y, y, deterministic=not train)
        x = x + y

        # Feed-forward branch
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.ffn_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(x.shape[-1])(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        return x + y


class _AtomEmbedding(nn.Module):
    """Project (batch, n_atoms, 3) → (batch, n_atoms, embed_dim) with positional encoding."""

    n_atoms: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        atom_proj = nn.Dense(self.embed_dim)(x)
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.n_atoms, self.embed_dim),
        )
        return atom_proj + pos_embed


class TransformerVAE(MolecularAutoencoder):
    """
    Atom-token transformer variational autoencoder.

    The encoder reshapes flat coordinates to (n_atoms, 3), projects each atom to
    embed_dim, applies N transformer blocks with self-attention, then mean-pools
    over atoms to produce z_mean and z_logvar.

    The decoder projects the latent vector to embed_dim, tiles it across n_atoms
    as initial atom queries, applies N transformer blocks, and projects each atom
    token back to 3 coordinates.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimension (n_atoms * 3).
    hidden_layers : tuple of int
        All elements must be equal to embed_dim.  Length = number of blocks.
    latents : int
        Latent dimensionality.
    dropout_rates : list of float
        Per-block dropout rate (one per transformer block).
    is_batchnorm : bool
        Accepted for API compatibility; transformer uses LayerNorm, not BatchNorm.
    num_heads : int
        Number of attention heads.  Must divide embed_dim evenly.
    ffn_mult : float
        Width of the FFN hidden layer as a multiple of embed_dim.
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    num_heads: int = 4
    ffn_mult: float = 4.0

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Return [embed_dim] * n_blocks.

        embed_dim defaults to min(256, input_size) and is rounded up to the
        nearest multiple of num_heads so attention head division is exact.
        Override by setting 'embed_dim' and optionally 'num_heads' in the JSON.
        """
        embed_dim = json_params.get("embed_dim", min(256, input_size))
        # num_heads must divide embed_dim; round embed_dim up if needed
        num_heads = json_params.get("num_heads", 4)
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + 1) * num_heads
        return [embed_dim] * len(dropout_rates)

    def setup(self):
        """Wire attention blocks, embedding, and projection layers."""
        self._n_atoms = self.input_size // 3
        embed_dim = self.hidden_layers[0]
        n_blocks = len(self.hidden_layers)
        ffn_dim = int(embed_dim * self.ffn_mult)

        self._atom_embed = _AtomEmbedding(self._n_atoms, embed_dim)
        self._enc_blocks = [
            _TransformerBlock(self.num_heads, ffn_dim, self.dropout_rates[i])
            for i in range(n_blocks)
        ]
        self._enc_norm = nn.LayerNorm()
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        # Decoder: latent → atom sequence → coordinates
        self._latent_proj = nn.Dense(embed_dim)
        self._dec_blocks = [
            _TransformerBlock(self.num_heads, ffn_dim, self.dropout_rates[i])
            for i in range(n_blocks)
        ]
        self._dec_norm = nn.LayerNorm()
        self._coord_proj = nn.Dense(3)

    def encode(self, x, z_rng=None, train: bool = False):
        """Encode (batch, n_atoms*3) → (z_mean, z_logvar), each (batch, latents)."""
        h = x.reshape(-1, self._n_atoms, 3)
        h = self._atom_embed(h)
        for block in self._enc_blocks:
            h = block(h, train=train)
        h = self._enc_norm(h)
        h = h.mean(axis=1)             # global mean-pool over atoms
        return self._z_mean(h), self._z_logvar(h)

    def decode(self, z, z_rng=None, train: bool = False):
        """Decode (batch, latents) → (batch, n_atoms*3)."""
        h = self._latent_proj(z)                        # (batch, embed_dim)
        h = jnp.tile(h[:, None, :], (1, self._n_atoms, 1))  # (batch, n_atoms, embed_dim)
        for block in self._dec_blocks:
            h = block(h, train=train)
        h = self._dec_norm(h)
        out = self._coord_proj(h)                       # (batch, n_atoms, 3)
        return out.reshape(-1, self.input_size)

    def __call__(self, x, z_rng, train: bool):
        """Full forward pass returning (reconstructed, z_mean, z_logvar)."""
        z_mean, z_logvar = self.encode(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, train=train), z_mean, z_logvar

    def construct(self, z_mean, z_logvar, z_rng, train=False):
        """Sample from posterior and decode without re-encoding."""
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, train=train)
