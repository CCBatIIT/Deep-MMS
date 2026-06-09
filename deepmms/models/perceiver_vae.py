"""
Perceiver VAE: O(n_atoms) cross-attention encoder for molecular coordinates.

The Perceiver architecture decouples the sequence length from the computation
by introducing a small set of latent query vectors that cross-attend to the
atom token sequence.  This scales linearly with the number of atoms rather
than quadratically.

JSON config extras
------------------
embed_dim        : int   – atom embedding and latent query width
                           (default min(256, input_size))
num_heads        : int   – attention heads in cross- and self-attention
                           (default 4)
n_latent_queries : int   – number of latent query slots (default 64)
ffn_mult         : float – FFN hidden width as multiple of embed_dim
                           (default 4.0)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder
from .vae import reparameterize


class _CrossAttentionBlock(nn.Module):
    """
    Cross-attention block: queries attend to key/value sequence.

    Applies LayerNorm → cross-MHA → residual, then LayerNorm → FFN → residual.

    Attributes
    ----------
    num_heads : int
    ffn_dim : int
    dropout_rate : float
    """

    num_heads: int
    ffn_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, queries, kv, train: bool):
        """
        Cross-attend queries to key-value sequence kv.

        Parameters
        ----------
        queries : array, shape (batch, n_q, embed_dim)
        kv : array, shape (batch, n_kv, embed_dim)
        train : bool

        Returns
        -------
        array, shape (batch, n_q, embed_dim)
        """
        dim = queries.shape[-1]
        q = nn.LayerNorm()(queries)
        k = nn.LayerNorm()(kv)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(q, k, deterministic=not train)
        queries = queries + y

        y = nn.LayerNorm()(queries)
        y = nn.Dense(self.ffn_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(dim)(y)
        return queries + y


class _SelfAttentionBlock(nn.Module):
    """
    Self-attention transformer block with pre-norm.

    Attributes
    ----------
    num_heads : int
    ffn_dim : int
    dropout_rate : float
    """

    num_heads: int
    ffn_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool):
        """
        Self-attention over sequence x.

        Parameters
        ----------
        x : array, shape (batch, seq_len, embed_dim)
        train : bool

        Returns
        -------
        array, same shape as x
        """
        dim = x.shape[-1]
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(y, y, deterministic=not train)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.ffn_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        y = nn.Dense(dim)(y)
        return x + y


class PerceiverVAE(MolecularAutoencoder):
    """
    Perceiver VAE for molecular coordinates.

    Encoder: atom tokens cross-attend into latent queries, then self-attend
    among queries; the resulting queries are mean-pooled → z_mean, z_logvar.

    Decoder: z is projected and tiled across atom positions, then N transformer
    self-attention blocks refine per-atom features → 3 coordinates per atom.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        All elements equal embed_dim; length = n_cross_layers.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-block dropout rates.
    is_batchnorm : bool
        Accepted for API compatibility; unused (LayerNorm is used instead).
    num_heads : int
        Attention heads (default 4).
    n_latent_queries : int
        Number of latent query slots (default 64).
    ffn_mult : float
        FFN width multiplier (default 4.0).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    num_heads: int = 4
    n_latent_queries: int = 64
    ffn_mult: float = 4.0

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Compute hidden layer widths from experiment config.

        Parameters
        ----------
        input_size : int
            Number of input features (n_atoms * 3).
        n_latents : int
            Latent dimensionality.
        dropout_rates : list of float
            Per-layer dropout rates; length = number of cross/self-attention blocks.
        json_params : dict
            Full JSON config.  Reads ``embed_dim`` (default min(256, input_size)),
            ``num_heads`` (default 4).

        Returns
        -------
        list of int
            [embed_dim] * n_blocks.
        """
        embed_dim = json_params.get("embed_dim", min(256, input_size))
        num_heads = json_params.get("num_heads", 4)
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + 1) * num_heads
        return [embed_dim] * len(dropout_rates)

    def setup(self):
        """Wire all sub-modules."""
        self._n_atoms = self.input_size // 3
        embed_dim = self.hidden_layers[0] if self.hidden_layers else min(256, self.input_size)
        n_blocks = len(self.hidden_layers)
        ffn_dim = int(embed_dim * self.ffn_mult)
        dr = self.dropout_rates[0] if self.dropout_rates else 0.0

        # Encoder sub-modules
        self._atom_proj = nn.Dense(embed_dim)
        self._pos_embed = self.param(
            "pos_embed",
            nn.initializers.truncated_normal(stddev=0.02),
            (self._n_atoms, embed_dim),
        )
        self._latent_queries = self.param(
            "latent_queries",
            nn.initializers.truncated_normal(stddev=0.02),
            (self.n_latent_queries, embed_dim),
        )
        self._cross_blocks = [
            _CrossAttentionBlock(self.num_heads, ffn_dim, dr)
            for _ in range(n_blocks)
        ]
        self._self_blocks = [
            _SelfAttentionBlock(self.num_heads, ffn_dim, dr)
            for _ in range(n_blocks)
        ]
        self._enc_norm = nn.LayerNorm()
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        # Decoder sub-modules
        self._latent_to_embed = nn.Dense(embed_dim)
        self._dec_blocks = [
            _SelfAttentionBlock(self.num_heads, ffn_dim, dr)
            for _ in range(n_blocks)
        ]
        self._dec_norm = nn.LayerNorm()
        self._coord_proj = nn.Dense(3)

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Encode flat coordinates to (z_mean, z_logvar).

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of arrays, each shape (batch, latents)
        """
        batch = x.shape[0]
        h = x.reshape(batch, self._n_atoms, 3)
        h = self._atom_proj(h) + self._pos_embed[None, :, :]   # (B, N, E)

        # Broadcast latent queries over batch
        q = jnp.tile(self._latent_queries[None, :, :], (batch, 1, 1))  # (B, Q, E)

        for cross_blk, self_blk in zip(self._cross_blocks, self._self_blocks):
            q = cross_blk(q, h, train=train)
            q = self_blk(q, train=train)

        q = self._enc_norm(q)
        pooled = jnp.mean(q, axis=1)                            # (B, E)
        return self._z_mean(pooled), self._z_logvar(pooled)

    def decode(self, z, z_rng=None, train: bool = False):
        """
        Decode latent vector to Cartesian coordinates.

        Parameters
        ----------
        z : array, shape (batch, latents)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        batch = z.shape[0]
        h = self._latent_to_embed(z)                             # (B, E)
        h = jnp.tile(h[:, None, :], (1, self._n_atoms, 1))      # (B, N, E)

        for blk in self._dec_blocks:
            h = blk(h, train=train)

        h = self._dec_norm(h)
        out = self._coord_proj(h)                                # (B, N, 3)
        return out.reshape(batch, self.input_size)

    def __call__(self, x, z_rng, train: bool):
        """
        Full forward pass returning (reconstructed, z_mean, z_logvar).

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        tuple of (decoded, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encode(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, train=train), z_mean, z_logvar

    def construct(self, z_mean, z_logvar, z_rng, train: bool = False):
        """
        Sample from the posterior and decode without re-encoding.

        Parameters
        ----------
        z_mean : array, shape (batch, latents)
        z_logvar : array, shape (batch, latents)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, train=train)
