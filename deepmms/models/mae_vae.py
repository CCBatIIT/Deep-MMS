"""
Masked Autoencoder (MAE) for molecular coordinates (ViT-style, atom-level masking).

During training a fraction ``mask_ratio`` of atoms are randomly masked from
the encoder.  The decoder reconstructs all atom positions, but the training
loss is computed only on the masked atoms.  At inference all atoms are encoded
and decoded normally.

JSON config extras
------------------
embed_dim  : int   – atom token embedding width (default min(128, input_size))
mask_ratio : float – fraction of atoms masked during training (default 0.75)
num_heads  : int   – attention heads in transformer blocks (default 4)
ffn_mult   : float – FFN width multiplier (default 4.0)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder
from .vae import reparameterize
from .transformer_vae import _TransformerBlock, _AtomEmbedding


class MaskedAutoencoder(MolecularAutoencoder):
    """
    Masked Autoencoder (MAE) VAE for molecular coordinates.

    Encoder: embeds all atoms, randomly drops masked tokens, applies N
    transformer blocks on the unmasked subset, then pools → z_mean, z_logvar.

    Decoder: expands z to all n_atoms positions, replacing masked positions
    with a learned mask token, applies N transformer blocks, then Dense(3) per
    atom.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        All elements equal embed_dim; length = number of transformer blocks.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-block dropout rates.
    is_batchnorm : bool
        Accepted for API compatibility; unused (LayerNorm is used).
    mask_ratio : float
        Fraction of atoms masked during training (default 0.75).
    num_heads : int
        Attention heads (default 4).
    ffn_mult : float
        FFN width multiplier (default 4.0).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    mask_ratio: float = 0.75
    num_heads: int = 4
    ffn_mult: float = 4.0

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Compute hidden layer widths from experiment config.

        Parameters
        ----------
        input_size : int
        n_latents : int
        dropout_rates : list of float
            Length = number of transformer blocks.
        json_params : dict
            Reads ``embed_dim`` (default min(128, input_size)),
            ``num_heads`` (default 4).

        Returns
        -------
        list of int
            [embed_dim] * n_blocks.
        """
        embed_dim = json_params.get("embed_dim", min(128, input_size))
        num_heads = json_params.get("num_heads", 4)
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + 1) * num_heads
        return [embed_dim] * len(dropout_rates)

    def setup(self):
        """Wire all sub-modules."""
        self._n_atoms = self.input_size // 3
        embed_dim = self.hidden_layers[0] if self.hidden_layers else min(128, self.input_size)
        n_blocks = len(self.hidden_layers)
        ffn_dim = int(embed_dim * self.ffn_mult)
        dr = self.dropout_rates[0] if self.dropout_rates else 0.0

        self._atom_embed = _AtomEmbedding(self._n_atoms, embed_dim)

        self._enc_blocks = [
            _TransformerBlock(self.num_heads, ffn_dim, dr)
            for _ in range(n_blocks)
        ]
        self._enc_norm = nn.LayerNorm()
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        # Decoder sub-modules
        self._mask_token = self.param(
            "mask_token",
            nn.initializers.truncated_normal(stddev=0.02),
            (embed_dim,),
        )
        self._latent_to_embed = nn.Dense(embed_dim)
        self._dec_blocks = [
            _TransformerBlock(self.num_heads, ffn_dim, dr)
            for _ in range(n_blocks)
        ]
        self._dec_norm = nn.LayerNorm()
        self._coord_proj = nn.Dense(3)

    def _get_mask(self, rng, n_atoms: int):
        """
        Sample a random atom mask.

        Parameters
        ----------
        rng : jax.random.PRNGKey
        n_atoms : int

        Returns
        -------
        mask : array of bool, shape (n_atoms,)
            True for atoms that are MASKED (excluded from encoder).
        """
        n_keep = max(1, int(round(n_atoms * (1.0 - self.mask_ratio))))
        perm = jax.random.permutation(rng, n_atoms)
        kept = perm[:n_keep]
        mask = jnp.ones(n_atoms, dtype=bool)
        mask = mask.at[kept].set(False)
        return mask

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Encode all atoms (no masking at inference).

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of arrays, each shape (batch, latents)
        """
        h = x.reshape(-1, self._n_atoms, 3)
        h = self._atom_embed(h)                               # (B, N, E)
        for blk in self._enc_blocks:
            h = blk(h, train=train)
        h = self._enc_norm(h)
        pooled = jnp.mean(h, axis=1)                          # (B, E)
        return self._z_mean(pooled), self._z_logvar(pooled)

    def _encode_masked(self, x, rng, train: bool):
        """
        Encode only unmasked atoms; return (z_mean, z_logvar, mask).

        Parameters
        ----------
        x : array, shape (batch, input_size)
        rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        z_mean : array, shape (batch, latents)
        z_logvar : array, shape (batch, latents)
        mask : array of bool, shape (n_atoms,)
        """
        mask = self._get_mask(rng, self._n_atoms)             # (N,) True=masked
        h = x.reshape(-1, self._n_atoms, 3)
        h = self._atom_embed(h)                               # (B, N, E)
        # Keep only unmasked tokens
        h_unmasked = h[:, ~mask, :]                           # (B, n_keep, E)
        for blk in self._enc_blocks:
            h_unmasked = blk(h_unmasked, train=train)
        h_unmasked = self._enc_norm(h_unmasked)
        pooled = jnp.mean(h_unmasked, axis=1)
        return self._z_mean(pooled), self._z_logvar(pooled), mask

    def decode(self, z, z_rng=None, train: bool = False):
        """
        Decode latent vector to all atom coordinates.

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
        h = self._latent_to_embed(z)                          # (B, E)
        h = jnp.tile(h[:, None, :], (1, self._n_atoms, 1))   # (B, N, E)
        for blk in self._dec_blocks:
            h = blk(h, train=train)
        h = self._dec_norm(h)
        out = self._coord_proj(h)                             # (B, N, 3)
        return out.reshape(batch, self.input_size)

    def _decode_with_mask(self, z, mask, train: bool):
        """
        Decode, inserting mask tokens for masked positions.

        Parameters
        ----------
        z : array, shape (batch, latents)
        mask : array of bool, shape (n_atoms,)
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        batch = z.shape[0]
        embed_dim = self._latent_to_embed.features
        h_base = self._latent_to_embed(z)                     # (B, E)
        h_base = jnp.tile(h_base[:, None, :], (1, self._n_atoms, 1))  # (B, N, E)
        # Replace masked positions with mask_token
        mask_tok = jnp.tile(
            self._mask_token[None, None, :], (batch, self._n_atoms, 1)
        )
        h = jnp.where(mask[None, :, None], mask_tok, h_base)
        for blk in self._dec_blocks:
            h = blk(h, train=train)
        h = self._dec_norm(h)
        out = self._coord_proj(h)
        return out.reshape(batch, self.input_size)

    def __call__(self, x, z_rng, train: bool):
        """
        Full MAE forward pass.

        During training, encoder sees only unmasked atoms; decoder reconstructs
        all atoms.  During inference, no masking is applied.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        tuple of (decoded_full, z_mean, z_logvar)
        """
        if train:
            rng1, rng2 = jax.random.split(z_rng)
            z_mean, z_logvar, mask = self._encode_masked(x, rng1, train)
            z = reparameterize(rng2, z_mean, z_logvar)
            decoded = self._decode_with_mask(z, mask, train)
        else:
            z_mean, z_logvar = self.encode(x, train=train)
            z = reparameterize(z_rng, z_mean, z_logvar)
            decoded = self.decode(z, train=train)
        return decoded, z_mean, z_logvar

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

    def get_mask(self, rng):
        """
        Expose mask generation for MAETrainer.

        Parameters
        ----------
        rng : jax.random.PRNGKey

        Returns
        -------
        mask : array of bool, shape (n_atoms,)
        """
        return self._get_mask(rng, self._n_atoms)
