"""
Hierarchical VAE (HVAE): two-level latent hierarchy for molecular coordinates.

Level 1 (z1) encodes slow global modes with a small dimensionality.
Level 2 (z2) encodes local / residual variation conditioned on z1.

The decoder concatenates z1 and z2 and reconstructs Cartesian coordinates.
Used with HVAETrainer, which sums KL divergences from both levels.

JSON config extras
------------------
None — uses standard keys (latent_dim, dropout_rates, is_batchnorm).
The z1/z2 split is fixed at latents // 4 (at least 1) vs. the remainder.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder
from .vae import BVEncoder, BVDecoder, reparameterize
from ..training.loss import KL_loss


class HierarchicalVAE(MolecularAutoencoder):
    """
    Two-level hierarchical VAE for molecular coordinates.

    Encoder: a shared MLP backbone maps inputs to a hidden representation h.
    Two branches project h to (z1_mean, z1_logvar) and (h + z1_sample) to
    (z2_mean, z2_logvar) respectively.

    Decoder: concatenated [z1, z2] → MLP → reconstructed coordinates.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        Hidden layer widths for the shared backbone and decoder.
    latents : int
        Total latent dimensionality; split into latents//4 and the remainder.
    dropout_rates : list of float
        Per-layer dropout rates.
    is_batchnorm : bool
        Enable / disable batch normalisation.
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool

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
            Per-layer dropout rates; length determines number of hidden layers.
        json_params : dict
            Full JSON config (no HVAE-specific keys).

        Returns
        -------
        list of int
            Hidden layer widths (all equal to input_size).
        """
        return [input_size] * len(dropout_rates)

    def setup(self):
        """Wire shared backbone, two latent branches, and decoder."""
        n1 = max(1, self.latents // 4)
        n2 = self.latents - n1
        self._n1 = n1
        self._n2 = n2

        # Shared backbone (no latent heads — use BVEncoder architecture)
        # We build it as Dense layers manually to separate backbone from heads.
        d_hidden = list(self.hidden_layers)
        self._backbone = BVEncoder(d_hidden, n1, self.dropout_rates, self.is_batchnorm)
        # z1 heads are inside _backbone (mean/logvar for n1 dims).

        # Branch 2: takes [h, z1_sample] and produces z2_mean, z2_logvar
        # Input dim = hidden_layers[-1] + n1; we use a smaller head
        backbone_out_dim = d_hidden[-1] if d_hidden else self.input_size
        self._z2_mean = nn.Dense(n2)
        self._z2_logvar = nn.Dense(n2)
        self._branch2_proj = nn.Dense(backbone_out_dim)

        # Prior network p(z2 | z1): generates z2 mean/logvar from z1 alone
        self._prior_z2_mean = nn.Dense(n2)
        self._prior_z2_logvar = nn.Dense(n2)

        # Decoder: [z1, z2] (total latents dims) → input_size
        # Entry projection maps full latent vector → first hidden dim
        self._z_in_proj = nn.Dense(d_hidden[0] if d_hidden else self.input_size)
        self.decoder = BVDecoder(d_hidden, self.input_size, self.dropout_rates, self.is_batchnorm)

    def _encode_full(self, x, z_rng, train: bool):
        """
        Full encode returning z1 and z2 parameters.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        tuple of (z1_mean, z1_logvar, z2_mean, z2_logvar)
        """
        z1_mean, z1_logvar = self._backbone(x, train=train)
        rng1, rng2 = jax.random.split(z_rng)
        z1_sample = reparameterize(rng1, z1_mean, z1_logvar)

        # Branch 2: use a small MLP on z1_sample
        h2 = nn.relu(self._branch2_proj(z1_sample))
        z2_mean = self._z2_mean(h2)
        z2_logvar = self._z2_logvar(h2)
        return z1_mean, z1_logvar, z2_mean, z2_logvar

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Return (z1_mean, z1_logvar) — the top-level global summary.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of arrays, each shape (batch, latents//4)
        """
        z1_mean, z1_logvar = self._backbone(x, train=train)
        return z1_mean, z1_logvar

    def decode(self, z, z_rng=None, train: bool = False):
        """
        Decode concatenated [z1, z2] → coordinate space.

        Parameters
        ----------
        z : array, shape (batch, latents)
            Concatenated z1 and z2.
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        h = nn.relu(self._z_in_proj(z))
        return self.decoder(h, train=train)

    def __call__(self, x, z_rng, train: bool):
        """
        Full hierarchical forward pass.

        Returns (decoded, z1_mean, z1_logvar) for interface compatibility.
        z2 information is computed internally for aux_loss.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        tuple of (decoded, z1_mean, z1_logvar)
        """
        rng1, rng2 = jax.random.split(z_rng)
        z1_mean, z1_logvar, z2_mean, z2_logvar = self._encode_full(x, rng1, train)
        z1 = reparameterize(rng1, z1_mean, z1_logvar)
        z2 = reparameterize(rng2, z2_mean, z2_logvar)
        z = jnp.concatenate([z1, z2], axis=-1)
        decoded = self.decode(z, train=train)
        # Ensure prior network params are initialized during the forward pass
        _ = self._prior_z2_mean(z1)
        _ = self._prior_z2_logvar(z1)
        return decoded, z1_mean, z1_logvar

    def construct(self, z1_mean, z1_logvar, z_rng, train: bool = False):
        """
        Sample z1 from posterior, predict z2 via prior p(z2|z1), decode.

        Parameters
        ----------
        z1_mean : array, shape (batch, latents//4)
        z1_logvar : array, shape (batch, latents//4)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        rng1, rng2 = jax.random.split(z_rng)
        z1 = reparameterize(rng1, z1_mean, z1_logvar)
        # Prior network generates z2 from z1
        prior_z2_mean = self._prior_z2_mean(z1)
        prior_z2_logvar = self._prior_z2_logvar(z1)
        z2 = reparameterize(rng2, prior_z2_mean, prior_z2_logvar)
        z = jnp.concatenate([z1, z2], axis=-1)
        return self.decode(z, train=train)

    def aux_loss(self, x, z_rng, train: bool = False):
        """
        Sum of KL divergences from both levels KL(z1) + KL(z2).

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        float
            Scalar total KL loss.
        """
        rng1, _ = jax.random.split(z_rng)
        z1_mean, z1_logvar, z2_mean, z2_logvar = self._encode_full(x, rng1, train)
        return KL_loss(z1_mean, z1_logvar) + KL_loss(z2_mean, z2_logvar)
