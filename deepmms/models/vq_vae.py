"""
VQ-VAE: Vector-Quantized Variational Autoencoder for molecular coordinates.

The encoder maps input coordinates to a continuous ``z_e`` vector.  A learnable
codebook of ``codebook_size`` embeddings is then searched to find the nearest
entry, yielding a discrete ``z_q``.  A straight-through estimator allows
gradients to flow through the non-differentiable argmin operation.

JSON config extras
------------------
codebook_size : int
    Number of discrete codebook entries (default 512).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder
from .vae import BVEncoder, BVDecoder, reparameterize


class VectorQuantizer(nn.Module):
    """
    Differentiable vector-quantization codebook.

    Finds the nearest codebook entry for each encoder output using
    squared Euclidean distance and applies a straight-through gradient.

    Attributes
    ----------
    codebook_size : int
        Number of embedding vectors in the codebook.
    latents : int
        Dimensionality of each embedding vector.
    """

    codebook_size: int
    latents: int

    @nn.compact
    def __call__(self, z_e):
        """
        Quantize continuous encoder output.

        Parameters
        ----------
        z_e : array, shape (batch, latents)
            Continuous encoder output.

        Returns
        -------
        z_q : array, shape (batch, latents)
            Quantized vectors (codebook entries).
        z_q_st : array, shape (batch, latents)
            Straight-through quantized vectors for the decoder.
        """
        codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=1.0 / self.codebook_size),
            (self.codebook_size, self.latents),
        )
        # Distances: ||z_e||^2 + ||e||^2 - 2 z_e · e^T
        dists = (
            jnp.sum(z_e ** 2, axis=-1, keepdims=True)
            + jnp.sum(codebook ** 2, axis=-1)[None, :]
            - 2.0 * z_e @ codebook.T
        )
        indices = jnp.argmin(dists, axis=-1)      # (batch,)
        z_q = codebook[indices]                   # (batch, latents)
        # Straight-through: copy gradients from z_q_st to z_e
        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)
        return z_q, z_q_st


class VQVAE(MolecularAutoencoder):
    """
    Vector-Quantized VAE for molecular coordinate reconstruction.

    The encoder produces a continuous latent ``z_e``, which is then
    quantized to the nearest codebook vector ``z_q``.  The decoder
    receives ``z_q`` via a straight-through estimator.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        Hidden layer widths shared by encoder and decoder.
    latents : int
        Dimensionality of the latent / codebook vectors.
    dropout_rates : list of float
        Per-layer dropout rates.
    is_batchnorm : bool
        Enable / disable batch normalisation.
    codebook_size : int
        Number of discrete codebook entries (default 512).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    codebook_size: int = 512

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
            Full JSON config.  Reads ``codebook_size`` (default 512).

        Returns
        -------
        list of int
            Hidden layer widths (all equal to input_size).
        """
        return [input_size] * len(dropout_rates)

    def setup(self):
        """Wire encoder, quantizer, and decoder sub-modules."""
        self.encoder = BVEncoder(
            list(self.hidden_layers),
            self.latents,
            self.dropout_rates,
            self.is_batchnorm,
        )
        self.quantizer = VectorQuantizer(self.codebook_size, self.latents)
        self.decoder = BVDecoder(
            list(self.hidden_layers),
            self.input_size,
            self.dropout_rates,
            self.is_batchnorm,
        )

    def __call__(self, x, z_rng, train: bool):
        """
        Full forward pass with vector quantization.

        Parameters
        ----------
        x : array, shape (batch, input_size)
            Input coordinates.
        z_rng : jax.random.PRNGKey
            RNG key (unused; kept for interface compatibility).
        train : bool
            Training-mode flag.

        Returns
        -------
        decoded : array, shape (batch, input_size)
        z_e : array, shape (batch, latents)
            Continuous pre-quantization encoding.
        z_q : array, shape (batch, latents)
            Quantized codebook vectors.
        """
        z_e, _ = self.encoder(x, train=train)
        z_q, z_q_st = self.quantizer(z_e)
        decoded = self.decoder(z_q_st, train=train)
        return decoded, z_e, z_q

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Return continuous pre-quantization encoding.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of (z_e, zeros)
            z_e is the continuous encoder output; zeros matches the logvar
            slot required by the interface.
        """
        z_e, _ = self.encoder(x, train=train)
        return z_e, jnp.zeros_like(z_e)

    def decode(self, z, z_rng=None, train: bool = False):
        """
        Decode a continuous or quantized latent vector.

        Parameters
        ----------
        z : array, shape (batch, latents)
            Latent vector (not quantized at inference).
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        return self.decoder(z, train=train)

    def construct(self, z_mean, z_logvar, z_rng, train: bool = False):
        """
        Quantize z_mean to the nearest codebook entry and decode.

        Parameters
        ----------
        z_mean : array, shape (batch, latents)
            Encoder mean output (used as the pre-quantization vector).
        z_logvar : array, shape (batch, latents)
            Ignored for VQ-VAE; present for interface compatibility.
        z_rng : jax.random.PRNGKey
            Ignored; present for interface compatibility.
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        z_q, _ = self.quantizer(z_mean)
        return self.decoder(z_q, train=train)

    def aux_loss(self, x, z_rng, train: bool = False):
        """
        VQ-VAE commitment loss.

        Commitment loss = mean((stop_grad(z_q) - z_e)^2)
                        + 0.25 * mean((z_q - stop_grad(z_e))^2)

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        float
            Scalar commitment loss.
        """
        _, z_e, z_q = self(x, z_rng, train=train)
        commitment = jnp.mean((jax.lax.stop_gradient(z_q) - z_e) ** 2)
        codebook_loss = 0.25 * jnp.mean((z_q - jax.lax.stop_gradient(z_e)) ** 2)
        return commitment + codebook_loss
