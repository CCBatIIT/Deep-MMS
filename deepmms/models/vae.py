"""
BatchNorm VAE architecture for molecular coordinate reconstruction.

Defines reparameterize, BVEncoder, BVDecoder, and the top-level BatchNorm_VAE
module that wires them together. All three classes support an is_batchnorm toggle
so that batch normalisation can be disabled without changing architecture depth
or the public API.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder


def reparameterize(z_rng, z_mean, z_logvar):
    """
    Draw a latent sample using the reparameterisation trick.

    Parameters
    ----------
    z_rng : jax.random.PRNGKey
        RNG key for sampling the noise vector.
    z_mean : array, shape (batch, n_latents)
        Predicted mean of the posterior.
    z_logvar : array, shape (batch, n_latents)
        Predicted log-variance of the posterior.

    Returns
    -------
    array, shape (batch, n_latents)
        Sampled latent vector z = mean + eps * std.
    """
    z_std = jnp.exp(0.5 * z_logvar)
    z_eps = jax.random.normal(z_rng, z_logvar.shape)
    return z_mean + z_eps * z_std


class BVEncoder(nn.Module):
    """
    Encoder half of the BatchNorm VAE.

    Applies Dense → ReLU → (optional BatchNorm) → Dropout for each hidden
    layer, then projects to separate mean and log-variance heads.

    Attributes
    ----------
    d_hidden : list of int
        Width of each hidden layer, in encoder order.
    latents : int
        Dimensionality of the latent space.
    dropout_rates : list of float
        Per-layer dropout rate (same length as d_hidden).
    is_batchnorm : bool
        Whether to insert BatchNorm after each activation.
    """

    d_hidden: list
    latents: int
    dropout_rates: list
    is_batchnorm: bool

    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(len(self.d_hidden)):
            x = nn.Dense(self.d_hidden[i])(x)
            x = nn.relu(x)
            if self.is_batchnorm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=not train)
        mean_x = nn.Dense(self.latents, name="fc5_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc5_logvar")(x)
        return mean_x, logvar_x


class BVDecoder(nn.Module):
    """
    Decoder half of the BatchNorm VAE.

    Mirrors BVEncoder by iterating hidden layers in reverse order:
    Dense → ReLU → (optional BatchNorm) → Dropout, followed by a final
    linear projection to the output (coordinate) dimension.

    Attributes
    ----------
    d_hidden : list of int
        Hidden layer widths (same list as encoder; iterated in reverse).
    out_dim : int
        Output dimensionality (n_atoms * 3).
    dropout_rates : list of float
        Per-layer dropout rate (applied in reverse order matching encoder).
    is_batchnorm : bool
        Whether to insert BatchNorm after each activation.
    """

    d_hidden: list
    out_dim: int
    dropout_rates: list
    is_batchnorm: bool

    @nn.compact
    def __call__(self, z, train: bool):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.Dense(self.d_hidden[i])(z)
            z = nn.relu(z)
            if self.is_batchnorm:
                z = nn.BatchNorm(use_running_average=not train)(z)
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=not train)
        z = nn.Dense(self.out_dim, name="f5")(z)
        return z


class BatchNorm_VAE(MolecularAutoencoder):
    """
    Variational autoencoder with optional batch normalisation for molecular coordinates.

    Wires BVEncoder and BVDecoder and exposes encode, decode, construct, and
    the full __call__ forward pass.  Inherits from MolecularAutoencoder so it
    satisfies the abstract interface required by Experiment.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        Hidden layer widths (broadcast to both encoder and decoder).
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-layer dropout rates; also determines number of hidden layers.
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
        """Return square hidden layers: one layer of width input_size per dropout rate."""
        return [input_size] * len(dropout_rates)

    def setup(self):
        self.encoder = BVEncoder(
            list(self.hidden_layers),
            self.latents,
            self.dropout_rates,
            self.is_batchnorm,
        )
        self.decoder = BVDecoder(
            list(self.hidden_layers),
            self.input_size,
            self.dropout_rates,
            self.is_batchnorm,
        )

    def __call__(self, x, z_rng, train: bool):
        """Full forward pass returning (reconstructed, z_mean, z_logvar)."""
        z_mean, z_logvar = self.encoder(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train), z_mean, z_logvar

    def construct(self, z_mean, z_logvar, z_rng, train=False):
        """Sample from the posterior and decode without re-encoding."""
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train)

    def encode(self, x, z_rng, train=False):
        """Return (z_mean, z_logvar) for input x."""
        return self.encoder(x, train=train)

    def decode(self, z, z_rng, train=False):
        """Decode latent vector z to coordinate space."""
        return self.decoder(z, train=train)
