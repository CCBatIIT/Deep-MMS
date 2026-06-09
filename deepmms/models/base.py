"""
Abstract base class for molecular autoencoder models built with Flax.

All concrete model classes (e.g. BatchNorm_VAE) should inherit from
MolecularAutoencoder and implement the three abstract methods so that
training and analysis code can be written against the interface rather
than a specific architecture.
"""

import abc
import flax.linen as nn


class MolecularAutoencoder(nn.Module, abc.ABC):
    """
    Abstract Flax module defining the interface for molecular autoencoders.

    Concrete subclasses must implement encode, decode, and __call__
    to be usable with the Experiment training harness.
    """

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Compute the hidden layer width list from experiment configuration.

        The default returns square layers (width == input_size) matching the
        original BatchNorm_VAE convention.  Subclasses override this to
        implement architecture-specific layer sizing (e.g. a fixed embed_dim
        for transformers, or a small start width for growing NEAT networks).

        Parameters
        ----------
        input_size : int
            Number of input features (n_atoms * 3).
        n_latents : int
            Latent dimensionality (unused by default but available to subclasses).
        dropout_rates : list of float
            Per-layer dropout rates; length determines number of hidden layers.
        json_params : dict
            Full JSON config; subclasses may read extra keys from it.

        Returns
        -------
        list of int
            Hidden layer widths, one per element in dropout_rates.
        """
        return [input_size] * len(dropout_rates)
    """
    Abstract Flax module defining the interface for molecular autoencoders.

    Concrete subclasses must implement encode, decode, and __call__
    to be usable with the Experiment training harness.
    """

    @abc.abstractmethod
    def encode(self, x, z_rng, train: bool = False):
        """
        Encode input coordinates to latent-space parameters.

        Parameters
        ----------
        x : array, shape (batch, n_features)
            Flattened coordinate input.
        z_rng : jax.random.PRNGKey
            RNG key (unused by some implementations, required by interface).
        train : bool
            Whether to run in training mode (affects dropout / batch-norm).

        Returns
        -------
        Depends on implementation; typically (z_mean, z_logvar).
        """

    @abc.abstractmethod
    def decode(self, z, z_rng, train: bool = False):
        """
        Decode a latent vector back to coordinate space.

        Parameters
        ----------
        z : array, shape (batch, n_latents)
            Latent coordinates.
        z_rng : jax.random.PRNGKey
            RNG key (unused by some implementations, required by interface).
        train : bool
            Whether to run in training mode.

        Returns
        -------
        array, shape (batch, n_features)
            Reconstructed flattened coordinates.
        """

    @abc.abstractmethod
    def __call__(self, x, z_rng, train: bool):
        """
        Full forward pass: encode then decode with reparameterisation.

        Parameters
        ----------
        x : array, shape (batch, n_features)
            Input coordinates.
        z_rng : jax.random.PRNGKey
            RNG key for the reparameterisation trick.
        train : bool
            Training-mode flag.

        Returns
        -------
        tuple of (reconstructed, z_mean, z_logvar)
        """
