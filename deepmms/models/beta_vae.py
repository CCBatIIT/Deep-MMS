"""
Beta-VAE: a disentangled variational autoencoder for molecular coordinates.

Subclasses BatchNorm_VAE with a single additional ``beta`` field that scales
the KL regularisation term.  The architecture is identical to BatchNorm_VAE;
the beta weight is applied by BetaVAETrainer during training.

JSON config extras
------------------
beta : float
    KL weight multiplier (default 4.0).  Higher values encourage
    more disentangled latent representations.
"""

import jax.numpy as jnp

from .vae import BatchNorm_VAE
from ..training.loss import KL_loss


class BetaVAE(BatchNorm_VAE):
    """
    Beta-VAE: BatchNorm_VAE with a tuneable KL weight.

    Identical MLP architecture as BatchNorm_VAE.  The ``beta`` field
    modulates the KL penalty when used with BetaVAETrainer.  The model
    itself does not change the forward pass; all beta logic lives in the
    trainer so that the same checkpoint can be evaluated with different
    beta values.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        Hidden layer widths shared by encoder and decoder.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-layer dropout rates.
    is_batchnorm : bool
        Enable / disable batch normalisation.
    beta : float
        KL weight multiplier (default 4.0).
    """

    beta: float = 4.0

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
            Full JSON config.  Reads ``beta`` (default 4.0); all other keys
            are delegated to BatchNorm_VAE.

        Returns
        -------
        list of int
            Hidden layer widths (all equal to input_size by default).
        """
        return [input_size] * len(dropout_rates)

    def aux_loss(self, x, z_rng, train: bool = False):
        """
        Compute the beta-weighted KL divergence auxiliary loss.

        Parameters
        ----------
        x : array, shape (batch, input_size)
            Input coordinate batch.
        z_rng : jax.random.PRNGKey
            RNG key for the reparameterisation trick.
        train : bool
            Training-mode flag.

        Returns
        -------
        float
            ``beta * KL_loss(z_mean, z_logvar)`` — the scaled KL penalty.
        """
        _, z_mean, z_logvar = self(x, z_rng, train=train)
        return self.beta * KL_loss(z_mean, z_logvar)
