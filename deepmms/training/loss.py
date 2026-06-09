"""
Loss functions for molecular coordinate reconstruction.

Provides the vmapped unweighted RMSD (atom_rmsd), a factory that builds a
mass-weighted variant (give_weighted_rmsd_func), and the KL divergence term
used in VAE training (KL_loss).
"""

import jax
import jax.numpy as jnp


@jax.vmap
def atom_rmsd(a, b):
    """
    Per-frame unweighted RMSD between flattened coordinate vectors a and b.

    Uses jax.vmap so it operates on a batch of frames, not individual frames.

    Parameters
    ----------
    a, b : array, shape (n_atoms*3,)
        Flattened coordinate frames (reshaped internally to (n_atoms, 3)).

    Returns
    -------
    float
        RMSD value in the same units as the input coordinates (nanometres).
    """
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    return jnp.sqrt(jnp.mean(jnp.sum((b - a) ** 2, axis=1)))


def give_weighted_rmsd_func(weights):
    """
    Build a mass-weighted RMSD function for a fixed weight vector.

    The returned function is vmapped over the batch dimension and computes
    sqrt(sum(w * d^2) / sum(w)) per frame.

    Parameters
    ----------
    weights : array, shape (n_atoms,)
        Per-atom weights (e.g. atomic masses).

    Returns
    -------
    callable
        Batched weighted-RMSD function with signature (a, b) → array of floats.
    """

    def weighted_atom_rmsd(a, b):
        """Per-frame mass-weighted RMSD; vmapped by give_weighted_rmsd_func."""
        a, b = a.reshape(-1, 3), b.reshape(-1, 3)
        return jnp.sqrt(
            jnp.sum(weights * jnp.sum((b - a) ** 2, axis=1)) / jnp.sum(weights)
        )

    return jax.vmap(weighted_atom_rmsd, in_axes=(0, 0))


def KL_loss(z_mean, z_logvar):
    """
    KL divergence between the posterior N(z_mean, exp(z_logvar)) and N(0,1).

    Returns the mean over the batch and summed over latent dimensions.

    Parameters
    ----------
    z_mean : array, shape (batch, n_latents)
    z_logvar : array, shape (batch, n_latents)

    Returns
    -------
    float
        Scalar KL divergence.
    """
    return jnp.mean(
        -0.5 * jnp.sum(1 + z_logvar - z_mean ** 2 - jnp.exp(z_logvar), axis=-1)
    )
