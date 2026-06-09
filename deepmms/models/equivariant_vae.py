"""
SE(3)-Invariant VAE using pairwise distance features (SchNet-style).

Because the training data are pre-aligned (superposed), the encoder can use
rotationally-invariant pairwise distances as features without losing information.
Distances are expanded with a radial basis function (RBF) and passed through
multiple interaction layers.  The decoder predicts Cartesian coordinates
directly, which is valid because the data are already in a fixed reference frame.

JSON config extras
------------------
embed_dim   : int   – atom feature width (default min(128, input_size))
n_rbf       : int   – number of RBF centres (default 32)
cutoff_dist : float – distance cutoff in nm (default 1.0)
n_interactions : int – number of SchNet interaction layers (default 3)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from .base import MolecularAutoencoder
from .vae import BVDecoder, reparameterize


class _RBFExpansion(nn.Module):
    """
    Radial basis function expansion of interatomic distances.

    Computes f_k(r) = exp(-gamma * (r - mu_k)^2) for k = 1..n_rbf with
    centres linearly spaced from 0 to cutoff_dist and a smooth envelope.

    Attributes
    ----------
    n_rbf : int
        Number of basis functions.
    cutoff : float
        Distance cutoff in nm.
    """

    n_rbf: int
    cutoff: float

    @nn.compact
    def __call__(self, D):
        """
        Expand pairwise distances to RBF features.

        Parameters
        ----------
        D : array, shape (...,)
            Pairwise distances.

        Returns
        -------
        array, shape (..., n_rbf)
        """
        mu = jnp.linspace(0.0, self.cutoff, self.n_rbf)
        delta = mu[1] - mu[0] + 1e-8
        gamma = 1.0 / (2.0 * delta ** 2)
        rbf = jnp.exp(-gamma * (D[..., None] - mu) ** 2)
        # Smooth envelope: u(r) = 1 - 0.5*(r/cutoff)^2 for r < cutoff, else 0
        envelope = jnp.where(
            D[..., None] < self.cutoff,
            1.0 - 0.5 * (D[..., None] / self.cutoff) ** 2,
            0.0,
        )
        return rbf * envelope


class _InteractionBlock(nn.Module):
    """
    SchNet-style interaction block for atom-level message passing.

    Attributes
    ----------
    embed_dim : int
        Atom feature width.
    n_rbf : int
        Size of the RBF expansion.
    """

    embed_dim: int
    n_rbf: int

    @nn.compact
    def __call__(self, h, rbf_ij):
        """
        One interaction step updating atom features.

        Parameters
        ----------
        h : array, shape (batch, n_atoms, embed_dim)
            Current atom features.
        rbf_ij : array, shape (batch, n_atoms, n_atoms, n_rbf)
            RBF-expanded pairwise distance features.

        Returns
        -------
        array, shape (batch, n_atoms, embed_dim)
            Updated atom features.
        """
        # Filter network: maps RBF → embed_dim continuous filters
        W = nn.Dense(self.embed_dim)(rbf_ij)  # (B, N, N, E)
        W = nn.tanh(W)

        # Message: for each atom i, aggregate filtered messages from all j
        # h_j: (B, 1, N, E) broadcast over i
        h_j = h[:, None, :, :]  # (B, 1, N, E)
        msg = jnp.sum(W * h_j, axis=2)  # (B, N, E)

        # Update rule
        msg = nn.Dense(self.embed_dim)(nn.tanh(nn.Dense(self.embed_dim)(msg)))
        return h + msg


class EquivariantVAE(MolecularAutoencoder):
    """
    SE(3)-invariant VAE using SchNet-style pairwise distance features.

    The encoder computes pairwise distances, expands them with RBFs, runs
    multiple interaction layers, then mean-pools over atoms to produce
    z_mean and z_logvar.  The decoder is a standard MLP from latents to
    Cartesian coordinates (valid because data are pre-superposed).

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        All elements equal embed_dim; length = n_interactions.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-layer dropout rates (applied in the MLP decoder).
    is_batchnorm : bool
        Enable / disable batch normalisation in decoder.
    n_rbf : int
        Number of RBF centres (default 32).
    cutoff_dist : float
        Distance cutoff in nm (default 1.0).
    n_interactions : int
        Number of SchNet interaction layers (default 3).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    n_rbf: int = 32
    cutoff_dist: float = 1.0
    n_interactions: int = 3

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
            Per-layer dropout rates; length not used — n_interactions controls depth.
        json_params : dict
            Full JSON config.  Reads ``embed_dim`` (default min(128, input_size)),
            ``n_interactions`` (default 3).

        Returns
        -------
        list of int
            [embed_dim] * n_interactions.
        """
        embed_dim = json_params.get("embed_dim", min(128, input_size))
        n_interactions = json_params.get("n_interactions", 3)
        return [embed_dim] * n_interactions

    def setup(self):
        """Initialise sub-modules."""
        self._n_atoms = self.input_size // 3
        embed_dim = self.hidden_layers[0] if self.hidden_layers else min(128, self.input_size)
        n_int = len(self.hidden_layers) if self.hidden_layers else self.n_interactions

        self._rbf = _RBFExpansion(self.n_rbf, self.cutoff_dist)
        self._atom_init = nn.Dense(embed_dim)
        self._interactions = [
            _InteractionBlock(embed_dim, self.n_rbf) for _ in range(n_int)
        ]
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        self.decoder = BVDecoder(
            list(self.hidden_layers),
            self.input_size,
            self.dropout_rates,
            self.is_batchnorm,
        )

    def _compute_distances(self, x):
        """
        Compute pairwise distances from flat coordinate array.

        Parameters
        ----------
        x : array, shape (batch, input_size)

        Returns
        -------
        D : array, shape (batch, n_atoms, n_atoms)
        """
        coords = x.reshape(-1, self._n_atoms, 3)  # (B, N, 3)
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 3)
        D = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)   # (B, N, N)
        return D

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Encode input to (z_mean, z_logvar) via pairwise distance features.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of arrays, each shape (batch, latents)
        """
        D = self._compute_distances(x)                        # (B, N, N)
        rbf_ij = self._rbf(D)                                 # (B, N, N, n_rbf)

        # Initial atom features from mean-pooled RBF
        h_init = jnp.mean(rbf_ij, axis=2)                    # (B, N, n_rbf)
        h = self._atom_init(h_init)                           # (B, N, embed_dim)

        for block in self._interactions:
            h = block(h, rbf_ij)

        pooled = jnp.mean(h, axis=1)                          # (B, embed_dim)
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
        return self.decoder(z, train=train)

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
