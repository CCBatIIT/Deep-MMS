"""
SE(3)-equivariant transformer (PaiNN-style) for molecular coordinates.

Each atom carries both scalar (invariant) and vector (equivariant) features.
Message passing layers update features by exchanging information along edges
weighted by pairwise distance.  Vector features transform correctly under
rotation because they are scaled by the unit direction vector.

After all message-passing layers the scalar features are mean-pooled over
atoms and projected to z_mean and z_logvar.  The decoder is a standard MLP.

JSON config extras
------------------
d_scalar    : int   – scalar feature dimension per atom (default 64)
d_vector    : int   – vector feature dimension per atom (default 16)
n_mp_layers : int   – number of message-passing layers (default 3)
cutoff_dist : float – distance cutoff in nm (default 1.0)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder
from .vae import BVDecoder, reparameterize


class _PaiNNLayer(nn.Module):
    """
    One PaiNN-style message-passing layer.

    Each neighbour j sends a scalar message and a vector message to i.
    The scalar message is an MLP of (distance_features, s_j).
    The vector message is a scaling of the direction (x_j - x_i) / D_ij.

    Attributes
    ----------
    d_scalar : int
        Scalar feature width.
    d_vector : int
        Vector feature width (number of independent direction channels).
    cutoff : float
        Distance cutoff; interactions beyond this radius are zeroed.
    """

    d_scalar: int
    d_vector: int
    cutoff: float

    @nn.compact
    def __call__(self, s, v, coords):
        """
        One message-passing step.

        Parameters
        ----------
        s : array, shape (batch, n_atoms, d_scalar)
            Scalar invariant features.
        v : array, shape (batch, n_atoms, d_vector, 3)
            Vector equivariant features.
        coords : array, shape (batch, n_atoms, 3)
            Atom Cartesian coordinates (needed for direction vectors).

        Returns
        -------
        s_new : array, same shape as s
        v_new : array, same shape as v
        """
        batch, n_atoms, _ = s.shape

        # Pairwise differences and distances
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 3)
        D = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)    # (B, N, N)
        unit = diff / (D[..., None] + 1e-8)                   # (B, N, N, 3)

        # Cutoff mask
        mask = (D < self.cutoff).astype(jnp.float32)          # (B, N, N)

        # Scalar message: phi_s(D_ij, s_j)
        # Encode distance as a simple feature: [D, 1/D, cos(pi*D/cutoff)]
        d_feat = jnp.stack([
            D,
            1.0 / (D + 1e-8),
            jnp.cos(jnp.pi * D / self.cutoff),
        ], axis=-1)  # (B, N, N, 3)

        # Concatenate s_j with distance features
        s_j = s[:, None, :, :]                               # (B, 1, N, d_s)
        s_j = jnp.broadcast_to(s_j, (batch, n_atoms, n_atoms, self.d_scalar))
        inp = jnp.concatenate([d_feat, s_j], axis=-1)        # (B, N, N, 3+d_s)

        # MLP → scalar update + vector scale
        h = nn.Dense(self.d_scalar)(inp)
        h = nn.silu(h)
        phi_s = nn.Dense(self.d_scalar)(h)                   # (B, N, N, d_s)
        phi_v = nn.Dense(self.d_vector)(h)                   # (B, N, N, d_v)

        # Apply mask
        phi_s = phi_s * mask[..., None]
        phi_v = phi_v * mask[..., None]

        # Aggregate scalar messages: sum over neighbours j
        ds = jnp.sum(phi_s, axis=2)                          # (B, N, d_s)
        s_new = s + nn.Dense(self.d_scalar)(ds)

        # Aggregate vector messages: phi_v_j * unit_ij, summed over j
        # unit_ij: (B, N, N, 3), phi_v: (B, N, N, d_v)
        # → (B, N, d_v, 3)
        dv = jnp.einsum("bijn,bijv->bivn", unit, phi_v)     # (B, N, d_v, 3)
        v_new = v + dv

        return s_new, v_new


class SE3TransformerVAE(MolecularAutoencoder):
    """
    SE(3)-equivariant VAE with PaiNN-style message passing.

    Scalar features are rotation-invariant; vector features transform
    equivariantly with rotation.  After message passing, scalar features
    are mean-pooled to produce z_mean and z_logvar.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        All elements equal d_scalar; length = n_mp_layers.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-layer dropout rates (applied in decoder MLP only).
    is_batchnorm : bool
        Enable / disable batch normalisation in decoder.
    d_scalar : int
        Scalar feature dimension per atom (default 64).
    d_vector : int
        Vector feature dimension per atom (default 16).
    n_mp_layers : int
        Number of message-passing layers (default 3).
    cutoff_dist : float
        Interaction cutoff in nm (default 1.0).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    d_scalar: int = 64
    d_vector: int = 16
    n_mp_layers: int = 3
    cutoff_dist: float = 1.0

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Compute hidden layer widths from experiment config.

        Parameters
        ----------
        input_size : int
        n_latents : int
        dropout_rates : list of float
        json_params : dict
            Reads ``d_scalar`` (default 64), ``n_mp_layers`` (default 3).

        Returns
        -------
        list of int
            [d_scalar] * n_mp_layers.
        """
        d_scalar = json_params.get("d_scalar", 64)
        n_mp = json_params.get("n_mp_layers", 3)
        return [d_scalar] * n_mp

    def setup(self):
        """Wire atom embedding, message-passing layers, and decoder."""
        self._n_atoms = self.input_size // 3
        n_mp = len(self.hidden_layers) if self.hidden_layers else self.n_mp_layers

        self._scalar_embed = nn.Dense(self.d_scalar)
        self._vector_embed = nn.Dense(self.d_vector)  # initialises v channels

        self._mp_layers = [
            _PaiNNLayer(self.d_scalar, self.d_vector, self.cutoff_dist)
            for _ in range(n_mp)
        ]
        self._pool_norm = nn.LayerNorm()   # stabilises pooled features after MP
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        self.decoder = BVDecoder(
            list(self.hidden_layers),
            self.input_size,
            self.dropout_rates,
            self.is_batchnorm,
        )

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Encode flat coordinates via equivariant message passing.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of arrays, each shape (batch, latents)
        """
        coords = x.reshape(-1, self._n_atoms, 3)              # (B, N, 3)

        # Initialise scalar features from raw coordinates (invariant proxy: norms)
        norms = jnp.linalg.norm(coords, axis=-1, keepdims=True)  # (B, N, 1)
        s = self._scalar_embed(jnp.concatenate([coords, norms], axis=-1))  # (B, N, d_s)

        # Initialise vector features as zero (they will be filled by MP)
        v = jnp.zeros((coords.shape[0], self._n_atoms, self.d_vector, 3))

        for layer in self._mp_layers:
            s, v = layer(s, v, coords)

        pooled = jnp.mean(s, axis=1)                           # (B, d_s)
        pooled = self._pool_norm(pooled)                       # stabilise after MP
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
