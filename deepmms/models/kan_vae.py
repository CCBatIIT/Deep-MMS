"""
Kolmogorov-Arnold Network (KAN) VAE for molecular coordinates.

Replaces standard Dense layers with KAN layers where each edge has a
learnable B-spline activation.  The encoder and decoder are stacks of
KANLayers with standard Dense heads for the latent mean and log-variance.

B-spline basis functions are computed differentiably in pure JAX using
the de Boor recurrence on a uniform grid from -3 to 3.

JSON config extras
------------------
embed_dim : int – KAN layer width (default min(64, input_size))
kan_n_grid : int – number of B-spline grid intervals (default 5)
kan_order : int – B-spline polynomial order (default 3, i.e. cubic)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

from .base import MolecularAutoencoder
from .vae import reparameterize


def _bspline_basis(x, grid, order: int):
    """
    Compute B-spline basis functions on a uniform grid using de Boor recurrence.

    Parameters
    ----------
    x : array, shape (batch,)
        Input values.
    grid : array, shape (n_grid + 2*order,)
        Extended knot vector (uniform interior + clamped boundary).
    order : int
        B-spline polynomial order (degree = order, so cubic = 3).

    Returns
    -------
    B : array, shape (batch, n_basis)
        B-spline basis values where n_basis = len(grid) - order - 1.
    """
    # Order-0 B-splines: indicator function on each interval
    # x shape: (batch,), grid shape: (K,)
    x_exp = x[:, None]                                        # (batch, 1)
    g_l = grid[:-1][None, :]                                 # (1, K-1)
    g_r = grid[1:][None, :]                                  # (1, K-1)
    B = ((x_exp >= g_l) & (x_exp < g_r)).astype(jnp.float32)  # (batch, K-1)

    # Handle right boundary
    right_boundary = x_exp >= grid[-1]
    B = B.at[:, -1].set(B[:, -1] + right_boundary[:, 0].astype(jnp.float32))

    # De Boor recurrence
    for k in range(1, order + 1):
        denom_l = grid[k:-1] - grid[:-k-1] + 1e-8
        denom_r = grid[k+1:] - grid[1:-k] + 1e-8

        left = (x_exp - grid[:-k-1][None, :]) / denom_l[None, :] * B[:, :-1]
        right = (grid[k+1:][None, :] - x_exp) / denom_r[None, :] * B[:, 1:]

        B = left + right

    return B  # (batch, n_basis)


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer.

    Each edge (i → j) has a learnable B-spline activation.  The layer also
    adds a SiLU residual path for same-dimension inputs.

    Parameters
    ----------
    in_features : int
    out_features : int
    n_grid : int
        Number of B-spline grid intervals (default 5).
    order : int
        B-spline polynomial order (default 3).
    grid_lo : float
        Lower bound of the B-spline domain (default -3.0).
    grid_hi : float
        Upper bound of the B-spline domain (default 3.0).
    """

    in_features: int
    out_features: int
    n_grid: int = 5
    order: int = 3
    grid_lo: float = -3.0
    grid_hi: float = 3.0

    @nn.compact
    def __call__(self, x):
        """
        Apply the KAN layer.

        Parameters
        ----------
        x : array, shape (batch, in_features)

        Returns
        -------
        array, shape (batch, out_features)
        """
        batch = x.shape[0]
        n_basis = self.n_grid + self.order  # number of B-spline basis functions

        # Learnable spline coefficients: (in_features, out_features, n_basis)
        C = self.param(
            "spline_coeff",
            nn.initializers.truncated_normal(stddev=0.1),
            (self.in_features, self.out_features, n_basis),
        )

        # Build extended knot vector (clamped at boundaries)
        interior = jnp.linspace(self.grid_lo, self.grid_hi, self.n_grid + 1)
        left_ext = jnp.full((self.order,), self.grid_lo)
        right_ext = jnp.full((self.order,), self.grid_hi)
        grid = jnp.concatenate([left_ext, interior, right_ext])  # length = n_grid+1+2*order

        # Compute basis values for each input feature
        # x[:, i] → B_i: (batch, n_basis)
        def basis_for_feature(xi):
            """Evaluate B-spline basis for one input feature across the batch."""
            return _bspline_basis(xi, grid, self.order)           # (batch, n_basis)

        # vmap over input features
        # x transposed: (in_features, batch)
        Bs = jax.vmap(basis_for_feature)(x.T)                    # (in_features, batch, n_basis)
        Bs = jnp.transpose(Bs, (1, 0, 2))                        # (batch, in_features, n_basis)

        # For each edge (i→j): psi_ij(x_i) = sum_k C[i,j,k] * B_i[k]
        # output[b,j] = sum_i psi_ij(x[b,i])
        # Bs: (batch, in, n_basis), C: (in, out, n_basis)
        # → (batch, in, out) = einsum("bin,ion->bio", Bs, C)
        # then sum over i → (batch, out)
        y = jnp.einsum("bin,ion->bo", Bs, C)                     # (batch, out)

        # Residual SiLU path for same-dimension layers
        if self.in_features == self.out_features:
            w_res = self.param(
                "w_residual",
                nn.initializers.ones,
                (self.out_features,),
            )
            y = y + w_res * nn.silu(x)

        return y


class KANVAE(MolecularAutoencoder):
    """
    Kolmogorov-Arnold Network VAE.

    Encoder: KANLayer stack → Dense heads for z_mean and z_logvar.
    Decoder: KANLayer stack (reversed widths) → output coordinates.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        Hidden layer widths; length = number of KAN layers.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Accepted for API compatibility; unused by KAN layers.
    is_batchnorm : bool
        Accepted for API compatibility; unused.
    kan_n_grid : int
        Number of B-spline intervals per edge (default 5).
    kan_order : int
        B-spline polynomial order (default 3).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    kan_n_grid: int = 5
    kan_order: int = 3

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Compute hidden layer widths from experiment config.

        Parameters
        ----------
        input_size : int
        n_latents : int
        dropout_rates : list of float
            Length = number of KAN layers.
        json_params : dict
            Reads ``embed_dim`` (default min(64, input_size)),
            ``kan_n_grid`` (default 5), ``kan_order`` (default 3).

        Returns
        -------
        list of int
            [embed_dim] * n_layers.
        """
        embed_dim = json_params.get("embed_dim", min(64, input_size))
        n_layers = len(dropout_rates)
        return [embed_dim] * n_layers

    def setup(self):
        """Wire KAN encoder, KAN decoder, and latent heads."""
        d = list(self.hidden_layers)
        n_grid = self.kan_n_grid
        order = self.kan_order

        # Encoder: input_size → d[0] → d[1] → ... → d[-1]
        enc_dims = [self.input_size] + d
        self._enc_layers = [
            KANLayer(enc_dims[i], enc_dims[i + 1], n_grid=n_grid, order=order)
            for i in range(len(d))
        ]
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        # Decoder: latents → d[-1] → d[-2] → ... → d[0] → input_size
        dec_dims = [self.latents] + list(reversed(d)) + [self.input_size]
        self._dec_layers = [
            KANLayer(dec_dims[i], dec_dims[i + 1], n_grid=n_grid, order=order)
            for i in range(len(d) + 1)
        ]

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Encode flat coordinates through KAN layers.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of arrays, each shape (batch, latents)
        """
        h = x
        for layer in self._enc_layers:
            h = layer(h)
        return self._z_mean(h), self._z_logvar(h)

    def decode(self, z, z_rng=None, train: bool = False):
        """
        Decode latent vector through reversed KAN layers.

        Parameters
        ----------
        z : array, shape (batch, latents)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        h = z
        for layer in self._dec_layers:
            h = layer(h)
        return h

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
