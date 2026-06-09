"""
Mamba-inspired selective State-Space Model (SSM) VAE for molecular coordinates.

Treats each atom as a sequence token and processes them with simplified Mamba
blocks implemented in pure JAX (no hardware-specific kernel tricks).  The
selective SSM uses input-dependent timescales (Δ) so each token can modulate
how much it retains from the past.

The associative scan parallelises the recurrence:
    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
using the monoid operator (A1, B1) ⊕ (A2, B2) = (A1*A2, A1*B2 + B1).

JSON config extras
------------------
embed_dim    : int   – atom embedding width (default min(128, input_size))
d_state      : int   – SSM state dimension (default 16)
d_inner_mult : float – d_inner = round(d_inner_mult * embed_dim) (default 2.0)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from .base import MolecularAutoencoder
from .vae import reparameterize


def _ssm_scan(A_bar, B_bar, x_seq):
    """
    Parallel selective SSM scan via jax.lax.associative_scan.

    Solves h_t = A_bar_t * h_{t-1} + B_bar_t * x_t with h_0 = 0.

    Parameters
    ----------
    A_bar : array, shape (batch, seq_len, d_state)
        Discretised transition matrices (diagonal, so stored as vector).
    B_bar : array, shape (batch, seq_len, d_state)
        Discretised input matrices.
    x_seq : array, shape (batch, seq_len, d_inner)
        Input sequence (projected).

    Returns
    -------
    h : array, shape (batch, seq_len, d_state)
        Hidden state at every time step.
    """
    # Lift to (A, B*x) pairs and scan with the associative operator.
    # State element: (a, b) where h = b after combining from left.
    bx = B_bar * x_seq[..., :B_bar.shape[-1]]  # (B, L, d_state)

    def combine(left, right):
        """Associative operator for the linear SSM parallel scan: (A1,b1) ⊕ (A2,b2) = (A1·A2, A1·b2+b1)."""
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, a_l * b_r + b_l

    # Scan over sequence dimension (axis=1)
    # Input elements: each is (A_bar[:, t, :], bx[:, t, :])
    # We need to transpose so axis=0 is the scan axis
    A_t = jnp.transpose(A_bar, (1, 0, 2))  # (L, B, d_state)
    B_t = jnp.transpose(bx, (1, 0, 2))     # (L, B, d_state)

    _, h_t = jax.lax.associative_scan(combine, (A_t, B_t), axis=0)
    h = jnp.transpose(h_t, (1, 0, 2))      # (B, L, d_state)
    return h


class _MambaBlock(nn.Module):
    """
    Simplified Mamba block with selective SSM.

    Input is projected to two streams: one goes through the SSM,
    the other acts as a gating signal.  The output is mixed back to d_model.

    Attributes
    ----------
    d_model : int
        Input/output token dimension.
    d_inner : int
        Inner (expanded) dimension.
    d_state : int
        SSM state dimension.
    dropout_rate : float
        Dropout rate applied to the output.
    """

    d_model: int
    d_inner: int
    d_state: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool):
        """
        Apply the Mamba block to a sequence.

        Parameters
        ----------
        x : array, shape (batch, seq_len, d_model)
        train : bool

        Returns
        -------
        array, same shape as x
        """
        residual = x
        x = nn.LayerNorm()(x)

        # Input projection: split into x_proj and gate
        xz = nn.Dense(2 * self.d_inner)(x)                   # (B, L, 2*d_inner)
        x_in, gate = jnp.split(xz, 2, axis=-1)               # each (B, L, d_inner)
        gate = nn.silu(gate)

        # Selective SSM parameters (input-dependent)
        delta = nn.softplus(nn.Dense(self.d_state)(x_in))     # (B, L, d_state)
        # Fixed A: diagonal state matrix initialised as -1 * I
        A = self.param(
            "A",
            lambda rng, shape: -jnp.ones(shape),
            (self.d_state,),
        )
        B = nn.Dense(self.d_state)(x_in)                      # (B, L, d_state)
        C = nn.Dense(self.d_state)(x_in)                      # (B, L, d_state)

        # Discretise: A_bar = exp(delta * A), B_bar = delta * B
        A_bar = jnp.exp(delta * A[None, None, :])             # (B, L, d_state)
        B_bar = delta * B                                     # (B, L, d_state)

        # Run SSM
        h = _ssm_scan(A_bar, B_bar, x_in)                    # (B, L, d_state)

        # Output via C gating
        y = jnp.sum(C * h, axis=-1, keepdims=True)           # (B, L, 1)
        y = x_in * y                                         # (B, L, d_inner) — element scale

        # Apply gate
        y = y * gate

        # Project back to d_model
        y = nn.Dense(self.d_model)(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
        return residual + y


class MambaVAE(MolecularAutoencoder):
    """
    Mamba-inspired selective SSM variational autoencoder.

    Treats atoms as sequence tokens; each atom's coordinates become an input
    token of dimension 3 → embed_dim.  N Mamba blocks refine the sequence.
    Mean-pooling + linear projection yields z_mean and z_logvar.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimensionality (n_atoms * 3).
    hidden_layers : tuple of int
        All elements equal embed_dim; length = number of Mamba blocks.
    latents : int
        Number of latent dimensions.
    dropout_rates : list of float
        Per-block dropout rates.
    is_batchnorm : bool
        Accepted for API compatibility; unused (LayerNorm is used).
    d_state : int
        SSM state dimension (default 16).
    d_inner_mult : float
        d_inner = round(d_inner_mult * embed_dim) (default 2.0).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    d_state: int = 16
    d_inner_mult: float = 2.0

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Compute hidden layer widths from experiment config.

        Parameters
        ----------
        input_size : int
        n_latents : int
        dropout_rates : list of float
            Length determines the number of Mamba blocks.
        json_params : dict
            Reads ``embed_dim`` (default min(128, input_size)), ``d_state``
            (default 16), ``d_inner_mult`` (default 2.0).

        Returns
        -------
        list of int
            [embed_dim] * n_blocks.
        """
        embed_dim = json_params.get("embed_dim", min(128, input_size))
        return [embed_dim] * len(dropout_rates)

    def setup(self):
        """Wire atom embedding, Mamba blocks, and latent heads."""
        self._n_atoms = self.input_size // 3
        embed_dim = self.hidden_layers[0] if self.hidden_layers else min(128, self.input_size)
        n_blocks = len(self.hidden_layers)
        d_inner = max(1, round(self.d_inner_mult * embed_dim))

        self._atom_embed = nn.Dense(embed_dim)
        self._enc_blocks = [
            _MambaBlock(embed_dim, d_inner, self.d_state,
                        self.dropout_rates[i] if i < len(self.dropout_rates) else 0.0)
            for i in range(n_blocks)
        ]
        self._enc_norm = nn.LayerNorm()
        self._z_mean = nn.Dense(self.latents)
        self._z_logvar = nn.Dense(self.latents)

        # Decoder: z → tile → Mamba blocks → Dense(3) per atom
        self._latent_proj = nn.Dense(embed_dim)
        self._dec_blocks = [
            _MambaBlock(embed_dim, d_inner, self.d_state,
                        self.dropout_rates[i] if i < len(self.dropout_rates) else 0.0)
            for i in range(n_blocks)
        ]
        self._dec_norm = nn.LayerNorm()
        self._coord_proj = nn.Dense(3)

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Encode flat coordinates via Mamba sequence model.

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
        batch = z.shape[0]
        h = self._latent_proj(z)                              # (B, E)
        h = jnp.tile(h[:, None, :], (1, self._n_atoms, 1))   # (B, N, E)
        for blk in self._dec_blocks:
            h = blk(h, train=train)
        h = self._dec_norm(h)
        out = self._coord_proj(h)                             # (B, N, 3)
        return out.reshape(batch, self.input_size)

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
