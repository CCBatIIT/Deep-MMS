"""
RealNVP normalising flow for molecular coordinates.

For normalising flows the ``input_dim == latent_dim`` — there is no
bottleneck compression.  The ``latents`` parameter controls how many of
the leading dimensions of the flow output are used for downstream analysis.

Each affine coupling layer splits the input at dim//2:
    y1 = x1,  y2 = x2 * exp(s(x1)) + t(x1)
Alternate which half is transformed each layer.
Actnorm (per-channel scale/shift initialised from first-batch statistics)
is inserted between coupling layers for stable training.

The model is trained by minimising the negative log-likelihood:
    NLL = 0.5 * sum(z^2) - log_det
using FlowTrainer.

JSON config extras
------------------
n_coupling_layers : int – number of coupling layers (default 8).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .base import MolecularAutoencoder


class _ActNorm(nn.Module):
    """
    Activation normalisation layer (actnorm).

    Scale and shift parameters are stored as learnable parameters (not
    initialised from data to keep Flax compatibility; we just initialise
    them to ones/zeros).

    Attributes
    ----------
    dim : int
        Feature dimension.
    """

    dim: int

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        """
        Apply actnorm forward or inverse.

        Parameters
        ----------
        x : array, shape (batch, dim)
        reverse : bool
            If True, apply the inverse transform.

        Returns
        -------
        y : array, shape (batch, dim)
        log_det : float
            Log determinant contribution (positive for forward).
        """
        log_scale = self.param("log_scale", nn.initializers.zeros, (self.dim,))
        shift = self.param("shift", nn.initializers.zeros, (self.dim,))
        if not reverse:
            y = (x + shift) * jnp.exp(log_scale)
            log_det = jnp.sum(log_scale)
        else:
            y = x * jnp.exp(-log_scale) - shift
            log_det = -jnp.sum(log_scale)
        return y, log_det


class _CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP.

    Splits input into two halves: the first half is unchanged and is used
    to compute scale and shift for the second half.

    Attributes
    ----------
    split_dim : int
        Number of features in the first (fixed) half.
    out_dim : int
        Total output dimension.
    hidden_dim : int
        Width of the scale/shift MLP.
    """

    split_dim: int
    out_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        """
        Apply coupling transform.

        Parameters
        ----------
        x : array, shape (batch, out_dim)
        reverse : bool

        Returns
        -------
        y : array, shape (batch, out_dim)
        log_det : float
        """
        x1 = x[:, :self.split_dim]
        x2 = x[:, self.split_dim:]

        # Scale/shift network operating on x1
        h = nn.Dense(self.hidden_dim)(x1)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        st = nn.Dense((self.out_dim - self.split_dim) * 2)(h)
        s, t = jnp.split(st, 2, axis=-1)
        s = jnp.tanh(s)                                # bounded scale

        if not reverse:
            y2 = x2 * jnp.exp(s) + t
            log_det = jnp.sum(s, axis=-1).mean()
        else:
            y2 = (x2 - t) * jnp.exp(-s)
            log_det = -jnp.sum(s, axis=-1).mean()

        y = jnp.concatenate([x1, y2], axis=-1)
        return y, log_det


class RealNVPFlow(MolecularAutoencoder):
    """
    RealNVP normalising flow for molecular coordinates.

    No compression — input_size == flow dimension.  The ``latents`` field
    controls how many dimensions to use for analysis (the leading dims of z).

    Coupling layers alternate which half of the input they transform.
    Actnorm is inserted between coupling layers.

    Attributes
    ----------
    input_size : int
        Flow dimension (equals n_atoms * 3).
    hidden_layers : tuple of int
        All elements equal input_size // 2; length = n_coupling_layers.
    latents : int
        Number of leading dimensions used for downstream analysis.
    dropout_rates : list of float
        Accepted for API compatibility; unused by flows.
    is_batchnorm : bool
        Accepted for API compatibility; unused.
    n_coupling_layers : int
        Number of affine coupling layers; set from hidden_layers length.
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
        n_latents : int
        dropout_rates : list of float
            Ignored; n_coupling_layers is used instead.
        json_params : dict
            Reads ``n_coupling_layers`` (default 8).

        Returns
        -------
        list of int
            [input_size // 2] * n_coupling_layers (MLP width per coupling layer).
        """
        n_cl = json_params.get("n_coupling_layers", 8)
        return [max(1, input_size // 2)] * n_cl

    def setup(self):
        """Wire actnorm and coupling layers."""
        n_layers = len(self.hidden_layers)
        hidden_dim = self.hidden_layers[0] if self.hidden_layers else max(1, self.input_size // 2)
        split = self.input_size // 2

        self._coupling = [
            _CouplingLayer(
                split_dim=split if i % 2 == 0 else (self.input_size - split),
                out_dim=self.input_size,
                hidden_dim=hidden_dim,
            )
            for i in range(n_layers)
        ]
        self._actnorm = [_ActNorm(self.input_size) for _ in range(n_layers)]

    def _forward(self, x):
        """
        Forward pass through all coupling and actnorm layers.

        Parameters
        ----------
        x : array, shape (batch, input_size)

        Returns
        -------
        z : array, shape (batch, input_size)
        log_det : float
            Total log determinant.
        """
        log_det = 0.0
        z = x
        split = self.input_size // 2
        for i, (coup, anorm) in enumerate(zip(self._coupling, self._actnorm)):
            # Alternate split: even layers fix first half, odd layers fix second half
            if i % 2 == 1:
                z = jnp.concatenate([z[:, split:], z[:, :split]], axis=-1)
            z, ld1 = coup(z)
            log_det = log_det + ld1
            z, ld2 = anorm(z)
            log_det = log_det + ld2
            if i % 2 == 1:
                z = jnp.concatenate([z[:, (self.input_size - split):], z[:, :(self.input_size - split)]], axis=-1)
        return z, log_det

    def _inverse(self, z):
        """
        Inverse pass (flow decoding).

        Parameters
        ----------
        z : array, shape (batch, input_size)

        Returns
        -------
        x : array, shape (batch, input_size)
        """
        split = self.input_size // 2
        x = z
        for i, (coup, anorm) in reversed(list(enumerate(zip(self._coupling, self._actnorm)))):
            if i % 2 == 1:
                x = jnp.concatenate([x[:, split:], x[:, :split]], axis=-1)
            x, _ = anorm(x, reverse=True)
            x, _ = coup(x, reverse=True)
            if i % 2 == 1:
                x = jnp.concatenate([x[:, (self.input_size - split):], x[:, :(self.input_size - split)]], axis=-1)
        return x

    def encode(self, x, z_rng=None, train: bool = False):
        """
        Map input through the flow to latent space.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey, optional
        train : bool

        Returns
        -------
        tuple of (z_mean, zeros)
            z_mean contains the first ``latents`` dims of the flow output.
        """
        z, _ = self._forward(x)
        z_mean = z[:, :self.latents]
        return z_mean, jnp.zeros_like(z_mean)

    def decode(self, z, z_rng=None, train: bool = False):
        """
        Invert the flow from latent space to coordinate space.

        Parameters
        ----------
        z : array, shape (batch, latents or input_size)
            If shorter than input_size, padded with zeros.
        z_rng : jax.random.PRNGKey, optional
            Used to sample noise for padding when z is shorter than input_size.
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        if z.shape[-1] < self.input_size:
            pad_size = self.input_size - z.shape[-1]
            if z_rng is not None:
                pad = jax.random.normal(z_rng, (z.shape[0], pad_size))
            else:
                pad = jnp.zeros((z.shape[0], pad_size))
            z_full = jnp.concatenate([z, pad], axis=-1)
        else:
            z_full = z
        return self._inverse(z_full)

    def __call__(self, x, z_rng, train: bool):
        """
        Full forward pass: encode → inverse → (x_recon, z, zeros).

        x_recon should equal x exactly up to floating-point precision
        since flow is bijective.

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        tuple of (x_recon, z, zeros)
        """
        z, _ = self._forward(x)
        x_recon = self._inverse(z)
        return x_recon, z, jnp.zeros_like(z)

    def construct(self, z_mean, z_logvar, z_rng, train: bool = False):
        """
        Invert from leading latents, padding remainder with zeros.

        Parameters
        ----------
        z_mean : array, shape (batch, latents)
        z_logvar : array, shape (batch, latents)
            Ignored for flows.
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        array, shape (batch, input_size)
        """
        return self.decode(z_mean, z_rng=z_rng, train=train)

    def aux_loss(self, x, z_rng, train: bool = False):
        """
        Negative log-likelihood loss for normalising flow training.

        NLL = 0.5 * mean(||z||^2) - log_det

        Parameters
        ----------
        x : array, shape (batch, input_size)
        z_rng : jax.random.PRNGKey
        train : bool

        Returns
        -------
        float
            Scalar NLL.
        """
        z, log_det = self._forward(x)
        nll = 0.5 * jnp.mean(jnp.sum(z ** 2, axis=-1)) - log_det
        return nll
