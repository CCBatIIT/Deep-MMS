"""
NEAT-inspired variational autoencoder for molecular coordinates.

NEAT (NeuroEvolution of Augmenting Topologies) begins with minimal networks
and grows them by adding neurons and connections only when complexity is
justified by improved fitness.  Three JAX-compatible design choices capture
that spirit here:

  1. Topology starts small and grows.  hidden_layers_from_config returns a
     compact initial topology; NEATTrainer adds layers at loss plateaus.

  2. tanh activations throughout (NEAT's original default; avoids dying-ReLU
     problems in evolving sparse networks).

  3. No batch normalisation — the evolutionary pressure and small initial size
     provide implicit regularisation.

The grow() module-level function creates a new, deeper model and transfers
weights from a smaller predecessor, initialising new-layer weights near zero
so the network's initial behaviour is preserved (NEAT's "minimal structural
innovation" principle).

JSON config extras (all optional):
    neat_start_dim    : int – hidden layer width at initialisation (default: 64)
    neat_start_layers : int – number of hidden layers at initialisation (default: 2)
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Any

from .base import MolecularAutoencoder
from .vae import reparameterize


class _NEATEncoder(nn.Module):
    """Tanh MLP encoder projecting coordinates to (z_mean, z_logvar)."""

    d_hidden: list
    latents: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        for width in self.d_hidden:
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
        mean_x = nn.Dense(self.latents, name="z_mean")(x)
        logvar_x = nn.Dense(self.latents, name="z_logvar")(x)
        return mean_x, logvar_x


class _NEATDecoder(nn.Module):
    """Tanh MLP decoder projecting latent to reconstructed coordinates."""

    d_hidden: list
    out_dim: int

    @nn.compact
    def __call__(self, z, train: bool = False):
        for width in reversed(self.d_hidden):
            z = nn.Dense(width)(z)
            z = nn.tanh(z)
        return nn.Dense(self.out_dim, name="output")(z)


class NEATAutoencoder(MolecularAutoencoder):
    """
    NEAT-inspired growing MLP variational autoencoder.

    Uses tanh activations and no batch normalisation.  The initial topology is
    intentionally small; the NEATTrainer drives topology growth automatically.
    Fully compatible with the Experiment training harness and gradient-based
    Adam optimisation.

    Attributes
    ----------
    input_size : int
        Flattened coordinate dimension (n_atoms * 3).
    hidden_layers : tuple of int
        Hidden layer widths.  Unlike BatchNorm_VAE, widths may differ per layer.
    latents : int
        Latent dimensionality.
    dropout_rates : list of float
        Accepted for API compatibility; NEAT uses no dropout (set to 0.0).
    is_batchnorm : bool
        Accepted for API compatibility; always ignored (no batch norm in NEAT).
    """

    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        Return a compact initial topology for NEAT evolution.

        Reads 'neat_start_dim' (default min(64, input_size)) and
        'neat_start_layers' (default len(dropout_rates)) from json_params.
        Intentionally smaller than the BatchNorm_VAE default so the network
        must earn additional capacity through topology growth.
        """
        start_dim = json_params.get("neat_start_dim", min(64, input_size))
        n_layers = json_params.get("neat_start_layers", len(dropout_rates))
        return [start_dim] * n_layers

    def setup(self):
        """Wire encoder and decoder tanh MLP sub-modules."""
        self._encoder = _NEATEncoder(list(self.hidden_layers), self.latents)
        self._decoder = _NEATDecoder(list(self.hidden_layers), self.input_size)

    def encode(self, x, z_rng=None, train: bool = False):
        """Return (z_mean, z_logvar) for input coordinates x."""
        return self._encoder(x, train=train)

    def decode(self, z, z_rng=None, train: bool = False):
        """Decode latent z to reconstructed coordinates."""
        return self._decoder(z, train=train)

    def __call__(self, x, z_rng, train: bool):
        """Full forward pass: encode → reparameterise → decode."""
        z_mean, z_logvar = self.encode(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, train=train), z_mean, z_logvar

    def construct(self, z_mean, z_logvar, z_rng, train=False):
        """Sample from posterior and decode without re-encoding."""
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, train=train)


def grow(model, old_params, new_width=None):
    """
    Return a new NEATAutoencoder with one additional hidden layer and transfer
    all existing weights, initialising the new layer near zero.

    The new layer is inserted at the end of the encoder (and beginning of the
    decoder, mirroring the symmetric encoder–decoder design).  Its weights are
    initialised to N(0, 0.01) so the network initially behaves close to its
    predecessor.

    Parameters
    ----------
    model : NEATAutoencoder
        The current, smaller model.
    old_params : dict
        Current Flax parameter tree (state.params).
    new_width : int or None
        Width of the new hidden layer.  Defaults to the last hidden layer width.

    Returns
    -------
    new_model : NEATAutoencoder
        Larger model with n_layers + 1 hidden layers.
    new_params : dict
        Parameter tree with old weights copied and new-layer weights near zero.
    new_dropout_rates : list of float
        Dropout rates list extended by one entry (0.0 for the new layer).
    """
    old_hidden = list(model.hidden_layers)
    if new_width is None:
        new_width = old_hidden[-1]
    new_hidden = old_hidden + [new_width]

    new_model = NEATAutoencoder(
        input_size=model.input_size,
        hidden_layers=tuple(new_hidden),
        latents=model.latents,
        dropout_rates=model.dropout_rates + [0.0],
        is_batchnorm=False,
    )

    # Build new parameter tree by copying existing Dense layers.
    # Flax names encoder layers Dense_0, Dense_1, ... Dense_n-1, then z_mean/z_logvar.
    # A new layer is appended; its kernel/bias start near zero.
    import copy
    new_params = copy.deepcopy(old_params)
    n_old = len(old_hidden)

    rng = jax.random.PRNGKey(0)

    for side in ["_encoder", "_decoder"]:
        # Find the existing final hidden Dense index and append a new one
        last_key = f"Dense_{n_old - 1}"
        last_width = old_hidden[-1]
        rng, k1, k2 = jax.random.split(rng, 3)
        new_params[side][f"Dense_{n_old}"] = {
            "kernel": jax.random.normal(k1, (last_width, new_width)) * 0.01,
            "bias": jax.random.normal(k2, (new_width,)) * 0.01,
        }

    new_dropout_rates = list(model.dropout_rates) + [0.0]
    return new_model, new_params, new_dropout_rates
