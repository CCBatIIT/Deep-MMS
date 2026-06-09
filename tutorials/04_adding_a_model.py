"""
Tutorial 4: Adding a Custom Architecture
==========================================
Run from the project root:
    python tutorials/04_adding_a_model.py

This tutorial walks through implementing a new architecture end-to-end:

  1. Subclass MolecularAutoencoder to create LinearAutoencoder — a minimal
     two-layer model with no hidden layers and no activation functions.
  2. Override hidden_layers_from_config to signal "no hidden layers".
  3. Train it for 10 epochs using the same low-level API as Tutorial 1.
  4. Compare its reconstruction RMSD to BatchNorm_VAE on the same data.

LinearAutoencoder is intentionally simple (encoder: D→K Dense, decoder: K→D
Dense) so you can see the full template with no distractions.  Real custom
architectures follow exactly this same pattern.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import deepmms.utils  # side-effect: sets jax_enable_x64

import flax.linen as nn
from deepmms.models.base import MolecularAutoencoder
from deepmms.models.vae import BatchNorm_VAE, reparameterize
from deepmms.training.loss import give_weighted_rmsd_func, atom_rmsd

# =============================================================================
# Step 1: Define LinearAutoencoder
# =============================================================================
# MolecularAutoencoder is the abstract base class that all Deep-MMS models
# must implement.  The three required methods are encode, decode, and __call__.
# The interface must also expose a construct() method (for perturbation analysis).

class LinearAutoencoder(MolecularAutoencoder):
    """
    Minimal linear autoencoder: no hidden layers, no activation functions.

    Encoder: Dense(D → K)
    Decoder: Dense(K → D)

    This is the simplest possible MolecularAutoencoder subclass and serves as
    a template for adding custom architectures to Deep-MMS.
    """

    # Flax module attributes — declared as class-level annotations
    input_size: int       # D = n_atoms * 3
    latents: int          # K
    hidden_layers: tuple  # ignored, but kept for interface compatibility
    dropout_rates: list   # ignored, but kept for interface compatibility
    is_batchnorm: bool    # ignored, but kept for interface compatibility

    @classmethod
    def hidden_layers_from_config(cls, input_size, n_latents, dropout_rates, json_params):
        """
        LinearAutoencoder has no hidden layers.

        We still return a list of length len(dropout_rates) so that the
        training harness does not complain about an empty list, but we never
        use these widths inside the model.
        """
        # Return a dummy list with the right length.  The actual model ignores it.
        return [1] * len(dropout_rates)

    def setup(self):
        """Wire the two Dense layers."""
        # Encoder projects D → K; decoder projects K → D.
        # We produce both a mean and a log-variance from the encoder,
        # just like BatchNorm_VAE, so the interface is identical.
        self._enc_mean   = nn.Dense(self.latents, name="enc_mean")
        self._enc_logvar = nn.Dense(self.latents, name="enc_logvar")
        self._dec        = nn.Dense(self.input_size, name="decoder")

    def encode(self, x, z_rng, train: bool = False):
        """Encode: project input directly to (z_mean, z_logvar)."""
        return self._enc_mean(x), self._enc_logvar(x)

    def decode(self, z, z_rng, train: bool = False):
        """Decode: project latent vector directly to coordinate space."""
        return self._dec(z)

    def __call__(self, x, z_rng, train: bool = False):
        """Full forward pass with reparameterisation."""
        z_mean, z_logvar = self.encode(x, z_rng, train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        recon = self.decode(z, z_rng, train)
        return recon, z_mean, z_logvar

    def construct(self, z_mean, z_logvar, z_rng, train=False):
        """Sample from posterior and decode (used by perturbation analysis)."""
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decode(z, z_rng, train)


# =============================================================================
# Step 2: Register in models/__init__.py? (Not required for use)
# =============================================================================
# You *can* add LinearAutoencoder to deepmms/models/__init__.py for convenient
# import, but it is not required.  The model can be used directly:
#
#   from tutorials.04_adding_a_model import LinearAutoencoder
#   exp = Experiment(json_params, from_json_params=True, model_cls=LinearAutoencoder)

# =============================================================================
# Shared setup: synthetic data, weights, train/eval helpers
# =============================================================================
N_FRAMES = 200
N_ATOMS  = 10
INPUT_SIZE = N_ATOMS * 3   # 30
LATENTS  = 4
DROPOUT  = [0.0, 0.0]

key = jax.random.PRNGKey(0)
key, dk = jax.random.split(key)
data = jax.random.normal(dk, (N_FRAMES, INPUT_SIZE))
train_data = data[:160]
test_data  = data[160:]

weights = jnp.ones(N_ATOMS)
atom_rmsd_fn = give_weighted_rmsd_func(weights)


def build_and_train(model_cls, n_epochs=10, lr=1e-3, extra_kwargs=None, seed=0):
    """
    Instantiate model_cls, train for n_epochs, and return (model, state, rmsd).

    This is the same low-level pattern shown in Tutorial 1, but now it accepts
    any MolecularAutoencoder subclass — showing that the interface is truly
    drop-in.
    """
    extra_kwargs = extra_kwargs or {}

    # Use the class-specific layer-sizing logic
    hidden = model_cls.hidden_layers_from_config(INPUT_SIZE, LATENTS, DROPOUT, {})
    model = model_cls(
        input_size=INPUT_SIZE, latents=LATENTS,
        hidden_layers=tuple(hidden), dropout_rates=DROPOUT,
        is_batchnorm=False, **extra_kwargs,
    )

    init_key = jax.random.PRNGKey(seed)
    params = model.init(init_key, train_data[:4], init_key, train=False)

    class TS(train_state.TrainState):
        key: jax.Array

    state = TS.create(
        apply_fn=model.apply,
        params=params["params"],
        key=init_key,
        tx=optax.adam(lr),
    )

    @jax.jit
    def step(state, batch):
        z_rng = jax.random.PRNGKey(state.step)
        drop_key = jax.random.fold_in(state.key, state.step)

        def loss_fn(p):
            recon, zm, zlv = model.apply(
                {"params": p}, batch, z_rng, train=True,
                rngs={"dropout": drop_key},
            )
            return jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_fn(batch, recon) ** 2)))

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    batch_size = 32
    n_batches  = len(train_data) // batch_size
    for epoch in range(n_epochs):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(train_data))
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            state = step(state, train_data[idx])

    # Evaluate final test RMSD
    z_rng = jax.random.PRNGKey(99)
    recon, _, _ = model.apply({"params": state.params}, test_data, z_rng, train=False)
    rmsd = float(jnp.mean(atom_rmsd(test_data, recon)))
    return model, state, rmsd


# =============================================================================
# Step 3: Train LinearAutoencoder and compare to BatchNorm_VAE
# =============================================================================
print("Training LinearAutoencoder (10 epochs) ...")
_, _, rmsd_linear = build_and_train(LinearAutoencoder, n_epochs=10)

print("Training BatchNorm_VAE    (10 epochs) ...")
_, _, rmsd_vae    = build_and_train(BatchNorm_VAE,    n_epochs=10)

print()
print("Results after 10 epochs:")
print(f"  LinearAutoencoder  test RMSD = {rmsd_linear * 10:.4f} A")
print(f"  BatchNorm_VAE      test RMSD = {rmsd_vae    * 10:.4f} A")
print()
if rmsd_linear <= rmsd_vae:
    print("LinearAutoencoder matches or beats BatchNorm_VAE on this tiny synthetic problem.")
else:
    ratio = rmsd_linear / rmsd_vae
    print(f"BatchNorm_VAE is {ratio:.2f}x better — expected: hidden layers help on real data.")

# =============================================================================
# Step 4: Show the Experiment usage pattern (with real files)
# =============================================================================
print("""
How to use LinearAutoencoder with the full Experiment harness
(requires a real DCD + topology file):

    from deepmms.training.trainer import Experiment
    from tutorials.four_adding_a_model import LinearAutoencoder

    json_params = {
        "fname_dcd": "Simulation/my_traj.dcd",
        "fname_topology": "Simulation/my_traj.pdb",
        "save_dir": "/tmp",
        "model_name": "LINEAR_TEST",
        "latent_dim": 4,
        "test_slice": 1,
        "data_slice_start": 0,
        "data_slice_end": "None",
        "batch_size": 64,
        "learning_rate": 1e-3,
        "dropout_rates": [0.0, 0.0],
        "resume_latest": False,
        "checkpoint_interval": 100,
        "max_epoch": 1000,
        "is_batchnorm": False,
        "atom_selection": "not element H",
        "weight_model": "Uniform_Heavy",
    }

    exp = Experiment(json_params, from_json_params=True, model_cls=LinearAutoencoder)
    exp.MAIN_train(n_epochs=200)
""")

print("Done. The four tutorials are complete.")
