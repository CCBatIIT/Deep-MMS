"""
Tutorial 1: Training Your First Model
======================================
Run from the project root:
    python tutorials/01_quickstart.py

This tutorial walks through the lowest-level API: constructing a model,
building a TrainState, running gradient steps by hand, and inspecting
the latent space — all without loading any real trajectory files.

We treat a synthetic (200, 30) dataset as a 10-atom molecule where each
frame is a flattened coordinate vector of shape (10 atoms * 3 dims = 30).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any

# ── Enable 64-bit floats (required by Deep-MMS) ──────────────────────────────
import deepmms.utils  # side-effect: sets jax_enable_x64

from deepmms.models.vae import BatchNorm_VAE, reparameterize
from deepmms.training.loss import give_weighted_rmsd_func, atom_rmsd
from deepmms.training.optimizer import create_warmup_cosine_schedule

# =============================================================================
# 1.  The JSON config format
# =============================================================================
# Every Deep-MMS run is driven by a Python dict (or JSON file) with these keys.
# You never need to touch the model code — just change the params.

json_params = {
    "latent_dim": 4,          # compress to 4 latent dimensions
    "dropout_rates": [0.0, 0.0],  # 2 hidden layers, no dropout for this demo
    "is_batchnorm": False,    # skip BatchNorm for clarity
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epoch": 500,
}

# =============================================================================
# 2.  Synthetic data: 200 frames, 10 atoms, 30 features
# =============================================================================
key = jax.random.PRNGKey(0)
key, data_key = jax.random.split(key)

N_FRAMES = 200
N_ATOMS = 10
INPUT_SIZE = N_ATOMS * 3   # 30

# Random Gaussian coordinates simulate a small molecule ensemble
data = jax.random.normal(data_key, (N_FRAMES, INPUT_SIZE))
train_data = data[:160]
test_data = data[160:]

print(f"Dataset: {data.shape}  (frames={N_FRAMES}, atoms={N_ATOMS}, features={INPUT_SIZE})")

# =============================================================================
# 3.  Instantiate BatchNorm_VAE using the low-level factory
# =============================================================================
# hidden_layers_from_config returns one square layer per dropout rate entry.
dropout_rates = json_params["dropout_rates"]
hidden_layers = BatchNorm_VAE.hidden_layers_from_config(
    INPUT_SIZE, json_params["latent_dim"], dropout_rates, json_params
)
print(f"\nHidden layer widths: {hidden_layers}  (square = input_size={INPUT_SIZE} each)")

model = BatchNorm_VAE(
    input_size=INPUT_SIZE,
    latents=json_params["latent_dim"],
    hidden_layers=tuple(hidden_layers),
    dropout_rates=dropout_rates,
    is_batchnorm=json_params["is_batchnorm"],
)

# Initialise parameters with a dummy forward pass
key, init_key, rng_key = jax.random.split(key, 3)
params = model.init(init_key, train_data[:4], rng_key, train=False)
print(f"Parameter dict keys: {list(params.keys())}")

# =============================================================================
# 4.  LR schedule — warmup then cosine decay
# =============================================================================
n_batches_per_epoch = len(train_data) // json_params["batch_size"]
total_steps = n_batches_per_epoch * json_params["max_epoch"]
warmup_steps = n_batches_per_epoch * 100  # 100-epoch warmup

lr_schedule = create_warmup_cosine_schedule(
    base_lr=json_params["learning_rate"],
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    final_lr=json_params["learning_rate"] / 100,
)

# Show what the LR looks like at three points in training
for step, label in [(0, "step 0 (start)"),
                    (warmup_steps, "step=warmup end"),
                    (total_steps // 2, "step=half total")]:
    print(f"  LR at {label:25s}: {float(lr_schedule(step)):.6f}")

# =============================================================================
# 5.  Build the TrainState (no BatchNorm → simple params-only state)
# =============================================================================
class SimpleTrainState(train_state.TrainState):
    key: jax.Array

key, dropout_key = jax.random.split(key)
state = SimpleTrainState.create(
    apply_fn=model.apply,
    params=params["params"],
    key=dropout_key,
    tx=optax.adam(lr_schedule),
)

# =============================================================================
# 6.  RMSD before training
# =============================================================================
weights = jnp.ones(N_ATOMS)  # uniform weights
atom_rmsd_fn = give_weighted_rmsd_func(weights)

def compute_rmsd(state, x):
    """Run inference and compute mean RMSD over the batch."""
    key_eval = jax.random.PRNGKey(42)
    recon, z_mean, z_logvar = model.apply(
        {"params": state.params}, x, key_eval, train=False
    )
    return float(jnp.mean(atom_rmsd_fn(x, recon)))

rmsd_before = compute_rmsd(state, test_data)
print(f"\nRMSD before training: {rmsd_before * 10:.4f} Å")

# =============================================================================
# 7.  Manual training steps
# =============================================================================
@jax.jit
def train_step(state, batch):
    """One gradient update step (no BatchNorm)."""
    z_rng = jax.random.PRNGKey(state.step)
    dropout_key = jax.random.fold_in(state.key, state.step)

    def loss_fn(params):
        recon, z_mean, z_logvar = model.apply(
            {"params": params}, batch, z_rng, train=True,
            rngs={"dropout": dropout_key},
        )
        loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_fn(batch, recon) ** 2)))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

N_TRAIN_STEPS = 5
print(f"\nRunning {N_TRAIN_STEPS} manual training steps:")
for i in range(N_TRAIN_STEPS):
    batch = train_data[i * 32:(i + 1) * 32]
    state, loss = train_step(state, batch)
    print(f"  step {i+1}: log-RMSD loss = {float(loss):.4f}")

rmsd_after = compute_rmsd(state, test_data)
print(f"\nRMSD after {N_TRAIN_STEPS} steps: {rmsd_after * 10:.4f} Å")
print(f"Change: {(rmsd_after - rmsd_before) * 10:+.4f} Å  "
      f"(positive = no real improvement yet — 5 steps is tiny)")

# =============================================================================
# 8.  Encode the test set and inspect the latent shape
# =============================================================================
key_encode = jax.random.PRNGKey(0)
z_mean, z_logvar = model.apply(
    {"params": state.params}, test_data, key_encode,
    method=model.encode, train=False
)
print(f"\nLatent mean shape:   {z_mean.shape}   "
      f"(frames={len(test_data)}, latent_dim={json_params['latent_dim']})")
print(f"Latent logvar range: [{float(z_logvar.min()):.3f}, {float(z_logvar.max()):.3f}]")
print("\nDone. See tutorials/02_architecture_tour.py for a multi-model comparison.")
