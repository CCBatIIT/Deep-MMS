"""
Tutorial 3: Exploring the Latent Space
========================================
Run from the project root:
    python tutorials/03_latent_space.py

After training a BatchNorm_VAE for 20 epochs on synthetic data we:
  1. Encode the full dataset and inspect per-dimension latent statistics.
  2. Perturb dimension 0 from -3σ to +3σ and decode each point to see
     how much the reconstruction changes (latent traversal).
  3. Compare BetaVAE (β=4) vs standard VAE (β=1) latent compactness:
     higher β forces a more constrained, unit-Gaussian-like posterior.

No plotting needed — all results printed to stdout.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

import deepmms.utils  # side-effect: sets jax_enable_x64

from deepmms.models.vae import BatchNorm_VAE, reparameterize
from deepmms.models.beta_vae import BetaVAE
from deepmms.training.loss import give_weighted_rmsd_func, KL_loss, atom_rmsd

# =============================================================================
# Synthetic dataset
# =============================================================================
N_FRAMES = 300
N_ATOMS  = 10
INPUT_SIZE = N_ATOMS * 3  # 30
LATENTS  = 4
DROPOUT  = [0.0, 0.0]

key = jax.random.PRNGKey(7)
key, dk = jax.random.split(key)
data = jax.random.normal(dk, (N_FRAMES, INPUT_SIZE))
train_data = data[:240]
test_data  = data[240:]

weights = jnp.ones(N_ATOMS)
atom_rmsd_fn = give_weighted_rmsd_func(weights)


# =============================================================================
# Helper: build model + TrainState
# =============================================================================
def build_model_state(model_cls, extra_kwargs=None, lr=1e-3, seed=0):
    extra_kwargs = extra_kwargs or {}
    hidden = model_cls.hidden_layers_from_config(INPUT_SIZE, LATENTS, DROPOUT, {})
    model = model_cls(
        input_size=INPUT_SIZE, latents=LATENTS,
        hidden_layers=tuple(hidden), dropout_rates=DROPOUT,
        is_batchnorm=False, **extra_kwargs,
    )
    key_init = jax.random.PRNGKey(seed)
    params = model.init(key_init, train_data[:4], key_init, train=False)

    class TS(train_state.TrainState):
        key: jax.Array

    state = TS.create(
        apply_fn=model.apply,
        params=params["params"],
        key=key_init,
        tx=optax.adam(lr),
    )
    return model, state


# =============================================================================
# Helper: train a model for n_epochs, returning the final state
# =============================================================================
def quick_train(model, state, n_epochs=20, batch_size=32, beta=1.0):
    """Train model for n_epochs; beta scales the KL penalty (1.0 = standard VAE)."""

    @jax.jit
    def step(state, batch):
        z_rng = jax.random.PRNGKey(state.step)
        drop_key = jax.random.fold_in(state.key, state.step)

        def loss_fn(params):
            recon, z_mean, z_logvar = model.apply(
                {"params": params}, batch, z_rng, train=True,
                rngs={"dropout": drop_key},
            )
            rmsd_loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd_fn(batch, recon) ** 2)))
            kl = KL_loss(z_mean, z_logvar)
            return rmsd_loss + beta * kl

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    n_batches = len(train_data) // batch_size
    for epoch in range(n_epochs):
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), len(train_data))
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            state = step(state, train_data[idx])
    return state


# =============================================================================
# 1. Train a BatchNorm_VAE for 20 epochs
# =============================================================================
print("=" * 60)
print("Part 1: Latent statistics after training (β=1, standard VAE)")
print("=" * 60)

model, state = build_model_state(BatchNorm_VAE)
state = quick_train(model, state, n_epochs=20, beta=1.0)

# Encode the full test set
z_rng = jax.random.PRNGKey(0)
z_mean, z_logvar = model.apply(
    {"params": state.params}, test_data, z_rng,
    method=model.encode, train=False,
)

print(f"\nTest set latent means  (shape {z_mean.shape}):")
print(f"  {'dim':>4}  {'min':>8}  {'max':>8}  {'std':>8}")
for k in range(LATENTS):
    col = z_mean[:, k]
    print(f"  {k:>4}  {float(col.min()):>8.3f}  {float(col.max()):>8.3f}  {float(col.std()):>8.3f}")

# =============================================================================
# 2. Latent traversal: sweep dimension 0 from -3σ to +3σ
# =============================================================================
print(f"\nPart 2: Latent traversal — sweep dim 0 from -3σ to +3σ")
mean_latent = jnp.mean(z_mean, axis=0)          # (LATENTS,)
std_dim0    = float(jnp.std(z_mean[:, 0]))       # σ of dimension 0

# Decode the mean structure
mean_struct = model.apply(
    {"params": state.params},
    mean_latent[None, :],      # shape (1, LATENTS)
    method=model.decode,
    z_rng=z_rng, train=False,
)

N_STEPS = 9
sweep_vals = jnp.linspace(-3 * std_dim0, 3 * std_dim0, N_STEPS)
print(f"\n  σ(dim 0) = {std_dim0:.3f} nm")
print(f"  {'perturb':>10}  {'RMSD from mean':>16}")
for v in sweep_vals:
    z_perturbed = mean_latent.at[0].set(float(v))
    recon = model.apply(
        {"params": state.params},
        z_perturbed[None, :],
        method=model.decode, z_rng=z_rng, train=False,
    )
    rmsd = float(jnp.mean(atom_rmsd(mean_struct, recon)))
    print(f"  {float(v):>10.3f}  {rmsd * 10:>14.3f} A")

# =============================================================================
# 3. BetaVAE (β=4) vs standard VAE (β=1): latent compactness
# =============================================================================
print(f"\nPart 3: β=4 (BetaVAE) vs β=1 (standard VAE) — latent std per dim")
print("Higher β pushes latents toward N(0,1), making std closer to 1.0")

model_b4, state_b4 = build_model_state(BetaVAE, extra_kwargs={"beta": 4.0}, seed=1)
state_b4 = quick_train(model_b4, state_b4, n_epochs=20, beta=4.0)

z_mean_b4, _ = model_b4.apply(
    {"params": state_b4.params}, test_data, z_rng,
    method=model_b4.encode, train=False,
)

print(f"\n  {'dim':>4}  {'std β=1':>10}  {'std β=4':>10}  {'β=4 more compact?':>20}")
for k in range(LATENTS):
    s1 = float(jnp.std(z_mean[:, k]))
    s4 = float(jnp.std(z_mean_b4[:, k]))
    flag = "yes" if s4 < s1 else "no"
    print(f"  {k:>4}  {s1:>10.4f}  {s4:>10.4f}  {flag:>20}")

print("\nConclusion: β=4 tends to produce a more compact, regularised latent.")
print("Done. See tutorials/04_adding_a_model.py to add a custom architecture.")
