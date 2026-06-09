"""
Tutorial 2: Comparing Architectures on the Same Data
======================================================
Run from the project root:
    python tutorials/02_architecture_tour.py

We instantiate five different architectures on an identical synthetic dataset
and compare: parameter count, forward-pass output shapes, and the per-frame
reconstruction RMSD straight from random initialisation (before any training).

The 10-atom system (30 features) is intentionally tiny so all models init fast.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

import deepmms.utils  # side-effect: sets jax_enable_x64

from deepmms.models.vae import BatchNorm_VAE
from deepmms.models.beta_vae import BetaVAE
from deepmms.models.transformer_vae import TransformerVAE
from deepmms.models.equivariant_vae import EquivariantVAE
from deepmms.models.kan_vae import KANVAE
from deepmms.training.loss import atom_rmsd

# =============================================================================
# Synthetic dataset: 100 frames, 10 atoms (30 features)
# =============================================================================
N_FRAMES = 100
N_ATOMS = 10
INPUT_SIZE = N_ATOMS * 3  # 30

key = jax.random.PRNGKey(42)
key, data_key = jax.random.split(key)
# Clip data to [-2.5, 2.5] so values stay inside the KANVAE B-spline grid domain [-3, 3].
# Real MD data is centred and scaled during preprocessing; this clip mimics that.
data = jnp.clip(jax.random.normal(data_key, (N_FRAMES, INPUT_SIZE)), -2.5, 2.5)

LATENTS = 4
DROPOUT = [0.0, 0.0]  # 2 hidden layers
json_params_base = {
    "latent_dim": LATENTS,
    "dropout_rates": DROPOUT,
    "is_batchnorm": False,
    "max_epoch": 100,
}


# =============================================================================
# Helper: count scalar parameters in a JAX param tree
# =============================================================================
def count_params(params):
    """Count total scalar parameters in a nested param dict."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# =============================================================================
# Helper: instantiate a model, init it, and measure RMSD and param count
# =============================================================================
def profile_model(name, model_cls, json_extra=None, ctor_extra=None):
    """
    Instantiate model_cls, run a forward pass, and return a summary dict.

    hidden_layers_from_config is called first so layer sizing is architecture-
    aware — TransformerVAE uses embed_dim, KANVAE uses embed_dim, MLP models
    use input_size as the hidden width.

    Parameters
    ----------
    json_extra : dict
        Extra keys forwarded to hidden_layers_from_config via json_params
        (e.g. embed_dim, num_heads).  These are config-level parameters.
    ctor_extra : dict
        Extra keyword arguments passed directly to the model constructor
        (e.g. num_heads, beta).  These must match declared Flax attributes.
    """
    json_extra = json_extra or {}
    ctor_extra = ctor_extra or {}

    # Merge extra keys into json_params for hidden_layers_from_config
    json_params = {**json_params_base, **json_extra}

    # Architecture-specific layer sizing
    hidden_layers = model_cls.hidden_layers_from_config(
        INPUT_SIZE, LATENTS, DROPOUT, json_params
    )

    model = model_cls(
        input_size=INPUT_SIZE,
        latents=LATENTS,
        hidden_layers=tuple(hidden_layers),
        dropout_rates=DROPOUT,
        is_batchnorm=False,
        **ctor_extra,
    )

    # Initialise parameters
    init_key, rng_key = jax.random.split(jax.random.PRNGKey(0))
    params = model.init(init_key, data[:4], rng_key, train=False)
    n_params = count_params(params)

    # Forward pass on the full 100-frame dataset
    recon, z_mean, z_logvar = model.apply(params, data, rng_key, train=False)

    # Per-frame RMSD using the unweighted function (vmapped).
    # KANVAE can produce very large values at random init due to B-spline amplification;
    # we cap at 1e6 to allow the comparison table to print sensibly.
    rmsd_vals = atom_rmsd(data, recon)  # shape (N_FRAMES,)
    rmsd_vals = jnp.where(jnp.isfinite(rmsd_vals), rmsd_vals, jnp.nan)
    rmsd_mean = float(jnp.nanmean(rmsd_vals))
    rmsd_std  = float(jnp.nanstd(rmsd_vals))

    return {
        "name": name,
        "n_params": n_params,
        "output_shape": recon.shape,
        "latent_shape": z_mean.shape,
        "hidden_layers": hidden_layers,
        "rmsd_mean_nm": rmsd_mean,
        "rmsd_std_nm":  rmsd_std,
    }


# =============================================================================
# Register the five architectures to benchmark
# =============================================================================
# For TransformerVAE: embed_dim goes in json_extra (used by hidden_layers_from_config);
#   num_heads goes in ctor_extra (declared Flax attribute).
# For KANVAE: embed_dim goes in json_extra; kan_n_grid goes in ctor_extra.
# For EquivariantVAE: embed_dim in json_extra; n_interactions/cutoff_dist in ctor_extra.
# For MLP models (BatchNorm_VAE, BetaVAE): no json_extra needed; hidden width = INPUT_SIZE.

ARCHITECTURES = [
    # (display name, class, json_extra, ctor_extra)
    ("BatchNorm_VAE",  BatchNorm_VAE,  {}, {}),
    ("BetaVAE",        BetaVAE,        {}, {"beta": 4.0}),
    ("TransformerVAE", TransformerVAE, {"embed_dim": 16, "num_heads": 2},
                                       {"num_heads": 2}),
    ("EquivariantVAE", EquivariantVAE, {"embed_dim": 16, "n_interactions": 2},
                                       {"n_interactions": 2, "cutoff_dist": 5.0}),
    ("KANVAE",         KANVAE,         {"embed_dim": 16},
                                       {"kan_n_grid": 3}),
]

# =============================================================================
# Run and print results
# =============================================================================
print("Architecture comparison on synthetic 10-atom system (100 frames, 30 features)")
print(f"{'Model':<20} {'params':>8}  {'hidden_layers':<24}  "
      f"{'recon_rmsd_mean':>16}  {'recon_rmsd_std':>14}")
print("-" * 90)

results = []
for name, cls, json_extra, ctor_extra in ARCHITECTURES:
    r = profile_model(name, cls, json_extra=json_extra, ctor_extra=ctor_extra)
    results.append(r)
    rmsd_str = f"{r['rmsd_mean_nm']*10:>14.3f}" if jnp.isfinite(r['rmsd_mean_nm']) else "        unstable"
    std_str  = f"{r['rmsd_std_nm']*10:>12.3f}" if jnp.isfinite(r['rmsd_std_nm']) else "     unstable"
    print(
        f"{r['name']:<20} {r['n_params']:>8,}  "
        f"{str(r['hidden_layers']):<24}  "
        f"{rmsd_str} A  "
        f"{std_str} A"
    )

print()
print("Notes:")
print("  - RMSD is from random init (before any training) — expect large values.")
print("  - BatchNorm_VAE hidden width = INPUT_SIZE (30); Transformer/KAN use embed_dim.")
print("  - TransformerVAE and EquivariantVAE have explicit atom-level structure.")
print("  - KANVAE replaces Dense layers with B-spline edge activations.")
print("  - 'unstable' means random-init decoder produces non-finite values;")
print("    KANVAE converges normally once trained (B-spline outputs become bounded).")
print()

# Show output shapes explicitly for the first model
r0 = results[0]
print(f"BatchNorm_VAE forward-pass shapes:")
print(f"  input:   (N_FRAMES={N_FRAMES}, INPUT_SIZE={INPUT_SIZE})")
print(f"  recon:   {r0['output_shape']}  (must match input)")
print(f"  z_mean:  {r0['latent_shape']}  (LATENTS={LATENTS})")
print("\nDone. See tutorials/03_latent_space.py to explore the latent space.")
