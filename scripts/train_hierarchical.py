"""
Train a Hierarchical VAE from a JSON configuration file.

Usage:
    python scripts/train_hierarchical.py <json_file>

The HVAE uses a two-level latent hierarchy: z1 captures global/slow modes
and z2 captures local variation conditioned on z1.  Training uses
HVAETrainer which sums KL divergences from both levels.

JSON config extras (on top of standard keys):
    None — standard latent_dim, dropout_rates, is_batchnorm are used.
    The z1/z2 split is latents // 4 vs. the remainder.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms.training.specialized import HVAETrainer
from deepmms.training.loss import atom_rmsd
from deepmms.utils import printf


def main(json_fn):
    """Entry point: parse args and run the HVAE training pipeline."""
    start = datetime.now()
    exp = HVAETrainer(json_fn, from_json_params=False)
    exp.MAIN_train(n_epochs=1000, verbose=100)
    train_time = datetime.now() - start

    printf(
        f"Time to train {exp.n_latents} latent HierarchicalVAE for {exp.epoch} epochs was"
        f" {train_time} averaging {train_time / max(exp.epoch, 1)} per epoch."
    )

    traingrp = exp.rootgrp["Train"]
    testgrp = exp.rootgrp["Test"]
    rmsd_train = np.mean(traingrp["RMSD_Loss_Term"][:, :], axis=-1)
    rmsd_test = np.mean(testgrp["RMSD_Loss_Term"][:, :], axis=-1)
    n1 = max(1, exp.n_latents // 4)
    n2 = exp.n_latents - n1
    printf(
        f"Final Epoch {exp.epoch} Train RMSD:{rmsd_train[-1]*10:2.3f} A,"
        f" Test RMSD:{rmsd_test[-1]*10:2.3f} A"
    )

    figure_dir = os.path.join(exp.data_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    plt.clf()
    plt.plot(np.arange(len(rmsd_train)), rmsd_train, label="Train")
    plt.plot(np.arange(len(rmsd_test)), rmsd_test, label="Test")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction RMSD (nm)")
    title = f"{exp.n_latents} Latents (z1={n1}, z2={n2}) - HierarchicalVAE RMSD Loss"
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    root_key = jax.random.PRNGKey(exp.epoch)
    main_key, _, dropout_key = jax.random.split(key=root_key, num=3)
    decoded, z1_mean, z1_logvar = exp.state.apply_fn(
        {"params": exp.state.params},
        exp.test_data, main_key, train=False,
        rngs={"dropout": dropout_key},
    )
    printf(f"{decoded.shape=}, {z1_mean.shape=}")

    rmsd_vals = atom_rmsd(exp.test_data, decoded) * 10
    plt.clf()
    plt.hist(rmsd_vals, bins=50)
    title = "HierarchicalVAE Reconstruction RMSD"
    plt.title(title)
    plt.xlabel("RMSD (Angstrom)")
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    printf(f"Summary: HierarchicalVAE z1={n1} z2={n2},"
           f" final_test_rmsd={rmsd_test[-1]*10:.3f} A")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_hierarchical.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
