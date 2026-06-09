"""
Train a Deep-MMS VAE model from a JSON configuration file.

Usage:
    python scripts/train.py <json_file>

Replaces _01_Run_and_Analyze_Heavy_Atom.py.  Trains the model with the
MAIN_train protocol (n_epochs warmup + auto-stop), then generates per-epoch
loss curves, a reconstruction RMSD histogram, and a latent-distribution
histogram in a figures/ subdirectory of the model's data_dir.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms import Experiment
from deepmms.training.loss import atom_rmsd


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    start = datetime.now()
    exp = Experiment(json_fn, from_json_params=False)
    exp.MAIN_train(n_epochs=1000, verbose=100)
    train_time = datetime.now() - start

    from deepmms.utils import printf
    printf(
        f"Time to train {exp.n_latents} model for {exp.epoch} epochs was"
        f" {train_time} averaging {train_time / exp.epoch} per epoch."
    )

    traingrp = exp.rootgrp["Train"]
    testgrp = exp.rootgrp["Test"]
    rmsd_train = np.mean(traingrp["RMSD_Loss_Term"][:, :], axis=-1)
    rmsd_test = np.mean(testgrp["RMSD_Loss_Term"][:, :], axis=-1)
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
    title = f"{exp.n_latents} Latents - RMSD Loss Term"
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    root_key = jax.random.PRNGKey(exp.epoch)
    main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

    if exp.is_batchnorm:
        decoded, latent_means, latent_vars = exp.state.apply_fn(
            {"params": exp.state.params, "batch_stats": exp.state.batch_stats},
            exp.test_data, main_key, train=False,
            rngs={"dropout": dropout_key},
        )
    else:
        decoded, latent_means, latent_vars = exp.state.apply_fn(
            {"params": exp.state.params},
            exp.test_data, main_key, train=False,
            rngs={"dropout": dropout_key},
        )

    printf(f"{decoded.shape=}, {latent_means.shape=}, {latent_vars.shape=}")

    rmsd_vals = atom_rmsd(exp.test_data, decoded) * 10
    plt.clf()
    plt.hist(rmsd_vals, bins=50)
    title = "Reconstruction RMSD"
    plt.title(title)
    plt.xlabel("RMSD (Angstrom)")
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    colors = [
        (i / exp.n_latents, 0.1, (exp.n_latents - i) / exp.n_latents)
        for i in range(1, exp.n_latents + 1)
    ]
    print(colors)
    plt.clf()
    plt.hist(latent_means.T, bins=100, histtype="step", alpha=0.5, color=colors)
    title = "All Latents"
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
