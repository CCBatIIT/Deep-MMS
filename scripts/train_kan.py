"""
Train a Kolmogorov-Arnold Network (KAN) VAE from a JSON configuration file.

Usage:
    python scripts/train_kan.py <json_file>

Replaces Dense layers with KANLayers where each edge has a learnable
B-spline activation function.  Uses the standard Experiment trainer.

JSON config extras (on top of standard keys):
    embed_dim  : int – KAN layer width (default min(64, input_size))
    kan_n_grid : int – number of B-spline grid intervals (default 5)
    kan_order  : int – B-spline polynomial order (default 3)
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms import Experiment
from deepmms.models import KANVAE
from deepmms.training.loss import atom_rmsd
from deepmms.utils import printf


def main(json_fn):
    """Entry point: parse args and run the KANVAE training pipeline."""
    start = datetime.now()
    exp = Experiment(json_fn, from_json_params=False, model_cls=KANVAE)
    exp.MAIN_train(n_epochs=1000, verbose=100)
    train_time = datetime.now() - start

    printf(
        f"Time to train {exp.n_latents} latent KANVAE for {exp.epoch} epochs was"
        f" {train_time} averaging {train_time / max(exp.epoch, 1)} per epoch."
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
    title = (
        f"{exp.n_latents} Latents - KANVAE RMSD Loss"
        f" (grid={exp.model.kan_n_grid}, order={exp.model.kan_order})"
    )
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    root_key = jax.random.PRNGKey(exp.epoch)
    main_key, _, dropout_key = jax.random.split(key=root_key, num=3)
    decoded, latent_means, latent_vars = exp.state.apply_fn(
        {"params": exp.state.params},
        exp.test_data, main_key, train=False,
        rngs={"dropout": dropout_key},
    )
    printf(f"{decoded.shape=}, {latent_means.shape=}")

    rmsd_vals = atom_rmsd(exp.test_data, decoded) * 10
    plt.clf()
    plt.hist(rmsd_vals, bins=50)
    title = "KANVAE Reconstruction RMSD"
    plt.title(title)
    plt.xlabel("RMSD (Angstrom)")
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    colors = [
        (i / exp.n_latents, 0.1, (exp.n_latents - i) / exp.n_latents)
        for i in range(1, exp.n_latents + 1)
    ]
    plt.clf()
    plt.hist(np.array(latent_means).T, bins=100, histtype="step", alpha=0.5, color=colors)
    title = "KANVAE All Latents"
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    printf(f"Summary: KANVAE latents={exp.n_latents},"
           f" kan_n_grid={exp.model.kan_n_grid}, kan_order={exp.model.kan_order},"
           f" final_test_rmsd={rmsd_test[-1]*10:.3f} A")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_kan.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
