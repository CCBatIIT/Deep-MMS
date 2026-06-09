"""
Train a Masked Autoencoder (MAE) for molecular coordinates.

Usage:
    python scripts/train_mae.py <json_file>

During training, a fraction of atoms are randomly masked from the encoder.
The decoder reconstructs all atoms but the training loss is RMSD on masked
atoms only.  Uses MAETrainer.

JSON config extras (on top of standard keys):
    embed_dim  : int   – atom token embedding width (default min(128, input_size))
    mask_ratio : float – fraction of atoms masked (default 0.75)
    num_heads  : int   – transformer attention heads (default 4)
    ffn_mult   : float – FFN width multiplier (default 4.0)
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms.training.specialized import MAETrainer
from deepmms.training.loss import atom_rmsd
from deepmms.utils import printf


def main(json_fn):
    """Entry point: parse args and run the MAE training pipeline."""
    start = datetime.now()
    exp = MAETrainer(json_fn, from_json_params=False)
    exp.MAIN_train(n_epochs=1000, verbose=100)
    train_time = datetime.now() - start

    printf(
        f"Time to train {exp.n_latents} latent MaskedAutoencoder for {exp.epoch} epochs was"
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
        f"{exp.n_latents} Latents - MAE RMSD Loss"
        f" (mask_ratio={exp.model.mask_ratio})"
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
    title = "MAE Reconstruction RMSD (All Atoms)"
    plt.title(title)
    plt.xlabel("RMSD (Angstrom)")
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    printf(f"Summary: MaskedAutoencoder latents={exp.n_latents},"
           f" mask_ratio={exp.model.mask_ratio},"
           f" final_test_rmsd={rmsd_test[-1]*10:.3f} A")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_mae.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
