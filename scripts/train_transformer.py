"""
Train a TransformerVAE model from a JSON configuration file.

Usage:
    python scripts/train_transformer.py <json_file>

The TransformerVAE treats each heavy atom as a sequence token and uses
multi-head self-attention to model long-range inter-atomic relationships.
Training follows the same MAIN_train protocol as the standard VAE.

JSON config extras (all optional):
    embed_dim  : int   – attention embedding width (default: min(256, input_size))
    num_heads  : int   – attention heads per block (default: 4)
    ffn_mult   : float – FFN hidden width multiple of embed_dim (default: 4.0)

Example JSON addition:
    "embed_dim": 128,
    "num_heads": 4
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms import Experiment
from deepmms.models import TransformerVAE
from deepmms.training.loss import atom_rmsd
from deepmms.utils import printf
import jax
import jax.numpy as jnp


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    start = datetime.now()
    exp = Experiment(json_fn, model_cls=TransformerVAE)
    exp.MAIN_train(n_epochs=1000, verbose=100)
    train_time = datetime.now() - start

    printf(
        f"TransformerVAE: {exp.n_latents} latents trained for {exp.epoch} epochs"
        f" in {train_time} ({train_time / exp.epoch} per epoch)"
    )

    traingrp = exp.rootgrp["Train"]
    testgrp = exp.rootgrp["Test"]
    rmsd_train = np.mean(traingrp["RMSD_Loss_Term"][:, :], axis=-1)
    rmsd_test = np.mean(testgrp["RMSD_Loss_Term"][:, :], axis=-1)
    printf(
        f"Final  Train RMSD: {rmsd_train[-1]*10:.3f} Å"
        f"  Test RMSD: {rmsd_test[-1]*10:.3f} Å"
    )

    figure_dir = os.path.join(exp.data_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    plt.clf()
    plt.plot(rmsd_train * 10, label="Train")
    plt.plot(rmsd_test * 10, label="Test")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction RMSD (Å)")
    plt.title(f"TransformerVAE — {exp.n_latents} latents")
    plt.legend()
    plt.savefig(
        os.path.join(figure_dir, "transformer_rmsd_loss.png"),
        dpi=300, bbox_inches="tight",
    )

    root_key = jax.random.PRNGKey(exp.epoch)
    main_key, _, dropout_key = jax.random.split(root_key, 3)
    decoded, latent_means, latent_vars = exp.state.apply_fn(
        {"params": exp.state.params},
        exp.test_data, main_key, train=False,
        rngs={"dropout": dropout_key},
    )
    rmsd_vals = atom_rmsd(exp.test_data, decoded) * 10
    plt.clf()
    plt.hist(np.array(rmsd_vals), bins=50)
    plt.xlabel("Reconstruction RMSD (Å)")
    plt.title("TransformerVAE — Reconstruction RMSD")
    plt.savefig(
        os.path.join(figure_dir, "transformer_recon_rmsd_hist.png"),
        dpi=300, bbox_inches="tight",
    )
    printf(f"Figures saved to {figure_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_transformer.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
