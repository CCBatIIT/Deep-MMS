"""
Train a RealNVP normalising flow from a JSON configuration file.

Usage:
    python scripts/train_flow.py <json_file>

The flow has no bottleneck — input_size == flow dimension.  Training
minimises the negative log-likelihood (NLL) = 0.5*||z||^2 - log_det.
RMSD is logged separately as a diagnostic metric.  Uses FlowTrainer.

JSON config extras (on top of standard keys):
    n_coupling_layers : int – number of affine coupling layers (default 8)

Note: latent_dim controls how many leading dimensions of z are used for
analysis, not the flow's internal dimension.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms.training.specialized import FlowTrainer
from deepmms.training.loss import atom_rmsd
from deepmms.utils import printf


def main(json_fn):
    """Entry point: parse args and run the RealNVPFlow training pipeline."""
    start = datetime.now()
    exp = FlowTrainer(json_fn, from_json_params=False)
    exp.MAIN_train(n_epochs=1000, verbose=100)
    train_time = datetime.now() - start

    printf(
        f"Time to train RealNVPFlow ({exp.n_latents} analysis latents) for"
        f" {exp.epoch} epochs was {train_time}"
        f" averaging {train_time / max(exp.epoch, 1)} per epoch."
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
    plt.plot(np.arange(len(rmsd_train)), rmsd_train, label="Train (RMSD diagnostic)")
    plt.plot(np.arange(len(rmsd_test)), rmsd_test, label="Test (RMSD diagnostic)")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction RMSD (nm)")
    title = f"RealNVP Flow RMSD Diagnostic ({exp.n_latents} analysis latents)"
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    root_key = jax.random.PRNGKey(exp.epoch)
    main_key, _, dropout_key = jax.random.split(key=root_key, num=3)
    x_recon, z, _ = exp.state.apply_fn(
        {"params": exp.state.params},
        exp.test_data, main_key, train=False,
        rngs={"dropout": dropout_key},
    )
    printf(f"{x_recon.shape=}, {z.shape=}")

    rmsd_vals = atom_rmsd(exp.test_data, x_recon) * 10
    plt.clf()
    plt.hist(rmsd_vals, bins=50)
    title = "RealNVP Flow Reconstruction RMSD"
    plt.title(title)
    plt.xlabel("RMSD (Angstrom)")
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    # Latent distribution plot (leading latents)
    z_np = np.array(z[:, :exp.n_latents])
    colors = [
        (i / exp.n_latents, 0.1, (exp.n_latents - i) / exp.n_latents)
        for i in range(1, exp.n_latents + 1)
    ]
    plt.clf()
    plt.hist(z_np.T, bins=100, histtype="step", alpha=0.5, color=colors)
    title = "RealNVP Flow Latents"
    plt.title(title)
    plt.savefig(os.path.join(figure_dir, title + ".png"), dpi=900, bbox_inches="tight")

    printf(f"Summary: RealNVPFlow {exp.n_latents} analysis latents,"
           f" final_test_rmsd={rmsd_test[-1]*10:.3f} A")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_flow.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
