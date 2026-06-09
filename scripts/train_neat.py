"""
Train a NEATAutoencoder using OpenES + topology growth.

Usage:
    python scripts/train_neat.py <json_file>

This script uses NEATTrainer, which drives gradient-free weight evolution via
OpenAI Evolution Strategies and grows the network topology whenever the fitness
plateaus.  Suitable for exploring non-gradient loss landscapes or when you want
automatic architecture search without manual hyperparameter tuning.

JSON config extras (all optional):
    neat_start_dim      : int   – initial hidden width (default: min(64, input_size))
    neat_start_layers   : int   – initial layer count (default: len(dropout_rates))
    es_population       : int   – perturbation population size (default: 50)
    es_sigma            : float – perturbation std deviation (default: 0.05)
    es_lr               : float – Adam lr applied to ES gradient (default: 0.01)
    neat_plateau_window : int   – generations without improvement before grow (default: 200)
    neat_plateau_thr    : float – fractional improvement threshold (default: 0.005)

Note: ES convergence is slower than Adam. For production-quality compression,
first train with scripts/train.py (BatchNorm_VAE + Adam) as a baseline.
NEAT is best suited for architecture search or gradient-hostile loss landscapes.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from deepmms.training.evolutionary import NEATTrainer
from deepmms.training.loss import atom_rmsd
from deepmms.utils import printf
import jax


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    start = datetime.now()
    trainer = NEATTrainer(json_fn)

    cutoff_epoch = trainer.json_params.get("max_epoch", 5000)
    trainer.train_n_epochs(cutoff_epoch, verbose=50)

    elapsed = datetime.now() - start
    printf(
        f"NEATTrainer finished: {trainer.epoch} epochs, "
        f"{trainer._grow_count} topology growths, "
        f"final topology {trainer.model.hidden_layers}, "
        f"elapsed {elapsed}"
    )

    figure_dir = os.path.join(trainer.data_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    rmsd_train = np.array(
        trainer.rootgrp["Train"]["RMSD_Loss_Term"][:, :]
    ).mean(axis=-1)
    rmsd_test = np.array(
        trainer.rootgrp["Test"]["RMSD_Loss_Term"][:, :]
    ).mean(axis=-1)

    plt.clf()
    plt.plot(rmsd_train * 10, label="Train (ES mean)")
    plt.plot(rmsd_test * 10, label="Test (ES mean)")
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Mean RMSD (Å)")
    plt.title(f"NEAT — {trainer.n_latents} latents  ({trainer._grow_count} growths)")
    plt.legend()
    plt.savefig(
        os.path.join(figure_dir, "neat_rmsd_loss.png"),
        dpi=300, bbox_inches="tight",
    )

    root_key = jax.random.PRNGKey(trainer.epoch)
    main_key, _, dropout_key = jax.random.split(root_key, 3)
    decoded, latent_means, _ = trainer.state.apply_fn(
        {"params": trainer.state.params},
        trainer.test_data, main_key, train=False,
        rngs={"dropout": dropout_key},
    )
    rmsd_vals = atom_rmsd(trainer.test_data, decoded) * 10
    plt.clf()
    plt.hist(np.array(rmsd_vals), bins=50)
    plt.xlabel("Reconstruction RMSD (Å)")
    plt.title(f"NEAT — Reconstruction RMSD ({trainer.model.hidden_layers})")
    plt.savefig(
        os.path.join(figure_dir, "neat_recon_rmsd_hist.png"),
        dpi=300, bbox_inches="tight",
    )
    printf(f"Figures saved to {figure_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_neat.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
