"""
Train a Deep-MMS model from a JSON config, dispatching on its ``architecture``
key.  Thin wrapper around :func:`deepmms.dispatch.build_harness` for use with
the config files written by scripts/generate_configs.py.

Usage:
    python scripts/train_dispatch.py <config.json> [n_epochs]

(For an ad-hoc run without a JSON file, use the ``deep-mms`` command instead.)
"""

import json
import os
import sys

# Make the project root importable so ``deepmms`` is found even when the package
# is not pip-installed -- important for SLURM array jobs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepmms.dispatch import build_harness
from deepmms.utils import printf


def main(json_fn, n_epochs=1000):
    """Read the architecture from the config, build the harness, and train."""
    with open(json_fn, "r") as fh:
        architecture = json.load(fh).get("architecture", "batchnorm_vae")
    printf(f"Dispatching architecture={architecture!r} for {json_fn}")
    build_harness(architecture, json_fn).MAIN_train(n_epochs=n_epochs, verbose=100)
    printf(f"Done training {architecture} for {json_fn}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/train_dispatch.py <config.json> [n_epochs]")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
