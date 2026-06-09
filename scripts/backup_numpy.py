"""
Copy .npy reconstruction error files to a versioned numpy_backups tree.

Usage:
    python scripts/backup_numpy.py <json_file>

Reads the data_dir from the JSON config, finds all .npy files there, and
copies them into numpy_backups/<model_name>/<n_latents>_latents/rpt_<n>/.

Replaces _03_preserve_numpy_data.py.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    with open(json_fn, "r") as g:
        json_params = json.load(g)

    data_dir = os.path.join(
        json_params["save_dir"],
        str(json_params["model_name"]),
        f'{json_params["latent_dim"]:04d}_latents/',
        f'rpt_{json_params["test_slice"]}/',
    )

    new_numpy_dir = os.path.join(
        json_params["save_dir"],
        "numpy_backups",
        str(json_params["model_name"]),
        f'{json_params["latent_dim"]:04d}_latents/',
        f'rpt_{json_params["test_slice"]}/',
    )
    os.makedirs(new_numpy_dir, exist_ok=True)

    for numpy_fn in glob.glob(os.path.join(data_dir, "*.npy")):
        new_fn = os.path.join(new_numpy_dir, os.path.basename(numpy_fn))
        arr = np.load(numpy_fn)
        np.save(new_fn, arr)
        print(f"Copied {numpy_fn} -> {new_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Back up .npy result files to numpy_backups/."
    )
    parser.add_argument("json_fn", help="JSON config file used to run the model")
    args = parser.parse_args()
    main(args.json_fn)
