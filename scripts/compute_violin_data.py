"""
Compute per-frame VAE and PCA reconstruction RMSD for a trained model.

Usage:
    python scripts/compute_violin_data.py <json_file>

Loads the trained checkpoint specified in the JSON, runs inference on the
test set, fits a matching PCA, and saves VAE_RMSD.npy, VAE_LOSS_RMSD.npy,
PCA_RMSD.npy, PCA_LOSS_RMSD.npy into the model's data_dir.

Replaces _02_write_viollin_data.py (the __main__ block).
"""

import sys

from deepmms.analysis import Analyzer, violin_data


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    analyzer = Analyzer(json_fn)
    violin_data(analyzer, save_npy=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_violin_data.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
