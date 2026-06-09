"""
Latent-space perturbation analysis: generate per-latent-dimension trajectories.

Usage:
    python scripts/perturbation.py <json_file>

Loads the trained model, encodes the test set to obtain latent mean/std,
then sweeps each latent dimension from mean-5σ to mean+5σ while holding all
other dimensions at their mean.  Writes one DCD per latent plus the raw test
frames and their reconstructions to <data_dir>/perturbation/.

Replaces _05_perturbation.py.
"""

import os
import sys

from deepmms.analysis import Analyzer
from deepmms.analysis.perturbation import run_perturbation_analysis


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    HA = Analyzer(json_fn=json_fn)
    work_dir = os.path.join(HA.data_dir, "perturbation")
    log_fn = run_perturbation_analysis(HA, work_dir)
    print(f"Perturbation analysis complete. Log at {log_fn}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/perturbation.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
