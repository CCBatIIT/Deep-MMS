"""
Evaluate clustering quality of VAE latent encodings vs PCA on test data.

Usage:
    python scripts/clustering_metrics.py <json_file>

Loads the trained model, computes or loads a pairwise RMSD distance matrix,
determines the optimal cluster count for KMeans / Agglomerative / HDBScan,
then scores both PCA and encoder representations with six clustering metrics.
Results are written to <data_dir>/clustering/clustering_metric_log.txt.

Replaces _04_clustering_metrics.py.
"""

import os
import sys
import numpy as np
import jax.numpy as jnp

from deepmms.analysis import Analyzer
from deepmms.analysis.clustering import (
    CLUSTERING_METHODS,
    METRIC_FUNCTIONS,
    load_or_compute_distance_matrix,
    num_clusters_by_method,
    validation_value,
    evaluate_pca_encoder_against_metric,
)


def main(json_fn):
    """Entry point: parse args and run the pipeline."""
    HA = Analyzer(json_fn=json_fn)
    test_frames = HA.test_data

    work_dir = os.path.join(HA.data_dir, "clustering")
    os.makedirs(work_dir, exist_ok=True)
    log_fn = os.path.join(work_dir, "clustering_metric_log.txt")

    with open(log_fn, "w") as logfile:
        dist_mat_fn = os.path.join(work_dir, "TestDistances.sqmat.npy")
        test_distance_matrix = load_or_compute_distance_matrix(
            test_frames, dist_mat_fn, logfile=logfile
        )

        hdbscan_epsilons = np.linspace(0, float(test_distance_matrix.max()))

        optimal_nums_clusters = {}
        logfile.write("Determine optimal number of clusters for each method...\n")
        for key in CLUSTERING_METHODS:
            logfile.write(f"\t{key}...\n")
            optimal_nums_clusters[key] = num_clusters_by_method(
                method_name=key,
                test_frames=test_frames,
                test_distance_matrix=test_distance_matrix,
                hdbscan_epsilons=hdbscan_epsilons,
                work_dir=work_dir,
                model_name=HA.model_name,
                plot=True,
                logfile=logfile,
            )
            logfile.write("\t\tDone!\n")

        logfile.write(f"{optimal_nums_clusters}\n")

        for method_name in CLUSTERING_METHODS:
            n_clusters = optimal_nums_clusters[method_name]

            for metric_name, metric in METRIC_FUNCTIONS.items():
                metric_function, is_supervised = metric
                logfile.write(
                    f"{method_name=}, {metric_name=}, {is_supervised=}\n"
                )

                validation_y, validation_y_err = validation_value(
                    n_clusters=n_clusters,
                    method_name=method_name,
                    metric_function=metric_function,
                    test_frames=test_frames,
                    test_distance_matrix=test_distance_matrix,
                    supervised=is_supervised,
                )

                pca_scores, enc_scores = evaluate_pca_encoder_against_metric(
                    n_clusters=n_clusters,
                    method_name=method_name,
                    metric_function=metric_function,
                    analyzer=HA,
                    test_frames=test_frames,
                    test_distance_matrix=test_distance_matrix,
                    supervised=is_supervised,
                )

                pca_mean = float(np.mean(pca_scores))
                pca_std = float(np.std(pca_scores))
                vae_mean = float(np.mean(enc_scores))
                vae_std = float(np.std(enc_scores))

                logfile.write(
                    f"For clustering method {method_name} and metric {metric_name}:\n"
                    f"        PCA scored {pca_mean:0.3f} +/- {pca_std:0.3f}\n"
                    f"        VAE scored {vae_mean:0.3f} +/- {vae_std:0.3f}\n"
                )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/clustering_metrics.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])
