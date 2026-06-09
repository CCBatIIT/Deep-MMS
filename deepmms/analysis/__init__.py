"""
Analysis subpackage: exports Analyzer, violin_data, violin_plots, and key clustering utilities.
"""

from .reconstruction import Analyzer, violin_data, save_dcd
from .plotting import violin_plots, violin_difference_plot, figure_log_to_csv
from .clustering import (
    rmsd_distance_matrix,
    load_or_compute_distance_matrix,
    num_clusters_by_method,
    nn_operate,
    validation_value,
    evaluate_pca_encoder_against_metric,
)
from .perturbation import run_perturbation_analysis

__all__ = [
    "Analyzer",
    "violin_data",
    "save_dcd",
    "violin_plots",
    "violin_difference_plot",
    "figure_log_to_csv",
    "rmsd_distance_matrix",
    "load_or_compute_distance_matrix",
    "num_clusters_by_method",
    "nn_operate",
    "validation_value",
    "evaluate_pca_encoder_against_metric",
    "run_perturbation_analysis",
]
