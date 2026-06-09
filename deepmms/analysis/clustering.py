"""
Clustering metric functions for comparing VAE latent encodings to PCA encodings.

Provides utilities to compute pairwise RMSD distance matrices, determine the
optimal cluster count by silhouette score, and evaluate supervised and
unsupervised clustering metrics for both PCA and encoder representations.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (
    rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

CLUSTERING_METHODS = {
    "KMeans": KMeans,
    "Agglomerative": AgglomerativeClustering,
    "HDBScan": HDBSCAN,
}

METRIC_FUNCTIONS = {
    "Rand Score": [rand_score, True],
    "normalized MI Score": [normalized_mutual_info_score, True],
    "Fowlkes-Mallows Score": [fowlkes_mallows_score, True],
    "Silhouette Score": [silhouette_score, False],
    "Davies-Bouldin Score": [davies_bouldin_score, False],
    "Calinksi-Harabasz Score": [calinski_harabasz_score, False],
}


def _one_off_rmsd(a, b):
    return jnp.sqrt(jnp.mean(jnp.sum((b.reshape(-1, 3) - a.reshape(-1, 3)) ** 2, axis=1)))


def _dist_matrix_row(a, bs):
    return jax.vmap(_one_off_rmsd, in_axes=(None, 0))(a, bs)


def rmsd_distance_matrix(frames):
    """
    Compute the full pairwise RMSD distance matrix for a set of frames.

    Parameters
    ----------
    frames : jnp.ndarray, shape (n_frames, n_atoms*3)
        Flattened coordinate frames.

    Returns
    -------
    jnp.ndarray, shape (n_frames, n_frames)
        Symmetric matrix of pairwise RMSD values.
    """
    dist_mat = jnp.empty((frames.shape[0], frames.shape[0]))
    for i in range(frames.shape[0]):
        dist_mat = dist_mat.at[i].set(_dist_matrix_row(frames[i], frames))
    return dist_mat


def load_or_compute_distance_matrix(frames, dist_mat_fn, logfile=None):
    """
    Load a precomputed distance matrix from disk or compute and save it.

    Parameters
    ----------
    frames : jnp.ndarray
        Coordinate frames used if recomputation is needed.
    dist_mat_fn : str
        Path to the .npy file for the square distance matrix.
    logfile : file-like or None
        Open writable file for status messages; no-op when None.

    Returns
    -------
    jnp.ndarray, shape (n_frames, n_frames)
    """
    def log(msg):
        if logfile is not None:
            logfile.write(msg)

    log("Get distance matrix of testing data... \n")
    try:
        log("Attempting to load distance matrix...\n")
        log(f"\tFrom:{dist_mat_fn}\n")
        matrix = jnp.load(dist_mat_fn)
        log("\tSuccess!\n")
    except Exception:
        log("\tSomething went wrong, regenerating distance matrix...\n")
        matrix = rmsd_distance_matrix(frames)
        log("\t\t Done!\n")
        jnp.save(dist_mat_fn, matrix)
        log(f"\t\t Saved to {dist_mat_fn}\n")
    return matrix


def _fit_predict(method_name, n_clusters, test_frames, test_distance_matrix):
    """Apply the named clustering method to get integer labels."""
    if method_name == "KMeans":
        return CLUSTERING_METHODS[method_name](n_clusters=n_clusters).fit_predict(test_frames)
    elif method_name == "Agglomerative":
        return CLUSTERING_METHODS[method_name](
            n_clusters=n_clusters, metric="precomputed", linkage="complete"
        ).fit_predict(test_distance_matrix)
    elif method_name == "HDBScan":
        return HDBSCAN(
            metric="precomputed", cluster_selection_epsilon=n_clusters
        ).fit_predict(np.array(test_distance_matrix, copy=True))
    raise ValueError(f"Unknown method: {method_name}")


def num_clusters_by_method(
    method_name,
    test_frames,
    test_distance_matrix,
    hdbscan_epsilons,
    work_dir,
    model_name,
    min_clusters=2,
    max_clusters=25,
    num_repetitions=5,
    plot=False,
    logfile=None,
):
    """
    Determine the optimal cluster count for a given method via silhouette score.

    Sweeps the number of clusters (or HDBSCAN epsilon) from min_clusters to
    max_clusters over num_repetitions random seeds and returns the count with
    the highest mean silhouette score.

    Parameters
    ----------
    method_name : str
        One of 'KMeans', 'Agglomerative', 'HDBScan'.
    test_frames : jnp.ndarray
    test_distance_matrix : jnp.ndarray
    hdbscan_epsilons : np.ndarray
        Epsilon sweep values used when method_name == 'HDBScan'.
    work_dir : str
        Directory to save the silhouette plot.
    model_name : str
        Used in the plot filename.
    min_clusters, max_clusters : int
        Range of cluster counts to sweep.
    num_repetitions : int
        Number of repeated fits per cluster count.
    plot : bool
        Save a silhouette-score plot when True.
    logfile : file-like or None
        Open writable file for status messages.

    Returns
    -------
    int
        Optimal cluster count (or epsilon index for HDBScan).
    """

    def log(msg):
        if logfile is not None:
            logfile.write(msg)

    sil_scores = np.empty((num_repetitions, 1 + max_clusters - min_clusters))
    nums_clusters = np.arange(min_clusters, max_clusters + 1)

    for j in range(num_repetitions):
        if method_name == "KMeans":
            for i in range(min_clusters, max_clusters + 1):
                labels = CLUSTERING_METHODS[method_name](n_clusters=i).fit_predict(test_frames)
                sil_scores[j, i - min_clusters] = silhouette_score(test_frames, labels)
        elif method_name == "Agglomerative":
            for i in range(min_clusters, max_clusters + 1):
                labels = CLUSTERING_METHODS[method_name](
                    n_clusters=i, metric="precomputed", linkage="complete"
                ).fit_predict(test_distance_matrix)
                sil_scores[j, i - min_clusters] = silhouette_score(test_frames, labels)
        elif method_name == "HDBScan":
            for i, k in enumerate(range(min_clusters, max_clusters + 1)):
                eps = hdbscan_epsilons[i]
                labels = HDBSCAN(
                    metric="precomputed", cluster_selection_epsilon=eps
                ).fit_predict(np.array(test_distance_matrix, copy=True))
                sil_scores[j, i - min_clusters] = silhouette_score(test_frames, labels)

    if plot:
        plt.clf()
        if method_name in ["KMeans", "Agglomerative"]:
            plt.errorbar(
                x=nums_clusters,
                y=np.mean(sil_scores, axis=0),
                yerr=np.std(sil_scores, axis=0),
            )
            plt.xlabel("Num Clusters")
            log(f"{method_name}, {np.argmax(np.mean(sil_scores, axis=0)) + min_clusters}\n")
        elif method_name == "HDBScan":
            plt.errorbar(
                x=hdbscan_epsilons[: len(range(min_clusters, max_clusters + 1))],
                y=np.mean(sil_scores, axis=0),
                yerr=np.std(sil_scores, axis=0),
            )
            plt.xlabel("HDBScan Epsilon")
            log(f"{method_name}, {hdbscan_epsilons[np.argmax(np.mean(sil_scores, axis=0))]}\n")
        plt.ylabel("Silhouette Score")
        plt.title(method_name)
        plt.savefig(
            os.path.join(work_dir, f"optimal_n_clusters_{model_name}_{method_name}.png"),
            dpi=900,
        )

    return np.argmax(np.mean(sil_scores, axis=0)) + min_clusters


def nn_operate(analyzer, data, rng_seed=69420):
    """
    Run a forward pass through the trained model without gradient tracking.

    Parameters
    ----------
    analyzer : Analyzer
        A restored Analyzer instance.
    data : jnp.ndarray
        Input coordinate frames.
    rng_seed : int
        RNG seed for dropout / reparameterisation keys.

    Returns
    -------
    decoded : jnp.ndarray
    latent_means : jnp.ndarray
    """
    key = jax.random.PRNGKey(rng_seed)
    main_key, dropout_key = jax.random.split(key, num=2)
    if analyzer.is_batchnorm:
        decoded, latent_means, latent_vars = analyzer.state.apply_fn(
            {"params": analyzer.state.params, "batch_stats": analyzer.state.batch_stats},
            data, main_key, train=False,
            rngs={"dropout": dropout_key},
        )
    else:
        decoded, latent_means, latent_vars = analyzer.state.apply_fn(
            {"params": analyzer.state.params},
            data, main_key, train=False,
            rngs={"dropout": dropout_key},
        )
    return decoded, latent_means


def validation_value(
    n_clusters, method_name, metric_function, test_frames,
    test_distance_matrix, supervised=True, num_outer=5, num_inner=10,
):
    """
    Estimate the self-consistency of a clustering method on raw test frames.

    Repeats clustering num_outer * num_inner times and returns the mean and
    propagated standard error of the chosen metric.

    Parameters
    ----------
    n_clusters : int or float
        Number of clusters or HDBSCAN epsilon.
    method_name : str
    metric_function : callable
    test_frames : jnp.ndarray
    test_distance_matrix : jnp.ndarray
    supervised : bool
        When True compare label sets; when False evaluate against raw frames.
    num_outer, num_inner : int

    Returns
    -------
    tuple (mean, std) of floats
    """
    mean_error = lambda err_arr: np.sqrt(np.sum(err_arr ** 2)) / err_arr.shape[0]
    final = []
    for j in range(num_outer):
        labels_true = _fit_predict(method_name, n_clusters, test_frames, test_distance_matrix)
        validation_set = []
        for i in range(num_inner):
            labels_pred = _fit_predict(method_name, n_clusters, test_frames, test_distance_matrix)
            if supervised:
                validation_set.append(metric_function(labels_true, labels_pred))
            else:
                validation_set.append(metric_function(test_frames, labels_pred))
        final.append([np.mean(validation_set), np.std(validation_set)])

    final = np.array(final)
    return (np.mean(final[:, 0]), mean_error(final[:, 1]))


def evaluate_pca_encoder_against_metric(
    n_clusters, method_name, metric_function, analyzer,
    test_frames, test_distance_matrix, rng_seed=2358,
    num_repetitions=5, supervised=True,
):
    """
    Score PCA and VAE encoder representations against a clustering metric.

    Clusters both PCA-projected and VAE-encoded test data, comparing each
    against the ground-truth labels derived from clustering the raw frames.

    Parameters
    ----------
    n_clusters : int or float
    method_name : str
    metric_function : callable
    analyzer : Analyzer
    test_frames : jnp.ndarray
    test_distance_matrix : jnp.ndarray
    rng_seed : int
    num_repetitions : int
    supervised : bool

    Returns
    -------
    pca_scores : np.ndarray, shape (num_repetitions,)
    enc_scores : np.ndarray, shape (num_repetitions,)
    """
    pca_scores = np.empty(num_repetitions)
    enc_scores = np.empty(num_repetitions)
    n_pca_components = min(analyzer.n_latents, 2000)

    for i in range(num_repetitions):
        labels_true = _fit_predict(method_name, n_clusters, test_frames, test_distance_matrix)

        pca = PCA(n_components=n_pca_components)
        pca.fit(analyzer.train_data)
        pca_latents = pca.transform(test_frames)
        _, enc_latents = nn_operate(analyzer, test_frames, rng_seed=rng_seed)

        if method_name in ["KMeans"]:
            pca_labels = CLUSTERING_METHODS[method_name](n_clusters=n_clusters).fit_predict(pca_latents)
            enc_labels = CLUSTERING_METHODS[method_name](n_clusters=n_clusters).fit_predict(enc_latents)
        elif method_name == "Agglomerative":
            pca_labels = CLUSTERING_METHODS[method_name](n_clusters=n_clusters, linkage="complete").fit_predict(pca_latents)
            enc_labels = CLUSTERING_METHODS[method_name](n_clusters=n_clusters, linkage="complete").fit_predict(enc_latents)
        elif method_name == "HDBScan":
            pca_labels = CLUSTERING_METHODS[method_name](cluster_selection_epsilon=n_clusters).fit_predict(np.array(pca_latents, copy=True))
            enc_labels = CLUSTERING_METHODS[method_name](cluster_selection_epsilon=n_clusters).fit_predict(np.array(enc_latents, copy=True))
        else:
            raise ValueError(f"Unknown method: {method_name}")

        if supervised:
            pca_scores[i] = metric_function(labels_true, pca_labels)
            enc_scores[i] = metric_function(labels_true, enc_labels)
        else:
            pca_scores[i] = metric_function(pca_latents, pca_labels)
            enc_scores[i] = metric_function(enc_latents, enc_labels)

    return pca_scores, enc_scores
