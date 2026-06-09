"""
Latent-space perturbation analysis: systematically vary each latent dimension
while holding all others at their mean to generate perturbation trajectories.

Provides nn_operate for running a trained Analyzer in inference mode and
run_perturbation_analysis for the full sweep.
"""

import os
import jax
import jax.numpy as jnp


def nn_operate(analyzer, data, rng_seed=69420):
    """
    Run a deterministic forward pass through the trained model.

    Parameters
    ----------
    analyzer : Analyzer
        A fully restored Analyzer instance.
    data : jnp.ndarray, shape (n_frames, n_features)
        Input coordinate frames.
    rng_seed : int
        RNG seed for all random keys.

    Returns
    -------
    decoded : jnp.ndarray, shape (n_frames, n_features)
    latent_means : jnp.ndarray, shape (n_frames, n_latents)
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


def run_perturbation_analysis(analyzer, work_dir, n_perturbation=2001, rng_seed=893467):
    """
    For each latent dimension, generate a trajectory sweeping that dimension from
    mean − 5σ to mean + 5σ while all other dimensions are held at their mean.

    Also saves the raw test frames and their reconstructions to DCD files.

    Parameters
    ----------
    analyzer : Analyzer
        A fully restored Analyzer instance.
    work_dir : str
        Directory where DCD files and the perturbation log are written.
    n_perturbation : int
        Number of points in the linspace sweep per latent dimension.
    rng_seed : int
        Starting RNG seed; incremented across latent dimensions.

    Returns
    -------
    log_fn : str
        Path to the written log file.
    """
    import datetime

    os.makedirs(work_dir, exist_ok=True)
    log_fn = os.path.join(work_dir, "perturbation_log.txt")
    logfile = open(log_fn, "w")
    logfile.write(f"{datetime.datetime.now()} - Begin {analyzer.model_name}\n")

    test_frames = analyzer.test_data
    decoded, latents = nn_operate(analyzer, test_frames)
    latent_means = jnp.mean(latents, axis=0)
    latent_stds = jnp.std(latents, axis=0)

    logfile.write(f"{datetime.datetime.now()} - Obtain Perturbation Data\n")
    perturb_space = jnp.linspace(
        latent_means - 5 * latent_stds,
        latent_means + 5 * latent_stds,
        n_perturbation,
        axis=0,
    )

    for i in range(perturb_space.shape[1]):
        key = jax.random.PRNGKey(rng_seed)
        main_key, dropout_key = jax.random.split(key, num=2)

        this_perturbation = perturb_space
        for j in range(perturb_space.shape[1]):
            if j != i:
                this_perturbation = this_perturbation.at[:, j].set(
                    jnp.repeat(latent_means[j], n_perturbation)
                )
        logfile.write(f"\t {datetime.datetime.now()} - Latent {i=}\n")

        params_dict = {"params": analyzer.state.params}
        if analyzer.is_batchnorm:
            params_dict["batch_stats"] = analyzer.state.batch_stats
        decoded_perturbed = analyzer.state.apply_fn(
            params_dict,
            this_perturbation, main_key, train=False,
            rngs={"dropout": dropout_key},
            method=analyzer.model.decode,
        )
        out_dcd = os.path.join(work_dir, f"{analyzer.model_name}_pLatent{i:04d}.dcd")
        analyzer.write_traj(None, decoded_perturbed, fname=out_dcd)
        logfile.write(f"\t {datetime.datetime.now()} - Wrote data for Latent {i=}\n")
        rng_seed += j

    analyzer.write_traj(
        None, test_frames,
        fname=os.path.join(work_dir, f"{analyzer.model_name}_test_data.dcd"),
    )
    analyzer.write_traj(
        None, decoded,
        fname=os.path.join(work_dir, f"{analyzer.model_name}_test_recon.dcd"),
    )
    logfile.close()
    return log_fn
