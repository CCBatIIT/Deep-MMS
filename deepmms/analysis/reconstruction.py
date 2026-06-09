"""
Analyzer: post-training analysis harness that loads a trained Experiment checkpoint
and exposes the model for reconstruction, latent encoding, and trajectory output.

Also provides violin_data for computing per-frame VAE vs PCA reconstruction RMSD,
and save_dcd for writing coordinate arrays to DCD files.
"""

import os
import json
import glob
import jax
import jax.numpy as jnp
import numpy as np
import mdtraj as md
import orbax.checkpoint
from sklearn.decomposition import PCA

from ..utils import printf
from ..data import load_and_align, train_test_split, mass_weights
from ..training.loss import atom_rmsd, give_weighted_rmsd_func
from ..training.optimizer import make_model_and_state
from ..training.trainer import Experiment
from ..models.vae import BatchNorm_VAE


def save_dcd(fname, traj_xyz):
    """
    Write a coordinate array to a DCD file.

    Coordinates are expected in nanometres and are multiplied by 10 on write
    to match the DCD convention of angstroms.

    Parameters
    ----------
    fname : str
        Output DCD path.
    traj_xyz : array, shape (n_frames, n_atoms*3) or (n_frames, n_atoms, 3)
        Coordinate data in nanometres.
    """
    if traj_xyz.shape[-1] != 3:
        traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
    with md.formats.DCDTrajectoryFile(fname, "w") as f:
        f.write(traj_xyz * 10)


class Analyzer(Experiment):
    """
    Analysis-mode subclass of Experiment that restores a trained checkpoint
    without starting a new training run.

    All training infrastructure from Experiment is initialised so that the
    model state is valid, but training methods are not called by __init__.

    Parameters
    ----------
    json_fn : str or dict
        JSON config file path or parameter dictionary.
    from_json_params : bool
        When True treat json_fn as a parameter dict.
    checkpoint_recency : int
        Index into the sorted list of checkpoint directories; -1 picks the
        most recently saved (highest-epoch) checkpoint.
    model_cls : type
        Model class (default: BatchNorm_VAE).
    """

    def __init__(self, json_fn, from_json_params=False, checkpoint_recency=-1, model_cls=BatchNorm_VAE):
        if not from_json_params:
            with open(json_fn, "r") as g:
                self.json_params = json.load(g)
        else:
            self.json_params = json_fn

        self.model_name = self.json_params["model_name"]
        self.n_latents = self.json_params["latent_dim"]
        test_slice = self.json_params["test_slice"]
        self.data_dir = os.path.join(
            self.json_params["save_dir"],
            f"{self.model_name}/",
            f"{self.n_latents:04d}_latents/",
            f"rpt_{test_slice}/",
        )
        self.is_batchnorm = self.json_params["is_batchnorm"]
        if (
            "data_dir" in self.json_params
            and self.json_params["data_dir"] is not None
        ):
            self.data_dir = self.json_params["data_dir"]

        nc_data_file = os.path.join(
            self.data_dir, f"model_{self.model_name}_{self.n_latents:04d}.nc"
        )
        self.rootgrp = self.establish_netcdf(nc_data_file, open_mode="r")

        checkpoint_dir_wc = os.path.join(self.data_dir, "checkpoint_managed", "*/")
        checkpoint_dir = sorted(glob.glob(checkpoint_dir_wc))[checkpoint_recency]

        assert all(
            os.path.exists(d) for d in [self.data_dir, nc_data_file, checkpoint_dir]
        )

        data_start = self.json_params["data_slice_start"]
        data_end = self.json_params["data_slice_end"]
        if data_end == "None":
            data_end = None

        c, coord_set = load_and_align(
            self.json_params["fname_dcd"],
            self.json_params["fname_topology"],
            self.json_params["atom_selection"],
            data_start,
            data_end,
        )
        printf(f"Coordinate set has shape {coord_set.shape}")
        num_samples, input_size = coord_set.shape

        self.train_data, self.test_data = train_test_split(coord_set, test_slice)
        self.batch_size = self.json_params["batch_size"]

        dropout_rates = self.json_params["dropout_rates"]
        learning_rate = self.json_params["learning_rate"]

        mass_sets = mass_weights(c)
        if "weight_model" in self.json_params:
            weight_model = self.json_params["weight_model"]
            assert weight_model in mass_sets.keys()
        else:
            weight_model = "Uniform_Heavy"
        printf(f"\t Using {weight_model=}")
        weights = jnp.array(mass_sets[weight_model])
        self.atom_rmsd_loss = give_weighted_rmsd_func(weights)

        self.model, self.state, self._step, self._evaluate = make_model_and_state(
            self, dropout_rates, coord_set, learning_rate, self.atom_rmsd_loss, model_cls=model_cls
        )
        self._step = jax.jit(self._step)
        self._evaluate = jax.jit(self._evaluate)

        self.state = orbax.checkpoint.PyTreeCheckpointer().restore(
            checkpoint_dir + "/default/", item=self.state
        )
        printf(f"Done restoring from {json_fn}")


def violin_data(analyzer, save_npy=True):
    """
    Compute per-frame reconstruction RMSD for VAE and PCA on the test set.

    Runs the trained VAE on test_data (no sampling noise; train=False),
    fits a PCA model on train_data with the same number of components as
    VAE latents, and records both unweighted and loss-weighted RMSD arrays.

    Optionally saves all four arrays as .npy files in the model's data_dir.

    Parameters
    ----------
    analyzer : Analyzer
        A fully restored Analyzer instance.
    save_npy : bool
        Whether to persist the four RMSD arrays to disk.

    Returns
    -------
    tuple of four jnp/np arrays
        (vae_rmsd, vae_loss_rmsd, pca_rmsd, pca_loss_rmsd)
        all in nanometres.
    """
    key = jax.random.PRNGKey(6969)
    main_key, dropout_key = jax.random.split(key, num=2)

    n_latents = analyzer.n_latents
    if analyzer.is_batchnorm:
        decoded, latent_means, latent_vars = analyzer.state.apply_fn(
            {"params": analyzer.state.params, "batch_stats": analyzer.state.batch_stats},
            analyzer.test_data, main_key, train=False,
            rngs={"dropout": dropout_key},
        )
    else:
        decoded, latent_means, latent_vars = analyzer.state.apply_fn(
            {"params": analyzer.state.params},
            analyzer.test_data, main_key, train=False,
            rngs={"dropout": dropout_key},
        )

    vae_rmsd = atom_rmsd(analyzer.test_data, decoded)
    vae_loss_rmsd = analyzer.atom_rmsd_loss(analyzer.test_data, decoded)

    pca = PCA(n_components=n_latents)
    pca.fit(analyzer.train_data)
    pca_test = pca.inverse_transform(pca.transform(analyzer.test_data))
    pca_rmsd = atom_rmsd(analyzer.test_data, pca_test)
    pca_loss_rmsd = analyzer.atom_rmsd_loss(analyzer.test_data, pca_test)

    if save_npy:
        for data, fn in zip(
            [vae_rmsd, vae_loss_rmsd, pca_rmsd, pca_loss_rmsd],
            ["VAE_RMSD.npy", "VAE_LOSS_RMSD.npy", "PCA_RMSD.npy", "PCA_LOSS_RMSD.npy"],
        ):
            out_path = os.path.join(analyzer.data_dir, fn)
            np.save(out_path, data)
            printf(f"    Wrote {out_path}")

    return vae_rmsd, vae_loss_rmsd, pca_rmsd, pca_loss_rmsd
