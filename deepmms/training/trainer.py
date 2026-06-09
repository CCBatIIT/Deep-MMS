"""
Experiment: the primary training harness for Deep-MMS molecular autoencoders.

Replaces HeavyAtom_NN_Experiment from pyscripts/heavy_atom_rmsd.py with an
identical interface and logic, but wired to the deepmms package internals.
The model class is injectable via model_cls so alternative architectures can
be dropped in without modifying this file.
"""

import os
import json
import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import mdtraj as md
import netCDF4 as nc
import orbax.checkpoint
from flax.training import orbax_utils
from datetime import datetime

from ..utils import printf
from ..data import Data_stream, load_and_align, train_test_split, mass_weights
from ..training.loss import give_weighted_rmsd_func
from ..training.optimizer import make_model_and_state
from ..models.vae import BatchNorm_VAE


class Experiment:
    """
    End-to-end training experiment for a molecular autoencoder.

    Loads trajectory data, builds the model and optimizer, manages NetCDF loss
    logging, and exposes methods for batched training, evaluation, and
    auto-stopping based on test-loss dynamics.

    Parameters
    ----------
    json_fn : str or dict
        Path to a JSON configuration file, or a pre-loaded parameter dictionary
        when from_json_params=True.
    make_dirs : bool
        Create output directories if they do not already exist.
    from_json_params : bool
        When True, treat json_fn as a dictionary of parameters rather than a path.
    model_cls : type
        Model class to instantiate; defaults to BatchNorm_VAE.  Must satisfy the
        MolecularAutoencoder interface.
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False, model_cls=BatchNorm_VAE):
        if not from_json_params:
            with open(json_fn, "r") as g:
                self.json_params = json.load(g)
        else:
            self.json_params = json_fn

        printf("Load Files")
        fname_dcd = self.json_params["fname_dcd"]
        fname_top = self.json_params["fname_topology"]
        self.n_latents = self.json_params["latent_dim"]
        test_slice = self.json_params["test_slice"]
        self.model_name = self.json_params["model_name"]
        save_dir = self.json_params["save_dir"]
        data_start = self.json_params["data_slice_start"]
        data_end = self.json_params["data_slice_end"]
        if data_end == "None":
            data_end = None
        self.batch_size = self.json_params["batch_size"]
        learning_rate = self.json_params["learning_rate"]
        dropout_rates = self.json_params["dropout_rates"]
        resume = self.json_params["resume_latest"]
        self.is_batchnorm = self.json_params["is_batchnorm"]

        printf("Establish Directories")
        model_dir = os.path.join(save_dir, f"{self.model_name}/")
        latent_dir = os.path.join(model_dir, f"{self.n_latents:04d}_latents/")
        self.data_dir = os.path.join(latent_dir, f"rpt_{test_slice}/")
        if "data_dir" in self.json_params and self.json_params["data_dir"] is not None:
            self.data_dir = self.json_params["data_dir"]

        if not os.path.isdir(self.data_dir) and make_dirs:
            os.makedirs(self.data_dir, exist_ok=True)

        printf("Load Data with MDTraj")
        c, coord_set = load_and_align(
            fname_dcd, fname_top,
            self.json_params["atom_selection"],
            data_start, data_end,
        )
        num_samples, input_size = coord_set.shape

        printf("Batch data")
        self.train_data, self.test_data = train_test_split(coord_set, test_slice)
        printf((self.train_data.shape, self.test_data.shape))

        printf("Model Init")
        printf("Calculate Mass Weighting Schemes")
        if (
            "weight_model" in self.json_params
            and "Uniform" not in self.json_params["weight_model"]
        ):
            weight_model = self.json_params["weight_model"]
            mass_sets = mass_weights(c)
            assert weight_model in mass_sets.keys()
            weights = jnp.array(mass_sets[weight_model])
        else:
            weight_model = "Uniform_Heavy"
            weights = jnp.ones(c.n_atoms)
        printf(f"\t Using {weight_model=}")

        printf("Make Loss Function")
        atom_rmsd_loss = give_weighted_rmsd_func(weights)

        printf("Make VAE")
        self.model, self.state, self._step, self._evaluate = make_model_and_state(
            self, dropout_rates, coord_set, learning_rate, atom_rmsd_loss, model_cls=model_cls
        )
        self._step = jax.jit(self._step)
        self._evaluate = jax.jit(self._evaluate)

        self.epoch = 0

        printf("Checkpointer")
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=2,
            save_interval_steps=self.json_params["checkpoint_interval"],
        )
        manager_dir = os.path.join(self.data_dir, "checkpoint_managed")
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
            manager_dir, self.orbax_checkpointer, options
        )

        printf("Batch Data")
        num_train = self.train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, self.batch_size)
        self.num_train_batches = num_complete_batches + bool(leftover)
        self.train_batches = Data_stream(
            self.n_latents, num_train, self.num_train_batches, self.batch_size, self.train_data
        )

        num_test = self.test_data.shape[0]
        num_complete_batches, leftover = divmod(num_test, self.batch_size)
        self.num_test_batches = num_complete_batches + bool(leftover)
        self.test_batches = Data_stream(
            self.n_latents, num_test, self.num_test_batches, self.batch_size, self.test_data
        )

        printf("Establish NCs")
        self.nc_data_file = os.path.join(
            self.data_dir, f"model_{self.model_name}_{self.n_latents:04d}.nc"
        )
        self.nc_checkpoint_file = os.path.join(
            self.data_dir, f"model_{self.model_name}_{self.n_latents:04d}_checkpoint.nc"
        )

        if resume:
            self.rootgrp = self.establish_netcdf(self.nc_data_file, open_mode="a")
            ckpt_fn = sorted(
                glob.glob(os.path.join(self.data_dir, "checkpoint_managed", "*/default/"))
            )[-1]
            self.epoch = int(ckpt_fn.split(os.sep)[-3]) + 1
            self.load_model_from_ckpt(ckpt_fn)
        else:
            self.rootgrp = self.establish_netcdf(self.nc_data_file)

        printf(f"Epoch {self.epoch}")
        printf(f"NC {self.rootgrp}")
        printf(f"Model {self.model}")
        printf("INITIALIZATION COMPLETE")

    def establish_netcdf(self, nc_filename, open_mode="w"):
        """
        Create or open a NetCDF file for logging per-epoch, per-batch losses.

        When open_mode is 'w' the file is created with Train/Test groups and
        unlimited epoch dimensions; when 'a' the existing file is appended to.

        Parameters
        ----------
        nc_filename : str
            Path to the NetCDF file.
        open_mode : str
            'w' to create, 'a' to append, 'r' to read-only.

        Returns
        -------
        netCDF4.Dataset
        """
        rootgrp = nc.Dataset(nc_filename, open_mode, format="NETCDF4")

        if open_mode == "w":
            traingrp = rootgrp.createGroup("Train")
            testgrp = rootgrp.createGroup("Test")

            traingrp.createDimension("epoch", None)
            traingrp.createDimension("batch", self.num_train_batches)

            testgrp.createDimension("epoch", None)
            testgrp.createDimension("batch", self.num_test_batches)

            for grp in [traingrp, testgrp]:
                rmsd_term = grp.createVariable("RMSD_Loss_Term", "f4", ("epoch", "batch"))
                rmsd_term.units = "Nanometer"
                grp.history = "Created" + time.ctime(time.time())

        return rootgrp

    def checkpoint_netcdf(self):
        """Write a snapshot of the current NetCDF loss log to the checkpoint file."""
        with nc.Dataset(self.nc_checkpoint_file, "w") as dst:
            for grp_name, grp in self.rootgrp.groups.items():
                dst.createGroup(grp_name)
                dst[grp_name].setncatts(grp.__dict__)
                for name, dimension in grp.dimensions.items():
                    dst[grp_name].createDimension(
                        name, (len(dimension) if not dimension.isunlimited() else None)
                    )
                for name, variable in grp.variables.items():
                    x = dst[grp_name].createVariable(name, variable.datatype, variable.dimensions)
                    dst[grp_name][name][:] = grp[name][:]
                    dst[grp_name][name].setncatts(grp[name].__dict__)

    def load_model_from_ckpt(self, chkpt_fn):
        """Restore model state from an Orbax checkpoint directory."""
        self.state = self.orbax_checkpointer.restore(chkpt_fn, item=self.state)

    def write_traj(self, identifier, traj_xyz, fname=None):
        """
        Write a coordinate array as a DCD trajectory file.

        Parameters
        ----------
        identifier : str or None
            Label used to construct a default filename when fname is None.
        traj_xyz : array, shape (n_frames, n_atoms*3) or (n_frames, n_atoms, 3)
            Coordinate data in nanometres.
        fname : str or None
            Explicit output path; overrides the auto-generated name.
        """
        if not fname:
            fname = os.path.join(
                self.data_dir, f"{identifier}_{self.model_name}{self.n_latents:04d}.dcd"
            )
        if traj_xyz.shape[-1] != 3:
            traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
        with md.formats.DCDTrajectoryFile(fname, "w") as f:
            f.write(traj_xyz * 10)

    def train_on_batches(self, batch_set):
        """
        Run one epoch of gradient updates over all batches in batch_set,
        then save a checkpoint.

        Parameters
        ----------
        batch_set : Data_stream
            Iterable that yields mini-batches for one epoch.
        """
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            batch = next(f)
            root_key = jax.random.PRNGKey(self.epoch)
            main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
            self.state, _ = self._step(self.state, batch, main_key, dropout_key)

        save_args = orbax_utils.save_args_from_target(self.state)
        self.checkpoint_manager.save(
            self.epoch, self.state, save_kwargs={"save_args": save_args}
        )
        if self.epoch > 0 and self.epoch % 100 == 0:
            self.checkpoint_netcdf()

    def eval_batches(self, batch_set):
        """
        Evaluate loss on all batches in batch_set and return a (n_batches, 1) array.

        Parameters
        ----------
        batch_set : Data_stream

        Returns
        -------
        jnp.ndarray, shape (n_batches, 1)
        """
        vals = jnp.empty((batch_set.num_batches, len(self.rootgrp["Train"].variables)))
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            batch = next(f)
            root_key = jax.random.PRNGKey(self.epoch)
            main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
            rmsd_term = self._evaluate(self.state, batch, main_key, dropout_key)
            vals = vals.at[i, :].set([rmsd_term])
        return vals

    def evaluate_loss(self):
        """
        Evaluate and log RMSD loss on both train and test sets for the current epoch.

        Returns
        -------
        list of two jnp.ndarrays
            [train_vals, test_vals], each shape (n_batches, 1).
        """
        guide = {"Train": self.train_batches, "Test": self.test_batches}
        most_recent_results = []
        for grp_key in ["Train", "Test"]:
            vals = self.eval_batches(guide[grp_key])
            self.rootgrp[grp_key].variables["RMSD_Loss_Term"][self.epoch, :] = vals[:, 0]
            most_recent_results.append(vals)
        return most_recent_results

    def verbose_print(self, last_result, epoch_time, verbose=True):
        """
        Conditionally print per-epoch loss summary.

        Parameters
        ----------
        last_result : list
            Output of evaluate_loss().
        epoch_time : timedelta
            Wall-clock duration of the epoch.
        verbose : bool or int
            True for every epoch; int N to print every N epochs; False to suppress.
        """
        if verbose is True:
            printf(f"Epoch {self.epoch:6d} took: {epoch_time}")
            printf(
                f"\t\tRMSD_Term\tTrain={jnp.mean(last_result[0][:,0]):.3E}"
                f"\tTest={jnp.mean(last_result[1][:,0]):.3E}"
            )
        elif type(verbose) is int and self.epoch % verbose == 0:
            printf(f"Epoch {self.epoch:6d} took: {epoch_time}")
            printf(
                f"\t\tRMSD_Term\tTrain={jnp.mean(last_result[0][:,0]):.3E}"
                f"\tTest={jnp.mean(last_result[1][:,0]):.3E}"
            )

    def train_n_epochs(self, n_epochs, verbose=True):
        """
        Train for exactly n_epochs gradient epochs.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train.
        verbose : bool or int
            Verbosity control forwarded to verbose_print.

        Returns
        -------
        int
            Current epoch after training.
        """
        while self.epoch < n_epochs:
            epoch_start = datetime.now()
            self.train_on_batches(self.train_batches)
            self.verbose_print(
                self.evaluate_loss(), datetime.now() - epoch_start, verbose=verbose
            )
            self.epoch += 1
        return self.epoch

    def train_to_auto_stop(self, cutoff_epoch=100000, verbose=True):
        """
        Train until overtraining is detected or cutoff_epoch is reached.

        Early stopping fires when the last 50 test-loss epochs simultaneously
        show a positive slope, a mean above 102.5% of train loss, and low variance.

        Parameters
        ----------
        cutoff_epoch : int
            Hard upper bound on the number of epochs.
        verbose : bool or int
            Verbosity forwarded to verbose_print.

        Returns
        -------
        int
            Current epoch after training.
        """
        should_early_stop = False

        while not should_early_stop and self.epoch < cutoff_epoch:
            epoch_start = datetime.now()
            self.train_on_batches(self.train_batches)
            self.verbose_print(
                self.evaluate_loss(), datetime.now() - epoch_start, verbose=verbose
            )
            self.epoch += 1

            last_50_train = np.mean(
                self.rootgrp["Train"]["RMSD_Loss_Term"][-50:, :], axis=1
            )
            last_50_test = np.mean(
                self.rootgrp["Test"]["RMSD_Loss_Term"][-50:, :], axis=1
            )

            test_rising = np.polyfit(np.arange(50), last_50_test, 1)[0] > 0.0001
            test_greater_than_train = np.mean(last_50_test) > 1.025 * np.mean(last_50_train)
            stable_vals = np.std(last_50_test) < 1

            if all([test_rising, test_greater_than_train, stable_vals]):
                should_early_stop = True
                printf("Auto Stopping Was Detected...")
                printf(
                    f"    Test_Rising = {np.polyfit(np.arange(50), last_50_test, 1)[0]:0.3f}"
                    " is greater than 0.0001"
                )
                printf(
                    f"    Test_Great_Than_Train = {np.mean(last_50_test)} is greater than"
                    f" {1.025 * np.mean(last_50_train):0.3f}"
                )
                printf(f"    Stable_Vals = {np.std(last_50_test):0.3f}")
        return self.epoch

    def MAIN_train(self, n_epochs=1000, cutoff_epoch=None, verbose=True):
        """
        Run the full training protocol: n_epochs of standard training followed by
        auto-stopping up to cutoff_epoch.

        Parameters
        ----------
        n_epochs : int
            Epochs of standard training before auto-stop phase begins.
        cutoff_epoch : int or None
            Maximum epoch for auto-stop phase; falls back to json_params['max_epoch'].
        verbose : bool or int
            Verbosity forwarded through training methods.

        Returns
        -------
        int
            Final epoch reached.
        """
        if cutoff_epoch is None:
            cutoff_epoch = self.json_params["max_epoch"]
        self.train_n_epochs(n_epochs, verbose=verbose)
        self.train_to_auto_stop(cutoff_epoch=cutoff_epoch, verbose=verbose)
        return self.epoch
