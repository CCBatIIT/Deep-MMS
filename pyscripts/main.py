import jax, optax, sys, os, json, pickle, time, glob
from datetime import datetime
import orbax.checkpoint
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import mdtraj as md
import netCDF4 as nc
import jax.numpy as jnp
from flax.training import train_state, orbax_utils
from typing import Any

#Neural Network Models
try:
    from NN_models import *
except:
    from .NN_models import *

jax.config.update("jax_enable_x64", True)
printf = lambda x : print(f"{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}//{x}", flush=True)

printf(jax.print_environment_info())
printf(f'Default JAX backend is {jax.default_backend()}')

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (np.exp(float(i) / float(n_vals-1)) - 1.0) / (np.e - 1.0) for i in range(n_vals)]

    
class NN_Experiment():
    """
    Methods:
    establish_netcdf(self, nc_filename, open_mode='w')
    checkpoint_netcdf(self):
    load_model_from_ckpt(self, chkpt_fn):
    write_traj(self, identifier, traj_xyz): #(n conf, n_atoms*3) OR (n conf, n_atoms, 3)
    write_decoded_traj(self, idfn=None):
    eval_batches(self, batch_set, eval_function, potential_coefficient):
    eval_rmsd_only(self, rng):
    eval_losses(self, rng, potential_coefficient):
    train_batches_on_step(self, batch_set, step_function, potential_coefficient):
    train_nepochs_on_rmsd_wo_reporting_potential(self, num_rmsd_epochs):
    train_rmsd_to_lowest_wo_reporting_potential(self, max_epoch=100000):
    train_nepochs_on_rmsd(self, num_rmsd_epochs):
    train_rmsd_threshold(self, nm_cutoff=0.1, num_move_ave=10):
    train_rmsd_to_lowest(self, num_move_ave=20, max_epoch=100000):
    train_potential_threshold(self, potential_coefficient=1, potential_threshold=1e-3, num_move_ave=10):
    train_scaling_potential(self):
    train_summation_threshold(self, potential_coefficient=1, potential_threshold=1e-3, num_move_ave=10):
    MAIN_train_rmsd_only(self, n_rmsd=1000, cutoff_epoch=100000):
    MAIN_train_rmsd_only_wo_reporting_potential(self, n_rmsd=1000, cutoff_epoch=100000):
    MAIN_scale_and_train_potential(self, n_rmsd=500, cutoff_epoch=200000):

    """
    def __init__(self, json_fn, make_dirs=True, from_json_params=False):
        """

        INIT PARAMETERS:
            json_fn: string or dict: if string, set from_json_params to False, and provide the json file name as a string
                                     if dict, set from_json_params to True, and provide the parameters as json_fn
            make_dirs: bool: 
        
        JSON PARAMETERS:
            #Explain all json params as they are assinged
        
        fname_dcd = #Trajectory file from which to obtain train and test sets
        
        fname_prmtop = #Parameter file for energy functions
        
        fname_pdb = #PDB file for topology, only if fname_prmtop is an xml file not a prmtop file
        
        save_dir = #Directory to build outputs
        
        data_dir = #Override the automatic saving with this directory
        
        max_epoch = #Cutoff epoch
        
        latent_dim = #Number of latent dimensions
        
        test_slice = #index of the data from which to derive the 20/80 test set split
        
        data_slice_start = #initial index of data from which to derive test and train sets
        
        data_slice_end = #final index of data from which to derive test and train sets
        
        model_name = #name for file-keeping purposes
        
        batch_size = #size of batches (must be an even divisor of total test and total train frames)
        
        learning_rate = #Learning rate for the adam optimizer
        
        dropout_rates = #Dropout rates for the hideen layers - also determines the quantity of layers
        
        potential_threshold = #IDK
        
        resume_latest = #Whether to resume a previous training or not
        
        report_potential = #whether to calculate the potential energy or not
        
        checkpoint_interval = #Interval of epochs to checkpoint the Neural Network
        
        scale_factor = #I forgor

        atom_select = #Selection string for mdtraj topology
        """
        #Get the information from the json file
        if not from_json_params:
            with open(json_fn, 'r') as g:
                self.json_params = json.load(g)
        else:
            #To use json parameters that are already in memeory, not just the file name, make from_json_params True and pass the params instead of the json_fn
            self.json_params = json_fn
        
        #Files
        printf('Load Files')
        fname_dcd = self.json_params["fname_dcd"]
        fname_prmtop = self.json_params["fname_prmtop"]
        fname_pdb = self.json_params["fname_pdb"]
        self.n_latents = self.json_params["latent_dim"] #number of latents
        test_slice = self.json_params["test_slice"] #int zero to four inclusive for 20/80 split of test and train
        model_name = self.json_params["model_name"] #string for the model name
        save_dir = self.json_params["save_dir"] #with trailing slash:
        data_start, data_end = self.json_params["data_slice_start"], self.json_params["data_slice_end"] #Slice of data
        self.batch_size = self.json_params["batch_size"]
        learning_rate = self.json_params["learning_rate"]
        dropout_rates = self.json_params["dropout_rates"]
        self.model_name = model_name
        resume = self.json_params["resume_latest"]
        self.report_potential = self.json_params["report_potential"]
        self.scale_factor = self.json_params["scale_factor"]

        printf('Establish Directories')
        #Establish Model Directory
        model_dir = os.path.join(save_dir, f'{self.model_name}/')
        if not os.path.isdir(model_dir) and make_dirs:
            os.mkdir(model_dir)
        
        #Establish Data Directory
        if self.json_params["data_dir"] == 'None':
            latent_dir = os.path.join(model_dir, f'{self.n_latents:04d}_latents/')
            self.data_dir = os.path.join(latent_dir, f'rpt_{test_slice}_{self.scale_factor}/')
            if not os.path.isdir(latent_dir) and make_dirs:
                os.mkdir(latent_dir)
        else:
            self.data_dir = self.json_params["data_dir"]
        
        if not os.path.isdir(self.data_dir) and make_dirs:
            os.mkdir(self.data_dir)
        
        if data_end == 'None':
            data_end = None
        
        #Load and Align
        if self.report_potential:
            raise Exception('Potential is currently FUUUUUUUUUUUUCKed')
            if fname_prmtop.endswith('prmtop'):
                from pyscripts.jax_MM_energy import AmberPRMTOPHandler
                self.energy_handler = AmberPRMTOPHandler(fname_prmtop)
            elif fname_prmtop.endswith('xml'):
                from pyscripts.jax_MM_energy import OpenMMSystemHandler
                self.energy_handler = OpenMMSystemHandler(fname_prmtop)
                raise Exception('OpenMMSystemHandler likely has energetics calculation errors')
            

        #Import loss functions after declaration of gas_fun
        global loss
        if not self.report_potential:
            try:
                import loss_rmsd_only as loss
            except:
                from . import loss_rmsd_only as loss
        elif self.report_potential:
            try:
                import loss as loss
            except:
                from . import loss as loss
        
        printf('Load Data with MDTraj')
        c = md.load(self.json_params['fname_dcd'], top=self.json_params['fname_prmtop'])
        if self.json_params['atom_select'] != 'all':
            c = c.atom_slice(c.topology.select(self.json_params['atom_select']))
        c = c.superpose(c) # FEED IN ALIGNED DATA
        coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))[data_start:data_end]
        num_samples, input_size = coord_set.shape

        #make test and train sets
        printf('Batch data')
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]
        printf((self.train_data.shape, self.test_data.shape))
        
        #Initialize Model
        printf('Model Init')
        #make hidden layers
        n_hidden = len(dropout_rates)
        hidden_layers = [int(val) for val in geometric_distribution(input_size, self.n_latents, n_hidden)]
        self.model = BatchNorm_VAE(input_size=input_size, latents=self.n_latents, hidden_layers=hidden_layers, dropout_rates=dropout_rates)
        rng_init = jax.random.PRNGKey(self.n_latents)
        main_key, params_key, dropout_key = jax.random.split(key=rng_init, num=3)
        
        variables = self.model.init(params_key, coord_set, rng_init, train=False)
        params = variables['params']
        batch_stats = variables['batch_stats']
        
        class TrainState(train_state.TrainState):
            batch_stats: Any
            key: jax.Array
            
        n_updates_per_epoch = 8000//self.batch_size
        #schedule = optax.schedules.cosine_decay_schedule(learning_rate, 100*n_updates_per_epoch, 0.25)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=params,
                                       batch_stats=batch_stats,
                                       key=dropout_key,
                                       tx=optax.adam(learning_rate=learning_rate))
        
        self.epoch = 0
        self.current_potential_coefficient = 0

        #Checkpointer
        printf('Checkpointer')
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=self.json_params['checkpoint_interval'])
        manager_dir = os.path.join(self.data_dir, 'checkpoint_managed')
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(manager_dir, self.orbax_checkpointer, options)

        
        printf('Batch Data')
        num_train = self.train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, self.batch_size)
        self.num_train_batches = num_complete_batches + bool(leftover)
        self.train_batches = Data_stream(self.n_latents, num_train, self.num_train_batches, self.batch_size, self.train_data)
        
        num_test = self.test_data.shape[0]
        num_complete_batches, leftover = divmod(num_test, self.batch_size)
        self.num_test_batches = num_complete_batches + bool(leftover)
        self.test_batches = Data_stream(self.n_latents, num_test, self.num_test_batches, self.batch_size, self.test_data)
        
        #Initialize data_storage
        printf('Establish NCs')
        self.nc_data_file = os.path.join(self.data_dir, f'model_{self.model_name}_{self.n_latents:04d}.nc')
        self.nc_checkpoint_file = os.path.join(self.data_dir, f'model_{self.model_name}_{self.n_latents:04d}_checkpoint.nc')
        
        #Handle Resuming a Training or not
        if resume:
            self.rootgrp = self.establish_netcdf(self.nc_data_file, open_mode='a')
            ckpt_fn = sorted(glob.glob(manager_dir + "/*/default/"))[-1]
            self.epoch = int(ckpt_fn.split(os.sep)[-3]) + 1
            if self.report_potential:
                self.current_potential_coefficient = self.rootgrp['Train'].variables['Potential Coefficient'][-1]
            self.load_model_from_ckpt(ckpt_fn)
        else:
            self.rootgrp = self.establish_netcdf(self.nc_data_file)
        printf(f'Epoch {self.epoch}')
        printf(f'NC {self.rootgrp}')
        printf(f'Model {self.model}')
        printf(f"INITIALIZATION COMPLETE for {self.n_latents} Latents, {self.scale_factor} Scale")

    def establish_netcdf(self, nc_filename, open_mode='w'):
        rootgrp = nc.Dataset(nc_filename, open_mode, format='NETCDF4')
        
        if open_mode == 'w':
            traingrp = rootgrp.createGroup('Train')
            testgrp = rootgrp.createGroup('Test')
        
            traingrp.createDimension('epoch', None)
            traingrp.createDimension('batch', self.num_train_batches)
        
            testgrp.createDimension('epoch', None)
            testgrp.createDimension('batch', self.num_test_batches)
        
            for grp in [traingrp, testgrp]:
                rmsd = grp.createVariable('RMSD', 'f4', ('epoch', 'batch',))
                rmsd.units = "Nanometer"
                if self.report_potential:
                    pot = grp.createVariable('Potential', 'f8', ('epoch', 'batch',))
                    pot.units = "KJ/mol"
                    summ = grp.createVariable('Summation', 'f8', ('epoch', 'batch',))
                    summ.units = 'Unitless'
                true_loss = grp.createVariable('Loss', 'f4', ('epoch', 'batch',))
                true_loss.units = 'Unitless'
                grp.history = "Created" + time.ctime(time.time())
            
            if self.report_potential:
                coef = traingrp.createVariable('Potential Coefficient', 'f8', ('epoch',))
                coef.units = 'Unitless'
            
        return rootgrp

    def checkpoint_netcdf(self):
        with nc.Dataset(self.nc_checkpoint_file, "w") as dst:
            for grp_name, grp in self.rootgrp.groups.items():
                dst.createGroup(grp_name)
                # copy global attributes all at once via dictionary
                dst[grp_name].setncatts(grp.__dict__)
                # copy dimensions
                for name, dimension in grp.dimensions.items():
                    dst[grp_name].createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
                # copy all file data
                for name, variable in grp.variables.items():
                    x = dst[grp_name].createVariable(name, variable.datatype, variable.dimensions)
                    dst[grp_name][name][:] = grp[name][:]
                    # copy variable attributes all at once via dictionary
                    dst[grp_name][name].setncatts(grp[name].__dict__)
    
    
    def load_model_from_ckpt(self, chkpt_fn):
        self.state = self.orbax_checkpointer.restore(chkpt_fn, item=self.state)
    
    def write_traj(self, identifier, traj_xyz): #(n conf, n_atoms*3) OR (n conf, n_atoms, 3)
        fname = self.data_dir + f'{identifier}_{self.model_name}{self.n_latents:04d}.dcd'
        
        if traj_xyz.shape[-1] != 3:
            traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
        
        with md.formats.DCDTrajectoryFile(fname, 'w') as f:
            f.write (traj_xyz*10) #*10 because mdtraj loads data in nm but saves it in angstrom
    
    def eval_batches(self, batch_set, eval_function, potential_coefficient):
        vals = jnp.empty((batch_set.num_batches, self.batch_size))
        f = iter(batch_set)
        
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            root_key = jax.random.PRNGKey(self.epoch)
            main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
            recon = self.state.apply_fn({'params':self.state.params, 'batch_stats':self.state.batch_stats}, batch, root_key, train=False, rngs={'dropout': dropout_key})[0]
            #Eval Batch
            vals = vals.at[i, :].set(eval_function(batch, recon, potential_coefficient))
        return vals
        
    def eval_rmsd_only(self):
        self.rootgrp['Train'].variables['RMSD'][self.epoch, :] = jnp.mean(self.eval_batches(self.train_batches, loss.atom_rmsd, None), axis=1)
        self.rootgrp['Test'].variables['RMSD'][self.epoch, :] = jnp.mean(self.eval_batches(self.test_batches, loss.atom_rmsd, None), axis=1)
        most_recent_results = []
        for grp in (self.rootgrp['Train'], self.rootgrp['Test']):
            most_recent_results.append(grp.variables['RMSD'][self.epoch, :].mean())
        return most_recent_results
        
    def eval_losses(self, potential_coefficient):
        """
        Shape of return is train_rmsd, train_potential, train_summ, test_rmsd, test_potential, test_summ, lambda
        """
        #After all batches seen this epoch
        self.rootgrp['Train'].variables['RMSD'][self.epoch, :] = jnp.mean(self.eval_batches(self.train_batches, loss.atom_rmsd, None), axis=1)
        self.rootgrp['Test'].variables['RMSD'][self.epoch, :] = jnp.mean(self.eval_batches(self.test_batches, loss.atom_rmsd, None), axis=1)
        
        self.rootgrp['Train'].variables['Potential'][self.epoch, :] = jnp.mean(self.eval_batches(self.train_batches, loss.scaled_pot_enr_diff, None), axis=1)
        self.rootgrp['Test'].variables['Potential'][self.epoch, :] = jnp.mean(self.eval_batches(self.test_batches, loss.scaled_pot_enr_diff, None), axis=1)
        
        self.rootgrp['Train'].variables['Summation'][self.epoch, :] = jnp.mean(self.eval_batches(self.train_batches, loss.different_summation_loss, potential_coefficient), axis=1)
        self.rootgrp['Test'].variables['Summation'][self.epoch, :] = jnp.mean(self.eval_batches(self.test_batches, loss.different_summation_loss, potential_coefficient), axis=1)

        self.rootgrp['Train'].variables['Potential Coefficient'][self.epoch] = potential_coefficient
        
        most_recent_results = []
        for grp in (self.rootgrp['Train'], self.rootgrp['Test']):
            for variable in ['RMSD', 'Potential', 'Summation']:
                most_recent_results.append(grp.variables[variable][self.epoch, :].mean())
        most_recent_results.append(self.rootgrp['Train'].variables['Potential Coefficient'][self.epoch])
        
        return most_recent_results
    
    def train_batches_on_step(self, batch_set, step_function, potential_coefficient):
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            #Train Batch
            root_key = jax.random.PRNGKey(self.epoch)
            main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
            self.state = step_function(self.state, batch, root_key, potential_coefficient, dropout_key)
        #After ANY EPOCH
        save_args = orbax_utils.save_args_from_target(self.state)
        self.checkpoint_manager.save(self.epoch, self.state, save_kwargs={'save_args': save_args})
        if self.epoch > 0 and self.epoch % 100 == 0:
            self.checkpoint_netcdf()
        
    def train_nepochs_on_rmsd_wo_reporting_potential(self, num_rmsd_epochs, verbose=True):
        while self.epoch < num_rmsd_epochs:
            epoch_start = datetime.now()
            #Training
            self.train_batches_on_step(self.train_batches, loss.rmsd_log_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_rmsd_only()
            epoch_end = datetime.now() - epoch_start
            if verbose:
                printf(f"epoch {self.epoch} atom_rmsd_nm {'%.4E'%last_loss[0]} {'%.4E'%last_loss[1]} Time: {epoch_end}")
            self.epoch += 1
        return self.epoch
    
    def train_rmsd_to_lowest_wo_reporting_potential(self, max_epoch=100000, verbose=True):
        """Train on the RMSD until overtraining is deteceted
            Stop training if the RMSD_loss of the test set is rising
        """
        should_early_stop = False

        while not should_early_stop and self.epoch < max_epoch:
            epoch_start = datetime.now()
            #Training
            self.train_batches_on_step(self.train_batches, loss.rmsd_log_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_rmsd_only()
            epoch_end = datetime.now() - epoch_start
            if verbose:
                printf(f"epoch {self.epoch} atom_rmsd_nm {'%.4E'%last_loss[0]} {'%.4E'%last_loss[1]} Time: {epoch_end}")
            self.epoch += 1
            
            #Evaluate if the last 50 epoch test loss is gr_or_eq than 0.1pm/epoch (incur 5pm=0.05 Angstrom loss)
            test_rising = np.polyfit(np.arange(50), np.mean(self.rootgrp['Test']['RMSD'][-50:, :], axis=1), 1)[0] > 0.00005
            
            #Evaluate if the average test value of the last 50 epochs is greater than 2.5% of the train value
            test_greater_than_train = np.mean(self.rootgrp['Test']['RMSD'][-50:, :]) > 1.025*np.mean(self.rootgrp['Train']['RMSD'][-50:, :])
            
            #Evaluate if the loss values are stable or not (last 50 epoch std < 1 angstrom)
            stable_vals = np.std(self.rootgrp['Test']['RMSD'][-50:, :]) < 1
            
            #If all are true, invoke early stopping
            if False not in [test_rising, test_greater_than_train, stable_vals]:
                should_early_stop = True
        
        return self.epoch
    
    def train_nepochs_on_rmsd(self, num_rmsd_epochs):
        """
        Train on the RMSD function alone, potential coefficient is zero
        """
        
        while self.epoch < num_rmsd_epochs:
            epoch_start = datetime.now()
            #Training
            self.train_batches_on_step(self.train_batches, loss.rmsd_log_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_losses(self.current_potential_coefficient)
            epoch_end = datetime.now() - epoch_start
            print('epoch', self.epoch,
                  'atom_rmsd_nm', '%.4E'%last_loss[0], '%.4E'%last_loss[3],
                  'dPotEnr', '%.4E'%last_loss[1], '%.4E'%last_loss[4],
                  'Summation', '%.4E'%last_loss[2], '%.4E'%last_loss[5],
                  'L=%.4E'%last_loss[6], 'Time:', epoch_end)
            
            self.epoch += 1
        return self.epoch

    def train_rmsd_threshold(self, nm_cutoff=0.1, num_move_ave=10):
        """Train on the RMSD until a nm_cutoff is reached of the last num_mov_ave epochs
            Stop training if the threshold is reached, or the RMSD_loss of the test set is rising
        """
        rmsd_above_threshold = True
        test_rmsd_decreasing = True

        while rmsd_above_threshold and test_rmsd_decreasing:
            #Training
            self.train_batches_on_step(self.train_batches, loss.rmsd_log_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_losses(self.current_potential_coefficient)
            print('epoch', self.epoch,
                  'atom_rmsd_nm', '%.4E'%last_loss[0], '%.4E'%last_loss[3],
                  'dPotEnr', '%.4E'%last_loss[1], '%.4E'%last_loss[4],
                  'Summation', '%.4E'%last_loss[2], '%.4E'%last_loss[5],
                  'L=%.4E'%last_loss[6])
            self.epoch += 1
            rmsd_above_threshold = self.rootgrp['Train']['RMSD'][-num_move_ave:, :].mean() > nm_cutoff or self.rootgrp['Test']['RMSD'][-num_move_ave:, :].mean() > nm_cutoff
            test_rmsd_decreasing = self.rootgrp['Test']['RMSD'][-2*num_move_ave:-num_move_ave, :].mean() > self.rootgrp['Test']['RMSD'][-num_move_ave:, :].mean()

        if not rmsd_above_threshold:
            print('RMSD Threshold Run - Break Reason - Threshold Reached')
        if not test_rmsd_decreasing:
            print('RMSD Threshold Run - Break Reason - Test Set Loss Increasing')
        
        return self.epoch

    def train_rmsd_to_lowest(self, num_move_ave=20, max_epoch=100000):
        """Train on the RMSD until overtraining is deteceted
            Stop training if the RMSD_loss of the test set is rising
        """
        test_rmsd_decreasing = True

        while test_rmsd_decreasing and self.epoch < max_epoch:
            #Training
            self.train_batches_on_step(self.train_batches, loss.rmsd_log_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_losses(self.current_potential_coefficient)
            print('epoch', self.epoch,
                  'atom_rmsd_nm', '%.4E'%last_loss[0], '%.4E'%last_loss[3],
                  'dPotEnr', '%.4E'%last_loss[1], '%.4E'%last_loss[4],
                  'Summation', '%.4E'%last_loss[2], '%.4E'%last_loss[5],
                  'L=%.4E'%last_loss[6])
            self.epoch += 1
            test_rmsd_decreasing = self.rootgrp['Test']['RMSD'][-2*num_move_ave:-num_move_ave, :].mean() > self.rootgrp['Test']['RMSD'][-num_move_ave:, :].mean()

        if not test_rmsd_decreasing:
            print('RMSD Lowest Run - Break Reason - Test Set Loss Increasing')
        
        return self.epoch
    
    def train_potential_threshold(self, potential_coefficient=1, potential_threshold=1e-3, num_move_ave=10):
        self.current_potential_coefficient = potential_coefficient
        potential_above_threshold = True
        test_potential_decreasing = True

        while potential_above_threshold or test_potential_decreasing:
            # Training
            self.train_batches_on_step(self.train_batches, loss.potential_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_losses(self.current_potential_coefficient)
            print('epoch', self.epoch,
                  'atom_rmsd_nm', '%.4E'%last_loss[0], '%.4E'%last_loss[3],
                  'dPotEnr', '%.4E'%last_loss[1], '%.4E'%last_loss[4],
                  'Summation', '%.4E'%last_loss[2], '%.4E'%last_loss[5],
                  'L=%.4E'%last_loss[6])
            self.epoch += 1
            potential_above_threshold = (self.rootgrp['Train']['Potential'][-num_move_ave:, :].mean() > potential_threshold or
                                         self.rootgrp['Test']['Potential'][-num_move_ave:, :].mean() > potential_threshold)
            test_potential_decreasing = self.rootgrp['Test']['Potential'][-2*num_move_ave:-num_move_ave, :].mean() > self.rootgrp['Test']['Potential'][-num_move_ave:, :].mean()

        if not potential_above_threshold:
            print('Potential Threshold Run - Break Reason - Threshold Reached')
        if not test_potential_decreasing:
            print('Potential Threshold Run - Break Reason - Test Set Loss Increasing')
        
        return self.epoch

    def train_scaling_potential(self):
        """Scale the potential in by frequently changing the coefficient to make potential equal to rmsd"""
        # Every five epochs choose lambda as min(1, max(1.01*lambda[-1], RMSD/NSD))
        print('Train Scaling Potential')
        while self.current_potential_coefficient < 1:
            # Training
            self.train_batches_on_step(self.train_batches, loss.different_summation_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_losses(self.current_potential_coefficient)
            #Choose the smaller between 1 and x, where x is the larger of (RMSD/Potential, 1% increase in the current coefficient)
            self.current_potential_coefficient = np.min((1, np.max((self.scale_factor * self.current_potential_coefficient, (jnp.nanmean(self.rootgrp['Train'].variables['RMSD'][-5:, :].filled()) / jnp.nanmean(self.rootgrp['Train'].variables['Potential'][-5:, :].filled()))))))
            print('epoch', self.epoch,
                  'atom_rmsd_nm', '%.4E'%last_loss[0], '%.4E'%last_loss[3],
                  'dPotEnr', '%.4E'%last_loss[1], '%.4E'%last_loss[4],
                  'Summation', '%.4E'%last_loss[2], '%.4E'%last_loss[5],
                  'L=%.4E'%last_loss[6])
            self.epoch += 1
        print('Scaling Complete')
        return self.epoch

    def train_summation_threshold(self, potential_coefficient=1, potential_threshold=1e-3, num_move_ave=10):
        """
        Exactly the same as potential threshold, except the summation step is used instead
        """
        print('Train Summation Threshold')
        self.current_potential_coefficient = potential_coefficient
        potential_above_threshold = True
        test_potential_decreasing = True

        while potential_above_threshold or test_potential_decreasing:
            # Training
            self.train_batches_on_step(self.train_batches, loss.different_summation_step, self.current_potential_coefficient)
            #After all batches seen this epoch
            last_loss = self.eval_losses(self.current_potential_coefficient)
            print('epoch', self.epoch,
                  'atom_rmsd_nm', '%.4E'%last_loss[0], '%.4E'%last_loss[3],
                  'dPotEnr', '%.4E'%last_loss[1], '%.4E'%last_loss[4],
                  'Summation', '%.4E'%last_loss[2], '%.4E'%last_loss[5],
                  'L=%.4E'%last_loss[6])
            self.epoch += 1
            potential_above_threshold = (self.rootgrp['Train']['Potential'][-num_move_ave:, :].mean() > potential_threshold or
                                         self.rootgrp['Test']['Potential'][-num_move_ave:, :].mean() > potential_threshold)
            test_potential_decreasing = self.rootgrp['Test']['Potential'][-2*num_move_ave:-num_move_ave, :].mean() > self.rootgrp['Test']['Potential'][-num_move_ave:, :].mean()
        
        if not potential_above_threshold:
            print('Potential Threshold Run - Break Reason - Threshold Reached')
        if not test_potential_decreasing:
            print('Potential Threshold Run - Break Reason - Test Set Loss Increasing')
        
        return self.epoch

    def MAIN_train_rmsd_only(self, n_rmsd=1000, cutoff_epoch=None):
        if cutoff_epoch is None:
            #Default to the one in the json
            cutoff_epoch = self.json_params['max_epoch']
        
        self.train_nepochs_on_rmsd(n_rmsd)
        self.train_rmsd_to_lowest(max_epoch=cutoff_epoch)
        return self.epoch

    def MAIN_train_rmsd_only_wo_reporting_potential(self, n_rmsd=1000, cutoff_epoch=None, verbose=True):
        if cutoff_epoch is None:
            #Default to the one in the json
            cutoff_epoch = self.json_params['max_epoch']
        
        self.train_nepochs_on_rmsd_wo_reporting_potential(n_rmsd, verbose=verbose)
        self.train_rmsd_to_lowest_wo_reporting_potential(max_epoch=cutoff_epoch, verbose=verbose)
        return self.epoch

    def MAIN_scale_and_train_potential(self, n_rmsd=500, cutoff_epoch=200000):
        self.train_nepochs_on_rmsd(n_rmsd)
        self.train_scaling_potential()
        self.train_summation_threshold(potential_threshold=self.json_params["potential_threshold"])
        return self.epoch
