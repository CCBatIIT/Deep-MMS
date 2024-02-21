import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import jax, optax, orbax, sys, os, json, pickle, glob, flax, time
from .Maths import Maths
from .NN_models import *
from . import jax_amber3 as jaa
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, orbax_utils
from flax.serialization import from_state_dict, to_state_dict

import mdtraj as md


class AutoEncoder_Experiment():
    def __init__(self, json_fn):
        """
        THIS DOCSTRING REQUIRES EDITING
        
        n_latents: int: number of latent dimensions
        coord_set: jnp.array(): data from which both test and train set will be derived
        test_slice: int in (0,1,2,3,4): which 80/20 slice of coord_set to take for test and train
        data_dir: string: directory (with trailing slash) in which to store all output data, images, etc.
        batch_size: int: default 400; number of frames in a training batch
        learning_rate: float: default 1e-4; optax adam learning rate
        dropout_rates: list of floats: encoder and decoder dropout rates forward for encoder and reverse for decoder
        """
        
        #Get the information from the json file
        with open(json_fn, 'r') as g:
            self.json_params = json.load(g)

        #Files
        fname_dcd = self.json_params["files"]["fname_dcd"]
        fname_prmtop = self.json_params["files"]["fname_prmtop"]
        save_dir = self.json_params["files"]["save_dir"]
        self.math = Maths(fname_prmtop)
        
        #Model
        self.model_name = self.json_params["model"]["model_name"]
        self.model_type = self.json_params["model"]["model_type"]
        
        #Training & Architecture
        self.n_latents = self.json_params["training"]["arch"]["latent_dim"]
        activators = self.json_params["training"]["arch"]["activators"]
        layer_ops = self.json_params["training"]["arch"]["layer_ops"]
        dropout_rates = self.json_params["training"]["arch"]["dropout_rates"]
        self.batch_size = self.json_params["training"]["arch"]["batch_size"]
        learning_rate = self.json_params["training"]["arch"]["learning_rate"]
        test_slice = self.json_params["training"]["data"]["test_slice"]
        rng_key = self.json_params["training"]["arch"]["rng_key"]
        data_start, data_end = self.json_params["training"]["data"]["data_slice_start"], self.json_params["training"]["data"]["data_slice_end"]

        #Establish Model Directory
        model_dir = os.path.join(save_dir, f'{self.model_name}/')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        
        #Establish Data Directory
        if self.json_params["files"]["data_dir"] == 'None':
            latent_dir = os.path.join(model_dir, f'{self.n_latents:02d}_latents/')
            self.data_dir = os.path.join(latent_dir, f'rpt_{test_slice}/')
            if not os.path.isdir(latent_dir):
                os.mkdir(latent_dir)
        else:
            self.data_dir = self.json_params["files"]["data_dir"]
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        #Get Data to train on
        if data_end == 'None':
            data_end = None
        
        #Load and Align
        c = md.load(fname_dcd, top=fname_prmtop)
        c = c.superpose(c) # FEED IN ALIGNED DATA
        coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))[data_start:data_end] # reshape to be n_conf, 3*n_atom 
        num_samples, input_size = coord_set.shape

        #Make Hidden Layers
        if self.json_params["training"]["arch"]["hidden_layers"] != "Auto":
            hidden_layers = self.json_params["training"]["arch"]["hidden_layers"]
        else:
            hidden_layers = [input_size]*3

        #Make test and train sets - indices, then sets
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]
        print(self.train_data.shape, self.test_data.shape)
        
        #Initialize Model
        print(f'### MODEL TYPE = {self.model_type} ###')
        activators = [eval(activator) for activator in activators]
        layer_ops = [eval(layer_op) for layer_op in layer_ops]
        self.model = AutoEncoder(input_size=input_size, n_latents=self.n_latents,
                                 hidden_layers=hidden_layers, activators=activators,
                                 layer_ops=layer_ops, dropout_rates=dropout_rates)
        
        rng_init = jax.random.PRNGKey(rng_key)
        rng, key = jax.random.split(rng_init)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                   params=self.model.init(key, coord_set, rng)['params'],
                                                   tx=optax.adam(learning_rate=learning_rate))
        #Checkpointer
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(os.path.join(self.data_dir, 'checkpoint_managed'), self.orbax_checkpointer, options)
        
        #Set Test and Train data into batches - Train
        num_train = self.train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, self.batch_size)
        self.num_train_batches = num_complete_batches + bool(leftover)
        self.train_batches = DataStream(self.n_latents, num_train, self.num_train_batches, self.batch_size, self.train_data)
        #Test
        num_test = self.test_data.shape[0]
        num_complete_batches, leftover = divmod(num_test, self.batch_size)
        self.num_test_batches = num_complete_batches + bool(leftover)
        self.test_batches = DataStream(self.n_latents, num_test, self.num_test_batches, self.batch_size, self.test_data)
        
        #Initialize data_storage
        self.nc_data_file = os.path.join(self.data_dir, f'model_{self.model_name}_{self.n_latents:02d}.nc')
        self.establish_netcdf(self.nc_data_file)
        self.epoch = 0
        
        print("######################################")
        print("#####     Initializing Done!     #####")
        print(self.model)
        print("######################################")

        #testing
        self.kernel_function = self.math.make_gaussian_kernel(self.math.rmsd_distance_matrix, 'mean', self.train_data)
        
    def establish_netcdf(self, nc_filename):
        rootgrp = nc.Dataset(nc_filename, 'w', format='NETCDF4')
        traingrp = rootgrp.createGroup('Train')
        testgrp = rootgrp.createGroup('Test')

        traingrp.createDimension('epoch', None)
        traingrp.createDimension('batch', self.num_train_batches)
        traingrp.createDimension('frame', self.batch_size)

        testgrp.createDimension('epoch', None)
        testgrp.createDimension('batch', self.num_test_batches)
        testgrp.createDimension('frame', self.batch_size)

        for grp in [traingrp, testgrp]:
            rmsd = grp.createVariable('RMSD', 'f4', ('epoch', 'batch', 'frame',))
            rmsd.units = "Nanometer"
            rmtd = grp.createVariable('RMTD', 'f4', ('epoch', 'batch', 'frame',))
            rmtd.units = "Radians"
            pot = grp.createVariable('Potential', 'f8', ('epoch', 'batch', 'frame',))
            pot.units = "KJ/mol"
            #repel = grp.createVariable('Repulsion', 'f8', ('epoch', 'batch', 'frame',))
            #repel.units = 'depends'
            grp.history = "Created" + time.ctime(time.time())
            
        self.traingrp, self.testgrp = traingrp, testgrp
        
    
    def write_model_to_ckpt(self, ckpt_fn=None):
        if ckpt_fn is None:
            ckpt_fn = os.path.join(self.data_dir, f'model_ckpt_{self.epoch:06d}.pkl')
        
        save_args = orbax_utils.save_args_from_target(self.state)
        self.orbax_checkpointer.save(ckpt_fn, self.state, save_args=save_args)

    def load_model_from_ckpt(self, chkpt_fn, restore_step):
        self.state = self.orbax_checkpointer.restore(chkpt_fn, item=self.state)

    # def restore_latest(self, restore_model=True, restore_numpy=False):
    #     #Restore Model
    #     if restore_model is True:
    #         self.state = self.checkpoint_manager.restore(self.checkpoint_manager.latest_step(), items=self.state)
        
    #     #Retrieve Loss Data
    #     npy_fns = glob.glob(os.path.join(self.data_dir, '*.npy'))
    #     loss_keys = [key for key in {'RMSD': 0, 'POTENTIAL': 0, 'SUM': 0, 'lambdas' : 0}]
    #     for key in ['RMSD', 'POTENTIAL', 'SUM', 'lambdas']:
    #         npy_ind = [key in npy_fn for npy_fn in npy_fns].index(True)
    #         npy_data = np.load(npy_fns[npy_ind])
    #         if key == 'RMSD':
    #             self.rmsd_loss = (list(npy_data[0]), list(npy_data[1]))
    #             self.epoch = len(self.rmsd_loss[0])
    #         elif key == 'POTENTIAL':
    #             self.pot_enr_loss = (list(npy_data[0]), list(npy_data[1]))
    #         elif key == 'SUM':
    #             self.summ_loss = (list(npy_data[0]), list(npy_data[1]))
    #         elif key == 'lambdas':
    #             self.potential_coefficients = list(npy_data)
    #         #print(self.rmsd_loss, self.pot_enr_loss, self.summ_loss, self.potential_coefficients)
    #         #assert len(self.rmsd_loss[0]) == len(self.pot_enr_loss[1])
    #         #assert len(self.pot_enr_loss[1]) == len(self.summ_loss[0])
    #         #assert len(self.lambdas) == len(self.summ_loss[0])
            
            
    def write_traj(self, identifier, traj_xyz): #(n conf, n_atoms*3) OR (n conf, n_atoms, 3)
        fname = os.path.join(self.data_dir, f'{identifier}_{self.model_name}{self.n_latents:02d}.dcd')
        if traj_xyz.shape[-1] != 3:
            traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)

        with md.formats.DCDTrajectoryFile(fname, 'w') as f:
            f.write(traj_xyz*10) #*10 because mdtraj loads data in nm but saves it in angstrom

    def reconstruct(self, data, rng_seed):
        rng = jax.random.PRNGKey(rng_seed)
        rng, key = jax.random.split(rng)
        decoded, latent = self.state.apply_fn({'params':self.state.params}, data, rng)
        return decoded, latent
    
    def write_decoded_traj(self, idfn=None):
        decoded, latent = self.reconstruct(self.test_data, self.epoch)
        if idfn != None:
            self.write_traj(idfn, decoded)
        else:
            self.write_traj(f"recon_test", decoded)

    def plot_losses(self):
        
        for var_name in ['RMSD', 'RMTD', 'Potential']:
            plt.clf()
            for grp in (self.traingrp, self.testgrp):
                _ = plt.plot(np.arange(self.epoch), np.mean(grp.variables[var_name], axis=(1, 2)))
            plt.title(var_name)
            if var_name == 'Potential':
                plt.yscale('log')
            plt.xlabel('Epoch')
            plt.show()
    
    def eval_batches(self, batch_set, eval_function, **kwargs):
        vals = []
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            decoded, latents = self.reconstruct(batch, self.epoch)
            #Eval Batch
            vals.append(eval_function(batch, decoded))
        return jnp.array(vals)
        

    def eval_losses(self):
        self.traingrp.variables['RMSD'][self.epoch, :, :] = self.eval_batches(self.train_batches, self.math.atom_rmsd)
        self.testgrp.variables['RMSD'][self.epoch, :, :] = self.eval_batches(self.test_batches, self.math.atom_rmsd)
        
        self.traingrp.variables['RMTD'][self.epoch, :, :] = self.eval_batches(self.train_batches, self.math.atom_rmtd)
        self.testgrp.variables['RMTD'][self.epoch, :, :] = self.eval_batches(self.test_batches, self.math.atom_rmtd)
        
        self.traingrp.variables['Potential'][self.epoch, :, :] = self.eval_batches(self.train_batches, self.math.scaled_pot_enr_diff)
        self.testgrp.variables['Potential'][self.epoch, :, :] = self.eval_batches(self.test_batches, self.math.scaled_pot_enr_diff)

        #FIX THIS NOW
        #self.traingrp.variables['Repulsion'][self.epoch, :, :] = self.eval_batches(self.train_batches, self.kernel_function)
        #self.testgrp.variables['Repulsion'][self.epoch, :, :] = self.eval_batches(self.test_batches, self.kernel_function)
        
        most_recent_results = []

        for grp in (self.traingrp, self.testgrp):
            for variable in ['RMSD', 'RMTD', 'Potential']:#, 'Repulsion']:
                most_recent_results.append(grp.variables[variable][self.epoch, :, :].mean())
        
        
        #most_recent_results = jnp.array((self.rmsd_loss[0][-1], self.rmsd_loss[1][-1],
        #                       self.tors_loss[0][-1], self.tors_loss[1][-1],
        #                       self.pot_enr_loss[0][-1], self.pot_enr_loss[1][-1]))

        return jnp.array(most_recent_results)


    def report_last_losses(self, weights):
        #Evaluate and Record
        last_losses = self.eval_losses()
        #Report Loss Values
        print(self.epoch, '%.4E'%last_losses[0], '%.4E'%last_losses[1], '%.4E'%last_losses[2],
              '%.4E'%last_losses[3], '%.4E'%last_losses[4], '%.4E'%last_losses[5], weights)
        NAN_check = jnp.isnan(last_losses)
        return NAN_check

    
    def train_batches_on_step(self, batch_set, step_function, **kwargs):
        f = iter(batch_set)
        #Before ANY EPOCH
        rng = jax.random.PRNGKey(self.epoch)
        rng, key = jax.random.split(rng)
        loss_vals = []
        #Dur batchs ANY EPOCH
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            #Train Batch
            self.state, loss_val = step_function(self.state, batch, z_rng=rng)
            loss_vals.append(loss_val)
        #After ANY EPOCH
        save_args = orbax_utils.save_args_from_target(self.state)
        self.checkpoint_manager.save(self.epoch, self.state, save_kwargs={'save_args': save_args})
        return np.array(loss_vals).mean()

    def post_epoch(self, weights, nan_check_ind=-1):
        """Things that happen after every epoch, no matter the loss_function"""
        #Record losses and run NAN check
        nan_check = self.report_last_losses(weights)[:nan_check_ind]
        isNAN_check = True in nan_check
        if isNAN_check:
            print("NAN is no-bueno")
            print(nan_check)            
            sys.exit(69)
        #iterate the epoch
        self.epoch += 1
        
    def define_step_function(self, loss_metrics:list, weights:list, averaging_method:str):
        #Establish Loss Function
        if len(loss_metrics) == 1 and len(weights) == 1 and weights[0] == 1:
            loss_function = loss_metrics[0]
        else:
            loss_function = self.math.make_summation_distance_function(loss_metrics, weights)
        #Establish Step Function
        step_fun = self.math.make_step_function(loss_function, averaging_method)
        return step_fun
    
    
    def train_nepochs(self, loss_metrics:list, weights:list, averaging_method:str,
                      num_epochs:int, nan_check_ind:int=-1):
        """
        Train for a given number of epochs
        """
        step_fun = self.define_step_function(loss_metrics, weights, averaging_method)
        #Training Loop
        for i in range(num_epochs):
            #Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun)
            #After all batches seen this epoch
            self.post_epoch(weights, nan_check_ind)
        
        return self.epoch

    def train_to_threshold(self, loss_metrics:list, weights:list, averaging_method:str,
                           threshold:float, cutoff_epoch:int, nan_check_ind:int=-1):
        """
        Train until the value of loss is at or below a threshold
        """
        step_fun = self.define_step_function(loss_metrics, weights, averaging_method)
        loss_this_epoch = 100 #something large just to start this loop, this is a ghost value
        while loss_this_epoch > threshold and self.epoch < cutoff_epoch:
            # Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun)
            #After all batches seen this epoch
            self.post_epoch(weights, nan_check_ind)
        return self.epoch

    def train_scaling_coef(self, loss_metrics:list, weights:list, averaging_method:str,
                           scaling_index:int, cutoff_epoch:int, freq:int=10, nan_check_ind:int=-1):
        """
        Scale a weight from an initial value up to one
            This is very typically a scaling of the potential from zero to one
        """
        step_fun = self.define_step_function(loss_metrics, weights, averaging_method)
        # Every ten epochs choose lambda as min(1, max(lambda[-1], RMSD/NSD))
        while weights[scaling_index] < 1 and self.epoch < cutoff_epoch:
            # Sometimes check to see if coef can be larger
            if self.epoch % freq == 0:
                #Hard Coded :(
                proposed_coef = (np.mean(self.rmsd_loss[0][-10:]) / np.mean(self.pot_enr_loss[0][-10:]))
                #Propose next weight value
                weights[scaling_index] = np.min((1, np.max((1.01*weights[scaling_index], proposed_coef))))
                #Redefine the step function to reflect the new weight
                step_fun = self.define_step_function(loss_metrics, weights, averaging_method)
                
            # Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun)
            #After all batches seen this epoch
            self.post_epoch(weights, nan_check_ind)
        return self.epoch

    def main(self):
        print('Main Invoked with the following params:')
        num_init_epochs = self.json_params["training"]["epoch"]["num_init_epochs"]
        struct_cutoff = self.json_params["training"]["epoch"]["struct_cutoff"]
        scaling_cutoff = self.json_params["training"]["epoch"]["scaling_cutoff"]
        final_cutoff = self.json_params["training"]["epoch"]["max_epoch"]
        
        struct_thresh = self.json_params["training"]["thresh"]["structural_go_to_scaling"]
        final_thresh = self.json_params["training"]["thresh"]["final_to_end"]

        print(f"Epochs - Init: {num_init_epochs}, Struct: {struct_cutoff}, Scale: {scaling_cutoff}, Final {final_cutoff}")
        print(f"Threshholds - Struct {struct_thresh}, Final {final_thresh}")
        
        #INIT EPOCHS
        print('##############################')
        print('#####', f'Init {self.epoch:05d}', '#####')
        print('##############################')
        end_init_epoch = self.train_nepochs(self.math.structural_distance, 'rmsd', num_init_epochs, nan_check_ind=-4)
        
        # REACH THRESHOLD BY STRUCTURE ALONE
        print('##############################')
        print('#####', f'Strt {self.epoch:05d}', '#####')
        print('##############################')
        end_struct_epoch = self.train_to_threshold(self.math.structural_distance, 'rmsd', struct_thresh,
                                                   struct_cutoff, potential_coefficient=0, nan_check_ind=-4)
        
        # SCALE IN POTENTIAL ENERGY
        print('##############################')
        print('#####', f'StSc {self.epoch:05d}', '#####')
        print('##############################')
        end_scaling_epoch = self.train_scaling_coef(self.math.summation_distance, 'rmsd', scaling_cutoff)

        # REACH A THRESHOLD OF LOSS AGAIN
        print('##############################')
        print('#####', f'Scld {self.epoch:05d}', '#####')
        print('##############################')
        end_training_epoch = self.train_to_threshold(self.math.summation_distance, 'rmsd', final_thresh, final_cutoff)

        # DONE REPORT THE LAST EPOCH
        print('##############################')
        print('#####', f'Done {self.epoch:05d}', '#####')
        print('##############################')