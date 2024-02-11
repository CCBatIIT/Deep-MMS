import numpy as np
import matplotlib.pyplot as plt
import jax, optax, orbax, sys, os, json, pickle, training_functions, glob
from NN_models import *

import jax.numpy as jnp
import jax_amber3 as jaa

from flax import linen as nn
from flax.training import train_state, orbax_utils
from flax.serialization import from_state_dict, to_state_dict

import mdtraj as md

class AutoEncoder_Experiment():
    def __init__(self, json_fn, run_main=False):
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
        
        #Model
        self.model_name = self.json_params["model"]["model_name"]
        self.model_type = self.json_params["model"]["model_type"]
        self.model_class = self.json_params["model"]["model_class"]
        
        #Training
        self.n_latents = self.json_params["training"]["arch"]["latent_dim"]
        batch_size = self.json_params["training"]["arch"]["batch_size"]
        learning_rate = self.json_params["training"]["arch"]["learning_rate"]
        dropout_rates = self.json_params["training"]["arch"]["dropout_rates"]
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
        
        #Get information about input data
        num_samples, input_size = coord_set.shape

        #Make Hidden Layers
        if self.json_params["training"]["arch"]["hidden_layers"] != "None":
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
        self.model = globals()[self.model_class](input_size=input_size, n_latents=self.n_latents,
                                                 hidden_layers=hidden_layers, dropout_rates=dropout_rates)
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
        num_complete_batches, leftover = divmod(num_train, batch_size)
        self.num_train_batches = num_complete_batches + bool(leftover)
        self.train_batches = DataStream(self.n_latents, num_train, self.num_train_batches, batch_size, self.train_data)
        #Test
        num_test = self.test_data.shape[0]
        num_complete_batches, leftover = divmod(num_test, batch_size)
        self.num_test_batches = num_complete_batches + bool(leftover)
        self.test_batches = DataStream(self.n_latents, num_test, self.num_test_batches, batch_size, self.test_data)
        
        #Initialize data_storage
        self.data_file = open(os.path.join(self.data_dir, f'model_{self.model_name}_{self.n_latents:02d}.out'), 'w')
        self.epoch = 0
        self.rmsd_loss = ([], [])
        self.tors_loss = ([], [])
        self.pot_enr_loss = ([], [])
        self.summ_loss = ([], [])
        self.potential_coefficients = []
        self.torsional_coefficients = []
        
        print("######################################")
        print("#####     Initializing Done!     #####")
        print(self.model)
        print("######################################")

        if run_main:
            print('Main Invoked with the following params:')
            num_init_epochs = self.json_params["training"]["epoch"]["num_init_epochs"]
            struct_cutoff = self.json_params["training"]["epoch"]["struct_cutoff"]
            scaling_cutoff = self.json_params["training"]["epoch"]["scaling_cutoff"]
            final_cutoff = self.json_params["training"]["epoch"]["max_epoch"]
            
            struct_thresh = self.json_params["training"]["thresh"]["structural_go_to_scaling"]
            final_thresh = self.json_params["training"]["thresh"]["final_to_end"]

            print(f"Epochs - Init: {num_init_epochs}, Struct: {struct_cutoff}, Scale: {scaling_cutoff}, Final {final_cutoff}")
            print(f"Threshholds - Struct {struct_thresh}, Final {final_thresh}")
            self.main_train(num_init_epochs,
                            struct_thresh, struct_cutoff,
                            scaling_cutoff,
                            final_thresh, final_cutoff)
            
        

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

    def save_loss_data(self):
        #LOSSES
        loss_names = ['RMSD', 'TORSION', 'POTENTIAL', 'SUMM']
        for arr in (self.rmsd_loss, self.tors_loss, self.pot_enr_loss, self.summ_loss):
            np.save(os.path.join(self.data_dir, f'{self.model_name}{self.n_latents:02d}_{loss_names.pop(0)}_{self.epoch:06d}.npy'), np.array(arr))
        #LAMBDAS (potential)
        np.save(os.path.join(self.data_dir, f'{self.model_name}{self.n_latents:02d}_lambdas_{self.epoch:06d}.npy'), np.array(self.potential_coefficients))
        #Betas (torsion)
        np.save(os.path.join(self.data_dir, f'{self.model_name}{self.n_latents:02d}_lambdas_{self.epoch:06d}.npy'), np.array(self.torsional_coefficients))


    def eval_batches(self, batch_set, eval_function, **kwargs):
        vals = []
        f = iter(batch_set)

        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            recon = self.reconstruct(batch, self.epoch)[0]
            #Eval Batch
            vals.append(eval_function(batch, recon, **kwargs))

        vals = jnp.array(vals)
        return vals.flatten()

    def eval_losses(self, **kwargs):
        self.rmsd_loss[0].append(self.eval_batches(self.train_batches, training_functions.atom_rmsd).mean())
        self.rmsd_loss[1].append(self.eval_batches(self.test_batches, training_functions.atom_rmsd).mean())

        self.tors_loss[0].append(self.eval_batches(self.train_batches, training_functions.atom_rmtd).mean())
        self.tors_loss[1].append(self.eval_batches(self.test_batches, training_functions.atom_rmtd).mean())
        
        self.pot_enr_loss[0].append(self.eval_batches(self.train_batches, training_functions.scaled_pot_enr_diff).mean())
        self.pot_enr_loss[1].append(self.eval_batches(self.test_batches, training_functions.scaled_pot_enr_diff).mean())

        self.summ_loss[0].append(self.eval_batches(self.train_batches, training_functions.summation_loss, **kwargs).mean())
        self.summ_loss[1].append(self.eval_batches(self.test_batches, training_functions.summation_loss, **kwargs).mean())

        self.potential_coefficients.append(kwargs["potential_coefficient"])
        self.torsional_coefficients.append(kwargs["torsional_coefficient"])

        most_recent_results = jnp.array((self.rmsd_loss[0][-1], self.rmsd_loss[1][-1],
                               self.tors_loss[0][-1], self.tors_loss[1][-1],
                               self.pot_enr_loss[0][-1], self.pot_enr_loss[1][-1],
                               self.summ_loss[0][-1], self.summ_loss[1][-1]))

        return most_recent_results


    def report_last_losses(self, torsional_coefficient, potential_coefficient):
        #Evaluate and Record
        last_losses = self.eval_losses(torsional_coefficient=torsional_coefficient, potential_coefficient=potential_coefficient)
        #Report Loss Values
        print(self.epoch, '%.4E'%last_losses[0], '%.4E'%last_losses[1],
              '%.4E'%last_losses[2], '%.4E'%last_losses[3], '%.4E'%last_losses[4],
              '%.4E'%last_losses[5], '%.4E'%last_losses[6], '%.4E'%last_losses[7],
              'B=%.4E'%torsional_coefficient, 'L=%.4E'%potential_coefficient)
        #print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E'%last_losses[0], '%.4E'%last_losses[1],
        #      'torsional', '%.4E'%last_losses[2], '%.4E'%last_losses[3],
        #      'dPotEnr', '%.4E'%last_losses[4], '%.4E'%last_losses[5],
        #      'Summation', '%.4E'%last_losses[6], '%.4E'%last_losses[7],
        #      'B=%.4E'%torsional_coefficient, 'L=%.4E'%potential_coefficient)
        #Run an NAN check
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
            self.state, loss_val = step_function(self.state, batch, z_rng=rng, **kwargs)
            loss_vals.append(loss_val)
        #After ANY EPOCH
        save_args = orbax_utils.save_args_from_target(self.state)
        self.checkpoint_manager.save(self.epoch, self.state, save_kwargs={'save_args': save_args})
        return np.array(loss_vals).mean()

    def post_epoch(self, coefficients, nan_check_ind=-1):
        """Things that happen after every epoch, no matter the loss_function"""
        #Record losses and run NAN check
        torsional_coefficient, potential_coefficient = coefficients
        nan_check = self.report_last_losses(torsional_coefficient, potential_coefficient)[:nan_check_ind]
        isNAN_check = True in nan_check
        if isNAN_check:
            print("NAN is no-bueno")
            print(nan_check)            
            sys.exit(69)
        #Record Numpy files sometimes
        if self.epoch % 200 == 0:
            self.save_loss_data()
        #iterate the epoch
        self.epoch += 1
        
    def train_nepochs(self, step_fun, num_epochs, coefficients=[0, 0], nan_check_ind=-1):
        """Choose a parameter to test the loss of step_fun
        step_fun = instance of training function that you would like to use
        (Recently Adjusted to actually go n_epochs, not up to the given epoch)
        Example: Train 100 epochs on RMSD and torsion, with B=0.5 and L=0
            train_nepochs(training_functions.rmsd_rng_step, 100, coefficients=(0.5, 0))
        """
        torsional_coefficient, potential_coefficient = coefficients
        for i in range(num_epochs):
            #Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun,
                                                         torsional_coefficient=torsional_coefficient,
                                                         potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            self.post_epoch(coefficients, nan_check_ind)
            
        return self.epoch

    def train_scaling_coef(self, step_fun, cutoff_epoch, scaling_coef_ind, freq=10, coefficients=[0, 0], nan_check_ind=-1):
        """
        Scale the torsion with scaling_coef_ind 0 and the potential with scaling_coef_ind 1
        Typically the potential must be scaled, not the torsion 
        Typical Usage - self.train_scaling_coef(training_functions.summation_rng_step, X, 1, coefficients=[1,0])
        """
        assert scaling_coef_ind in [0, 1]
        # Every ten epochs choose lambda as min(1, max(lambda[-1], RMSD/NSD))
        while coefficients[scaling_coef_ind] != 1 and self.epoch < cutoff_epoch:
            # Sometime check to see if coef can be larger
            if self.epoch % freq == 0:
                #A small part that hard codes which is which
                if scaling_coef_ind == 0:
                    prop_coef = (np.mean(self.rmsd_loss[0][-10:]) / np.mean(self.tors_loss[0][-10:]))
                elif scaling_coef_ind == 1:
                    prop_coef = (np.mean(self.rmsd_loss[0][-10:]) / np.mean(self.pot_enr_loss[0][-10:]))
                #Propose next coefficient
                coefficients[scaling_coef_ind] = np.min((1, np.max((coefficients[scaling_coef_ind], proposed_coef))))
                
            # Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun,
                                                         torsional_coefficient=coefficients[0],
                                                         potential_coefficient=coefficients[1])
            #After all batches seen this epoch
            self.post_epoch(coefficients, nan_check_ind)
            
        return self.epoch

    def train_to_threshold(self, step_fun, threshold, cutoff_epoch, coefficients=[1, 1], nan_check_ind=-1):
        """Train the NN on the given step_fun, until the value of that function is below threshold"""
        
        torsional_coefficient, potential_coefficient = coefficients
        loss_this_epoch = 100 #something large just to start this loop, value is actually reported elsewhere
        while loss_this_epoch > threshold and self.epoch < cutoff_epoch:
            #Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun,
                                                         torsional_coefficient=torsional_coefficient,
                                                         potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            self.post_epoch(coefficients, nan_check_ind)
            
        return self.epoch

    def run_main(self):
        """ Run structural training with torsion in (first for 100 epoch, then until it reaches struct_thresh) until struct_cutoff
            Followed by potential scaling (up to scaling_cutoff_epoch)
            Followed by final training (training of full function to final_thresh)"""
        
        print('Main Invoked with the following params:')
        num_init_epochs = self.json_params["training"]["epoch"]["num_init_epochs"]
        struct_cutoff = self.json_params["training"]["epoch"]["struct_cutoff"]
        scaling_cutoff = self.json_params["training"]["epoch"]["scaling_cutoff"]
        final_cutoff = self.json_params["training"]["epoch"]["max_epoch"]
        
        struct_thresh = self.json_params["training"]["thresh"]["structural_go_to_scaling"]
        final_thresh = self.json_params["training"]["thresh"]["final_to_end"]

        print(f"Epochs - Init: {num_init_epochs}, Struct: {struct_cutoff}, Scale: {scaling_cutoff}, Final {final_cutoff}")
        print(f"Threshholds - Struct {struct_thresh}, Final {final_thresh}")
        
        print('##############################')
        print('#####', f'Init {self.epoch:05d}', '#####')
        print('##############################')
        end_init_epoch = self.train_nepochs(training_functions.structural_rng_step,
                                            num_init_epochs, coefficients=[1,0], nan_check_ind=-4)
        print('##############################')
        print('#####', f'Strt {self.epoch:05d}', '#####')
        print('##############################')
        end_struct_epoch = self.train_to_threshold(training_functions.structural_rng_step,
                                                   struct_thresh, struct_cutoff, coefficients=[1,0], nan_check_ind=-4)
        print('##############################')
        print('#####', f'StSc {self.epoch:05d}', '#####')
        print('##############################')
        end_scaling_epoch = self.train_scaling_coef(training_functions.summation_rng_step,
                                                    scaling_cutoff, 1, coefficients=[1,0], nan_check_ind=-1)
        print('##############################')
        print('#####', f'Scld {self.epoch:05d}', '#####')
        print('##############################')
        end_training_epoch = self.train_to_threshold(training_functions.summation_rng_step,
                                                     final_thresh, final_cutoff, coefficients=[1, 1], nan_check_ind=-1)
        print('##############################')
        print('#####', f'Done {self.epoch:05d}', '#####')
        print('##############################')