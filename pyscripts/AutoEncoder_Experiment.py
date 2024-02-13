import numpy as np
import matplotlib.pyplot as plt
import jax, optax, orbax, sys, os, json, pickle, glob
from .training_functions import Maths
from .NN_models import *
from . import jax_amber3 as jaa
import jax.numpy as jnp
import flax
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
        batch_size = self.json_params["training"]["arch"]["batch_size"]
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

    def eval_batches(self, batch_set, eval_function, **kwargs):
        vals = []
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            decoded, latents = self.reconstruct(batch, self.epoch)
            #Eval Batch
            vals.append(eval_function(batch, decoded, **kwargs))
        vals = jnp.array(vals)
        return vals.flatten()

    def eval_losses(self, **kwargs):
        self.rmsd_loss[0].append(self.eval_batches(self.train_batches, self.math.atom_rmsd).mean())
        self.rmsd_loss[1].append(self.eval_batches(self.test_batches, self.math.atom_rmsd).mean())

        self.tors_loss[0].append(self.eval_batches(self.train_batches, self.math.atom_rmtd).mean())
        self.tors_loss[1].append(self.eval_batches(self.test_batches, self.math.atom_rmtd).mean())
        
        self.pot_enr_loss[0].append(self.eval_batches(self.train_batches, self.math.scaled_pot_enr_diff).mean())
        self.pot_enr_loss[1].append(self.eval_batches(self.test_batches, self.math.scaled_pot_enr_diff).mean())

        self.summ_loss[0].append(self.eval_batches(self.train_batches, self.math.summation_distance, **kwargs).mean())
        self.summ_loss[1].append(self.eval_batches(self.test_batches, self.math.summation_distance, **kwargs).mean())

        self.potential_coefficients.append(kwargs["potential_coefficient"])

        most_recent_results = jnp.array((self.rmsd_loss[0][-1], self.rmsd_loss[1][-1],
                               self.tors_loss[0][-1], self.tors_loss[1][-1],
                               self.pot_enr_loss[0][-1], self.pot_enr_loss[1][-1],
                               self.summ_loss[0][-1], self.summ_loss[1][-1]))

        return most_recent_results


    def report_last_losses(self, potential_coefficient):
        #Evaluate and Record
        last_losses = self.eval_losses(potential_coefficient=potential_coefficient)
        #Report Loss Values
        print(self.epoch, '%.4E'%last_losses[0], '%.4E'%last_losses[1],
              '%.4E'%last_losses[2], '%.4E'%last_losses[3], '%.4E'%last_losses[4],
              '%.4E'%last_losses[5], '%.4E'%last_losses[6], '%.4E'%last_losses[7],
              'L=%.4E'%potential_coefficient)
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

    def post_epoch(self, potential_coefficient, nan_check_ind=-1):
        """Things that happen after every epoch, no matter the loss_function"""
        #Record losses and run NAN check
        nan_check = self.report_last_losses(potential_coefficient)[:nan_check_ind]
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
        
    def train_nepochs(self, loss_metric, averaging_method, num_epochs, potential_coefficient=0, nan_check_ind=-1):
        step_fun = self.math.make_step_function(loss_metric, averaging_method)
        for i in range(num_epochs):
            #Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun, potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            self.post_epoch(potential_coefficient, nan_check_ind)
        return self.epoch

    def train_scaling_coef(self, loss_metric, averaging_method, cutoff_epoch, init_potential_coef=0, freq=10, nan_check_ind=-1):
        step_fun = self.math.make_step_function(loss_metric, averaging_method)
        potential_coefficient = init_potential_coef
        # Every ten epochs choose lambda as min(1, max(lambda[-1], RMSD/NSD))
        while potential_coefficient != 1 and self.epoch < cutoff_epoch:
            # Sometime check to see if coef can be larger
            if self.epoch % freq == 0:
                #A small part that hard codes which is which
                proposed_coef = (np.mean(self.rmsd_loss[0][-10:]) / np.mean(self.pot_enr_loss[0][-10:]))
                #Propose next coefficient
                potential_coefficient = np.min((1, np.max((potential_coefficient, proposed_coef))))
            # Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun, potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            self.post_epoch(potential_coefficient, nan_check_ind)
        return self.epoch

    def train_to_threshold(self, loss_metric, averaging_method, threshold, cutoff_epoch, potential_coefficient=1, nan_check_ind=-1):
        step_fun = self.math.make_step_function(loss_metric, averaging_method)
        loss_this_epoch = 100 #something large just to start this loop, this is a ghost value
        while loss_this_epoch > threshold and self.epoch < cutoff_epoch:
            # Training
            loss_this_epoch = self.train_batches_on_step(self.train_batches, step_fun, potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            self.post_epoch(potential_coefficient, nan_check_ind)
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