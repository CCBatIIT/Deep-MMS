import numpy as np
import matplotlib.pyplot as plt

import jax, optax, orbax, sys, os, json, pickle, NN_models, training_functions, glob
import jax.numpy as jnp
import jax_amber2 as jaa

from flax import linen as nn
from flax.training import train_state, orbax_utils
from flax.serialization import from_state_dict, to_state_dict

import mdtraj as md

class AutoEncoder_Experiment():
    def __init__(self, json_fn):
        """
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
        fname_dcd = self.json_params["fname_dcd"]
        fname_prmtop = self.json_params["fname_prmtop"]
        self.n_latents = self.json_params["latent_dim"] #number of latents
        test_slice = self.json_params["test_slice"] #int zero to four inclusive for 20/80 split of test and train
        model_name = self.json_params["model_name"] #string for the model name
        save_dir = self.json_params["save_dir"] #with trailing slash:
        data_start, data_end = self.json_params["data_slice_start"], self.json_params["data_slice_end"] #Slice of data
        batch_size = self.json_params["batch_size"]
        learning_rate = self.json_params["learning_rate"]
        model_type = self.json_params["model_type"]
        dropout_rates = self.json_params["dropout_rates"]

        model_dir = save_dir + f'{model_name}/'

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if self.json_params["data_dir"] == 'None':
            latent_dir = model_dir + f'{self.n_latents:02d}_latents/'
            data_dir = latent_dir + f'rpt_{test_slice}/'
            if not os.path.isdir(latent_dir):
                os.mkdir(latent_dir)
        else:
            data_dir = self.json_params['data_dir']

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)

        if data_end == 'None':
            data_end = None

        c = md.load(fname_dcd, top=fname_prmtop)
        c = c.superpose(c) # FEED IN ALIGNED DATA
        coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))[data_start:data_end]

        #Get information about input data
        num_samples, input_size = coord_set.shape

        #make hidden layers
        hidden_layers = [input_size]*3

        #make test and train sets
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]

        print(self.train_data.shape, self.test_data.shape)

        #Initialize Model
        print(f'### MODEL TYPE = {model_type} ###')
        self.model_type = model_type

        self.model = NN_models.Sigmoid_Dropout_AutoEncoder(input_size=input_size,
                                                           n_latents=self.n_latents,
                                                           hidden_layers=hidden_layers,
                                                           dropout_rates=dropout_rates)

        rng_init = jax.random.PRNGKey(self.n_latents)
        rng, key = jax.random.split(rng_init)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                   params=self.model.init(key, coord_set, rng)['params'],
                                                   tx=optax.adam(learning_rate=learning_rate))
        #Checkpointer
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(os.path.join(data_dir, 'checkpoint_managed'), self.orbax_checkpointer, options)
        
        #Initialize data_storage_file
        self.data_dir = data_dir
        self.model_name = model_name
        self.data_file = open(self.data_dir + f'model_{self.model_name}_{self.n_latents:02d}.out', 'w')
        print(self.model)

        self.epoch = 0
        self.rmsd_loss = ([], [])
        self.tors_loss = ([], [])
        self.pot_enr_loss = ([], [])
        self.summ_loss = ([], [])
        self.potential_coefficients = []
        self.torsional_coefficients = []

        num_train = self.train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        self.num_train_batches = num_complete_batches + bool(leftover)
        self.train_batches = NN_models.DataStream(self.n_latents, num_train, self.num_train_batches, batch_size, self.train_data)

        num_test = self.test_data.shape[0]
        num_complete_batches, leftover = divmod(num_test, batch_size)
        self.num_test_batches = num_complete_batches + bool(leftover)
        self.test_batches = NN_models.DataStream(self.n_latents, num_test, self.num_test_batches, batch_size, self.test_data)
        print("INITIALIZATION COMPLETE")

    def write_model_to_ckpt(self, ckpt_fn=None):
        if ckpt_fn is None:
            ckpt_fn = self.data_dir + f'model_ckpt_{self.epoch:06d}.pkl'
        
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
        fname = self.data_dir + f'{identifier}_{self.model_name}{self.n_latents:02d}.dcd'
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
        decoded, latent = reconstruct(self.test_data, self.epoch)
        if idfn != None:
            self.write_traj(idfn, decoded)
        else:
            self.write_traj(f"recon_test", decoded)

    def save_loss_data(self):
        #LOSSES
        loss_names = ['RMSD', 'TORSION', 'POTENTIAL', 'SUMM']
        for arr in (self.rmsd_loss, self.tors_loss, self.pot_enr_loss, self.summ_loss):
            np.save(self.data_dir + f'{self.model_name}{self.n_latents:02d}_{loss_names.pop(0)}_{self.epoch:06d}.npy', np.array(arr))
        #LAMBDAS (potential)
        np.save(self.data_dir + f'{self.model_name}{self.n_latents:02d}_lambdas_{self.epoch:06d}.npy', np.array(self.potential_coefficients))
        #Betas (torsion)
        np.save(self.data_dir + f'{self.model_name}{self.n_latents:02d}_lambdas_{self.epoch:06d}.npy', np.array(self.torsional_coefficients))

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

        self.tors_loss[0].append(self.eval_batches(self.train_batches, training_functions.torsional_diff).mean())
        self.tors_loss[1].append(self.eval_batches(self.test_batches, training_functions.torsional_diff).mean())
        
        self.pot_enr_loss[0].append(self.eval_batches(self.train_batches, training_functions.scaled_pot_enr_diff).mean())
        self.pot_enr_loss[1].append(self.eval_batches(self.test_batches, training_functions.scaled_pot_enr_diff).mean())

        self.summ_loss[0].append(self.eval_batches(self.train_batches, training_functions.summation_loss, **kwargs).mean())
        self.summ_loss[1].append(self.eval_batches(self.test_batches, training_functions.summation_loss, **kwargs).mean())

        most_recent_results = (self.rmsd_loss[0][-1], self.rmsd_loss[1][-1],
                               self.tors_loss[0][-1], self.tors_loss[1][-1],
                               self.pot_enr_loss[0][-1], self.pot_enr_loss[1][-1],
                               self.summ_loss[0][-1], self.summ_loss[1][-1])

        return most_recent_results


    def report_last_losses(self, torsional_coefficient, potential_coefficient):
        last_losses = self.eval_losses(torsional_coefficient=torsional_coefficient, potential_coefficient=potential_coefficient)
        print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E'%last_losses[0], '%.4E'%last_losses[1],
              'torsional', '%.4E'%last_losses[2], '%.4E'%last_losses[3],
              'dPotEnr', '%.4E'%last_losses[4], '%.4E'%last_losses[5],
              'Summation', '%.4E'%last_losses[6], '%.4E'%last_losses[7],
              'B=%.4E'%torsional_coefficient, 'L=%.4E'%potential_coefficient)
        return last_losses

    
    def train_batches_on_step(self, batch_set, step_function, **kwargs):
        f = iter(batch_set)
        #Before EVERY EPOCH
        rng = jax.random.PRNGKey(self.epoch)
        rng, key = jax.random.split(rng)
        #Ddur batchs EVERY EPOCH
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            #Train Batch
            self.state = step_function(self.state, batch, z_rng=rng, **kwargs)
        #After EVERY EPOCH
        save_args = orbax_utils.save_args_from_target(self.state)
        self.checkpoint_manager.save(self.epoch, self.state, save_kwargs={'save_args': save_args})


    def train_nepochs_on_rmsd(self, num_rmsd_epochs):
        """
        Train on the RMSD function alone, potential coefficient is zero
        """
        torsional_coefficient = 0
        potential_coefficient = 0
        while self.epoch < num_rmsd_epochs:
            #Training
            self.train_batches_on_step(self.train_batches, training_functions.rmsd_rng_step)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()
            
            #Record Data
            self.potential_coefficients.append(potential_coefficient)
            self.torsional_coefficients.append(torsional_coefficient)
            self.epoch += 1

    def train_rmsd_threshold(self, nm_cutoff, num_move_ave, cutoff_epoch):
        """
        Train on the RMSD until a nm_cutoff is reached of the last num_mov_ave epochs
        """
        torsional_coefficient = 0
        potential_coefficient = 0
        while np.mean(self.rmsd_loss[-1][-num_move_ave:]) > nm_cutoff and self.epoch < cutoff_epoch:
            #Training
            self.train_batches_on_step(self.train_batches, training_functions.rmsd_rng_step)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()
            #Record Data
            self.potential_coefficients.append(potential_coefficient)
            self.torsional_coefficients.append(torsional_coefficient)
            if self.epoch % 100 == 0:# Once every 100 epoch, dump all loss data to a file
                self.save_loss_data()
            self.epoch += 1
        return self.epoch

    def train_nepochs_on_torsion(self, num_tors_epochs):
        """
        Train on the torsional difference
        """
        torsional_coefficient = 1
        potential_coefficient = 0
        while self.epoch < num_tors_epochs:
            #Training
            self.train_batches_on_step(self.train_batches, training_functions.torsional_rng_step)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()
            
            #Record Data
            self.potential_coefficients.append(potential_coefficient)
            self.torsional_coefficients.append(torsional_coefficient)
            self.epoch += 1

    def train_potential(self, potential_threshold, num_mov_ave, cutoff_epoch):
        potential_not_below_threshold = True
        potential_is_decreasing = True
        torsional_coefficient = 0
        potential_coefficient = 1

        while (potential_not_below_threshold or potential_is_decreasing) and self.epoch < cutoff_epoch:
            # Training
            self.train_batches_on_step(self.train_batches, training_functions.potential_rng_step)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()
            #Record Data
            self.potential_coefficients.append(potential_coefficient)
            self.torsional_coefficients.append(torsional_coefficient)
            
            # if self.epoch % 50 == 0:
            #     #Check if NAN and abort if so
            #     recent_loss = jnp.array([pot_enr_train_loss, pot_enr_test_loss])
            #     if True in jnp.isnan(recent_loss):
            #         raise NotImplementedError('Potential is not meant to be NAN')
            if self.epoch % 100 == 0:  # Periodically save loss
                self.save_loss_data()
            # if self.epoch % 500 == 0:  # Periodically save a traj
            #     self.write_decoded_traj()
            self.epoch += 1
            # Should break loop?
            doub_mov_ave = 2 * num_mov_ave
            potential_not_below_threshold = (np.mean(self.pot_enr_loss[-1][-num_mov_ave:]) > potential_threshold or
                                             np.mean(self.pot_enr_loss[0][-num_mov_ave:]) > potential_threshold)  # Test and train should be below the threshold
            potential_is_decreasing = np.mean(self.pot_enr_loss[-1][-doub_mov_ave:-num_mov_ave]) > np.mean(self.pot_enr_loss[-1][-num_mov_ave:])  # Test should continue decreasing
        return self.epoch

    def train_scaling_torsional(self, cutoff_epoch, freq=10):
        """
        Scale the torsion in by frequently changing the coefficient to make torsion equal to rmsd
        """
        torsional_coefficient = 0        
        # Every ten epochs choose lambda as min(1, max(lambda[-1], RMSD/NSD))
        while torsional_coefficient != 1 and self.epoch < cutoff_epoch:
            # Sometime check to see if lambda can be larger
            if self.epoch % freq == 0:
                torsional_coefficient = np.min((1, np.max((torsional_coefficient, (self.rmsd_loss[0][-1] / self.pot_enr_loss[0][-1])))))
            # Training
            self.train_batches_on_step(self.train_batches, training_functions.summation_rng_step, torsional_coefficient=torsional_coefficient)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()

            self.torsional_coefficients.append(torsional_coefficient)

            # if self.epoch % 50 == 0:
            #     #Check if NAN and abort if so
            #     recent_loss = jnp.array([pot_enr_train_loss, pot_enr_test_loss])
            #     if True in jnp.isnan(recent_loss):
            #         raise NotImplementedError('Potential is not meant to be NAN')
            if self.epoch % 100 == 0:
                self.save_loss_data()
            self.epoch += 1
        return self.epoch

    def train_scaling_potential(self, cutoff_epoch, freq=10):
        """
        Scale the potential in by frequently changing the coefficient to make potential equal to rmsd
        """
        potential_coefficient = 0        
        # Every ten epochs choose lambda as min(1, max(lambda[-1], RMSD/NSD))
        while potential_coefficient != 1 and self.epoch < cutoff_epoch:
            # Sometime check to see if lambda can be larger
            if self.epoch % freq == 0:
                potential_coefficient = np.min((1, np.max((potential_coefficient, (self.rmsd_loss[0][-1] / self.pot_enr_loss[0][-1])))))
            # Training
            self.train_batches_on_step(self.train_batches, training_functions.summation_rng_step, potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()

            self.potential_coefficients.append(potential_coefficient)

            # if self.epoch % 50 == 0:
            #     #Check if NAN and abort if so
            #     recent_loss = jnp.array([pot_enr_train_loss, pot_enr_test_loss])
            #     if True in jnp.isnan(recent_loss):
            #         raise NotImplementedError('Potential is not meant to be NAN')
            if self.epoch % 100 == 0:
                self.save_loss_data()
            self.epoch += 1
        return self.epoch

    def train_summation(self, potential_threshold, num_mov_ave, cutoff_epoch):
        potential_not_below_threshold = True
        potential_is_decreasing = True
        torsional_coefficient = 1
        potential_coefficient = 1

        while (potential_not_below_threshold or potential_is_decreasing) and self.epoch < cutoff_epoch:
            # Training
            self.train_batches_on_step(self.train_batches, training_functions.summation_rng_step, torsional_coefficient=torsional_coefficient, potential_coefficient=potential_coefficient)
            #After all batches seen this epoch
            last_losses = self.report_last_losses()
            #Record Data
            self.potential_coefficients.append(potential_coefficient)
            self.torsional_coefficients.append(torsional_coefficient)
            # if self.epoch % 50 == 0:
            #     #Check if NAN and abort if so
            #     recent_loss = jnp.array([pot_enr_train_loss, pot_enr_test_loss])
            #     if True in jnp.isnan(recent_loss):
            #         raise NotImplementedError('Potential is not meant to be NAN')
            if self.epoch % 100 == 0:  # Periodically save loss
                self.save_loss_data()
            # if self.epoch % 500 == 0:  # Periodically save a traj
            #     self.write_decoded_traj()
            self.epoch += 1
            # Should break loop?
            doub_mov_ave = 2*num_mov_ave
            potential_not_below_threshold = (np.mean(self.pot_enr_loss[-1][-num_mov_ave:]) > potential_threshold or
                                                np.mean(self.pot_enr_loss[0][-num_mov_ave:]) > potential_threshold)  # Test and train should be below the threshold
            potential_is_decreasing = np.mean(self.pot_enr_loss[-1][-doub_mov_ave:-num_mov_ave]) > np.mean(self.pot_enr_loss[-1][-num_mov_ave:])  # Test should continue decreasing
        return self.epoch

    # def train_model(self, nm_cutoff, max_rmsd_epoch, potential_threshold, cutoff_epoch):
    #     """ CURRENT MAIN USAGE CASE """
    #     # RMSD BLOCK 1
    #     # Train on RMSD until average of last 100 epochs <1 angstrom
    #     # Get first 100 vals
    #     print('START RMSD')
    #     self.train_nepochs_on_rmsd(100)
    #     self.save_loss_data()
    #     self.write_model_to_ckpt()

    #     # RMSD BLOCK 2
    #     # Train until last 100 vals average less than predefined cutoff, always make sure we never train longer than cutoff_epoch
    #     begin_scaling_epoch = self.train_rmsd_threshold(nm_cutoff=nm_cutoff, num_move_ave=100, cutoff_epoch=max_rmsd_epoch)
    #     self.write_model_to_ckpt()

    #     # SCALE IN POTENTIAL BLOCK
    #     print('START SCALING POTENTIAL')
    #     end_scaling_epoch = self.train_scaling_potential(cutoff_epoch=cutoff_epoch)
    #     print('END SCALING POTENTIAL')
    #     self.save_loss_data()
    #     self.write_model_to_ckpt()

    #     #TRAIN ON POTENTIAL BLOCK (MAINTAIN RMSD IF THIS IS AE, DROP THAT TERM IF IT IS VAE
    #     if self.model_type == 'VAE':
    #         self.train_potential(potential_threshold=potential_threshold, cutoff_epoch=cutoff_epoch)
    #     elif self.model_type == 'AE':
    #         self.train_summation(potential_coefficient=self.potential_coefficients[-1], potential_threshold=potential_threshold, cutoff_epoch=cutoff_epoch)
    #     print('END TRAINING')
    #     self.save_loss_data()
    #     self.write_model_to_ckpt()

    #     return begin_scaling_epoch, end_scaling_epoch


    def graph_losses(self, begin_scaling_epoch=None, end_scaling_epoch=None, yscale='log'):
        """
        Produce graphs of the three loss functions
        scaling_epochs: tuple of two ints: epochs at which scaling started and finished
        """

        def plot_and_save_data(data_sets, title, ylabel):
            plt.clf()
            for data_set in data_sets:
                _ = plt.plot(np.array(range(self.epoch)), data_set)
            if begin_scaling_epoch != None:
                plt.axvline(begin_scaling_epoch, color='r', linestyle='dashed')
            if end_scaling_epoch != None:
                plt.axvline(end_scaling_epoch, color='r', linestyle='dashed')
            plt.title(title)
            plt.ylabel(ylabel)
            plt.yscale(yscale)
            plt.xlabel('epoch')
            plt.legend(['train','test'])
            plt.savefig(self.data_dir + f'{ylabel}_{self.model_name}{self.n_latents}.png')
            plt.show()

        #RMSD
        plot_and_save_data(self.rmsd_loss, f'RMSD - {self.n_latents} latents', 'RMSD(nm)')

        #POTENTIAL
        plot_and_save_data(self.pot_enr_loss, f'Norm Square Deviation of Potential - {self.n_latents} latents', 'NSD Potential')

        #Summation
        plot_and_save_data(self.summ_loss, f'Summation - {self.n_latents} latents', 'SumLoss')
