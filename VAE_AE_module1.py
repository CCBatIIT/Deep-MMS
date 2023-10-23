import jax_amber2 as jaa
import jax_obc2, jax_model

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import jax, optax, sys, os, json
import jax.numpy as jnp
from flax import linen as nn 
from flax.training import train_state

import mdtraj as md

# USAGE TO RUN THE MODEL "$python VAE_module2.py json_fn"

def get_positions_from_pdb(fname_pdb):
    nameMembrane = ['DPP', 'POP']

    f_pdb = open(fname_pdb)
    l_pdb = f_pdb.read().split('\n')
    f_pdb.close()

    coords = []
    prt_heavy_atoms = []
    mem_heavy_atoms = []
    iatom = 0
    for line in l_pdb[:-1]:
        if line[:6] in ['ATOM  ', 'HETATM']:
            words = line[30:].split()
            x = float(words[0])
            y = float(words[1])
            z = float(words[2])

            coords.append(Vec3(x, y, z))

            if line[17:20] in nameMembrane and words[-1] != 'H':
                mem_heavy_atoms.append(iatom)
            elif line[:6] in ['ATOM  '] and words[-1] != 'H':
                prt_heavy_atoms.append(iatom)

            iatom += 1

    return np.array(coords), prt_heavy_atoms, mem_heavy_atoms


#Gas Function
def get_amber_gas_energy_functions(fname_prmtop):
    prm_raw_data = jaa.amber_prmtop_load (fname_prmtop)
    ener_bonded_fn, ener_bond_fn = jaa.ener_bonded (prm_raw_data)
    
    chgs = jaa.prm_get_charges (prm_raw_data)
    atom_types = jaa.prm_get_atom_types (prm_raw_data)
    
    sigma, epsilon = jaa.prm_get_nonbond_terms (prm_raw_data)
    
    nonbond_pairs = jaa.prm_get_nonbond_pairs (prm_raw_data)
    ener_nbond_fn = jaa.ener_nonbonded_pair (atom_types, nonbond_pairs, sigma, epsilon, chgs)
    
    nbonds14 = jaa.prm_get_nonbond14_info (prm_raw_data)
    ener_nbond14_fn = jaa.ener_nonbonded14 (atom_types, nbonds14, sigma, epsilon, chgs)
    
    ener_bonded_fn = jax.jit (ener_bonded_fn)
    ener_bond_fn = jax.jit (ener_bond_fn)
    ener_nbond_fn = jax.jit(ener_nbond_fn)
    ener_nbond14_fn = jax.jit(ener_nbond14_fn)
    #vmax0 = jnp.float32(100.0)
    @jax.vmap
    def compute_fun (R):
        """
        R (natom * 3)
        """
        #print(R.shape)
        R = R.reshape(-1,3)
        en_bonded = ener_bonded_fn (R)
        en_lj, en_chg = ener_nbond_fn (R)
        en_lj14, en_chg14 = ener_nbond14_fn (R)
        
        return en_bonded + en_lj + en_chg + en_lj14 + en_chg14

    return compute_fun, ener_bonded_fn, ener_nbond_fn, ener_nbond14_fn

#Batching training data
def data_stream(rng_seed, num_train, num_batches, batch_size, train_data):
    rng = npr.RandomState(rng_seed)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield train_data[batch_idx]

def reparameterize(z_rng, z_mean, z_logvar):
    z_std = jnp.exp(0.5*z_logvar)
    z_eps = jax.random.normal(z_rng, z_logvar.shape)
    return z_mean + z_eps*z_std 


#VAE Classes
class VEncoder(nn.Module):
    d_hidden: list
    latents: int
    dropout_rates: list
    
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.d_hidden)):
            x = nn.relu(nn.Dense(self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        
        mean_x = nn.Dense(self.latents, name='fc5_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc5_logvar')(x)

        return mean_x, logvar_x 

class Encoder(nn.Module):
    d_hidden: list
    latents: int
    dropout_rates: list

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.d_hidden)):
            x = nn.relu(nn.Dense(self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)

        x = nn.Dense(self.latents, name='f5')(x)

        return x
    
class Decoder(nn.Module):
    d_hidden: list
    out_dim: int
    dropout_rates: list

    @nn.compact
    def __call__(self, z):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.relu(nn.Dense(self.d_hidden[i])(z))
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=True)
        
        z = nn.Dense(self.out_dim, name='f5')(z)

        return z
        
class AE(nn.Module):
    input_size: int
    hidden_layers: tuple
    dropout_rates: list
    latents: int

    def setup(self):
        self.encoder = Encoder(list(self.hidden_layers), self.latents, self.dropout_rates)
        self.decoder = Decoder(list(self.hidden_layers), self.input_size, self.dropout_rates)
    
    def __call__(self, x):
        z_latent = self.encoder(x)
        return self.decoder(z_latent), z_latent
        

class VAE(nn.Module):
    input_size: int
    hidden_layers: tuple
    dropout_rates: list
    latents: int
        
    def setup(self):
        self.encoder = VEncoder(list(self.hidden_layers), self.latents, self.dropout_rates)
        self.decoder = Decoder(list(self.hidden_layers), self.input_size, self.dropout_rates)

    def __call__(self, x, z_rng):
        z_mean, z_logvar = self.encoder(x)
        z = reparameterize(z_rng, z_mean, z_logvar)
        recon_x = self.decoder(z)
        return recon_x, z_mean, z_logvar
    
    def construct(self, z_mean, z_logvar, z_rng):
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z)
    
    def mvn_latent_model(self, x, z_rng):
        return NotImplemented()


@jax.vmap
def atom_rmsd(a, b): # for arrays of (n_conf, n_atom*3)
    mn = a.shape[-1]//3
    x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
    return jnp.sqrt(jnp.mean((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))

@jax.vmap
def msd_fun(recon, target):
    return jnp.sum((recon-target)**2)

@jax.jit
def scaled_pot_enr_diff(a, b): # WITH A AS BATCH AND B AS RECON
    return (((gas_fun(a) - gas_fun(b))/gas_fun(a))**2).mean() #Unitless quantity

@jax.jit
def summation_loss(a, b, potential_coefficient, weights=(1,1)): # LET A BE BATCH AND B BE RECON
    # Make this the square root of the mean of the sum of squares of elements
    return jnp.sqrt(jnp.sum(atom_rmsd(a,b)**2)) + (potential_coefficient * scaled_pot_enr_diff(a, b))

@jax.jit
def rmsd_step(state, batch_x, z_rng):
    def loss_fn(params, apply_fn):
        recon_x, z_mean, z_logvar = apply_fn({'params':params}, batch_x, z_rng)
        return summation_loss(batch_x, recon_x, 0) 
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients (grads=grads)


@jax.jit
def potential_step(state, batch_x, z_rng):
    def loss_fn(params, apply_fn):
        recon_x, z_mean, z_logvar = apply_fn({'params':params}, batch_x, z_rng)
        return scaled_pot_enr_diff(batch_x, recon_x)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)    


@jax.jit
def summation_step(state, batch_x, potential_coefficient, z_rng, weights=(1,1)):
    def loss_fn(params, apply_fn):
        recon_x, z_mean, z_logvar = apply_fn({'params':params}, batch_x, z_rng)
        return summation_loss(batch_x, recon_x, potential_coefficient)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)


class NN_Experiment():
    def __init__(self, n_latents, coord_set, test_slice, data_dir, model_name,
                 batch_size=400, learning_rate=3e-6, dropout_rates=[0.4, 0.4, 0.4], model_type='VAE',
                 from_json_fn=None):
        """
        n_latents: int: number of latent dimensions
        coord_set: jnp.array(): data from which both test and train set will be derived
        test_slice: int in (0,1,2,3,4): which 80/20 slice of coord_set to take for test and train
        data_dir: string: directory (with trailing slash) in which to store all output data, images, etc.
        batch_size: int: default 400; number of frames in a training batch
        learning_rate: float: default 1e-4; optax adam learning rate
        dropout_rates: list of floats: encoder and decoder dropout rates forward for encoder and reverse for decoder
        """
        #Get information about input data
        num_samples, input_size = coord_set.shape
        self.n_latents = n_latents
        
        #make hidden layers
        hidden_layers = [input_size]*3
        
        #make test and train sets
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]
        
        #Initialize Model
        assert model_type in ['VAE', 'AE']
        self.model_type = model_type
        if model_type == 'VAE':
            self.model = VAE(input_size=input_size, latents=self.n_latents,
                             hidden_layers=hidden_layers, dropout_rates=dropout_rates)
        elif model_type == 'AE':
            self.model = AE(input_size=input_size, latents=self.n_latents,
                            hidden_layers=hidden_layers, dropout_rates=dropout_rates)
        
        rng_init = jax.random.PRNGKey(self.n_latents)
        rng, key = jax.random.split(rng_init)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply,
                                              params=self.model.init(key, coord_set, rng_init)['params'],
                                              tx=optax.adam(learning_rate=learning_rate))
        #Initialize data_storage_file
        self.data_dir = data_dir
        self.model_name = model_name        
        self.data_file = open(self.data_dir + f'model_{self.model_name}_{self.n_latents:02d}.out', 'w')
        print(self.model)
        
        self.epoch = 0
        self.rmsd_loss = ([], [])
        self.pot_enr_loss = ([], [])
        self.summ_loss = ([], [])
        self.potential_coefficients = []
        
        num_train = self.train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        self.num_batches = num_complete_batches + bool(leftover)
        
        self.batches = data_stream(n_latents, num_train, self.num_batches, batch_size, self.train_data)
        print("INITIALIZATION COMPLETE")
        
    def write_traj(self, identifier, traj_xyz): #(n conf, n_atoms*3) OR (n conf, n_atoms, 3)
        fname = self.data_dir + f'{identifier}_{self.model_name}{self.n_latents:02d}.dcd'
        
        if traj_xyz.shape[-1] != 3:
            traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
        
        with md.formats.DCDTrajectoryFile(fname, 'w') as f:
            f.write (traj_xyz*10) #*10 because mdtraj loads data in nm but saves it in angstrom
    
    def write_decoded_traj(self, idfn=None):
        rng = jax.random.PRNGKey(self.epoch)
        rng, key = jax.random.split(rng)
        recon_test = self.state.apply_fn({'params':self.state.params}, self.test_data, rng)
        if idfn != None:
            self.write_traj(idfn, recon_test[0])
        else:
            self.write_traj(f"recon_test", recon_test[0])
    
    def eval_losses(self, rng, potential_coefficient):
        #After all batches seen this epoch
        recon_train = self.state.apply_fn({'params':self.state.params}, self.train_data, rng)
        recon_test = self.state.apply_fn({'params':self.state.params}, self.test_data, rng)

        self.rmsd_loss[0].append(atom_rmsd(self.train_data, recon_train[0]).mean())
        self.rmsd_loss[1].append(atom_rmsd(self.test_data, recon_test[0]).mean())

        self.pot_enr_loss[0].append(scaled_pot_enr_diff(self.train_data, recon_train[0]))
        self.pot_enr_loss[1].append(scaled_pot_enr_diff(self.test_data, recon_test[0]))

        self.summ_loss[0].append(summation_loss(self.train_data, recon_train[0], potential_coefficient))
        self.summ_loss[1].append(summation_loss(self.test_data, recon_test[0], potential_coefficient))
        
        most_recent_results = (self.rmsd_loss[0][-1], self.rmsd_loss[1][-1],
                               self.pot_enr_loss[0][-1], self.pot_enr_loss[1][-1],
                               self.summ_loss[0][-1], self.summ_loss[1][-1])
        
        return most_recent_results
    
    def save_loss_data(self):
        #LOSSES
        loss_names = ['RMSD', 'POTENTIAL', 'SUMM']
        for arr in (self.rmsd_loss, self.pot_enr_loss, self.summ_loss):
            np.save(self.data_dir + f'{self.model_name}{self.n_latents:02d}_{loss_names.pop(0)}.npy', np.array(arr))
        #LAMBDAS
        np.save(self.data_dir + f'{self.model_name}{self.n_latents:02d}_lambdas.npy', np.array(self.potential_coefficients))
    
    def train_rmsd_inner_loop(self):
        #train on batches
        for i in range(self.num_batches):
            #Get Batch
            batch = next(self.batches)
            #Train Batch
            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            self.state = rmsd_step(self.state, batch, rng)
    
    def train_potential_inner_loop(self):
        # train on batches
        for i in range(self.num_batches):
            # Get Batch
            batch = next(self.batches)
            # Train Batch
            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            self.state = potential_step(self.state, batch, rng)

    def train_summation_inner_loop(self, potential_coefficient):
        #train on batches 
        for i in range(self.num_batches):
            #Get Batch
            batch = next(self.batches)
            #Train Batch
            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            self.state = summation_step(self.state, batch, potential_coefficient, rng, weights=(1,1))
    
    def train_nepochs_on_rmsd(self, num_rmsd_epochs):
        """
        Train on the RMSD function alone, potential coefficient is zero
        """
        print('START RMSD')
        potential_coefficient = 0
        
        while self.epoch < num_rmsd_epochs:
            #Training
            self.train_rmsd_inner_loop()
            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            #After all batches seen this epoch
            rmsd_train_loss, rmsd_test_loss, pot_enr_train_loss, pot_enr_test_loss, summ_train_loss, summ_test_loss = self.eval_losses(rng, potential_coefficient)
            
            print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E'%rmsd_train_loss, '%.4E'%rmsd_test_loss,
                  'dPotEnr', '%.4E'%pot_enr_train_loss, '%.4E'%pot_enr_test_loss,
                  'Summation', '%.4E'%summ_train_loss, '%.4E'%summ_test_loss, 'L=%.4E'%potential_coefficient)
            self.potential_coefficients.append(potential_coefficient)
            self.epoch += 1
        print('END RMSD')

    def train_rmsd_threshold(self, nm_cutoff=0.1, num_move_ave=100, cutoff_epoch=25000):
        """Train on the RMSD until a nm_cutoff is reached of the last num_mov_ave epochs"""
        potential_coefficient=0
        while np.mean(self.rmsd_loss[-1][-num_move_ave:]) > nm_cutoff and self.epoch < cutoff_epoch:
            #Training
            self.train_rmsd_inner_loop()
            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            # After all batches seen this epoch evaluate losses
            rmsd_train_loss, rmsd_test_loss, pot_enr_train_loss, pot_enr_test_loss, summ_train_loss, summ_test_loss = self.eval_losses(rng, potential_coefficient)
            # Record Data
            print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E'%rmsd_train_loss, '%.4E'%rmsd_test_loss,
                  'dPotEnr', '%.4E'%pot_enr_train_loss, '%.4E'%pot_enr_test_loss,
                  'Summation', '%.4E'%summ_train_loss, '%.4E'%summ_test_loss, 'L=%.4E'%potential_coefficient)
            self.potential_coefficients.append(potential_coefficient)
            if self.epoch % 100 == 0:# Once every 100 epoch, dump all loss data to a file
                self.save_loss_data()
            self.epoch += 1
        return self.epoch

    def train_potential(self, potential_coefficient=1, potential_threshold=1e-3, num_mov_ave=50, cutoff_epoch=25000):
        potential_not_below_threshold = True
        potential_is_decreasing = True

        while (potential_not_below_threshold or potential_is_decreasing) and self.epoch < cutoff_epoch:
            # Training
            self.train_potential_inner_loop()

            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            # After all batches seen this epoch
            (rmsd_train_loss, rmsd_test_loss, pot_enr_train_loss, pot_enr_test_loss, summ_train_loss,
             summ_test_loss) = self.eval_losses(rng, potential_coefficient)

            print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E' % rmsd_train_loss, '%.4E' % rmsd_test_loss,
                  'dPotEnr', '%.4E' % pot_enr_train_loss, '%.4E' % pot_enr_test_loss,
                  'Summation', '%.4E' % summ_train_loss, '%.4E' % summ_test_loss, 'L=%.4E' % potential_coefficient)

            self.potential_coefficients.append(potential_coefficient)
            if self.epoch % 100 == 0:  # Periodically save loss
                self.save_loss_data()
            if self.epoch % 500 == 0:  # Periodically save a traj
                self.write_decoded_traj()
            self.epoch += 1
            # Should break loop?
            doub_mov_ave = 2 * num_mov_ave
            potential_not_below_threshold = (np.mean(self.pot_enr_loss[-1][-num_mov_ave:]) > potential_threshold or
                                             np.mean(self.pot_enr_loss[0][-num_mov_ave:]) > potential_threshold)  # Test and train should be below the threshold
            potential_is_decreasing = np.mean(self.pot_enr_loss[-1][-doub_mov_ave:-num_mov_ave]) > np.mean(
                self.pot_enr_loss[-1][-num_mov_ave:])  # Test should continue decreasing
        return self.epoch

    def train_scaling_potential(self, potential_coefficient=0, cutoff_epoch=25000):
        """Scale the potential in by frequently changing the coefficient to make potential equal to rmsd"""

        # Every ten epochs choose lambda as min(1, max(lambda[-1], RMSD/NSD))
        while potential_coefficient != 1 and self.epoch < cutoff_epoch:
            # Sometime check to see if lambda can be larger
            if self.epoch % 10 == 0:
                potential_coefficient = np.min(
                    (1, np.max((potential_coefficient, (self.rmsd_loss[0][-1] / self.pot_enr_loss[0][-1])))))
            # Training
            self.train_summation_inner_loop(potential_coefficient)
            # After all batches seen this epoch
            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            (rmsd_train_loss, rmsd_test_loss, pot_enr_train_loss, pot_enr_test_loss, summ_train_loss,
             summ_test_loss) = self.eval_losses(rng, potential_coefficient)

            print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E' % rmsd_train_loss, '%.4E' % rmsd_test_loss,
                  'dPotEnr', '%.4E' % pot_enr_train_loss, '%.4E' % pot_enr_test_loss,
                  'Summation', '%.4E' % summ_train_loss, '%.4E' % summ_test_loss, 'L=%.4E' % potential_coefficient)

            self.potential_coefficients.append(potential_coefficient)
            if self.epoch % 100 == 0:
                self.save_loss_data()
            self.epoch += 1
        return self.epoch

    def train_summation(self, potential_coefficient=1, potential_threshold=1e-3, num_mov_ave=50, cutoff_epoch=25000):
        potential_not_below_threshold = True
        potential_is_decreasing = True

        while (potential_not_below_threshold or potential_is_decreasing) and self.epoch < cutoff_epoch:
            # Training
            self.train_summation_inner_loop(potential_coefficient)

            rng = jax.random.PRNGKey(self.epoch)
            rng, key = jax.random.split(rng)
            # After all batches seen this epoch
            (rmsd_train_loss, rmsd_test_loss, pot_enr_train_loss, pot_enr_test_loss, summ_train_loss,
             summ_test_loss) = self.eval_losses(rng, potential_coefficient)

            print('epoch', self.epoch, 'atom_rmsd_nm', '%.4E' % rmsd_train_loss, '%.4E' % rmsd_test_loss,
                  'dPotEnr', '%.4E' % pot_enr_train_loss, '%.4E' % pot_enr_test_loss,
                  'Summation', '%.4E' % summ_train_loss, '%.4E' % summ_test_loss, 'L=%.4E' % potential_coefficient)

            self.potential_coefficients.append(potential_coefficient)
            if self.epoch % 100 == 0:  # Periodically save loss
                self.save_loss_data()
            if self.epoch % 500 == 0:  # Periodically save a traj
                self.write_decoded_traj()
            self.epoch += 1
            # Should break loop?
            doub_mov_ave = 2*num_mov_ave
            potential_not_below_threshold = (np.mean(self.pot_enr_loss[-1][-num_mov_ave:]) > potential_threshold or
                                                np.mean(self.pot_enr_loss[0][-num_mov_ave:]) > potential_threshold)  # Test and train should be below the threshold
            potential_is_decreasing = np.mean(self.pot_enr_loss[-1][-doub_mov_ave:-num_mov_ave]) > np.mean(self.pot_enr_loss[-1][-num_mov_ave:])  # Test should continue decreasing
        return self.epoch

    def train_model(self, nm_cutoff=0.1, potential_threshold=1e-3, cutoff_epoch=25000):
        """ CURRENT MAIN USAGE CASE """
        # RMSD BLOCK 1
        # Train on RMSD until average of last 100 epochs <1 angstrom
        # Get first 100 vals
        print('START RMSD')
        self.train_nepochs_on_rmsd(100)
        self.save_loss_data()
        
        # RMSD BLOCK 2
        # Train until last 100 vals average less than predefined cutoff, always make sure we never train longer than cutoff_epoch
        begin_scaling_epoch = self.train_rmsd_threshold(nm_cutoff=nm_cutoff, num_move_ave=100, cutoff_epoch=cutoff_epoch)
        
        # SCALE IN POTENTIAL BLOCK
        print('START SCALING POTENTIAL')
        end_scaling_epoch = self.train_scaling_potential(cutoff_epoch=cutoff_epoch)
        print('END SCALING POTENTIAL')
        
        #TRAIN ON POTENTIAL BLOCK (MAINTAIN RMSD IF THIS IS AE, DROP THAT TERM IF IT IS VAE
        if self.model_type == 'VAE':
            self.train_potential(potential_threshold=potential_threshold, cutoff_epoch=cutoff_epoch)
        elif self.model_type == 'AE':
            self.train_summation(potential_coefficient=self.potential_coefficients[-1], potential_threshold=potential_threshold, cutoff_epoch=cutoff_epoch)
        print('END TRAINING')
        return begin_scaling_epoch, end_scaling_epoch
        
    
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


if __name__ == "__main__":
    json_fn = sys.argv[1]
    with open(json_fn, 'r') as g:
        params = json.load(g)

    jax.config.update("jax_enable_x64", True)
    #Files
    fname_pdb = params["fname_pdb"]
    fname_dcd = params["fname_dcd"]
    fname_prmtop = params["fname_prmtop"]
    gas_fun, _, _, _ = get_amber_gas_energy_functions(fname_prmtop)
    #Parameters
    latent_dim = params["latent_dim"] #number of latents
    test_slice = params["test_slice"] #int zero to four inclusive for 20/80 split of test and train
    start, stop = params["data_slice_start"], params["data_slice_end"] #Slice of data
    #Directories
    model_name = params["model_name"] #string for the model name
    save_dir = params["save_dir"] #with trailing slash:
    model_dir = save_dir + f'{model_name}/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if params["data_dir"] == 'None':
        latent_dir = model_dir + f'{latent_dim:02d}_latents/'
        data_dir = latent_dir + f'rpt_{test_slice}/'
        if not os.path.isdir(latent_dir):
            os.mkdir(latent_dir)
    else:
        data_dir = model_dir + params['data_dir']
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    #Data
    c = md.load(fname_dcd, top=fname_pdb)
    c = c.superpose(c) # FEED IN ALIGNED DATA
    my_rmsd = md.rmsd(c, c)
    coords = jnp.array(c.xyz[start:stop].reshape(stop-start, -1))
    print('OG Data shape', c.xyz.shape, 'Data 4 VAE shape', coords.shape)
    
    #Init Experiment
    experiment = NN_Experiment(latent_dim, coords, test_slice, data_dir, model_name,
                               batch_size=params["batch_size"], learning_rate=params["learning_rate"],
                               dropout_rates=params["dropout_rates"], model_type=params["model_type"])
    #Save Testing Data
    experiment.write_traj("test_data", experiment.test_data)
    # Automatic training
    begin_scaling_epoch, end_scaling_epoch = experiment.train_model(nm_cutoff=0.15, potential_threshold=params["potential_threshold"], cutoff_epoch=params["max_epoch"])
    experiment.save_loss_data()
    experiment.graph_losses(begin_scaling_epoch=begin_scaling_epoch, end_scaling_epoch=end_scaling_epoch)
    #Obtain Decoded Trajectory
    experiment.write_decoded_traj('final_recon_test')