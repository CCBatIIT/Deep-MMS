import flax, jax, optax, sys, os, json, pickle, time, glob
import flax.linen as nn
import jax.numpy as jnp
from jax.scipy.special import digamma
from flax.training import train_state, orbax_utils
import orbax.checkpoint
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import mdtraj as md
from datetime import datetime
import netCDF4 as nc
from typing import Any
from sklearn.feature_selection import mutual_info_regression
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

####################################################################
#    UTILITIES SECTION
####################################################################
jax.config.update("jax_enable_x64", True)
printf = lambda x : print(f"{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}//{x}", flush=True)

def mass_weights(traj):
    import mdtraj as md
    import numpy as np
    H = md.element.hydrogen
    traj_heavy = traj.atom_slice(traj.top.select('not element H'))
    masses = np.array([traj.top.atom(i).element.mass for i in range(traj.n_atoms)])
    index_map = np.array([[i, 0] for i in range(traj_heavy.n_atoms)])
    info = lambda atom: (atom.residue.name, atom.residue.index, atom.name, atom.element.mass)
    for i in range(traj_heavy.n_atoms):
        for j in range(traj.n_atoms):
            atom_i, atom_j = traj_heavy.top.atom(i), traj.top.atom(j)
            if info(atom_i) == info(atom_j):
                index_map[i, 1] = j
                break
            else: 
                continue
    
    assert np.allclose(traj.xyz[:, index_map[:, 1]], traj_heavy.xyz[:, index_map[:, 0]])
    
    heavy_masses = np.array([traj_heavy.top.atom(i).element.mass for i in range(traj_heavy.n_atoms)])
    assert np.all(heavy_masses[index_map[:, 0]] == masses[index_map[:, 1]])

    from copy import deepcopy
    mass_unified = deepcopy(heavy_masses)
    mass_valence = np.ones(heavy_masses.shape)
    
    for bond in traj.top.bonds:
        if bond.atom1.element == H or bond.atom2.element == H:
            #print(bond.atom1.index, bond.atom1.element, bond.atom2.index, bond.atom2.element)
            #if so, add the mass of hydrogen to the mass of the heavy atom (unified model)
            mass_unified[np.where(index_map[:, 1] == bond.atom1.index)[0]] += masses[bond.atom2.index]
            #also if so, increment the valence model of the heavy atom by one
            mass_valence[np.where(index_map[:, 1] == bond.atom1.index)[0]] += 1

    return {'Uniform': np.ones(traj.n_atoms), 'Uniform_Heavy': np.ones(traj_heavy.n_atoms),
            'Mass': masses, 'Mass_Heavy': heavy_masses, 'Mass_United': mass_unified, 'H-Valence': mass_valence}
    
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Data_stream():
    """
    A class for batching data into random permutations
    """
    def __init__(self, rng_seed, num_total, num_batches, batch_size, data):
        self.rng_seed = rng_seed
        self.num_total = num_total
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = data
        
    def __iter__(self):
        rng = npr.RandomState(self.rng_seed)
        while True:
            perm = rng.permutation(self.num_total)
            for i in range(self.num_batches):
                batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                yield self.data[batch_idx]


def reparameterize(z_rng, z_mean, z_logvar):
    z_std = jnp.exp(0.5*z_logvar)
    z_eps = jax.random.normal(z_rng, z_logvar.shape)
    return z_mean + z_eps*z_std



####################################################################
#    NEURAL NETWORK SECTION
####################################################################
class BatchNorm_VAE(nn.Module):
    input_size: int
    hidden_layers: tuple
    latents: int
    dropout_rates: list
    is_batchnorm: bool
        
    def setup(self):
        self.encoder = BVEncoder(list(self.hidden_layers), self.latents, self.dropout_rates, self.is_batchnorm)
        self.decoder = BVDecoder(list(self.hidden_layers), self.input_size, self.dropout_rates, self.is_batchnorm)

    def __call__(self, x, z_rng, train:bool):
        z_mean, z_logvar = self.encoder(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train), z_mean, z_logvar
    
    def construct(self, z_mean, z_logvar, z_rng, train=False):
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train)
    
    def encode(self, x, z_rng, train=False):
        return self.encoder(x, train=train)
    
    def decode(self, z, z_rng, train=False):
        return self.decoder(z, train=train)

#BatchNorm Variational AutoEncoder
class BVEncoder(nn.Module):
    d_hidden: list
    latents: int
    dropout_rates: list
    is_batchnorm: bool
    
    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(len(self.d_hidden)):
            x = nn.Dense(self.d_hidden[i])(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            if self.is_batchnorm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            #x = nn.leaky_relu(x, negative_slope=0.2)
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=not train)
        mean_x = nn.Dense(self.latents, name='fc5_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc5_logvar')(x)
        return mean_x, logvar_x 

class BVDecoder(nn.Module):
    d_hidden: list
    out_dim: int
    dropout_rates: list
    is_batchnorm: bool

    @nn.compact
    def __call__(self, z, train: bool):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.Dense(self.d_hidden[i])(z)
            z = nn.leaky_relu(z, negative_slope=0.2)
            if self.is_batchnorm:
                z = nn.BatchNorm(use_running_average=not train)(z)
            #z = nn.leaky_relu(z, negative_slope=0.2)
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=not train)
        z = nn.Dense(self.out_dim, name='f5')(z)
        return z

        
####################################################################
#    LOSS SECTION
####################################################################
@jax.vmap
def atom_rmsd(a, b):
    """
    Atom RMSD of vectorized frames a and b
    Due to vmapping does not work on individual frames, but only collections of frames
    """
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    return jnp.sqrt(jnp.mean(jnp.sum((b - a)**2, axis=1)))


def give_weighted_rmsd_func(weights):
    def weighted_atom_rmsd(a, b):
        a, b = a.reshape(-1, 3), b.reshape(-1, 3)
        return jnp.sqrt(jnp.mean(weights*jnp.sum((b - a)**2, axis=1)))
    weighted_atom_rmsd = jax.vmap(weighted_atom_rmsd, in_axes=(0,0))
    return weighted_atom_rmsd

# #KL Divergence between a set of means stds against standard normal distributions
# KL_loss = lambda mus, log_vars: 0.5 * jnp.sum(mus**2 + jnp.exp(log_vars) - log_vars - 1)
# KL_loss = jax.jit(KL_loss)
#Mutual Information Regression for the latents (should be minimized) - take the maximum as loss value
# def pairwise_linf_distances(X):
#     """Compute pairwise L∞ distances (Chebyshev norm)."""
#     X = X[:, None, :]  # (N, 1, D)
#     Y = X.transpose((1, 0, 2))  # (1, N, D)
#     return jnp.max(jnp.abs(X - Y), axis=2)  # (N, N)

# def knn_mutual_information(x, y, k=5):
#     N = x.shape[0]
#     z = jnp.concatenate([x, y], axis=1)

#     dists_z = pairwise_linf_distances(z) + jnp.eye(N) * 1e10
#     epsilons = jnp.sort(dists_z, axis=1)[:, k - 1]

#     dists_x = pairwise_linf_distances(x)
#     dists_y = pairwise_linf_distances(y)

#     def count_neighbors(dists, eps):
#         return jnp.sum(dists < eps - 1e-10)

#     n_x = jax.vmap(count_neighbors, in_axes=(0, 0))(dists_x, epsilons)
#     n_y = jax.vmap(count_neighbors, in_axes=(0, 0))(dists_y, epsilons)

#     return digamma(k) + digamma(N) - jnp.mean(digamma(n_x + 1) + digamma(n_y + 1))

# def knn_mi_batch(Z, k=5):
#     """
#     Estimate mutual information between columns of Z[:, 0] and Z[:, 1]
#     where Z is shape (N_samples, 2)
#     """
#     x = Z[:, 0:1]
#     y = Z[:, 1:2]
#     return knn_mutual_information(x, y, k)

# def compute_mi_matrix_vmap(X, k=5):
#     """
#     Vectorized computation of MI matrix between all feature pairs in X.

#     Args:
#         X: jnp.ndarray of shape (N_samples, n_features)
#         k: KNN parameter for MI estimation

#     Returns:
#         MI matrix: shape (n_features, n_features)
#     """
#     n_samples, n_features = X.shape

#     # Get upper triangle indices (i < j)
#     i_indices, j_indices = jnp.triu_indices(n_features, k=1)
#     n_pairs = i_indices.shape[0]

#     # Create (n_pairs, N_samples, 2) array of feature pairs using vmap
#     def extract_pair(i, j):
#         xi = X[:, i]
#         xj = X[:, j]
#         return jnp.stack([xi, xj], axis=-1)  # (N_samples, 2)

#     Z_pairs = jax.vmap(extract_pair)(i_indices, j_indices)  # (n_pairs, N_samples, 2)

#     # vmap MI computation
#     mi_values = jax.vmap(lambda z: knn_mi_batch(z, k))(Z_pairs)  # (n_pairs,)
#     return jnp.max(mi_values)
#     # # Fill symmetric MI matrix
#     # MI = jnp.zeros((n_features, n_features))
#     # MI = MI.at[i_indices, j_indices].set(mi_values)
#     # MI = MI.at[j_indices, i_indices].set(mi_values)

#     #return MI

#MI_loss = jax.jit(lambda latent_means: compute_mi_matrix_vmap(latent_means, k=5))


# @jax.jit
# def step(state, batch_x, z_rng, dropout_key):
#     dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
#     def loss_fn(params):
#         #Logits is the output of calling the NN (Decoded, Latent_Means, Latent_Vars)
#         logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
#                                          batch_x, z_rng, train=True,
#                                          rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
#         #Loss term representing the Root Mean Square reconstruction error
#         loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0])**2)))
#         #Loss term representing the KL Divergence between latent space and standard normals
#         #loss += KL_loss(logits[1], logits[2])
#         #Loss term representing the MI between latent Dimensions
#         #loss += MI_loss(logits[1])
#         return loss, (logits, updates)
#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (loss, (logits, updates)), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     state = state.replace(batch_stats=updates['batch_stats'])
#     return state, loss

# @jax.jit
# def evaluate(state, batch_x, z_rng, dropout_key):
#     dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
#     def loss_fn(params):
#         #Logits is the output of calling the NN (Decoded, Latent_Means, Latent_Vars)
#         logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
#                                          batch_x, z_rng, train=False,
#                                          rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
#         #Loss term representing the Root Mean Square reconstruction error
#         rmsd_term = jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0])**2))
#         #Loss term representing the KL Divergence between latent space and standard normals
#         #KL_term = KL_loss(logits[1], logits[2])
#         #Loss term representing the MI between latent Dimensions
#         #MI_term = MI_loss(logits[1])
#         return (rmsd_term), (logits, updates)
#     return loss_fn(state.params)[0]


####################################################################
#    MAIN CUSTOM CLASS SECTION
####################################################################

class HeavyAtom_NN_Experiment():
    def __init__(self, json_fn, make_dirs=True, from_json_params=False):
        """
        json_fn = string or dictionary
            If string, from_json_params should be false: attempts to load json file from the string provided
            If dictionary, from_json_params should be True: attempts to use the dictionary as the json input
        make_dirs = bool
        from_json_params = bool: see above
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
        fname_dcd = self.json_params["fname_dcd"] #DCD with coordinates for mdtraj
        fname_top = self.json_params["fname_topology"] #Topology file for mdtraj
        self.n_latents = self.json_params["latent_dim"] #number of latents
        test_slice = self.json_params["test_slice"] #int zero to four inclusive for 20/80 split of test and train
        self.model_name = self.json_params["model_name"] #string for the model name
        save_dir = self.json_params["save_dir"] #with trailing slash:
        data_start, data_end = self.json_params["data_slice_start"], self.json_params["data_slice_end"] #Slice of data
        if data_end == 'None':
            data_end = None
        self.batch_size = self.json_params["batch_size"]
        learning_rate = self.json_params["learning_rate"]
        dropout_rates = self.json_params["dropout_rates"]
        resume = self.json_params["resume_latest"]
        self.is_batchnorm = self.json_params["is_batchnorm"]
        
        #Establish Directories
        printf('Establish Directories')
        model_dir = os.path.join(save_dir, f'{self.model_name}/')
        latent_dir = os.path.join(model_dir, f'{self.n_latents:04d}_latents/')
        self.data_dir = os.path.join(latent_dir, f'rpt_{test_slice}/')
        if "data_dir" in self.json_params.keys():
            if self.json_params["data_dir"] is not None:
                self.data_dir = self.json_params["data_dir"]
        
        if not os.path.isdir(self.data_dir) and make_dirs:
            os.makedirs(self.data_dir, exist_ok=True)
        
        #Load and Align
        printf('Load Data with MDTraj')
        c = md.load(self.json_params['fname_dcd'], top=self.json_params['fname_topology'])
        c = c.atom_slice(c.topology.select(self.json_params["atom_selection"]))
        c = c.superpose(c) # FEED IN ALIGNED DATA
        coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))[data_start:data_end]
        num_samples, input_size = coord_set.shape
        
        printf('Calculate Mass Weighting Schemes')
        mass_sets = mass_weights(c)

        #Make Test and Train Sets
        printf('Batch data')
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]
        printf((self.train_data.shape, self.test_data.shape))
        
        #Initialize Model
        printf('Model Init')
        from .NN_constructor import make_model_and_state
        
        if 'weight_model' in self.json_params.keys():
            weight_model = self.json_params['weight_model']
            assert weight_model in mass_sets.keys()
        else:
            weight_model = 'Uniform_Heavy'
        printf(f'\t Using {weight_model=}')
        weights = jnp.array(mass_sets[weight_model])
        printf('Make Loss Function')
        atom_rmsd_loss = give_weighted_rmsd_func(weights)
        printf('Make VAE')
        global step, evaluate
        self.model, self.state, step, evaluate = make_model_and_state(self, dropout_rates, coord_set, learning_rate, atom_rmsd_loss)
        step, evaluate = jax.jit(step), jax.jit(evaluate)
        
        self.epoch = 0

        #Checkpointer
        printf('Checkpointer')
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=self.json_params['checkpoint_interval'])
        manager_dir = os.path.join(self.data_dir, 'checkpoint_managed')
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager(manager_dir, self.orbax_checkpointer, options)

        #Put Data into Batches
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
            self.load_model_from_ckpt(ckpt_fn)
        else:
            self.rootgrp = self.establish_netcdf(self.nc_data_file)
        
        #Provide some final information
        printf(f'Epoch {self.epoch}')
        printf(f'NC {self.rootgrp}')
        printf(f'Model {self.model}')
        printf(f"INITIALIZATION COMPLETE")
    

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
                rmsd_term = grp.createVariable('RMSD_Loss_Term', 'f4', ('epoch', 'batch',))
                rmsd_term.units = 'Nanometer'
                #KL_term = grp.createVariable('KL_Loss_Term', 'f4', ('epoch', 'batch',))
                #KL_term.units = 'Unitless'
                #MI_term = grp.createVariable('MI_Loss_Term', 'f4', ('epoch', 'batch',))
                #MI_term.units = 'Unitless'
                grp.history = "Created" + time.ctime(time.time())
            
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
        fname = os.path.join(self.data_dir, f'{identifier}_{self.model_name}{self.n_latents:04d}.dcd')
        
        if traj_xyz.shape[-1] != 3:
            traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
        
        with md.formats.DCDTrajectoryFile(fname, 'w') as f:
            f.write(traj_xyz*10) #*10 because mdtraj loads data in nm but DCD saves it in angstrom

    
    def train_on_batches(self, batch_set):
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            #Train Batch
            root_key = jax.random.PRNGKey(self.epoch)
            main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
            self.state, _ = step(self.state, batch, main_key, dropout_key)
        #After ANY EPOCH
        save_args = orbax_utils.save_args_from_target(self.state)
        self.checkpoint_manager.save(self.epoch, self.state, save_kwargs={'save_args': save_args})
        if self.epoch > 0 and self.epoch % 100 == 0:
            self.checkpoint_netcdf()

    
    def eval_batches(self, batch_set):
        vals = jnp.empty((batch_set.num_batches, len(self.rootgrp['Train'].variables)))
        
        f = iter(batch_set)
        for i in range(batch_set.num_batches):
            #Get Batch
            batch = next(f)
            root_key = jax.random.PRNGKey(self.epoch)
            main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)
            #Eval Batch
            rmsd_term = evaluate(self.state, batch, main_key, dropout_key)
            vals = vals.at[i, :].set([rmsd_term])
        return vals
        
    
    def evaluate_loss(self):
        guide = {'Train': self.train_batches, 'Test': self.test_batches}
        most_recent_results = []
        for grp_key in ['Train', 'Test']:
            vals = self.eval_batches(guide[grp_key])
            self.rootgrp[grp_key].variables['RMSD_Loss_Term'][self.epoch, :] = vals[:, 0]
            #self.rootgrp[grp_key].variables['KL_Loss_Term'][self.epoch, :] = vals[:, 1]
            #self.rootgrp[grp_key].variables['MI_Loss_Term'][self.epoch, :] = vals[:, 2]
            most_recent_results.append(vals)
        return most_recent_results
    
    
    def verbose_print(self, last_result, epoch_time, verbose=True):
        if verbose == True:
            printf(f"Epoch {self.epoch:6d} took: {epoch_time}")
            printf(f"\t\tRMSD_Term\tTrain={jnp.mean(last_result[0][:,0]):.3E}\tTest={jnp.mean(last_result[1][:,0]):.3E}")
            #printf(f"\t\t KL_Term \tTrain={jnp.mean(last_result[0][:,1]):.3E}\tTest={jnp.mean(last_result[1][:,1]):.3E}")
            #printf(f"\t\t MI_Term \tTrain={jnp.mean(last_result[0][:,2]):.3E}\tTest={jnp.mean(last_result[1][:,2]):.3E}")
        elif type(verbose) == int and self.epoch % verbose == 0:
            printf(f"Epoch {self.epoch:6d} took: {epoch_time}")
            printf(f"\t\tRMSD_Term\tTrain={jnp.mean(last_result[0][:,0]):.3E}\tTest={jnp.mean(last_result[1][:,0]):.3E}")
            #printf(f"\t\t KL_Term \tTrain={jnp.mean(last_result[0][:,1]):.3E}\tTest={jnp.mean(last_result[1][:,1]):.3E}")
            #printf(f"\t\t MI_Term \tTrain={jnp.mean(last_result[0][:,2]):.3E}\tTest={jnp.mean(last_result[1][:,2]):.3E}")
        return None
            
    
    def train_n_epochs(self, n_epochs, verbose=True):
        while self.epoch < n_epochs:
            #Training
            epoch_start = datetime.now()
            self.train_on_batches(self.train_batches)
            #After all batches seen this epoch
            self.verbose_print(self.evaluate_loss(), datetime.now() - epoch_start, verbose=verbose)
            self.epoch += 1
        return self.epoch
    
    
    def train_to_auto_stop(self, cutoff_epoch=100000, verbose=True):
        """
        Train until overtraining is detected
        """
        should_early_stop = False

        while not should_early_stop and self.epoch < cutoff_epoch:
            #Training
            epoch_start = datetime.now()
            self.train_on_batches(self.train_batches)
            #After all batches seen this epoch
            self.verbose_print(self.evaluate_loss(), datetime.now() - epoch_start, verbose=verbose)
            self.epoch += 1
            
            #Obtain the last 50 losses
            last_50_train_total = np.mean(self.rootgrp['Train']['RMSD_Loss_Term'][-50:, :], axis=1)
            last_50_test_total = np.mean(self.rootgrp['Test']['RMSD_Loss_Term'][-50:, :], axis=1)
            
            #If the last 50 epoch test losses are grth 0.1pm/epoch (incur 0.05 Angstrom Loss)
            test_rising = np.polyfit(np.arange(50), last_50_test_total, 1)[0] > 0.0001
            #If the average test value of the last 50 epochs is more than 2.5% of the train value
            test_greater_than_train = np.mean(last_50_test_total) > 1.025*np.mean(last_50_train_total)
            #If the test loss values are stable (std of most recent 50 epochs < 1 loss unit)
            stable_vals = np.std(last_50_test_total) < 1
            #If all are true, invoke early stopping
            if False not in [test_rising, test_greater_than_train, stable_vals]:
                should_early_stop = True
        return self.epoch
    
    
    def MAIN_train(self, n_epochs=1000, cutoff_epoch=None, verbose=True):
        if cutoff_epoch is None:
            #Default to the one in the json
            cutoff_epoch = self.json_params['max_epoch']
        self.train_n_epochs(n_epochs, verbose=verbose)
        self.train_to_auto_stop(cutoff_epoch=cutoff_epoch, verbose=verbose)
        return self.epoch

