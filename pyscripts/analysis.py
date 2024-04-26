import jax

jax.config.update("jax_enable_x64", True)
jax.print_environment_info(), jax.default_backend()

@jax.vmap
def atom_rmsd(a, b, potential_coefficient=None): # for arrays of (n_conf, n_atom*3)
    mn = a.shape[-1]//3
    x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
    return jnp.sqrt(jnp.mean((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))

@jax.jit
def scaled_pot_enr_diff(a, b, potential_coefficient=None): # WITH A AS BATCH AND B AS RECON
    return ((gas_fun(a) - gas_fun(b))/gas_fun(a))**2 #Unitless quantity

@jax.jit
def summation_loss(a, b, potential_coefficient): # LET A BE BATCH AND B BE RECON
    # Make this the square root of the mean of the sum of squares of elements
    return atom_rmsd(a,b) + potential_coefficient * scaled_pot_enr_diff(a, b)

@jax.jit
def rmsd_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return jnp.sqrt(jnp.sum(atom_rmsd(batch_x, recon_x)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def rmsd_log_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return jnp.log(jnp.sqrt(jnp.sum(atom_rmsd(batch_x, recon_x)**2)))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def potential_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return scaled_pot_enr_diff(batch_x, recon_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)    

@jax.jit
def summation_step(state, batch_x, z_rng, potential_coefficient, weights=(1,1)):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return summation_loss(batch_x, recon_x, potential_coefficient).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)


import numpy.random as npr
import flax.linen as nn

class Data_stream():
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


#VAE Classes
class V_Encoder(nn.Module):
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
    
    def __call__(self, x, z_rng):
        z_latent = self.encoder(x)
        return self.decoder(z_latent), z_latent
    
    def encode(self, x, z_rng):
        return self.encoder(x)
    
    def decode(self, z, z_rng):
        return self.decoder(z)
        
class VAE(nn.Module):
    input_size: int
    hidden_layers: tuple
    dropout_rates: list
    latents: int
        
    def setup(self):
        self.encoder = V_Encoder(list(self.hidden_layers), self.latents, self.dropout_rates)
        self.decoder = Decoder(list(self.hidden_layers), self.input_size, self.dropout_rates)

    def __call__(self, x, z_rng):
        z_mean, z_logvar = self.encoder(x)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z), z_mean, z_logvar
    
    def construct(self, z_mean, z_logvar, z_rng):
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z)
    
    def encode(self, x, z_rng):
        return self.encoder(x)
    
    def decode(self, z, z_rng):
        return self.decoder(z)
    
    def mvn_latent_model(self, x, z_rng):
        return NotImplemented()


import json, orbax.checkpoint, optax, os
import pyscripts.jax_amber3 as jaa
import mdtraj as md
import jax.numpy as jnp
import numpy as np
from flax.training import train_state, orbax_utils
import netCDF4 as nc
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class NN_Experiment_Analyzer():

    def __init__(self, json_fn, netcdf_fn, checkpoint_dir):

        if type(json_fn) == dict:
            #Json has already been parsed
            self.json_params = json_fn
        elif type(json_fn) == str:
            #Attempt to parse
            with open(json_fn, 'r') as g:
                self.json_params = json.load(g)
        else:
            raise Exception('Json Load Error: json_fn should be dict or string')
        
        #Files
        print('Load Files')
        print(self.json_params)
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
        model_type = self.json_params["model_type"]
        coordinate_scheme = self.json_params["coordinate_scheme"]
        self.model_name = model_name
        resume = self.json_params["resume_latest"]

        #Load and Align
        print('Load Data with MDTraj')
        c = md.load(fname_dcd, top=fname_prmtop)
        c = c.superpose(c) # FEED IN ALIGNED DATA
        coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))
        num_samples, input_size = coord_set.shape
        
        global gas_fun
        gas_fun, _ = jaa.get_amber_functions(fname_prmtop)

        #make test and train sets
        print('Batch data')
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]
        print(self.train_data.shape, self.test_data.shape)
        
        #Initialize Model
        print('Model Init')
        assert model_type in ['VAE', 'AE']
        self.model_type = model_type
        #make hidden layers
        hidden_layers = [input_size]*3
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

        print('Batch Data')
        num_train = self.train_data.shape[0]
        num_complete_batches, leftover = divmod(num_train, self.batch_size)
        self.num_train_batches = num_complete_batches + bool(leftover)
        self.train_batches = Data_stream(self.n_latents, num_train, self.num_train_batches, self.batch_size, self.train_data)
        
        num_test = self.test_data.shape[0]
        num_complete_batches, leftover = divmod(num_test, self.batch_size)
        self.num_test_batches = num_complete_batches + bool(leftover)
        self.test_batches = Data_stream(self.n_latents, num_test, self.num_test_batches, self.batch_size, self.test_data)

        print('Read NetCDF and Checkpoint')
        self.rootgrp = nc.Dataset(netcdf_fn, 'r', format='NETCDF4')
        self.state = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_dir, item=self.state)

    def plot_results(self, print_only=False, yscale=None, xscale=None):
        
        traingrp = self.rootgrp['Train']
        testgrp = self.rootgrp['Test']
        keys = ['RMSD', 'Potential']
        if not print_only:
            for foo in keys:
                plt.clf()
                plt.title(foo + f' {self.n_latents} Latents')
                for grp in [traingrp, testgrp]:
                    _ = plt.plot(np.arange(grp.variables[foo][:, :].shape[0]),
                                 np.mean(grp.variables[foo][:, :], axis=1))
                plt.xlabel('Epoch')
                if foo == 'Potential':
                    plt.yscale('log')
                    plt.ylabel('Potential (kJ/mol)')
                
                elif foo == 'RMSD':
                    plt.ylabel('RMSD (nm)')
    
                if xscale:
                    plt.xscale(xscale)
                if yscale:
                    plt.yscale(yscale)
                
                plt.legend(['Train','Test'])
                plt.show()
        
        print(f'Num Epochs {traingrp.variables['RMSD'][:, :].shape[0]}')
        print(f'Last RMSD (nm): Train: {np.mean(traingrp.variables['RMSD'][-1, :])} Test {np.mean(testgrp.variables['RMSD'][-1, :])}')
        print(f'Last PotDiff (% Deviation): Train: {np.mean(traingrp.variables['Potential'][-1, :])} Test {np.mean(testgrp.variables['Potential'][-1, :])}')

    def operate(self, rng_seed):
        rng = jax.random.PRNGKey(rng_seed)
        rng, key = jax.random.split(rng)
        decoded, latent = self.state.apply_fn({'params':self.state.params}, self.test_data, rng)
        return decoded, latent

    def decode(self, rng_seed, data):
        rng = jax.random.PRNGKey(rng_seed)
        rng, key = jax.random.split(rng)
        decoded = self.model.apply({'params': self.state.params}, data, rng, method=self.model.decode)
        return decoded
    
    def plot_latents_sequential(self, rng_seed):
        decoded, latents = self.operate(rng_seed)
        for i in range(self.n_latents):
            plt.clf()
            _ = plt.hist(latents[:, i], bins=50)
            plt.title(f'Latent {i}')
            plt.show()

    def plot_latents_grid(self, rng_seed, figsize=(15, 10)):
        decoded, latents = self.operate(rng_seed)
        fig, axs = plt.subplots(self.n_latents, self.n_latents, figsize=figsize, sharex='col')
        for i in range(self.n_latents):
            for j in range(self.n_latents):
                #axs[i,j].scatter(latent[:,j], latent[:,i])
                if i > j:
                    axs[i,j].scatter(latents[:,j], latents[:,i])
                elif i == j:
                    axs[i,j].hist(latents[:, i], bins=25)

    def plot_Akaike_Bayes(self, rng_seed, min_comp=1, max_comp=30):
        decoded, latents = self.operate(rng_seed)
        ics = []
        for i in range(min_comp, max_comp+1):
            MM = GaussianMixture(n_components=i).fit(latents)
            ics.append((i, MM.aic(latents), MM.bic(latents)))
        ics = np.array(ics)
        plt.clf()
        _ = plt.plot(ics[:, 0], ics[:, 1])
        _ = plt.plot(ics[:, 0], ics[:, 2])
        plt.legend(('Akaike Info Criterion', 'Bayes Info Criterion'))
        plt.xlabel('Num Components')
        plt.show()

    def gaussian_mm_fit(self, rng_seed, num_components, num_samples='Auto'):
        decoded, latents = self.operate(rng_seed)
        MM = GaussianMixture(n_components=5).fit(latents) #chosen based on above graph
        if num_samples == 'Auto':
            samples = MM.sample(latents.shape[0])[0]
        else:
            samples = MM.sample(num_samples)[0]
        return MM, samples

    def plot_GMM_w_latents(self, rng_seed, num_components):
        decoded, latents = self.operate(rng_seed)
        MM, samples = self.gaussian_mm_fit(rng_seed, num_components)
        
        for i in range(samples.shape[-1]):
            plt.clf()
            _ = plt.hist(latents[:,i], bins=25, histtype='step', color='g')
            _ = plt.hist(samples[:,i], bins=25, histtype='step', color='b')
            plt.legend(('Encoded from Test', 'Sampled from Mixture Model'))
            plt.title(f'Latent {i}')
            plt.show()

    def write_traj(self, prefix, traj_xyz): #(n conf, n_atoms*3) OR (n conf, n_atoms, 3)
        fname = f'{prefix}_{self.model_name}{self.n_latents:04d}.dcd'
        
        if traj_xyz.shape[-1] != 3:
            traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
        
        with md.formats.DCDTrajectoryFile(fname, 'w') as f:
            f.write(traj_xyz*10) #*10 because mdtraj loads data in nm but writes it as angstrom (charmdcd standard is angstrom)
    
    def test_decoded_modeled_trajs(self, rng_seed, num_components, dcd_prefixes:[str]=['TEST','DECODED','GMM'], write=True):
        decoded, latents = self.operate(rng_seed)
        MM, samples = self.gaussian_mm_fit(rng_seed, num_components)
        mm_decoded = self.decode(rng_seed, samples)
        data_to_write = [self.test_data, decoded, mm_decoded]
        assert len(dcd_prefixes) == 3
        if write:
            for i in range(3):
                self.write_traj(dcd_prefixes[i], data_to_write[i])
        return data_to_write

    def plot_rmsd(self, rng_seed, num_components, wrt_ind=0, plot=None):
        test, decoded, modeled = self.test_decoded_modeled_trajs(rng_seed, num_components, write=False)
        
        def atom_rmsd(a, b): # for arrays of (n_conf, n_atom*3)
            mn = a.shape[-1]//3
            x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
            return jnp.sqrt(jnp.mean((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))

        test_rmsd = jax.vmap(atom_rmsd, in_axes=(0, None))(test, test[wrt_ind])
        decoded_rmsd = jax.vmap(atom_rmsd, in_axes=(0, None))(decoded, decoded[wrt_ind])
        modeled_rmsd = jax.vmap(atom_rmsd, in_axes=(0, None))(modeled, modeled[wrt_ind])

        if plot == 'Frame-Series':
            plt.clf()
            plt.title(f"RMSD w.r.t. frame {wrt_ind}")
            plt.xlabel('Frame')
            plt.ylabel('RMSD (nm)')
            for data in test_rmsd, decoded_rmsd, modeled_rmsd:
                _ = plt.plot(jnp.arange(data.shape[0]), data)
            plt.legend(('Test', 'Decoded', 'Modeled'))
            plt.show()

        elif plot == 'Hist':
            plt.clf()
            plt.title(f"RMSD w.r.t. frame {wrt_ind}")
            plt.xlabel('RMSD (nm)')
            plt.ylabel('Count')
            for data in test_rmsd, decoded_rmsd, modeled_rmsd:
                _ = plt.hist(data, bins=50, histtype='step')
            plt.legend(('Test', 'Decoded', 'Modeled'))
            plt.show()

        return test_rmsd, decoded_rmsd, modeled_rmsd

    def cluster_test(self, num_clusters=0, min_clusters=2, max_clusters=40, plot=None):
        """
        If num_clusters is zero, plot the Silhouette score against a range of clusters.
        If num_clusters is integer, just cluster with that many
        """

        if num_clusters == 0:
            sil_scores = []
            clusters = []
            for i in range(min_clusters, max_clusters+1):
                clusters.append(i)
                kmeans = KMeans(n_clusters=i).fit_predict(self.test_data)
                sil_scores.append(silhouette_score(self.test_data, kmeans))
            
            sil_scores = np.array(sil_scores)
            clusters = np.array(clusters)

            if plot is True:
                plt.clf()
                _ = plt.plot(clusters, sil_scores)
                plt.xlabel('Num Clusters')
                plt.ylabel('Silhouette Score')
                plt.show()
            
            return clusters, sil_scores
        
        elif type(num_clusters) == int and num_clusters > 0:
            kmeans = KMeans(n_clusters=num_clusters).fit_predict(self.test_data)

            if plot is True:
                plt.clf()
                _ = plt.scatter(np.arange(self.test_data.shape[0]), kmeans)
                plt.xlabel('Frame')
                plt.ylabel('Cluster ID')
                plt.yticks(np.arange(num_clusters))
                plt.show()
            return kmeans

    def plot_latents_clustered_grid(self, rng_seed, num_clusters, figsize=(15, 10)):
        test_labels = self.cluster_test(num_clusters=num_clusters)
        decoded, latents = self.operate(rng_seed)
        fig, axs = plt.subplots(self.n_latents, self.n_latents, figsize=figsize, sharex='col')
        for i in range(self.n_latents):
            for j in range(self.n_latents):
                for k in range(num_clusters):
                    data = latents[np.where(test_labels == k)]
                    if i > j:
                        axs[i,j].scatter(data[:,j], data[:,i], alpha=0.35)
                    elif i == j:
                        axs[i,j].hist(data[:, i], bins=25, histtype='step')


