from pyscripts.heavy_atom_rmsd import *

def save_dcd(fname, traj_xyz):
    if traj_xyz.shape[-1] != 3:
        traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
    
    with md.formats.DCDTrajectoryFile(fname, 'w') as f:
        f.write(traj_xyz*10) #*10 because mdtraj loads data in nm but writes it as angstrom (charmdcd standard is angstrom)    

class HeavyAtom_Analyzer(HeavyAtom_NN_Experiment):
    def __init__(self, json_fn, from_json_params=False, checkpoint_recency=-1):
        """
        Provide the json file that ran the experiment
        To use json parameters that are already in memeory, not just the file name,
        make from_json_params True and pass the dictionary instead of the file name

        Checkpoint_recency, an index of sorted checkpoint dirs -1 = most well trained.
        """
        #Unpack json
        if not from_json_params:
            with open(json_fn, 'r') as g:
                self.json_params = json.load(g)
        else:
            self.json_params = json_fn
        
        #Determine the location of the save_dir from the json file
        self.model_name = self.json_params["model_name"]
        self.n_latents = self.json_params["latent_dim"]
        test_slice = self.json_params["test_slice"]
        self.data_dir = os.path.join(self.json_params["save_dir"], f'{self.model_name}/', f'{self.n_latents:04d}_latents/', f'rpt_{test_slice}/')
        self.is_batchnorm = self.json_params["is_batchnorm"]
        if "data_dir" in self.json_params.keys():
            if self.json_params["data_dir"] is not None:
                self.data_dir = self.json_params["data_dir"]
                
        #Find the netcdf and checkpoint
        nc_data_file = os.path.join(self.data_dir, f'model_{self.model_name}_{self.n_latents:04d}.nc')
        self.rootgrp = self.establish_netcdf(nc_data_file, open_mode='r')

        checkpoint_dir_wc = os.path.join(self.data_dir, 'checkpoint_managed', '*/')
        checkpoint_dir = sorted(glob.glob(checkpoint_dir_wc))[checkpoint_recency]

        assert False not in [os.path.exists(direc) for direc in [self.data_dir, nc_data_file, checkpoint_dir]]

        #Load and Align
        c = md.load(self.json_params['fname_dcd'], top=self.json_params['fname_topology'])
        c = c.atom_slice(c.topology.select(self.json_params["atom_selection"]))
        c = c.superpose(c) # FEED IN ALIGNED DATA
        mass_sets = mass_weights(c)
        data_start, data_end = self.json_params["data_slice_start"], self.json_params["data_slice_end"] #Slice of data
        if data_end == 'None':
            data_end = None
        
        coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))[data_start:data_end]
        num_samples, input_size = coord_set.shape

        #Make Test and Train Sets
        test_indices = np.array(range(test_slice, num_samples, 5)) #every fifth frame
        train_indices = np.array([element for element in range(num_samples) if element not in test_indices])
        self.test_data = coord_set[test_indices]
        self.train_data = coord_set[train_indices]
        #printf((self.train_data.shape, self.test_data.shape))
        self.batch_size = self.json_params["batch_size"]
        
        #Load the model and state
        dropout_rates = self.json_params["dropout_rates"]
        learning_rate = self.json_params["learning_rate"]
        from pyscripts.NN_constructor import make_model_and_state
        
        if 'weight_model' in self.json_params.keys():
            weight_model = self.json_params['weight_model']
            assert weight_model in mass_sets.keys()
        else:
            weight_model = 'Uniform_Heavy'
        printf(f'\t Using {weight_model=}')
        weights = jnp.array(mass_sets[weight_model])
        self.atom_rmsd_loss = give_weighted_rmsd_func(weights)
        global step, evaluate
        self.model, self.state, step, evaluate = make_model_and_state(self, dropout_rates, coord_set, learning_rate, self.atom_rmsd_loss)
        step, evaluate = jax.jit(step), jax.jit(evaluate)
        
        
        self.state = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_dir+'/default/', item=self.state)

        printf(f"Done restoring from {json_fn}")


def violin_data(heavy_atom_analyzer, save_npy=True):
    from sklearn.decomposition import PCA
    key = jax.random.PRNGKey(6969)
    main_key, dropout_key = jax.random.split(key, num=2)


    n_latents = heavy_atom_analyzer.n_latents
    decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn({'params': heavy_atom_analyzer.state.params, 'batch_stats': heavy_atom_analyzer.state.batch_stats},
                                                                            heavy_atom_analyzer.test_data, main_key, train=False,
                                                                            rngs={'dropout': dropout_key})
    vae_rmsd = atom_rmsd(heavy_atom_analyzer.test_data, decoded)
    vae_loss_rmsd = heavy_atom_analyzer.atom_rmsd_loss(heavy_atom_analyzer.test_data, decoded)
    pca = PCA(n_components=n_latents)
    _ = pca.fit(heavy_atom_analyzer.train_data)
    comps = pca.transform(heavy_atom_analyzer.test_data)
    pca_test = pca.inverse_transform(pca.transform(heavy_atom_analyzer.test_data))
    pca_rmsd = atom_rmsd(heavy_atom_analyzer.test_data, pca_test)
    pca_loss_rmsd = heavy_atom_analyzer.atom_rmsd_loss(heavy_atom_analyzer.test_data, pca_test)

    if save_npy:
        for data, fn in zip([vae_rmsd, vae_loss_rmsd, pca_rmsd, pca_loss_rmsd],['VAE_RMSD.npy', 'VAE_LOSS_RMSD.npy', 'PCA_RMSD.npy', 'PCA_LOSS_RMSD.npy']):
            np.save(os.path.join(heavy_atom_analyzer.data_dir, fn), data)
            printf(f"    Wrote {os.path.join(heavy_atom_analyzer.data_dir, fn)}")
    
    return vae_rmsd, vae_loss_rmsd, pca_rmsd, pca_loss_rmsd

if __name__ == '__main__':
    import sys

    json_fn = sys.argv[1]
    HA_analyzer = HeavyAtom_Analyzer(json_fn)
    _ = violin_data(HA_analyzer)