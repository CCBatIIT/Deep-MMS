#USAGE: python _03b_...py json_fn
import inspect, jax
import jax.numpy as jnp
import os, sys, jax, glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from _02_write_viollin_data import HeavyAtom_Analyzer

def dir_tree(base_dir, heavy_atom_analyzer):
    return os.path.join(base_dir,
                        f'{heavy_atom_analyzer.model_name}/',
                        f'{heavy_atom_analyzer.n_latents:04d}_latents/',
                        f'rpt_{heavy_atom_analyzer.json_params["test_slice"]}/')

def nn_operate(heavy_atom_analyzer, data, rng_seed=69420):
    key = jax.random.PRNGKey(rng_seed)
    main_key, dropout_key = jax.random.split(key, num=2)
    if heavy_atom_analyzer.is_batchnorm:
        decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn({'params': heavy_atom_analyzer.state.params, 'batch_stats': heavy_atom_analyzer.state.batch_stats},
                                                                                data, main_key, train=False,
                                                                                rngs={'dropout': dropout_key})
    else:
        decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn({'params': heavy_atom_analyzer.state.params},
                                                                                data, main_key, train=False,
                                                                                rngs={'dropout': dropout_key})
    return decoded, latent_means

euclidean_vmapped = jax.vmap(lambda a, b: jnp.sqrt(jnp.sum((b-a)**2)), in_axes=(None, 0))

def euclidean_distance_matrix(vectors, func=euclidean_vmapped):
    """
    Calculate the euclidean distance matrix between the array of vectors, give as (n_samples, n_features) returns
    (n_samples, n_samples) distance matrix where the ith row is all particles compared to sample i
        func - optionally provide a function to use for the distance between two vectors - otherwise the Euclidean (L2) distance will be used.
    
    """
    distance = jnp.empty((vectors.shape[0], vectors.shape[0]))
    for i in range(vectors.shape[0]): #Iterate over rows, distribute pair op across GPU for each row
        distance = distance.at[i, :].set(func(vectors[i], vectors)) #See inputs here, compare to in_axes of vmap, note actual functional form
    return distance

def atom_rmsd(a, b):
    """
    Atom RMSD of vectorized frames a and b
    Due to vmapping does not work on individual frames, but only collections of frames
    """
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    return jnp.sqrt(jnp.mean(jnp.sum((b - a)**2, axis=1)))

def get_latent_pca(json_fn):
    
    #establish the analyzer
    ha = HeavyAtom_Analyzer(json_fn)
    data_dir = dir_tree(data_directory, ha)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    #Get the latent means representation
    decoded, latent = nn_operate(ha, ha.test_data)
    
    #Get the PCA representation (train on train and fit test)
    pca = PCA(n_components=ha.n_latents)
    _ = pca.fit(ha.train_data)
    components = pca.transform(ha.test_data)
    
    #Save these to the data directory
    vae_fn, pca_fn = os.path.join(data_dir, "VAE_latents.npy"), os.path.join(data_dir, "PCA_latents.npy")
    np.save(vae_fn, latent)
    print(vae_fn)
    np.save(pca_fn, components)
    print(pca_fn)
    #Save the L2 distance matrices of these
    np.save(vae_fn.replace('.npy','_L2_distance_matrix.npy'),
            euclidean_distance_matrix(latent))
    print(vae_fn.replace('.npy','_L2_distance_matrix.npy'))
    np.save(pca_fn.replace('.npy','_L2_distance_matrix.npy'),
            euclidean_distance_matrix(components))
    print(pca_fn.replace('.npy','_L2_distance_matrix.npy'))
    #Save the L2 and RMSD distance matrices of the testing data
    np.save(os.path.join(data_dir, 'L2_test_distance_matrix.npy'),
            euclidean_distance_matrix(ha.test_data))
    print(os.path.join(data_dir, 'L2_test_distance_matrix.npy'))
    np.save(os.path.join(data_dir, 'RMSD_test_distance_matrix.npy'),
            euclidean_distance_matrix(ha.test_data, func=jax.vmap(atom_rmsd, in_axes=(None, 0))))
    print(os.path.join(data_dir, 'RMSD_test_distance_matrix.npy'))
######################################################
###  End Function Definitions and begin main block ###
######################################################

if __name__ == '__main__':
    data_directory = '/ocean/projects/cis250004p/josephdb/Deep-MMS/comp_stor/'
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory, exist_ok=True)
    _ = get_latent_pca(sys.argv[1])
    
    