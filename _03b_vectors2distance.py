#Usage python _03b_...py numpy_file
#Where numpy file is of shape (n_samples, n_features) - writes Euclidean (or custom if provided) distance matrix of shape (n_samples, n_samples)
import inspect, jax
import jax.numpy as jnp
#Distance function - Euclidean
#written expecting input as pair (a and b have shape (n_features))
euclidean_vmapped = jax.vmap(lambda a, b: jnp.sqrt(jnp.sum((b-a)**2)), in_axes=(None, 0))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:^^^^^^^^^^^^^^^^^^^^^^^^^^^^XXXXXXXXXXXXXXXXXXXX
# use your own function by editing lambda function here (^) not here (X)
jax.print_environment_info()

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

if __name__ == '__main__':
    import sys
    import numpy as np
    np_fn = sys.argv[1]
    np.save(np_fn.replace('.npy','_distance_matrix.npy'), euclidean_distance_matrix(np.load(np_fn)))