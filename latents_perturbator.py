import numpy as np
import matplotlib.pyplot as plt
import jax

#########################################
#
#   Written by Timo Srinarmwong. github.com/tsrinarmwong Created 01/31/2024       
#
#       This is a methods file for latents perturbation used in VAE research
#         project with Joseph DePaolo-Boisvert, Dr.David Minh
#      
#    Functions list:
#        perturb_single_latent(orig_latents, perturb_value=-0.5, latent_target=0, perturb_method=1):
#        perturb_all_latents(orig_latents, perturb_values)
#        calculate_displacement_vectors(orig_decoded, pert_decoded)
#        find_frame_with_max_displacement(displacement_vectors)
#        find_frame_with_min_displacement(displacement_vectors)
#        plot_mean_displacement(displacement_vectors)
#
#        Example usage at the end of file
# 
########################################

@jax.jit
def perturb_single_latent(orig_latents, perturb_value, latent_target, perturb_method):
    """
    Perturb the specified latent column in the input latents.

    Parameters:
    - orig_latents: Original latent array.
    - perturb_value: Value to perturb the latent column by.
    - latent_target: Index of the latent column to perturb.
    - perturb_method: Perturbation method ('0 = add', '1 = subtract', '2 = multiply', '3 = divide').

    Returns:
    - pert_latents: Perturbed latent array.
    """
    
    if perturb_method == 0:
        pert_latents = orig_latents.at[:, latent_target].add(perturb_value)
    elif perturb_method == 1:
        pert_latents = orig_latents.at[:, latent_target].subtract(perturb_value)
    elif perturb_method == 2:
        pert_latents = orig_latents.at[:, latent_target].multiply(perturb_value)
    elif perturb_method == 3:
        pert_latents = orig_latents.at[:, latent_target].divide(perturb_value)
    else:
        raise ValueError("Invalid perturbation method. Choose from 'add', 'subtract', 'multiply', 'divide'.")
    return pert_latents

@jax.jit
def perturb_all_latents(orig_latents, perturb_values):
    """
    Perturb all latent columns in the input latents. Distinct value can be chosen for each latents

    Parameters:
    - orig_latents: Original latent array. (recon[1])
    - perturb_values: List of values to perturb all latent columns.

    Returns:
    - pert_latents: Perturbed latent array.
    """
    pert_latents = jnp.array(orig_latents)
    
    for value, latent_index in zip(perturb_values, range(orig_latents.shape[1])):
        pert_latents[:, latent_index].add(value)
    
    return pert_latents

import jax.numpy as jnp
from jax import jit

@jax.jit
def perturb_all_latents_prot(orig_latents, perturb_values, perturb_method='add'):
    """
    Perturb all latent columns in the input latents. Distinct value can be chosen for each latents

    Parameters:
    - orig_latents: Original latent array. (recon[1])
    - perturb_values: List of values to perturb all latent columns.
    - perturb_method: Perturbation method ('add', 'subtract', 'multiply', 'divide').

    Returns:
    - pert_latents: Perturbed latent array.
    """
    
    for value, latent_index in zip(perturb_values, range(orig_latents.shape[1])):
        if perturb_method == 'add':
            pert_latents = pert_latents.at[:, latent_index].add(value)
        elif perturb_method == 'subtract':
            pert_latents = pert_latents.at[:, latent_index].subtract(value)
        elif perturb_method == 'multiply':
            pert_latents = pert_latents.at[:, latent_index].multiply(value)
        elif perturb_method == 'divide':
            pert_latents = pert_latents.at[:, latent_index].divide(value)
        else:
            raise ValueError("Invalid perturbation method. Choose from 'add', 'subtract', 'multiply', 'divide'.")
    
    return pert_latents


def calculate_displacement_vectors(orig_decoded, pert_decoded):
    """
    Calculate displacement vectors between original and perturbed decoded arrays.

    Parameters:
    - orig_decoded: Original decoded array.
    - pert_decoded: Perturbed decoded array.

    Returns:
    - displacement_vectors: Displacement vectors array.
    """
    motion = orig_decoded - pert_decoded
    displacement_vectors = motion.reshape(motion.shape[0], -1, 3)
    return displacement_vectors

def find_frame_with_max_displacement(displacement_vectors):
    """
    Find the frame index with the maximum displacement.

    Parameters:
    - displacement_vectors: Displacement vectors array.

    Returns:
    - frame_index: Index of the frame with the maximum displacement.
    """
    norms = np.linalg.norm(displacement_vectors, axis=2) #Calculate displacement for each atom in each frame
    total_displacement_per_frame = norms.sum(axis=1) #Calculate total displacement for each frame
    frame_with_max_displacement = np.argmax(total_displacement_per_frame) #Find Max index
    return frame_with_max_displacement + 1 #Convert to frame index

def find_frame_with_min_displacement(displacement_vectors):
    """
    Find the frame index with the minimum displacement.

    Parameters:
    - displacement_vectors: Displacement vectors array.

    Returns:
    - frame_index: Index of the frame with the minimum displacement.
    """
    norms = np.linalg.norm(displacement_vectors, axis=2) #Calculate displacement for each atom in each frame
    total_displacement_per_frame = norms.sum(axis=1) #Calculate total displacement for each frame
    frame_with_min_displacement = np.argmin(total_displacement_per_frame) #Find Min index
    return frame_with_min_displacement + 1 #Convert to frame index

def plot_mean_displacement(displacement_vectors):
    """
    Plot the mean displacement along X, Y, and Z axes.

    Parameters:
    - displacement_vectors: Displacement vectors array.
    """
    means = np.mean(displacement_vectors, axis=0)
    titles = ['Average Displacement X', 'Average Displacement Y', 'Average Displacement Z']
    
    for i in range(means.shape[-1]):
        plt.clf()
        plt.bar(np.arange(means.shape[0]), means[:, i])
        plt.title(titles.pop(0))
        plt.ylabel('Mean displacement (nm)')
        plt.xlabel('Atom Index')
        plt.show()

# Example usage:
#   orig_latents = jnp.array(recon[1])
#   pert_latents = perturb_single_latent(orig_latents, perturb_value=-0.5, latent_target=0)

#   perturb_values = [-0.5, 0.2, 0.1]
#   pert_latents = perturb_all_latents(orig_latents, perturb_values)

#   orig_decoded = ...
#   pert_decoded = ...

#   displacement_vectors = calculate_displacement_vectors(orig_decoded, pert_decoded)
#   frame_max = find_frame_with_max_displacement(displacement_vectors)
#   frame_min = find_frame_with_min_displacement(displacement_vectors)
#   plot_mean_displacement(displacement_vectors)
