#USAGE python _05_...py json_file

#How do the best models compare to PCA in terms of clustering?
import os, sys, jax, glob, datetime
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pyscripts.heavy_atom_rmsd import *

from _02_write_viollin_data import HeavyAtom_Analyzer

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
print('start')

#Obtain the test set date from the VAE Model
json_fn = sys.argv[1]
HA = HeavyAtom_Analyzer(json_fn=json_fn)

model_name = HA.model_name
work_dir = os.path.join(HA.data_dir, 'perturbation')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir, exist_ok=True)

log_fn = os.path.join(work_dir, 'perturbation_log.txt')
logfile = open(log_fn, 'w')
logfile.write(f"{datetime.now()} - Begin {model_name}\n")

#obtain trajectories of test data, reconstructed test data, and along a latent axis
logfile.write(f"{datetime.now()} - Obtain Test Data\n")
test_frames = HA.test_data
decoded, latents = nn_operate(HA, test_frames)
latent_means, latent_stds = jnp.mean(latents, axis=0), jnp.std(latents, axis=0)

logfile.write(f"{datetime.now()} - Obtain Perturbation Data\n")
n_perturbation = 2001
perturb_space = jnp.linspace(latent_means - 5*latent_stds, latent_means+5*latent_stds, n_perturbation, axis=0)

#iterate over the linspaces, keeping the other coords as mean
rng_seed = 893467
for i in range(perturb_space.shape[1]):
    key = jax.random.PRNGKey(rng_seed)
    main_key, dropout_key = jax.random.split(key, num=2)
    
    this_perturbation = perturb_space
    for j in range(perturb_space.shape[1]):
        if j != i:
            this_perturbation = this_perturbation.at[:, j].set(jnp.repeat(latent_means[j], n_perturbation))
    logfile.write(f"\t {datetime.now()} - Latent {i=}\n")
    
    decoded_perturbed = HA.state.apply_fn({'params': HA.state.params},
                                          this_perturbation, main_key, train=False,
                                          rngs={'dropout': dropout_key}, method=HA.model.decode)
    HA.write_traj(None, decoded_perturbed, fname=os.path.join(work_dir, f"{model_name}_pLatent{i:04d}.dcd"))
    logfile.write(f"\t {datetime.now()} - Wrote data for Latent {i=}\n")
    rng_seed += j

HA.write_traj(None, test_frames, fname=os.path.join(work_dir, f"{model_name}_test_data.dcd"))
HA.write_traj(None, decoded, fname=os.path.join(work_dir, f"{model_name}_test_recon.dcd"))
logfile.close()



