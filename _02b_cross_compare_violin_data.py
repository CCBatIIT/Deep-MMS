#USAGE python -02b_...py json1 json2
import jax.numpy as jnp
import numpy as np
import glob, os, jax, sys
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
import mdtraj as md

from _02_write_viollin_data import HeavyAtom_Analyzer, atom_rmsd
from pyscripts.plotting_utils import *

def operate(heavy_atom_analyzer, data, rng_seed):
    key = jax.random.PRNGKey(rng_seed)
    main_key, dropout_key = jax.random.split(key, num=2)
    arg_dict = {'params': heavy_atom_analyzer.state.params}
    if heavy_atom_analyzer.is_batchnorm:
        arg_dict['batch_stats'] = heavy_atom_analyzer.state.batch_stats
    decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn(arg_dict, data, main_key, train=False, rngs={'dropout': dropout_key})
    return decoded, latent_means, latent_vars

def cross_compare_models(json1, json2, rng_seed=69420, align_first=False):
    """
    Operate on and calculate reconstructive error for test data of the other model
    returns 
        
        arr1 = error vals associated with model 1 reconstructing the test data of model 2
        arr2 = error vals associated with model 2 reconstructing the test data of model 1

    if align_first:
        align data such that the each model is operating on data aligned to its first training frame
    """
    haas = [HeavyAtom_Analyzer(j_fn) for j_fn in [json1, json2]]
    assert haas[0].n_latents == haas[-1].n_latents
    if align_first:
        #align model 2 data to model 1
        traj1 = md.Trajectory(xyz=haas[0].test_data.reshape(haas[0].test_data.shape[0], -1, 3),
                              topology=md.load(haas[0].json_params['fname_topology'].replace('prmtop', 'pdb')).atom_slice(md.load(haas[0].json_params['fname_topology'].replace('prmtop', 'pdb')).topology.select('not element H')).topology) 
        traj2 = md.Trajectory(xyz=haas[1].test_data.reshape(haas[1].test_data.shape[0], -1, 3),
                              topology=md.load(haas[1].json_params['fname_topology'].replace('prmtop', 'pdb')).atom_slice(md.load(haas[1].json_params['fname_topology'].replace('prmtop', 'pdb')).topology.select('not element H')).topology) 

        traj1_on2 = traj1.superpose(traj2)
        traj2_on1 = traj2.superpose(traj1)
        del traj1, traj2
        
        data1_on2 = jnp.array(traj1_on2.xyz.reshape(traj1_on2.xyz.shape[0], -1))
        data2_on1 = jnp.array(traj2_on1.xyz.reshape(traj2_on1.xyz.shape[0], -1))
        
        decoded12, _, _ = operate(haas[0], data2_on1, rng_seed=rng_seed)
        decoded21, _, _ = operate(haas[1], data1_on2, rng_seed=rng_seed)
    else:
        decoded12, _, _ = operate(haas[0], haas[1].test_data, rng_seed=rng_seed)
        decoded21, _, _ = operate(haas[1], haas[0].test_data, rng_seed=rng_seed)
    
    err12 = atom_rmsd(haas[1].test_data, decoded12)
    err21 = atom_rmsd(haas[0].test_data, decoded21)

    return err12, err21

#'/ocean/projects/cis250004p/josephdb/Deep-MMS/difference/'
def compare_and_save(json1, json2, save_dir, align_first=False):
    model_names = [elem.split('/')[-2] for elem in [json1, json2]]
    latents = [int(elem.split('_')[-2]) for elem in [json1, json2]]
    er1, er2 = cross_compare_models(json1, json2, align_first=align_first)
    np.save(os.path.join(save_dir, f'{model_names[0]}_{latents[0]:04d}_reconstructing_{model_names[1]}.npy'), er1)
    np.save(os.path.join(save_dir, f'{model_names[1]}_{latents[1]:04d}_reconstructing_{model_names[0]}.npy'), er2)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 3:
        align = bool(sys.argv[3])
    else:
        align = False
    compare_and_save(sys.argv[1], sys.argv[2], '/ocean/projects/cis250004p/josephdb/Deep-MMS/difference/', align_first=align)

