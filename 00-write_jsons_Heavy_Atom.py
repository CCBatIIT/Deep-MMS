import json, os
import numpy as np
import mdtraj as md

def primes_up_to(limit):
    if limit < 2:
        return np.array([], dtype=int)
    
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i : limit + 1 : i] = False
            
    return np.flatnonzero(is_prime)

def powers_of_two_up_to(limit):
    log_limit = np.log2(limit)
    log_limit = log_limit // 1
    return [int(i) for i in 2**np.arange(log_limit+1, dtype=int)]

def latent_nums(limit):
    primes = primes_up_to(limit)
    twos = powers_of_two_up_to(limit)
    latents = []
    for i in np.arange(limit+1):
        if i in primes or i in twos or i == limit:
            latents.append(int(i))
    return latents


#Don't forget the NAME!
model_base = 'X008-2'

#UnComment to test on crambin
# dcd_fns = ["/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/1crn_split2.dcd"]

# top_fns = ["/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/1crn_H.prmtop"]

# model_names = [f'CR_{model_base}']
#END UnComment Section
#UnComment below to make all jsons
dcd_fns = ["/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/oxycodone.dcd",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/decaalanine_1us_split3.dcd",
           "/media/volume/Josephs-Volume/SYW_da_stretch/DA_stretch_super.dcd",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/1crn_split2.dcd",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/3mxf_split2.dcd",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/KOR_protein_ligand.dcd"]

top_fns = ["/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/oxycodone.prmtop",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/ala_deca_peptide.prmtop",
           "/media/volume/Josephs-Volume/SYW_da_stretch/ala_deca_peptide.prmtop",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/1crn_H.prmtop",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/3mxf_protein_ligand.pdb",
           "/media/volume/Josephs-Volume/githubs/Deep-MMS/Simulation/KOR_protein_ligand.pdb"]

model_names = [f'OX_{model_base}', f'DA_{model_base}',
               f'DA_stretch_{model_base}', f'CR_{model_base}',
               f'BR_{model_base}', f"KOR_{model_base}"]
#END UnComment Section



json_dir = f'/media/volume/Josephs-Volume/githubs/Deep-MMS/json_inputs/{model_base}'
if not os.path.isdir(json_dir):
    os.makedirs(json_dir, exist_ok=True)

for dcd_fn, top_fn, model_name in zip(dcd_fns, top_fns, model_names):
    c = md.load(dcd_fn, top=top_fn)
    c = c.atom_slice(c.topology.select('not element H'))
    latent_dims = powers_of_two_up_to(c.n_atoms) + [c.n_atoms] 

    lrs = [10**-3 for lr in latent_dims]

    assert False not in [os.path.isfile(fn) for fn in [dcd_fn, top_fn]]

    if not os.path.isdir(os.path.join(json_dir, model_name)):
        os.mkdir(os.path.join(json_dir, model_name))
    
    for latent_dim, lr in zip(latent_dims, lrs):
        for test_slice in [1, 2, 3, 4, 5]:
            json_fn = os.path.join(json_dir, model_name, f"{model_name}_{latent_dim:04d}_{test_slice:02d}.json")
            #Dropout rates for the hideen layers - also determines the quantity of layers
            dropout_rates = [0.5, 0.4, 0.3, 0.2, 0.1, 0.1] #ORIGINAL
            
            #Directory to build outputs
            save_dir = "/media/volume/Josephs-Volume/githubs/Deep-MMS/"
            #initial and final index of data from which to derive test and train sets
            data_slice_start, data_slice_end = 0, "None"
            #size of batches (must be an even divisor of total test and total train frames)
            batch_size = 1000
            #Learning rate for the adam optimizer
            learning_rate = lr
            #Whether to resume a previous training or not
            resume_latest = False
            #Interval of epochs to checkpoint the Neural Network
            checkpoint_interval = 200
            #Cutoff epoch
            max_epoch = 15001
            #Batchnorm?
            is_batchnorm = True
            
            json_params = dict(fname_dcd=dcd_fn, fname_topology=top_fn,
                               save_dir=save_dir,
                               max_epoch=max_epoch, latent_dim=latent_dim, test_slice=test_slice,
                               data_slice_start=data_slice_start, data_slice_end=data_slice_end,
                               model_name=model_name, batch_size=batch_size, learning_rate=learning_rate,
                               dropout_rates=dropout_rates, 
                               resume_latest=resume_latest, 
                               checkpoint_interval=checkpoint_interval,
                               is_batchnorm=is_batchnorm)

            with open(json_fn, 'w') as f:
                json.dump(json_params, f, indent=4)
