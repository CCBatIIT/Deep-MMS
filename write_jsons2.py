import json, sys, os
import numpy as np

# USAGE python write_jsons2.py folder_fn highest_latent_n

assert len(sys.argv) == 3

#Trajectory file from which to obtain train and test sets
fname_dcd = "/ocean/projects/cis250004p/josephdb/Deep-MMS/Simulation/decaalanine_1us_split3.dcd"
assert os.path.isfile(fname_dcd)

#Parameter file for energy functions
fname_prmtop = "/ocean/projects/cis250004p/josephdb/Deep-MMS/Simulation/ala_deca_peptide.prmtop"

#PDB file for topology, only if fname_prmtop is an xml file not a prmtop file
fname_pdb = "None"

#name for file-keeping purposes
model_name = "DA_VAE_Dropout_RMSD_wKL"

#Directory to build outputs
save_dir = f"/ocean/projects/cis250004p/josephdb/Deep-MMS_runs/"

#Override the automatic saving with this directory
data_dir = "None"

#Cutoff epoch
max_epoch = 20000

#Number of latent dimensions
latent_dim = None

#index of the data from which to derive the 20/80 test set split
test_slice = 3

#initial index of data from which to derive test and train sets
data_slice_start = 0

#final index of data from which to derive test and train sets
data_slice_end = "None"

#name for file-keeping purposes
model_name = "DA_AE_Dropout_RMSD_wKL"

#size of batches (must be an even divisor of total test and total train frames)
batch_size = 1000

#Learning rate for the adam optimizer
learning_rate = 1e-4

#Dropout rates for the hideen layers - also determines the quantity of layers
dropout_rates = [0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]

#IDK
potential_threshold = 0.05

#Whether to resume a previous training or not
resume_latest = False

#whether to calculate the potential energy or not
report_potential = False

#Interval of epochs to checkpoint the Neural Network
checkpoint_interval = 200

#I forgor
scale_factor = 1.0

#Whether to use all atom or heavy atom coordinates ('!H' or 'all')
atom_select = 'not element H'


json_params = dict(fname_dcd=fname_dcd, fname_prmtop=fname_prmtop, fname_pdb=fname_pdb,
                   save_dir=save_dir, data_dir=data_dir,
                   max_epoch=max_epoch, latent_dim=latent_dim, test_slice=test_slice,
                   data_slice_start=data_slice_start, data_slice_end=data_slice_end,
                   model_name=model_name, batch_size=batch_size, learning_rate=learning_rate,
                   dropout_rates=dropout_rates, potential_threshold=potential_threshold,
                   resume_latest=resume_latest, report_potential=report_potential,
                   checkpoint_interval=checkpoint_interval, scale_factor=scale_factor,
                   atom_select=atom_select)


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
    return 2**np.arange(log_limit+1, dtype=int)

def latent_nums(limit):
    primes = primes_up_to(limit)
    twos = powers_of_two_up_to(limit)
    latents = []
    for i in np.arange(limit+1):
        if i in primes or i in twos or i == limit:
            latents.append(i)
    return np.array(latents)


#Determine Latents Quantities
latents = latent_nums(int(sys.argv[2]))
#Determine directory to put jsons in
json_stor_dir = sys.argv[1]
if not os.path.isdir(json_stor_dir):
    os.mkdir(json_stor_dir)

json_stor_dir = os.path.join(json_stor_dir, model_name)
if not os.path.isdir(json_stor_dir):
    os.mkdir(json_stor_dir)

for lat in latents:
    for tes in [0,1,2,3,4]:
        json_params["latent_dim"] = int(lat)
        json_params["test_slice"] = tes
        with open(os.path.join(json_stor_dir, f'{model_name}_{lat:04d}_{tes}.json'), 'w') as g:
            json.dump(json_params, g)

