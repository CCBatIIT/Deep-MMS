from AutoEncoder_Experiment import *
import sys, jax
from datetime import datetime
jax.config.update("jax_enable_x64", True)

#JSON is arg
json_fn = 'Sigmoids_Dropout.json'
#To be added to json
num_rmsd_epochs = 250 #Initial Run
nm_cutoff = 0.1 #target rmsd in nm
rmsd_cutoff = 1500 #num epochs to force training to summation
num_move_ave = 50 #number of previous points for moving average calculations
potential_threshold = 1e-3 #Target Potential NSD tolerance
cutoff_epoch = 10000 #maximum number of epochs of training

#Init Experiment
experiment = AutoEncoder_Experiment(json_fn)
