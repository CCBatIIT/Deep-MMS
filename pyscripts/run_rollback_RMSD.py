from rollback import *
import sys, jax, json, time
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

json_fn = sys.argv[1]
with open(json_fn, 'r') as g:
        json_params = json.load(g)

#Init Experiment
experiment = AutoEncoder_Experiment(json_fn)

#Train RMSD into oblivion
experiment.MAIN_train_rmsd_only()