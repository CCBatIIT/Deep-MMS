from DEV_main import NN_Experiment
import sys, jax, json, time
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

json_fn = sys.argv[1]

#Init Experiment
experiment = NN_Experiment(json_fn)

#Train 
experiment.MAIN_scale_and_train_potential(n_rmsd=1000)
