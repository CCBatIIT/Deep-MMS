from AutoEncoder_Experiment import *
import sys, jax, json
import jax.numpy as jnp
from datetime import datetime

import jax_amber3 as ja
import training_functions as tf

jax.config.update("jax_enable_x64", True)

json_fn = sys.argv[1]
with open(json_fn, 'r') as g:
        json_params = json.load(g)

#Init Experiment
experiment = AutoEncoder_Experiment(json_fn)

#Try training n_epochs on the structural loss function
experiment.train_nepochs(tf.structural_rng_step, 100, coefficients=[1,0])
experiment.train_scaling_coef(tf.summation_rng_step, 5000, 1, coefficients=[1,0])
experiment.write_decoded_traj()
