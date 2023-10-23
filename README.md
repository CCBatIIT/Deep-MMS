# ACS_DEPRECATED IS A STALE BRANCH, CONTAINING THE CODE USED TO GENERATE FIGURES AND GMM FOR JOSEPHS ACS SAN FRAN PRESENTATION

# Deep-MMS
Repository for construction of JAX/FLAX neural networks which are applied to the mapping of molecular mechanics ensembles.

### Modules
NN_models - a python module containing various flax.linen.nn modules.  Used to construct AutoEncoder and Decoder pairs for experiment./
training_functions - jax vmap and jit functions which are applied during training./
AutoEncoder_Experiment - A wrapping class that actually executes the training and assembly.
