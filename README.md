# Deep-MMS
Repository for construction of JAX/FLAX neural networks which are applied to the mapping of molecular mechanics ensembles.


### Steps to setup conda environment

The conda environment can be set up for linux systems with access to cuda12 by using the include yml file
```
conda env create --file=jetstream2_env.yml
```

It can also be constructed by doing the following:
Add conda-forge to the front of the channels list, then create the new environment
```
conda config --prepend channels conda-forge
conda create -n ENV_NAME
conda activate ENV_NAME
```

Install necessary packages
```
conda install python pip jupyter netCDF4 mdtraj matplotlib openmm scikit-learn
which pip     #Verify that the pip being used is from the environment
pip install -U "jax[cuda13]"
pip install flax
```

In an ipython instance - verify that JAX detects gpu
```
ipython
[1] >>> print(jax.print_environment_info(), jax.default_backend())
#Should provide and nvidia-smi output and say gpu
```
