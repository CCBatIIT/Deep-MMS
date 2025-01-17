# Deep-MMS
Repository for construction of JAX/FLAX neural networks which are applied to the mapping of molecular mechanics ensembles.


### Steps to setup conda environment
Add conda-forge to the front of the channels list, then create the new environment
```
conda config --prepend channels conda-forge
conda create -n ENV_NAME
```

Install necessary packages
```
conda install python pip jupyter
which pip #Verify that the pip being used is from the environment
pip install -U "jax[cuda12]"
pip install flax
conda install netCDF4 mdtraj matplotlib
```

In an ipython instance - verify that JAX detects gpu
```
ipython
[1] >>> print(jax.print_environment_info(), jax.default_backend())
#Should provide and nvidia-smi output and say gpu
```




### pyscripts Modules
  - jax_amber3 - Construction of Molecular Mechanics functions in jax/flax
  - NN_models - Construction of Encoding-decoding pairs
  - Maths - Mathematical functions for evaluating metrics such as 
    - Evaluation of Root Mean Square Distances with euclidean and torsional distances available
    - Comparisons of potential energy (functions constructed in jax_amber3)
    - Training "Step" functions - invoke these metrics as loss values and apply the gradients to the AutoEncoder
  - AutoEncoder_Experiment - Main class
    - Parsing of json input file
    - Construction of NN
    - Batching of Data
    - Training by (on any distance function from Maths):
       - Number of epochs
       - Reaching a Threshold
       - Scaling in a metric (potential only at the moment)
