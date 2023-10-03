# Deep-MMS
Repository for construction of JAX/FLAX neural networks which are applied to the mapping of molecular mechanics ensembles.

The branch focuses on Visualization aspect.
The main work is done in `perturbation_model` notebooks. 
  The notebooks uses custom json file to specify the data and load a model from checkpoint,
  It auto encodes and decodes the data. The notebook perturbate the latent space then decoded it again.
  Producing data in # 5 sets of data
      **input data = experiment.test_data**
      **original latents = recon[1]**
      **original decoded = recon[0]**
      **perturb latents = perturbed_latents**
      **perturb decoded = perturbed_decoded**

  It write the data into .dcd file to furthur visualize
