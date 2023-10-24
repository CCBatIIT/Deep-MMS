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

*** 10/03 Meeting update ***
  Exploring ChimeraX
    
    Step-0 .dcd file seems to be in nanometers rather than angstrom. Check the decimals conversion before writing it to .dcd again.
    Step-1 explore the ChimeraX program. ### use it only as a visualizer! Try to finish the perturbation and algorithms before putting it in!
    Step-2 find a way to compare original vs perturbed .dcd. Analyze your insights afterthat, move on to other latents.

*** 10/10 Meeting update *** 
  Implementing GitHub workflows
  
    Step-0 Update the github to match the main branch. (solve the commits behind problem)
    Step-1 Urvi and Timo. We need to manage this visualization branch. Let's get an appointment.
    Step-2 Urvi raise an issue on github and communicate with Joseph. 

*** 10/17 Meeting update ***
  Perturbing the latents with new modelL
    
    Step-0 Continue working on GitHub branch issue
    Step-1 Perturb the latents using modelL.
    Step-2 visualize the difference after perturbation.
    Step-3 Write a latents by latents histogram-scatterplot analysis
    
*** 10/23 Meeting update *** 
    
  

=======

### Modules
NN_models - a python module containing various flax.linen.nn modules.  Used to construct AutoEncoder and Decoder pairs for experiment./
training_functions - jax vmap and jit functions which are applied during training./
AutoEncoder_Experiment - A wrapping class that actually executes the training and assembly.

