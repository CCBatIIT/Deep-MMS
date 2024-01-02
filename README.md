# Deep-MMS
Repository for construction of JAX/FLAX neural networks which are applied to the mapping of molecular mechanics ensembles.

The latest version of the visualization after perturbation :
visualize how does the perturbed *Value* affect final decoding. (Red = added 0.5, light red = subtract 0.5)
<img width="864" alt="image" src="https://github.com/CCBatIIT/Deep-MMS/assets/125150205/c797f60c-ec26-4d88-923c-15d21769331d">

The first version of the visualization after perturbation: 
visualize how does the perturbed *Latents* affect final decoding.
<img width="853" alt="modelLframe1" src="https://github.com/CCBatIIT/Deep-MMS/assets/125150205/d1420725-2ef5-44f0-9385-15b2f9e7402b">

The branch focuses on Visualization aspect.

The main work is done in `Model_Analysis_2-Perturbation.ipynb` notebook. 
  
  The notebooks uses custom json file to specify the data and load a model from checkpoint,
  It auto encodes and decodes the data. The notebook perturbate the latent space then decoded it again.
  Producing data in # 5 sets of data
  
    input data = experiment.test_data
    original latents = recon[1]
    original decoded = recon[0]
    perturb latents = perturbed_latents
    perturb decoded = perturbed_decoded

  It write the data into .dcd file to furthur visualize

  It also build visualizer arrows in `Arrows_builder.ipynb` notebook.

  Results are written in modelL folder

***11/21 Meeting update*** 

    No meeting. Sick and holiday~
    
***11/14 Meeting update*** 
      
    Step 1-Perturb with other method like minus and see if it has pararell difference.
    Step 2-Explore Relationship between applied Gaussian Structures and Displacement between latents.

***11/07 Meeting update*** 
      
    Step 1-Perturb different latent spaces and visualization
    Step 2-Set up the environment in fully-awake-wahoo
    Step 3-Clean up the Github

***10/31 Meeting update*** 
      
    Step 1 - Give the new data for Urvi to test the visualization out
    Step 2 - "What is the correlation between latents and atomic variance?" Big question needs to be answered
       2.1 - Start with motion vector
    And Tada~! the arrows are born~ (Green is Original, Red is Perturbed)

***10/24 Meeting update*** 
  
    Running modelL on practical environment
    Finallize python the visualization functions 

    On modelL
      Step-1 Configure the computation environment to run it. (Probally best to have Joseph's help)
      
    On histogram-on-scatterplots
      Step-1 Get rid of empty boxes
      Step-2 Inverse the scatter box from top right - to -> bottom left
      Step-3 Make it bigger or at least readable.

    On the whole notebook
      Step-1 Make it a universally importable python file. 

***10/17 Meeting update***
  
    Perturbing the latents with new modelL
    
    Step-1 Continue working on GitHub branch issue
    Step-2 Perturb the latents using modelL.
    Step-3 visualize the difference after perturbation.
    Step-4 Write a latents by latents histogram-scatterplot analysis

***10/10 Meeting update*** 

    Implementing GitHub workflows
  
    Step-1 Update the github to match the main branch. (solve the commits behind problem)
    Step-2 Urvi and Timo. We need to manage this visualization branch. Let's get an appointment.
    Step-3 Urvi raise an issue on github and communicate with Joseph. 

***10/03 Meeting update***
  
    Exploring ChimeraX
    
    Step-1 .dcd file seems to be in nanometers rather than angstrom. Check the decimals conversion before writing it to .dcd again.
    Step-2 explore the ChimeraX program. ### use it only as a visualizer! Try to finish the perturbation and algorithms before putting it in!
    Step-3 find a way to compare original vs perturbed .dcd. Analyze your insights afterthat, move on to other latents.

=======

### Modules
NN_models - a python module containing various flax.linen.nn modules.  Used to construct AutoEncoder and Decoder pairs for experiment./
training_functions - jax vmap and jit functions which are applied during training./
AutoEncoder_Experiment - A wrapping class that actually executes the training and assembly.

