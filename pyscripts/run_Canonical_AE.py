import Canonical_AutoEncoder_Experiment
import sys, jax
jax.config.update("jax_enable_x64", True)


#JSON is arg
json_fn = sys.argv[1]

#To be added to json
num_rmsd_epochs = 100 #Initial Run
nm_cutoff = 0.1 #target rmsd in nm
rmsd_cutoff = 5000 #num epochs to force training to summation
num_move_ave = 50 #number of previous points for moving average calculations
potential_threshold = 1e-3 #Target Potential NSD tolerance
cutoff_epoch = 30000 #maximum number of epochs of training

#Init Experiment
experiment = Canonical_AutoEncoder_Experiment.NN_Experiment(json_fn)

#Save Testing Data
experiment.write_traj("test_data", experiment.test_data)

# Automatic training

# RMSD BLOCK 1
# Train on RMSD until average of last 100 epochs <1 angstrom
# Get first 100 vals
print('START RMSD')
experiment.train_nepochs_on_rmsd(num_rmsd_epochs)
experiment.save_loss_data()
experiment.write_model_to_ckpt()

# RMSD BLOCK 2
# Train until last 100 vals average less than predefined cutoff, always make sure we never train longer than cutoff_epoch
begin_scaling_epoch = experiment.train_rmsd_threshold(nm_cutoff, num_move_ave, rmsd_cutoff)
experiment.write_model_to_ckpt()
print('END RMSD')

# SCALE IN POTENTIAL BLOCK
print('START SCALING POTENTIAL')
end_scaling_epoch = experiment.train_scaling_potential(cutoff_epoch)
print('END SCALING POTENTIAL')
experiment.save_loss_data()
experiment.write_model_to_ckpt()

experiment.train_summation(potential_threshold, num_move_ave, cutoff_epoch)
print('END TRAINING')
experiment.save_loss_data()
experiment.write_model_to_ckpt()
experiment.graph_losses(begin_scaling_epoch=begin_scaling_epoch, end_scaling_epoch=end_scaling_epoch)
