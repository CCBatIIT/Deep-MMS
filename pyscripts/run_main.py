
try:
    from main import *
except:
    from .main import *




import sys

assert len(sys.argv) == 2

#Set json file
json_fn = sys.argv[1]

#Init
experiment = NN_Experiment(json_fn)

#Run RMSD Only
experiment.MAIN_train_rmsd_only_wo_reporting_potential()

