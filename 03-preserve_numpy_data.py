import os, sys, argparse, glob, json
import numpy as np

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('json_fn', help="input json given to run the model")

args = parser.parse_args()

with open(args.json_fn, 'r') as g:
    json_params = json.load(g)

data_dir = os.path.join(json_params["save_dir"],
                        str(json_params["model_name"]),
                        f'{json_params["latent_dim"]:04d}_latents/',
                        f'rpt_{json_params["test_slice"]}/')

new_numpy_dir = os.path.join(json_params["save_dir"],
                             'numpy_backups',
                             str(json_params["model_name"]),
                             f'{json_params["latent_dim"]:04d}_latents/',
                             f'rpt_{json_params["test_slice"]}/')

if not os.path.isdir(new_numpy_dir):
    os.makedirs(new_numpy_dir, exist_ok=True)

numpy_files = glob.glob(os.path.join(data_dir, '*.npy'))
numpy_arrays = [np.load(numpy_file) for numpy_file in numpy_files]

for numpy_fn in numpy_files:
    new_numpy_fn = os.path.join(new_numpy_dir, os.path.basename(numpy_fn))
    arr = np.load(numpy_fn)
    _ = np.save(new_numpy_fn, arr)