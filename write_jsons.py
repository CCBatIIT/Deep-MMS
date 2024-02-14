import json, sys, os

with open(sys.argv[1], 'r') as g:
    B = json.load(g)

model_name = sys.argv[1].split('.')[0]
if not os.path.isdir(f'jsons/{model_name}'):
    os.mkdir(f'jsons/{model_name}')

for lat in range(2, 51, 2):
    for tes in [2, 0]:
        B["latent_dim"] = lat
        B["test_slice"] = tes
        with open(f'jsons/{model_name}/{model_name}_{lat:02d}_{tes}.json', 'w') as g:
            json.dump(B, g)
