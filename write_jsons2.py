import json, sys, os

with open(sys.argv[1], 'r') as g:
    B = json.load(g)

model_name = B["model_name"]
if not os.path.isdir('./jsons'):
    os.mkdir('jsons')

if not os.path.isdir(f'jsons/{model_name}'):
    os.mkdir(f'jsons/{model_name}')

latents = [2, 4, 8, 16, 32, 64, 128, 256, 306]

for lat in latents:
    for tes in [0,1,2,3,4]:
        B["latent_dim"] = lat
        B["test_slice"] = tes
        with open(f'jsons/{model_name}/{model_name}_{lat:04d}_{tes}.json', 'w') as g:
            json.dump(B, g)
