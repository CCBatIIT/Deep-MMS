import json, sys, os

# USAGE python write_jsons2.py json_fn max_latents

with open(sys.argv[1], 'r') as g:
    B = json.load(g)

max_latents = int(sys.argv[2])

model_name = B["model_name"]
if not os.path.isdir('./jsons'):
    os.mkdir('jsons')

if not os.path.isdir(f'jsons/{model_name}'):
    os.mkdir(f'jsons/{model_name}')

latents = []
i = 0
while 2**i < max_latents:
    latents.append(2**i)
    i += 1
latents.append(max_latents)

for lat in latents:
    for tes in [0,1,2,3,4]:
        B["latent_dim"] = lat
        B["test_slice"] = tes
        with open(f'jsons/{model_name}/{model_name}_{lat:04d}_{tes}.json', 'w') as g:
            json.dump(B, g)
