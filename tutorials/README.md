# Deep-MMS Tutorials

Four standalone Python scripts that teach the Deep-MMS API from the ground up.
All tutorials use synthetic data — no trajectory files required.

Run each script from the **project root** directory:

```bash
cd /path/to/Deep-MMS
python tutorials/01_quickstart.py
python tutorials/02_architecture_tour.py
python tutorials/03_latent_space.py
python tutorials/04_adding_a_model.py
```

---

## Tutorial 1 — Training Your First Model (`01_quickstart.py`)

Covers the complete minimal workflow using the low-level API:
- The JSON config format (Python dict, no files needed)
- Instantiating `BatchNorm_VAE` via `hidden_layers_from_config`
- Building a `TrainState` and running 5 gradient steps by hand
- The warmup-cosine LR schedule (`create_warmup_cosine_schedule`)
- Encoding test data and inspecting the latent shape

---

## Tutorial 2 — Comparing Architectures (`02_architecture_tour.py`)

Instantiates five architectures on the same 100-frame dataset and prints:
- Parameter count per model
- Hidden layer widths from `hidden_layers_from_config`
- Forward-pass output shapes
- Mean and std of per-frame reconstruction RMSD at random init

Models covered: `BatchNorm_VAE`, `BetaVAE`, `TransformerVAE`,
`EquivariantVAE`, `KANVAE`.

---

## Tutorial 3 — Exploring the Latent Space (`03_latent_space.py`)

After training a `BatchNorm_VAE` for 20 epochs:
- Per-dimension latent statistics (min, max, std)
- Latent traversal: sweep dimension 0 from −3σ to +3σ and print RMSD
- β=4 vs β=1 comparison: shows that `BetaVAE` produces a more constrained
  posterior (smaller per-dimension std)

---

## Tutorial 4 — Adding a Custom Architecture (`04_adding_a_model.py`)

Walks through the full custom-model template:
- Subclassing `MolecularAutoencoder` with a `LinearAutoencoder`
  (encoder: D→K Dense, decoder: K→D Dense, no hidden layers, no activation)
- Overriding `hidden_layers_from_config` to return no hidden layers
- Training it with the same low-level step loop used throughout the tutorials
- Comparing final RMSD to `BatchNorm_VAE` on the same data
- Showing the `Experiment` usage pattern for real trajectory files

---

## Prerequisites

```bash
pip install jax flax optax  # core requirements (GPU optional)
```

The full package install (including MDTraj for real trajectories) is described
in the project `README.md`.
