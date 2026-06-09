# Deep-MMS

**Deep Molecular Mechanics Surrogate** — a JAX/Flax library for compressing and reconstructing molecular dynamics ensembles using variational autoencoders.

Given a trajectory of heavy-atom Cartesian coordinates, Deep-MMS trains a VAE to learn a low-dimensional latent representation and reconstruct frames with minimal RMSD.  Multiple architectures are provided so you can benchmark compression quality as a function of latent dimensionality and model class.

---

## Environment setup

### Linux + CUDA 12/13

```bash
conda config --prepend channels conda-forge
conda create -n deepmms
conda activate deepmms
conda install python pip jupyter netCDF4 mdtraj matplotlib openmm scikit-learn
pip install -U "jax[cuda13]"
pip install flax optax orbax-checkpoint
```

Or from the provided environment file:
```bash
conda env create --file jetstream2_env.yml
```

Verify GPU detection in Python:
```python
import jax
print(jax.print_environment_info(), jax.default_backend())
# Should print nvidia-smi output and report 'gpu'
```

---

## Background

Classical force-field MD simulations produce trajectories that are expensive to store, redundant in coordinate space, and difficult to analyse at scale.  Deep-MMS addresses this by learning a compact, differentiable encoding of the conformational ensemble.  The latent space can be used for:

- **Conformational clustering** — compare VAE vs PCA cluster separation quality across all six metrics
- **Landscape visualization** — project high-dimensional ensembles onto 2–3 latent axes
- **Latent perturbation** — sweep individual latent dimensions to generate physically meaningful conformational series
- **Cross-model comparison** — evaluate whether different architectures capture the same conformational modes

---

## Package structure

```
deepmms/                         # installable Python package
├── utils.py                     # printf timestamp logger; JAX x64 config
├── data.py                      # trajectory loading, train/test split, batching, mass weights
├── models/
│   ├── base.py                  # MolecularAutoencoder abstract base class
│   ├── vae.py                   # BatchNorm_VAE  (default; MLP with optional BN/Dropout)
│   ├── transformer_vae.py       # TransformerVAE (atom-token self-attention)
│   └── neat_vae.py              # NEATAutoencoder (growing tanh MLP)
├── training/
│   ├── loss.py                  # atom_rmsd, give_weighted_rmsd_func, KL_loss
│   ├── optimizer.py             # warmup-cosine LR schedule, TrainState factory
│   ├── trainer.py               # Experiment — gradient-based Adam training harness
│   └── evolutionary.py          # EvolutionaryTrainer (OpenES) + NEATTrainer (ES + growth)
└── analysis/
    ├── reconstruction.py        # Analyzer, violin_data — VAE vs PCA RMSD comparison
    ├── clustering.py            # KMeans / Agglomerative / HDBScan metric evaluation
    ├── perturbation.py          # per-latent sweep trajectory generation
    └── plotting.py              # violin plots, difference plots, figure log CSVs

scripts/                         # CLI entry points (all take a JSON config file)
├── generate_configs.py          # write a sweep of JSON configs for a trajectory set
├── train.py                     # train BatchNorm_VAE with Adam
├── train_transformer.py         # train TransformerVAE with Adam
├── train_neat.py                # train NEATAutoencoder with OpenES + topology growth
├── compute_violin_data.py       # compute per-frame VAE vs PCA RMSD arrays (.npy)
├── backup_numpy.py              # copy .npy files to versioned numpy_backups/
├── clustering_metrics.py        # evaluate clustering metrics on latent representations
└── perturbation.py              # generate latent-sweep DCD trajectories

archive/pyscripts/               # superseded code preserved for reference
```

---

## Architectures

### `BatchNorm_VAE` (default)

Symmetric encoder–decoder MLP.  Each hidden layer is a square dense transform (width = input size), making it a series of invertible-like linear mappings regularised by optional BatchNorm and Dropout.

```
Input (n_atoms×3) → [Dense → ReLU → (BN) → Dropout] × n_layers → z_mean, z_logvar
                  ← [Dense → ReLU → (BN) → Dropout] × n_layers ← z
```

Best for: establishing a baseline; fast training on any system size.

### `TransformerVAE`

Treats each heavy atom as a sequence token with 3-dimensional features (x, y, z).  Multi-head self-attention in the encoder allows every atom to attend to every other atom, capturing long-range contacts without explicit bond topology.

```
Input → reshape (n_atoms, 3) → AtomEmbedding+PE → [MHA + FFN] × n_blocks → mean-pool → z_mean, z_logvar
      ← reshape (n_atoms×3)  ← Dense(3)          ← LayerNorm  ← [MHA + FFN] × n_blocks ← tile(z, n_atoms)
```

Best for: large proteins where inter-residue coupling drives conformational change (BR, HIV1p).

JSON extras:

| Key | Default | Description |
|---|---|---|
| `embed_dim` | `min(256, input_size)` | Attention embedding width per atom token |
| `num_heads` | `4` | Attention heads per block (must divide embed_dim) |
| `ffn_mult` | `4.0` | FFN hidden width as multiple of embed_dim |

### `NEATAutoencoder` + `NEATTrainer`

NEAT-inspired growing MLP.  Starts with a compact tanh network and adds hidden layers whenever the fitness (negative reconstruction RMSD) plateaus.  Weights are optimised by **OpenAI Evolution Strategies** (OpenES) — gradient-free perturbation-based gradient estimation — which avoids vanishing gradients and requires no differentiable loss.

Topology growth follows NEAT's *minimal structural innovation* principle: new layer weights initialise near zero so the network's output changes smoothly upon layer addition.

```
Generation t:
  1. Sample N perturbations: θᵢ = θ + σ·εᵢ, εᵢ ~ N(0, I)  [antithetic pairs]
  2. Evaluate fitness F(θᵢ) = −mean RMSD on one training batch
  3. Rank-normalise fitnesses → [-0.5, 0.5]
  4. ES gradient estimate: ĝ = −(1/Nσ) Σ Fᵢ·εᵢ
  5. Apply Adam moment estimates to ĝ → update θ
  6. If max(fitness[-window:]) − max(fitness[:-window]) < threshold → grow topology
```

JSON extras:

| Key | Default | Description |
|---|---|---|
| `neat_start_dim` | `min(64, input_size)` | Initial hidden layer width |
| `neat_start_layers` | `len(dropout_rates)` | Initial number of hidden layers |
| `es_population` | `50` | Perturbation population size N |
| `es_sigma` | `0.05` | Perturbation standard deviation σ |
| `es_lr` | `0.01` | Adam learning rate applied to ES gradient |
| `neat_plateau_window` | `200` | Generations without threshold improvement before growth |
| `neat_plateau_thr` | `0.005` | Minimum fractional fitness improvement to reset plateau window |

---

## Quickstart

### 1. Generate JSON configs

Edit the trajectory/model lists at the top of `scripts/generate_configs.py`, then:

```bash
cd /path/to/Deep-MMS
python scripts/generate_configs.py
```

This writes one JSON per (latent dimension × test-set repeat) combination under `json_inputs/<model_base>/`.

### 2. Train a model

```bash
# Standard MLP VAE (Adam)
python scripts/train.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json

# Transformer VAE
python scripts/train_transformer.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json

# NEAT (OpenES + automatic topology growth)
python scripts/train_neat.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json
```

All scripts must be run from the project root (the directory containing `deepmms/`).

### 3. Compute reconstruction RMSD (for violin plots)

```bash
python scripts/compute_violin_data.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json
```

Writes `VAE_RMSD.npy`, `VAE_LOSS_RMSD.npy`, `PCA_RMSD.npy`, `PCA_LOSS_RMSD.npy` to the model's output directory.

### 4. Evaluate clustering

```bash
python scripts/clustering_metrics.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json
```

Writes `clustering_metric_log.txt` with all six metrics (Rand, NMI, Fowlkes-Mallows, Silhouette, Davies-Bouldin, Calinski-Harabasz) for KMeans, Agglomerative, and HDBScan.

### 5. Generate latent perturbation trajectories

```bash
python scripts/perturbation.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json
```

Sweeps each latent dimension from mean−5σ to mean+5σ, writing one DCD per latent.

### 6. Back up numpy arrays

```bash
python scripts/backup_numpy.py json_inputs/X013-2/CR_X013-2/CR_X013-2_0004_01.json
```

---

## JSON configuration reference

| Key | Type | Description |
|---|---|---|
| `fname_dcd` | str | Path to DCD trajectory file |
| `fname_topology` | str | Path to topology (PDB or PRMTOP) |
| `save_dir` | str | Root output directory |
| `model_name` | str | Identifier (e.g. `CR_X013-2`) |
| `latent_dim` | int | Number of latent dimensions |
| `test_slice` | int | Test-set offset 1–5 (every 5th frame) |
| `data_slice_start` | int | First frame index |
| `data_slice_end` | int or `"None"` | Last frame index (exclusive) |
| `batch_size` | int | Mini-batch size |
| `learning_rate` | float | Peak learning rate |
| `dropout_rates` | list[float] | Per-layer dropout rates; length = number of hidden layers |
| `resume_latest` | bool | Resume from most recent checkpoint |
| `checkpoint_interval` | int | Save checkpoint every N epochs |
| `max_epoch` | int | Training cutoff epoch |
| `is_batchnorm` | bool | Enable BatchNorm (BatchNorm_VAE only) |
| `atom_selection` | str | MDTraj DSL selection string (e.g. `"not element H"`) |
| `weight_model` | str | Mass-weighting scheme (`"Uniform_Heavy"`, `"Mass_Heavy"`, `"Mass_United"`, `"H-Valence"`) |

Architecture-specific keys are described in the table for each model above.

---

## Output structure

For each JSON config, training produces:

```
<save_dir>/<model_name>/<latent_dim>_latents/rpt_<test_slice>/
├── model_<name>_<latent>.nc           # NetCDF4 loss log (Train/Test RMSD per epoch/batch)
├── model_<name>_<latent>_checkpoint.nc
├── checkpoint_managed/                # Orbax checkpoints (max 2 kept)
├── figures/
│   ├── <N> Latents - RMSD Loss Term.png
│   ├── Reconstruction RMSD.png
│   └── All Latents.png
├── VAE_RMSD.npy                       # per-frame unweighted VAE RMSD (nm)
├── VAE_LOSS_RMSD.npy                  # per-frame weighted VAE RMSD
├── PCA_RMSD.npy                       # per-frame unweighted PCA RMSD
├── PCA_LOSS_RMSD.npy
├── clustering/
│   ├── TestDistances.sqmat.npy        # pairwise RMSD distance matrix
│   ├── clustering_metric_log.txt      # 6 metrics × 3 clustering methods
│   └── optimal_n_clusters_*.png
└── perturbation/
    ├── <model>_pLatent<i>.dcd         # one DCD per latent dimension
    ├── <model>_test_data.dcd
    ├── <model>_test_recon.dcd
    └── perturbation_log.txt
```

---

## Adding a new architecture

1. Create `deepmms/models/my_model.py` subclassing `MolecularAutoencoder`.
2. Implement `encode(x, z_rng, train)`, `decode(z, z_rng, train)`, and `__call__(x, z_rng, train)`.
3. Override `hidden_layers_from_config` if the default square-MLP layer sizing doesn't apply.
4. Register it in `deepmms/models/__init__.py`.
5. Pass `model_cls=MyModel` to `Experiment` — no other files need to change.

---

## Molecular systems

| System | Label | Heavy atoms | Length |
|---|---|---|---|
| Oxycodone | OX | 23 | — |
| Deca-alanine | DA | 50 | — |
| Crambin | CR | 327 | — |
| BPTI | BR | 1093 | 100 ns |
| HIV-1 Protease | HIV1p | 1599 | 1 µs |

---

## Dependencies

| Package | Purpose |
|---|---|
| `jax` / `jaxlib` | Array ops, JIT compilation, vmap |
| `flax` | Neural network modules (linen API) |
| `optax` | Optimisers, LR schedules |
| `orbax-checkpoint` | Model checkpointing |
| `mdtraj` | Trajectory loading and topology |
| `netCDF4` | Loss-curve storage |
| `scikit-learn` | PCA baseline, clustering metrics |
| `numpy` / `matplotlib` | Numerics and plotting |

---

## Citation

If you use Deep-MMS in published work, please cite the associated manuscript (in preparation, 2026).
