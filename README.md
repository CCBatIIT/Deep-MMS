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

Then install the package to get the `deep-mms` command (use `--no-deps` so the
CUDA-specific jax build in your environment is left untouched):

```bash
pip install -e . --no-deps
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
├── cli.py                       # `deep-mms` command: train one model from CLI args
├── dispatch.py                  # architecture id -> training harness (build_harness)
├── config.py                    # assemble a config dict from CLI/sweep knobs
├── utils.py                     # printf timestamp logger; JAX x64 config
├── data.py                      # trajectory loading, train/test split, batching, mass weights
├── models/
│   ├── base.py                  # MolecularAutoencoder abstract base class
│   ├── vae.py                   # BatchNorm_VAE  (default; MLP with optional BN/Dropout)
│   ├── beta_vae.py              # BetaVAE        (tuneable KL weight β)
│   ├── transformer_vae.py       # TransformerVAE (atom-token self-attention)
│   ├── equivariant_vae.py       # EquivariantVAE (SchNet-style pairwise distances)
│   ├── perceiver_vae.py         # PerceiverVAE   (O(N·M) cross-attention)
│   ├── hierarchical_vae.py      # HierarchicalVAE (two-level global/local latent)
│   ├── se3_transformer.py       # SE3TransformerVAE (PaiNN equivariant message passing)
│   ├── mamba_vae.py             # MambaVAE       (selective SSM, O(N) cost)
│   ├── flow_vae.py              # RealNVPFlow    (invertible normalising flow)
│   ├── mae_vae.py               # MaskedAutoencoder (atom-level masking)
│   ├── kan_vae.py               # KANVAE         (B-spline edge activations)
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
├── generate_configs.py          # write the profiling grid (arch × latent × depth × ensemble)
├── train_dispatch.py            # train any cell: dispatches on the config's "architecture" key
├── run_sweep.sbatch             # SLURM array runner over the generated configs
├── train.py                     # train BatchNorm_VAE with Adam
├── train_beta_vae.py            # train BetaVAE with Adam
├── train_transformer.py         # train TransformerVAE with Adam
├── train_equivariant.py         # train EquivariantVAE with Adam
├── train_perceiver.py           # train PerceiverVAE with Adam
├── train_hierarchical.py        # train HierarchicalVAE with Adam
├── train_se3.py                 # train SE3TransformerVAE with Adam
├── train_mamba.py               # train MambaVAE with Adam
├── train_flow.py                # train RealNVPFlow with Adam
├── train_mae.py                 # train MaskedAutoencoder with Adam
├── train_kan.py                 # train KANVAE with Adam
├── train_neat.py                # train NEATAutoencoder with OpenES + topology growth
├── train_vq_vae.py              # train VQVAE with Adam
├── compute_violin_data.py       # compute per-frame VAE vs PCA RMSD arrays (.npy)
├── backup_numpy.py              # copy .npy files to versioned numpy_backups/
├── clustering_metrics.py        # evaluate clustering metrics on latent representations
├── perturbation.py              # generate latent-sweep DCD trajectories
└── report_sigma.py              # audit reparameterization sigma (posterior collapse) from checkpoints
```

Training outputs and bulk trajectory data are git-ignored (`runs/`,
`results_archive/`, `Simulation/`, `**/checkpoint_managed/`, `*.nc`, …); see
`.gitignore`.  Prior experiment outputs were relocated to `results_archive/`
during the 2026 cleanup — nothing was deleted.

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

### 0. Train one model from the command line

For an ad-hoc run, the `deep-mms` command builds the config from its arguments —
no JSON file needed.  Give the trajectory and topology, then any number of
`key=value` overrides:

```bash
deep-mms Simulation/oxycodone.dcd Simulation/oxycodone.prmtop \
    architecture=batchnorm_vae latents=8 depth=4 batchnorm=True

deep-mms Simulation/1crn_split2.dcd Simulation/1crn_H.prmtop \
    architecture=transformer latents=16 depth=6 epochs=5000

deep-mms --list-architectures                 # show the 13 architectures
deep-mms <dcd> <top> latents=8 depth=4 --dry-run   # print the config, don't train
```

Common keys (aliases in parentheses): `architecture`, `latents` (`latent_dim`),
`depth` (→ number of hidden layers), `batchnorm` (`is_batchnorm`), `epochs`
(`max_epoch`), `lr` (`learning_rate`), `batch_size`, `test_slice`,
`weight_model`, `save_dir`, `model_name`.  Any other config key may be passed the
same way.  Output goes to `save_dir/<model_name>/<latent>_latents/rpt_<slice>/`
(`save_dir` defaults to `runs/`).

### 1. Generate the profiling-sweep configs

`scripts/generate_configs.py` writes one JSON per cell of the grid
**architecture × latent {1,2,4,8,16,32} × depth {2,4,6,8,10} × ensemble**
(latents are clamped per system to ≤ heavy-atom count).  Edit the `CONFIG`
section at the top (`ENSEMBLES`, `ARCHITECTURES`, `LATENTS`, `DEPTHS`, `REPEATS`)
to change any axis, then:

```bash
cd /path/to/Deep-MMS
python scripts/generate_configs.py
```

Configs are written under `json_inputs/profiling_2026/<ensemble>_<arch>_D<depth>/`
and training output goes to `runs/` (both git-ignored).  Network depth is set by
`len(dropout_rates)` for every architecture except `se3`/`equivariant`/`flow`,
whose depth keys (`n_mp_layers`/`n_interactions`/`n_coupling_layers`) are emitted
automatically.

### 2. Train a model

Every generated config carries an `"architecture"` key, so a single dispatcher
trains any cell:

```bash
python scripts/train_dispatch.py json_inputs/profiling_2026/CR_batchnorm_vae_D04/CR_batchnorm_vae_D04_0008_01.json
```

The per-architecture scripts (`scripts/train.py`, `train_transformer.py`,
`train_neat.py`, …) remain available if you prefer to pin the model explicitly.
On an HPC cluster, submit the whole sweep as a SLURM array:

```bash
find json_inputs/profiling_2026 -name '*.json' | sort > sweep_configs.txt
sbatch --array=1-$(wc -l < sweep_configs.txt)%50 scripts/run_sweep.sbatch sweep_configs.txt
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

Architecture-specific keys are documented in `deepmms/MODEL_DOCUMENTATION.md` under the section for each model.

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

| System | Label | Heavy atoms | Data (dcd / topology) |
|---|---|---|---|
| Oxycodone | OX | 23 | `Simulation/oxycodone.dcd` / `oxycodone.prmtop` |
| Deca-alanine | DA | 50 | `Simulation/decaalanine_1us_split3.dcd` / `ala_deca_peptide.prmtop` |
| Crambin | CR | 327 | `Simulation/1crn_split2.dcd` / `1crn_H.prmtop` |
| Bromodomain (3mxf) | BRD | 1093 | `Simulation/3mxf_protein_ligand_only.dcd` / `.pdb` |
| HIV-1 Protease | HIV1p | 1599 | `Simulation/HIV1p_protein_only.dcd` / `.pdb` |

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
