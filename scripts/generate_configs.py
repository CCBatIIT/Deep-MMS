"""
Generate JSON configuration files for the Deep-MMS architecture-profiling sweep.

Usage:
    python scripts/generate_configs.py

Writes one JSON per cell of the grid

    architecture x latent_dim x depth x molecular_ensemble x repeat

under ``json_inputs/<SWEEP_TAG>/<model_name>/``, where

    model_name = "<ENSEMBLE>_<architecture>_D<depth>"
    filename   = "<model_name>_<latent:04d>_<repeat:02d>.json"

Each config is consumed by ``scripts/train_dispatch.py`` (or the per-architecture
``scripts/train_*.py``).  The architecture is recorded in the ``architecture``
JSON key so a single runner can train any cell.

Edit the CONFIG section to change the sweep.  Network depth is controlled by the
length of ``dropout_rates`` for every architecture EXCEPT se3 / equivariant /
flow, whose depth is set by ``n_mp_layers`` / ``n_interactions`` /
``n_coupling_layers`` respectively (emitted automatically here).
"""

import json
import os
import sys

import mdtraj as md

# Make the project root importable so the shared config helpers are found even
# when the package is not pip-installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepmms.config import atom_selection_for, depth_specific_keys

# ---------------------------------------------------------------------------
# CONFIG -- edit these to change the sweep
# ---------------------------------------------------------------------------
SWEEP_TAG = "profiling_2026"

# Molecular ensembles: (fname_dcd, fname_topology, ensemble_label).
# Paths are relative to the project root; the multi-GB trajectories for the
# larger systems are git-ignored but must exist on disk to be trained.
ENSEMBLES = [
    ("Simulation/oxycodone.dcd",                 "Simulation/oxycodone.prmtop",                "OX"),
    ("Simulation/decaalanine_1us_split3.dcd",    "Simulation/ala_deca_peptide.prmtop",         "DA"),
    ("Simulation/1crn_split2.dcd",               "Simulation/1crn_H.prmtop",                   "CR"),
    ("Simulation/3mxf_protein_ligand_only.dcd",  "Simulation/3mxf_protein_ligand_only.pdb",    "BRD"),   # bromodomain (3mxf)
    ("Simulation/HIV1p_protein_only.dcd",        "Simulation/HIV1p_protein_only.pdb",          "HIV1p"),
]

# Architecture identifiers -> dispatched to model classes by train_dispatch.py.
ARCHITECTURES = [
    "batchnorm_vae", "beta_vae", "vq_vae", "transformer", "perceiver",
    "hierarchical", "equivariant", "se3", "mamba", "flow", "mae", "kan", "neat",
]

LATENTS = [1, 2, 4, 8, 16, 32]     # clamped per-ensemble to <= n_heavy_atoms
DEPTHS = [2, 4, 6, 8, 10]          # number of hidden layers
REPEATS = [1]                      # test_slice values (1..5); one fold by default

WEIGHT_MODEL = "Uniform_Heavy"     # -> atom_selection "not element H"
LEARNING_RATE = 1e-3
BATCH_SIZE = 100
MAX_EPOCH = 10001
CHECKPOINT_INTERVAL = 200
DATA_SLICE_START = 0
DATA_SLICE_END = "None"            # "None" = use all frames
SAVE_DIR = "runs"                  # git-ignored output root
# ---------------------------------------------------------------------------


def n_heavy_atoms(top_fn, atom_selection):
    """Count selected atoms from the topology alone (no trajectory load)."""
    topology = md.load_topology(top_fn)
    return len(topology.select(atom_selection))


def main():
    """Entry point: write JSON config files for the profiling sweep."""
    root = os.getcwd()
    if not (os.path.isdir(os.path.join(root, "deepmms"))
            and os.path.isdir(os.path.join(root, "Simulation"))):
        raise RuntimeError(
            "Run this script from the Deep-MMS project root "
            "(the directory containing deepmms/ and Simulation/)."
        )

    atom_selection = atom_selection_for(WEIGHT_MODEL)
    n_written = 0

    for dcd_rel, top_rel, ensemble in ENSEMBLES:
        dcd_fn = os.path.join(root, dcd_rel)
        top_fn = os.path.join(root, top_rel)
        if not os.path.isfile(dcd_fn):
            print(f"  SKIP {ensemble}: trajectory not found ({dcd_rel})")
            continue
        if not os.path.isfile(top_fn):
            print(f"  SKIP {ensemble}: topology not found ({top_rel})")
            continue

        n_atoms = n_heavy_atoms(top_fn, atom_selection)
        latents = [l for l in LATENTS if l <= n_atoms]
        print(f"{ensemble}: {n_atoms} selected atoms -> latents {latents}")

        for architecture in ARCHITECTURES:
            for depth in DEPTHS:
                model_name = f"{ensemble}_{architecture}_D{depth:02d}"
                out_dir = os.path.join(root, "json_inputs", SWEEP_TAG, model_name)
                os.makedirs(out_dir, exist_ok=True)

                for latent_dim in latents:
                    for test_slice in REPEATS:
                        params = dict(
                            architecture=architecture,
                            fname_dcd=dcd_fn,
                            fname_topology=top_fn,
                            save_dir=os.path.join(root, SAVE_DIR),
                            max_epoch=MAX_EPOCH,
                            latent_dim=latent_dim,
                            test_slice=test_slice,
                            data_slice_start=DATA_SLICE_START,
                            data_slice_end=DATA_SLICE_END,
                            model_name=model_name,
                            batch_size=BATCH_SIZE,
                            learning_rate=LEARNING_RATE,
                            dropout_rates=[0.0] * depth,
                            resume_latest=False,
                            checkpoint_interval=CHECKPOINT_INTERVAL,
                            is_batchnorm=(architecture == "batchnorm_vae"),
                            atom_selection=atom_selection,
                            weight_model=WEIGHT_MODEL,
                        )
                        params.update(depth_specific_keys(architecture, depth))

                        json_fn = os.path.join(
                            out_dir,
                            f"{model_name}_{latent_dim:04d}_{test_slice:02d}.json",
                        )
                        with open(json_fn, "w") as f:
                            json.dump(params, f, indent=4)
                        n_written += 1

    print(f"\nWrote {n_written} config(s) under json_inputs/{SWEEP_TAG}/")


if __name__ == "__main__":
    main()
