"""
Generate JSON configuration files for a Deep-MMS training sweep.

Usage:
    python scripts/generate_configs.py

Edit the CONFIG section below to control which trajectories, model variants,
and hyperparameters are swept.  One JSON file is written per
(trajectory × latent_dim × test_slice) combination.
"""

import json
import os
import sys
import numpy as np
import mdtraj as md

from deepmms.data import powers_of_two_up_to


def primes_up_to(limit):
    """Return all prime numbers up to limit as a numpy integer array."""
    if limit < 2:
        return np.array([], dtype=int)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i * i : limit + 1 : i] = False
    return np.flatnonzero(is_prime)


def latent_nums(limit):
    """Combine primes and powers-of-two into a sorted latent sweep sequence."""
    primes = primes_up_to(limit)
    twos = powers_of_two_up_to(limit)
    return sorted({int(i) for i in np.arange(limit + 1)
                   if i in primes or i in twos or i == limit})


def main():
    """Entry point: write JSON config files for a training sweep."""
    if os.path.basename(os.getcwd()) != "Deep-MMS":
        raise RuntimeError("Run this script from the Deep-MMS project root directory.")

    # ------------------------------------------------------------------
    # CONFIG — edit these lists to change the sweep
    # ------------------------------------------------------------------
    dcd_fns = ["Simulation/1crn_split2.dcd"]
    top_fns = ["Simulation/1crn_H.prmtop"]

    dcd_fns = [os.path.join(os.getcwd(), fn) if fn.startswith("Simulation") else fn
               for fn in dcd_fns]
    top_fns = [os.path.join(os.getcwd(), fn) if fn.startswith("Simulation") else fn
               for fn in top_fns]

    assert all(os.path.isfile(fn) for fn in dcd_fns), "DCD file(s) not found"
    assert all(os.path.isfile(fn) for fn in top_fns), "Topology file(s) not found"
    assert len(dcd_fns) == len(top_fns)

    sweep_params = [
        # (weight_model, model_base)
        ("Uniform_Heavy", "X013-2"),
    ]
    # ------------------------------------------------------------------

    for weight_model, model_base in sweep_params:
        if weight_model in ("Uniform", "Mass"):
            atom_selection = "all"
        elif weight_model in ("Uniform_Heavy", "Mass_Heavy", "Mass_United", "H-Valence"):
            atom_selection = "not element H"
        else:
            raise ValueError(f"Unknown weight_model: {weight_model!r}")

        model_names = [f"CR_small_{model_base}"]
        assert len(dcd_fns) == len(model_names)

        json_dir = os.path.join(os.getcwd(), "json_inputs", model_base)
        os.makedirs(json_dir, exist_ok=True)

        for dcd_fn, top_fn, model_name in zip(dcd_fns, top_fns, model_names):
            c = md.load(dcd_fn, top=top_fn)
            c = c.atom_slice(c.topology.select(atom_selection))
            latent_dims = powers_of_two_up_to(c.n_atoms) + [c.n_atoms]
            lrs = [1e-3] * len(latent_dims)

            os.makedirs(os.path.join(json_dir, model_name), exist_ok=True)

            for latent_dim, lr in zip(latent_dims, lrs):
                for test_slice in [1, 2, 3, 4, 5]:
                    json_fn = os.path.join(
                        json_dir, model_name,
                        f"{model_name}_{latent_dim:04d}_{test_slice:02d}.json",
                    )
                    json_params = dict(
                        fname_dcd=dcd_fn,
                        fname_topology=top_fn,
                        save_dir=os.getcwd(),
                        max_epoch=10001,
                        latent_dim=latent_dim,
                        test_slice=test_slice,
                        data_slice_start=0,
                        data_slice_end=500,
                        model_name=model_name,
                        batch_size=100,
                        learning_rate=lr,
                        dropout_rates=[0.0, 0.0, 0.0],
                        resume_latest=False,
                        checkpoint_interval=200,
                        is_batchnorm=False,
                        atom_selection=atom_selection,
                        weight_model=weight_model,
                    )
                    with open(json_fn, "w") as f:
                        json.dump(json_params, f, indent=4)
                    print(f"Wrote {json_fn}")


if __name__ == "__main__":
    main()
