"""
Data loading, alignment, splitting, and batching utilities for molecular dynamics trajectories.

Provides Data_stream for mini-batch iteration, load_and_align for MDTraj-based
coordinate preparation, train_test_split for the standard every-fifth-frame split,
mass_weights for per-atom weighting schemes, and powers_of_two_up_to for generating
canonical latent-dimension sweep sequences.
"""

import numpy as np
import numpy.random as npr
import jax.numpy as jnp
import mdtraj as md
from copy import deepcopy


class Data_stream:
    """
    Infinite iterator that yields randomly permuted mini-batches from a fixed dataset.

    Parameters
    ----------
    rng_seed : int
        Seed for the NumPy RandomState used to shuffle indices.
    num_total : int
        Total number of samples in the dataset.
    num_batches : int
        Number of batches per epoch (ceil(num_total / batch_size)).
    batch_size : int
        Number of samples per batch.
    data : array-like
        Full dataset array indexed on axis 0.
    """

    def __init__(self, rng_seed, num_total, num_batches, batch_size, data):
        self.rng_seed = rng_seed
        self.num_total = num_total
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = data

    def __iter__(self):
        rng = npr.RandomState(self.rng_seed)
        while True:
            perm = rng.permutation(self.num_total)
            for i in range(self.num_batches):
                batch_idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
                yield self.data[batch_idx]


def load_and_align(fname_dcd, fname_topology, atom_selection, data_start, data_end):
    """
    Load a trajectory with MDTraj, slice to heavy atoms, superpose, and return
    both the MDTraj object and a flattened JAX coordinate array.

    Parameters
    ----------
    fname_dcd : str
        Path to the DCD trajectory file.
    fname_topology : str
        Path to the topology file (PDB, PRMTOP, etc.).
    atom_selection : str
        MDTraj DSL selection string (e.g. 'not element H').
    data_start : int
        First frame index to include.
    data_end : int or None
        One-past-last frame index; None means include all remaining frames.

    Returns
    -------
    c : md.Trajectory
        Aligned MDTraj trajectory sliced to the requested atoms and frames.
    coord_set : jnp.ndarray, shape (n_frames, n_atoms*3)
        Flattened coordinate array in nanometres.
    """
    c = md.load(fname_dcd, top=fname_topology)
    c = c.atom_slice(c.topology.select(atom_selection))
    c = c.superpose(c)
    coord_set = jnp.array(c.xyz.reshape(c.xyz.shape[0], -1))[data_start:data_end]
    return c, coord_set


def train_test_split(coord_set, test_slice):
    """
    Split coordinates into train and test sets using an every-fifth-frame rule.

    Parameters
    ----------
    coord_set : jnp.ndarray, shape (n_frames, n_features)
        Full aligned coordinate array.
    test_slice : int
        Offset (0–4 inclusive) selecting which frames become the test set.

    Returns
    -------
    train_data : jnp.ndarray
    test_data : jnp.ndarray
    """
    num_samples = coord_set.shape[0]
    test_indices = np.array(range(test_slice, num_samples, 5))
    test_set = set(test_indices.tolist())
    train_indices = np.array([i for i in range(num_samples) if i not in test_set])
    return coord_set[train_indices], coord_set[test_indices]


def mass_weights(traj):
    """
    Compute several per-heavy-atom mass weighting schemes for a trajectory.

    Builds a mapping between heavy-atom indices (in the heavy-only topology)
    and all-atom indices, then accumulates hydrogen masses onto their bonded
    heavy partners for the unified model.

    Parameters
    ----------
    traj : md.Trajectory
        Full all-atom trajectory (used to read topology and heavy-atom subset).

    Returns
    -------
    dict with keys:
        'Uniform'        – ones over all atoms
        'Uniform_Heavy'  – ones over heavy atoms only
        'Mass'           – actual masses of all atoms
        'Mass_Heavy'     – actual masses of heavy atoms only
        'Mass_United'    – heavy-atom masses with H contributions folded in
        'H-Valence'      – count of hydrogen bonds per heavy atom (+ 1)
    """
    H = md.element.hydrogen
    traj_heavy = traj.atom_slice(traj.top.select("not element H"))
    masses = np.array([traj.top.atom(i).element.mass for i in range(traj.n_atoms)])
    index_map = np.array([[i, 0] for i in range(traj_heavy.n_atoms)])
    info = lambda atom: (
        atom.residue.name,
        atom.residue.index,
        atom.name,
        atom.element.mass,
    )
    for i in range(traj_heavy.n_atoms):
        for j in range(traj.n_atoms):
            atom_i, atom_j = traj_heavy.top.atom(i), traj.top.atom(j)
            if info(atom_i) == info(atom_j):
                index_map[i, 1] = j
                break

    print(traj, traj.topology, traj_heavy, traj_heavy.topology)
    assert np.allclose(traj.xyz[:, index_map[:, 1]], traj_heavy.xyz[:, index_map[:, 0]])

    heavy_masses = np.array(
        [traj_heavy.top.atom(i).element.mass for i in range(traj_heavy.n_atoms)]
    )
    assert np.all(heavy_masses[index_map[:, 0]] == masses[index_map[:, 1]])

    mass_unified = deepcopy(heavy_masses)
    mass_valence = np.ones(heavy_masses.shape)

    for bond in traj.top.bonds:
        if bond.atom1.element == H or bond.atom2.element == H:
            mass_unified[
                np.where(index_map[:, 1] == bond.atom1.index)[0]
            ] += masses[bond.atom2.index]
            mass_valence[
                np.where(index_map[:, 1] == bond.atom1.index)[0]
            ] += 1

    return {
        "Uniform": np.ones(traj.n_atoms),
        "Uniform_Heavy": np.ones(traj_heavy.n_atoms),
        "Mass": masses,
        "Mass_Heavy": heavy_masses,
        "Mass_United": mass_unified,
        "H-Valence": mass_valence,
    }


def powers_of_two_up_to(limit):
    """
    Return the canonical latent-dimension sweep sequence: [1, 2, 3, 4, 8, 16, …]
    up to the highest power of two not exceeding *limit*.

    Parameters
    ----------
    limit : int
        Upper bound (inclusive for exact powers of two).

    Returns
    -------
    list of int
    """
    log_limit = np.log2(limit) // 1
    latents = [1, 2, 3] + [int(i) for i in 2 ** np.arange(2, log_limit + 1, dtype=int)]
    return latents
