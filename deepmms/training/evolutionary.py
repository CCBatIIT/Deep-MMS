"""
OpenAI Evolution Strategies (OpenES) trainer for molecular autoencoders.

OpenES is a gradient-free black-box optimiser that estimates the gradient of the
expected fitness via population perturbations rather than backpropagation.  It
converges more slowly than Adam on smooth loss landscapes but is immune to
vanishing/exploding gradients, requires no differentiable loss, and naturally
produces diverse weight distributions useful for ensemble analysis.

Algorithm (Salimans et al. 2017):
    For each generation t:
        1. Sample N perturbations εᵢ ~ N(0, I) around the current mean θ.
        2. Evaluate fitness F(θ + σεᵢ) for each perturbation.
        3. Rank-normalise fitnesses to reduce sensitivity to scale.
        4. Gradient estimate: ĝ = (1/Nσ) Σ Fᵢ * εᵢ
        5. Update: θ ← θ + α * ĝ  (Adam moment estimates applied to ĝ)

The NEATTrainer subclass wraps this with topology growth: when the best fitness
has not improved by more than plateau_threshold for plateau_window consecutive
generations, it calls deepmms.models.neat_vae.grow() to add a hidden layer and
continues evolution with the larger network.

Usage
-----
Use EvolutionaryTrainer as a drop-in replacement for Experiment when you want
gradient-free weight optimisation (any model class).  Use NEATTrainer for the
full NEAT experience (gradient-free + topology growth on NEATAutoencoder).

Both are compatible with the same JSON config format; extra keys:

    es_population   : int   – perturbation population size (default: 50)
    es_sigma        : float – perturbation std deviation (default: 0.05)
    es_lr           : float – Adam step size applied to ES gradient (default: 0.01)
    neat_plateau_window : int   – generations without improvement before growth (default: 200)
    neat_plateau_thr    : float – minimum fractional improvement to reset window (default: 0.005)
    neat_start_dim      : int   – initial hidden width (passed to NEATAutoencoder)
    neat_start_layers   : int   – initial layer count (passed to NEATAutoencoder)
"""

import os
import time
import json
import glob
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax.training import train_state, orbax_utils

from .trainer import Experiment
from .loss import give_weighted_rmsd_func
from .optimizer import create_warmup_cosine_schedule
from ..data import Data_stream, load_and_align, train_test_split, mass_weights
from ..models.vae import BatchNorm_VAE
from ..utils import printf


# ---------------------------------------------------------------------------
# Flat-parameter utilities
# ---------------------------------------------------------------------------

def _flatten_params(params):
    """Flatten a nested Flax param dict to a single 1-D array."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat = jnp.concatenate([x.ravel() for x in leaves])
    return flat, treedef, [x.shape for x in leaves]


def _unflatten_params(flat, treedef, shapes):
    """Reconstruct a nested Flax param dict from a flat array and metadata."""
    leaves = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        leaves.append(flat[offset: offset + size].reshape(shape))
        offset += size
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ---------------------------------------------------------------------------
# OpenES update (one generation)
# ---------------------------------------------------------------------------

def _es_generation(flat_params, state, model, train_batch, z_rng,
                   treedef, shapes, sigma, population, apply_batchnorm):
    """
    Run one generation of OpenES and return (new_flat_params, new_opt_state, mean_fitness).

    All JAX operations; safe to call inside jax.jit if shapes are static.
    """
    n = population
    # Sample perturbations using antithetic sampling (pairs +ε, -ε)
    half = n // 2
    rng_keys = jax.random.split(z_rng, half)
    eps_pos = jax.vmap(lambda k: jax.random.normal(k, flat_params.shape))(rng_keys)
    eps = jnp.concatenate([eps_pos, -eps_pos], axis=0)  # (n, n_params)

    # Evaluate fitness for each perturbation
    def fitness_one(eps_i):
        p_i = _unflatten_params(flat_params + sigma * eps_i, treedef, shapes)
        if apply_batchnorm:
            decoded, *_ = state.apply_fn(
                {"params": p_i, "batch_stats": state.batch_stats},
                train_batch, z_rng, train=False,
                rngs={"dropout": z_rng},
            )
        else:
            decoded, *_ = state.apply_fn(
                {"params": p_i},
                train_batch, z_rng, train=False,
                rngs={"dropout": z_rng},
            )
        # Per-frame RMSD (nm); fitness = negative RMSD (higher = better)
        diff = (decoded - train_batch).reshape(train_batch.shape[0], -1, 3)
        rmsd = jnp.sqrt(jnp.mean(jnp.sum(diff ** 2, axis=-1), axis=-1))
        return -jnp.mean(rmsd)

    fitnesses = jax.vmap(fitness_one)(eps)  # (n,)

    # Rank-normalise to [-0.5, 0.5] for robustness
    ranks = jnp.argsort(jnp.argsort(fitnesses))
    normalised = (ranks / (n - 1)) - 0.5

    # Gradient estimate
    grad_estimate = -(1.0 / (n * sigma)) * jnp.sum(
        normalised[:, None] * eps, axis=0
    )

    return grad_estimate, jnp.mean(fitnesses)


class EvolutionaryTrainer(Experiment):
    """
    OpenES-based trainer: replaces gradient descent with evolution strategies.

    Inherits all data-loading, checkpointing, and NetCDF infrastructure from
    Experiment.  The training loop replaces train_on_batches with an ES
    generation step.

    Parameters are the same as Experiment.  Additional JSON keys:
        es_population, es_sigma, es_lr  (see module docstring).
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False,
                 model_cls=BatchNorm_VAE):
        super().__init__(
            json_fn, make_dirs=make_dirs,
            from_json_params=from_json_params, model_cls=model_cls,
        )
        self._es_pop = self.json_params.get("es_population", 50)
        self._es_sigma = self.json_params.get("es_sigma", 0.05)
        es_lr = self.json_params.get("es_lr", 0.01)
        self._es_opt = optax.adam(es_lr)

        flat, treedef, shapes = _flatten_params(self.state.params)
        self._treedef = treedef
        self._shapes = shapes
        self._es_opt_state = self._es_opt.init(flat)

    def _es_step(self, flat_params, batch):
        """One ES generation: return updated flat_params, opt_state, mean_fitness."""
        rng = jax.random.PRNGKey(self.epoch)
        grad_est, mean_fit = _es_generation(
            flat_params, self.state, self.model, batch, rng,
            self._treedef, self._shapes,
            self._es_sigma, self._es_pop, self.is_batchnorm,
        )
        updates, new_opt_state = self._es_opt.update(grad_est, self._es_opt_state)
        new_flat = optax.apply_updates(flat_params, updates)
        return new_flat, new_opt_state, float(mean_fit)

    def train_n_epochs(self, n_epochs, verbose=True):
        """Run ES for n_epochs; each epoch uses one random training batch."""
        flat, _, _ = _flatten_params(self.state.params)

        while self.epoch < n_epochs:
            epoch_start = datetime.now()
            batch = next(iter(self.train_batches))
            flat, self._es_opt_state, fit = self._es_step(flat, batch)

            # Write back to state
            new_params = _unflatten_params(flat, self._treedef, self._shapes)
            self.state = self.state.replace(params=new_params)

            # Log
            rmsd_nm = -fit
            if verbose is True or (isinstance(verbose, int) and self.epoch % verbose == 0):
                printf(
                    f"Epoch {self.epoch:6d}  ES mean fitness={fit:+.4f}"
                    f"  RMSD≈{rmsd_nm*10:.3f} Å  [{datetime.now()-epoch_start}]"
                )

            # NetCDF: store RMSD
            self.rootgrp["Train"].variables["RMSD_Loss_Term"][self.epoch, :] = rmsd_nm
            self.rootgrp["Test"].variables["RMSD_Loss_Term"][self.epoch, :] = rmsd_nm

            # Checkpoint
            save_args = orbax_utils.save_args_from_target(self.state)
            self.checkpoint_manager.save(
                self.epoch, self.state,
                save_kwargs={"save_args": save_args},
            )
            self.epoch += 1

        return self.epoch

    def MAIN_train(self, n_epochs=1000, cutoff_epoch=None, verbose=True):
        """Run ES for up to cutoff_epoch (or json_params['max_epoch']) generations."""
        if cutoff_epoch is None:
            cutoff_epoch = self.json_params["max_epoch"]
        return self.train_n_epochs(cutoff_epoch, verbose=verbose)


class NEATTrainer(EvolutionaryTrainer):
    """
    Full NEAT trainer: OpenES weight evolution + topology growth on plateau.

    Extends EvolutionaryTrainer by monitoring fitness across a sliding window.
    When the best fitness fails to improve by more than plateau_threshold for
    plateau_window consecutive generations, grow() is called on the model to
    add a hidden layer, and evolution continues with the larger network.

    Requires model_cls=NEATAutoencoder (default when using scripts/train_neat.py).

    JSON extras:
        neat_plateau_window : int   (default 200)
        neat_plateau_thr    : float (default 0.005)
    """

    def __init__(self, json_fn, make_dirs=True, from_json_params=False,
                 model_cls=None):
        from ..models.neat_vae import NEATAutoencoder
        if model_cls is None:
            model_cls = NEATAutoencoder
        super().__init__(
            json_fn, make_dirs=make_dirs,
            from_json_params=from_json_params, model_cls=model_cls,
        )
        self._plateau_window = self.json_params.get("neat_plateau_window", 200)
        self._plateau_thr = self.json_params.get("neat_plateau_thr", 0.005)
        self._best_fitness_history = []
        self._grow_count = 0

    def _check_and_grow(self):
        """Grow topology if fitness has plateaued; return True if grown."""
        if len(self._best_fitness_history) < self._plateau_window:
            return False

        window = self._best_fitness_history[-self._plateau_window:]
        best_recent = max(window)
        best_old = max(self._best_fitness_history[:-self._plateau_window]
                       if len(self._best_fitness_history) > self._plateau_window else window[:1])

        improvement = (best_recent - best_old) / max(abs(best_old), 1e-9)
        if improvement < self._plateau_thr:
            self._grow()
            return True
        return False

    def _grow(self):
        """Add one hidden layer to the model and reinitialise ES state."""
        from ..models.neat_vae import grow as neat_grow

        printf(
            f"[NEATTrainer] Plateau detected at epoch {self.epoch} "
            f"(growth #{self._grow_count + 1}).  Adding hidden layer."
        )
        printf(f"  Old topology: {self.model.hidden_layers}")

        new_model, new_params, new_dropout = neat_grow(
            self.model, self.state.params
        )
        self.model = new_model

        # Rebuild state with new params (same TrainState class)
        self.state = self.state.replace(params=new_params)

        # Re-derive flat-param metadata for ES
        flat, treedef, shapes = _flatten_params(new_params)
        self._treedef = treedef
        self._shapes = shapes
        es_lr = self.json_params.get("es_lr", 0.01)
        self._es_opt = optax.adam(es_lr)
        self._es_opt_state = self._es_opt.init(flat)

        self._grow_count += 1
        self._best_fitness_history = []  # reset plateau window

        printf(f"  New topology: {self.model.hidden_layers}")

    def train_n_epochs(self, n_epochs, verbose=True):
        """
        Run NEAT evolution for n_epochs, growing topology on plateau.

        After each generation, checks whether the fitness sliding window
        has stalled and calls _grow() if so.  Verbose output includes the
        current layer count so topology growth is visible in logs.
        """
        flat, _, _ = _flatten_params(self.state.params)

        while self.epoch < n_epochs:
            epoch_start = datetime.now()
            batch = next(iter(self.train_batches))
            flat, self._es_opt_state, fit = self._es_step(flat, batch)

            new_params = _unflatten_params(flat, self._treedef, self._shapes)
            self.state = self.state.replace(params=new_params)

            self._best_fitness_history.append(fit)

            if verbose is True or (isinstance(verbose, int) and self.epoch % verbose == 0):
                n_layers = len(self.model.hidden_layers)
                printf(
                    f"Epoch {self.epoch:6d}  fit={fit:+.4f}"
                    f"  RMSD≈{-fit*10:.3f} Å"
                    f"  layers={n_layers}"
                    f"  [{datetime.now()-epoch_start}]"
                )

            self.rootgrp["Train"].variables["RMSD_Loss_Term"][self.epoch, :] = -fit
            self.rootgrp["Test"].variables["RMSD_Loss_Term"][self.epoch, :] = -fit

            save_args = orbax_utils.save_args_from_target(self.state)
            self.checkpoint_manager.save(
                self.epoch, self.state,
                save_kwargs={"save_args": save_args},
            )

            grew = self._check_and_grow()
            if grew:
                flat, _, _ = _flatten_params(self.state.params)

            self.epoch += 1

        return self.epoch
