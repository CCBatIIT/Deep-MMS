"""
Optimizer, learning-rate schedule, and Flax TrainState construction for Deep-MMS.

Provides create_warmup_cosine_schedule for a linear-warmup / cosine-decay LR
policy, define_step to build JIT-compiled step and evaluate closures that are
aware of whether batch normalisation is active, and make_model_and_state to
wire everything together from an Experiment configuration object.
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state, orbax_utils
from typing import Any

from ..models.vae import BatchNorm_VAE


def create_warmup_cosine_schedule(base_lr, warmup_steps, total_steps, final_lr=1e-5):
    """
    Build a learning-rate schedule with linear warmup followed by cosine decay.

    Parameters
    ----------
    base_lr : float
        Peak learning rate reached at the end of warmup.
    warmup_steps : int
        Number of gradient steps over which LR rises linearly to base_lr.
    total_steps : int
        Total gradient steps; cosine decay fills the remaining steps after warmup.
    final_lr : float
        Minimum learning rate at the end of decay (also the LR at step 0).

    Returns
    -------
    callable
        Optax-compatible schedule function mapping step → learning rate.
    """
    warmup_fn = optax.linear_schedule(
        init_value=final_lr, end_value=base_lr, transition_steps=warmup_steps
    )
    decay_steps = total_steps - warmup_steps
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=decay_steps, alpha=final_lr
    )

    def schedule(step):
        return jnp.where(
            step < warmup_steps, warmup_fn(step), cosine_fn(step - warmup_steps)
        )

    return schedule


def define_step(is_batchnorm, atom_rmsd):
    """
    Return JIT-compiled (step, evaluate) closures appropriate for the batchnorm setting.

    When is_batchnorm is True the step function propagates batch_stats mutations
    and the TrainState is expected to carry a batch_stats field.  When False,
    the standard params-only state is used.

    Parameters
    ----------
    is_batchnorm : bool
        Whether the model uses batch normalisation.
    atom_rmsd : callable
        Batched weighted-RMSD function produced by give_weighted_rmsd_func.

    Returns
    -------
    step : callable
        Training step: (state, batch_x, z_rng, dropout_key) → (state, loss).
    evaluate : callable
        Evaluation step: (state, batch_x, z_rng, dropout_key) → rmsd_term.
    """
    if is_batchnorm:
        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            """One gradient step with BatchNorm mutable batch_stats."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    batch_x, z_rng, train=True,
                    rngs={"dropout": dropout_train_key},
                    mutable=["batch_stats"],
                )
                loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0]) ** 2)))
                return loss, (logits, updates)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, updates)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates["batch_stats"])
            return state, loss

        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            """Evaluate RMSD loss in inference mode (batch_stats frozen)."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits, updates = state.apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    batch_x, z_rng, train=False,
                    rngs={"dropout": dropout_train_key},
                    mutable=["batch_stats"],
                )
                rmsd_term = jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0]) ** 2))
                return (rmsd_term), (logits, updates)

            return loss_fn(state.params)[0]

    else:
        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            """One gradient step without BatchNorm."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=True,
                    rngs={"dropout": dropout_train_key},
                )
                loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0]) ** 2)))
                return loss, (logits, None)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, updates)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            """Evaluate RMSD loss in inference mode (no BatchNorm)."""
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_fn(params):
                logits = state.apply_fn(
                    {"params": params},
                    batch_x, z_rng, train=False,
                    rngs={"dropout": dropout_train_key},
                )
                rmsd_term = jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0]) ** 2))
                return (rmsd_term), (logits, None)

            return loss_fn(state.params)[0]

    return step, evaluate


def make_model_and_state(experiment, dropout_rates, coord_set, learning_rate, atom_rmsd_loss, model_cls=BatchNorm_VAE):
    """
    Instantiate the model, initialise parameters, build the optimizer, and
    return the model, TrainState, and compiled step/evaluate functions.

    Parameters
    ----------
    experiment : Experiment
        Configuration object exposing n_latents, is_batchnorm, batch_size,
        train_data, and json_params.
    dropout_rates : list of float
        Per-layer dropout rates; determines the number of hidden layers.
    coord_set : jnp.ndarray, shape (n_frames, n_features)
        Full coordinate set used to determine input_size.
    learning_rate : float
        Peak learning rate for the warmup-cosine schedule.
    atom_rmsd_loss : callable
        Weighted RMSD function (from give_weighted_rmsd_func).
    model_cls : type, optional
        Model class to instantiate (default: BatchNorm_VAE).

    Returns
    -------
    model : model_cls instance
    state : TrainState
    step : callable
    evaluate : callable
    """
    num_samples, input_size = coord_set.shape
    hidden_layers = model_cls.hidden_layers_from_config(
        input_size, experiment.n_latents, dropout_rates, experiment.json_params
    )

    model = model_cls(
        input_size=input_size,
        latents=experiment.n_latents,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        is_batchnorm=experiment.is_batchnorm,
    )

    rng_init = jax.random.PRNGKey(experiment.n_latents)
    main_key, params_key, dropout_key = jax.random.split(key=rng_init, num=3)
    variables = model.init(params_key, coord_set, rng_init, train=False)
    params = variables["params"]

    n_updates_per_epoch = experiment.train_data.shape[0] // experiment.batch_size
    lr = create_warmup_cosine_schedule(
        base_lr=learning_rate,
        warmup_steps=n_updates_per_epoch * 1000,
        total_steps=n_updates_per_epoch * experiment.json_params["max_epoch"],
        final_lr=learning_rate / 100,
    )

    if experiment.is_batchnorm:
        batch_stats = variables["batch_stats"]

        class TrainState(train_state.TrainState):
            batch_stats: Any
            key: jax.Array

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            batch_stats=batch_stats,
            key=dropout_key,
            tx=optax.adam(lr),
        )
    else:
        class TrainState(train_state.TrainState):
            key: jax.Array

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            key=dropout_key,
            tx=optax.adam(lr),
        )

    step_func, evaluate_func = define_step(experiment.is_batchnorm, atom_rmsd_loss)
    return model, state, step_func, evaluate_func
