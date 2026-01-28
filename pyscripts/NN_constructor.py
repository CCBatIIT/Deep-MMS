import flax, jax, optax, sys, os, json, pickle, time, glob
import flax.linen as nn
import jax.numpy as jnp
from flax.training import train_state, orbax_utils
from typing import Any
from .heavy_atom_rmsd import printf, BatchNorm_VAE



        
# ####################################################################
# #    LOSS SECTION
# ####################################################################
# @jax.vmap
# def atom_rmsd(a, b):
#     """
#     Atom RMSD of vectorized frames a and b
#     Due to vmapping does not work on individual frames, but only collections of frames
#     """
#     a, b = a.reshape(-1, 3), b.reshape(-1, 3)
#     return jnp.sqrt(jnp.mean(jnp.sum((b - a)**2, axis=1)))


def define_step(NN_exp, atom_rmsd):
    if NN_exp.is_batchnorm:
        #Define Step
        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
            def loss_fn(params):
                #Logits is the output of calling the NN (Decoded, Latent_Means, Latent_Vars)
                logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                 batch_x, z_rng, train=True,
                                                 rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
                #Loss term representing the Root Mean Square reconstruction error
                loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0])**2)))
                #Loss term representing the KL Divergence between latent space and standard normals
                #loss += KL_loss(logits[1], logits[2])
                #Loss term representing the MI between latent Dimensions
                #loss += MI_loss(logits[1])
                return loss, (logits, updates)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, updates)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])
            return state, loss
        #Define Evaluation
        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
            def loss_fn(params):
                #Logits is the output of calling the NN (Decoded, Latent_Means, Latent_Vars)
                logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                 batch_x, z_rng, train=False,
                                                 rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
                #Loss term representing the Root Mean Square reconstruction error
                rmsd_term = jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0])**2))
                #Loss term representing the KL Divergence between latent space and standard normals
                #KL_term = KL_loss(logits[1], logits[2])
                #Loss term representing the MI between latent Dimensions
                #MI_term = MI_loss(logits[1])
                return (rmsd_term), (logits, updates)
            return loss_fn(state.params)[0]
    
    elif not NN_exp.is_batchnorm:
        #Define Step
        @jax.jit
        def step(state, batch_x, z_rng, dropout_key):
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
            def loss_fn(params):
                #Logits is the output of calling the NN (Decoded, Latent_Means, Latent_Vars)
                logits = state.apply_fn({'params': params},
                                        batch_x, z_rng, train=True,
                                        rngs={'dropout': dropout_train_key})
                #Loss term representing the Root Mean Square reconstruction error
                loss = jnp.log(jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0])**2)))
                #Loss term representing the KL Divergence between latent space and standard normals
                #loss += KL_loss(logits[1], logits[2])
                #Loss term representing the MI between latent Dimensions
                #loss += MI_loss(logits[1])
                return loss, (logits, None)
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (logits, updates)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss
        #Define Evaluation
        @jax.jit
        def evaluate(state, batch_x, z_rng, dropout_key):
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
            def loss_fn(params):
                #Logits is the output of calling the NN (Decoded, Latent_Means, Latent_Vars)
                logits = state.apply_fn({'params': params},
                                                 batch_x, z_rng, train=False,
                                                 rngs={'dropout': dropout_train_key})
                #Loss term representing the Root Mean Square reconstruction error
                rmsd_term = jnp.sqrt(jnp.mean(atom_rmsd(batch_x, logits[0])**2))
                #Loss term representing the KL Divergence between latent space and standard normals
                #KL_term = KL_loss(logits[1], logits[2])
                #Loss term representing the MI between latent Dimensions
                #MI_term = MI_loss(logits[1])
                return (rmsd_term), (logits, None)
            return loss_fn(state.params)[0]
        
    return step, evaluate


def create_warmup_cosine_schedule(base_lr, warmup_steps, total_steps, final_lr=1e-5):
    # During warmup: linearly go from 0 -> base_lr
    warmup_fn = optax.linear_schedule(init_value=final_lr, end_value=base_lr, transition_steps=warmup_steps)
    
    # After warmup: cosine decay from base_lr -> 0
    decay_steps = total_steps - warmup_steps
    cosine_fn = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=decay_steps, alpha=final_lr)

    def schedule(step):
        # Use warmup for the first warmup_steps
        return jnp.where(step < warmup_steps,
                         warmup_fn(step),
                         cosine_fn(step - warmup_steps))
    return schedule


def make_model_and_state(NN_exp, dropout_rates, coord_set, learning_rate, atom_rmsd_loss):

    #Initialize Model
    num_samples, input_size = coord_set.shape
    n_hidden = len(dropout_rates) #Num Hidden Layers determined by quantity of dropout rates
    #SIZE OF HIDDEN LAYERS
    #geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (jnp.exp(float(i) / float(n_vals-1)) - 1.0) / (jnp.e - 1.0) for i in range(n_vals)]
    #hidden_layers = [int(val) if int(val) >= int(NN_exp.n_latents) else int(NN_exp.n_latents) for val in geometric_distribution(input_size, NN_exp.n_latents, n_hidden)]
    # OG OG OG (X012)
    hidden_layers = [input_size]*len(dropout_rates)
    
    model = BatchNorm_VAE(input_size=input_size,
                          latents=NN_exp.n_latents,
                          hidden_layers=hidden_layers,
                          dropout_rates=dropout_rates,
                          is_batchnorm=NN_exp.is_batchnorm)
    
    rng_init = jax.random.PRNGKey(NN_exp.n_latents)
    main_key, params_key, dropout_key = jax.random.split(key=rng_init, num=3)
    variables = model.init(params_key, coord_set, rng_init, train=False)
    params = variables['params']
    n_updates_per_epoch = NN_exp.train_data.shape[0]//NN_exp.batch_size
    
    lr = create_warmup_cosine_schedule(base_lr=learning_rate,
                                       warmup_steps = n_updates_per_epoch * 1000,
                                       total_steps = n_updates_per_epoch * NN_exp.json_params['max_epoch'],
                                       final_lr=learning_rate/100)
    
    if NN_exp.is_batchnorm:
        batch_stats = variables['batch_stats']
        class TrainState(train_state.TrainState):
            batch_stats: Any
            key: jax.Array

        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  batch_stats=batch_stats,
                                  key=dropout_key,
                                  tx=optax.adam(lr))
    else:
        class TrainState(train_state.TrainState):
            key: jax.Array
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  key=dropout_key,
                                  tx=optax.adam(lr))
    step_func, evaluate_func = define_step(NN_exp, atom_rmsd_loss)
    
    return model, state, step_func, evaluate_func
