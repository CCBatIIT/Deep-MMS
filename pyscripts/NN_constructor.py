import flax, jax, optax, sys, os, json, pickle, time, glob
import flax.linen as nn
import jax.numpy as jnp
from flax.training import train_state, orbax_utils
from typing import Any
from .heavy_atom_rmsd import printf, BatchNorm_VAE

# ####################################################################
# #    NEURAL NETWORK SECTION
# ####################################################################


# class BatchNorm_VAE(nn.Module):
#     input_size: int
#     hidden_layers: tuple
#     latents: int
#     dropout_rates: list
#     is_batchnorm: bool
        
#     def setup(self):
#         encoder = BVEncoder(list(hidden_layers), latents, dropout_rates, is_batchnorm)
#         decoder = BVDecoder(list(hidden_layers), input_size, dropout_rates, is_batchnorm)

#     def __call__(self, x, z_rng, train:bool):
#         z_mean, z_logvar = encoder(x, train=train)
#         z = reparameterize(z_rng, z_mean, z_logvar)
#         return decoder(z, train=train), z_mean, z_logvar
    
#     def construct(self, z_mean, z_logvar, z_rng, train=False):
#         z = reparameterize(z_rng, z_mean, z_logvar)
#         return decoder(z, train=train)
    
#     def encode(self, x, z_rng, train=False):
#         return encoder(x, train=train)
    
#     def decode(self, z, z_rng, train=False):
#         return decoder(z, train=train)

# #BatchNorm Variational AutoEncoder
# class BVEncoder(nn.Module):
#     d_hidden: list
#     latents: int
#     dropout_rates: list
#     is_batchnorm: bool
    
#     @nn.compact
#     def __call__(self, x, train: bool):
#         for i in range(len(d_hidden)):
#             x = nn.Dense(d_hidden[i])(x)
#             x = nn.leaky_relu(x, negative_slope=0.2)
#             if is_batchnorm:
#                 x = nn.BatchNorm(use_running_average=not train)(x)
#             x = nn.Dropout(rate=dropout_rates[i])(x, deterministic=not train)
#         mean_x = nn.Dense(latents, name='fc5_mean')(x)
#         logvar_x = nn.Dense(latents, name='fc5_logvar')(x)
#         return mean_x, logvar_x 

# #BatchNorm "Variational" Decoder (all variation is in encoding)
# class BVDecoder(nn.Module):
#     d_hidden: list
#     out_dim: int
#     dropout_rates: list
#     is_batchnorm: bool

#     @nn.compact
#     def __call__(self, z, train: bool):
#         for i in range(len(d_hidden))[::-1]:
#             z = nn.Dense(d_hidden[i])(z)
#             z = nn.leaky_relu(z, negative_slope=0.2)
#             if is_batchnorm:
#                 z = nn.BatchNorm(use_running_average=not train)(z)
#             z = nn.Dropout(rate=dropout_rates[i])(z, deterministic=not train)
#         z = nn.Dense(out_dim, name='f5')(z)
#         return z

        
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



def make_model_and_state(NN_exp, dropout_rates, coord_set, learning_rate, atom_rmsd_loss):

    #Initialize Model
    num_samples, input_size = coord_set.shape
    n_hidden = len(dropout_rates) #Num Hidden Layers determined by quantity of dropout rates
    hidden_layers = [int(elem) for elem in jnp.round(jnp.logspace(jnp.log10(input_size), jnp.log10(NN_exp.n_latents), n_hidden+1))][:n_hidden]
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
    
    if NN_exp.is_batchnorm:
        batch_stats = variables['batch_stats']
        class TrainState(train_state.TrainState):
            batch_stats: Any
            key: jax.Array

        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  batch_stats=batch_stats,
                                  key=dropout_key,
                                  tx=optax.adam(learning_rate=learning_rate))
    else:
        class TrainState(train_state.TrainState):
            key: jax.Array
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  key=dropout_key,
                                  tx=optax.adam(learning_rate=learning_rate))
    step_func, evaluate_func = define_step(NN_exp, atom_rmsd_loss)
    
    return model, state, step_func, evaluate_func
