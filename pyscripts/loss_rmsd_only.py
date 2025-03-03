import jax
import jax.numpy as jnp

@jax.vmap
def atom_rmsd(a, b, pot_coef=None):
    """
    Atom RMSD of vectorized frames a and b
    Due to vmapping does not work on individual frames, but only collections of frames
    """
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    return jnp.sqrt(jnp.mean(jnp.sum((b - a)**2, axis=1)))

KL_loss = lambda mus, log_vars: 0.5 * jnp.sum(mus**2 + jnp.exp(log_vars) - log_vars - 1)

def step(state, batch_x, z_rng, dropout_key, compute_fun):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True,
                                         rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
        loss = jnp.sqrt(jnp.mean(compute_fun(batch_x, logits[0])**2)) #RMSE of loss function
        loss += KL_loss(logits[1], logits[2])
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state, loss

step = jax.jit(step, static_argnums=[4])

@jax.jit
def rmsd_log_step(state, batch_x, z_rng, potential_coefficient, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True,
                                         rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
        loss = jnp.log(jnp.sqrt(jnp.sum(atom_rmsd(batch_x, logits[0])**2)))
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state