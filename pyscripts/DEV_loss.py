import jax
import jax.numpy as jnp
try:
    from DEV_main import gas_fun
except:
    from .DEV_main import gas_fun


@jax.vmap
def atom_rmsd(a, b, pot_coef=None):
    """
    Atom RMSD of vectorized frames a and b
    Due to vmapping does not work on individual frames, but only collections of frames
    """
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    return jnp.sqrt(jnp.mean(jnp.sum((b - a)**2, axis=1)))

@jax.jit
def scaled_pot_enr_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return ((gas_fun(a) - gas_fun(b))/gas_fun(a))**2 #Unitless quantity

@jax.jit
def weighted_summation_loss(a, b, potential_coefficient): # LET A BE BATCH AND B BE RECON
    # Make this the square root of the mean of the sum of squares of elements
    return atom_rmsd(a,b) + potential_coefficient*scaled_pot_enr_diff(a, b)

@jax.jit
def rmsd_log_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True, mutable=['batch_stats'])
        loss = jnp.log(jnp.sqrt(jnp.sum(atom_rmsd(batch_x, logits[0])**2)))
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state

@jax.jit
def potential_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True, mutable=['batch_stats'])
        loss = scaled_pot_enr_diff(batch_x, logits[0]).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state

@jax.jit
def weighted_summation_step(state, batch_x, z_rng, potential_coefficient, weights=(1,1)):
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True, mutable=['batch_stats'])
        loss = weighted_summation_loss(batch_x, logits[0], potential_coefficient).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state