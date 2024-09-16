import jax
import jax.numpy as jnp
try:
    from main import gas_fun
except:
    from .main import gas_fun


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
def rmsd_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return jnp.sqrt(jnp.sum(atom_rmsd(batch_x, recon_x)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def rmsd_log_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return jnp.log(jnp.sqrt(jnp.sum(atom_rmsd(batch_x, recon_x)**2)))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def potential_step(state, batch_x, z_rng, potential_coefficient):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return scaled_pot_enr_diff(batch_x, recon_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)    

@jax.jit
def summation_step(state, batch_x, z_rng, potential_coefficient, weights=(1,1)):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return summation_loss(batch_x, recon_x, potential_coefficient).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def weighted_summation_step(state, batch_x, z_rng, potential_coefficient, weights=(1,1)):
    def loss_fn(params, apply_fn):
        recon_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return weighted_summation_loss(batch_x, recon_x, potential_coefficient).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)