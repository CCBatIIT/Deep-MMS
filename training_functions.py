import jax, os
import jax.numpy as jnp
import jax_amber3 as ja

fname_prmtop = os.path.join(os.getcwd(), 'Simulation/1crn_H.prmtop')
ener_fun, tors_fun = ja.get_amber_gas_energy_function(fname_prmtop)

#Mathematical Functions
@jax.vmap
def atom_rmsd(a, b, **kwargs): # for arrays of (n_conf, n_atom*3)
    mn = a.shape[-1]//3
    x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
    return jnp.mean(jnp.sqrt((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))

@jax.jit
def torsional_diff(a, b):
    return jnp.min(jnp.stack((tors_fun(a) - tors_fun(b), 2*jnp.pi - tors_fun(a) + tors_fun(b))), axis=0)

@jax.jit
def scaled_pot_enr_diff(a, b, **kwargs): # WITH A AS BATCH AND B AS RECON
    return ((ener_fun(a) - ener_fun(b))/ener_fun(a))**2 #Unitless quantity

@jax.jit
def summation_loss(a, b, torsional_coefficient, potential_coefficient, **kwargs): # LET A BE BATCH AND B BE RECON
    return jnp.sqrt(jnp.sum(atom_rmsd(a,b)**2)) + \
           torsional_coefficient * torsional_diff(a, b).mean() + \
           potential_coefficient * scaled_pot_enr_diff(a, b).mean()

#Steps with no rng
@jax.jit
def rmsd_step(state, batch_x, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return jnp.sqrt(jnp.sum(atom_rmsd(batch_x, decoded_x)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients (grads=grads)

@jax.jit
def torsional_step(state, batch_x, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return torsionl_diff(batch_x, decoded_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients (grads=grads)

@jax.jit
def potential_step(state, batch_x, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return scaled_pot_enr_diff(batch_x, decoded_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def summation_step(state, batch_x, potential_coefficient, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return summation_loss(batch_x, decoded_x, potential_coefficient)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

#Steps with rng
@jax.jit
def rmsd_rng_step(state, batch_x, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return jnp.sqrt(jnp.sum(atom_rmsd(batch_x, decoded_x)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients (grads=grads)

@jax.jit
def torsional_rng_step(state, batch_x, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return torsional_diff(batch_x, decoded_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def potential_rng_step(state, batch_x, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return scaled_pot_enr_diff(batch_x, decoded_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def summation_rng_step(state, batch_x, z_rng, potential_coefficient, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return summation_loss(batch_x, decoded_x, potential_coefficient)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)
