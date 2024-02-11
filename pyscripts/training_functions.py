import jax, os
import jax_amber3 as jaa
import jax.numpy as jnp

fname_prmtop = os.path.join(os.getcwd(), 'Simulation/ala_deca_peptide.prmtop')
ener_fun, tors_fun = jaa.get_amber_functions(fname_prmtop)

#Maths
@jax.vmap
def atom_rmsd(a, b, **kwargs): # for arrays of (n_conf, n_atom*3)
    """VMAPED iterates over the n_configurational array, read a and b below as iterations over a,b which are actually provided
    Ex. if a has shape (10, 300), then in code below the expected shape of a is (300)"""
    mn = a.shape[-1]//3
    x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
    return jnp.sqrt(jnp.mean((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))

@jax.jit
def cos_torsional_dist(a, b, **kwargs):
    """For molecular sets, a, b
        returns 1/2 of the square of the L2 distance between two angles which are cast onto the unit circle.
        L2d(Theta1, Theta2) = sqrt(2*(1-cos(a-b)))"""
    a, b = tors_fun(a), tors_fun(b)
    return 1 - jnp.cos(b-a) #invariant to order of opt, as cosine is an even function

@jax.jit
def atom_rmtd(a, b, **kwargs):
    """Extension of cos_torsional_dist to calculat the RMTD (Root Mean Torsional Deviation)"""
    return jnp.sqrt(jnp.mean(cos_torsional_dist(a, b), axis=1)) #axis invoked here and not for rmSd because this function is not vmapped

@jax.jit
def structural_loss(a, b, torsional_coefficient, **kwargs): # LET A BE BATCH AND B BE RECON
    """RMSD plus RMTD, RMTD can be scaled by torsional_coefficient"""
    return jnp.sqrt(jnp.sum(atom_rmsd(a,b)**2)) + jnp.sqrt(jnp.sum(torsional_coefficient*atom_rmtd(a,b)**2))

@jax.jit
def scaled_pot_enr_diff(a, b, **kwargs): # WITH A AS BATCH AND B AS RECON
    """Square of the percent deviation of energy of b compared to energy of a"""
    return ((ener_fun(b) - ener_fun(a))/ener_fun(a))**2 #Unitless quantity

@jax.jit
def summation_loss(a, b, torsional_coefficient, potential_coefficient, **kwargs): # LET A BE BATCH AND B BE RECON
    """A loss function incorporating RMSD, RMTD, and potential deviation"""
    return structural_loss(a, b, torsional_coefficient=torsional_coefficient) + potential_coefficient * scaled_pot_enr_diff(a, b).mean()

#Repulsion
def rmsd_one_off(a, b):
    """A distance function where A is one particle, and B is a set of particles (A = (3*natom), B = (n_conf>1, 3*natom))"""
    rmsd = jnp.sqrt(jnp.mean((b - a)**2 + (b - a)**2 + (b - a)**2))
    return rmsd

def rmtd_one_off(a, b):
    """Where A, and B are torsional particles, analogously defined as in rmsd one off"""
    rmtd = jnp.sqrt(jnp.mean(1 - jnp.cos(b-a)))
    return rmtd

rmsd_one_off = jax.vmap(rmsd_one_off, in_axes=(None, 0))
rmtd_one_off = jax.vmap(rmtd_one_off, in_axes=(None, 0))

@jax.jit
def rmsd_distance_matrix(a, b):
    rmsd = jnp.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        #print(i)
        rmsd = rmsd.at[i,:].set(rmsd_one_off(a[i],b))
    return rmsd

@jax.jit
def rmtd_distance_matrix(a, b):
    rmtd = jnp.zeros((a.shape[0], b.shape[0]))
    for i in range(a.shape[0]):
        #print(i)
        rmtd = rmtd.at[i,:].set(rmtd_one_off(a[i],b))
    return rmtd

@jax.jit
def structural_distance_matrix(a, b):
    tors_a, tors_b = torsion_fun(a), torsion_fun(b)
    return rmsd_distance_matrix(a, b) + rmtd_distance_matrix(tors_a, tors_b)

#General Step functions (in development)
def establish_step_function(state, batch, loss_func, method='mean'):
    @jax.jit
    def custom_step(state, batch, loss_fn, **kwargs):
        def loss(params, apply_fn):
            decoded, latents = apply_fn({'params':params}, batch, z_rng)
            return loss_fn(batch, decoded, **kwargs)
        grads = jax.grad(loss)(state.params, state.apply_fn)
        return state.apply_gradients(grads=grads)
    return custom_step


#Steps with rng
@jax.jit
def rmsd_rng_step(state, batch, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded, _ = apply_fn({'params':params}, batch, z_rng)
        return jnp.sqrt(jnp.sum(atom_rmsd(batch, decoded)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)

@jax.jit
def rmtd_rng_step(state, batch, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded, _ = apply_fn({'params':params}, batch, z_rng)
        return jnp.sqrt(jnp.sum(atom_rmtd(batch, decoded)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)

@jax.jit
def potential_rng_step(state, batch, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded, _ = apply_fn({'params':params}, batch, z_rng)
        return scaled_pot_enr_diff(batch, decoded).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)

@jax.jit
def structural_rng_step(state, batch_x, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return structural_loss(batch_x, decoded_x, **kwargs)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)

@jax.jit
def summation_rng_step(state, batch_x, z_rng, **kwargs):
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return summation_loss(batch_x, decoded_x, **kwargs)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)