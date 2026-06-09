import jax
import jax.numpy as jnp
try:
    from main import bonds_fun, angles_fun, torsions_fun, lj14_fun, coul14_fun, lj_fun, coul_fun
except:
    from .main import bonds_fun, angles_fun, torsions_fun, lj14_fun, coul14_fun, lj_fun, coul_fun


@jax.vmap
def atom_rmsd(a, b, pot_coef=None):
    """
    Atom RMSD of vectorized frames a and b
    Due to vmapping does not work on individual frames, but only collections of frames
    """
    a, b = a.reshape(-1, 3), b.reshape(-1, 3)
    return jnp.sqrt(jnp.mean(jnp.sum((b - a)**2, axis=1)))

@jax.jit
def bond_ener_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(bonds_fun(a) - bonds_fun(b))

@jax.jit
def angle_ener_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(angles_fun(a) - angles_fun(b))

@jax.jit
def torsion_ener_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(torsions_fun(a) - torsions_fun(b))

@jax.jit
def LJ_14_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(lj14_fun(a) - lj14_fun(b))

@jax.jit
def Coul_14_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(coul14_fun(a) - coul14_fun(b))

@jax.jit
def LJ_NB_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(lj_fun(a) - lj_fun(b))

@jax.jit
def Coul_NB_diff(a, b, pot_coef=None):
    """
    Potential Energy Difference between Ensembles a and b
    jitted here because it is already vmapped within gas_fun
    Requires the global definition of gas_fun
    """
    return jnp.abs(coul_fun(a) - coul_fun(b))


# @jax.jit
# def scaled_pot_enr_diff(a, b, pot_coef=None):
#     """
#     Potential Energy Difference between Ensembles a and b
#     jitted here because it is already vmapped within gas_fun
#     Requires the global definition of gas_fun
#     """
#     return jnp.abs(gas_fun(a) - gas_fun(b))

@jax.jit
def sum1_loss(a, b): # LET A BE BATCH AND B BE RECON
    return atom_rmsd(a,b) + bond_ener_diff(a, b)

@jax.jit
def sum2_loss(a, b): # LET A BE BATCH AND B BE RECON
    return sum1_loss(a, b) + angle_ener_diff(a, b)

@jax.jit
def sum3_loss(a, b): # LET A BE BATCH AND B BE RECON
    return sum2_loss(a, b) + torsion_ener_diff(a, b)

@jax.jit
def sum4_loss(a, b): # LET A BE BATCH AND B BE RECON
    return sum3_loss(a, b) + LJ_14_diff(a, b)

@jax.jit
def sum5_loss(a, b): # LET A BE BATCH AND B BE RECON
    return sum4_loss(a, b) + Coul_14_diff(a, b)

@jax.jit
def sum6_loss(a, b): # LET A BE BATCH AND B BE RECON
    return sum5_loss(a, b) + LJ_NB_diff(a, b)

@jax.jit
def sum7_loss(a, b): # LET A BE BATCH AND B BE RECON
    return sum6_loss(a, b) + Coul_NB_diff(a, b)

@jax.jit
def weighted_summation_loss(a, b, potential_coefficient): # LET A BE BATCH AND B BE RECON
    # Make this the square root of the mean of the sum of squares of elements
    return atom_rmsd(a,b) + potential_coefficient*scaled_pot_enr_diff(a, b)

@jax.jit
def different_summation_loss(a, b, potential_coefficient): # LET A BE BATCH AND B BE RECON
    # Make this the square root of the mean of the sum of squares of elements
    ener = scaled_pot_enr_diff(a, b)
    return atom_rmsd(a,b) + potential_coefficient*ener


def step(state, batch_x, z_rng, dropout_key, compute_fun):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True,
                                         rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
        loss = jnp.sqrt(jnp.mean(compute_fun(batch_x, logits[0])**2)) #RMSE of loss function
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

@jax.jit
def potential_step(state, batch_x, z_rng, potential_coefficient, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True,
                                         rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
        loss = scaled_pot_enr_diff(batch_x, logits[0]).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state

@jax.jit
def weighted_summation_step(state, batch_x, z_rng, potential_coefficient, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True,
                                         rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
        loss = weighted_summation_loss(batch_x, logits[0], potential_coefficient).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state

@jax.jit
def different_summation_step(state, batch_x, z_rng, potential_coefficient, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def loss_fn(params):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                         batch_x, z_rng, train=True,
                                         rngs={'dropout': dropout_train_key}, mutable=['batch_stats'])
        loss = different_summation_loss(batch_x, logits[0], potential_coefficient).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state