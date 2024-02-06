import jax, os
import jax.numpy as jnp
import jax_amber2 as ja



fname_prmtop = os.path.join(os.getcwd(), 'Simulation/ala_deca_peptide.prmtop')
ener_fun = ja.get_amber_gas_energy_function(fname_prmtop)

@jax.vmap
def atom_rmsd(a, b, **kwargs): # for arrays of (n_conf, n_atom*3)
    '''
    Calculate the root mean square deviation (RMSD) between two sets of atomic conformations.

    Parameters:
        - a (array): Array of shape (n_conf, n_atom*3) representing the set of input conformations of atoms.
        - b (array): Array of shape (n_conf, n_atom*3) representing the set of encoded/decoded conformations of atoms.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - float: The RMSD value calculated between the two sets of atomic conformations.
    '''
    mn = a.shape[-1]//3
    x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
    return jnp.sqrt(jnp.mean((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))

@jax.jit
def scaled_pot_enr_diff(a, b, **kwargs): # WITH A AS BATCH AND B AS RECON
    '''
    Calculate the scaled potential energy difference between two sets of conformations.

    Parameters:
        - a (array): Array representing the batch (input) conformation.
        - b (array): Array representing the reconstruction (output) conformation.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - float: The scaled potential energy difference.

    Notes:
        - The function assumes the existence of "ener_fun" that computes the potential energy of a conformation.
        - The returned value is a unitless quantity.
    '''
    return ((ener_fun(a) - ener_fun(b))/ener_fun(a))**2 #Unitless quantity

@jax.jit
def summation_loss(a, b, potential_coefficient, **kwargs): # LET A BE BATCH AND B BE RECON
    '''
    Calculate the defined loss function between two sets of conformations for this neural network. 

    Parameters:
        - a (array): Array representing the set of input conformations.
        - b (array): Array representing the set of encoded/decoded conformations; output conformations.
        - potential_coefficient (float): Coefficient used in scaling of potential energy difference.

    Returns:
        - float: The calculated value for the loss function.

    Notes:
        - The function assumes the existence of "atom_rmsd" and "scaled_pot_enr_diff" functions.
    '''
    return jnp.sqrt(jnp.sum(atom_rmsd(a,b)**2)) + potential_coefficient * scaled_pot_enr_diff(a, b).mean()

@jax.jit
def rmsd_step(state, batch_x, **kwargs):
    '''
    Perform one step of optimization to minimize the root mean square deviation (RMSD) loss between batch_x and its reconstruction.

    Parameters:
        - state (OptimizerState): Object representing the optimizer state, including its parameters and apply functions.
        - batch_x (array): Array representing the batch input data.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - OptimizerState: Updated optimizer state following application of gradients.

    Notes:
        - This function assumes the existence of "atom_rmsd" to calculate RMSD between conformations.
        - It also defines "loss_fn" to calculate RMSD loss based on optimizer state parameters and apply function(s).
        - Gradients are computed using automatic differentiation provided by JAX; used to update the parameters of the model.
    '''
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return jnp.sqrt(jnp.sum(atom_rmsd(batch_x, decoded_x)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients (grads=grads)

@jax.jit
def rmsd_rng_step(state, batch_x, z_rng, **kwargs):
    '''
    Perform one step of optimization to minimize the root mean square deviation (RMSD) loss between batch_x and its reconstruction.

    Parameters:
        - state (OptimizerState): Object representing the optimizer state, including its parameters and apply functions.
        - batch_x (array): Array representing the batch input data.
        - z_rng (array): Array representing the random noise added for reconstruction.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - OptimizerState: Updated optimizer state following application of gradients.

    Notes:
        - This function assumes the existence of "atom_rmsd" to calculate RMSD between conformations.
        - It also defines "loss_fn" to calculate RMSD loss based on optimizer state parameters and apply function(s).
            - The "apply_fn" function accepts z_rng as random noise to increase generalizability of the model.
        - Gradients are computed using automatic differentiation provided by JAX; used to update the parameters of the model.
    '''
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return jnp.sqrt(jnp.sum(atom_rmsd(batch_x, decoded_x)**2))
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients (grads=grads)

@jax.jit
def potential_step(state, batch_x, **kwargs):
    '''
    Perform one step of optimization to minimize the scaled potential energy difference loss between batch_x and its reconstruction.

    Parameters:
        - state (OptimizerState): Object representing the optimizer state, including its parameters and apply functions.
        - batch_x (array): Array representing the batch input data.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - OptimizerState: Updated optimizer state following application of gradients.

    Notes:
        - This function assumes the existence of "scaled_pot_enr_diff" to calculate the scaled potential energy difference between conformations.
        - It also defines "loss_fn" to calculate potential energy difference loss based on optimizer state parameters and apply function(s).
        - Gradients are computed using automatic differentiation provided by JAX; used to update the parameters of the model.
    '''
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return scaled_pot_enr_diff(batch_x, decoded_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def potential_rng_step(state, batch_x, z_rng, **kwargs):
    '''
    Perform one step of optimization to minimize the scaled potential energy difference loss between batch_x and its reconstruction.

    Parameters:
        - state (OptimizerState): Object representing the optimizer state, including its parameters and apply functions.
        - batch_x (array): Array representing the batch input data.
        - z_rng (array): Array representing the random noise added for reconstruction.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - OptimizerState: Updated optimizer state following application of gradients.

    Notes:
        - This function assumes the existence of "scaled_pot_enr_diff" to calculate scaled potential energy difference between conformations.
        - It also defines "loss_fn" to calculate scaled potential energy differnce based on optimizer state parameters and apply function(s).
            - The "apply_fn" function accepts z_rng as random noise to increase generalizability of the model.
        - Gradients are computed using automatic differentiation provided by JAX; used to update the parameters of the model.
    '''
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return scaled_pot_enr_diff(batch_x, decoded_x).mean()
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def summation_step(state, batch_x, potential_coefficient, **kwargs):
    '''
    Perform one step of optimization to minimize the total loss function value between batch_x and its reconstruction.

    Parameters:
        - state (OptimizerState): Object representing the optimizer state, including its parameters and apply functions.
        - batch_x (array): Array representing the batch input data.
        - potential_coefficient (float): Coefficient to scale the potential energy difference contribution to the loss function.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - OptimizerState: Updated optimizer state following application of gradients.

    Notes:
        - This function assumes the existence of "summation_loss" to calculate the loss function value between conformations.
        - It also defines "loss_fn" to calculate the loss function value based on optimizer state parameters and apply function(s).
        - Gradients are computed using automatic differentiation provided by JAX; used to update the parameters of the model.
    '''
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x)[0]
        return summation_loss(batch_x, decoded_x, potential_coefficient)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)

@jax.jit
def summation_rng_step(state, batch_x, z_rng, potential_coefficient, **kwargs):
    '''
     Perform one step of optimization to minimize the total loss function value between batch_x and its reconstruction.

    Parameters:
        - state (OptimizerState): Object representing the optimizer state, including its parameters and apply functions.
        - batch_x (array): Array representing the batch input data.
        - z_rng (array): Array representing the random noise added for reconstruction.
        - potential_coefficient (float): Coefficient to scale the potential energy difference contribution in the loss function.
        - **kwargs: Additional keyword arguments (optional).

    Returns:
        - OptimizerState: Updated optimizer state following application of gradients.

    Notes:
        - This function assumes the existence of "summation_loss" to calculate the loss function value between conformations.
        - It also defines "loss_fn" to calculate loss function value based on optimizer state parameters and apply function(s).
            - The "apply_fn" function accepts z_rng as random noise to increase generalizability of the model.
        - Gradients are computed using automatic differentiation provided by JAX; used to update the parameters of the model.
    '''
    def loss_fn(params, apply_fn):
        decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
        return summation_loss(batch_x, decoded_x, potential_coefficient)
    grads = jax.grad(loss_fn)(state.params, state.apply_fn)
    return state.apply_gradients(grads=grads)
