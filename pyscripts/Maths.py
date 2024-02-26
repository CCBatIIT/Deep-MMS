import jax, os
from . import jax_amber3 as jaa
import jax.numpy as jnp
import numpy as np

######################################################################################################################################################
# Last Amended February 14, 2024
#     Author: J.A.DePaolo-Boisvert
######################################################################################################################################################
# List of Functions
#    Maths: Class: A container for mathematical functions, the potential and torsion functions defined by the input prmtop
#
#    Distances:
#    atom_rmsd: A one-to-one mapping of atomic RMSDs
#    atom_rmtd: A one-to-one mapping of Torsional RMSDs (RMTDs)
#    SOON TO BE DEPRECATED structural_distance: A one-to-one mapping of the sum of RMSD and RMTD
#    scaled_pot_enr_diff: A one-to-one mapping of Squared % deviation in potential energy (sqrt this for %deviation magnitude
#    SOON TO BE DEPRECATED summation_distance: A one-to-one mapping of the sum of structural_distance and scaled_pot_enr_distance
#    
#    All of the above can also do full pairwise distance matrices with:
#        rmsd_distance_matrix, rmtd_distance_matrix, potential_distance_matrix
#
#    make_summation_distance_function(distance_functions:list, weights:list)
#        Deprecated functions are to be replaced with this
#        Constructs a weighted summation of loss functions (rmsd, rmtd, potential, repulsion)
#        The ith loss function is weighted by the ith weight in particle (conformation) wise summation
#
#    establish_step_function(loss_function, averaging_method)
#        Any one-to-one distance mapping can be chosen as the loss_function
#        A summation from make_summation_distance_function can also be used
#        averaging method must be in ['mean', 'rmsd'] where:
#            'mean' yields the true average of the particles' loss values
#            'rmsd' yields the rmsd of the particles' loss values
#
#    make_gaussian_kernel(distance_matrix_function, averaging_method, train_data):
#        Utilize a distance matrix function to construct a gaussian kernel repulsion term
######################################################################################################################################################


class Maths():
    """
    These Functions that need to invoke Potential Energy Function or Torsional Function
    This structure will allow specification of the prmtop file outside this module
    """
    def __init__(self, fname_prmtop):
        """
        Provide an AMBER prmtop file, in order to get the MM Potential Energy and Torsion Functions
        """
        self.prmtop_fn = fname_prmtop
        self.ener_fun, self.tors_fun = jaa.get_amber_functions(fname_prmtop)

        self._map_functions()

    def _map_functions(self):
        """
        In moving all of these functions to methods, decorators can no longer be used
        (Else it will attempt to map over self, which raises and error)
        This function invoked in __init__ in order to appropriately map everything
        """
        #ParticleWise Distances returns n_conf distances
        self.atom_rmsd, self.atom_rmsd_jit = jax.vmap(self.atom_rmsd, in_axes=(0, 0)), jax.jit(self.atom_rmsd)
        self.cos_torsional_dist = jax.vmap(self.cos_torsional_dist, in_axes=(0, 0))
        self.atom_rmtd = jax.jit(self.atom_rmtd)
        self.scaled_pot_enr_diff = jax.jit(self.scaled_pot_enr_diff)
        #self.structural_distance = jax.jit(self.structural_distance)
        #self.summation_distance = jax.jit(self.summation_distance)
        # Helper functions for distance matrices
        self.rmsd_one_off = jax.vmap(self.rmsd_one_off, in_axes=(None, 0))
        self.rmtd_one_off = jax.vmap(self.rmtd_one_off, in_axes=(None, 0))
        self.simple_scaled_diff = jax.vmap(self.simple_scaled_diff, in_axes=(None, 0))
        # Pairwise matrix operations
        self.rmsd_distance_matrix = jax.jit(self.rmsd_distance_matrix)
        #self.rmtd_distance_matrix = jax.jit(self.rmtd_distance_matrix)
        #self.structural_distance_matrix = jax.jit(self.structural_distance_matrix)
        #self.potential_distance_matrix = jax.jit(self.potential_distance_matrix)
        #self.summation_distance_matrix = jax.jit(self.summation_distance_matrix)
        return None
        
    def report_num_atoms(self):
        """
        This function exists as a sanity check.  There will be no errors if the
        prmtop in this file is different than that in the json.  This function
        is used to report the number of atoms in fname_prmtop, in order to check
        that there is an equivalent number of atoms in the prmtop from the json file
        """
        return len(jaa.prm_get_atom_types(jaa.amber_prmtop_load(self.prmtop_fn)))
    
    def atom_rmsd(self, a, b): # for arrays of (n_conf, n_atom*3)
        """VMAPED iterates over the n_configurational array, read a and b below as iterations over a,b which are actually provided
        Ex. if a has shape (10, 300), then in code below the expected shape of a is (300)"""
        mn = a.shape[-1]//3
        x_inds, y_inds, z_inds = jnp.arange(0,mn), jnp.arange(mn, 2*mn), jnp.arange(2*mn, 3*mn)
        return jnp.sqrt(jnp.mean((b[x_inds] - a[x_inds])**2 + (b[y_inds] - a[y_inds])**2 + (b[z_inds] - a[z_inds])**2))
            
    def rmsd_one_off(self, a, b):
        """A distance function where A is one particle, and B is a set of particles (A = (3*natom), B = (n_conf>1, 3*natom))"""
        rmsd = jnp.sqrt(jnp.mean((b - a)**2 + (b - a)**2 + (b - a)**2))
        return rmsd
        
   # def rmsd_distance_matrix(self, a, b):
   #     """
   #     Where A and B are 2D arrays of positions (n_conf, 3*n_atom)
   #     """
   #     rmsd = jnp.zeros((a.shape[0], b.shape[0]))
   #     for i in range(a.shape[0]):
   #         rmsd = rmsd.at[i,:].set(self.rmsd_one_off(a[i],b))
   #     return rmsd

    def rmsd_distance_matrix(self, xs, ys):
      return jax.vmap(lambda x: jax.vmap(lambda y: self.atom_rmsd_jit(x, y))(xs))(ys)
    
    def cos_torsional_dist(self, a, b):
        """For molecular sets, a, b
            returns 1/2 of the square of the L2 distance between two angles which are cast onto the unit circle.
            L2d(Theta1, Theta2) = sqrt(2*(1-cos(a-b)))"""
        return 1 - jnp.cos(b-a) #invariant to order of opt, as cosine is an even function
    
    def atom_rmtd(self, a, b):
        """Extension of cos_torsional_dist to calculat the RMTD (Root Mean Torsional Deviation)"""
        a, b = self.tors_fun(a), self.tors_fun(b)
        return jnp.sqrt(jnp.mean(self.cos_torsional_dist(a, b), axis=1)) #axis invoked here and not for rmSd because rmsd is sqrt summed inside vmapping 
    
    def rmtd_one_off(self, a, b):
        """Where A, and B are torsional particles, analogously defined as in rmsd one off"""
        rmtd = jnp.sqrt(jnp.mean(1 - jnp.cos(b-a)))
        return rmtd

    def rmtd_distance_matrix(self, a, b):
        """
        Where A and B are 2D arrays of positions (n_conf, 3*n_atom)
        """
        rmtd = jnp.zeros((a.shape[0], b.shape[0]))
        a, b = self.tors_fun(a), self.tors_fun(b)
        for i in range(a.shape[0]):
            rmtd = rmtd.at[i,:].set(self.rmtd_one_off(a[i],b))
        return rmtd
    
    def simple_scaled_diff(self, a, b):
        """The same operation as scaled_pot_enr_diff, but with values defined"""
        return ((b-a)/(a))**2
        
    def scaled_pot_enr_diff(self, a, b): # WITH A AS BATCH AND B AS RECON
        """Square of the percent deviation of energy of b compared to energy of a"""
        return ((self.ener_fun(b) - self.ener_fun(a))/self.ener_fun(a))**2 #Unitless quantity

    def potential_distance_matrix(self, a, b):
        """
        Where A and B are 2D arrays of positions (n_conf, 3*n_atom)
        """
        potentials = jnp.zeros((a.shape[0], b.shape[0]))
        a, b = self.ener_fun(a), self.ener_fun(b)
        for i in range(a.shape[0]):
            potentials = potentials.at[i,:].set(self.simple_scaled_diff(a[i],b))
        return potentials
    
    def make_summation_distance_function(self, distance_functions:list, weights:list):
        """
        This function will better facilitate the addition of loss terms
        Each ith distance function will be summed over particle (conformation) wise, and summed with the ith weight
        This is the optimal way to construct choice summations of different metrics.
        The output of this function can be passed directly to Maths.make_step_function
        
        Parameters:
            distance_functions - A list of Maths distance functions
            weights - Weight each distance function by the weight of the corresponding index

        Returns:
            loss_function - The created loss function
        """
        assert len(distance_functions) == len(weights)

        @jax.jit
        def loss_function(batch, decoded):
            return jnp.sum(jnp.array([weights[i] * distance_functions[i](batch, decoded) for i in range(len(weights))]), axis=0)
        return loss_function
    
    #General Step function
    def make_step_function(self, loss_function, averaging_method):
        """
        Create the step_function, the function which evaluates the loss metric provided,
        averages values by the averaging method provided, and applies gradients based on the average
        loss function may be any distance function from Maths, or combinations of them (made with Maths.make_summation_loss_function)

        Parameters:
            loss_function: one of the functions from the metrics list at the top of the module
                            or a function made with Maths.make_summation_loss_function
            averaging_method: 'mean' or 'rmsd' whether to take the true average or the rmsd of loss metric values

        Returns:
            step_function: A function which evaluates loss metric, coming to a single loss value by averaging method
                           and applies gradients to the NN based on the final value
        """
        assert averaging_method in ['mean', 'rmsd']
        if averaging_method == 'mean':
            def average(vals):
                return vals.mean()
        elif averaging_method == 'rmsd':
            def average(vals):
                return jnp.sqrt(jnp.sum(vals**2)/vals.shape[0])

        @jax.jit
        def custom_step(state, batch_x, z_rng):
            def loss(params, apply_fn):
                decoded, latents = apply_fn({'params':params}, batch_x, z_rng)
                return average(loss_function(batch_x, decoded))
            grads = jax.grad(loss)(state.params, state.apply_fn)
            return state.apply_gradients(grads=grads), loss(state.params, state.apply_fn)
        return custom_step

    #Repulsion
    def make_gaussian_kernel(self, distance_matrix_function, averaging_method, train_data, batch_size):
        """
        Create a gaussian kernel function (of the form e^(-1*(distance_metric**2)/h))
        Where distance function is an elementwise distance matrix calculation (rmsd, rmtd, potential, etc)
        h is chosen as the minimum (non-zero) distance between two particles
        """
        #determine h
        matrix = distance_matrix_function(train_data, train_data) #THIS LOOKS LIKE IT WILL TAKE A WHILE
        bandwidth = jnp.min(matrix[jnp.where(matrix != 0)])
        
        #make the kernel function
        kernel = lambda x : jnp.exp(-1*(x**2)/bandwidth)
        
        # How to assemble matrix into loss
        assert averaging_method in ['mean', 'rmsd']
        if averaging_method == 'mean':
            def average(vals):
                return vals.mean()
        elif averaging_method == 'rmsd':
            def average(vals):
                return jnp.sqrt(jnp.sum(vals**2)/vals.shape[0])
        
        #This is the number of entries in the upper tiangle (excluding diagonal) of a batch_size X batch_size matrix
        size = int(batch_size**2/2 - batch_size/2)
        
        @jax.jit
        def repulsion_term(batch, decoded):
            matrix = distance_matrix_function(decoded, decoded) #This is not a typo, decoded against itself
            triu = jnp.triu(kernel(matrix), k=1)
            return average(triu[jnp.nonzero(triu, size=size)])

        return repulsion_term

#############################################################################################################################
##############      DEPRACTION WARNING - EVERYTHING BELOW THIS LINE MAY BE DEPRECATED WITHOUT WARNING     ###################
#############################################################################################################################


    
    def summation_distance(self, a, b, potential_coefficient): # LET A BE BATCH AND B BE RECON
        """A loss function incorporating RMSD, RMTD, and potential deviation"""
        return self.structural_distance(a, b) + potential_coefficient * self.scaled_pot_enr_diff(a, b)
    
    def summation_distance_matrix(self, a, b, potential_coefficient):
        return self.rmsd_distance_matrix(a, b) + self.rmtd_distance_matrix(a, b) + potential_coefficient * self.potential_distance_matrix(a, b)
        
    def structural_distance(self, a, b, **kwargs): # LET A BE BATCH AND B BE RECON
        """
        RMSD plus RMTD
        The values of RMSD and RMTD are squared summed and square rooted (an RMSD of RMSD or RMTD)
        To penalize higher values better
        """
        return self.atom_rmsd(a,b) + self.atom_rmtd(a,b)

    def structural_distance_matrix(self, a, b):
        """
        Simply a single function to automatically provide the some of the two distance matrices (RMSD and RMTD)
        """
        return self.rmsd_distance_matrix(a, b) + self.rmtd_distance_matrix(a, b)

    def rmsd_rng_step(self, state, batch, z_rng, **kwargs):
        print('DEPRACTION WARNING')
        def loss_fn(params, apply_fn):
            decoded, _ = apply_fn({'params':params}, batch, z_rng)
            return jnp.sqrt(jnp.sum(atom_rmsd(batch, decoded)**2))
        grads = jax.grad(loss_fn)(state.params, state.apply_fn)
        return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)
    
    def rmtd_rng_step(self, state, batch, z_rng, **kwargs):
        print('DEPRACTION WARNING')
        def loss_fn(params, apply_fn):
            decoded, _ = apply_fn({'params':params}, batch, z_rng)
            return jnp.sqrt(jnp.sum(self.atom_rmtd(batch, decoded)**2))
        grads = jax.grad(loss_fn)(state.params, state.apply_fn)
        return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)
    
    def potential_rng_step(self, state, batch, z_rng, **kwargs):
        print('DEPRACTION WARNING')
        def loss_fn(params, apply_fn):
            decoded, _ = apply_fn({'params':params}, batch, z_rng)
            return self.scaled_pot_enr_diff(batch, decoded).mean()
        grads = jax.grad(loss_fn)(state.params, state.apply_fn)
        return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)
    
    def structural_rng_step(self, state, batch_x, z_rng, **kwargs):
        print('DEPRACTION WARNING')
        def loss_fn(params, apply_fn):
            decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
            return np.sqrt(jnp.sum(self.structural_distance(batch_x, decoded_x, **kwargs)**2))
        grads = jax.grad(loss_fn)(state.params, state.apply_fn)
        return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)
    
    def summation_rng_step(self, state, batch_x, z_rng, **kwargs):
        print('DEPRACTION WARNING')
        def loss_fn(params, apply_fn):
            decoded_x = apply_fn({'params':params}, batch_x, z_rng)[0]
            return self.summation_distance(batch_x, decoded_x, **kwargs)
        grads = jax.grad(loss_fn)(state.params, state.apply_fn)
        return state.apply_gradients(grads=grads), loss_fn(state.params, state.apply_fn)