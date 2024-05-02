import jax.numpy as jnp
import jax
from datetime import datetime
from openmm.app import *
from openmm import *
from openmm.unit import *

def obtain_energetics_information(sys:System):
    """
    Parses an OpenMM System to retrieve the energetic interaction parameters in jax_numpy arrays.
    
    Parameters:
        sys : OpenMM System : The system to parse

    Returns:
        Nonbonded Params : 3-tuple of jax.numpy.array : Charges (natom) as elementary charge, Sigmas (natom) as nanometer, Epsilons (natom) as kJ/mol
        Covalent Bond Params : 2-tuple of jax.numpy.array : Bonded Pairs (nBond, 2) as ints, Bond Parameters (nBond, 2) as (nm, kJ/mol/nm^2)
        Angle Bond Params : 2-tuple of jax.numpy.array : Angle Pairs (nAngle, 3) as ints, Angle Parameters (nAngle, 2) as (rad, kJ/mol/rad^2)
        Torsion Bond Params : 2-tuple of jax.numpy.array : Torsion Pairs (nTorsion, 4) as ints, Torsion Parameters (nTorsion, 3) as (int, rad, 
        
    """
    #HarmonicBond
    hbf = sys.getForce([type(force) == HarmonicBondForce for force in sys.getForces()].index(True))
    covbond_pairs = jnp.zeros((hbf.getNumBonds(), 2), dtype=jnp.int32) #particle1, particle2
    covbond_params = jnp.zeros((hbf.getNumBonds(), 2), dtype=jnp.float32) #length (nm), constant (kJ/mol/nm^2)
    for i in range(hbf.getNumBonds()):
        covbond_pairs = covbond_pairs.at[i, :].set(hbf.getBondParameters(i)[:2])
        covbond_params = covbond_params.at[i, :].set([elem._value for elem in hbf.getBondParameters(i)[2:]])

    #HarmonicAngle
    haf = sys.getForce([type(force) == HarmonicAngleForce for force in sys.getForces()].index(True))
    angle_pairs = jnp.zeros((haf.getNumAngles(), 3), dtype=jnp.int32) #particle1, particle2, particle3
    angle_params = jnp.zeros((haf.getNumAngles(), 2), dtype=jnp.float32) # angle (radians), constant (kJ/mol/radian^2)
    for i in range(haf.getNumAngles()):
        angle_pairs = angle_pairs.at[i,:].set(haf.getAngleParameters(i)[:3])
        angle_params = angle_params.at[i, :].set([elem._value for elem in haf.getAngleParameters(i)[3:]])

    #PeriodicTorsion
    ptf = sys.getForce([type(force) == PeriodicTorsionForce for force in sys.getForces()].index(True))
    torsion_pairs = jnp.zeros((ptf.getNumTorsions(), 4), dtype=jnp.int32) #particle 1, 2, 3, 4
    torsion_params = jnp.zeros((ptf.getNumTorsions(), 3), dtype=jnp.float32) #periodicity, phase_offset (Radians), constant
    for i in range(ptf.getNumTorsions()):
        torsion_pairs = torsion_pairs.at[i,:].set(ptf.getTorsionParameters(i)[:4])
        torsion_params = torsion_params.at[i, :].set([elem._value if type(elem) != int else elem for elem in ptf.getTorsionParameters(i)[4:]])
        
    #NonBonded
    nbf = sys.getForce([type(force) == NonbondedForce for force in sys.getForces()].index(True))
    if nbf.getNumParticleParameterOffsets() != 0:
        raise NotImplementedError
    charges = jnp.zeros(nbf.getNumParticles())
    sigmas = jnp.zeros(nbf.getNumParticles())
    epsilons = jnp.zeros(nbf.getNumParticles())
    for i in range(nbf.getNumParticles()):
        charges = charges.at[i].set(nbf.getParticleParameters(i)[0]._value)
        sigmas = sigmas.at[i].set(nbf.getParticleParameters(i)[1]._value)
        epsilons = epsilons.at[i].set(nbf.getParticleParameters(i)[2]._value)
    all_pairs = []
    for i in range(nbf.getNumParticles()):
        for j in range(i+1, nbf.getNumParticles()):
            all_pairs.append([i, j])
    all_pairs = jnp.array(all_pairs)
    # Set all params to the unexceptioned values
    ind1 = all_pairs[:, 0]
    ind2 = all_pairs[:, 1]
    charge_products = charges[ind1] * charges[ind2]
    sigma_products = jnp.float32(0.5)*(sigmas[ind1] + sigmas[ind2])
    epsilon_products = jnp.sqrt(epsilons[ind1] * epsilons[ind2])
    nonbond_params = jnp.array((charge_products, sigma_products, epsilon_products)).T
    #get array of exceptions
    for i in range(nbf.getNumExceptions()):
        #Retrieve Exception
        exception = nbf.getExceptionParameters(i)
        #Determine the Excepted Pair
        excepted_pair = jnp.array(exception[:2])
        #Get the parameters for this pair
        excepted_params = jnp.array([val._value for val in exception[2:]])
        #Determine the index of all_pairs that has the excepted pair
        inds, counts = jnp.unique(jnp.where(all_pairs == excepted_pair)[0], return_counts=True)
        spec_ind = inds[jnp.where(counts>1)]
        #At that index, replace the parameters with the excepted parameters
        nonbond_params = nonbond_params.at[spec_ind, :].set(excepted_params)
    
    return (all_pairs, nonbond_params), (covbond_pairs, covbond_params), (angle_pairs, angle_params), (torsion_pairs, torsion_params)

def square_distance(dR:jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(dR**2, axis=-1)

def distance(dR:jnp.ndarray):
    dr = square_distance(dR)
    return jnp.sqrt(dr)

def nonbonded_pairs_within_cutoff(R:jnp.ndarray, cutoff):
    distance_matrix = jnp.zeros((R.shape[0], R.shape[0]))
    for i in range(R.shape[0]):
        distance_matrix = distance_matrix.at[i, :].set(jax.vmap(distance)(R - R[i, :]))
    return jnp.array(jnp.where(distance_matrix <= cutoff)).T

def cosine_angle_between_two_vectors(dR_12:jnp.ndarray, dR_13:jnp.ndarray) -> jnp.ndarray:
    dr_12 = distance(dR_12) + 1e-7
    dr_13 = distance(dR_13) + 1e-7
    cos_angle = jnp.dot(dR_12, dR_13) / dr_12 / dr_13
    return jnp.clip(cos_angle, -1.0, 1.0)

def harmonic_interaction(r, r0, k0):
    return 0.5*k0*(r - r0)**2

def torsion_interaction(theta, theta0, n0, k0):
    # theta : torsional angle
    # cos_phase0 = cos (theta0), where theta0 is 0 or pi
    # U_torsion = k0 * (1 + cos (n0 theta - theta0))
    # U_torsion = k0 * (1 + cos (n0 theta) * cos_phase0)
    return k0*(1.0 + jnp.cos(n0*theta - theta0))

def get_ener_harm_fun(bond_pairs, bond_params):
    '''
    Parameters:
        bond_pairs : jnp.array((nbonds, 2), dtype=int) : sets of bonded atom indices (atom1, atom2)
        bond_params : jnp.array((nbonds, 2), dtype=float) : list of bond parameters (equilibrium distance (nm), force constant (kJ/mol/nm^2))
    Returns:
        Harmonic Energy Function : A function, which when given a coordinate array, returns the energy of all harmonic bonds
    '''
    def compute_fn(R):
        '''
        Parameters:
            R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
        Returns:
            Total Energy of harmonic bond interactinos (kJ/mol)
        '''
        dR = R[bond_pairs[:,1]] - R[bond_pairs[:,0]] # (n_bond, 3)
        r = jax.vmap(distance)(dR)
        en_val = jax.vmap(harmonic_interaction)(r, bond_params[:, 0], bond_params[:, 1])
        return jnp.sum(en_val)
    return compute_fn

def get_ener_angle_fun(angle_pairs, angle_params):
    '''
    Parameters:
        angle_pairs : jnp.array((nAngles, 3), dtype=int) : Sets of atom indices which have angular interaction (atom1, atom2, atom3)
        angle_params : jnp.array((nAngles, 2), dtype=float) : Sets of Angle interaction parameters (equilibrium angle (radians), force constant (kJ/mol/radian^2))
    Returns:
        Angular Energy Function : A function, which when given a coordinate array returns the energy of all harmonic angle interactions
    '''
    def compute_fn(R):
        '''
        Parameters:
            R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
        Returns:
            Total Energy of harmonic angle interactinos (kJ/mol)
        '''
        dR21 = R[angle_pairs[:, 0]] - R[angle_pairs[:, 1]]# (n_angle, 3)
        dR23 = R[angle_pairs[:, 2]] - R[angle_pairs[:, 1]]
        cos_angle = jax.vmap(cosine_angle_between_two_vectors)(dR21, dR23)
        theta = jnp.arccos(cos_angle)
        en_val = jax.vmap(harmonic_interaction)(theta, angle_params[:, 0], angle_params[:, 1])
        return jnp.sum(en_val)
    return compute_fn

def get_ener_torsion_fun(torsion_pairs, torsion_params):
    '''
    Parameters:
        torsion_pairs : jnp.array((nTorsions, 4), dtype=int) : Sets of atom indices which have torsional interactions (atom1, atom2, atom3, atom4)
        torsion_params : jnp.array((nTorsions, 3), dtype=float) : Sets of torsional parameters (periodicity (int), phase_offset (radians), force_constant(kJ/mol))
    Returns:
        Torsional Energy Function: A function, which takes a coordinate array as input and returns the energy of all torsional interactions
    '''
    def theta_fn(dR12, dR23, dR34):
        '''
        Calculate the angle between two planes.  Plane1 contains dR12 and dR23, Plane2 contains dR23 and dR34.
        The two planes intersect along the vector dR23
        '''
        dRT = jnp.cross(dR12, dR23)
        dRU = jnp.cross(dR23, dR34)
        dTU = jnp.cross(dRT, dRU)

        rt = distance(dRT) + 1.e-7
        ru = distance(dRU) + 1.e-7
        r23 = distance(dR23) + 1.e-7

        cos_angle = jnp.dot(dRT, dRU)/(rt*ru)
        sin_angle = jnp.dot(dR23, dTU)/(r23*rt*ru)
        theta = jnp.arctan2(sin_angle, cos_angle)
        return theta

    def compute_fn(R):
        '''
        Parameters:
            R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
        Returns:
            Total Energy of torsional angle interactinos (kJ/mol)
        '''
        dR12 = R[torsion_pairs[:,1]] - R[torsion_pairs[:,0]]
        dR23 = R[torsion_pairs[:,2]] - R[torsion_pairs[:,1]]
        dR34 = R[torsion_pairs[:,3]] - R[torsion_pairs[:,2]]
        theta = jax.vmap(theta_fn)(dR12, dR23, dR34)
        en_val = jax.vmap(torsion_interaction)(theta, torsion_params[:, 1], torsion_params[:, 0], torsion_params[:, 2])
        return jnp.sum(en_val)
    return compute_fn

_ONE_4PI_EPS0 = 138.905
#_ONE_4PI_EPS0 = 138.935456
nonbonded_Coul = lambda dr, chg_ij: _ONE_4PI_EPS0*chg_ij/dr
nonbonded_LJ = lambda dr, sigma_ij, epsilon_ij: 4*epsilon_ij*((sigma_ij/dr)**12 - (sigma_ij/dr)**6)

def get_coulomb_nocutoff_fun(nonbonded_pairs, charge_products):

    def compute_fn(R):
        '''
        Parameters:
            R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
        Returns:
            Total Energy of Nonbonded interactions (kJ/mol)
        '''
        #distances between pairs
        Rab = R[nonbonded_pairs[:,1]] - R[nonbonded_pairs[:,0]]
        dr = jax.vmap(distance)(Rab)
        #Energy of Nonbonded Pairs
        E_Cou = jax.vmap(nonbonded_Coul)(dr, charge_products)
        return jnp.sum(E_Cou)
    return compute_fn

def get_LJ_nocutoff_fun(nonbonded_pairs, sigma_products, epsilon_products):
    assert sigma_products.shape == epsilon_products.shape

    def compute_fn(R):
        '''
        Parameters:
            R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
        Returns:
            Total Energy of Nonbonded interactions (kJ/mol)
        '''
        #distances between pairs
        Rab = R[nonbonded_pairs[:,1]] - R[nonbonded_pairs[:,0]]
        dr = jax.vmap(distance)(Rab)
        #Energy of Nonbonded Pairs
        E_LJ = jax.vmap(nonbonded_LJ)(dr, sigma_products, epsilon_products)
        return jnp.sum(E_LJ)
    return compute_fn

def get_openmm_energy_functions(in_file_fn, forcefields='Default', mode='SUM'):
    """
    Obtain the energy function based on either a pdb file, or an openmm system.
    If in_file_fn is an .xml file, the openmm system will be created from the xml
    If in_file_fn is a pdb file, the openmm system will be created using either the default forcefield or
    optionally a user defined forcefield with the pdbfiles topology.

    Parameters:
        in_file_fn: string: .pdb or .xml file
    """
    if in_file_fn.endswith('.pdb'):
        if forcefields == 'Default':
            ff = ForceField('amber14/protein.ff14SB.xml')
        elif type(forcefields) == list:
            ff = ForceField(*forcefields)
        else:
            raise Exception('Forcefield file error')
        pdb = PDBFile(in_file_fn)
        sys = ff.createSystem(pdb.topology)
    elif in_file_fn.endswith('.xml'):
        with open(in_file_fn, 'r') as f:
            sys = XmlSerializer.deserialize(f.read())
    else:
        raise Exception('pdb or xml file required')

    nonbonded, harmonic, angular, torsional = obtain_energetics_information(sys)
    harm_bond_func = get_ener_harm_fun(*harmonic)
    harm_angl_func = get_ener_angle_fun(*angular)
    peri_tors_func = get_ener_torsion_fun(*torsional)
    coulomb_func = get_coulomb_nocutoff_fun(nonbonded[0], nonbonded[1][:, 0])
    LJ_func = get_LJ_nocutoff_fun(nonbonded[0], nonbonded[1][:, 1], nonbonded[1][:, 2])
    
    if mode == 'SUM':
        def compute_fn(R):
            R = R.reshape(-1,3)
            return harm_bond_func(R) + harm_angl_func(R) + peri_tors_func(R) + coulomb_func(R) + LJ_func(R)
    elif mode == 'DECOMP':
        def compute_fn(R):
            R = R.reshape(-1,3)
            return jnp.array((harm_bond_func(R), harm_angl_func(R), peri_tors_func(R), coulomb_func(R), LJ_func(R)))
    return jax.vmap(compute_fn)