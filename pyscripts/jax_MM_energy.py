"""
 OBC METHOD
 A. Onufriev, D. Bashford and D. A. Case, "Exploring Protein
     Native States and Large-Scale Conformational Changes with a
     Modified Generalized Born Model", PROTEINS, 55, 383-394 (2004)

 ACE METHOD
 M. Schaefer, C. Bartels and M. Karplus, "Solution Conformations
     and Thermodynamics of Structured Peptides: Molecular Dynamics
     Simulation with an Implicit Solvation Model", J. Mol. Biol.,
     284, 835-848 (1998)
"""

#PACKAGE IMPORTS
import jax, os, sys, re
import jax.numpy as jnp
import numpy as np 
import mdtraj as md
from openmm.app import *
from openmm import *
from openmm.unit import *

#CUSTOM IMPORTS
try:
    from .general_utility import *
except:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from general_utility import *

#CONSTANTS
E_CHARGE = 1.602176634e-19
AVOGADRO = 6.02214076e23
# (e^2 Na/(kJ nm)) == (e^2/(kJ mol nm)) */ 
EPSILON0 = 1e-6*8.8541878128e-12/(E_CHARGE*E_CHARGE*AVOGADRO)
ONE_4PI_EPS0 = 1.0/(4.0*jnp.pi*EPSILON0)

#MATHEMATICS FUNCTIONS
square_distance = lambda dR: jnp.sum(dR**2, axis=-1)
distance = lambda dR: jnp.sqrt(square_distance(dR))
cosine_angle_between_two_vectors = lambda dR_12, dR_13: jnp.clip(jnp.dot(dR_12, dR_13)/(distance(dR_12) + 1e-7)/(distance(dR_13) + 1e-7), -1.0, 1.0)
harmonic_interaction = lambda r, r0, k0: jnp.float32(0.5)*k0*(r-r0)**2
torsion_interaction = lambda theta, cos_phase0, n0, k0: k0*(1.0+jnp.cos(n0*theta)*cos_phase0)
nonbonded_Coul = lambda dr, chg_ij: ONE_4PI_EPS0*chg_ij/dr
nonbonded_LJ = lambda dr, sigma_ij, epsilon_ij: 4*epsilon_ij*((sigma_ij/dr)**12 - (sigma_ij/dr)**6)


#HANDLE AMBER PRMTOP FILES
class AmberPRMTOPHandler():
    """
    Based on original code by Soohaeng Yoo Willow 2022

    Parse an Amber prmtop file and generate jax functions for calculating the energy

    Functions that are returned by build... methods should typically be vmapped over the configurations.
        They are not decorated with the jax.vmap decorator by default.
    """
    def __init__(self, fname_prmtop, OBC2=False, verbose=False):
        """
        fname_prmtop :string = filename of amber prmtop file
        OBC2 :bool = Whether to use OBC2 parameters (EXPERIMENTAL USE WITH CAUTION)
        verbose :bool = Whether to print progress
        """
        self.is_OBC2 = OBC2
        self.verbose = verbose
        
        #Load Raw Data
        prmtop_raw_data = self._amber_prmtop_load(fname_prmtop)
        printv(verbose, "Loaded prmtop data")
       
        #Parse information about atoms
        self.atomTypes = self._prm_get_atom_types(prmtop_raw_data)
        printv(verbose, "Collected Atom Types")
        
        #Parse Nonbonded terms and Pairs
        self.charges = self._prm_get_charges(prmtop_raw_data)
        self.sigmas, self.epsilons = self._prm_get_nonbond_terms(prmtop_raw_data)
        printv(verbose, "Collected Charges Sigmas and Epsilons")
        
        self.nonbondedPairs = self._prm_get_nonbond_pairs(prmtop_raw_data)
        self.nonbonded14Pairs = self._prm_get_nonbond14_info(prmtop_raw_data)
        printv(verbose, "Collected Nonbonded information")
        
        #Parse Bonded Terms and Pairs
        self.bondPairs, self.bondTypes, self.bondEquils, self.bondConstants = self._get_bonds_info(prmtop_raw_data)
        printv(verbose, "Collected Harmonic Bond information")
        
        #Parse Angle Terms and Pairs
        self.anglePairs, self.angleTypes, self.angleEquils, self.angleConstants = self._get_angles_info(prmtop_raw_data)
        printv(verbose, "Collected Harmonic Angle information")
        
        #Parse Torsion Terms and Pairs
        self.torsPairs, self.torsTypes, self.torsParams = self._get_dihedrals_info(prmtop_raw_data)
        printv(verbose, "Collected Periodic Torsion information")

        #Parse OBC2 Terms (If Necessary)
        if self.is_OBC2 == True:
            self.obc2Radii, self.obc2Screen = self._prm_get_gb_parms(prmtop_raw_data)
            printv(verbose, "Collected Generalized Born params")
    
    #BEGIN SECTION WHICH OBTAINS AND OPERATES ON RAW PRMTOP DATA
    def _amber_prmtop_load(self, fname_prmtop):
        ''' From openmm/wrappers/python/openmm/app/internal/amber_file_parser.py
        fname_prmtop: string 
        '''
        FORMAT_RE_PATTERN=re.compile("([0-9]+)\(?([a-zA-Z]+)([0-9]+)\.?([0-9]*)\)?")
        
        _flags = []
        _raw_data = {}
        _raw_format = {}
        with open(fname_prmtop, 'r') as fIn:
            for line in fIn:
                if line[0] == '%':
                    if line.startswith('%VERSION'):
                        tag, _prmtopVersion = line.rstrip().split(None, 1)
                    elif line.startswith('%FLAG'):
                        tag, flag = line.rstrip().split(None, 1)
                        _flags.append (flag)
                        _raw_data[flag] = []
                    elif line.startswith('%FORMAT'):
                        format = line.rstrip()
                        index0 = format.index('(')
                        index1 = format.index(')')
                        format = format[index0+1:index1]
                        m = FORMAT_RE_PATTERN.search(format)
                        _raw_format[_flags[-1]] = (format, m.group(1), m.group(2), int(m.group(3)), m.group(4))
                    elif line.startswith('%COMMENT'):
                        continue
                elif _flags and 'TITLE' == _flags[-1] and not _raw_data['TITLE']:
                        _raw_data['TITLE'] = line.rstrip()
                else:
                    flag = _flags[-1]
                    (format, numItems, itemType, iLength, itemPrecision) = _raw_format[flag]
                    line = line.rstrip()
                    for index in range (0, len(line), iLength):
                        item = line[index:index+iLength]
                        if item:
                            _raw_data[flag].append(item.strip())
                    
        return _raw_data

    def _prm_get_charges(self, prm_raw_data):
        charge_list = [float(x)/18.2223 for x in prm_raw_data['CHARGE']]
        return jnp.array(charge_list)
    
    def _prm_get_atom_types(self, prm_raw_data):
        atomType = [int(x) - 1 for x in prm_raw_data['ATOM_TYPE_INDEX']]
        return jnp.array(atomType)
    
    def _prm_get_nonbond_terms(self, prm_raw_data):
        numTypes = int(prm_raw_data['POINTERS'][1])
    
        LJ_ACOEF = prm_raw_data['LENNARD_JONES_ACOEF']
        LJ_BCOEF = prm_raw_data['LENNARD_JONES_BCOEF']
        # kcal/mol --> kJ/mol
        energyConversionFactor = 4.184
        # A -> nm
        lengthConversionFactor = 0.1
    
        sigma = np.zeros(numTypes)
        epsilon = np.zeros(numTypes)
    
        for i in range(numTypes):
            index = int(prm_raw_data['NONBONDED_PARM_INDEX'][numTypes*i+i]) - 1    
            acoef = float(LJ_ACOEF[index])
            bcoef = float(LJ_BCOEF[index])
    
            try:
                sig = (acoef/bcoef)**(1.0/6.0)
                eps = 0.25*bcoef*bcoef/acoef
            except ZeroDivisionError:
                sig = 1.0
                eps = 0.0
    
            sigma[i] = sig*lengthConversionFactor
            epsilon[i] = eps*energyConversionFactor
    
        return jnp.array(sigma), jnp.array(epsilon)
    
    def _prm_get_nonbond_pairs(self, prm_raw_data):
        num_excl_atoms = prm_raw_data['NUMBER_EXCLUDED_ATOMS']
        excl_atoms_list = prm_raw_data['EXCLUDED_ATOMS_LIST']
        total = 0
        numAtoms = int(prm_raw_data['POINTERS'][0])
        nonbond_pairs = []
        
        for iatom in range(numAtoms):
            index0 = total
            n = int(num_excl_atoms[iatom])
            total += n
            index1 = total
            excl_list = []
            for jatom in excl_atoms_list[index0:index1]:
                j = int(jatom) - 1
                excl_list.append(j)
    
            for jatom in range (iatom+1, numAtoms):
                if jatom in excl_list:
                    continue
                nonbond_pairs.append([iatom, jatom])
            
        return jnp.array(nonbond_pairs)
    
    def _prm_get_nonbond14_info(self, prm_raw_data):
        dihedralPointers = prm_raw_data["DIHEDRALS_WITHOUT_HYDROGEN"] + prm_raw_data["DIHEDRALS_INC_HYDROGEN"] 
        nonbond14_pairs = []
        
        for ii in range(0, len(dihedralPointers), 5):
            if int(dihedralPointers[ii+2]) > 0 and int(dihedralPointers[ii+3]) > 0:
                iAtom = int(dihedralPointers[ii])//3
                lAtom = int(dihedralPointers[ii+3])//3
                nonbond14_pairs.append ((iAtom, lAtom))
    
        return jnp.array(nonbond14_pairs)
    
    def _get_bonds_info(self, prm_raw_data):
        # kcal/mol/A^2 --> kJ/mol/nm^2
        forceConstConversionFactor = jnp.float32(418.4)
        # Amber : k(r - r0)^2
        # openmm and this code : 0.5 * k' (r - r0)^2
        # k' = 2 * k
        forceConstant = jnp.float32(2.0)*jnp.array([float(k0) for k0 in prm_raw_data['BOND_FORCE_CONSTANT']])*forceConstConversionFactor
        # A --> nm
        lengthConversionFactor = jnp.float32 (0.1)
        bondEquil = jnp.array([float(r0) for r0 in prm_raw_data['BOND_EQUIL_VALUE']])*lengthConversionFactor       
        bondPointers = prm_raw_data['BONDS_WITHOUT_HYDROGEN'] + prm_raw_data['BONDS_INC_HYDROGEN']
        
        bonds = []
        bond_types = []
        for ii in range(0, len(bondPointers), 3):
            iType = int(bondPointers[ii+2]) - 1
            bonds.append((int(bondPointers[ii])//3, int(bondPointers[ii+1])//3))
            bond_types.append(iType)
    
        return jnp.array(bonds), jnp.array(bond_types), bondEquil, forceConstant
    
    def _get_angles_info(self, prm_raw_data):
        # kcal/mol/rad^2 --> kJ/mol/rad^2
        forceConstConversionFactor = jnp.float32(4.184) 
        # Amber : k(r - r0)^2
        # openmm and this code : 0.5 * k' (r - r0)^2
        # k' = 2 * k
        forceConstant = jnp.float32(2.0)*jnp.array([float(k0) for k0 in prm_raw_data['ANGLE_FORCE_CONSTANT']])*forceConstConversionFactor
        angleEquil = jnp.array([float(r0) for r0 in prm_raw_data['ANGLE_EQUIL_VALUE']])
    
        anglePointers = prm_raw_data['ANGLES_WITHOUT_HYDROGEN'] + prm_raw_data['ANGLES_INC_HYDROGEN']
        
        angles = []
        angle_types = []
        for ii in range(0, len(anglePointers), 4):
            iType = int(anglePointers[ii+3]) - 1
            angles.append((int(anglePointers[ii]) //3, int(anglePointers[ii+1])//3, int(anglePointers[ii+2])//3))
            angle_types.append(iType)
    
        return jnp.array(angles), jnp.array(angle_types), angleEquil, forceConstant
    
    def _get_dihedrals_info(self, prm_raw_data):
        # kcal/mol/rad^2 --> kJ/mol/rad^2
        forceConstConversionFactor = jnp.float32(4.184) 
        forceConstant = jnp.array([float(k0) for k0 in prm_raw_data['DIHEDRAL_FORCE_CONSTANT']])*forceConstConversionFactor
        
        cos0 = jnp.array([jnp.cos(float(ph0)) for ph0 in prm_raw_data['DIHEDRAL_PHASE']])
        cos_phase0 = jnp.where (cos0 < 0, jnp.float32(-1), jnp.float32(1.0))
        periodicity = jnp.array([int (0.5 + float(n0)) for n0 in prm_raw_data['DIHEDRAL_PERIODICITY']])
        
        dihedralPointers = prm_raw_data['DIHEDRALS_WITHOUT_HYDROGEN'] + prm_raw_data['DIHEDRALS_INC_HYDROGEN']
        
        dihedrals = []
        dihedral_types = []
        for ii in range(0, len(dihedralPointers), 5):
            iType = int(dihedralPointers[ii+4]) - 1
            dihedrals.append((int(dihedralPointers[ii]) //3, int(dihedralPointers[ii+1])//3, abs(int(dihedralPointers[ii+2]))//3, abs(int(dihedralPointers[ii+3]))//3))
            dihedral_types.append(iType)
    
        dihedral_type_values = (periodicity, cos_phase0, forceConstant)
            
        return jnp.array(dihedrals), jnp.array(dihedral_types), dihedral_type_values

    def _prm_get_gb_parms(self, prm_raw_data):
        screen = [float(s) for s in prm_raw_data['SCREEN']]
        radii  = [float(r)/10 for r in prm_raw_data['RADII']]
        return (jnp.array(radii), jnp.array(screen))

    
    #BEGIN SECTION WHICH USES THE RAW DATA TO BUILD JAX FUNCTIONS
    def build_harmonic_bond_func(self):
        '''
        Returns a function which calculates the energy of harmonic bond interactions

        bondPairs : jnp.array ( (n_bond, 2), dtype=int )
        bondTypes: jnp.array ( (n_bond), dtype=int )
        bondEquils : jnp.array ( (n_bond_type), dtype=float )
        bondConstants : jnp.array ( (n_bond_type), dtype=float )
        '''
        r0 = self.bondEquils[self.bondTypes]
        k0 = self.bondConstants[self.bondTypes]
        
        def compute_func(R):
            '''
            R ((n_atom, 3),dtype=float)
            '''
            dR12 = R[self.bondPairs[:,1]] - R[self.bondPairs[:,0]] # (n_bond, 3)
            # r (n_bond)
            r  = jax.vmap(distance)(dR12)
            # en_val (n_bond)
            en_val = jax.vmap(harmonic_interaction)(r, r0, k0)
            return jnp.sum (en_val)
        return compute_func

    def build_harmonic_angle_func(self):
        '''
        Returns a function which calculates the energy of harmonic angle interactions
        
        anglePairs: jnp.array ((n_angle, 3), dtype=int) 
        angleTypes: jnp.array ( (n_angle), dtype=int )
        angleEquils : jnp.array ( (n_angle_type), dtype=float )
        angleConstants : jnp.array ( (n_angle_torsion), dtype=float )
        '''
        theta0 = self.angleEquils[self.angleTypes]
        k0 = self.angleConstants[self.angleTypes]
        
        def compute_func(R):
            '''
            R: jnp.array ( (n_atom, 3), dtype=float)
            '''
            dR21 = R[self.anglePairs[:, 0]] - R[self.anglePairs[:, 1]]# (n_angle, 3)
            dR23 = R[self.anglePairs[:, 2]] - R[self.anglePairs[:, 1]]
            # theta (n_angle)
            cos_angle = jax.vmap(cosine_angle_between_two_vectors)(dR21, dR23)
            theta = jnp.arccos(cos_angle)
            # en_val (n_angle)
            en_val = jax.vmap(harmonic_interaction)(theta, theta0, k0)
            return jnp.sum (en_val)
        return compute_func

    def build_torsional_angle_func(self):
        '''
        Returns a function which calculates the value of torsional angles
        
        torsPairs: jnp.array ((n_torsion, 4), dtype=int)
        '''
        
        def theta_func(dR12, dR23, dR34):
            '''
            Estimate torsional angle using four atoms: R1, R2, R3, R4
            R1, R2, R3, R4: jnp.array ( (3), dtype=float)
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
            
        def compute_func(R):
            '''
            R: jnp.array( (n_atom, 3), dtype=float)
            '''
            dR12 = R[self.torsPairs[:,1]] - R[self.torsPairs[:,0]] # R1 (n_torsion, 3)
            dR23 = R[self.torsPairs[:,2]] - R[self.torsPairs[:,1]]
            dR34 = R[self.torsPairs[:,3]] - R[self.torsPairs[:,2]]
            # theta (n_torsion)
            theta = jax.vmap(theta_func)(dR12, dR23, dR34)
            return theta
        return compute_func
        
    def build_periodic_torsion_func(self):
        """
        Returns a function which calculates the energy of periodic torsion interactions
        
        torsTypes: jnp.array ( (n_torsion), dtype=int )
        n_theta0 : jnp.array ( (n_torsion_type), dtype=int )
        cos_phase0 : jnp.array ( (n_torsion_type), dtype=float )
        k_theta0 : jnp.array ( (n_torsion_torsion), dtype=float )
        """
        n_theta0, cosine_phase0, k_theta0 = self.torsParams
        n0 = n_theta0[self.torsTypes]
        cos_phase0 = cosine_phase0[self.torsTypes]
        k0 = k_theta0[self.torsTypes]
        theta_func = self.build_torsional_angle_func()
        
        def compute_func(R):
            '''
            R: jnp.array( (n_atom, 3), dtype=float)
            '''
            theta = theta_func(R)
            en_val = jax.vmap(torsion_interaction)(theta, cos_phase0, n0, k0)
            return jnp.sum(en_val)
        return compute_func

    def build_nonbonded14_func(self, term='Both'):
        """
        Returns a function which calculates the energy of nonbonded 14 interactions

        term: string in ['Coulomb', 'Lennard-Jones', 'Both']
            The energetics terms to calculate, this distinction is made to reduce the number
            of pairwise distance calculations where possible
        """
        assert term in ['Coulomb', 'Lennard-Jones', 'Both']
        
        if term in ['Coulomb', 'Both']:
            scee0 = jnp.float32(1.0/1.2)
            chg_ab =  self.charges[self.nonbonded14Pairs[:,0]]*self.charges[self.nonbonded14Pairs[:,1]]
        
        if term in ['Lennard-Jones', 'Both']:
            scnb0 = jnp.float32(1.0/2.0)
            at_type_a = self.atomTypes[self.nonbonded14Pairs[:,0]]
            at_type_b = self.atomTypes[self.nonbonded14Pairs[:,1]]
            sig_ab = 0.5*(self.sigmas[at_type_a]+self.sigmas[at_type_b])
            eps_ab = np.sqrt(self.epsilons[at_type_a]*self.epsilons[at_type_b])

        if term == 'Coulomb':
            def compute_func(R):
                '''
                R: jnp.array( (n_atom, 3), dtype=float)
                '''
                # Ra and Rb (n_pairs, 3)
                Rab = R[self.nonbonded14Pairs[:,1]] - R[self.nonbonded14Pairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                U_chg = scee0*jax.vmap(nonbonded_Coul)(dr, chg_ab)
                return jnp.sum(U_chg)
        elif term == 'Lennard-Jones':
            def compute_func(R):
                '''
                R: jnp.array( (n_atom, 3), dtype=float)
                '''
                # Ra and Rb (n_pairs, 3)
                Rab = R[self.nonbonded14Pairs[:,1]] - R[self.nonbonded14Pairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                U_lj = scnb0*jax.vmap(nonbonded_LJ)(dr, sig_ab, eps_ab)
                return jnp.sum(U_lj)
        elif term == 'Both':
            def compute_func(R):
                '''
                R: jnp.array( (n_atom, 3), dtype=float)
                '''
                # Ra and Rb (n_pairs, 3)
                Rab = R[self.nonbonded14Pairs[:,1]] - R[self.nonbonded14Pairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                U_lj = scnb0*jax.vmap(nonbonded_LJ)(dr, sig_ab, eps_ab)
                U_chg = scee0*jax.vmap(nonbonded_Coul)(dr, chg_ab)
                return jnp.sum(U_lj) + jnp.sum(U_chg)
        else:
            raise Exception('How did that happen?')
        return compute_func

    def build_nonbonded_func(self, term='Both'):
        """
        Returns a function which calculates the energy of nonbonded interactions

        term: string in ['Coulomb', 'Lennard-Jones', 'Both']
            The energetics terms to calculate, this distinction is made to reduce the number
            of pairwise distance calculations where possible
        """
        assert term in ['Coulomb', 'Lennard-Jones', 'Both']
        
        if term in ['Coulomb', 'Both']:
            scee0 = jnp.float32(1.0/1.2)
            chg_ab =  self.charges[self.nonbondedPairs[:,0]]*self.charges[self.nonbondedPairs[:,1]]
        
        if term in ['Lennard-Jones', 'Both']:
            scnb0 = jnp.float32(1.0/2.0)
            at_type_a = self.atomTypes[self.nonbondedPairs[:,0]]
            at_type_b = self.atomTypes[self.nonbondedPairs[:,1]]
            sig_ab = 0.5*(self.sigmas[at_type_a]+self.sigmas[at_type_b])
            eps_ab = np.sqrt(self.epsilons[at_type_a]*self.epsilons[at_type_b])

        if term == 'Coulomb':
            def compute_func(R):
                '''
                R: jnp.array( (n_atom, 3), dtype=float)
                '''
                # Ra and Rb (n_pairs, 3)
                Rab = R[self.nonbondedPairs[:,1]] - R[self.nonbondedPairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                U_chg = scee0*jax.vmap(nonbonded_Coul)(dr, chg_ab)
                return jnp.sum(U_chg)
        elif term == 'Lennard-Jones':
            def compute_func(R):
                '''
                R: jnp.array( (n_atom, 3), dtype=float)
                '''
                # Ra and Rb (n_pairs, 3)
                Rab = R[self.nonbondedPairs[:,1]] - R[self.nonbondedPairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                U_lj = scnb0*jax.vmap(nonbonded_LJ)(dr, sig_ab, eps_ab)
                return jnp.sum(U_lj)
        elif term == 'Both':
            def compute_func(R):
                '''
                R: jnp.array( (n_atom, 3), dtype=float)
                '''
                # Ra and Rb (n_pairs, 3)
                Rab = R[self.nonbondedPairs[:,1]] - R[self.nonbondedPairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                U_lj = scnb0*jax.vmap(nonbonded_LJ)(dr, sig_ab, eps_ab)
                U_chg = scee0*jax.vmap(nonbonded_Coul)(dr, chg_ab)
                return jnp.sum(U_lj) + jnp.sum(U_chg)
        else:
            raise Exception('How did that happen?')
        return compute_func

    def build_gbsa_obc2_func(self):
        """
        Returns a tuple of two functions
            The first is the energy function which requires the configuration and the born radii
            The second obtains the born radii for a configuration
        """
        if not self.is_OBC2:
            raise Exception("The class must be initialized with the OBC2 flag set to True in order to use this function")

        # The following values are from 'ObcParameters.cpp'
        # OpenMM used water dielectric constant : 78.5
        # ddcosmo used : 78.3553
        solvent_dielectric = jnp.float32(78.5) # or 78.3553
        solute_dielectric = jnp.float32(1.0)
        electric_constant = -jnp.float32(0.5)*ONE_4PI_EPS0
        preFactor = 2.0*electric_constant*(1.0/solute_dielectric - 1.0/solvent_dielectric)
        probe_radius = jnp.float32(0.14)
        pi4_Asolv = jnp.float32(28.3919551)
        dielectric_offset = jnp.float32(0.009) # nm
        rad_offset = self.obc2Radii - dielectric_offset # (nAtoms)
        rad_offset_i = rad_offset.reshape(-1,1) # (nAtoms, 1)
        rad_scaled = jnp.einsum('i,i->i', rad_offset, self.obc2Screen)
        rad_scaled2 = jnp.einsum('i,i->i', rad_scaled, rad_scaled)
    
        nAtoms = self.obc2Radii.shape[0]
    
        # OBC2 Parameters
        alphaObc = jnp.float32(1.0)
        betaObc = jnp.float32(0.8)
        gammaObc = jnp.float32(4.85)
    
        def get_born_radii(R):
            # rad_offset_i : radius_offset at atom i (Float)
            # rij : the inter-distance between atom i and others : Array (Float)
            '''
            rij = []
            for ia in range (nAtoms):
                rab = R[ia] - R
                dab = jax_md.space.distance (rab)
                rij.append(dab)
            rij = jnp.array(rij)
            '''
            rij = jnp.linalg.norm(R.reshape(-1,1,3) - R, axis=2)
    
            # shape = (n_atom,n_atom)
            rij_rscaled = rad_scaled + rij
            diff_rad = rad_scaled - rij
            abs_diff_rad = abs (diff_rad)
            
            l_ij = 1.0/jnp.maximum (rad_offset_i, abs_diff_rad)
            u_ij = 1.0/rij_rscaled 
            l_ij2 = l_ij*l_ij
            u_ij2 = u_ij*u_ij 
    
            inv_rij = 1.0/rij 
            log_ratio = jnp.log ( (u_ij/l_ij) )
            term = l_ij - u_ij + 0.25*rij*(u_ij2-l_ij2) + 0.5*inv_rij*log_ratio + (0.25*rad_scaled2*inv_rij)*(l_ij2-u_ij2)
            term2 = 2.0*(1.0/rad_offset_i - l_ij)
    
            zeros = jnp.float32(0.0) #jnp.zeros (rij.shape[0])
    
            #  shape : (n_atom, n_atom) --> (n_atom)
            sum = jnp.sum(jnp.where (rad_offset_i<rij_rscaled, term, zeros), axis=-1)
            sum += jnp.sum(jnp.where (rad_offset_i<diff_rad, term2, zeros), axis=-1)
            sum *= 0.5*rad_offset
            
            tanh_sum = jnp.tanh(alphaObc*(sum) - betaObc*(sum**2) + gammaObc*(sum**3))
            born_radii = 1.0/(1.0/rad_offset - tanh_sum/self.obc2Radii)
            return born_radii
    
    
        def get_ACE_nonpolar_energy(ri, rb):
            return pi4_Asolv * (ri+probe_radius)**2 * (ri/rb)**6
            
    
        def compute_func(R, born_radii):
            
            ener_ace = jax.vmap (get_ACE_nonpolar_energy) (self.obc2Radii, born_radii)
            ener_ace = jnp.sum(ener_ace)
            
            # GB electrostatic polarization energy term 
            def get_GB_polar_energy(chg_i, r_i, born_rad_i):
                pchg_ij = preFactor*chg_i*self.charges
                drab = r_i - R
                r_ij2 = jnp.sum(drab**2, axis=-1)
                alpha2 = born_rad_i*born_radii 
                expTerm = jnp.exp(-0.25*r_ij2/alpha2)
                fgb = jnp.sqrt (r_ij2 + alpha2*expTerm)
                return jnp.float32(0.5)*jnp.sum(pchg_ij/fgb)
    
            ener_obc2 = jax.vmap(get_GB_polar_energy) (self.charges, R, born_radii)
            ener_obc2 = jnp.sum(ener_obc2)
            return ener_ace + ener_obc2
    
        return compute_func, get_born_radii



#HANDLE OPENMM XML FILES (SYSTEMS):
class OpenMMSystemHandler():
    """
    Based on original code by J. DePaolo-Boisvert 2024

    Parse an OpenMM Serialized System XML file and generate jax functions for calculating the energy

    A PDB file can also be provided, in which case a list of xml files should be provided (to be passed to ForceField),
        if none are provided - parameterization with amber14/ff14sb will be attempted.

    Functions that are returned by build... methods should typically be vmapped over the configurations.
        They are not decorated with the jax.vmap decorator by default.
    """
    def __init__(self, filename, xml_filelist=None, verbose=False):
        self.verbose = verbose
        #If a System was passed, just use it
        if type(filename) == System:
            self.system = filename
        #Attempt to Deserialize an xml
        elif filename.endswith('xml'):
            printv(verbose, "Attempting to load xml as OpenMM System")
            #Deserialize and check is OpenMM System
            with open(filename, 'r') as f:
                self.system = XmlSerializer.deserialize(f.read())
                assert type(self.system) == System
        #Attempt to parameterize a pdb with either ff14SB or user-defined
        elif filename.endswith('pdb'):
            if xml_filelist is None:
                printv(verbose, "Will attempt to parameterize the provided pdb file with ff14SB")
                xml_filelist = ['amber14/protein.ff14SB.xml']
            ff = ForceField(*xml_filelist)
            pdb = PDBFile(filename)
            self.system = ff.createSystem(pdb.topology)
        else:
            raise Exception('Filename must have pdb or xml extension')

        self.Nonbond, self.Bond, self.Angle, self.Torsion = self._obtain_energetics_information()

    
    def _obtain_energetics_information(self):
        """
        Parses an OpenMM System to retrieve the energetic interaction parameters in jax_numpy arrays.
        
        Parameters:
            system : OpenMM System : The system to parse
    
        Returns:
            Nonbonded Params : 2-tuple of jax.numpy.array : NonBonded Pairs (nPairs, 2) as ints, NonBonded Params (Charges (natom) as elementary charge, Sigmas (natom) as nanometer, Epsilons (natom) as kJ/mol)
            Covalent Bond Params : 2-tuple of jax.numpy.array : Bonded Pairs (nBond, 2) as ints, Bond Parameters (nBond, 2) as (nm, kJ/mol/nm^2)
            Angle Bond Params : 2-tuple of jax.numpy.array : Angle Pairs (nAngle, 3) as ints, Angle Parameters (nAngle, 2) as (rad, kJ/mol/rad^2)
            Torsion Bond Params : 2-tuple of jax.numpy.array : Torsion Pairs (nTorsion, 4) as ints, Torsion Parameters (nTorsion, 3) as (int, rad, 
            
        """
        #HarmonicBond
        hbf = self.system.getForce([type(force) == HarmonicBondForce for force in self.system.getForces()].index(True))
        covbond_pairs = jnp.zeros((hbf.getNumBonds(), 2), dtype=jnp.int32) #particle1, particle2
        covbond_params = jnp.zeros((hbf.getNumBonds(), 2), dtype=jnp.float32) #length (nm), constant (kJ/mol/nm^2)
        for i in range(hbf.getNumBonds()):
            covbond_pairs = covbond_pairs.at[i, :].set(hbf.getBondParameters(i)[:2])
            covbond_params = covbond_params.at[i, :].set([elem._value for elem in hbf.getBondParameters(i)[2:]])
        printv(self.verbose, 'Harmonic Bonds Done')
        
        #HarmonicAngle
        haf = self.system.getForce([type(force) == HarmonicAngleForce for force in self.system.getForces()].index(True))
        angle_pairs = jnp.zeros((haf.getNumAngles(), 3), dtype=jnp.int32) #particle1, particle2, particle3
        angle_params = jnp.zeros((haf.getNumAngles(), 2), dtype=jnp.float32) # angle (radians), constant (kJ/mol/radian^2)
        for i in range(haf.getNumAngles()):
            angle_pairs = angle_pairs.at[i,:].set(haf.getAngleParameters(i)[:3])
            angle_params = angle_params.at[i, :].set([elem._value for elem in haf.getAngleParameters(i)[3:]])
        printv(self.verbose, 'Harmonic Angles Done')
        
        #PeriodicTorsion
        ptf = self.system.getForce([type(force) == PeriodicTorsionForce for force in self.system.getForces()].index(True))
        torsion_pairs = jnp.zeros((ptf.getNumTorsions(), 4), dtype=jnp.int32) #particle 1, 2, 3, 4
        torsion_params = jnp.zeros((ptf.getNumTorsions(), 3), dtype=jnp.float32) #periodicity, phase_offset (Radians), constant
        for i in range(ptf.getNumTorsions()):
            torsion_pairs = torsion_pairs.at[i,:].set(ptf.getTorsionParameters(i)[:4])
            torsion_params = torsion_params.at[i, :].set([elem._value if type(elem) != int else elem for elem in ptf.getTorsionParameters(i)[4:]])
        printv(self.verbose, 'Periodic Torsions Done')
        
        #NonBonded
        nbf = self.system.getForce([type(force) == NonbondedForce for force in self.system.getForces()].index(True))
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
        printv(self.verbose, 'Nonbonded with Exceptions Done')
        return (all_pairs, nonbond_params), (covbond_pairs, covbond_params), (angle_pairs, angle_params), (torsion_pairs, torsion_params)

    def build_harmonic_bond_func(self):
        '''
        Parameters:
            bond_pairs : jnp.array((nbonds, 2), dtype=int) : sets of bonded atom indices (atom1, atom2)
            bond_params : jnp.array((nbonds, 2), dtype=float) : list of bond parameters (equilibrium distance (nm), force constant (kJ/mol/nm^2))
        Returns:
            Harmonic Energy Function : A function, which when given a coordinate array, returns the energy of all harmonic bonds
        '''
        bond_pairs, bond_params = self.Bond
        def compute_func(R):
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
        return compute_func
    
    def build_harmonic_angle_func(self):
        '''
        Parameters:
            angle_pairs : jnp.array((nAngles, 3), dtype=int) : Sets of atom indices which have angular interaction (atom1, atom2, atom3)
            angle_params : jnp.array((nAngles, 2), dtype=float) : Sets of Angle interaction parameters (equilibrium angle (radians), force constant (kJ/mol/radian^2))
        Returns:
            Angular Energy Function : A function, which when given a coordinate array returns the energy of all harmonic angle interactions
        '''
        angle_pairs, angle_params = self.Angle
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
    
    def build_torsional_angle_func(self):
        '''
        Returns a function which calculates the value of torsional angles
        
        torsPairs: jnp.array ((n_torsion, 4), dtype=int)
        '''
        torsPairs, _ = self.Torsion
        def theta_func(dR12, dR23, dR34):
            '''
            Estimate torsional angle using four atoms: R1, R2, R3, R4
            R1, R2, R3, R4: jnp.array ( (3), dtype=float)
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
            
        def compute_func(R):
            '''
            R: jnp.array( (n_atom, 3), dtype=float)
            '''
            dR12 = R[torsPairs[:,1]] - R[torsPairs[:,0]] # R1 (n_torsion, 3)
            dR23 = R[torsPairs[:,2]] - R[torsPairs[:,1]]
            dR34 = R[torsPairs[:,3]] - R[torsPairs[:,2]]
            # theta (n_torsion)
            theta = jax.vmap(theta_func)(dR12, dR23, dR34)
            return theta
        return compute_func
        
    def build_periodic_torsion_func(self):
        '''
        Parameters:
            torsion_pairs : jnp.array((nTorsions, 4), dtype=int) : Sets of atom indices which have torsional interactions (atom1, atom2, atom3, atom4)
            torsion_params : jnp.array((nTorsions, 3), dtype=float) : Sets of torsional parameters (periodicity (int), phase_offset (radians), force_constant(kJ/mol))
        Returns:
            Torsional Energy Function: A function, which takes a coordinate array as input and returns the energy of all torsional interactions
        '''
        _, torsion_params = self.Torsion
        theta_func = self.build_torsional_angle_func()
        
        def compute_fn(R):
            '''
            Parameters:
                R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
            Returns:
                Total Energy of torsional angle interactinos (kJ/mol)
            '''
            theta = theta_func(R)
            en_val = jax.vmap(torsion_interaction)(theta, torsion_params[:, 1], torsion_params[:, 0], torsion_params[:, 2])
            return jnp.sum(en_val)
        return compute_fn

    def build_nonbonded_func(self, term='Both'):
        """
        Returns a function which calculates the energy of nonbonded interactions

        term: string in ['Coulomb', 'Lennard-Jones', 'Both']
            The energetics terms to calculate, this distinction is made to reduce the number
            of pairwise distance calculations where possible
        """
        assert term in ['Coulomb', 'Lennard-Jones', 'Both']
        nonbonded_pairs, nonbonded_params = self.Nonbond
        
        if term == 'Coulomb':
            def compute_func(R):
                '''
                Coulomb Function
                Parameters:
                    R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
                Returns:
                    Total Energy of Nonbonded interactions (kJ/mol)
                '''
                #distances between pairs
                Rab = R[nonbonded_pairs[:,1]] - R[nonbonded_pairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                #Energy of Nonbonded Pairs
                E_Cou = jax.vmap(nonbonded_Coul)(dr, nonbonded_params[:, 0])
                return jnp.sum(E_Cou)

        elif term == 'Lennard-Jones':
            def compute_func(R):
                '''
                Lennard-Jones Function
                Parameters:
                    R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
                Returns:
                    Total Energy of Nonbonded interactions (kJ/mol)
                '''
                #distances between pairs
                Rab = R[nonbonded_pairs[:,1]] - R[nonbonded_pairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                #Energy of Nonbonded Pairs
                E_LJ = jax.vmap(nonbonded_LJ)(dr, nonbonded_params[:, 1], nonbonded_params[:, 2])
                return jnp.sum(E_LJ)
                
        elif term == 'Both':
            def compute_func(R):
                '''
                Coulomb + Lennard-Jones Summation Function
                Parameters:
                    R : jnp.array((n_atom, 3), dtype=float) : Coordinate Array of Positions (nm)
                Returns:
                    Total Energy of Nonbonded interactions (kJ/mol)
                '''
                #distances between pairs
                Rab = R[nonbonded_pairs[:,1]] - R[nonbonded_pairs[:,0]]
                dr = jax.vmap(distance)(Rab)
                #Energy of Nonbonded Pairs
                E_Cou = jax.vmap(nonbonded_Coul)(dr, nonbonded_params[:, 0])
                E_LJ = jax.vmap(nonbonded_LJ)(dr, nonbonded_params[:, 1], nonbonded_params[:, 2])
                return jnp.sum(E_Cou) + jnp.sum(E_LJ)
        else:
            raise Exception('How did that happen?')
        return compute_func
    
#OTHER FUNCTIONS (UTILITY)
def obtain_system_subset(old_system_fn:str, old_pdb_fn:str, new_system_fn:str, new_pdb_fn:str, old_selection:str):
    """
    EXPERIMENTAL - THE AUTHOR IS UNSURE OF THE CURRENT WORKING STATUS OF THIS FUNCTION, RESULTS ARE LIKELY INCORRECT!
    
    WARNING: This function is constructed under the assumption that new_pdb_fn is a pdb file that is
    already reduced to the selection of the system that you would like

    I.E. It is assumed that selection (old_selection) on the pdb file associated with the old system (old_pdb_fn, old_system_fn)
    was already used to write the pdb file from which new indices are retrieved (new_pdb_fn)

    Parameters:
        old_system_fn: string: the xml file from which the system to be 'sliced' will be loaded
        old_pdb_fn: string: pdb file associated with the old_system_fn xml file
            Note: (old_pdb_fn, old_system_fn) are likely a pdb xml pair used to run previous simulation
        new_system_fn: string: the filename to write the newly constructed 'sliced' system
        new_pdb_fn: string: filename pointing to a pdb which already contains the atoms to be sliced from old_system
            Note: new_pdb_fn should have been generated by making old_selection against old_pdb_fn and saving the selection to pdb
        old_selection: string: a selection string associated with the portion of old_system and old_pdb that you would like expressed in a new system

    Returns:
        None - the newly built system is written to new_system_fn    
    """
    #Load PDBs
    print('Load Files')
    old_pdb = md.load(old_pdb_fn)
    new_pdb = md.load(new_pdb_fn)
    #Map old indices to new indices
    assert old_pdb.top.select(old_selection).shape == new_pdb.top.select('all').shape
    index_map = jnp.array((old_pdb.top.select(old_selection), new_pdb.top.select('all'))).T
    assert jnp.all(index_map[:, 1] == jnp.arange(index_map.shape[0]))
    
    with open(old_system_fn, 'r') as f:
        old_system = XmlSerializer.deserialize(f.read())
    
    #Get Forces from old system
    print('Read Forces')
    hbf = old_system.getForce([type(force) == HarmonicBondForce for force in old_system.getForces()].index(True))
    haf = old_system.getForce([type(force) == HarmonicAngleForce for force in old_system.getForces()].index(True))
    ptf = old_system.getForce([type(force) == PeriodicTorsionForce for force in old_system.getForces()].index(True))
    nbf = old_system.getForce([type(force) == NonbondedForce for force in old_system.getForces()].index(True))

    #Get Nonbonded parameters for all particles from old system
    print('Obtain Charges, Sigmas, Epsilons')
    if nbf.getNumParticleParameterOffsets() != 0:
        raise NotImplementedError
    charges = jnp.zeros(nbf.getNumParticles())
    sigmas = jnp.zeros(nbf.getNumParticles())
    epsilons = jnp.zeros(nbf.getNumParticles())
    for i in range(nbf.getNumParticles()):
        charges = charges.at[i].set(nbf.getParticleParameters(i)[0]._value)
        sigmas = sigmas.at[i].set(nbf.getParticleParameters(i)[1]._value)
        epsilons = epsilons.at[i].set(nbf.getParticleParameters(i)[2]._value)

    #Build the New System from scratch
    print('Building New System')
    build = System()
    #Add Particles
    #iterate over the list of new indices, retrieve the previous index and the mass of that particle
    for i in range(index_map.shape[0]):
        build.addParticle(old_system.getParticleMass(index_map[:, 0][jnp.where(index_map[:, 1] == i)].item()))
    
    #Add Constraints
    print('Adding New Constraints')
    for i in range(old_system.getNumConstraints()):
        params = old_system.getConstraintParameters(i)
        if jnp.any(index_map[:, 0] == params[0]) and jnp.any(index_map[:, 0] == params[1]):
            params[0] = index_map[:, 1][jnp.where(index_map[:, 0] == params[0])].item()
            params[1] = index_map[:, 1][jnp.where(index_map[:, 0] == params[1])].item()
            build.addConstraint(*params)

    #Build new Harmonic Bond Force and replace the old indices with the new ones based on the map
    print('Adding New HarmonicBondForce')
    new_hbf = HarmonicBondForce()
    for i in range(hbf.getNumBonds()):
        params = hbf.getBondParameters(i)
        params[0] = index_map[:, 1][jnp.where(index_map[:, 0] == params[0])].item()
        params[1] = index_map[:, 1][jnp.where(index_map[:, 0] == params[1])].item()
        new_hbf.addBond(*params)
    build.addForce(new_hbf)

    #Build new Harmonic Angle Force
    print('Adding New HarmonicAngleForce') 
    new_haf = HarmonicAngleForce()
    for i in range(haf.getNumAngles()):
        params = haf.getAngleParameters(i)
        params[0] = index_map[:, 1][jnp.where(index_map[:, 0] == params[0])].item()
        params[1] = index_map[:, 1][jnp.where(index_map[:, 0] == params[1])].item()
        params[2] = index_map[:, 1][jnp.where(index_map[:, 0] == params[2])].item()
        new_haf.addAngle(*params)
    build.addForce(new_haf)

    #Build new Periodic Torsion Force
    print('Adding New PeriodicTorsionForce') 
    new_ptf = PeriodicTorsionForce()
    for i in range(ptf.getNumTorsions()):
        params = ptf.getTorsionParameters(i)
        params[0] = index_map[:, 1][jnp.where(index_map[:, 0] == params[0])].item()
        params[1] = index_map[:, 1][jnp.where(index_map[:, 0] == params[1])].item()
        params[2] = index_map[:, 1][jnp.where(index_map[:, 0] == params[2])].item()
        params[3] = index_map[:, 1][jnp.where(index_map[:, 0] == params[3])].item()
        new_ptf.addTorsion(*params)
    build.addForce(new_ptf)

    #Build new Nonbonded Force
    print('Adding New NonBondedForce') 
    new_charges = charges[index_map[:, 0]]
    new_sigmas = sigmas[index_map[:, 0]]
    new_epsilons = epsilons[index_map[:, 0]]
    
    assert build.getNumParticles() == new_charges.shape[0]
    assert new_sigmas.shape[0] == new_charges.shape[0]
    assert new_epsilons.shape[0] == new_charges.shape[0]
    
    new_nbf = NonbondedForce()
    print('NonBonded Parameters') 
    for i in range(build.getNumParticles()):
        new_nbf.addParticle(new_charges[i].item() * elementary_charge,
                            new_sigmas[i].item() * nanometer,
                            new_epsilons[i].item() * kilojoule_per_mole)
    #Build Exceptions for new Nonbonded force
    print('NonBonded Exceptions')
    for i in range(nbf.getNumExceptions()):
        params = nbf.getExceptionParameters(i)
        if jnp.any(index_map[:, 0] == params[0]) and jnp.any(index_map[:, 0] == params[1]):
            params[0] = index_map[:, 1][jnp.where(index_map[:, 0] == params[0])].item()
            params[1] = index_map[:, 1][jnp.where(index_map[:, 0] == params[1])].item()
            new_nbf.addException(*params)
    build.addForce(new_nbf)
    
    #Write out the new system
    print(f'Writing Built System to {new_system_fn}')
    with open(new_system_fn, 'w') as f:
        f.write(XmlSerializer.serialize(build))
    