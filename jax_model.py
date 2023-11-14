import jax
import jax.numpy as jnp
import numpy as np

import jax_amber
import jax_obc2 


def get_heavy_atom_index (fname_prmtop):
    
    prm_raw_data = jax_amber.amber_prmtop_load (fname_prmtop)
    znum = jnp.array([int (val) for val in prm_raw_data['ATOMIC_NUMBER']])
    
    return jnp.where(znum > 1)[0]


def get_selected_atoms_index (fname_prmtop, selected_atoms):
    
    #backbone_atom_name = ['CA', 'C', 'O', 'N']
    #backbone_atom_name = ['CA']
    prm_raw_data = jax_amber.amber_prmtop_load (fname_prmtop)

    return jnp.array([ia for ia, val in enumerate(prm_raw_data['ATOM_NAME']) if val in selected_atoms])
    
    
        
def get_amber_energy_gas_fun (fname_prmtop):
    
    prm_raw_data = jax_amber.amber_prmtop_load (fname_prmtop)
    
    ener_bonded_fn, ener_bond_fn = \
         jax_amber.ener_bonded (prm_raw_data)
    ener_bonded_fn = jax.jit (ener_bonded_fn)
    ener_bond_fn = jax.jit (ener_bond_fn)

    chgs = jax_amber.prm_get_charges (prm_raw_data)
    atom_types = jax_amber.prm_get_atom_types (prm_raw_data)

    sigma, epsilon = jax_amber.prm_get_nonbond_terms (prm_raw_data)

    nonbond_pairs = jax_amber.prm_get_nonbond_pairs (prm_raw_data)

    ener_nbond_fn = jax_amber.ener_nonbonded_pair (atom_types, nonbond_pairs,
                                                    sigma, epsilon, chgs)
    ener_nbond_fn = jax.jit(ener_nbond_fn)

    nbonds14 = jax_amber.prm_get_nonbond14_info (prm_raw_data)
    ener_nbond14_fn = jax_amber.ener_nonbonded14 (atom_types, nbonds14,  
                                                sigma, epsilon, chgs)
    ener_nbond14_fn = jax.jit(ener_nbond14_fn)

    def compute_fun (R, vmax0):
        """
        R (natom, 3)
        """
        en_bonded = ener_bonded_fn (R)
        en_lj, en_chg = ener_nbond_fn (R, vmax0)
        en_lj14, en_chg14 = ener_nbond14_fn (R, vmax0)
        
        return en_bonded + en_lj + en_chg + en_lj14 + en_chg14

    return compute_fun, ener_bond_fn


def get_amber_energy_obc2_fun (fname_prmtop):

    prm_raw_data = jax_amber.amber_prmtop_load (fname_prmtop)
    chgs = jax_amber.prm_get_charges (prm_raw_data)
    gb_parms = jax_obc2.prm_get_gb_parms (prm_raw_data)
    ener_obc2_fn, get_born_radii = jax_obc2.ener_gbsa_obc2 (chgs, gb_parms)
    ener_obc2_fn = jax.jit(ener_obc2_fn)
    get_born_radii = jax.jit(get_born_radii)

    def compute_fun (R, born_radii):
        en_obc2 = ener_obc2_fn(R, born_radii)
        return en_obc2

    return compute_fun, get_born_radii


def get_bound_state_fun (fname_com_prmtop, fname_prt_prmtop, fname_lig_prmtop):

    ener_prt_fn, ener_prt_bond_fn = get_amber_energy_gas_fun (fname_prt_prmtop)
    ener_lig_fn, ener_lig_bond_fn = get_amber_energy_gas_fun (fname_lig_prmtop)
    ener_com_obc2_fn, get_born_radii_fn = get_amber_energy_obc2_fun (fname_com_prmtop)

    # PRT
    prm_raw_data = jax_amber.amber_prmtop_load (fname_prt_prmtop)
    nAtoms_prt = int(prm_raw_data['POINTERS'][0])
    # LIG
    prm_raw_data = jax_amber.amber_prmtop_load (fname_lig_prmtop)
    nAtoms_lig = int(prm_raw_data['POINTERS'][0])

    prm_raw_data = jax_amber.amber_prmtop_load (fname_com_prmtop)
    chgs = jax_amber.prm_get_charges (prm_raw_data)
    atom_types = jax_amber.prm_get_atom_types (prm_raw_data)
    sigma, epsilon = jax_amber.prm_get_nonbond_terms (prm_raw_data)
    
    nonbonds = []
    for ia in range (nAtoms_prt):
        for ja in range (nAtoms_lig):
            nonbonds.append ( [ia, ja+nAtoms_prt] )
    nonbonds = jnp.array(nonbonds)

    ener_nbond_fn = jax_amber.ener_nonbonded_pair (atom_types, nonbonds, sigma, epsilon, chgs)
    ener_nbond_fn = jax.jit (ener_nbond_fn)

    def compute_gas_fun (R, vmax0=jnp.float32(500.0)):
        '''
        R: Array (n_atom, 3)
        '''
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]
        en_prt = ener_prt_fn (R_prt, vmax0)
        en_lig = ener_lig_fn (R_lig, vmax0)

        # LJ/Coul energies between prt and lig
        U_lj, U_chg = ener_nbond_fn (R, vmax0)

        return en_prt + en_lig + U_lj + U_chg

  
    def compute_obc2_fun (R, born_radii, 
                          vmax0=jnp.float32(500.0)):
        '''
        R: Array (n_atom, 3)
        '''
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]
        en_prt = ener_prt_fn (R_prt, vmax0)
        en_lig = ener_lig_fn (R_lig, vmax0)

        U_lj, U_chg = ener_nbond_fn (R, vmax0)
        en_obc2 = ener_com_obc2_fn (R, born_radii) 
        
        return en_prt + en_lig + U_lj + U_chg + en_obc2


    def compute_bond_fun (R):
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]
        en_prt = ener_prt_bond_fn (R_prt)
        en_lig = ener_lig_bond_fn (R_lig)
        
        return en_prt + en_lig
    
    
    return (compute_gas_fun, compute_obc2_fun), compute_bond_fun, get_born_radii_fn



def get_bound_state_fun_com (fname_com_prmtop):

    
    ener_com_fn, ener_com_bond_fn = get_amber_energy_gas_fun (fname_com_prmtop)
    ener_com_obc2_fn, get_born_radii_fn = get_amber_energy_obc2_fun (fname_com_prmtop)

    def compute_gas_fun (R, vmax0=jnp.float32(500.0)):
        '''
        R: Array (n_atom, 3)
        '''
        
        return ener_com_fn (R, vmax0)

  
    def compute_obc2_fun (R, born_radii, 
                          vmax0=jnp.float32(500.0)):
        '''
        R: Array (n_atom, 3)
        '''
        
        en_gas = ener_com_fn (R, vmax0)
        en_obc2 = ener_com_obc2_fn (R, born_radii) 
        
        return en_gas + en_obc2


    def compute_bond_fun (R):
        return ener_com_bond_fn (R)
        
    return (compute_gas_fun, compute_obc2_fun), compute_bond_fun, get_born_radii_fn



def get_unbound_state_fun (fname_prt_prmtop, fname_lig_prmtop):

    ener_prt_fn, ener_prt_bond_fn = get_amber_energy_gas_fun (fname_prt_prmtop)
    ener_lig_fn, ener_lig_bond_fn = get_amber_energy_gas_fun (fname_lig_prmtop)

    prm_raw_data = jax_amber.amber_prmtop_load (fname_prt_prmtop)
    nAtoms_prt = int(prm_raw_data['POINTERS'][0])

    ener_prt_obc2_fn, get_prt_born_radii_fn = get_amber_energy_obc2_fun (fname_prt_prmtop)
    ener_lig_obc2_fn, get_lig_born_radii_fn = get_amber_energy_obc2_fun (fname_lig_prmtop)

    def compute_gas_fun (R, vmax0=jnp.float32(500.0)):
        '''
        R: Array (n_atom, 3)
        '''
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]
        return ener_prt_fn (R_prt, vmax0) + ener_lig_fn (R_lig, vmax0)
        
    def compute_obc2_fun (R, born_radii, 
                          vmax0=jnp.float32(500.0)):
        '''
        R: Array (n_atom, 3)
        '''
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]
        born_radii_prt = born_radii[:nAtoms_prt]
        born_radii_lig = born_radii[nAtoms_prt:]
        en_gas = ener_prt_fn (R_prt, vmax0) + ener_lig_fn (R_lig, vmax0) 
        en_obc = ener_prt_obc2_fn (R_prt, born_radii_prt) + \
            ener_lig_obc2_fn(R_lig, born_radii_lig)
        return en_gas + en_obc

    def compute_bond_fun (R):
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]
        return ener_prt_bond_fn(R_prt) + ener_lig_bond_fn (R_lig)
    
    def get_born_radii_fn (R):
        R_prt = R[:nAtoms_prt]
        R_lig = R[nAtoms_prt:]

        born_radii_prt = get_prt_born_radii_fn (R_prt)
        born_radii_lig = get_lig_born_radii_fn (R_lig)

        return jnp.hstack ( (born_radii_prt, born_radii_lig) )

    return (compute_gas_fun, compute_obc2_fun), compute_bond_fun, get_born_radii_fn


def get_bonded_fun (fname_prmtop):
                    
    prm_raw_data = jax_amber.amber_prmtop_load (fname_prmtop)
    ener_bonded_fn, ener_bond_fn = \
         jax_amber.ener_bonded (prm_raw_data)
    ener_bonded_fn = jax.jit (ener_bonded_fn)
    ener_bond_fn = jax.jit (ener_bond_fn)
    
 
    
    def compute_bonded_fun (R):
        '''
        R: Array (n_atom, 3)
        '''
        return ener_bonded_fn (R)
        
    
    def compute_bond_fun (R):
        
        return ener_bond_fn(R) 
    
    return compute_bonded_fun, compute_bond_fun




if __name__ == '__main__':
    import MDAnalysis as mda

    fname_prmtop = 'complex.prmtop'
    fname_pdb = 'complex.pdb'
    fname_dcd = 'traj_complex.dcd'
    fname_prt_prmtop = 'protein.prmtop'
    fname_prt_pdb = 'Sim1/stage3_protein_bnd_500.pdb'
    fname_lig_prmtop = 'ligand.prmtop'
    fname_lig_pdb = 'Sim1/stage3_ligand_bnd_500.pdb'

    ener_bound_state_funs, ener_bond_bnd_fun = \
        get_bound_state_fun (fname_prt_prmtop, fname_lig_prmtop) 

    ener_LJ_fun, ener_LJ_Coul_fun, _ = ener_bound_state_funs

    
    ener_prt_fun, _ = get_amber_energy_gas_fun(fname_prt_prmtop)
    ener_lig_fun, _ = get_amber_energy_gas_fun(fname_lig_prmtop)
    ener_com_fun, _ = get_amber_energy_gas_fun(fname_prmtop)

    
    u = mda.Universe(fname_pdb)
    x_Ai = jnp.array(u.atoms.positions)*jnp.float32(0.1) # A-->nm
    
    en_com = ener_com_fun (x_Ai)
    print ('en_com', en_com/4.184)

    u = mda.Universe(fname_prt_pdb)
    x_prt = jnp.array(u.atoms.positions)*jnp.float32(0.1) # A-->nm
    
    en_prt = ener_prt_fun (x_prt)
    print ('en_prt', en_prt/4.184)
    
    u = mda.Universe(fname_lig_pdb)
    x_lig = jnp.array(u.atoms.positions)*jnp.float32(0.1) # A-->nm
    
    en_lig = ener_lig_fun (x_lig)
    print ('en_lig', en_lig/4.184)

    en_com_LJ = ener_LJ_fun (x_prt, x_lig)
    en_com_LJ_Coul = ener_LJ_Coul_fun (x_prt, x_lig)
    print ('en_com2', en_com_LJ/4.184, en_com_LJ_Coul/4.184)

    
