from typing import Sequence, Optional
import numpy as np
import jax.numpy as jnp
from flax import linen as nn 
from flax.training import train_state 
from flax.linen.initializers import zeros as nn_zeros
from flax.linen.initializers import lecun_normal
import jax, optax, pymbar, sys, pickle

import jax_amber_realnvp as jax_amber

default_kernel_init = lecun_normal()
RT = jnp.float32(8.3144621E-3 * 300.0) 
beta = jnp.float32(1.0)/RT 
nm2ang = jnp.float32(10.0)
ang2nm = jnp.float32(0.1)

class AfflineCoupling(nn.Module):
    input_size: int 
    i_dim: int 
    hidden_layers: int 
    hidden_dim : int 
    fixed_atoms: Sequence[int]

    @nn.compact
    def __call__ (self, inputs, reverse=False):

        fixed_mask = jnp.ones ((self.input_size), dtype=jnp.int32).reshape(-1,3)
        fixed_mask = fixed_mask.at[:,self.i_dim].set(0)
        moved_mask = jnp.int32(1) - fixed_mask
        moved_mask = moved_mask.at[self.fixed_atoms,self.i_dim].set(0)
        moved_mask = moved_mask.reshape (1,-1)
        fixed_mask = fixed_mask.reshape (1,-1)
        y = inputs*fixed_mask
        
        for _ in range (self.hidden_layers):
            y = nn.relu (nn.Dense (features=self.hidden_dim, kernel_init=default_kernel_init) (y))
    
        log_scale = nn.Dense (features=self.input_size, kernel_init=nn_zeros) (y)
        shift     = nn.Dense (features=self.input_size, kernel_init=nn_zeros) (y)
        shift     = shift*moved_mask 
        log_scale = log_scale*moved_mask
        
        if reverse:
            log_scale = -log_scale
            outputs = (inputs-shift)*jnp.exp(log_scale)
        else:
            outputs = inputs*jnp.exp(log_scale) + shift
      
        return outputs, log_scale 


class realNVP3 (nn.Module):
    input_size: int 
    hidden_layers: int 
    hidden_dim : int 
    fixed_atoms: Sequence[int]
    
    def setup (self):
        
        self.af_x = AfflineCoupling (self.input_size, i_dim=0, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)
        self.af_y = AfflineCoupling (self.input_size, i_dim=1, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)
        self.af_z = AfflineCoupling (self.input_size, i_dim=2, 
                                     hidden_layers=self.hidden_layers, 
                                     hidden_dim=self.hidden_dim,
                                     fixed_atoms=self.fixed_atoms)

    @nn.compact
    def __call__ (self, inputs, reverse=False):
        #print(type(inputs), inputs.shape)
        n_conf, n_atoms, n_dim = inputs.shape 
        
        outputs = inputs.reshape (n_conf, -1)
        if reverse:
            outputs, log_J_z = self.af_z (outputs, reverse)
            outputs, log_J_y = self.af_y (outputs, reverse)
            outputs, log_J_x = self.af_x (outputs, reverse)
        else:
            outputs, log_J_x = self.af_x (outputs)
            outputs, log_J_y = self.af_y (outputs)
            outputs, log_J_z = self.af_z (outputs)

        return outputs.reshape(n_conf, n_atoms, n_dim), \
                (log_J_x + log_J_y + log_J_z).sum(axis=-1)


class DoublerealNVP3(nn.Module):
    input_size: int 
    hidden_layers: int 
    hidden_dim : int 
    fixed_atoms: Sequence[int]

    def setup(self):
        self.forward = realNVP3(input_size=self.input_size, hidden_layers=self.hidden_layers,
                                hidden_dim=self.hidden_dim, fixed_atoms=self.fixed_atoms)
        self.reverse = realNVP3(input_size=self.input_size, hidden_layers=self.hidden_layers,
                                hidden_dim=self.hidden_dim, fixed_atoms=self.fixed_atoms)

    @nn.compact
    def __call__(self, inputs):
        """For inputs (X_a, X_b)"""
        m_B, log_J_F = self.forward(inputs[0])
        m_A, log_J_R = self.reverse(inputs[1])
        return m_B, log_J_F, m_A, log_J_R


def checkpoint_save (fname, ckpt):
    with open(fname, 'wb') as fp:        
        pickle.dump (ckpt, fp)

def checkpoint_load (fname):
    with open (fname, 'rb') as fp:
        return pickle.load (fp)

def get_energy_values (x, ener_funs, R0):
    ener_nHO_fun, ener_wHO_fun, ener_bond_fun = ener_funs 
    enr_bnd = jax.vmap(ener_bond_fun) (x)
    enr_nHO = jax.vmap(ener_nHO_fun) (x)
    enr_wHO = jax.vmap(ener_wHO_fun, in_axes=(0,None)) (x, R0)
    return enr_bnd, enr_nHO, enr_wHO



def print_progress (state, inputs, ener_funs, ener_ref0, fixed_iatom, fixed_R0, fout):
    
    R0_A, R0_B, dE0 = fixed_R0
    (enr_nHO_A0, enr_nHO_B0), (enr_wHO_A0, enr_wHO_B0,_,_) = ener_ref0
    m_B, log_J_F, m_A, log_J_R = state.apply_fn({'params':state.params}, inputs) #Mapped

    enr_bond_A, enr_nHO_A, enr_wHO_A = get_energy_values (m_A, ener_funs, R0_A)
    enr_bond_B, enr_nHO_B, enr_wHO_B = get_energy_values (m_B, ener_funs, R0_B)
    
    dU_F = enr_wHO_B - enr_wHO_A0
    dU_R = enr_wHO_A - enr_wHO_B0 
    phi_F = beta*dU_F - log_J_F
    phi_R = beta*dU_R - log_J_R 

    Z_A = m_A[:,fixed_iatom,2].mean()
    Z_B = m_B[:,fixed_iatom,2].mean()

    print ('R_A R_B dE          {:12.6f} {:12.6f} {:8.4f}'.format (R0_A[-1], R0_B[-1], dE0), file=fout)
    print ('Fixed_Z             {:12.6f} {:12.6f}'.format(Z_A, Z_B), file=fout)
    print ('<-log_J>(kJ/mol)    {:12.6f} {:12.6f}'.format(RT*(-log_J_F-log_J_R).mean(), -RT*log_J_F.mean()), file=fout)
    print (' <U_bond>(kJ/mol)   {:12.6f} {:12.6f}'.format (enr_bond_A.mean(), enr_bond_B.mean()), file=fout)
    print (' <U_wHO>(kJ/mol)    {:12.6f} {:12.6f}'.format (enr_wHO_A.mean(), enr_wHO_B.mean()), file=fout)
    print ('<dU_wHO>(kJ/mol)    {:12.6f} {:12.6f}'.format (dU_R.mean(), dU_F.mean()), file=fout)
    print ('<phi_wHO>(kJ/mol)   {:12.6f} {:12.6f}'.format (RT*phi_R.mean(), RT*phi_F.mean()), file=fout) 

    f_BAR_wHO = pymbar.BAR (phi_F, phi_R, 
                        relative_tolerance=1.0e-5,
                        verbose=False,
                        compute_uncertainty=False)
    print ('LBAR(kJ/mol)        {:12.6f}'.format ( RT*f_BAR_wHO ), file=fout)
    return RT*f_BAR_wHO #LBAR in KJ/mol

    


def loss_value (ener_wHO_fn, ener_bond_fn, enr0_wHO, m_B, log_J_F, m_A, log_J_R, fixed_R0):

    enr_wHO_A0, enr_wHO_B0, enr_bnd_A0, enr_bnd_B0 = enr0_wHO
    R0_A, R0_B, dE0 = fixed_R0

    enr_A = jax.vmap(ener_wHO_fn, in_axes=(0,None)) (m_A, R0_A)
    enr_B = jax.vmap(ener_wHO_fn, in_axes=(0,None)) (m_B, R0_B)
    enr_bnd_A = jax.vmap(ener_bond_fn) (m_A)
    enr_bnd_B = jax.vmap(ener_bond_fn) (m_B)

    loss_F = beta*(enr_B - enr_wHO_A0) - log_J_F 
    loss_R = beta*(enr_A - enr_wHO_B0) - log_J_R
    diff_bnd_A = beta*(enr_bnd_A.mean() - enr_bnd_A0)
    diff_bnd_B = beta*(enr_bnd_B.mean() - enr_bnd_B0)
    diff_bnd_A2 = diff_bnd_A**2
    diff_bnd_B2 = diff_bnd_B**2
    loss = loss_F.mean() + loss_R.mean() 
    loss_wBnd = loss + diff_bnd_A2 + diff_bnd_B2 #+ (log_J_F.mean()+log_J_R.mean())**2
    return loss_wBnd, loss


def write_traj (fname, traj_xyz_nm):
    from mdtraj.formats import DCDTrajectoryFile

    traj_xyz = traj_xyz_nm*10
    n_conf = traj_xyz.shape[0]
    if traj_xyz.shape[-1] != 3:
        traj_xyz = traj_xyz.reshape(n_conf, -1, 3)
    
    with DCDTrajectoryFile(fname, 'w') as f:
        f.write (traj_xyz)


def get_trajectory (fname_prmtop, fname_dcd, nsamp):
    import mdtraj as md

    c = md.load (fname_dcd, top=fname_prmtop)
    crds = jnp.array (c.xyz)
    return crds[-nsamp:], crds[:-nsamp] # in nm unit

def main_train(json_data):
    
    fout = open (json_data['fname_log'], 'w', 1)
    x_A, tx_A = get_trajectory(json_data['fname_prmtop'], json_data['fname_dcd_A'], json_data['nsamp'])
    x_B, tx_B = get_trajectory(json_data['fname_prmtop'], json_data['fname_dcd_B'], json_data['nsamp'])
    inputs, test_inputs = (x_A, x_B), (tx_A, tx_B)
    nconf = x_A.shape[0]

    fixed_atoms = jnp.array (json_data['fixed']['atoms']) - 1
    R0_A = jnp.array (json_data['fixed']['R0_A'])
    R0_B = jnp.array (json_data['fixed']['R0_B'])
    #dF0  = jnp.float32 (json_data['fixed']['dF0'])
    kval = jnp.float32 (json_data['fixed']['kval'])
    dR0_AB = R0_B - R0_A
    d_lam = json_data['d_lambda']

    fixed_iatom = fixed_atoms[-1]
    
    ener_funs = jax_amber.get_amber_energy_funs(json_data['fname_prmtop'], fixed_iatom, kval)
    ener_nHO_fun, ener_wHO_fun, ener_bond_fun = ener_funs 
    enr_bnd_A0, enr_nHO_A0, enr_wHO_A0 = get_energy_values(x_A, ener_funs, R0_A)
    enr_bnd_B0, enr_nHO_B0, enr_wHO_B0 = get_energy_values(x_B, ener_funs, R0_B)
    ###(TESTING)
    _, _, enr_wHO_A0_test = get_energy_values(tx_A, ener_funs, R0_A)
    _, _, enr_wHO_B0_test = get_energy_values(tx_B, ener_funs, R0_B)
    ###
    dE0 = (enr_wHO_B0-enr_wHO_A0).mean()
    Z_A = x_A[:,fixed_iatom,2].mean()
    Z_B = x_B[:,fixed_iatom,2].mean()
    enr_bnd_A0 = enr_bnd_A0.mean()
    enr_bnd_B0 = enr_bnd_B0.mean()
    print (' Fixed_Z:           {:12.6f} {:12.6f}'.format(Z_A, Z_B), file=fout)
    print (' <U_wHO0>(kJ/mol):  {:12.6f} {:12.6f}'.format (enr_wHO_A0.mean(), enr_wHO_B0.mean()), file=fout)
    print (' <U_nHO0>(kJ/mol):  {:12.6f} {:12.6f}'.format (enr_nHO_A0.mean(), enr_nHO_B0.mean()), file=fout)
    print ('<dU>[w/no](kJ/mol): {:12.6f} {:12.6f}'.format( dE0, (enr_nHO_B0-enr_nHO_A0).mean() ), file=fout)
    print ('<enr_bond>(kJ/mol): {:12.6f} {:12.6f}'.format(enr_bnd_A0, enr_bnd_B0), file=fout)

    ener_ref0 = (enr_nHO_A0, enr_nHO_B0), (enr_wHO_A0, enr_wHO_B0, enr_bnd_A0, enr_bnd_B0 )
    ener_wHO_ref0_test = (enr_wHO_A0_test, enr_wHO_B0_test, enr_bnd_A0, enr_bnd_B0 )
    
    lr = json_data['optax']['learning_rate']
    total_steps = json_data['optax']['total_steps']
    alpha = json_data['optax']['alpha']
    scheduler = optax.cosine_decay_schedule(lr, decay_steps=total_steps, alpha=alpha)
    #opt_method = optax.adam (learning_rate=scheduler)
    opt_method = optax.chain(optax.clip(1.0), optax.adam(learning_rate=scheduler))

    rng = jax.random.PRNGKey(json_data['random_seed'])
    rng, key = jax.random.split(rng)
    
    input_size = x_A.shape[1]*3
    hidden_dim = json_data['realNVP']['hidden_dim']
    hidden_layers=json_data['realNVP']['hidden_layers']
    mask_fixed = jnp.array(json_data['realNVP']['mask_fixed']) - 1
    
    model = DoublerealNVP3(input_size=input_size, hidden_layers=hidden_layers, hidden_dim=hidden_dim, fixed_atoms=mask_fixed)
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=model.init(key, inputs)['params'], tx=opt_method)
    
    lam_max = jnp.float32(1.0)
    lam = lam_max
    
    @jax.jit
    def train_step(state, inputs, ener_wHO_ref0, fixed_R0):
        def loss_fn (params, apply_fn):
            m_B, log_J_F, m_A, log_J_R = apply_fn({'params':params}, inputs) #Mapped
            loss_wBnd, loss = loss_value(ener_wHO_fun, ener_bond_fun, ener_wHO_ref0, m_B, log_J_F, m_A, log_J_R, fixed_R0)
            return loss_wBnd
        grads = jax.grad(loss_fn)(state.params, state.apply_fn)
        return state.apply_gradients (grads=grads)

    R_A = R0_B - lam*dR0_AB 
    R_B = R0_A + lam*dR0_AB
    dE  = lam*dE0
    fixed_R0 = (R_A, R_B, dE)

    loss_old = 0.0
    loss_test_min = float('inf') #1000.0
    loss_test_list = []
    for epoch in range (json_data['nepoch']):
   
        for ist0 in range (0,nconf,1000):
            ied0 = ist0 + 1000
            ied0 = jnp.where (ied0 < nconf, ied0, nconf)
            batch = (x_A[ist0:ied0], x_B[ist0:ied0])
            ener_wHO_ref0 = (enr_wHO_A0[ist0:ied0], enr_wHO_B0[ist0:ied0], enr_bnd_A0, enr_bnd_B0)
            state = train_step (state, batch, ener_wHO_ref0, fixed_R0)
        
        if (epoch+1)%10 == 0: #m_B, log_J_F, m_A, log_J_R
            m_B, log_J_F, m_A, log_J_R = state.apply_fn ({'params':state.params}, inputs)
            loss_Wbnd, loss = loss_value (ener_wHO_fun, ener_bond_fun, ener_ref0[1], m_B, log_J_F, m_A, log_J_R, fixed_R0)
            diff = loss_Wbnd - loss_old 
            loss_old = loss_Wbnd

            m_B, log_J_F, m_A, log_J_R = state.apply_fn ({'params':state.params}, test_inputs)
            _, loss_test = loss_value (ener_wHO_fun, ener_bond_fun, ener_wHO_ref0_test, m_B, log_J_F, m_A, log_J_R, fixed_R0)
            print ('loss {:8d} {:12.4f} {:12.4f} {:12.4f} {:14.4f}'.format(epoch+1, loss, loss_Wbnd, diff, loss_test),file=fout)
            
            if loss_test < loss_test_min:
                loss_test_min = loss_test 
                test_ckpt = {'params': state.params, 'opt_state':state.opt_state}
            if loss < jnp.float32(0.0):
                break 

            loss_test_list.append (loss_test)
            
        
        if (epoch+1)%200 == 0:
            _ = print_progress(state, inputs, ener_funs, ener_ref0, fixed_iatom, fixed_R0, fout)
            loss_test = jnp.array (loss_test_list).min()
            print ('loss_test_min', loss_test, loss_test_min, file=fout)
            if loss_test > loss_test_min + 3.0:
                break
            loss_test_list = []
        
        
    ckpt = {'params': state.params, 'opt_state': state.opt_state, 'lam': lam}
    checkpoint_save(json_data['fname_nn_pkl'], ckpt)
    checkpoint_save(json_data['fname_nn_test_pkl'], test_ckpt)
    state = state.replace(step=0, params=test_ckpt['params'], opt_state=test_ckpt['opt_state'])
    print("===SUMMARY===", file=fout)
    return print_progress(state, inputs, ener_funs, ener_ref0, fixed_iatom, fixed_R0, fout)