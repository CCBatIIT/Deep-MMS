from tqdm import tqdm
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import mdtraj as md
from MDAnalysis.analysis.bat import BAT
import MDAnalysis as mda
from dataclasses import field
import dataclasses
import matplotlib.pyplot as plt
import functools
import dill
import os



def save_checkpoint(generator_state, discriminator_state, epoch, file_name):
    generator_checkpoint = {
        "params": generator_state.params,
        "opt_state": generator_state.opt_state,
    }

    discriminator_checkpoint = {
        "params": discriminator_state.params,
        "opt_state": discriminator_state.opt_state,
    }
    checkpoint = {
        "generator_state": generator_checkpoint,
        "discriminator_state": discriminator_checkpoint,
        "epoch": epoch,
    }
    with open(file_name, "wb") as f:
        dill.dump(checkpoint, f)
    print(f"Checkpoint saved at epoch {epoch} to file: {file_name}")

def load_checkpoint(file_name):
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
        print("No valid checkpoint file found. Starting training from scratch.")
        return None, None, 0

    try:
        with open(file_name, 'rb') as f:
            checkpoint = dill.load(f)

        # Reconstruct generator state
        generator_state = TrainState.create(
            apply_fn=generator.apply,
            params=checkpoint['generator_state']['params'],
            tx=optax.adam(learning_rate=5e-6)
        ).replace(opt_state=checkpoint['generator_state']['opt_state'])

        # Reconstruct discriminator state
        discriminator_state = TrainState.create(
            apply_fn=discriminator.apply,
            params=checkpoint['discriminator_state']['params'],
            tx=optax.adam(learning_rate=5e-6)
        ).replace(opt_state=checkpoint['discriminator_state']['opt_state'])

        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from file: {file_name}, resuming at epoch {start_epoch}")
        return generator_state, discriminator_state, start_epoch

    except Exception as e:
        print(f"Failed to load checkpoint due to error: {e}")
        return None, None, 0

 #--- Energy Functions ---
def ener_bond(bond_lengths, bond_eq_lengths, bond_k):
    return 0.5 * bond_k * (bond_lengths - bond_eq_lengths) ** 2

def ener_angle(angles, angle_eq, angle_k):
    angle_eq_broadcast = angle_eq[:, np.newaxis]
    angle_k_broadcast = angle_k[:, np.newaxis]
    return 0.5 * angle_k_broadcast * (angles - angle_eq_broadcast) ** 2


def ener_torsion(torsions, torsion_k, torsion_n, torsion_phi_eq):
    # Ensure shapes align for broadcasting
    torsion_k = jnp.resize(torsion_k, (torsions.shape[0], 1))
    torsion_n = jnp.resize(torsion_n, (torsions.shape[0], 1))
    torsion_phi_eq = jnp.resize(torsion_phi_eq, (torsions.shape[0], 1))

    # Compute torsion energy
    return torsion_k * (1 + jnp.cos(torsion_n * torsions - torsion_phi_eq))



# --- Nonbonded Interactions ---
def nonbonded_LJ(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def nonbonded_Coul(q1, q2, r, epsilon=1.0):
    return q1 * q2 / (4 * jnp.pi * epsilon * r)


# --- Force Field Parameters ---
def amber_prmtop_load(prmtop_filename):

    return {
        "bond_eq_lengths": np.random.random(100),
        "bond_k": np.random.random(100),
        "angle_eq": np.random.random(50),
        "angle_k": np.random.random(50),
        "torsion_k": np.random.random(30),
        "torsion_n": np.random.randint(1, 4, 30),
        "torsion_phi_eq": np.random.random(30),
        "charges": np.random.random(100),
        "lj_epsilon": np.random.random(100),
        "lj_sigma": np.random.random(100),
    }


# --- Restraint Functions ---
def restraint_fun(positions, target_positions, weights):
    return weights * jnp.sum((positions - target_positions) ** 2)

def reparameterize(z_rng, z_mean, z_logvar):
    z_std = jnp.exp(0.5 * z_logvar)
    z_eps = jax.random.normal(z_rng, shape=z_mean.shape)
    return z_mean + z_eps * z_std

def detect_bonds(positions, bond_threshold=1.6, min_threshold=1.0):
    bonds = []
    num_atoms = positions.shape[0]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            if min_threshold <= distance <= bond_threshold:
                bonds.append([i, j])
    return np.array(bonds)

def detect_torsion_bonds(bonds):
    """
    Detect torsion bonds (sets of 4 connected atoms) from the bonds array.
    Args:
        bonds: Array of pairs of connected atoms.
    Returns:
        torsion_bonds: Array of torsion bonds, each with 4 indices.
    """
    torsion_bonds = []
    bond_dict = {i: set() for i in np.unique(bonds)}
    for bond in bonds:
        bond_dict[bond[0]].add(bond[1])
        bond_dict[bond[1]].add(bond[0])

    for atom1 in bond_dict:
        for atom2 in bond_dict[atom1]:
            for atom3 in bond_dict[atom2] - {atom1}:
                for atom4 in bond_dict[atom3] - {atom2}:
                    torsion_bonds.append([atom1, atom2, atom3, atom4])
    return np.array(torsion_bonds)

def gradient_penalty(discriminator, real_data, fake_data, discriminator_params):
    # Generate random interpolation weights (same batch size as real_data)
    batch_size = real_data.shape[0]
    epsilon = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, 1, 1))

    # Ensure fake_data and real_data have the same shape
    if real_data.shape != fake_data.shape:
        print(f"⚠️ Shape mismatch detected in gradient_penalty! real_data: {real_data.shape}, fake_data: {fake_data.shape}")
        min_batch_size = min(real_data.shape[0], fake_data.shape[0])
        real_data = real_data[:min_batch_size]
        fake_data = fake_data[:min_batch_size]
        print(f"✅ Adjusted shapes -> real_data: {real_data.shape}, fake_data: {fake_data.shape}")

    # Interpolate between real and fake data
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data  # Now correctly broadcasted

    # Compute gradients
    grads = jax.grad(lambda x: jnp.sum(discriminator.apply({'params': discriminator_params}, x)))(interpolated)

    # Compute the norm of the gradients
    grad_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=(1, 2)) + 1e-6)  # Added small constant

    # Calculate gradient penalty
    return jnp.mean((grad_norm - 1) ** 2)


def normalize_coordinates(data, reference_scale=None):
    """
    Normalize coordinates by centering and scaling.
    If reference_scale is given, use it to avoid mismatched scale in generated structures.
    """
    center = jnp.mean(data, axis=0)
    data = data - center

    # Use a fixed reference scale if provided, else compute from data
    if reference_scale is None:
        scale = jnp.linalg.norm(data, axis=1).max()
    else:
        scale = reference_scale

    normalized = data / scale
    return normalized, scale


# Structural constraint: Bond length
def bond_length_constraint(fake_data, bonds, real_bond_lengths_batch):
    fake_bond_lengths = jnp.linalg.norm(fake_data[:, bonds[:, 0]] - fake_data[:, bonds[:, 1]], axis=-1)
    bond_length_loss = jnp.mean((fake_bond_lengths - real_bond_lengths_batch) ** 2)
    return bond_length_loss



def torsional_angle_constraint(real_data, fake_data, bonds):
    real_angles = calculate_torsional_angles(real_data, bonds)
    fake_angles = calculate_torsional_angles(fake_data, bonds)
    angle_loss = jnp.mean((real_angles - fake_angles) ** 2)
    return angle_loss


# Structural constraint: Centering the structure
def center_of_mass_constraint(fake_data):
    center_of_mass = jnp.mean(fake_data, axis=1)
    com_loss = jnp.mean(jnp.sum(center_of_mass ** 2, axis=-1))
    return com_loss

def structure_similarity_loss(real_data, fake_data):
    """
    Align the generated data to the real data before computing similarity loss.
    """
    # Ensure batch sizes match before alignment
    if real_data.shape[0] != fake_data.shape[0]:
        print(f"⚠️ Shape mismatch detected! real_data: {real_data.shape}, fake_data: {fake_data.shape}")
        min_batch_size = min(real_data.shape[0], fake_data.shape[0])
        real_data = real_data[:min_batch_size]
        fake_data = fake_data[:min_batch_size]
        print(f"✅ Adjusted shapes -> real_data: {real_data.shape}, fake_data: {fake_data.shape}")

    aligned_fake_data, _ = align_structures(real_data, fake_data)  # Unpack the aligned data
    similarity_loss = jnp.mean((real_data - aligned_fake_data) ** 2)
    return similarity_loss



def latent_space_regularization(fake_data, latent_samples):
    coverage_loss = jnp.mean(jnp.std(latent_samples, axis=0))
    return coverage_loss

def torsion_similarity_loss(fake_torsions, real_torsions):
    delta = jnp.abs(fake_torsions - real_torsions)
    angular_diff = 180 - jnp.abs(delta - 180)
    return jnp.mean(angular_diff ** 2)


def generator_loss(fake_output, real_data, fake_data, bonds, real_bond_lengths, epoch, num_epochs, latent_samples, torsion_bonds, real_angles, params):
    adversarial_loss = -jnp.mean(jnp.log(jnp.clip(fake_output, 1e-6, 1.0 - 1e-6)))

    # Calculate bond lengths and angles
    bond_lengths = jnp.linalg.norm(fake_data[:, bonds[:, 0]] - fake_data[:, bonds[:, 1]], axis=-1)
    angle_values = calculate_angles(fake_data, angle_bonds)
    real_angle_values = calculate_angles(real_data, angle_bonds)

    # Calculate torsions (fake and real)
    fake_torsions = calculate_torsional_angles(fake_data, torsion_bonds)
    real_torsions = calculate_torsional_angles(real_data, torsion_bonds)

    # Energy using updated similarity constraints
    bond_energy = ener_bond(bond_lengths, real_bond_lengths, params['bond_k'])
    angle_energy = ener_angle(angle_values, real_angle_values, params['angle_k'])
    torsion_energy = torsion_similarity_loss(fake_torsions, real_torsions)

    # Structural similarity and regularization
    similarity_loss = structure_similarity_loss(real_data, fake_data)
    similarity_weight = 1.0 + 9.0 * (epoch / num_epochs)
    torsional_weight = 20
    latent_regularization = latent_space_regularization(fake_data, latent_samples)

    total_loss = (
        adversarial_loss
        + bond_energy.mean()
        + angle_energy.mean()
        + torsional_weight * torsion_energy
        + similarity_weight * similarity_loss
        + latent_regularization
    )
    return total_loss


def form_angle_bonds(bonds):
    """
    Create angle bonds (triplets) from a list of bonds (pairs).
    Args:
        bonds: Array of pairs of connected atoms.
    Returns:
        angle_bonds: Array of triplets of atoms forming angles.
    """
    angle_bonds = []
    bond_dict = {i: set() for i in np.unique(bonds)}
    for bond in bonds:
        bond_dict[bond[0]].add(bond[1])
        bond_dict[bond[1]].add(bond[0])

    for atom1 in bond_dict:
        for atom2 in bond_dict[atom1]:
            for atom3 in bond_dict[atom2]:
                if atom3 != atom1:  # Avoid duplicate or invalid triplets
                    angle_bonds.append([atom1, atom2, atom3])
    return np.array(angle_bonds)


def calculate_torsional_angles(data, torsion_bonds):
    """
    Calculate torsional angles for a given structure and torsion bonds.
    Args:
        data: Array of atomic coordinates, shape (num_atoms, 3).
        torsion_bonds: Array of torsion bond groups, shape (num_torsions, 4).
    Returns:
        Array of torsional angles.
    """
    if torsion_bonds.shape[1] != 4:
        raise ValueError("Torsion bonds must have 4 indices per torsion.")

    v1 = data[torsion_bonds[:, 1]] - data[torsion_bonds[:, 0]]
    v2 = data[torsion_bonds[:, 2]] - data[torsion_bonds[:, 1]]
    v3 = data[torsion_bonds[:, 3]] - data[torsion_bonds[:, 2]]

    n1 = jnp.cross(v1, v2)
    n2 = jnp.cross(v2, v3)

    n1 = n1 / jnp.linalg.norm(n1, axis=1, keepdims=True)
    n2 = n2 / jnp.linalg.norm(n2, axis=1, keepdims=True)

    cos_theta = jnp.sum(n1 * n2, axis=1)
    torsional_angles = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))

    return torsional_angles


def calculate_angles(data, angle_bonds):
    """
    Calculate angles for a given structure and angle bonds.
    Args:
        data: Array of atomic coordinates, shape (num_atoms, 3).
        angle_bonds: Array of angle bond groups, shape (num_angles, 3).
    Returns:
        Array of angles in radians.
    """
    if angle_bonds.shape[1] != 3:
        raise ValueError("Angle bonds must have 3 indices per angle.")

    v1 = data[angle_bonds[:, 1]] - data[angle_bonds[:, 0]]
    v2 = data[angle_bonds[:, 1]] - data[angle_bonds[:, 2]]

    # Normalize vectors
    v1 = v1 / jnp.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / jnp.linalg.norm(v2, axis=1, keepdims=True)

    # Calculate cosine of the angle
    cos_theta = jnp.sum(v1 * v2, axis=1)
    angles = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))

    return angles

def align_structures(ref, target):
    """
    Aligns two structures using the Kabsch algorithm with a fallback for SVD using eigen decomposition.
    Args:
        ref: Reference structure (shape: [batch_size, num_atoms, 3]).
        target: Target structure (shape: [batch_size, num_atoms, 3]).
    Returns:
        Aligned target structure, RMSD penalty.
    """
    ref_center = jnp.mean(ref, axis=1, keepdims=True)
    target_center = jnp.mean(target, axis=1, keepdims=True)

    ref_centered = ref - ref_center
    target_centered = target - target_center

    print(f"Shape of ref_centered: {ref_centered.shape}")
    print(f"Shape of target_centered: {target_centered.shape}")

    covariance_matrix = jnp.einsum('bij,bik->bjk', ref_centered, target_centered)

    u, _, vh = jnp.linalg.svd(covariance_matrix, full_matrices=False)
    rotation_matrix = jnp.matmul(u, vh)

    det = jnp.linalg.det(rotation_matrix)
    adjust_sign = jnp.where(det < 0, -1.0, 1.0)
    u = u.at[:, :, -1].set(u[:, :, -1] * adjust_sign[:, None])
    rotation_matrix = jnp.matmul(u, vh)

    aligned_target = jnp.einsum('bij,bnj->bni', rotation_matrix, target_centered) + ref_center

    # RMSD calculation
    rmsd = jnp.sqrt(jnp.mean((aligned_target - ref) ** 2))
    return aligned_target, rmsd



def write_traj(identifier, traj_xyz, model_name, n_latents, file_name):
    if traj_xyz.shape[-1] != 3:
        traj_xyz = traj_xyz.reshape(traj_xyz.shape[0], -1, 3)
    with md.formats.DCDTrajectoryFile(file_name, 'w') as f:
        f.write(traj_xyz)
    print(f"Trajectory saved to {file_name}")


# BAT_jax class for BAT coordinate transformation
class BAT_jax(BAT):
    def __init__(self, ag, initial_atom=None, filename=None, **kwargs):
        super(BAT, self).__init__(ag.universe.trajectory, **kwargs)
        self._ag = ag
        if not hasattr(self._ag, 'bonds'):
            raise AttributeError('AtomGroup has no attribute bonds')
        if len(self._ag.fragments) > 1:
            raise ValueError('AtomGroup has more than one molecule')
        terminal_atoms = sorted([a for a in self._ag.atoms if len(a.bonds) == 1], key=lambda x: x.mass, reverse=True)
        initial_atom = terminal_atoms[0] if initial_atom is None else initial_atom
        if initial_atom not in terminal_atoms:
            raise ValueError('Initial atom is not a terminal atom')
        second_atom = initial_atom.bonded_atoms[0]
        third_atom = sorted(
            [a for a in second_atom.bonded_atoms if a != initial_atom and len(a.bonded_atoms) > 1],
            key=lambda x: x.mass,
            reverse=True
        )[0]
        self._root = mda.AtomGroup([initial_atom, second_atom, third_atom])
        self._torsions = self._find_torsions(self._root, self._ag)

    def _find_torsions(self, root, atoms):
        torsions = []
        selected_atoms = list(root)
        while len(selected_atoms) < len(atoms):
            for a1 in selected_atoms:
                a0_list = sorted([a for a in a1.bonded_atoms if a not in selected_atoms], key=lambda x: x.mass)
                for a0 in a0_list:
                    a2_list = sorted([a for a in a1.bonded_atoms if a != a0 and a in selected_atoms],
                                     key=lambda x: x.mass)
                    for a2 in a2_list:
                        a3_list = sorted([a for a in a2.bonded_atoms if a != a1 and a in selected_atoms],
                                         key=lambda x: x.mass)
                        for a3 in a3_list:
                            torsions.append(mda.AtomGroup([a0, a1, a2, a3]))
                            selected_atoms.append(a0)
                            break
        return torsions

# Data Loading and Preprocessing with BAT Conversion Option
def load_md_data(dcd_filename, prmtop_filename, coord_type="Cartesian", test_split=0.2):
    u = mda.Universe(prmtop_filename, dcd_filename)
    ag = u.select_atoms("all")
    if coord_type == "BAT":
        bat_converter = BAT_jax(ag)
        bat_converter.run()
        coord_set = jnp.array(bat_converter.results.bat)
    else:
        coord_set = np.array([u.trajectory.ts.positions.reshape(-1) for ts in u.trajectory])
    coord_set = jnp.array(coord_set)
    num_samples, input_size = coord_set.shape
    np.random.seed(42)
    test_indices = np.random.choice(num_samples, size=int(test_split * num_samples), replace=False)
    train_indices = np.array([i for i in range(num_samples) if i not in test_indices])
    train_data = coord_set[train_indices]
    test_data = coord_set[test_indices]
    return train_data, test_data, input_size

class TrainState(train_state.TrainState):
    pass  # No need for batch_stats if not using BatchNorm


# Transformer Model
class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = jnp.array(pe)  # (max_len, d_model)

    def __call__(self, x):
        """
        Apply positional encoding to the input tensor.

        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape  # Ensure the expected shape

        # Ensure positional encoding has the correct shape
        pe_broadcasted = self.pe[:seq_len, :].reshape(1, seq_len, d_model)  # (1, seq_len, d_model)
        pe_broadcasted = jnp.broadcast_to(pe_broadcasted, (batch_size, seq_len, d_model))  # (batch_size, seq_len, d_model)

        return x + pe_broadcasted  # Ensure correct shape for addition



class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    dim_feedforward: int = 2048
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool):
        attn_output = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x, x, x)
        x = nn.LayerNorm()(x + attn_output)

        feedforward = nn.Sequential([
            nn.Dense(self.dim_feedforward),
            nn.relu,
            nn.Dense(self.d_model)
        ])
        x = nn.LayerNorm()(x + feedforward(x))
        return x

class TransformerModel(nn.Module):
    input_size: int
    num_layers: int
    num_heads: int
    d_model: int
    dim_feedforward: int = 2048
    dropout: float = 0.1

    def setup(self):
        self.embedding = nn.Dense(self.d_model)  # Converts input to (batch_size, d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        self.transformer_blocks = [
            TransformerBlock(self.d_model, self.num_heads, self.dim_feedforward, self.dropout)
            for _ in range(self.num_layers)
        ]
        self.output_layer = nn.Dense(self.input_size)

    def __call__(self, x, train: bool):
        """ Forward pass through Transformer Model """

        if len(x.shape) == 2:  #  Handle 2D input case
            batch_size, input_size = x.shape
            seq_len = input_size // self.d_model

            if seq_len * self.d_model != input_size:
                raise ValueError(f"Input size {input_size} must be divisible by d_model {self.d_model}")

            x = x.reshape(batch_size, seq_len, self.d_model)  #  Correctly reshaping to 3D

        elif len(x.shape) == 3:
            batch_size, seq_len, d_model = x.shape
            if d_model != self.d_model:
                raise ValueError(f"Expected d_model={self.d_model}, but got {d_model}")

        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")  # Catch any wrong shape

        x = self.embedding(x)  # Expected shape: (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)  # Apply positional encoding

        for block in self.transformer_blocks:
            x = block(x, train=train)

        x = self.output_layer(x)  # Expected output shape: (batch_size, seq_len, input_size)
        return x


# Discriminator Model
@dataclasses.dataclass
class Discriminator(nn.Module):
    input_size: int
    d_hidden: list = field(default_factory=lambda: [256, 128, 64, 32])
    dropout_rates: list = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.4])

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.input_size)
        for i in range(len(self.d_hidden)):
            x = nn.relu(nn.Dense(self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        return nn.sigmoid(nn.Dense(1)(x))


# Discriminator Loss Functions
def discriminator_loss(real_output, fake_output, real_data, fake_data, discriminator_params):
    real_loss = -jnp.mean(jnp.log(jnp.clip(real_output, 1e-6, 1.0 - 1e-6)))
    fake_loss = -jnp.mean(jnp.log(jnp.clip(1 - fake_output, 1e-6, 1.0 - 1e-6)))
    gp = gradient_penalty(discriminator, real_data, fake_data, discriminator_params)
    return real_loss + fake_loss + 10 * gp


# TrainState class
from flax.training import train_state

class TrainState(train_state.TrainState):
    pass  # Remove batch_stats



# Training Step with Constraints
@functools.partial(jax.jit, static_argnums=(9, 10))
def train_step(generator_state, discriminator_state, batch, params, bonds, real_bond_lengths_batch, epoch, num_epochs, real_angles, latent_dim, torsion_bonds, z_rng):
    dropout_rng_key, z_rng = jax.random.split(z_rng)
    torsion_bonds = np.array(torsion_bonds)

    #  Sample latent vectors and reshape to match (batch_size, seq_len, d_model)
    latent_samples = jax.random.normal(z_rng, shape=(batch.shape[0], latent_dim))
    generator_input = latent_samples.reshape(batch.shape[0], latent_dim // 102, 102)

    def loss_fn(generator_params, discriminator_params, batch):
        # Forward pass through generator using latent input
        fake_data = generator.apply({'params': generator_params}, generator_input, train=True)
        fake_data = fake_data.reshape(-1, input_size // 3, 3)
        batch = batch.reshape(-1, input_size // 3, 3)

        # Discriminator output
        fake_output = discriminator.apply({'params': discriminator_params}, fake_data.reshape(-1, input_size))
        real_output = discriminator.apply({'params': discriminator_params}, batch.reshape(-1, input_size))

        # Compute generator and discriminator loss
        g_loss = generator_loss(
            fake_output,
            batch,
            fake_data,
            bonds,
            real_bond_lengths_batch,
            epoch,
            num_epochs,
            latent_samples,
            torsion_bonds,
            real_angles,
            params
        )

        d_loss = discriminator_loss(real_output, fake_output, batch, fake_data, discriminator_params)

        total_loss = g_loss + d_loss
        return total_loss, (g_loss, d_loss)

    # Backpropagation
    (total_loss, aux_outputs), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
        generator_state.params, discriminator_state.params, batch
    )
    g_loss, d_loss = aux_outputs

    # Apply gradients
    generator_state = generator_state.apply_gradients(grads=grads[0])
    discriminator_state = discriminator_state.apply_gradients(grads=grads[1])


    angle_loss = 0.0
    torsion_loss = 0.0

    return generator_state, discriminator_state, g_loss, d_loss, angle_loss, torsion_loss





# Generate probabilities
def get_generated_data_probabilities(generator, discriminator, generator_state, discriminator_state, num_samples, latent_dim):
    z_rng = jax.random.PRNGKey(42)
    z_samples = jax.random.normal(z_rng, shape=(num_samples, latent_dim))
    generator_variables = {'params': generator_state.params, 'batch_stats': generator_state.batch_stats}
    fake_data, _, _ = generator.apply(generator_variables, z_samples, z_rng, rngs={'dropout': z_rng}, train=False)
    fake_probabilities = discriminator.apply({'params': discriminator_state.params}, fake_data)
    return fake_data, fake_probabilities


# Reshape to 3D
def reshape_to_3d_structure(data):
    """
    Reshapes a dataset to 3D structure if necessary.
    """
    if len(data.shape) == 2:  # Case: (num_samples, input_size)
        num_samples, input_size = data.shape
        assert input_size % 3 == 0, "Input size must be divisible by 3 for 3D visualization."
        num_atoms = input_size // 3
        return data.reshape((num_samples, num_atoms, 3))  #  Correct 3D shape

    elif len(data.shape) == 3:  # Case: (num_samples, seq_len, d_model) - Already in correct shape
        return data

    else:
        raise ValueError(f"Unexpected input shape {data.shape}, expected (num_samples, input_size) or (num_samples, seq_len, d_model).")


def latent_space_coverage(z_samples):
    return jnp.mean(jnp.std(z_samples, axis=0))


# Main Execution
batch_size = 600
num_epochs = 1000
latent_dim = 102

dcd_filename = "decaalanine_1us_split3.dcd"
prmtop_filename = "ala_deca_peptide.prmtop"


train_data, test_data, input_size = load_md_data(dcd_filename, prmtop_filename, coord_type="Cartesian")
params = amber_prmtop_load(prmtop_filename)  # Load parameters here
train_data_reshaped = train_data.reshape(-1, input_size // 3, 3)

# Detect bonds and torsion bonds
initial_positions = train_data_reshaped[0]
example_bonds = detect_bonds(initial_positions, bond_threshold=1.5)
torsion_bonds = detect_torsion_bonds(example_bonds)

# Adjust bond parameters to match the number of detected bonds
num_bonds = example_bonds.shape[0]
params['bond_eq_lengths'] = params['bond_eq_lengths'][:num_bonds]
params['bond_k'] = params['bond_k'][:num_bonds]

# Generate angle_bonds from bonds
angle_bonds = form_angle_bonds(example_bonds)

# Adjust angle parameters to match the number of detected angle bonds
num_angle_bonds = angle_bonds.shape[0]
params['angle_eq'] = np.resize(params['angle_eq'], num_angle_bonds)
params['angle_k'] = np.resize(params['angle_k'], num_angle_bonds)


# Convert torsion_bonds to a hashable format
torsion_bonds_hashable = tuple(map(tuple, torsion_bonds))

# Precompute real torsional angles
real_angles = calculate_torsional_angles(train_data_reshaped[0], torsion_bonds)

# Calculate bond lengths
real_bond_lengths = []
for sample in train_data_reshaped:
    bond_lengths = np.linalg.norm(sample[example_bonds[:, 0]] - sample[example_bonds[:, 1]], axis=1)
    real_bond_lengths.append(bond_lengths)
real_bond_lengths = np.array(real_bond_lengths)

print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
print("Input Size:", input_size)
print("Detected Bonds:\n", example_bonds)
print("Detected Torsion Bonds:\n", torsion_bonds)


# Initialize Transformer Generator and Discriminator
generator = TransformerModel(input_size=306, num_layers=6, num_heads=6, d_model=102)
#Matches latent_dim
discriminator = Discriminator(input_size=input_size)

# Create PRNG keys
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

# Initialize the Transformer Generator
generator_variables = generator.init(subkey, jnp.ones((batch_size, input_size)), train=False)
key, subkey = jax.random.split(key)

# Initialize Discriminator
discriminator_params = discriminator.init(subkey, jnp.ones((batch_size, input_size)))

# Define optimizers
generator_tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
    optax.adam(learning_rate=1e-4)
)
discriminator_tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4)  # Restore higher learning rate
)



# Create Training States (No batch_stats required)
generator_state = TrainState.create(
    apply_fn=generator.apply,
    params=generator_variables['params'],
    tx=generator_tx
)  # No batch_stats



discriminator_state = TrainState.create(
    apply_fn=discriminator.apply,
    params=discriminator_params['params'],
    tx=discriminator_tx,
)


# Initialize tracking lists
num_generated_samples = 2000
real_data_accuracy = []
generated_data_fooling_rate = []
real_outputs_all_epochs = []
fake_outputs_all_epochs = []


# Set logging interval and checkpoint file name
log_interval = 10
file_name = "gan_checkpoint.pkl"
log_file_name = "training_log.txt"


def initialize_generator_state():
    generator_variables = generator.init(subkey, jnp.ones((batch_size, input_size)), train=False)

    # Debugging print statement
    print("Generator Variables Keys:", generator_variables.keys())  # Should not contain 'batch_stats'

    generator_tx = optax.adam(learning_rate=1e-4)

    return TrainState.create(
        apply_fn=generator.apply,
        params=generator_variables['params'],
        tx=generator_tx  # No batch_stats
    )


def check_nan_value(value, name, epoch):
    """Check for NaN or Inf values in loss terms."""
    if jnp.isnan(value).any() or jnp.isinf(value).any():
        print(f"⚠️ NaN/Inf detected in {name} at epoch {epoch}")
        return True
    return False


def initialize_discriminator_state():
    discriminator_params = discriminator.init(subkey, jnp.ones((batch_size, input_size)))
    discriminator_tx = optax.adam(learning_rate=1e-4)
    return train_state.TrainState.create(
        apply_fn=discriminator.apply,
        params=discriminator_params['params'],
        tx=discriminator_tx
    )


# Load checkpoint or initialize states
generator_state, discriminator_state, start_epoch = load_checkpoint(file_name)

if generator_state is None or discriminator_state is None:
    generator_state = initialize_generator_state()
    discriminator_state = initialize_discriminator_state()
    start_epoch = 0  # Start from scratch if no checkpoint is found
    print("No checkpoint found. Starting training from scratch.")
else:
    print(f"Resuming training from epoch {start_epoch}.")

# Initialize lists to store loss values
generator_losses = []
discriminator_losses = []
angle_losses = []
torsion_losses = []


# Main Training Loop
try:
    with open(log_file_name, "a") as log_file:  # Open the log file in append mode
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Progress"):
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                real_bond_lengths_batch = real_bond_lengths[i:i + batch_size]
                assert isinstance(key, jax.Array)
                key, subkey = jax.random.split(key)
                generator_state, discriminator_state, g_loss, d_loss, angle_loss, torsion_loss = train_step(
                    generator_state,
                    discriminator_state,
                    batch,
                    params,
                    example_bonds,
                    real_bond_lengths_batch,
                    epoch,
                    num_epochs,
                    real_angles,
                    latent_dim,
                    torsion_bonds_hashable,
                    subkey
                )

                #  Check for NaN values and stop execution if detected
                if check_nan_value(g_loss, "Generator Loss", epoch) or check_nan_value(d_loss, "Discriminator Loss", epoch):
                    print("⚠️ Terminating training due to NaN values.")
                    break  # Stop training safely

            # Store loss values for this epoch
            generator_losses.append(float(g_loss))
            discriminator_losses.append(float(d_loss))
            angle_losses.append(float(angle_loss))
            torsion_losses.append(float(torsion_loss))

            # Log outputs and save trajectory every log_interval epochs
            if epoch % log_interval == 0:
                real_output = discriminator.apply({'params': discriminator_state.params}, train_data)
                z_rng, subkey = jax.random.split(key)
                latent_samples = jax.random.normal(z_rng, shape=(num_generated_samples, latent_dim))

                # Ensure latent_samples shape is compatible with d_model
                batch_size, latent_dim = latent_samples.shape
                seq_len = latent_dim // 102  # Ensure seq_len is valid

                if seq_len * 102 != latent_dim:
                    raise ValueError(f"Latent dim {latent_dim} must be divisible by d_model {102}")

                # Reshape latent samples before passing to generator
                latent_samples = latent_samples.reshape(batch_size, seq_len, 102)

                # Now call generator.apply with correct shape
                fake_data = generator.apply(
                    {'params': generator_state.params},
                    latent_samples,
                    rngs={'dropout': subkey},
                    train=False
                )

                fake_output = discriminator.apply({'params': discriminator_state.params}, fake_data)
                fake_data_reshaped = reshape_to_3d_structure(fake_data)
                dcd_file_name = f"generated_epoch_{epoch}.dcd"

                try:
                    write_traj("generated_sample", fake_data_reshaped, "GAN_Model", latent_dim, dcd_file_name)
                except Exception as e:
                    print(f"Error saving DCD file for epoch {epoch}: {e}")

                # Log metrics
                real_data_correct = (real_output > 0.5).mean() * 100
                fake_data_fooling = (fake_output > 0.5).mean() * 100
                latent_coverage = latent_space_coverage(latent_samples)

                log_message = (
                    f"Epoch {epoch}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}, "
                    f"Angle Loss: {angle_loss:.4f}, Torsion Loss: {torsion_loss:.4f}, "
                    f"Real Data Accuracy: {real_data_correct:.2f}%, "
                    f"Fake Data Fooling: {fake_data_fooling:.2f}%, Latent Coverage: {latent_coverage:.4f}\n"
                )
                print(log_message.strip())
                log_file.write(log_message)  # Save log to file

            # Save checkpoint every log_interval epochs or at the final epoch
            if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
                try:
                    save_checkpoint(generator_state, discriminator_state, epoch + 1, file_name)
                except Exception as e:
                    print(f"Error saving checkpoint at epoch {epoch + 1}: {e}")



except KeyboardInterrupt:
    print("Training interrupted. Saving checkpoint...")
    save_checkpoint(generator_state, discriminator_state, epoch, file_name)
    print("Checkpoint saved. You can resume training later.")


# Function to Plot Losses
def plot_all_losses(generator_losses, discriminator_losses, angle_losses, torsion_losses):
    if len(generator_losses) == 0 or len(discriminator_losses) == 0:
        print("⚠️ No loss data available for plotting.")
        return

    plt.figure(figsize=(14, 6))

    # GAN Losses
    plt.subplot(1, 2, 1)
    plt.plot(generator_losses, label="Generator Loss", color='blue')
    plt.plot(discriminator_losses, label="Discriminator Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator vs Discriminator Loss")
    plt.legend()
    plt.grid(True)

    # Structural Losses
    plt.subplot(1, 2, 2)
    plt.plot(angle_losses, label="Angle Loss", color='orange')
    plt.plot(torsion_losses, label="Torsion Loss", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Angle vs Torsion Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    loss_plot_filename = "all_losses_plot.png"
    plt.savefig(loss_plot_filename, dpi=300, bbox_inches='tight')
    print(f"📉 All losses plot saved as {loss_plot_filename}")
    plt.show()


# Call the function to plot and save the updated loss visualizations
plot_all_losses(generator_losses, discriminator_losses, angle_losses, torsion_losses)


real_outputs_all_epochs = np.concatenate(real_outputs_all_epochs)
fake_outputs_all_epochs = np.concatenate(fake_outputs_all_epochs)

# Final evaluation and generation
print("\nGenerating final samples...")
final_latent_samples = jax.random.normal(jax.random.PRNGKey(42), shape=(num_generated_samples, latent_dim))

# Generate fake data
key, subkey = jax.random.split(key)
final_fake_data = generator.apply(
    {'params': generator_state.params},  # Only keep 'params'
    final_latent_samples,
    subkey,
    rngs={'dropout': subkey},
    train=False
)


# Pass the fake data through the discriminator to compute probabilities
final_fake_probabilities = discriminator.apply({'params': discriminator_state.params}, final_fake_data)

# Reshape to 3D and normalize coordinates
final_fake_data_3d = reshape_to_3d_structure(final_fake_data)


# Apply normalization for each sample
# --- Normalize using reference from training sample ---
_, scale = normalize_coordinates(train_data_reshaped[0])  # Reference from real structure

final_fake_data_normalized = []
for sample in final_fake_data_3d:
    normalized_sample, _ = normalize_coordinates(sample, reference_scale=scale)
    final_fake_data_normalized.append(normalized_sample)

final_fake_data_normalized = np.array(final_fake_data_normalized)

aligned_real, _ = align_structures(train_data_reshaped[0:1], final_fake_data_normalized[0:1])
generated_bonds = detect_bonds(aligned_real[0], bond_threshold=1.5, min_threshold=1.0)


# Detect bonds for the aligned structure
generated_bonds = detect_bonds(aligned_real[0], bond_threshold=1.6, min_threshold=1.0)

real_angles = calculate_torsional_angles(train_data_reshaped[0], torsion_bonds)
generated_angles = calculate_torsional_angles(final_fake_data_normalized[0], torsion_bonds)
print("Real Torsional Angles:", real_angles)
print("Generated Torsional Angles:", generated_angles)

angle_similarity_metric = jnp.mean(jnp.abs(real_angles - generated_angles))
print(f"Angular Similarity Metric: {angle_similarity_metric:.4f}")


# Evaluate discriminator output distribution for real and fake data
plt.figure(figsize=(10, 6))
plt.hist(real_outputs_all_epochs, bins=200, alpha=0.7, label="Real Data Outputs (All Epochs)")
plt.hist(fake_outputs_all_epochs, bins=200, alpha=0.7, label="Generated Data Outputs (All Epochs)")
plt.xlabel("Discriminator Output")
plt.ylabel("Frequency")
plt.title("Discriminator Outputs Across All Epochs")
plt.legend()
plt.grid()
plt.show()

# Display generated data probabilities
print("\nGenerated Data Probabilities (Sample):")
for i, prob in enumerate(final_fake_probabilities):
    print(f"Sample {i + 1}: Real Probability: {prob[0]:.4f}, Fake Probability: {1 - prob[0]:.4f}")

print("\nTraining and generation completed.")
