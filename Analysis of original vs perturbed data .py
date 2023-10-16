#!/usr/bin/env python
# coding: utf-8

# In[4]:


import Canonical_AutoEncoder_Experiment
import sys
import jax
import mdtraj as md
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Configure JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)

# JSON configuration file path
json_fn = "/media/volume/sdb/Timooo/Auto_Encoding_FE/AE_TEST_SMALL_08_3.json"

# Initialize the experiment
experiment = Canonical_AutoEncoder_Experiment.NN_Experiment(json_fn)

# Loading the model checkpoint
ckpt_fn = "model_ckpt_005000.pkl"
experiment.load_model_from_ckpt(ckpt_fn)

# Save testing data as a trajectory
experiment.write_traj("test_data", experiment.test_data)

# Load simulation data
c = md.load('/media/volume/sdb/Timooo/Auto_Encoding_FE/Simulation/small_test.dcd', top='/media/volume/sdb/Timooo/Auto_Encoding_FE/Simulation/ala_deca_peptide.prmtop')
coords = jnp.array(c.superpose(c).xyz)
print(coords.shape)

# Flatten the dimensions
coords = coords.reshape(coords.shape[0], -1)
print(coords.shape)

# Encode the test data, obtaining both decoded and latent representations
recon = experiment.state.apply_fn({'params': experiment.state.params}, experiment.test_data)
decoded_data = recon[0]
original_latents = recon[1]

# Perturb Latent Representations
perturbed_latents = jnp.array(original_latents)  # Copy original latents
perturbed_latents = perturbed_latents.at[:, 0].add(31)  # Perturb the first column

# Print the shape of the perturbed latents
print("Perturbed Latents Shape:", perturbed_latents.shape)

# Visualize original latents and perturbed latents
for i in range(original_latents.shape[1]):
    _ = plt.hist(original_latents[:, i], bins=100, color='blue', alpha=0.5, label='Original Latents')
    _ = plt.hist(perturbed_latents[:, i], bins=100, color='red', alpha=0.5, label='Perturbed Latents')
    plt.legend()
    plt.show()

# Plot histograms and analysis model to compare original and perturbed latents
def compare_latents(original_latents, perturbed_latents):
    for i in range(original_latents.shape[1]):
        plt.figure(figsize=(8, 4))

        # Original Latents Histogram
        plt.subplot(1, 2, 1)
        plt.hist(original_latents[:, i], bins=100, color='blue', alpha=0.5)
        plt.xlabel(f'Latent Dimension {i + 1}')
        plt.ylabel('Frequency')
        plt.title('Original Latents')

        # Perturbed Latents Histogram
        plt.subplot(1, 2, 2)
        plt.hist(perturbed_latents[:, i], bins=100, color='red', alpha=0.5)
        plt.xlabel(f'Latent Dimension {i + 1}')
        plt.ylabel('Frequency')
        plt.title('Perturbed Latents')

        plt.tight_layout()
        plt.show()

    # Add an analysis model here to compare original and perturbed latents
    # You can calculate and display any relevant metrics or comparisons
    # For example, you can compute the mean, variance, or any other statistical analysis.

# Compare original and perturbed latents
compare_latents(original_latents, perturbed_latents)


# In[5]:


# Example: T-test
from scipy.stats import ttest_ind

# Perform a t-test on the first dimension of original and perturbed latents
t_stat, p_value = ttest_ind(original_latents[:, 0], perturbed_latents[:, 0])
print(f"T-statistic: {t_stat}, p-value: {p_value}")


# In[6]:


# Example: Scatter Plot
plt.scatter(original_latents[:, 0], original_latents[:, 1], label='Original Latents', c='blue', alpha=0.5)
plt.scatter(perturbed_latents[:, 0], perturbed_latents[:, 1], label='Perturbed Latents', c='red', alpha=0.5)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.show()



# In[8]:


get_ipython().system('pip install scikit-learn')



# In[9]:


# Example: PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
original_latents_pca = pca.fit_transform(original_latents)
perturbed_latents_pca = pca.fit_transform(perturbed_latents)

plt.scatter(original_latents_pca[:, 0], original_latents_pca[:, 1], label='Original Latents', c='blue', alpha=0.5)
plt.scatter(perturbed_latents_pca[:, 0], perturbed_latents_pca[:, 1], label='Perturbed Latents', c='red', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[13]:


# Example: Pearson Correlation with Handling of Constant Columns
def calculate_pearson_correlation(data):
    # Check for constant columns and remove them
    std_dev = np.std(data, axis=0)
    constant_columns = np.where(std_dev == 0)[0]

    # Remove constant columns
    data_filtered = np.delete(data, constant_columns, axis=1)

    # Calculate Pearson correlation on filtered data
    correlation_matrix = np.corrcoef(data_filtered, rowvar=False)
    
    return correlation_matrix

# Calculate Pearson correlation for original and perturbed latents
correlation_matrix_original = calculate_pearson_correlation(original_latents)
correlation_matrix_perturbed = calculate_pearson_correlation(perturbed_latents)

# Handle any remaining NaN or infinity values in the correlation matrices if needed
correlation_matrix_original = np.nan_to_num(correlation_matrix_original)
correlation_matrix_perturbed = np.nan_to_num(correlation_matrix_perturbed)




# In[17]:


pip install seaborn matplotlib


# In[18]:


# Example: Heatmap
import seaborn as sns

# Create correlation matrices (as calculated earlier)
sns.heatmap(correlation_matrix_original, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Original)")
plt.show()

sns.heatmap(correlation_matrix_perturbed, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Perturbed)")
plt.show()

