import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_latents(original_latents, perturbed_latents):
    """
    Visualize original and perturbed latents using histograms.

    Parameters:
    - original_latents: Original latents array.
    - perturbed_latents: Perturbed latents array.
    """
    for i in range(original_latents.shape[1]):
        plt.hist(original_latents[:, i], bins=100, color='blue', alpha=0.5, label='Original Latents')
        plt.hist(perturbed_latents[:, i], bins=100, color='red', alpha=0.5, label='Perturbed Latents')
        plt.legend()
        plt.show()

def compare_latents_histogram(original_latents, perturbed_latents):
    """
    Compare original and perturbed latents using side-by-side histograms.

    Parameters:
    - original_latents: Original latents array.
    - perturbed_latents: Perturbed latents array.
    """
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

def scatter_plot_latents(original_latents, perturbed_latents):
    """
    Create a scatter plot to compare original and perturbed latents.

    Parameters:
    - original_latents: Original latents array.
    - perturbed_latents: Perturbed latents array.
    """
    plt.scatter(original_latents[:, 0], original_latents[:, 1], label='Original Latents', c='blue', alpha=0.5)
    plt.scatter(perturbed_latents[:, 0], perturbed_latents[:, 1], label='Perturbed Latents', c='red', alpha=0.5)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.show()

# Example usage:
#     visualize_latents(original_latents, perturbed_latents)
#     compare_latents_histogram(original_latents, perturbed_latents)
#     scatter_plot_latents(original_latents, perturbed_latents)

def calculate_pearson_correlation(data):
    """
    Calculate Pearson correlation matrix with handling of constant columns.

    Parameters:
    - data: Input data array.

    Returns:
    - correlation_matrix: Pearson correlation matrix.
    """
    std_dev = np.std(data, axis=0)
    constant_columns = np.where(std_dev == 0)[0]
    data_filtered = np.delete(data, constant_columns, axis=1)
    correlation_matrix = np.corrcoef(data_filtered, rowvar=False)
    return correlation_matrix

def plot_heatmap(correlation_matrix, title):
    """
    Plot a heatmap for the given correlation matrix.

    Parameters:
    - correlation_matrix: Pearson correlation matrix.
    - title: Title for the heatmap.
    """
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()

# Example usage:
#     correlation_matrix_original = calculate_pearson_correlation(original_latents)
#     correlation_matrix_perturbed = calculate_pearson_correlation(perturbed_latents)
#     
#     correlation_matrix_original = np.nan_to_num(correlation_matrix_original)
#     correlation_matrix_perturbed = np.nan_to_num(correlation_matrix_perturbed)
#     
#     plot_heatmap(correlation_matrix_original, "Correlation Heatmap (Original)")
#     plot_heatmap(correlation_matrix_perturbed, "Correlation Heatmap (Perturbed)")

def scatterplot_correlations(original_latents, perturbed_latents, title):
    """
    Create scatter plots for each pair of latent dimensions.

    Parameters:
    - original_latents: Original latents array.
    - perturbed_latents: Perturbed latents array.
    - title: Title for the scatter plots.
    """
    num_dimensions = original_latents.shape[1]

    plt.figure(figsize=(40, 90))

    for i in range(num_dimensions):
        for j in range(i + 1, num_dimensions):
            plt.subplot(num_dimensions - 1, num_dimensions - 1, i * (num_dimensions - 1) + j)

            # Scatter plot
            plt.scatter(original_latents[:, i], original_latents[:, j], color='blue', label='Original Latents', alpha=0.5)
            plt.scatter(perturbed_latents[:, i], perturbed_latents[:, j], color='red', label='Perturbed Latents', alpha=0.5)
            plt.xlabel(f'Latent Dimension {i + 1}')
            plt.ylabel(f'Latent Dimension {j + 1}')
            plt.title(f'Latent Dimensions {i + 1} vs {j + 1}')
            plt.legend()

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

# Example usage:
#     scatterplot_correlations(original_latents, perturbed_latents, "Correlations Between Original Latents")
#     scatterplot_correlations(original_latents, perturbed_latents, "Correlations Between Perturbed Latents")

def scatterplot_histograms(latents, title):
    """
    Create scatter plots and histograms for each pair of latent dimensions.

    Parameters:
    - latents: Latents array.
    - title: Title for the scatter plots and histograms.
    """
    num_dimensions = latents.shape[1]

    # Create a new figure for the scatter plots and histograms
    fig, axes = plt.subplots(num_dimensions, num_dimensions, figsize=(35, 35))

    for i in range(num_dimensions):
        for j in range(i, num_dimensions):
            if i == j:
                # Create histograms on the diagonal
                axes[i, j].hist(latents[:, i], bins=50, color='blue', alpha=0.5, label='Latent Dimension')
                axes[i, j].set_xlabel(f'Latent Dimension {i + 1}')
                axes[i, j].set_ylabel('Frequency')
                axes[i, j].set_title(f'Latent Dimension {i + 1}')

            else:
                # Create scatter plots off-diagonal
                axes[i, j].scatter(latents[:, i], latents[:, j], color='blue', alpha=0.5, label='Latent Dimension')
                axes[i, j].set_xlabel(f'Latent Dimension {i + 1}')
                axes[i, j].set_ylabel(f'Latent Dimension {j + 1}')
                axes[i, j].set_title(f'Latent Dimensions {i + 1} vs {j + 1}')

            # Remove x and y ticks for plots not on the edges
            if i < num_dimensions - 1:
                axes[i, j].set_xticks([])
            if j > 0:
                axes[i, j].set_yticks([])

            # Invert the scatter plot layout
            if i != j:
                axes[j, i].scatter(latents[:, j], latents[:, i], color='blue', alpha=0.5, label='Latent Dimension')
                axes[j, i].set_xlabel(f'Latent Dimension {j + 1}')
                axes[j, i].set_ylabel(f'Latent Dimension {i + 1}')
                axes[j, i].set_title(f'Latent Dimensions {j + 1} vs {i + 1}')

    # Set the title for the entire plot
    fig.suptitle(title, y=1.02)

    # Display the legend in the last subplot
    axes[0, num_dimensions - 1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Example usage:
#     scatterplot_histograms(original_latents, "Latent Dimension Correlations and Histograms")
#     scatterplot_histograms(perturbed_latents, "Perturbed Latent Dimension Correlations and Histograms")
