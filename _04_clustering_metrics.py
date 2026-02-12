#USAGE python _04_...py json_file

#How do the best models compare to PCA in terms of clustering?
import os, sys, jax, glob
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pyscripts.heavy_atom_rmsd import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from datetime import datetime

decode_pca = lambda data, pca: pca.inverse_transform(pca.transform(heavy_atom_analyzer.data))

from _02_write_viollin_data import HeavyAtom_Analyzer
#Obtain the test set date from the VAE Model
json_fn = sys.argv[1]

HA = HeavyAtom_Analyzer(json_fn=json_fn)
model_name = HA.model_name
test_frames = HA.test_data

work_dir = os.path.join(HA.data_dir, 'clustering')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir, exist_ok=True)
log_fn = os.path.join(work_dir, 'clustering_metric_log.txt')
logfile = open(log_fn, 'w')

one_off_rmsd = lambda a, b: jnp.sqrt(jnp.mean(jnp.sum((b.reshape(-1, 3) - a.reshape(-1, 3))**2, axis=1)))

#Get distance matrix of the test frames
def dist_matrix_row(a, bs):
    return jax.vmap(one_off_rmsd, in_axes=(None, 0))(a, bs)

def rmsd_distance_matrix(frames):
    dist_mat = jnp.empty((frames.shape[0], frames.shape[0]))
    for i in range(frames.shape[0]):
        dist_mat = dist_mat.at[i].set(dist_matrix_row(frames[i], frames))
    return dist_mat

logfile.write('Get distance matrix of testing data... \n')
dist_mat_fn = os.path.join(work_dir, 'TestDistances.sqmat.npy')
try:
    logfile.write("Attempting to load distance matrix...\n")
    logfile.write(f"\tFrom:{dist_mat_fn}\n")
    test_distance_matrix = jnp.load(dist_mat_fn)
    logfile.write("\tSuccess!\n")
except:
    logfile.write("\tSomething went wrong, regenerating distance matrix...\n")
    test_distance_matrix = rmsd_distance_matrix(test_frames)
    logfile.write('\t\t Done!\n')
    jnp.save(dist_mat_fn, test_distance_matrix)
    logfile.write(f'\t\t Saved to {dist_mat_fn}\n')

#Entries {key=name, value=clustering_method}
clustering_methods = {'KMeans': KMeans,
                      'Agglomerative': AgglomerativeClustering,
                      'HDBScan': HDBSCAN}
methods_without_nums_clusters = ['HDBScan']

#Entries {key=name, value=[function, is_supervised]}
metric_functions = {"Rand Score":[rand_score, True],
                    "normalized MI Score":[normalized_mutual_info_score, True],
                    "Fowlkes-Mallows Score":[fowlkes_mallows_score, True],
                    "Silhouette Score":[silhouette_score, False],
                    "Davies-Bouldin Score":[davies_bouldin_score, False],
                    "Calinksi-Harabasz Score":[calinski_harabasz_score, False]}

hdbscan_epsilons = np.linspace(0, test_distance_matrix.max())

def num_clusters_by_method(method_name, min_clusters=2, max_clusters=25, num_repetitions=5, plot=False):
    assert method_name in clustering_methods.keys()
    
    sil_scores = np.empty((num_repetitions, 1+max_clusters-min_clusters))
    nums_clusters = np.arange(min_clusters, max_clusters+1)
    for j in range(num_repetitions):
        if method_name == 'KMeans':
             #Iterate over number of clusters
            for i in range(min_clusters, max_clusters+1):
                labels = clustering_methods[method_name](n_clusters=i).fit_predict(test_frames)
                sil_scores[j, i-min_clusters] = silhouette_score(test_frames, labels)
        
        elif method_name == 'Agglomerative':
             #Iterate over number of clusters
            for i in range(min_clusters, max_clusters+1):
                labels = clustering_methods[method_name](n_clusters=i, metric='precomputed', linkage='complete').fit_predict(test_distance_matrix)
                sil_scores[j, i-min_clusters] = silhouette_score(test_frames, labels)
        
        elif method_name == 'HDBScan':
             #Iterate over threshold
            for i, k in enumerate(range(min_clusters, max_clusters+1)):
                eps = hdbscan_epsilons[i]
                labels = HDBSCAN(metric='precomputed', cluster_selection_epsilon=eps).fit_predict(np.array(test_distance_matrix, copy=True))
                sil_scores[j, i-min_clusters] = silhouette_score(test_frames, labels)
        

    if plot is True:
        plt.clf()
        if method_name in ['KMeans', 'Agglomerative']:
            _ = plt.errorbar(x=nums_clusters, y=np.mean(sil_scores, axis=0), yerr=np.std(sil_scores, axis=0))
            plt.xlabel('Num Clusters')
            logfile.write(f"{method_name}, {np.argmax(np.mean(sil_scores, axis=0)) + min_clusters}\n")
        elif method_name in ['HDBScan']:
            _ = plt.errorbar(x=hdbscan_epsilons[:len(range(min_clusters, max_clusters+1))], y=np.mean(sil_scores, axis=0), yerr=np.std(sil_scores, axis=0))
            plt.xlabel('HDBScan Epsilon')
            logfile.write(f"{method_name}, {hdbscan_epsilons[np.argmax(np.mean(sil_scores, axis=0))]}\n")
        plt.ylabel('Silhouette Score')
        plt.title(method_name)
        plt.savefig(os.path.join(work_dir, f'optimal_n_clusters_{HA.model_name}_{method_name}.png'), dpi=900)
        #plt.show()
            
    return np.argmax(np.mean(sil_scores, axis=0))+min_clusters

optimal_nums_clusters = {}
logfile.write('Determine optimal number of clusters for each method...\n')
for key in clustering_methods.keys():
    logfile.write(f"\t{key}...\n")
    optimal_nums_clusters[key] = num_clusters_by_method(key, plot=True)
    logfile.write("\t\tDone!\n")
        
logfile.write(f"{optimal_nums_clusters}")


mean_error = lambda err_arr: np.sqrt(np.sum(err_arr**2)) / err_arr.shape[0]

def nn_operate(heavy_atom_analyzer, data, rng_seed=69420):
    key = jax.random.PRNGKey(rng_seed)
    main_key, dropout_key = jax.random.split(key, num=2)
    if heavy_atom_analyzer.is_batchnorm:
        decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn({'params': heavy_atom_analyzer.state.params, 'batch_stats': heavy_atom_analyzer.state.batch_stats},
                                                                                data, main_key, train=False,
                                                                                rngs={'dropout': dropout_key})
    else:
        decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn({'params': heavy_atom_analyzer.state.params},
                                                                                data, main_key, train=False,
                                                                                rngs={'dropout': dropout_key})
    return decoded, latent_means


def validation_value(n_clusters, clustering_method, metric_function, supervised=True):
    final = []
    for j in range(5):
        # Report rand score for repeated clusterings of the test data
        if method_name == 'KMeans':
            labels_true = clustering_methods[clustering_method](n_clusters=n_clusters).fit_predict(test_frames)
        elif method_name == 'Agglomerative':
            labels_true = clustering_methods[clustering_method](n_clusters=n_clusters, metric='precomputed', linkage='complete').fit_predict(test_distance_matrix)
        elif method_name == 'HDBScan':
            labels_true = clustering_methods[clustering_method](cluster_selection_epsilon=n_clusters, metric='precomputed').fit_predict(np.array(test_distance_matrix, copy=True))
        else:
            raise Exception("How did that happen")
        
        validation_set = []
        for i in range(10):
            if method_name == 'KMeans':
                labels_pred = clustering_methods[clustering_method](n_clusters=n_clusters).fit_predict(test_frames)
            elif method_name == 'Agglomerative':
                labels_pred = clustering_methods[clustering_method](n_clusters=n_clusters, metric='precomputed', linkage='complete').fit_predict(test_distance_matrix)
            elif method_name == 'HDBScan':
                labels_pred = clustering_methods[clustering_method](cluster_selection_epsilon=n_clusters, metric='precomputed').fit_predict(np.array(test_distance_matrix, copy=True))
            else:
                raise Exception("How did that happen")
                
            if supervised:
                #Compare labels_a to labels_b
                validation_set.append(metric_function(labels_true, labels_pred))
            else:
                #Compare samples and labels_b
                validation_set.append(metric_function(test_frames, labels_pred))
                
        final.append([np.mean(validation_set), np.std(validation_set)])

    final = np.array(final)
    final = (np.mean(final[:, 0]), mean_error(final[:, 1]))
    return final

def evaluate_pca_encoder_against_metric(n_clusters, clustering_method, metric_function,
                                        rng_seed=2358, num_repetitions=5, supervised=True, recon_data=False):
    #domain = np.array([exp.n_latents for key, exp in self.experiments.items()])
    pca_scores = np.empty((num_repetitions))
    enc_scores = np.empty((num_repetitions))

    for i in range(num_repetitions):
        if method_name == 'KMeans':
            labels_true = clustering_methods[clustering_method](n_clusters=n_clusters).fit_predict(test_frames)
        elif method_name == 'Agglomerative':
            labels_true = clustering_methods[clustering_method](n_clusters=n_clusters, metric='precomputed', linkage='complete').fit_predict(test_distance_matrix)
        elif method_name == 'HDBScan':
            labels_true = clustering_methods[clustering_method](cluster_selection_epsilon=n_clusters, metric='precomputed').fit_predict(np.array(test_distance_matrix, copy=True))
        else:
            raise Exception("How did that happen")
        
        # Perform PCA "encoding" on test data
        if HA.n_latents < 2000:
            pca = PCA(n_components=HA.n_latents)
        else:
            pca = PCA(n_components=2000)
        _ = pca.fit(HA.train_data)
        pca_latents = pca.transform(HA.test_data)

        _, enc_latents = nn_operate(HA, HA.test_data, rng_seed=rng_seed)
        
        
        #Cluster both
        if method_name == 'KMeans':
            pca_labels = clustering_methods[clustering_method](n_clusters=n_clusters).fit_predict(pca_latents)
            enc_labels = clustering_methods[clustering_method](n_clusters=n_clusters).fit_predict(enc_latents)
        elif method_name == 'Agglomerative':
            pca_labels = clustering_methods[clustering_method](n_clusters=n_clusters, linkage='complete').fit_predict(pca_latents)
            enc_labels = clustering_methods[clustering_method](n_clusters=n_clusters, linkage='complete').fit_predict(enc_latents)
        elif method_name == 'HDBScan':
            pca_labels = clustering_methods[clustering_method](cluster_selection_epsilon=n_clusters).fit_predict(np.array(pca_latents, copy=True))
            enc_labels = clustering_methods[clustering_method](cluster_selection_epsilon=n_clusters).fit_predict(np.array(enc_latents, copy=True))
        else:
            raise Exception("How did that happen")

        if supervised:
            pca_scores[i] = metric_function(labels_true, pca_labels)
            enc_scores[i] = metric_function(labels_true, enc_labels)
        else:
            pca_scores[i] = metric_function(pca_latents, pca_labels)
            enc_scores[i] = metric_function(enc_latents, enc_labels)
    
    return pca_scores, enc_scores

def clustering_metric_plot(xs, pcays, encys,
                           valid_y, valid_y_err,
                           n_clusters, title,
                           ylabel, xlabel='Number of Components (log base 2)'):
    
    plt.clf()
    plt.hlines(y=[valid_y, valid_y+valid_y_err, valid_y-valid_y_err],
               xmin=np.log2(xs).min() - 0.25, xmax=np.log2(xs).max() + 0.25,
               colors='black', alpha=[0.5, 0.3, 0.3],
               linestyles=['solid', 'dashed', 'dashed'])
    
    _ = plt.errorbar(x=np.log2(xs),
                     y=np.mean(pcays, axis=0),
                     yerr=np.std(pcays, axis=0), label='PCA')
    
    _ = plt.errorbar(x=np.log2(xs)+0.05,
                     y=np.mean(encys, axis=0),
                     yerr=np.std(encys, axis=0), label='Encoder')
    
    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylabel == 'Calinksi-Harabasz Score':
        plt.yscale('log')
    save_fn = f"clustering_figures/{system.replace(' ', '_')}_{ylabel.replace(' ', '_')}_{title.replace(' ', '_')}.png"
    plt.savefig(save_fn, dpi=900)
    #plt.show()


for method_name in clustering_methods.keys():
    n_clusters = optimal_nums_clusters[method_name]
    
    for metric_name, metric in metric_functions.items():
        metric_function, is_supervised = metric
        logfile.write(f"{method_name=}, {metric_name=}, {is_supervised=}\n")
        #Run repeated clusterings on the test data
        validation_y, validation_y_err = validation_value(n_clusters=n_clusters,
                                                          clustering_method=method_name,
                                                          metric_function=metric_function,
                                                          supervised=is_supervised)
        #Multiple rounds of clustering with PCA and Encoder
        pca_scores, enc_scores = evaluate_pca_encoder_against_metric(n_clusters=n_clusters,
                                                                     clustering_method=method_name,
                                                                     metric_function=metric_function,
                                                                     supervised=is_supervised,
                                                                     recon_data=True)

        pca_mean, pca_std, vae_mean, vae_std = jnp.mean(pca_scores), jnp.std(pca_scores), jnp.mean(enc_scores), jnp.std(enc_scores)

        logfile.write(f"""For clustering method {method_name} and metric {metric_name}:
        PCA scored {pca_mean:0.3f} +/- {pca_std:0.3f}
        VAE scored {vae_mean:0.3f} +/- {vae_std:0.3f}
        """)
        
        # #Make a plot
        # clustering_metric_plot(xs=domain, pcays=pca_ys, encys=enc_ys,
        #                        valid_y=validation_y, valid_y_err=validation_y_err,
        #                        n_clusters=n_clusters, title=f"{method_name} Comparison",
        #                        ylabel=metric_name)                       

logfile.close()







