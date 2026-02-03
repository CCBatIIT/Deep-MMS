import os, sys, glob
from pyscripts.heavy_atom_rmsd import *

from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
    
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

json_params = sys.argv[1]

start = datetime.now()
self = HeavyAtom_NN_Experiment(json_params, from_json_params=False)
self.MAIN_train(n_epochs=1000, verbose=100)
train_time = datetime.now() - start
printf(f"Time to train {self.n_latents} model for {self.epoch} epochs was {train_time} averaging {train_time/self.epoch} per epoch.") 

traingrp = self.rootgrp['Train']
testgrp = self.rootgrp['Test']
rmsd_train = np.mean(traingrp['RMSD_Loss_Term'][:, :], axis=-1)
rmsd_test  = np.mean(testgrp['RMSD_Loss_Term'][:, :], axis=-1)
printf(f"Final Epoch {self.epoch} Train RMSD:{rmsd_train[-1]*10:2.3f} A, Test RMSD:{rmsd_test[-1]*10:2.3f} A")

figure_dir = os.path.join(self.data_dir, 'figures')
if not os.path.isdir(figure_dir):
    os.mkdir(figure_dir)

#RMSD TERM
plt.clf()
_ = plt.plot(np.arange(len(rmsd_train)), rmsd_train, label='Train')
_ = plt.plot(np.arange(len(rmsd_test)), rmsd_test, label='Test')
plt.legend()
#plt.ylim(1, 10)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction RMSD (nm)')
title = f"{self.n_latents} Latents - RMSD Loss Term"
plt.title(title)
plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
#plt.show()

#Perform VAE on Test Data
root_key = jax.random.PRNGKey(self.epoch)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

if self.is_batchnorm:
    decoded, latent_means, latent_vars = self.state.apply_fn({'params': self.state.params, 'batch_stats': self.state.batch_stats},
                                                              self.test_data, main_key, train=False, rngs={'dropout': dropout_key})
else:
    decoded, latent_means, latent_vars = self.state.apply_fn({'params': self.state.params},
                                                             self.test_data, main_key, train=False, rngs={'dropout': dropout_key})



printf(f"{decoded.shape=}, {latent_means.shape=}, {latent_vars.shape=}")

#Plot the rmsd histogram of decoded to encoded
rmsd_vals = atom_rmsd(self.test_data, decoded)*10
plt.clf()
_ = plt.hist(rmsd_vals, bins=50)
title = "Reconstruction RMSD"
plt.title(title)
plt.xlabel("RMSD (Angstrom)")
plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
#plt.show()

#Plot the latent dimensional distributions
colors = [(i/self.n_latents, 0.1, (self.n_latents - i)/self.n_latents) for i in range(1, self.n_latents+1)]
print(colors)
plt.clf()
_ = plt.hist(latent_means.T, bins=100, histtype='step', alpha=0.5, color=colors)
title = 'All Latents'
plt.title(title)
plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
#plt.show()

# #Calculate MI/correlation metrics
# pearson_Rs = []
# MI_regressions = []
# for i in range(self.n_latents):
#     for j in range(i+1, self.n_latents):
#         pearson_Rs.append(stats.pearsonr(latent_means[:, i], latent_means[:, j])[0])
#         MI_regressions.append(mutual_info_regression(latent_means[:, i].reshape(-1, 1),
#                                                      latent_means[:, j].reshape(-1, 1))[0])
# pearson_Rs = jnp.array(pearson_Rs)
# MI_regressions = jnp.array(MI_regressions)

# #Plot distribution of pearsons R's between latents
# plt.clf()
# _ = plt.hist(jnp.abs(pearson_Rs), density=True, bins=20)
# title = 'Pearson R'
# plt.title(title)
# plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
# #plt.show()
# #Plot distribution of Mutual Informations between latents
# plt.clf()
# _ = plt.hist(MI_regressions, density=True, bins=20)
# title = 'Mutual Information'
# plt.title(title)
# plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
# #plt.show()

# #Plot the AIC/BIC
# ics = []
# for i in range(1, 40):
#     MM = GaussianMixture(n_components=i).fit(latent_means)
#     ics.append((i, MM.aic(latent_means), MM.bic(latent_means)))
# ics = np.array(ics)
# plt.clf()
# _ = plt.plot(ics[:, 0], ics[:, 1], color='blue', label='Akaike Info Criterion')
# _ = plt.scatter(ics[:, 0], ics[:, 1], color='blue')
# _ = plt.plot(ics[:, 0], ics[:, 2], color='orange', label='Bayes Info Criterion')
# _ = plt.scatter(ics[:, 0], ics[:, 2], color='orange')
# plt.legend()
# title = "AIC_BIC"
# plt.title(title)
# plt.xlabel('Num Components')
# plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
# #plt.show()


# #Gaussian Mixture Modeling
# component_nums = [1, 1, 2, np.argmin(ics[:, 1])+1, np.argmin(ics[:, 2])+1]
# sample_mults = [1, 50, 1, 1, 1]

# for num_components, mult in zip(component_nums, sample_mults):
#     MM = GaussianMixture(n_components=num_components).fit(latent_means)
#     samples, _ = MM.sample(latent_means.shape[0]*mult)

#     if not os.path.isdir(os.path.join(figure_dir, 'GMM_hists')):
#         os.mkdir(os.path.join(figure_dir, 'GMM_hists'))

#     for i in range(samples.shape[-1]):
#         plt.clf()
#         _ = plt.hist(latent_means[:,i], bins=25, histtype='step', color='g', label='Encoded from Test', density=True)
#         _ = plt.hist(samples[:,i], bins=25, histtype='step', color='b', label='GMM Samples', density=True)
#         title = f'Latent {i} - {num_components} Component GMM'
#         plt.title(title)
#         plt.legend()
#         plt.savefig(os.path.join(figure_dir, 'GMM_hists', title+'.png'), dpi=900, bbox_inches='tight')
#         #plt.show()

#     key = jax.random.PRNGKey(self.epoch)
#     main_key, dropout_key = jax.random.split(key, num=2)
#     decoded_samples, _ = self.state.apply_fn({'params': self.state.params, 'batch_stats': self.state.batch_stats},
#                                           samples, main_key, train=False,
#                                           method=self.model.decode,
#                                           rngs={'dropout': dropout_key}, mutable=['batch_stats'])
#     printf(f"{self.test_data.shape=}, {latent_means.shape=}, {decoded.shape=}, {samples.shape=}, {decoded_samples.shape=}")

#     # Contour plot of Test Data, Reconstructed Data, and Generated Data
#     fit_all, fit_test = False, True

#     pca_test_fit = PCA(n_components=2)
#     pca_all_fit = PCA(n_components=2)
#     pca_test_1 = pca_test_fit.fit_transform(self.test_data)
#     _ = pca_all_fit.fit(jnp.concatenate((self.test_data, decoded, decoded_samples)))
#     pca_test_2 = pca_all_fit.transform(self.test_data)

#     pca_recon_1 = pca_test_fit.transform(decoded)
#     pca_recon_2 = pca_all_fit.transform(decoded)
#     pca_gener_1 = pca_test_fit.transform(decoded_samples)
#     pca_gener_2 = pca_all_fit.transform(decoded_samples)

#     names = ["Test Set", "Reconstructed", "Generated"]
#     colors = ['red', 'green', 'blue']
# #    xx, yy = np.meshgrid(np.linspace(pca_test[:,0].min()-2, pca_test[:,0].max()+2, 100),
# #                         np.linspace(pca_test[:,1].min()-2, pca_test[:,1].max()+2, 100))
# #
# #    custom_lines = []
# #    for i, data in enumerate([pca_test, pca_recon, pca_gener]):
# #        kde = gaussian_kde(data.T)
# #        z = kde(np.vstack([xx.flatten(), yy.flatten()]))
# #        z = z.reshape(xx.shape)
# #        levels = np.linspace(z.max()*0.2, z.max(), 6)
# #        print(i, levels)
# #        plt.contour(xx, yy, z, colors=colors[i], levels=levels)
# #        custom_lines.append(plt.Line2D([0], [0], color=colors[i], lw=3))
# #    lgd = plt.legend(custom_lines, names, bbox_to_anchor=(1.32,1))
# #    title = f'PCA Contour Comparison - {num_components} Component GMM - {decoded_samples.shape[0]:1.1E} Samples'
# #    plt.title(title)
# #    plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
# #    #plt.show()
# #
#     plt.clf()
#     custom_lines = []
#     for i, data in enumerate([pca_test_1, pca_recon_1, pca_gener_1]):
#         plt.scatter(data[:,0], data[:,1], color=colors[i], label=names[i], alpha=0.05)
#         custom_lines.append(plt.Line2D([0], [0], color=colors[i], lw=3))
#     lgd = plt.legend(custom_lines, names, bbox_to_anchor=(1.32,1))
#     title = f'PCA (Test) Scatter Comparison - {num_components} Component GMM - {decoded_samples.shape[0]:1.1E} Samples'
#     plt.title(title)
#     plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
#     #plt.show()

#     plt.clf()
#     custom_lines = []
#     for i, data in enumerate([pca_test_2, pca_recon_2, pca_gener_2]):
#         plt.scatter(data[:,0], data[:,1], color=colors[i], label=names[i], alpha=0.05)
#         custom_lines.append(plt.Line2D([0], [0], color=colors[i], lw=3))
#     lgd = plt.legend(custom_lines, names, bbox_to_anchor=(1.32,1))
#     title = f'PCA (All) Scatter Comparison - {num_components} Component GMM - {decoded_samples.shape[0]:1.1E} Samples'
#     plt.title(title)
#     plt.savefig(os.path.join(figure_dir, title+'.png'), dpi=900, bbox_inches='tight')
#     #plt.show()

