import numpy as np
import glob, os, jax
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy

from _02_write_viollin_data import HeavyAtom_Analyzer, atom_rmsd

base_dict = {"VAE_RMSD": None,
             "VAE_LOSS_RMSD": None,
             "PCA_RMSD": None,
             "PCA_LOSS_RMSD": None}

topology_map = {'BR_': 'Simulation/3mxf_implicit.pdb',
                'CR_': 'Simulation/1crn_H.pdb',
                'DA_': 'Simulation/ala_deca_peptide.pdb',
                'DA_stretch_': 'Simulation/ala_deca_peptide.pdb',
                'HIV1p_': 'Simulation/HIV1p_protein_only.pdb',
                'OX_': 'Simulation/oxycodone.pdb'}

pref2heavycount = {'OX_': 23, 'DA_': 50, 'DA_stretch_': 50, 'CR_': 327, 'CR_small_': 327, 'BR_': 1093, 'HIV1p_':1599}

def load_files(numpy_files, verbose=True):
    fns = sorted(numpy_files)
    attrs = [fn.split('/')[1:] if fn.split('/')[0] == 'numpy_backups' else fn.split('/') for fn in fns]
    model_name = attrs[0][0]
    assert False not in [elem[0] == model_name for elem in attrs]
    
    model_data = {}
    for attr_set in attrs:
        print(attr_set)
        n_latents = int(attr_set[1].split('_')[0])
        if n_latents not in model_data.keys():
            model_data[n_latents] = {}
        rpt = attr_set[2]
        if rpt not in model_data[n_latents].keys():
            model_data[n_latents][rpt] = deepcopy(base_dict)

        try:
            if attr_set[-1].split('.')[0] in [key for key in base_dict.keys()]:
                model_data[n_latents][rpt][attr_set[-1].split('.')[0]] = np.load(os.path.join(*attr_set))
            else:
                if verbose:
                    print(f"Excluding ", os.path.join(*attr_set))
        except:
            if verbose:
                    print(f"Failure on", os.path.join(*attr_set))
            
    return model_data

def lowest_rpts(chart_data):
    means = {}

    for key, val in chart_data.items():
        if key not in means.keys():
            means[key] = {}
        for key2, val2 in chart_data[key].items():
            if key2 not in means[key].keys():
                means[key][key2] = {}
            for key3, val3 in chart_data[key][key2].items():
                means[key][key2][key3] = np.mean(chart_data[key][key2][key3])
    
    best_rpts_RMSD = {key: None for key in means.keys()}
    best_rpts_LOSS = {key: None for key in means.keys()}
    
    for key, val in means.items():
        #discover the lowest mean rpt
        best_rpts_RMSD[key] = min(val.items(), key=lambda x: x[-1]['VAE_RMSD'])[0]
        best_rpts_LOSS[key] = min(val.items(), key=lambda x: x[-1]['VAE_LOSS_RMSD'])[0]
    
    return means, best_rpts_RMSD, best_rpts_LOSS

two_to = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
def index_highest_two_mod(value):
    return np.where(value - two_to < 0)[0]

#Plot Violin of the best

def violin_plots(vae_dat, pca_dat, latents, title, pref, ceil=10.0, num_stds=3, show=False, figure_dir=None, test_hists=False):
    import matplotlib.pyplot as plt
    figure_log = f"Began {datetime.now()} \n"
    figure_log += f"Log for making figure {title=} \n"
    
    #rmsds and pca rmsds are a dict of results with keys n_latent and vals hist_vals
    rpt_rmsds = [10*vae_dat[key] for key in vae_dat.keys()] #convert to angstrom
    rpt_pca_rmsds = [10*pca_dat[key] for key in pca_dat.keys()] #convert to angstrom
    og_shapes, og_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]
    
    #remove nan values
    rpt_rmsds = [data[np.where(~np.isnan(data))] for data in rpt_rmsds]
    rpt_pca_rmsds = [data[np.where(~np.isnan(data))] for data in rpt_pca_rmsds]
    non_nan_shapes, non_nan_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]

    #Report the number of Nan values, and remove datasets with all Nan values
    indices_all_nan = []
    for i, (latent, old1, old2, new1, new2) in enumerate(zip(latents, og_shapes, og_pca_shapes, non_nan_shapes, non_nan_pca_shapes)):
        figure_log += f"Analyze NAN content for {latent} Latents \n"
        figure_log += f"For model with {latent} Latents, \n \t {old1-new1} Nan values were found in VAE \n \t {old2 - new2} Nan values were found in PCA \n"
        if new1 == 0:
            figure_log += f"For model with {latent} Latents, \n \t All values are NAN - a dashed line on the plot \n"
            indices_all_nan.append(i)

    #Now remove values above a ridiculous threshold, like 10 angstroms
    rpt_rmsds = [data[np.where(data < ceil)] for data in rpt_rmsds]
    rpt_pca_rmsds = [data[np.where(data < ceil)] for data in rpt_pca_rmsds]
    in_thresh_shapes, in_thresh_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]

    #Report the number of values above the threshold
    for i, (latent, old1, old2, new1, new2) in enumerate(zip(latents, non_nan_shapes, non_nan_pca_shapes, in_thresh_shapes, in_thresh_pca_shapes)):
        figure_log += f"Analyze Outliers content for {latent} Latents \n"
        figure_log += f"For model with {latent} Latents, \n \t {old1-new1} values were found in VAE above the ceiling of {ceil} Angstroms \n \t {old2 - new2} values were found in PCA above the ceiling of {ceil} Angstroms \n"

    #Retrim - all datasets should be composed of real values between 0 and ceil, remove empty sets now (if they were all nan essentially)
    rpt_rmsds = [data for i, data in enumerate(rpt_rmsds) if i not in indices_all_nan]
    rpt_pca_rmsds = [data for i, data in enumerate(rpt_pca_rmsds) if i not in indices_all_nan]
    latents2plot = [data for i, data in enumerate(latents) if i not in indices_all_nan]
    
    #detect outliers - calculate median, std, and do not plot above n stds
    rpt_rmsd_medians = [np.median(arr) for arr in rpt_rmsds]
    rpt_rmsd_stds = [np.std(arr) for arr in rpt_rmsds]
    rpt_pca_rmsd_medians = [np.median(arr) for arr in rpt_pca_rmsds]
    rpt_pca_rmsd_stds = [np.std(arr) for arr in rpt_pca_rmsds]
    rpt_maxima = [median + 3*std for median, std in zip(rpt_rmsd_medians, rpt_rmsd_stds)]
    rpt_pca_maxima = [median + 3*std for median, std in zip(rpt_pca_rmsd_medians, rpt_pca_rmsd_stds)]

    figure_log += f"Report the median and standard deviation used for each Latent \n  \t VAE \n"
    for elem in [f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n" for latent, median, std in zip(latents2plot, rpt_rmsd_medians, rpt_rmsd_stds)]:
        figure_log += elem
    figure_log += f"  \t PCA \n"
    for elem in [f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n" for latent, median, std in zip(latents2plot, rpt_pca_rmsd_medians, rpt_pca_rmsd_stds)]:
        figure_log += elem
    figure_log += '\n'
    
    #Reduce charting data to those within the threshold
    rpt_data2plot = [rpt_rmsd[np.where(rpt_rmsd < maximum)] for rpt_rmsd, maximum in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2plot = [rpt_pca_rmsd[np.where(rpt_pca_rmsd < maximum)] for rpt_pca_rmsd, maximum in zip(rpt_pca_rmsds, rpt_pca_maxima)]
    
    rpt_data2exclude = [np.where(rpt_rmsd > maximum)[0].shape[0] for rpt_rmsd, maximum in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2exclude = [np.where(rpt_pca_rmsd > maximum)[0].shape[0] for rpt_pca_rmsd, maximum in zip(rpt_pca_rmsds, rpt_pca_maxima)]

    figure_log += (f"Number of points that are {num_stds:0.3f} stds above the median \n \t VAE \n")
    for elem in [f"\t\t L-{latent} Num-{elem} \n" for latent, elem in zip(latents2plot, rpt_data2exclude)]:
        figure_log += elem
    figure_log += (f" \t PCA \n")
    for elem in [f"\t\t L-{latent} Num-{elem} \n" for latent, elem in zip(latents2plot, rpt_pca_data2exclude)]:
        figure_log += elem
    figure_log += '\n'

    figure_log += (f"Final number of points comprising violin \n \t VAE \n")
    for elem in [f"\t\t L-{latent} Num-{elem.shape[0]} \n" for latent, elem in zip(latents2plot, rpt_data2plot)]:
        figure_log += elem
    figure_log += (f" \t PCA \n")
    for elem in [f"\t\t L-{latent} Num-{elem.shape[0]} \n" for latent, elem in zip(latents2plot, rpt_pca_data2plot)]:
        figure_log += elem
    figure_log += '\n'
    
    plt.clf()
    fig = plt.figure(figsize=(3.25, 3.25))
    #Make the domain linear on n_latents = 1, 2, 3, 4 and log for 8, 16 etc
    domain = [n-0.05 if n <= 4 else 1.95+np.log2(n) if n in two_to else 1.95+index_highest_two_mod(pref2heavycount[pref])[0] for n in latents2plot]
    v1 = plt.violinplot(rpt_data2plot, domain, showextrema=False)
    domain = [n+0.05 if n <= 4 else 2.05+np.log2(n) if n in two_to else 2.05+index_highest_two_mod(pref2heavycount[pref])[0] for n in latents2plot]
    v2 = plt.violinplot(rpt_pca_data2plot, domain, showextrema=False)
    for i, b in enumerate(v1['bodies']):
        b.set_edgecolor('black')
        b.set_alpha(1)
        # get the center (the center is actually the domain point)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color('r')
        
    for b in v2['bodies']:
        b.set_edgecolor('black')
        b.set_alpha(1)
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color('b')
    
    #add some verticle lines where we excluded the NAN values, min and max set by the plotted data
    global_min = np.min([np.min(dataset) for dataset in rpt_data2plot])
    global_max = np.max([np.max(dataset) for dataset in rpt_data2plot])
    latents2vline = [n for n in latents if n not in latents2plot]
    latents2vline = [n-0.05 if n <= 4 else 1.95+np.log2(n) if n in two_to else 1.95+index_highest_two_mod(pref2heavycount[pref])[0] for n in latents2vline]
    
    plt.vlines(latents2vline, ymin=global_min, ymax=global_max,
              colors='red', linestyles='dashed')
    
    plt.xlabel('N_Latents')
    plt.ylabel('Reconstruction Error (Angstrom)')
    plt.title(title)
    #rotation = [0,0,0,0] + [45]*(len(latents)-4)
    labels = [''+str(latent) for latent in latents[:-1]] + [str(pref2heavycount[pref])]
    plt.xticks(ticks=np.arange(1, len(latents)+1),
               labels=labels,
               rotation='vertical', ha='left')
    
    plt.savefig(os.path.join(figure_dir, f'reconstruction_rmsd_{title=}.png'), bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    
    with open(os.path.join(figure_dir, f'figure_generation_log.txt'), 'w') as f:
        figure_log += f"Successfully generated and saved figure and log by {datetime.now()} \n"
        f.write(figure_log)
    
    return fig, figure_log
    

def title_of_model(name):
    if name.startswith('OX_X'):
        return 'Oxycodone'
    elif name.startswith('DA_X'):
        return 'Deca-Alanine'
    elif name.startswith('DA_stretch'):
        return 'Deca-Alanine Helix Stretch'
    elif name.startswith('CR_X'):
        return 'Crambin'
    elif name.startswith('BR_X'):
        return 'BRD4/JQ1'
    elif name.startswith('HIV1p_X'):
        return 'HIV1-Protease'
    elif name.startswith('KOR_X'):
        return 'KOR/ak'
    else:
        raise Exception(f'No title found for {name}!')

def print_lat(i):
    """
    Convert expected index to expected latent (not heavy atom quantity)
    """
    if i in [0,1,2,3]:
        return i + 1
    else:
        return 2**(i-1)

def farlier_part(lines):
    latents, vae_nans, pca_nans = [], [], []
    for line in lines:
        if line.startswith('Analyze Outliers content for'):
            num_latents = line.split(' ')[4]
            index_this_line = lines.index(line)
            #print(index_this_line, num_latents)
            vae_line, pca_line = lines[index_this_line + 2], lines[index_this_line + 3]
            latents.append(num_latents)
            vae_nans.append([elem for elem in vae_line.split(' ') if elem][1])
            pca_nans.append([elem for elem in pca_line.split(' ') if elem][1])
    return latents, vae_nans, pca_nans


def NAN_part(lines):
    latents, vae_nans, pca_nans = [], [], []
    for line in lines:
        if line.startswith('Analyze NAN content for'):
            num_latents = line.split(' ')[4]
            index_this_line = lines.index(line)
            #print(index_this_line, num_latents)
            vae_line, pca_line = lines[index_this_line + 2], lines[index_this_line + 3]
            latents.append(num_latents)
            vae_nans.append([elem for elem in vae_line.split(' ') if elem][1])
            pca_nans.append([elem for elem in pca_line.split(' ') if elem][1])
    return latents, vae_nans, pca_nans


def mean_std_part(lines):
    start_line_index = lines.index([line for line in lines if line.startswith("Report the median and standard deviation")][0])
    end_line_index = lines.index([line for line in lines if line.startswith("Number of points that are ")][0]) - 1
    lines = lines[start_line_index+1:end_line_index]
    halfway = len(lines) // 2
    vae_lines, pca_lines = lines[1:halfway], lines[halfway+1:]

    vae_lats = [[elem for elem in line.split(' ') if elem.startswith('L-')][0][2:] for line in vae_lines]
    vae_meds = [[elem for elem in line.split(' ') if elem.startswith('Med-')][0][4:] for line in vae_lines]
    vae_stds = [[elem for elem in line.split(' ') if elem.startswith('Std-')][0][4:] for line in vae_lines]
    pca_lats = [[elem for elem in line.split(' ') if elem.startswith('L-')][0][2:] for line in pca_lines]
    pca_meds = [[elem for elem in line.split(' ') if elem.startswith('Med-')][0][4:] for line in pca_lines]
    pca_stds = [[elem for elem in line.split(' ') if elem.startswith('Std-')][0][4:] for line in pca_lines]
    #Same Latents, same length of everything
    assert vae_lats == pca_lats and len(vae_meds) == len(vae_stds) and \
           len(pca_meds) == len(pca_stds) and len(vae_lats) == len(vae_meds) and len(vae_meds) == len(vae_stds)

    return vae_lats, vae_meds, vae_stds, pca_meds, pca_stds

def num_above_mean_std_part(lines):
    start_line_index = lines.index([line for line in lines if line.startswith("Number of points that are ")][0])
    end_line_index = lines.index([line for line in lines if line.startswith("Final number of points comprising violin")][0]) - 1
    lines = lines[start_line_index+1:end_line_index]
    halfway = len(lines) // 2
    vae_lines, pca_lines = lines[1:halfway], lines[halfway+1:]

    vae_lats = [[elem for elem in line.split(' ') if elem.startswith('L-')][0][2:] for line in vae_lines]
    vae_nums = [[elem for elem in line.split(' ') if elem.startswith('Num-')][0][4:] for line in vae_lines]
    
    pca_lats = [[elem for elem in line.split(' ') if elem.startswith('L-')][0][2:] for line in pca_lines]
    pca_nums = [[elem for elem in line.split(' ') if elem.startswith('Num-')][0][4:] for line in pca_lines]

    #Same latents, same length of everything
    assert vae_lats == pca_lats and len(vae_nums) == len(vae_lats) and len(pca_nums) == len(vae_nums)

    return vae_lats, vae_nums, pca_nums

def num_in_violin(lines):
    start_line_index = lines.index([line for line in lines if line.startswith("Final number of points comprising violin")][0])
    end_line_index = lines.index([line for line in lines if line.startswith("Successfully generated and saved figure")][0]) - 1
    lines = lines[start_line_index+1:end_line_index]
    halfway = len(lines) // 2
    vae_lines, pca_lines = lines[1:halfway], lines[halfway+1:]

    vae_lats = [[elem for elem in line.split(' ') if elem.startswith('L-')][0][2:] for line in vae_lines]
    vae_nums = [[elem for elem in line.split(' ') if elem.startswith('Num-')][0][4:] for line in vae_lines]
    
    pca_lats = [[elem for elem in line.split(' ') if elem.startswith('L-')][0][2:] for line in pca_lines]
    pca_nums = [[elem for elem in line.split(' ') if elem.startswith('Num-')][0][4:] for line in pca_lines]

    #Same latents, same length of everything
    assert vae_lats == pca_lats and len(vae_nums) == len(vae_lats) and len(pca_nums) == len(vae_nums)

    return vae_lats, vae_nums, pca_nums

def figure_log_to_csv(figure_log_fn):
    """
    Convert the figure log string to a csv string

    Columns:
        NAN - Number of particles that were NAN on passthrough
        > 10 A - had reconstruction error greater than 1 nanometer
        Median (Ang) - median of the violin distribution reported in Angstrom
        Std (Ang) - standard deviation of the violin distribution in Angstrom
        N Omit - Number of particles that are greater than n standard deviations from median (default n=3)
        N Violin - Number of particles that are present in the violin histogram

    Each column should be generated for each latent and each method (for latent in latents: for method in VAE PCA:)
    """
    with open(figure_log_fn, 'r') as f:
        lines = [line for line in f.readlines()]
    name = [line for line in lines if 'title' in line][0][29:-3]
    file_contents = f",,{name},,,,,,\n"
    cols = ["NAN", "> 10 A", "Median (Ang)", "Std (Ang)", "N Omit", "N Violin"]
    file_contents += f"Latents,,"+ ','.join(cols) + '\n'

    latents, vae_nan, pca_nan = NAN_part(lines)
    _, vae_farlier, pca_farlier = farlier_part(lines)
    latents_plotted, vae_meds, vae_stds, pca_meds, pca_stds = mean_std_part(lines)
    _, vae_outlier, pca_outlier = num_above_mean_std_part(lines)
    _, vae_violin, pca_violin = num_in_violin(lines)

    vae_lines = []
    for i, latent in enumerate(latents):
        num_nan, num_far = vae_nan[i], vae_farlier[i]
        if latent in latents_plotted:
            i = latents_plotted.index(latent)
            median, std, num_outlier, num_violin = vae_meds[i], vae_stds[i], vae_outlier[i], vae_violin[i]
        else:
            median, std, num_outlier, num_violin = 0, 0, 0, 0
        vae_lines.append(f"{latent},VAE (red),{num_nan},{num_far},{median},{std},{num_outlier},{num_violin}\n")
    
    pca_lines = []
    for i, latent in enumerate(latents):
        num_nan, num_far = pca_nan[i], pca_farlier[i]
        if latent in latents_plotted:
            i = latents_plotted.index(latent)
            median, std, num_outlier, num_violin = pca_meds[i], pca_stds[i], pca_outlier[i], pca_violin[i]
        else:
            median, std, num_outlier, num_violin = 0, 0, 0, 0
        pca_lines.append(f"{latent},PCA (blue),{num_nan},{num_far},{median},{std},{num_outlier},{num_violin}\n")

    assert len(vae_lines) == len(pca_lines)
    
    for i, line in enumerate(vae_lines):
        file_contents += line
        file_contents += pca_lines[i]
        
    return file_contents

def reorganize_figures_and_logs(figure_dir):
    import glob, shutil
    unique_models = []
    for content in os.listdir(figure_dir):
        if content.startswith('X') or '_' not in content:
            continue
        name_parts = content.split('_')
        if name_parts[-1].startswith('X') and name_parts[-1] not in unique_models:
            unique_models.append(name_parts[-1])
    #print(unique_models)
    for model_name in sorted(unique_models):
        if not os.path.isdir(os.path.join(figure_dir, model_name)):
            os.makedirs(os.path.join(figure_dir, model_name))
        for content in os.listdir(figure_dir):
            if content.startswith('X') or '_' not in content:
                continue
            name_parts = content.split('_')
            if name_parts[-1] == model_name:
                pngs = sorted(glob.glob(os.path.join(figure_dir, content, '*.png')))
                #print(pngs)
                logs = sorted(glob.glob(os.path.join(figure_dir, content, 'figure_generation_log.txt')))
                #print(logs)
                for png, log in zip(pngs, logs):
                    shutil.copy(png, os.path.join(figure_dir, model_name, os.path.basename(png)))
                    shutil.copy(log, os.path.join(figure_dir, model_name, os.path.basename(png)[:-4]+"_"+os.path.basename(log)))


def parse_best_models(model_list_fn):
    with open(model_list_fn, 'r') as f:
        lines = [elem.split('/') for elem in f.read().split('\n') if elem]
    model_names = []
    prefs = []
    for line_parts in lines:
        model_name2load = line_parts[2]
        pref = line_parts[2].split('_')[0] + '_'
        if model_name2load not in model_names:
            model_names.append(model_name2load)
            prefs.append(pref)
    latent_sets = [[] for model_name in model_names]
    rpt_sets = [[] for model_name in model_names]
    
    
    for i, (model_name, pref) in enumerate(zip(model_names, prefs)):
        for line_parts in lines:
            #print(line_parts)
            model_name2load = line_parts[2]
            this_pref = line_parts[2].split('_')[0] + '_'
            latent = int(line_parts[-1].split('_')[-2])
            rpt = int(line_parts[-1].split('_')[-1].split('.')[0])
            #print(latent not in latent_sets[i], model_name2load==model_name, this_pref==pref)
            if latent not in latent_sets[i] and model_name2load==model_name and this_pref==pref:
                latent_sets[i].append(latent)
                rpt_sets[i].append(rpt)
    return model_names, prefs, latent_sets, rpt_sets


def operate(heavy_atom_analyzer, data, rng_seed):
    key = jax.random.PRNGKey(rng_seed)
    main_key, dropout_key = jax.random.split(key, num=2)
    arg_dict = {'params': heavy_atom_analyzer.state.params}
    if heavy_atom_analyzer.is_batchnorm:
        arg_dict['batch_stats'] = heavy_atom_analyzer.state.batch_stats
    decoded, latent_means, latent_vars = heavy_atom_analyzer.state.apply_fn(arg_dict, data, main_key, train=False, rngs={'dropout': dropout_key})
    return decoded, latent_means, latent_vars

def cross_compare_models(json1, json2, rng_seed=69420):
    """
    Operate on and calculate reconstructive error for test data of the other model
    returns 
        
        arr1 = error vals associated with model 1 reconstructing the test data of model 2
        arr2 = error vals associated with model 2 reconstructing the test data of model 1
    """
    haas = [HeavyAtom_Analyzer(j_fn) for j_fn in [json1, json2]]
    assert haas[0].n_latents == haas[-1].n_latents
    decoded12, _, _ = operate(haas[0], haas[1].test_data, rng_seed=rng_seed)
    decoded21, _, _ = operate(haas[1], haas[0].test_data, rng_seed=rng_seed)
    
    err12 = atom_rmsd(haas[1].test_data, decoded12)
    err21 = atom_rmsd(haas[0].test_data, decoded21)

    return err12, err21

#'/ocean/projects/cis250004p/josephdb/Deep-MMS/difference/'
def compare_and_save(json1, json2, save_dir):
    model_names = [elem.split('/')[-2] for elem in [json1, json2]]
    er1, er2 = cross_compare_models('json_inputs/X013-2/DA_X013-2/DA_X013-2_0004_03.json',
                                    'json_inputs/X013-1/DA_X013-1/DA_X013-1_0004_03.json')
    np.save(os.path.join(save_dir, f'{model_names[0]}_reconstructing_{model_names[1]}.npy'), er1)
    np.save(os.path.join(save_dir, f'{model_names[1]}_reconstructing_{model_names[0]}.npy'), er2)
    return True

def determine_comparisons(base_model):
    best_models_fns = sorted(glob.glob('best_model_list_X013-*.txt'))
    base_model_index = [base_model in name for name in best_models_fns].index(True)
    with open(best_models_fns[base_model_index], 'r') as f:
        base_model_lines = [elem.split('/') for elem in f.read().split('\n') if elem]

    other_model_lines = []
    for i, best_fn in enumerate(best_models_fns):
        if i != base_model_index:
            with open(best_fn, 'r') as f:
                other_model_lines.append([elem.split('/') for elem in f.read().split('\n') if elem])

    for base, (others) in zip(base_model_lines, zip(*other_model_lines)):
        for other in others:
            path1, path2 = '/'.join(base), '/'.join(other)
            assert False not in [os.path.isfile(path) for path in [path1, path2]]
            print(f"sbatch compare_violin.job {path1} {path2}")

#Difference Violin Plot
def pself(self):
    print(f"{self=}")


def violin_difference_plot(model_A_name2load, model_B_name2load, pref, figure_dir=None, ceil=10.0, num_stds=3.0, show=False):
    """
    Plot a doube violin plot where:
        Left violin indicates the difference error (of the two models) of operating on model A's test data
        Right violin indicates the difference error (of the two models) of operating on model B's test data
        Values above zero indicate Model A is more accurate on reconstructing the data
        Values below zero indicate Model B is more accurate on reconstructing the data
    """

    title = f"Compare_{model_A_name2load}_{model_B_name2load}"
        
    figure_log = f"Began {datetime.now()} \n"
    figure_log += f"Log for making figure {title=} \n"

    #Load model A on its own data
    chart_data_11 = load_files(glob.glob(f'numpy_backups/{model_A_name2load}/*/*/*.npy'), verbose=False)
    means_11, best_rpts_rmsd_11, best_rpts_loss_11 = lowest_rpts(chart_data_11)
    rmsds_11 = {key: val[best_rpts_rmsd_11[key]]['VAE_RMSD'] for key, val in chart_data_11.items()}
    #Load MOdel B on its own data
    chart_data_22 = load_files(glob.glob(f'numpy_backups/{model_B_name2load}/*/*/*.npy'), verbose=False)
    means_22, best_rpts_rmsd_22, best_rpts_loss_22 = lowest_rpts(chart_data_22)
    rmsds_22 = {key: val[best_rpts_rmsd_22[key]]['VAE_RMSD'] for key, val in chart_data_22.items()}

    assert [int(key) for key in rmsds_11] == [int(key) for key in rmsds_22]
    #Data of ModelA operating on modelB's test data
    data_files_12 = sorted([file for file in glob.glob(f'difference/{model_A_name2load}*reconstructing*{model_B_name2load}*.npy') if '_0' in file or '_1' in file])
    rmsds_12 = {key: np.load(fn) for key, fn in zip([int(os.path.basename(fn).split('reconstructing')[0][-5:-1]) for fn in data_files_12], data_files_12)}
    #Data of modelB on modelA's test data
    data_files_21 = sorted([file for file in glob.glob(f'difference/{model_B_name2load}*reconstructing*{model_A_name2load}*.npy') if '_0' in file or '_1' in file])
    rmsds_21 = {key: np.load(fn) for key, fn in zip([int(os.path.basename(fn).split('reconstructing')[0][-5:-1]) for fn in data_files_21], data_files_21)}
    assert [int(key) for key in rmsds_11] == [int(key) for key in rmsds_12]
    assert [int(key) for key in rmsds_11] == [int(key) for key in rmsds_21]

    #Data_ij = the error associated with operating modeli against testdataj  perform substractions where j matches
    # Above zero should indicate model A more accurate (B-A) gives positive when A is smaller than be, always subtract i=2 - i=1
    
    sizes_A = [elem_0.shape[0] - elem_1.shape[0] if elem_0.shape[0] - elem_1.shape[0] != 0  else None for elem_0, elem_1 in zip(rmsds_21.values(), rmsds_11.values())]
    err_diff_A = {}
    for (key, elem_0), elem_1, size in zip(rmsds_21.items(), rmsds_11.values(), sizes_A):
        if size is None:
            err_diff_A[key] = elem_0 - elem_1
        elif size > 0:
            err_diff_A[key] = elem_0[:elem_1.shape[0]] - elem_1
        elif size < 0:
            err_diff_A[key] = elem_0 - elem_1[:elem_0.shape[0]]
        else:
            raise Exception("That should'n't'v'e been possible")

    sizes_B = [elem_0.shape[0] - elem_1.shape[0] if elem_0.shape[0] - elem_1.shape[0] != 0  else None for elem_0, elem_1 in zip(rmsds_22.values(), rmsds_12.values())]
    err_diff_B = {}
    for (key, elem_0), elem_1, size in zip(rmsds_22.items(), rmsds_12.values(), sizes_B):
        if size is None:
            err_diff_B[key] = elem_0 - elem_1
        elif size > 0:
            err_diff_B[key] = elem_0[:elem_1.shape[0]] - elem_1
        elif size < 0:
            err_diff_B[key] = elem_0 - elem_1[:elem_0.shape[0]]
        else:
            raise Exception("That should'n't'v'e been possible")
    latents = [int(key) for key in err_diff_A]
    
    #slyly convert to this system
    #rmsds and pca rmsds are a dict of results with keys n_latent and vals hist_vals
    rpt_rmsds = [10*err_diff_A[key] for key in err_diff_A.keys()] #convert to angstrom
    rpt_pca_rmsds = [10*err_diff_B[key] for key in err_diff_B.keys()] #convert to angstrom
    og_shapes, og_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]
    
    #remove nan values
    rpt_rmsds = [data[np.where(~np.isnan(data))] for data in rpt_rmsds]
    rpt_pca_rmsds = [data[np.where(~np.isnan(data))] for data in rpt_pca_rmsds]
    non_nan_shapes, non_nan_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]
    pself(rpt_rmsds)
    pself(rpt_pca_rmsds)
    #Report the number of Nan values, and remove datasets with all Nan values
    indices_all_nan = []
    for i, (latent, old1, old2, new1, new2) in enumerate(zip(latents, og_shapes, og_pca_shapes, non_nan_shapes, non_nan_pca_shapes)):
        figure_log += f"Analyze NAN content for {latent} Latents \n"
        figure_log += f"For model with {latent} Latents, \n \t {old1-new1} Nan values were found in VAE \n \t {old2 - new2} Nan values were found in PCA \n"
        if new1 == 0:
            figure_log += f"For model with {latent} Latents, \n \t All values are NAN - a dashed line on the plot \n"
            indices_all_nan.append(i)

    #Now remove values above a ridiculous threshold, like 10 angstroms
    rpt_rmsds = [data[np.where(data < ceil)] for data in rpt_rmsds]
    rpt_pca_rmsds = [data[np.where(data < ceil)] for data in rpt_pca_rmsds]
    in_thresh_shapes, in_thresh_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]

    #Report the number of values above the threshold
    for i, (latent, old1, old2, new1, new2) in enumerate(zip(latents, non_nan_shapes, non_nan_pca_shapes, in_thresh_shapes, in_thresh_pca_shapes)):
        figure_log += f"Analyze Outliers content for {latent} Latents \n"
        figure_log += f"For model with {latent} Latents, \n \t {old1-new1} values were found in VAE above the ceiling of {ceil} Angstroms \n \t {old2 - new2} values were found in PCA above the ceiling of {ceil} Angstroms \n"

    #And specially for this, above a floor
    rpt_rmsds = [data[np.where(data > -1*ceil)] for data in rpt_rmsds]
    rpt_pca_rmsds = [data[np.where(data > -1*ceil)] for data in rpt_pca_rmsds]
    in_thresh_shapes, in_thresh_pca_shapes = [data.shape[0] for data in rpt_rmsds], [data.shape[0] for data in rpt_pca_rmsds]

    #Report the number of values above the threshold
    for i, (latent, old1, old2, new1, new2) in enumerate(zip(latents, non_nan_shapes, non_nan_pca_shapes, in_thresh_shapes, in_thresh_pca_shapes)):
        figure_log += f"Analyze Outliers content for {latent} Latents \n"
        figure_log += f"For model with {latent} Latents, \n \t {old1-new1} values were found in VAE above the ceiling of {ceil} Angstroms or below the floor of {-1*ceil} \n \t {old2 - new2} values were found in PCA above the ceiling of {ceil} Angstroms \n"


    #Retrim - all datasets should be composed of real values between 0 and ceil, remove empty sets now (if they were all nan essentially)
    rpt_rmsds = [data for i, data in enumerate(rpt_rmsds) if i not in indices_all_nan]
    rpt_pca_rmsds = [data for i, data in enumerate(rpt_pca_rmsds) if i not in indices_all_nan]
    latents2plot = [data for i, data in enumerate(latents) if i not in indices_all_nan]
    
    #detect outliers - calculate median, std, and do not plot above n stds
    rpt_rmsd_medians = [np.median(arr) for arr in rpt_rmsds]
    rpt_rmsd_stds = [np.std(arr) for arr in rpt_rmsds]
    rpt_pca_rmsd_medians = [np.median(arr) for arr in rpt_pca_rmsds]
    rpt_pca_rmsd_stds = [np.std(arr) for arr in rpt_pca_rmsds]
    rpt_maxima = [median + 3*std for median, std in zip(rpt_rmsd_medians, rpt_rmsd_stds)]
    rpt_pca_maxima = [median + 3*std for median, std in zip(rpt_pca_rmsd_medians, rpt_pca_rmsd_stds)]

    figure_log += f"Report the median and standard deviation used for each Latent \n  \t VAE \n"
    for elem in [f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n" for latent, median, std in zip(latents2plot, rpt_rmsd_medians, rpt_rmsd_stds)]:
        figure_log += elem
    figure_log += f"  \t PCA \n"
    for elem in [f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n" for latent, median, std in zip(latents2plot, rpt_pca_rmsd_medians, rpt_pca_rmsd_stds)]:
        figure_log += elem
    figure_log += '\n'
    
    #Reduce charting data to those within the threshold
    rpt_data2plot = [rpt_rmsd[np.where(rpt_rmsd < maximum)] for rpt_rmsd, maximum in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2plot = [rpt_pca_rmsd[np.where(rpt_pca_rmsd < maximum)] for rpt_pca_rmsd, maximum in zip(rpt_pca_rmsds, rpt_pca_maxima)]
    
    rpt_data2exclude = [np.where(rpt_rmsd > maximum)[0].shape[0] for rpt_rmsd, maximum in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2exclude = [np.where(rpt_pca_rmsd > maximum)[0].shape[0] for rpt_pca_rmsd, maximum in zip(rpt_pca_rmsds, rpt_pca_maxima)]

    figure_log += (f"Number of points that are {num_stds:0.3f} stds above the median \n \t VAE \n")
    for elem in [f"\t\t L-{latent} Num-{elem} \n" for latent, elem in zip(latents2plot, rpt_data2exclude)]:
        figure_log += elem
    figure_log += (f" \t PCA \n")
    for elem in [f"\t\t L-{latent} Num-{elem} \n" for latent, elem in zip(latents2plot, rpt_pca_data2exclude)]:
        figure_log += elem
    figure_log += '\n'

    figure_log += (f"Final number of points comprising violin \n \t VAE \n")
    for elem in [f"\t\t L-{latent} Num-{elem.shape[0]} \n" for latent, elem in zip(latents2plot, rpt_data2plot)]:
        figure_log += elem
    figure_log += (f" \t PCA \n")
    for elem in [f"\t\t L-{latent} Num-{elem.shape[0]} \n" for latent, elem in zip(latents2plot, rpt_pca_data2plot)]:
        figure_log += elem
    figure_log += '\n'
    
    plt.clf()
    fig = plt.figure(figsize=(3.25, 3.25))
    plt.hlines(y=0.0, xmin=0.9, xmax=2.1+index_highest_two_mod(pref2heavycount[pref])[0], linestyle='--', color='k', zorder=-1)
    #Make the domain linear on n_latents = 1, 2, 3, 4 and log for 8, 16 etc
    domain = [n-0.05 if n <= 4 else 1.95+np.log2(n) if n in two_to else 1.95+index_highest_two_mod(pref2heavycount[pref])[0] for n in latents2plot]
    print([val.shape for val in rpt_data2plot])
    v1 = plt.violinplot(rpt_data2plot, domain, showextrema=False)
    domain = [n+0.05 if n <= 4 else 2.05+np.log2(n) if n in two_to else 2.05+index_highest_two_mod(pref2heavycount[pref])[0] for n in latents2plot]
    print([val.shape for val in rpt_pca_data2plot])
    v2 = plt.violinplot(rpt_pca_data2plot, domain, showextrema=False)

    for i, b in enumerate(v1['bodies']):
        b.set_edgecolor('black')
        b.set_alpha(1)
        # get the center (the center is actually the domain point)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color((0,0.5,0.5))
        
    for b in v2['bodies']:
        b.set_edgecolor('black')
        b.set_alpha(1)
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further left than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color((0.5,0,0.5))
    
        #add some verticle lines where we excluded the NAN values, min and max set by the plotted data
    global_min = np.min([np.min(dataset) for dataset in rpt_data2plot])
    global_max = np.max([np.max(dataset) for dataset in rpt_data2plot])
    latents2vline = [n for n in latents if n not in latents2plot]
    latents2vline = [n-0.05 if n <= 4 else 1.95+np.log2(n) if n in two_to else 1.95+index_highest_two_mod(pref2heavycount[pref])[0] for n in latents2vline]
    
    plt.vlines(latents2vline, ymin=global_min, ymax=global_max,
              colors='red', linestyles='dashed')
    plt.xlabel('N_Latents')
    plt.ylabel('Error Difference (Angstrom)')
    plt.title(title.replace('_', ' '))
    #rotation = [0,0,0,0] + [45]*(len(latents)-4)
    plt.xticks(ticks=np.arange(1, len(latents)+1),
               labels=[''+str(latent) for latent in latents],
               rotation='vertical')#, ha='left')
    
    if figure_dir:
        print(f"Saving at {os.path.join(figure_dir, f'reconstruction_diff_err_{title=}.png')}")
        plt.savefig(os.path.join(figure_dir, f'reconstruction_diff_err_{title=}.png'), bbox_inches='tight')
        with open(os.path.join(figure_dir, f'figure_generation_log_diff_err_{title=}.txt'), 'w') as f:
            figure_log += f"Successfully generated and saved figure and log by {datetime.now()} \n"
            f.write(figure_log)
        
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, figure_log
