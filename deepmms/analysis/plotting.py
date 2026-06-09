"""
Plotting utilities for reconstruction-error violin charts and figure log parsing.

Provides violin_plots for split half-violin comparisons of VAE vs PCA RMSD,
violin_difference_plot for cross-model comparisons, figure_log_to_csv for
converting figure logs to CSV summaries, and several helper functions for
loading multi-run numpy result files.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

two_to = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

topology_map = {
    "BR_": "Simulation/3mxf_implicit.pdb",
    "CR_": "Simulation/1crn_H.pdb",
    "DA_": "Simulation/ala_deca_peptide.pdb",
    "DA_stretch_": "Simulation/ala_deca_peptide.pdb",
    "HIV1p_": "Simulation/HIV1p_protein_only.pdb",
    "OX_": "Simulation/oxycodone.pdb",
}

pref2heavycount = {
    "OX_": 23,
    "DA_": 50,
    "DA_stretch_": 50,
    "CR_": 327,
    "CR_small_": 327,
    "BR_": 1093,
    "HIV1p_": 1599,
}

base_dict = {
    "VAE_RMSD": None,
    "VAE_LOSS_RMSD": None,
    "PCA_RMSD": None,
    "PCA_LOSS_RMSD": None,
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def index_highest_two_mod(value):
    """Return indices of powers-of-two in two_to that are less than value."""
    return np.where(value - two_to < 0)[0]


def load_files(numpy_files, verbose=True):
    """
    Load per-run .npy reconstruction error files into a nested dict.

    Expects file paths of the form model_name/n_latents_latents/rpt/filename.npy.

    Parameters
    ----------
    numpy_files : list of str
        Glob result of .npy paths to load.
    verbose : bool
        Print messages for excluded or failed files.

    Returns
    -------
    dict
        {n_latents: {rpt: {metric_name: np.ndarray}}}
    """
    fns = sorted(numpy_files)
    attrs = [
        fn.split("/")[1:] if fn.split("/")[0] == "numpy_backups" else fn.split("/")
        for fn in fns
    ]
    model_name = attrs[0][0]
    assert all(elem[0] == model_name for elem in attrs)

    model_data = {}
    for attr_set in attrs:
        print(attr_set)
        n_latents = int(attr_set[1].split("_")[0])
        model_data.setdefault(n_latents, {})
        rpt = attr_set[2]
        model_data[n_latents].setdefault(rpt, deepcopy(base_dict))

        try:
            key = attr_set[-1].split(".")[0]
            if key in base_dict:
                model_data[n_latents][rpt][key] = np.load(os.path.join(*attr_set))
            elif verbose:
                print("Excluding", os.path.join(*attr_set))
        except Exception:
            if verbose:
                print("Failure on", os.path.join(*attr_set))

    return model_data


def lowest_rpts(chart_data):
    """
    For each latent count find the repetition with the lowest mean VAE RMSD/loss.

    Parameters
    ----------
    chart_data : dict
        Output of load_files.

    Returns
    -------
    means : dict
    best_rpts_RMSD : dict  {n_latents: rpt_key}
    best_rpts_LOSS : dict  {n_latents: rpt_key}
    """
    means = {}
    for key, val in chart_data.items():
        means[key] = {}
        for key2, val2 in val.items():
            means[key][key2] = {k: np.mean(v) for k, v in val2.items()}

    best_rpts_RMSD = {
        key: min(val.items(), key=lambda x: x[-1]["VAE_RMSD"])[0]
        for key, val in means.items()
    }
    best_rpts_LOSS = {
        key: min(val.items(), key=lambda x: x[-1]["VAE_LOSS_RMSD"])[0]
        for key, val in means.items()
    }
    return means, best_rpts_RMSD, best_rpts_LOSS


def title_of_model(name):
    """Map a model prefix string to a human-readable title."""
    mapping = {
        "OX_X": "Oxycodone",
        "DA_X": "Deca-Alanine",
        "DA_stretch": "Deca-Alanine Helix Stretch",
        "CR_X": "Crambin",
        "BR_X": "BRD4/JQ1",
        "HIV1p_X": "HIV1-Protease",
        "KOR_X": "KOR/ak",
    }
    for prefix, title in mapping.items():
        if name.startswith(prefix):
            return title
    raise ValueError(f"No title found for {name}!")


def parse_best_models(model_list_fn):
    """
    Parse a best-model list file into structured model/latent/rpt lists.

    Parameters
    ----------
    model_list_fn : str
        Path to a text file with one model path per line.

    Returns
    -------
    model_names : list of str
    prefs : list of str
    latent_sets : list of lists
    rpt_sets : list of lists
    """
    with open(model_list_fn, "r") as f:
        lines = [elem.split("/") for elem in f.read().split("\n") if elem]
    model_names, prefs = [], []
    for line_parts in lines:
        model_name2load = line_parts[2]
        pref = line_parts[2].split("_")[0] + "_"
        if model_name2load not in model_names:
            model_names.append(model_name2load)
            prefs.append(pref)

    latent_sets = [[] for _ in model_names]
    rpt_sets = [[] for _ in model_names]
    for i, (model_name, pref) in enumerate(zip(model_names, prefs)):
        for line_parts in lines:
            model_name2load = line_parts[2]
            this_pref = line_parts[2].split("_")[0] + "_"
            latent = int(line_parts[-1].split("_")[-2])
            rpt = int(line_parts[-1].split("_")[-1].split(".")[0])
            if latent not in latent_sets[i] and model_name2load == model_name and this_pref == pref:
                latent_sets[i].append(latent)
                rpt_sets[i].append(rpt)
    return model_names, prefs, latent_sets, rpt_sets


# ---------------------------------------------------------------------------
# Main violin plot
# ---------------------------------------------------------------------------

def violin_plots(
    vae_dat, pca_dat, latents, title, pref,
    ceil=10.0, num_stds=3, show=False, figure_dir=None, test_hists=False,
):
    """
    Produce a split half-violin plot comparing VAE (red, left) to PCA (blue, right).

    Removes NaN values, clips outliers beyond ceil angstroms, then further clips
    points more than num_stds standard deviations above the median.  Appends a
    full figure-generation log to a text file in figure_dir.

    Parameters
    ----------
    vae_dat : dict  {n_latents: np.ndarray}
        Per-frame VAE RMSD arrays in nanometres; converted to Å internally.
    pca_dat : dict  {n_latents: np.ndarray}
        Per-frame PCA RMSD arrays in nanometres.
    latents : list of int
        Ordered latent counts corresponding to dict keys.
    title : str
        Figure title and part of the saved filename.
    pref : str
        Model prefix key into pref2heavycount (e.g. 'CR_').
    ceil : float
        Upper threshold in angstroms for outlier removal.
    num_stds : int or float
        Number of median-standard-deviations above which points are excluded.
    show : bool
        Call plt.show() when True; otherwise close the figure.
    figure_dir : str or None
        Directory to save the figure and log; skipped when None.
    test_hists : bool
        Reserved for future use; not currently used.

    Returns
    -------
    fig : matplotlib.figure.Figure
    figure_log : str
        Full generation log string.
    """
    figure_log = f"Began {datetime.now()} \n"
    figure_log += f"Log for making figure {title=} \n"

    rpt_rmsds = [10 * vae_dat[key] for key in vae_dat]
    rpt_pca_rmsds = [10 * pca_dat[key] for key in pca_dat]
    og_shapes = [d.shape[0] for d in rpt_rmsds]
    og_pca_shapes = [d.shape[0] for d in rpt_pca_rmsds]

    rpt_rmsds = [d[~np.isnan(d)] for d in rpt_rmsds]
    rpt_pca_rmsds = [d[~np.isnan(d)] for d in rpt_pca_rmsds]
    non_nan_shapes = [d.shape[0] for d in rpt_rmsds]
    non_nan_pca_shapes = [d.shape[0] for d in rpt_pca_rmsds]

    indices_all_nan = []
    for i, (latent, old1, old2, new1, new2) in enumerate(
        zip(latents, og_shapes, og_pca_shapes, non_nan_shapes, non_nan_pca_shapes)
    ):
        figure_log += f"Analyze NAN content for {latent} Latents \n"
        figure_log += (
            f"For model with {latent} Latents, \n"
            f" \t {old1-new1} Nan values were found in VAE \n"
            f" \t {old2-new2} Nan values were found in PCA \n"
        )
        if new1 == 0:
            figure_log += (
                f"For model with {latent} Latents, \n"
                f" \t All values are NAN - a dashed line on the plot \n"
            )
            indices_all_nan.append(i)

    rpt_rmsds = [d[d < ceil] for d in rpt_rmsds]
    rpt_pca_rmsds = [d[d < ceil] for d in rpt_pca_rmsds]
    in_thresh_shapes = [d.shape[0] for d in rpt_rmsds]
    in_thresh_pca_shapes = [d.shape[0] for d in rpt_pca_rmsds]

    for i, (latent, old1, old2, new1, new2) in enumerate(
        zip(latents, non_nan_shapes, non_nan_pca_shapes, in_thresh_shapes, in_thresh_pca_shapes)
    ):
        figure_log += f"Analyze Outliers content for {latent} Latents \n"
        figure_log += (
            f"For model with {latent} Latents, \n"
            f" \t {old1-new1} values were found in VAE above the ceiling of {ceil} Angstroms \n"
            f" \t {old2-new2} values were found in PCA above the ceiling of {ceil} Angstroms \n"
        )

    rpt_rmsds = [d for i, d in enumerate(rpt_rmsds) if i not in indices_all_nan]
    rpt_pca_rmsds = [d for i, d in enumerate(rpt_pca_rmsds) if i not in indices_all_nan]
    latents2plot = [d for i, d in enumerate(latents) if i not in indices_all_nan]

    rpt_rmsd_medians = [np.median(a) for a in rpt_rmsds]
    rpt_rmsd_stds = [np.std(a) for a in rpt_rmsds]
    rpt_pca_rmsd_medians = [np.median(a) for a in rpt_pca_rmsds]
    rpt_pca_rmsd_stds = [np.std(a) for a in rpt_pca_rmsds]
    rpt_maxima = [m + 3 * s for m, s in zip(rpt_rmsd_medians, rpt_rmsd_stds)]
    rpt_pca_maxima = [m + 3 * s for m, s in zip(rpt_pca_rmsd_medians, rpt_pca_rmsd_stds)]

    figure_log += "Report the median and standard deviation used for each Latent \n  \t VAE \n"
    for latent, median, std in zip(latents2plot, rpt_rmsd_medians, rpt_rmsd_stds):
        figure_log += f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n"
    figure_log += "  \t PCA \n"
    for latent, median, std in zip(latents2plot, rpt_pca_rmsd_medians, rpt_pca_rmsd_stds):
        figure_log += f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n"
    figure_log += "\n"

    rpt_data2plot = [d[d < m] for d, m in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2plot = [d[d < m] for d, m in zip(rpt_pca_rmsds, rpt_pca_maxima)]
    rpt_data2exclude = [np.where(d > m)[0].shape[0] for d, m in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2exclude = [np.where(d > m)[0].shape[0] for d, m in zip(rpt_pca_rmsds, rpt_pca_maxima)]

    figure_log += f"Number of points that are {num_stds:0.3f} stds above the median \n \t VAE \n"
    for latent, elem in zip(latents2plot, rpt_data2exclude):
        figure_log += f"\t\t L-{latent} Num-{elem} \n"
    figure_log += " \t PCA \n"
    for latent, elem in zip(latents2plot, rpt_pca_data2exclude):
        figure_log += f"\t\t L-{latent} Num-{elem} \n"
    figure_log += "\n"

    figure_log += "Final number of points comprising violin \n \t VAE \n"
    for latent, elem in zip(latents2plot, rpt_data2plot):
        figure_log += f"\t\t L-{latent} Num-{elem.shape[0]} \n"
    figure_log += " \t PCA \n"
    for latent, elem in zip(latents2plot, rpt_pca_data2plot):
        figure_log += f"\t\t L-{latent} Num-{elem.shape[0]} \n"
    figure_log += "\n"

    plt.clf()
    fig = plt.figure(figsize=(3.25, 3.25))
    domain = [
        n - 0.05 if n <= 4
        else 1.95 + np.log2(n) if n in two_to
        else 1.95 + index_highest_two_mod(pref2heavycount[pref])[0]
        for n in latents2plot
    ]
    v1 = plt.violinplot(rpt_data2plot, domain, showextrema=False)
    domain = [
        n + 0.05 if n <= 4
        else 2.05 + np.log2(n) if n in two_to
        else 2.05 + index_highest_two_mod(pref2heavycount[pref])[0]
        for n in latents2plot
    ]
    v2 = plt.violinplot(rpt_pca_data2plot, domain, showextrema=False)

    for b in v1["bodies"]:
        b.set_edgecolor("black")
        b.set_alpha(1)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color("r")

    for b in v2["bodies"]:
        b.set_edgecolor("black")
        b.set_alpha(1)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color("b")

    global_min = np.min([np.min(d) for d in rpt_data2plot])
    global_max = np.max([np.max(d) for d in rpt_data2plot])
    latents2vline = [n for n in latents if n not in latents2plot]
    latents2vline = [
        n - 0.05 if n <= 4
        else 1.95 + np.log2(n) if n in two_to
        else 1.95 + index_highest_two_mod(pref2heavycount[pref])[0]
        for n in latents2vline
    ]
    plt.vlines(latents2vline, ymin=global_min, ymax=global_max, colors="red", linestyles="dashed")

    plt.xlabel("N_Latents")
    plt.ylabel("Reconstruction Error (Angstrom)")
    plt.title(title)
    labels = ["" + str(latent) for latent in latents[:-1]] + [str(pref2heavycount[pref])]
    plt.xticks(
        ticks=np.arange(1, len(latents) + 1),
        labels=labels,
        rotation="vertical",
        ha="left",
    )

    if figure_dir:
        plt.savefig(
            os.path.join(figure_dir, f"reconstruction_rmsd_{title=}.png"),
            bbox_inches="tight",
        )
        with open(os.path.join(figure_dir, "figure_generation_log.txt"), "w") as f:
            figure_log += f"Successfully generated and saved figure and log by {datetime.now()} \n"
            f.write(figure_log)

    if show:
        plt.show()
    else:
        plt.close()

    return fig, figure_log


# ---------------------------------------------------------------------------
# Difference violin plot
# ---------------------------------------------------------------------------

def violin_difference_plot(
    model_A_name2load, model_B_name2load, pref,
    figure_dir=None, ceil=10.0, num_stds=3.0, show=False,
):
    """
    Plot a split violin of per-frame RMSD differences between two models.

    Left violin: model B operating on model A's test data minus model A on its own.
    Right violin: model A operating on model B's test data minus model B on its own.
    Values above zero indicate model A is more accurate.

    Parameters
    ----------
    model_A_name2load : str
        Model A name, used to glob numpy_backups and difference directories.
    model_B_name2load : str
        Model B name.
    pref : str
        Molecule prefix key into pref2heavycount.
    figure_dir : str or None
        Directory for figure and log output.
    ceil : float
        Ceiling in angstroms for outlier removal.
    num_stds : float
        Standard deviations above median for additional outlier removal.
    show : bool
        Whether to display the figure interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
    figure_log : str
    """
    title = f"Compare_{model_A_name2load}_{model_B_name2load}"
    figure_log = f"Began {datetime.now()} \n"
    figure_log += f"Log for making figure {title=} \n"

    chart_data_11 = load_files(glob.glob(f"numpy_backups/{model_A_name2load}/*/*/*.npy"), verbose=False)
    means_11, best_rpts_rmsd_11, _ = lowest_rpts(chart_data_11)
    rmsds_11 = {key: val[best_rpts_rmsd_11[key]]["VAE_RMSD"] for key, val in chart_data_11.items()}

    chart_data_22 = load_files(glob.glob(f"numpy_backups/{model_B_name2load}/*/*/*.npy"), verbose=False)
    means_22, best_rpts_rmsd_22, _ = lowest_rpts(chart_data_22)
    rmsds_22 = {key: val[best_rpts_rmsd_22[key]]["VAE_RMSD"] for key, val in chart_data_22.items()}

    assert [int(k) for k in rmsds_11] == [int(k) for k in rmsds_22]

    data_files_12 = sorted([
        f for f in glob.glob(f"difference/{model_A_name2load}*reconstructing*{model_B_name2load}*.npy")
        if "_0" in f or "_1" in f
    ])
    rmsds_12 = {
        int(os.path.basename(fn).split("reconstructing")[0][-5:-1]): np.load(fn)
        for fn in data_files_12
    }
    data_files_21 = sorted([
        f for f in glob.glob(f"difference/{model_B_name2load}*reconstructing*{model_A_name2load}*.npy")
        if "_0" in f or "_1" in f
    ])
    rmsds_21 = {
        int(os.path.basename(fn).split("reconstructing")[0][-5:-1]): np.load(fn)
        for fn in data_files_21
    }

    assert [int(k) for k in rmsds_11] == [int(k) for k in rmsds_12]
    assert [int(k) for k in rmsds_11] == [int(k) for k in rmsds_21]

    def safe_diff(a_dict, b_dict):
        result = {}
        for (key, a), b in zip(a_dict.items(), b_dict.values()):
            size = a.shape[0] - b.shape[0]
            if size == 0:
                result[key] = a - b
            elif size > 0:
                result[key] = a[: b.shape[0]] - b
            else:
                result[key] = a - b[: a.shape[0]]
        return result

    err_diff_A = safe_diff(rmsds_21, rmsds_11)
    err_diff_B = safe_diff(rmsds_22, rmsds_12)
    latents = [int(k) for k in err_diff_A]

    rpt_rmsds = [10 * err_diff_A[k] for k in err_diff_A]
    rpt_pca_rmsds = [10 * err_diff_B[k] for k in err_diff_B]
    og_shapes = [d.shape[0] for d in rpt_rmsds]
    og_pca_shapes = [d.shape[0] for d in rpt_pca_rmsds]

    rpt_rmsds = [d[~np.isnan(d)] for d in rpt_rmsds]
    rpt_pca_rmsds = [d[~np.isnan(d)] for d in rpt_pca_rmsds]
    non_nan_shapes = [d.shape[0] for d in rpt_rmsds]
    non_nan_pca_shapes = [d.shape[0] for d in rpt_pca_rmsds]

    indices_all_nan = []
    for i, (latent, old1, old2, new1, new2) in enumerate(
        zip(latents, og_shapes, og_pca_shapes, non_nan_shapes, non_nan_pca_shapes)
    ):
        figure_log += f"Analyze NAN content for {latent} Latents \n"
        figure_log += (
            f"For model with {latent} Latents, \n"
            f" \t {old1-new1} Nan values were found in VAE \n"
            f" \t {old2-new2} Nan values were found in PCA \n"
        )
        if new1 == 0:
            figure_log += (
                f"For model with {latent} Latents, \n"
                f" \t All values are NAN - a dashed line on the plot \n"
            )
            indices_all_nan.append(i)

    rpt_rmsds = [d[(d < ceil) & (d > -1 * ceil)] for d in rpt_rmsds]
    rpt_pca_rmsds = [d[(d < ceil) & (d > -1 * ceil)] for d in rpt_pca_rmsds]
    in_thresh_shapes = [d.shape[0] for d in rpt_rmsds]
    in_thresh_pca_shapes = [d.shape[0] for d in rpt_pca_rmsds]

    for i, (latent, old1, old2, new1, new2) in enumerate(
        zip(latents, non_nan_shapes, non_nan_pca_shapes, in_thresh_shapes, in_thresh_pca_shapes)
    ):
        figure_log += f"Analyze Outliers content for {latent} Latents \n"
        figure_log += (
            f"For model with {latent} Latents, \n"
            f" \t {old1-new1} values were found in VAE above the ceiling of {ceil} Angstroms"
            f" or below the floor of {-1*ceil} \n"
            f" \t {old2-new2} values were found in PCA above the ceiling of {ceil} Angstroms \n"
        )

    rpt_rmsds = [d for i, d in enumerate(rpt_rmsds) if i not in indices_all_nan]
    rpt_pca_rmsds = [d for i, d in enumerate(rpt_pca_rmsds) if i not in indices_all_nan]
    latents2plot = [d for i, d in enumerate(latents) if i not in indices_all_nan]

    rpt_rmsd_medians = [np.median(a) for a in rpt_rmsds]
    rpt_rmsd_stds = [np.std(a) for a in rpt_rmsds]
    rpt_pca_rmsd_medians = [np.median(a) for a in rpt_pca_rmsds]
    rpt_pca_rmsd_stds = [np.std(a) for a in rpt_pca_rmsds]
    rpt_maxima = [m + 3 * s for m, s in zip(rpt_rmsd_medians, rpt_rmsd_stds)]
    rpt_pca_maxima = [m + 3 * s for m, s in zip(rpt_pca_rmsd_medians, rpt_pca_rmsd_stds)]

    figure_log += "Report the median and standard deviation used for each Latent \n  \t VAE \n"
    for latent, median, std in zip(latents2plot, rpt_rmsd_medians, rpt_rmsd_stds):
        figure_log += f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n"
    figure_log += "  \t PCA \n"
    for latent, median, std in zip(latents2plot, rpt_pca_rmsd_medians, rpt_pca_rmsd_stds):
        figure_log += f"\t\t L-{latent} Med-{median:0.3f} Std-{std:0.3f} \n"
    figure_log += "\n"

    rpt_data2plot = [d[d < m] for d, m in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2plot = [d[d < m] for d, m in zip(rpt_pca_rmsds, rpt_pca_maxima)]
    rpt_data2exclude = [np.where(d > m)[0].shape[0] for d, m in zip(rpt_rmsds, rpt_maxima)]
    rpt_pca_data2exclude = [np.where(d > m)[0].shape[0] for d, m in zip(rpt_pca_rmsds, rpt_pca_maxima)]

    figure_log += f"Number of points that are {num_stds:0.3f} stds above the median \n \t VAE \n"
    for latent, elem in zip(latents2plot, rpt_data2exclude):
        figure_log += f"\t\t L-{latent} Num-{elem} \n"
    figure_log += " \t PCA \n"
    for latent, elem in zip(latents2plot, rpt_pca_data2exclude):
        figure_log += f"\t\t L-{latent} Num-{elem} \n"
    figure_log += "\n"

    figure_log += "Final number of points comprising violin \n \t VAE \n"
    for latent, elem in zip(latents2plot, rpt_data2plot):
        figure_log += f"\t\t L-{latent} Num-{elem.shape[0]} \n"
    figure_log += " \t PCA \n"
    for latent, elem in zip(latents2plot, rpt_pca_data2plot):
        figure_log += f"\t\t L-{latent} Num-{elem.shape[0]} \n"
    figure_log += "\n"

    plt.clf()
    fig = plt.figure(figsize=(3.25, 3.25))
    plt.hlines(
        y=0.0,
        xmin=0.9,
        xmax=2.1 + index_highest_two_mod(pref2heavycount[pref])[0],
        linestyle="--", color="k", zorder=-1,
    )
    domain = [
        n - 0.05 if n <= 4
        else 1.95 + np.log2(n) if n in two_to
        else 1.95 + index_highest_two_mod(pref2heavycount[pref])[0]
        for n in latents2plot
    ]
    print([v.shape for v in rpt_data2plot])
    v1 = plt.violinplot(rpt_data2plot, domain, showextrema=False)
    domain = [
        n + 0.05 if n <= 4
        else 2.05 + np.log2(n) if n in two_to
        else 2.05 + index_highest_two_mod(pref2heavycount[pref])[0]
        for n in latents2plot
    ]
    print([v.shape for v in rpt_pca_data2plot])
    v2 = plt.violinplot(rpt_pca_data2plot, domain, showextrema=False)

    for b in v1["bodies"]:
        b.set_edgecolor("black")
        b.set_alpha(1)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color((0, 0.5, 0.5))

    for b in v2["bodies"]:
        b.set_edgecolor("black")
        b.set_alpha(1)
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color((0.5, 0, 0.5))

    global_min = np.min([np.min(d) for d in rpt_data2plot])
    global_max = np.max([np.max(d) for d in rpt_data2plot])
    latents2vline = [n for n in latents if n not in latents2plot]
    latents2vline = [
        n - 0.05 if n <= 4
        else 1.95 + np.log2(n) if n in two_to
        else 1.95 + index_highest_two_mod(pref2heavycount[pref])[0]
        for n in latents2vline
    ]
    plt.vlines(latents2vline, ymin=global_min, ymax=global_max, colors="red", linestyles="dashed")
    plt.xlabel("N_Latents")
    plt.ylabel("Error Difference (Angstrom)")
    plt.title(title.replace("_", " "))
    plt.xticks(
        ticks=np.arange(1, len(latents) + 1),
        labels=["" + str(latent) for latent in latents],
        rotation="vertical",
    )

    if figure_dir:
        save_fig_path = os.path.join(figure_dir, f"reconstruction_diff_err_{title=}.png")
        print(f"Saving at {save_fig_path}")
        plt.savefig(save_fig_path, bbox_inches="tight")
        log_path = os.path.join(figure_dir, f"figure_generation_log_diff_err_{title=}.txt")
        with open(log_path, "w") as f:
            figure_log += f"Successfully generated and saved figure and log by {datetime.now()} \n"
            f.write(figure_log)

    if show:
        plt.show()
    else:
        plt.close()

    return fig, figure_log


# ---------------------------------------------------------------------------
# Figure log parsing helpers
# ---------------------------------------------------------------------------

def _NAN_part(lines):
    latents, vae_nans, pca_nans = [], [], []
    for line in lines:
        if line.startswith("Analyze NAN content for"):
            num_latents = line.split(" ")[4]
            idx = lines.index(line)
            vae_line, pca_line = lines[idx + 2], lines[idx + 3]
            latents.append(num_latents)
            vae_nans.append([e for e in vae_line.split(" ") if e][1])
            pca_nans.append([e for e in pca_line.split(" ") if e][1])
    return latents, vae_nans, pca_nans


def _farlier_part(lines):
    latents, vae_nans, pca_nans = [], [], []
    for line in lines:
        if line.startswith("Analyze Outliers content for"):
            num_latents = line.split(" ")[4]
            idx = lines.index(line)
            vae_line, pca_line = lines[idx + 2], lines[idx + 3]
            latents.append(num_latents)
            vae_nans.append([e for e in vae_line.split(" ") if e][1])
            pca_nans.append([e for e in pca_line.split(" ") if e][1])
    return latents, vae_nans, pca_nans


def _mean_std_part(lines):
    start = lines.index([l for l in lines if l.startswith("Report the median and standard deviation")][0])
    end = lines.index([l for l in lines if l.startswith("Number of points that are ")][0]) - 1
    lines = lines[start + 1 : end]
    halfway = len(lines) // 2
    vae_lines, pca_lines = lines[1:halfway], lines[halfway + 1 :]

    def parse_field(prefix, line):
        return [e for e in line.split(" ") if e.startswith(prefix)][0][len(prefix):]

    vae_lats = [parse_field("L-", l) for l in vae_lines]
    vae_meds = [parse_field("Med-", l) for l in vae_lines]
    vae_stds = [parse_field("Std-", l) for l in vae_lines]
    pca_lats = [parse_field("L-", l) for l in pca_lines]
    pca_meds = [parse_field("Med-", l) for l in pca_lines]
    pca_stds = [parse_field("Std-", l) for l in pca_lines]
    assert vae_lats == pca_lats
    return vae_lats, vae_meds, vae_stds, pca_meds, pca_stds


def _num_above_mean_std_part(lines):
    start = lines.index([l for l in lines if l.startswith("Number of points that are ")][0])
    end = lines.index([l for l in lines if l.startswith("Final number of points comprising violin")][0]) - 1
    lines = lines[start + 1 : end]
    halfway = len(lines) // 2
    vae_lines, pca_lines = lines[1:halfway], lines[halfway + 1 :]

    vae_lats = [[e for e in l.split(" ") if e.startswith("L-")][0][2:] for l in vae_lines]
    vae_nums = [[e for e in l.split(" ") if e.startswith("Num-")][0][4:] for l in vae_lines]
    pca_lats = [[e for e in l.split(" ") if e.startswith("L-")][0][2:] for l in pca_lines]
    pca_nums = [[e for e in l.split(" ") if e.startswith("Num-")][0][4:] for l in pca_lines]
    assert vae_lats == pca_lats
    return vae_lats, vae_nums, pca_nums


def _num_in_violin(lines):
    start = lines.index([l for l in lines if l.startswith("Final number of points comprising violin")][0])
    end = lines.index([l for l in lines if l.startswith("Successfully generated and saved figure")][0]) - 1
    lines = lines[start + 1 : end]
    halfway = len(lines) // 2
    vae_lines, pca_lines = lines[1:halfway], lines[halfway + 1 :]

    vae_lats = [[e for e in l.split(" ") if e.startswith("L-")][0][2:] for l in vae_lines]
    vae_nums = [[e for e in l.split(" ") if e.startswith("Num-")][0][4:] for l in vae_lines]
    pca_lats = [[e for e in l.split(" ") if e.startswith("L-")][0][2:] for l in pca_lines]
    pca_nums = [[e for e in l.split(" ") if e.startswith("Num-")][0][4:] for l in pca_lines]
    assert vae_lats == pca_lats
    return vae_lats, vae_nums, pca_nums


def figure_log_to_csv(figure_log_fn):
    """
    Convert a figure_generation_log.txt file to a CSV-formatted string.

    The CSV has one row per latent per method (VAE / PCA) with columns:
    NAN count, > 10 A count, Median (Ang), Std (Ang), N Omit, N Violin.

    Parameters
    ----------
    figure_log_fn : str
        Path to the figure generation log text file.

    Returns
    -------
    str
        CSV-formatted string (not yet written to disk).
    """
    with open(figure_log_fn, "r") as f:
        lines = [line for line in f.readlines()]
    name = [line for line in lines if "title" in line][0][29:-3]
    file_contents = f",,{name},,,,,,\n"
    cols = ["NAN", "> 10 A", "Median (Ang)", "Std (Ang)", "N Omit", "N Violin"]
    file_contents += "Latents,," + ",".join(cols) + "\n"

    latents, vae_nan, pca_nan = _NAN_part(lines)
    _, vae_farlier, pca_farlier = _farlier_part(lines)
    latents_plotted, vae_meds, vae_stds, pca_meds, pca_stds = _mean_std_part(lines)
    _, vae_outlier, pca_outlier = _num_above_mean_std_part(lines)
    _, vae_violin, pca_violin = _num_in_violin(lines)

    vae_lines, pca_lines_out = [], []
    for i, latent in enumerate(latents):
        num_nan, num_far = vae_nan[i], vae_farlier[i]
        if latent in latents_plotted:
            j = latents_plotted.index(latent)
            median, std, n_omit, n_viol = vae_meds[j], vae_stds[j], vae_outlier[j], vae_violin[j]
        else:
            median, std, n_omit, n_viol = 0, 0, 0, 0
        vae_lines.append(f"{latent},VAE (red),{num_nan},{num_far},{median},{std},{n_omit},{n_viol}\n")

        num_nan, num_far = pca_nan[i], pca_farlier[i]
        if latent in latents_plotted:
            j = latents_plotted.index(latent)
            median, std, n_omit, n_viol = pca_meds[j], pca_stds[j], pca_outlier[j], pca_violin[j]
        else:
            median, std, n_omit, n_viol = 0, 0, 0, 0
        pca_lines_out.append(f"{latent},PCA (blue),{num_nan},{num_far},{median},{std},{n_omit},{n_viol}\n")

    for vl, pl in zip(vae_lines, pca_lines_out):
        file_contents += vl
        file_contents += pl

    return file_contents


def reorganize_figures_and_logs(figure_dir):
    """
    Move figures and logs into per-experiment-version subdirectories.

    Scans figure_dir for files whose name contains an X-prefixed version tag,
    creates a subdirectory for that version, and copies matching PNGs and logs.

    Parameters
    ----------
    figure_dir : str
        Root directory containing unsorted figures.
    """
    import shutil

    unique_models = []
    for content in os.listdir(figure_dir):
        if content.startswith("X") or "_" not in content:
            continue
        name_parts = content.split("_")
        if name_parts[-1].startswith("X") and name_parts[-1] not in unique_models:
            unique_models.append(name_parts[-1])

    for model_name in sorted(unique_models):
        dest = os.path.join(figure_dir, model_name)
        os.makedirs(dest, exist_ok=True)
        for content in os.listdir(figure_dir):
            if content.startswith("X") or "_" not in content:
                continue
            if content.split("_")[-1] == model_name:
                pngs = sorted(glob.glob(os.path.join(figure_dir, content, "*.png")))
                logs = sorted(glob.glob(os.path.join(figure_dir, content, "figure_generation_log.txt")))
                for png, log in zip(pngs, logs):
                    shutil.copy(png, os.path.join(dest, os.path.basename(png)))
                    shutil.copy(
                        log,
                        os.path.join(dest, os.path.basename(png)[:-4] + "_" + os.path.basename(log)),
                    )
