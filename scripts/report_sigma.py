"""
Report the reparameterization sigma (posterior standard deviation) of trained
Deep-MMS VAE checkpoints -- i.e. test whether the "variational" models
collapsed to deterministic autoencoders.

Why this matters
----------------
The Deep-MMS VAEs are trained with a reconstruction-only objective.  In
``deepmms/training/optimizer.py`` the loss is::

    loss = log(sqrt(mean(atom_rmsd(batch, recon) ** 2)))

There is NO KL term.  Yet the forward pass still samples the latent with the
reparameterization trick (``deepmms/models/vae.py``)::

    z_std = exp(0.5 * z_logvar)          # <-- this is "sigma"
    z     = z_mean + eps * z_std,   eps ~ N(0, 1)

Sampling noise can only *hurt* reconstruction and nothing rewards a spread-out
posterior, so gradient descent is free to drive ``z_logvar`` toward -inf,
collapsing sigma -> 0 and making the model effectively deterministic.  This is
exactly the failure mode colleagues are worried about; this script measures how
small sigma actually became, for every trained checkpoint.

How sigma is measured (checkpoint-only, no trajectory files needed)
-------------------------------------------------------------------
The final encoder layer that produces ``z_logvar`` is a linear map::

    logvar = h @ W + b

where ``h`` is the last hidden activation, ``W`` = ``fc5_logvar/kernel`` and
``b`` = ``fc5_logvar/bias``.  In these models a BatchNorm layer sits directly
before that head, and its running statistics + affine parameters tell us the
distribution of ``h`` over the *training data* without re-running anything:

    h_i  ~  mean = BatchNorm.bias_i ,  std = |BatchNorm.scale_i|

Propagating that through the linear head gives the data-representative
distribution of the log-variance, and hence of sigma::

    logvar_j ~ Normal( mean = b_j + sum_i W_ij * BN.bias_i,
                       std  = sqrt( sum_i (W_ij * BN.scale_i)^2 ) )
    sigma_j  = exp(0.5 * logvar_j)

We report sigma at the mean of that distribution ("typical") and at +/- 2 std
("data-representative range").  This is architecture-agnostic: it works on the
original funnel/leaky-relu checkpoints and on the refactored square/relu ones
alike, because it only inspects the log-variance head and the BatchNorm that
feeds it -- located automatically in the parameter tree.

If a log-variance head has no BatchNorm feeding it (e.g. a non-BatchNorm
variant), the script falls back to reporting sigma at the bias with a
conservative kernel-norm-based upper bound, and says so.

Interpretation
--------------
A properly regularised VAE posterior has sigma of order 1 (matching the N(0,1)
prior).  sigma << 1 means the dimension carries essentially no stochasticity:
the "variational" model behaves as a plain deterministic autoencoder on that
axis.  A latent is flagged COLLAPSED when its sigma is below --threshold
(default 0.1).

Usage
-----
    # Scan every checkpoint under the repo (latest epoch of each model):
    python scripts/report_sigma.py .

    # Scan one experiment subtree and write a CSV:
    python scripts/report_sigma.py CR_X008-2 --csv sigma_report.csv

    # Point at one checkpoint_managed dir, or a single <epoch>/default dir:
    python scripts/report_sigma.py CR_X008-2/0004_latents/rpt_3/checkpoint_managed

    # Watch sigma collapse across every saved epoch of one model:
    python scripts/report_sigma.py CR_X008-2/0004_latents/rpt_3 --all-epochs --per-latent
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

# Keys whose {kernel, bias} subtree is a log-variance head.  Covers fc5_logvar
# (BatchNorm_VAE), z_logvar / _z_logvar (transformer, mamba, perceiver, kan,
# mae, se3, equivariant, neat) and z2_logvar (hierarchical).
_LOGVAR_KEY_RE = re.compile(r"log_?var|log_?sigma", re.IGNORECASE)
# Keys identifying a BatchNorm layer (its {scale, bias} params and {mean, var}
# running stats).
_BN_KEY_RE = re.compile(r"batch_?norm", re.IGNORECASE)
_BN_EPS = 1e-5  # Flax nn.BatchNorm default epsilon.


def _lazy_orbax():
    """Import orbax only when a checkpoint is actually restored."""
    import orbax.checkpoint as ocp
    return ocp


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def find_leaf_checkpoints(path, recency=-1, all_epochs=False):
    """
    Resolve a user path into a list of (label, leaf_dir) checkpoints.

    A "leaf_dir" is an Orbax item directory ending in ``/<epoch>/default``.
    ``path`` may be a leaf dir, a ``checkpoint_managed`` dir, or any ancestor
    directory which is searched recursively for every ``checkpoint_managed``
    directory beneath it.

    Parameters
    ----------
    path : str
        Path to resolve.
    recency : int
        Index into each model's ascending-epoch checkpoint list; -1 = latest.
        Ignored when all_epochs is True.
    all_epochs : bool
        Emit every saved epoch of each model instead of a single checkpoint.

    Returns
    -------
    list of (str, str)
        (label, leaf_dir) pairs.
    """
    path = os.path.abspath(path)

    if os.path.basename(path) == "default":
        return [(_model_label(path), path)]

    if os.path.basename(path) == "checkpoint_managed":
        managed_dirs = [path]
    else:
        managed_dirs = sorted(
            glob.glob(os.path.join(path, "**", "checkpoint_managed"), recursive=True)
        )
        direct = os.path.join(path, "checkpoint_managed")
        if os.path.isdir(direct) and direct not in managed_dirs:
            managed_dirs.append(direct)

    results = []
    for managed in sorted(set(managed_dirs)):
        epochs = _sorted_epoch_dirs(managed)
        if not epochs:
            continue
        chosen = epochs if all_epochs else [epochs[recency]]
        for leaf in chosen:
            results.append((_model_label(leaf), leaf))
    return results


def _sorted_epoch_dirs(managed_dir):
    """Return leaf dirs under a checkpoint_managed dir, sorted by epoch number."""
    epoch_dirs = []
    for entry in os.listdir(managed_dir):
        full = os.path.join(managed_dir, entry)
        if entry.isdigit() and os.path.isdir(full):
            leaf = os.path.join(full, "default")
            if os.path.isdir(leaf):
                epoch_dirs.append((int(entry), leaf))
    epoch_dirs.sort(key=lambda t: t[0])
    return [leaf for _, leaf in epoch_dirs]


def _model_label(leaf_dir):
    """
    Build a compact label from a leaf checkpoint path.

    .../CR_X008-2/0004_latents/rpt_3/checkpoint_managed/15000/default
        -> "CR_X008-2/0004_latents/rpt_3 @epoch 15000"
    """
    parts = leaf_dir.rstrip(os.sep).split(os.sep)
    try:
        epoch = parts[-2]
        managed_idx = parts.index("checkpoint_managed")
        model_parts = parts[max(0, managed_idx - 3):managed_idx]
        return f"{'/'.join(model_parts)} @epoch {epoch}"
    except (ValueError, IndexError):
        return leaf_dir


# --------------------------------------------------------------------------- #
# Locating the log-variance head and the BatchNorm that feeds it
# --------------------------------------------------------------------------- #
def find_logvar_heads(params):
    """
    Recursively locate every log-variance head in a Flax param pytree.

    A head is a dict carrying both 'kernel' and 'bias' whose own key matches the
    log-variance pattern.

    Parameters
    ----------
    params : dict
        The 'params' subtree of a restored checkpoint.

    Returns
    -------
    list of (list_of_str, np.ndarray, np.ndarray)
        (path_keys, kernel, bias) for each head; path_keys locates it in the tree.
    """
    heads = []

    def walk(node, trail):
        if not isinstance(node, dict):
            return
        key = trail[-1] if trail else ""
        if _LOGVAR_KEY_RE.search(str(key)) and "kernel" in node and "bias" in node:
            heads.append((list(trail),
                          np.asarray(node["kernel"], dtype=np.float64),
                          np.asarray(node["bias"], dtype=np.float64)))
        for child_key, child in node.items():
            if isinstance(child, dict):
                walk(child, trail + [str(child_key)])

    walk(params, [])
    return heads


def _get(tree, path_keys):
    """Follow a list of keys into a nested dict, or return None if absent."""
    node = tree
    for key in path_keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def find_feeding_batchnorm(params, batch_stats, head_path, in_dim):
    """
    Find the BatchNorm layer in the same scope that feeds a log-variance head.

    The encoder applies ``... -> BatchNorm -> (dropout) -> fc5_logvar``, so the
    head's input is the output of the last BatchNorm in its parent scope.  We
    match by feature dimension (== the head kernel's input dim) and, among ties,
    take the highest-indexed BatchNorm.

    Parameters
    ----------
    params : dict
        Full 'params' subtree.
    batch_stats : dict or None
        Full 'batch_stats' subtree (running mean/var); None if the model has none.
    head_path : list of str
        Key path to the log-variance head.
    in_dim : int
        Number of input features to the head (head kernel's first-axis length).

    Returns
    -------
    dict or None
        {'name', 'scale', 'bias', 'mean', 'var'} of the feeding BatchNorm, or
        None if no matching BatchNorm with running stats is found.
    """
    if batch_stats is None:
        return None
    parent_path = head_path[:-1]
    parent_params = _get(params, parent_path)
    parent_stats = _get(batch_stats, parent_path)
    if not isinstance(parent_params, dict) or not isinstance(parent_stats, dict):
        return None

    candidates = []
    for name, sub in parent_params.items():
        if not (_BN_KEY_RE.search(str(name)) and isinstance(sub, dict)):
            continue
        if "scale" not in sub or "bias" not in sub:
            continue
        stat = parent_stats.get(name)
        if not isinstance(stat, dict) or "mean" not in stat or "var" not in stat:
            continue
        scale = np.asarray(sub["scale"], dtype=np.float64)
        if scale.shape[0] != in_dim:
            continue
        # Sort key: trailing integer in the name (BatchNorm_5 -> 5), else -1.
        m = re.search(r"(\d+)\s*$", str(name))
        order = int(m.group(1)) if m else -1
        candidates.append((order, name, sub, stat))

    if not candidates:
        return None
    _, name, sub, stat = max(candidates, key=lambda t: t[0])
    return {
        "name": name,
        "scale": np.asarray(sub["scale"], dtype=np.float64),
        "bias": np.asarray(sub["bias"], dtype=np.float64),
        "mean": np.asarray(stat["mean"], dtype=np.float64),
        "var": np.asarray(stat["var"], dtype=np.float64),
    }


# --------------------------------------------------------------------------- #
# Sigma computation
# --------------------------------------------------------------------------- #
def summarise_head(kernel, bias, bn):
    """
    Compute the data-representative sigma distribution for one log-variance head.

    Parameters
    ----------
    kernel : np.ndarray, shape (in_dim, n_latents)
        The head's weight matrix W.
    bias : np.ndarray, shape (n_latents,)
        The head's bias b.
    bn : dict or None
        Output of find_feeding_batchnorm; None if no BatchNorm feeds the head.

    Returns
    -------
    dict
        Per-latent arrays and the method used.  Keys:
        method, bias, kcol_norm, logvar_mean, logvar_std,
        sigma_typ, sigma_lo, sigma_hi.
    """
    bias = np.asarray(bias, dtype=np.float64).ravel()
    kernel = np.asarray(kernel, dtype=np.float64)
    kcol_norm = np.linalg.norm(kernel, axis=0) if kernel.ndim == 2 else np.zeros_like(bias)

    if bn is not None and kernel.ndim == 2:
        # BatchNorm output over the data: mean ~ bn.bias, std ~ |bn.scale|.
        h_mean = bn["bias"]
        h_std = np.abs(bn["scale"])
        logvar_mean = bias + kernel.T @ h_mean
        logvar_std = np.sqrt((kernel.T ** 2) @ (h_std ** 2))
        method = f"bn-analytic (feeds from {bn['name']})"
    else:
        # No BatchNorm to pin the input scale: anchor at the bias and use the
        # kernel column norm as a conservative +/- proxy for the logvar spread.
        logvar_mean = bias.copy()
        logvar_std = kcol_norm.copy()
        method = "bias-only (no BatchNorm feeding head; spread is a rough bound)"

    sigma_typ = np.exp(0.5 * logvar_mean)
    sigma_lo = np.exp(0.5 * (logvar_mean - 2.0 * logvar_std))
    sigma_hi = np.exp(0.5 * (logvar_mean + 2.0 * logvar_std))
    return {
        "method": method,
        "bias": bias,
        "kcol_norm": kcol_norm,
        "logvar_mean": logvar_mean,
        "logvar_std": logvar_std,
        "sigma_typ": sigma_typ,
        "sigma_lo": sigma_lo,
        "sigma_hi": sigma_hi,
    }


def scan_checkpoint(leaf_dir):
    """
    Restore one checkpoint and summarise its log-variance head(s).

    Parameters
    ----------
    leaf_dir : str
        Path to an Orbax "<epoch>/default" item directory.

    Returns
    -------
    list of dict
        One record per log-variance head (see summarise_head), each also
        carrying 'head' (dotted path) and 'n_latents'.
    """
    ocp = _lazy_orbax()
    restored = ocp.PyTreeCheckpointer().restore(leaf_dir)
    if not isinstance(restored, dict) or "params" not in restored:
        raise ValueError("checkpoint has no 'params' subtree")

    params = restored["params"]
    batch_stats = restored.get("batch_stats")
    heads = find_logvar_heads(params)
    if not heads:
        raise ValueError("no log-variance head found (not a reparameterized VAE?)")

    records = []
    for head_path, kernel, bias in heads:
        in_dim = kernel.shape[0] if kernel.ndim == 2 else 0
        bn = find_feeding_batchnorm(params, batch_stats, head_path, in_dim)
        rec = summarise_head(kernel, bias, bn)
        rec["head"] = ".".join(head_path)
        rec["n_latents"] = int(bias.size)
        records.append(rec)
    return records


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _verdict(sigmas, threshold):
    """
    Classify an array of sigmas.

    Returns
    -------
    (n_collapsed, n_total, n_nan, verdict_str)
        n_collapsed counts finite sigmas below threshold; n_nan counts
        non-finite entries (a NaN/inf log-variance means the training run
        diverged and the checkpoint is not trustworthy).
    """
    sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
    n_total = sigmas.size
    finite = np.isfinite(sigmas)
    n_nan = int(np.sum(~finite))
    n_collapsed = int(np.sum(sigmas[finite] < threshold))

    if n_nan == n_total:
        verdict = "DIVERGED (all latents NaN/inf; training blew up)"
    elif n_nan > 0:
        verdict = f"DIVERGED ({n_nan}/{n_total} latents NaN/inf)"
    elif n_collapsed == n_total:
        verdict = "COLLAPSED (all latents deterministic)"
    elif n_collapsed > 0:
        verdict = f"PARTIAL ({n_collapsed}/{n_total} latents collapsed)"
    else:
        verdict = "OK (posterior retains spread)"
    return n_collapsed, n_total, n_nan, verdict


def print_record(label, records, threshold, show_per_latent):
    """Pretty-print one model's result to stdout."""
    print("=" * 80)
    print(label)
    for rec in records:
        # Judge collapse by the upper edge of the data-representative range so
        # we never over-claim collapse.
        n_coll, n_tot, n_nan, verdict = _verdict(rec["sigma_hi"], threshold)
        print(f"  head '{rec['head']}'  ({rec['n_latents']} latents)   [{rec['method']}]")
        print(f"    logvar mean      : min {rec['logvar_mean'].min():+8.3f} "
              f"med {np.median(rec['logvar_mean']):+8.3f}  max {rec['logvar_mean'].max():+8.3f}")
        print(f"    sigma typical    : min {rec['sigma_typ'].min():.3e} "
              f"med {np.median(rec['sigma_typ']):.3e}  max {rec['sigma_typ'].max():.3e}")
        print(f"    sigma data range : [{rec['sigma_lo'].min():.3e} .. "
              f"{rec['sigma_hi'].max():.3e}]  (-2sd .. +2sd over data)")
        print(f"    VERDICT          : {verdict}   (threshold sigma < {threshold})")
        if show_per_latent:
            print("    per-latent [idx: sigma_typ  (sigma_-2sd .. sigma_+2sd)]:")
            for j in range(rec["n_latents"]):
                print(f"      {j:4d}: {rec['sigma_typ'][j]:.3e}  "
                      f"({rec['sigma_lo'][j]:.3e} .. {rec['sigma_hi'][j]:.3e})")


def write_csv(csv_path, rows):
    """Write accumulated per-head summary rows to a CSV file."""
    fieldnames = [
        "model", "head", "method", "n_latents",
        "logvar_mean_med", "sigma_typ_med", "sigma_typ_min", "sigma_typ_max",
        "sigma_range_lo", "sigma_range_hi", "n_collapsed", "verdict",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    """Construct the argparse parser."""
    p = argparse.ArgumentParser(
        description="Report reparameterization sigma of trained Deep-MMS VAEs "
                    "(detect posterior collapse toward a deterministic AE).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="*", default=["."],
                   help="Checkpoint dirs, checkpoint_managed dirs, or trees to "
                        "search recursively (default: current directory).")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="sigma below this counts as collapsed (default: 0.1).")
    p.add_argument("--recency", type=int, default=-1,
                   help="Which checkpoint per model: -1 = latest (default).")
    p.add_argument("--all-epochs", action="store_true",
                   help="Report every saved epoch (watch collapse over training).")
    p.add_argument("--per-latent", action="store_true",
                   help="Print every latent dimension, not just summary stats.")
    p.add_argument("--csv", metavar="FILE",
                   help="Write a machine-readable per-head summary CSV.")
    return p


def main(argv=None):
    """Entry point."""
    args = build_parser().parse_args(argv)

    checkpoints = []
    for path in args.paths:
        checkpoints.extend(find_leaf_checkpoints(path, args.recency, args.all_epochs))
    seen = set()
    checkpoints = [(l, d) for (l, d) in checkpoints if not (d in seen or seen.add(d))]

    if not checkpoints:
        print("No checkpoints found. Looked for "
              "'<...>/checkpoint_managed/<epoch>/default' directories under the "
              "given path(s).", file=sys.stderr)
        return 1

    print(f"Found {len(checkpoints)} checkpoint(s).\n")
    csv_rows = []
    n_models = n_collapsed_models = n_diverged_models = n_errors = 0

    for label, leaf in checkpoints:
        try:
            records = scan_checkpoint(leaf)
        except Exception as exc:  # noqa: BLE001 - keep scanning remaining models
            n_errors += 1
            print("=" * 80)
            print(f"{label}\n    SKIPPED: {exc}")
            continue

        n_models += 1
        print_record(label, records, args.threshold, args.per_latent)
        model_collapsed = False
        model_diverged = False
        for rec in records:
            n_coll, n_tot, n_nan, verdict = _verdict(rec["sigma_hi"], args.threshold)
            if n_nan > 0:
                model_diverged = True
            elif n_coll == n_tot:
                model_collapsed = True
            csv_rows.append({
                "model": label, "head": rec["head"], "method": rec["method"],
                "n_latents": rec["n_latents"],
                "logvar_mean_med": f"{np.median(rec['logvar_mean']):.4f}",
                "sigma_typ_med": f"{np.median(rec['sigma_typ']):.6e}",
                "sigma_typ_min": f"{rec['sigma_typ'].min():.6e}",
                "sigma_typ_max": f"{rec['sigma_typ'].max():.6e}",
                "sigma_range_lo": f"{rec['sigma_lo'].min():.6e}",
                "sigma_range_hi": f"{rec['sigma_hi'].max():.6e}",
                "n_collapsed": n_coll, "verdict": verdict,
            })
        if model_diverged:
            n_diverged_models += 1
        elif model_collapsed:
            n_collapsed_models += 1

    print("=" * 80)
    print(f"SUMMARY: {n_models} model(s) scanned | {n_collapsed_models} fully collapsed "
          f"(all latents sigma < {args.threshold}) | {n_diverged_models} diverged (NaN/inf) "
          f"| {n_errors} skipped.")
    if args.csv:
        write_csv(args.csv, csv_rows)
        print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
