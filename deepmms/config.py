"""
Config assembly helpers shared by the ``deep-mms`` CLI and the sweep generator.

Turns a small set of user-facing knobs (architecture, latent dimension, network
depth, mass-weighting scheme, …) into the flat json_params dictionary consumed
by ``Experiment`` / the specialized trainers.  Keeping this logic in one place
means the command line, the config sweep, and any notebook build identical
configs.
"""

import os

# Default config values before any user override is applied.  Mirrors the
# constants in scripts/generate_configs.py.
_DEFAULTS = dict(
    architecture="batchnorm_vae",
    latent_dim=8,
    max_epoch=10001,
    learning_rate=1e-3,
    batch_size=100,
    test_slice=1,
    data_slice_start=0,
    data_slice_end="None",
    resume_latest=False,
    checkpoint_interval=200,
    weight_model="Uniform_Heavy",
    save_dir="runs",
)

# Friendly CLI alias -> canonical json_params key.
_ALIASES = {
    "latents": "latent_dim",
    "arch": "architecture",
    "batchnorm": "is_batchnorm",
    "epochs": "max_epoch",
    "lr": "learning_rate",
    "dcd": "fname_dcd",
    "top": "fname_topology",
    "topology": "fname_topology",
}


def parse_value(text):
    """Convert a CLI string to bool / int / float where possible, else str."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def atom_selection_for(weight_model):
    """Map a mass-weighting scheme to its MDTraj atom-selection string."""
    if weight_model in ("Uniform", "Mass"):
        return "all"
    if weight_model in ("Uniform_Heavy", "Mass_Heavy", "Mass_United", "H-Valence"):
        return "not element H"
    raise ValueError(f"Unknown weight_model: {weight_model!r}")


def depth_specific_keys(architecture, depth):
    """
    Extra JSON keys that make the depth axis effective for the architectures
    whose depth is NOT taken from ``len(dropout_rates)`` (se3 / equivariant /
    flow).  Returns an empty dict for every other architecture.
    """
    if architecture == "se3":
        return {"n_mp_layers": depth}
    if architecture == "equivariant":
        return {"n_interactions": depth}
    if architecture == "flow":
        return {"n_coupling_layers": depth}
    return {}


def assemble_config(fname_dcd, fname_topology, overrides=None):
    """
    Build a full json_params dict from two data paths plus key=value overrides.

    Applies alias resolution, a ``depth`` -> dropout_rates expansion (with the
    per-architecture depth keys), an is_batchnorm default tied to the
    architecture, an atom_selection derived from weight_model, and a descriptive
    model_name.  Explicit overrides always win over the derived defaults.

    Parameters
    ----------
    fname_dcd : str
        Path to the trajectory (.dcd).
    fname_topology : str
        Path to the topology (.pdb / .prmtop).
    overrides : dict, optional
        Already-parsed {key: value} pairs; aliases (e.g. ``latents``,
        ``batchnorm``, ``depth``) are accepted.

    Returns
    -------
    dict
        A config accepted by ``build_harness(..., from_json_params=True)``.
    """
    overrides = dict(overrides or {})
    cfg = dict(_DEFAULTS)

    depth = overrides.pop("depth", None)
    for key, value in overrides.items():
        cfg[_ALIASES.get(key, key)] = value

    # Absolute paths: Orbax requires an absolute checkpoint dir, and absolute
    # data paths make runs independent of the working directory.
    cfg["fname_dcd"] = os.path.abspath(fname_dcd)
    cfg["fname_topology"] = os.path.abspath(fname_topology)
    architecture = cfg["architecture"]

    # Depth -> number of hidden layers, unless dropout_rates is given directly.
    if depth is not None and "dropout_rates" not in cfg:
        depth = int(depth)
        cfg["dropout_rates"] = [0.0] * depth
        for key, value in depth_specific_keys(architecture, depth).items():
            cfg.setdefault(key, value)
    cfg.setdefault("dropout_rates", [0.0, 0.0, 0.0, 0.0])

    cfg.setdefault("is_batchnorm", architecture == "batchnorm_vae")
    cfg.setdefault("atom_selection", atom_selection_for(cfg["weight_model"]))

    if "model_name" not in cfg:
        stem = os.path.splitext(os.path.basename(fname_dcd))[0]
        cfg["model_name"] = f"{stem}_{architecture}_D{len(cfg['dropout_rates']):02d}"

    cfg["save_dir"] = os.path.abspath(cfg["save_dir"])
    return cfg
