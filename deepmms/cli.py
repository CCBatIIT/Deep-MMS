"""
``deep-mms`` command-line entry point: train a model in one shot from the shell.

Installed as the ``deep-mms`` console script via pyproject.toml.  Assembles a
config from CLI arguments (no JSON file required) and trains the selected
architecture.

Usage
-----
    deep-mms <fname_dcd> <fname_topology> [key=value ...] [--n-epochs N] [--dry-run]

Positional arguments
    fname_dcd         trajectory (.dcd)
    fname_topology    topology (.pdb / .prmtop)

key=value overrides (any config key; common aliases in parentheses)
    architecture=batchnorm_vae   model to train (see --list-architectures)
    latents=8                    (latent_dim) number of latent dimensions
    depth=4                      number of hidden layers -> dropout_rates
    batchnorm=True               (is_batchnorm)
    epochs=10001                 (max_epoch)
    lr=1e-3                      (learning_rate)
    batch_size=100
    test_slice=1
    weight_model=Uniform_Heavy   sets atom_selection unless given explicitly
    save_dir=runs                output root
    model_name=...               default: "<dcd-stem>_<arch>_D<depth>"

Examples
    deep-mms traj.dcd top.prmtop latents=8 depth=4
    deep-mms traj.dcd top.pdb architecture=transformer latents=16 depth=6 epochs=5000
    deep-mms traj.dcd top.prmtop latents=8 --dry-run
"""

import argparse
import json
import sys

from .config import assemble_config, parse_value
from .dispatch import build_harness, known_architectures
from .utils import printf


def build_parser():
    """Construct the argparse parser for the ``deep-mms`` command."""
    p = argparse.ArgumentParser(
        prog="deep-mms",
        description="Train a Deep-MMS model from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="key=value overrides: latents, depth, batchnorm, architecture, "
               "epochs, lr, batch_size, weight_model, save_dir, model_name, ...",
    )
    p.add_argument("fname_dcd", help="trajectory .dcd file")
    p.add_argument("fname_topology", help="topology .pdb/.prmtop file")
    p.add_argument("overrides", nargs="*", metavar="key=value",
                   help="config overrides, e.g. latents=8 depth=4 batchnorm=True")
    p.add_argument("--n-epochs", type=int, default=1000,
                   help="epochs of standard training before the auto-stop phase "
                        "(default: 1000)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the assembled config and exit without training")
    p.add_argument("--list-architectures", action="store_true",
                   help="list the available architectures and exit")
    return p


def main(argv=None):
    """Entry point for the ``deep-mms`` console script."""
    argv = sys.argv[1:] if argv is None else list(argv)

    # Handle the info flag before requiring the positional data paths.
    if "--list-architectures" in argv:
        print("\n".join(known_architectures()))
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = {}
    for token in args.overrides:
        if "=" not in token:
            parser.error(f"expected key=value, got {token!r}")
        key, value = token.split("=", 1)
        overrides[key] = parse_value(value)

    cfg = assemble_config(args.fname_dcd, args.fname_topology, overrides)

    if cfg["architecture"] not in known_architectures():
        parser.error(
            f"unknown architecture {cfg['architecture']!r}; "
            f"choose from {known_architectures()}"
        )

    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0

    printf(
        f"Training architecture={cfg['architecture']!r} "
        f"latent_dim={cfg['latent_dim']} depth={len(cfg['dropout_rates'])} "
        f"is_batchnorm={cfg['is_batchnorm']}"
    )
    harness = build_harness(cfg["architecture"], cfg, from_json_params=True)
    harness.MAIN_train(n_epochs=args.n_epochs, verbose=100)
    printf("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
