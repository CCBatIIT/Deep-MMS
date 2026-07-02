"""
Architecture dispatch: map an ``architecture`` identifier to a training harness.

Both the ``deep-mms`` CLI (deepmms/cli.py) and the JSON-driven
scripts/train_dispatch.py build their harness through :func:`build_harness`, so
the architecture-to-model mapping lives in exactly one place.
"""

from .training.trainer import Experiment
from .models import (
    TransformerVAE, EquivariantVAE, PerceiverVAE,
    SE3TransformerVAE, MambaVAE, KANVAE,
)
from .training.specialized import (
    BetaVAETrainer, VQVAETrainer, HVAETrainer, FlowTrainer, MAETrainer,
)
from .training.evolutionary import NEATTrainer

# architecture -> model class for the generic Experiment harness
# (None means Experiment's default, BatchNorm_VAE).
_EXPERIMENT_MODELS = {
    "batchnorm_vae": None,
    "transformer": TransformerVAE,
    "equivariant": EquivariantVAE,
    "perceiver": PerceiverVAE,
    "se3": SE3TransformerVAE,
    "mamba": MambaVAE,
    "kan": KANVAE,
}

# architecture -> specialized Trainer (carries its own model_cls + custom loss).
_TRAINERS = {
    "beta_vae": BetaVAETrainer,
    "vq_vae": VQVAETrainer,
    "hierarchical": HVAETrainer,
    "flow": FlowTrainer,
    "mae": MAETrainer,
    "neat": NEATTrainer,
}


def known_architectures():
    """Return the sorted list of accepted architecture identifiers."""
    return sorted(set(_EXPERIMENT_MODELS) | set(_TRAINERS))


def build_harness(architecture, config, from_json_params=False):
    """
    Construct the training harness for an architecture and config.

    Parameters
    ----------
    architecture : str
        Identifier from :func:`known_architectures`.
    config : str or dict
        Path to a JSON config file, or a parameter dict when
        ``from_json_params`` is True.
    from_json_params : bool
        Treat ``config`` as a dict rather than a file path.

    Returns
    -------
    object
        A harness exposing ``MAIN_train(n_epochs, verbose)``.
    """
    if architecture in _EXPERIMENT_MODELS:
        model_cls = _EXPERIMENT_MODELS[architecture]
        if model_cls is None:
            return Experiment(config, from_json_params=from_json_params)
        return Experiment(config, from_json_params=from_json_params, model_cls=model_cls)
    if architecture in _TRAINERS:
        return _TRAINERS[architecture](config, from_json_params=from_json_params)
    raise ValueError(
        f"Unknown architecture {architecture!r}. Known: {known_architectures()}"
    )
