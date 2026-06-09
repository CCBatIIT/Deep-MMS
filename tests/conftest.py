"""
Shared pytest fixtures for Deep-MMS tests.

All fixtures use small synthetic inputs so tests run in seconds without
requiring real trajectory files or GPU resources.
"""

import sys
import os

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

BATCH = 4
N_ATOMS = 10          # 10 atoms × 3 = 30 features
INPUT_SIZE = N_ATOMS * 3   # 30
LATENTS = 4
DROPOUT = [0.0, 0.0]       # 2 hidden layers, no dropout for determinism


@pytest.fixture(scope="session")
def rng():
    """Reproducible JAX PRNGKey for all tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def x(rng):
    """Synthetic coordinate batch, shape (BATCH, INPUT_SIZE)."""
    return jax.random.normal(rng, (BATCH, INPUT_SIZE))


@pytest.fixture(scope="session")
def json_params_base():
    """Minimal JSON param dict used by hidden_layers_from_config tests."""
    return {
        "max_epoch": 10,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "dropout_rates": DROPOUT,
        "is_batchnorm": False,
        "resume_latest": False,
        "checkpoint_interval": 5,
        "atom_selection": "not element H",
        "weight_model": "Uniform_Heavy",
        "data_slice_start": 0,
        "data_slice_end": "None",
        "test_slice": 1,
        "latent_dim": LATENTS,
        "model_name": "TEST",
        "save_dir": "/tmp",
    }


def make_model(cls, **kwargs):
    """
    Instantiate a model with minimal valid arguments, merging any overrides.

    Returns (model, params) ready for apply() calls.
    """
    defaults = dict(
        input_size=INPUT_SIZE,
        hidden_layers=tuple([16] * len(DROPOUT)),
        latents=LATENTS,
        dropout_rates=DROPOUT,
        is_batchnorm=False,
    )
    defaults.update(kwargs)
    model = cls(**defaults)
    key = jax.random.PRNGKey(0)
    x_init = jnp.ones((BATCH, INPUT_SIZE))
    params = model.init(key, x_init, key, train=False)
    return model, params
