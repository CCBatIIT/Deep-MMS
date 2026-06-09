"""
Model forward-pass tests.

Each test verifies:
  - model.init() succeeds with synthetic inputs
  - __call__ returns (decoded, z_mean, z_logvar) with correct shapes
  - decoded.shape == (BATCH, INPUT_SIZE)
  - z_mean.shape[0] == BATCH
  - encode() and decode() are individually callable
  - construct() roundtrips: sample from posterior and decode
  - hidden_layers_from_config() returns a list of correct length
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from conftest import BATCH, INPUT_SIZE, LATENTS, DROPOUT, N_ATOMS, make_model


# ---------------------------------------------------------------------------
# Import all model classes
# ---------------------------------------------------------------------------

from deepmms.models.vae import BatchNorm_VAE
from deepmms.models.beta_vae import BetaVAE
from deepmms.models.vq_vae import VQVAE
from deepmms.models.equivariant_vae import EquivariantVAE
from deepmms.models.perceiver_vae import PerceiverVAE
from deepmms.models.hierarchical_vae import HierarchicalVAE
from deepmms.models.se3_transformer import SE3TransformerVAE
from deepmms.models.transformer_vae import TransformerVAE
from deepmms.models.mamba_vae import MambaVAE
from deepmms.models.flow_vae import RealNVPFlow
from deepmms.models.mae_vae import MaskedAutoencoder
from deepmms.models.kan_vae import KANVAE
from deepmms.models.neat_vae import NEATAutoencoder


# ---------------------------------------------------------------------------
# Registry: (class, extra_kwargs)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    ("BatchNorm_VAE",    BatchNorm_VAE,    {}),
    ("BetaVAE",          BetaVAE,          {"beta": 2.0}),
    ("VQVAE",            VQVAE,            {"codebook_size": 8}),
    ("EquivariantVAE",   EquivariantVAE,   {"hidden_layers": (16, 16, 16), "dropout_rates": [0., 0., 0.]}),
    ("PerceiverVAE",     PerceiverVAE,     {"num_heads": 2, "n_latent_queries": 4}),
    ("HierarchicalVAE",  HierarchicalVAE,  {}),
    # SE3: needs well-separated atoms; supply via custom x fixture in conftest
    ("SE3TransformerVAE",SE3TransformerVAE,{"hidden_layers": (16,16,16), "dropout_rates":[0.,0.,0.], "d_scalar": 16, "d_vector": 4, "cutoff_dist": 10.0}),
    ("TransformerVAE",   TransformerVAE,   {"num_heads": 2, "hidden_layers": (16, 16)}),
    ("MambaVAE",         MambaVAE,         {"d_state": 4}),
    ("RealNVPFlow",      RealNVPFlow,      {"hidden_layers": tuple([INPUT_SIZE // 2] * 4), "dropout_rates": [0.]*4}),
    ("MaskedAutoencoder",MaskedAutoencoder,{"num_heads": 2, "mask_ratio": 0.5}),
    ("KANVAE",           KANVAE,           {"hidden_layers": (8, 8), "kan_n_grid": 3}),
    ("NEATAutoencoder",  NEATAutoencoder,  {"hidden_layers": (8, 8)}),
]

MODEL_IDS = [name for name, _, _ in MODEL_REGISTRY]


def _make_x(name, key):
    """
    Return a well-conditioned coordinate batch for the given model.

    SE3TransformerVAE uses direction vectors (r_j - r_i)/d_ij; atoms placed on
    a regular grid ensure no two atoms coincide and distances are well-defined.
    All other models use iid-normal coordinates.
    """
    if name == "SE3TransformerVAE":
        # Place N_ATOMS atoms on a line spaced 0.2 nm apart → no zero distances
        positions = jnp.tile(
            jnp.linspace(0.0, (N_ATOMS - 1) * 0.2, N_ATOMS)[:, None],
            (1, 3)
        ).reshape(-1)  # (D,)
        return jnp.tile(positions[None, :], (BATCH, 1))
    return jax.random.normal(key, (BATCH, INPUT_SIZE))


@pytest.fixture(scope="module", params=MODEL_REGISTRY, ids=MODEL_IDS)
def model_and_params(request):
    """
    Instantiate each registered model with its extra kwargs.
    Returns (name, model, params, x_batch, rng_key).
    """
    name, cls, extra = request.param
    model, params = make_model(cls, **extra)
    key = jax.random.PRNGKey(7)
    x = _make_x(name, key)
    return name, model, params, x, key


# ---------------------------------------------------------------------------
# Tests applied to every model
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Verify __call__ output shapes and types for all models."""

    def test_decoded_shape(self, model_and_params):
        """Decoded output must be (BATCH, INPUT_SIZE)."""
        name, model, params, x, key = model_and_params
        decoded, z_mean, z_logvar = model.apply(params, x, key, train=False)
        assert decoded.shape == (BATCH, INPUT_SIZE), \
            f"{name}: decoded shape {decoded.shape} != {(BATCH, INPUT_SIZE)}"

    def test_z_mean_batch_dim(self, model_and_params):
        """z_mean must have BATCH as its first dimension."""
        name, model, params, x, key = model_and_params
        _, z_mean, _ = model.apply(params, x, key, train=False)
        assert z_mean.shape[0] == BATCH, \
            f"{name}: z_mean.shape[0]={z_mean.shape[0]} != {BATCH}"

    def test_z_logvar_shape_matches_mean(self, model_and_params):
        """z_logvar must have the same shape as z_mean."""
        name, model, params, x, key = model_and_params
        _, z_mean, z_logvar = model.apply(params, x, key, train=False)
        assert z_mean.shape == z_logvar.shape, \
            f"{name}: z_mean {z_mean.shape} != z_logvar {z_logvar.shape}"

    def test_no_nans_in_output(self, model_and_params):
        """Decoded output must contain no NaN values."""
        name, model, params, x, key = model_and_params
        decoded, z_mean, _ = model.apply(params, x, key, train=False)
        assert not jnp.any(jnp.isnan(decoded)), f"{name}: NaN in decoded"
        assert not jnp.any(jnp.isnan(z_mean)), f"{name}: NaN in z_mean"

    def test_train_mode_runs(self, model_and_params):
        """Forward pass with train=True must not raise."""
        name, model, params, x, key = model_and_params
        # BatchNorm models need mutable=['batch_stats']
        try:
            result = model.apply(params, x, key, train=True,
                                 mutable=["batch_stats"])
            if isinstance(result, tuple) and len(result) == 2:
                (decoded, z_mean, z_logvar), _ = result
            else:
                decoded, z_mean, z_logvar = result
        except Exception:
            decoded, z_mean, z_logvar = model.apply(params, x, key, train=True)
        assert decoded.shape[0] == BATCH


class TestEncodeDecodeInterface:
    """Verify encode() and decode() methods independently."""

    def test_encode_returns_two_arrays(self, model_and_params):
        """encode() must return a 2-tuple (z_mean, z_logvar)."""
        name, model, params, x, key = model_and_params
        result = model.apply(params, x, method=model.encode,
                              z_rng=key, train=False)
        assert len(result) == 2, f"{name}: encode returned {len(result)} values"

    def test_decode_accepts_z_mean(self, model_and_params):
        """decode(z) must return an array of shape (BATCH, INPUT_SIZE).

        For HierarchicalVAE, encode() returns only z1 (global level); decode()
        expects the full concatenated [z1, z2] latent.  We construct a full-rank
        zero-padded input to test decode in isolation.
        """
        name, model, params, x, key = model_and_params
        _, z_mean, _ = model.apply(params, x, key, train=False)

        if name == "HierarchicalVAE":
            # z_mean is z1 (shape B, K//4); decode needs [z1, z2] (shape B, K)
            k1 = z_mean.shape[-1]
            k2 = LATENTS - k1
            z_full = jnp.concatenate([z_mean, jnp.zeros((BATCH, k2))], axis=-1)
            decoded = model.apply(params, z_full, method=model.decode,
                                  z_rng=key, train=False)
        else:
            decoded = model.apply(params, z_mean, method=model.decode,
                                  z_rng=key, train=False)

        assert decoded.shape == (BATCH, INPUT_SIZE), \
            f"{name}: decode output {decoded.shape}"

    def test_construct_runs(self, model_and_params):
        """construct(z_mean, z_logvar, z_rng) must return (BATCH, INPUT_SIZE)."""
        name, model, params, x, key = model_and_params
        _, z_mean, z_logvar = model.apply(params, x, key, train=False)
        out = model.apply(params, z_mean, z_logvar, key,
                          method=model.construct)
        assert out.shape == (BATCH, INPUT_SIZE), \
            f"{name}: construct output {out.shape}"


class TestHiddenLayersConfig:
    """Verify hidden_layers_from_config returns a valid configuration."""

    @pytest.mark.parametrize("cls,extra", [(c, e) for _, c, e in MODEL_REGISTRY],
                             ids=MODEL_IDS)
    def test_returns_list(self, cls, extra, json_params_base):
        """hidden_layers_from_config must return a non-empty list of ints."""
        result = cls.hidden_layers_from_config(
            INPUT_SIZE, LATENTS, DROPOUT, json_params_base
        )
        assert isinstance(result, list), f"{cls.__name__}: returned {type(result)}"
        assert len(result) > 0, f"{cls.__name__}: returned empty list"
        assert all(isinstance(w, int) and w > 0 for w in result), \
            f"{cls.__name__}: non-positive widths in {result}"


class TestRealNVPSpecific:
    """RealNVP-specific tests: invertibility and NLL loss."""

    def test_forward_inverse_roundtrip(self):
        """f^{-1}(f(x)) should equal x up to float32 tolerance."""
        model, params = make_model(
            RealNVPFlow,
            hidden_layers=tuple([INPUT_SIZE // 2] * 4),
            dropout_rates=[0.] * 4,
        )
        key = jax.random.PRNGKey(99)
        x = jax.random.normal(key, (BATCH, INPUT_SIZE))

        # Forward: x → z (encoded)
        decoded, z, _ = model.apply(params, x, key, train=False)
        # decoded is f^{-1}(f(x)), should ≈ x
        np.testing.assert_allclose(
            np.array(decoded), np.array(x), atol=1e-4,
            err_msg="RealNVPFlow: f^{-1}(f(x)) != x"
        )


class TestVQVAESpecific:
    """VQ-VAE specific tests: codebook quantisation."""

    def test_quantised_output_in_codebook(self):
        """After quantisation, z_q must be one of the codebook entries."""
        codebook_size = 8
        model, params = make_model(VQVAE, codebook_size=codebook_size)
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (BATCH, INPUT_SIZE))

        # __call__ returns (decoded, z_e, z_q); z_q is the quantised vector
        _, z_e, z_q = model.apply(params, x, key, train=False)
        codebook = params["params"]["quantizer"]["codebook"]  # (C, K)

        # Each z_q row should be the nearest codebook entry (dist ≈ 0)
        for b in range(BATCH):
            dists = jnp.sum((codebook - z_q[b]) ** 2, axis=-1)
            assert float(jnp.min(dists)) < 1e-5, \
                f"Row {b} of z_q not in codebook (min dist={float(jnp.min(dists)):.2e})"


class TestHVAESpecific:
    """Hierarchical VAE: z1 must be smaller than z2."""

    def test_z1_smaller_than_z2(self):
        """z1 has latents//4 dims; full latent has latents dims."""
        model, params = make_model(HierarchicalVAE)
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (BATCH, INPUT_SIZE))
        _, z_mean, _ = model.apply(params, x, key, train=False)
        assert z_mean.shape[-1] == max(1, LATENTS // 4), \
            f"HierarchicalVAE z1 dim {z_mean.shape[-1]} != {max(1, LATENTS // 4)}"
