"""
Tests for deepmms.training.loss: atom_rmsd, give_weighted_rmsd_func, KL_loss.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from deepmms.training.loss import atom_rmsd, give_weighted_rmsd_func, KL_loss


BATCH = 4
N_ATOMS = 6
D = N_ATOMS * 3   # 18


class TestAtomRMSD:
    """atom_rmsd must satisfy symmetry, identity, and scaling properties."""

    def test_identical_frames_give_zero(self):
        """RMSD of a frame with itself must be exactly zero."""
        x = jnp.ones((BATCH, D))
        result = atom_rmsd(x, x)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6)

    def test_output_shape(self):
        """atom_rmsd must return one scalar per frame in the batch."""
        x = jnp.zeros((BATCH, D))
        y = jnp.ones((BATCH, D))
        result = atom_rmsd(x, y)
        assert result.shape == (BATCH,)

    def test_non_negative(self):
        """RMSD values must always be non-negative."""
        x = jnp.array(np.random.randn(BATCH, D))
        y = jnp.array(np.random.randn(BATCH, D))
        assert jnp.all(atom_rmsd(x, y) >= 0)

    def test_symmetric(self):
        """atom_rmsd(a, b) must equal atom_rmsd(b, a)."""
        a = jnp.array(np.random.randn(BATCH, D))
        b = jnp.array(np.random.randn(BATCH, D))
        np.testing.assert_allclose(
            np.array(atom_rmsd(a, b)),
            np.array(atom_rmsd(b, a)),
            atol=1e-6,
        )

    def test_known_value(self):
        """Manually computed RMSD for two simple frames must match."""
        # Frame a: all zeros. Frame b: all ones.
        # Per-atom displacement: (1,1,1) → squared: 3. Mean: 3. sqrt: sqrt(3).
        a = jnp.zeros((1, D))
        b = jnp.ones((1, D))
        expected = float(np.sqrt(3.0))
        result = float(atom_rmsd(a, b)[0])
        assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"


class TestWeightedRMSD:
    """give_weighted_rmsd_func must produce mass-weighted RMSD correctly."""

    def test_uniform_weights_match_atom_rmsd(self):
        """Uniform weights should give the same result as atom_rmsd."""
        weights = jnp.ones(N_ATOMS)
        weighted_fn = give_weighted_rmsd_func(weights)
        a = jnp.array(np.random.randn(BATCH, D))
        b = jnp.array(np.random.randn(BATCH, D))
        np.testing.assert_allclose(
            np.array(weighted_fn(a, b)),
            np.array(atom_rmsd(a, b)),
            atol=1e-5,
        )

    def test_identical_frames_give_zero(self):
        """Weighted RMSD of a frame with itself must be zero."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fn = give_weighted_rmsd_func(weights)
        x = jnp.ones((BATCH, D))
        np.testing.assert_allclose(np.array(fn(x, x)), 0.0, atol=1e-6)

    def test_output_shape(self):
        """Weighted RMSD must return (BATCH,) array."""
        weights = jnp.ones(N_ATOMS)
        fn = give_weighted_rmsd_func(weights)
        result = fn(jnp.zeros((BATCH, D)), jnp.ones((BATCH, D)))
        assert result.shape == (BATCH,)

    def test_non_negative(self):
        """Weighted RMSD must always be non-negative."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        fn = give_weighted_rmsd_func(weights)
        a = jnp.array(np.random.randn(BATCH, D))
        b = jnp.array(np.random.randn(BATCH, D))
        assert jnp.all(fn(a, b) >= 0)


class TestKLLoss:
    """KL_loss must return zero for a standard normal posterior."""

    def test_standard_normal_gives_zero(self):
        """KL[N(0,1) || N(0,1)] = 0."""
        mu = jnp.zeros((BATCH, 4))
        log_var = jnp.zeros((BATCH, 4))
        result = float(KL_loss(mu, log_var))
        assert abs(result) < 1e-5, f"Expected 0, got {result}"

    def test_non_negative(self):
        """KL divergence must be >= 0."""
        mu = jnp.array(np.random.randn(BATCH, 4))
        log_var = jnp.array(np.random.randn(BATCH, 4))
        assert float(KL_loss(mu, log_var)) >= -1e-5

    def test_larger_mean_increases_kl(self):
        """Moving the mean away from zero must increase KL."""
        mu0 = jnp.zeros((BATCH, 4))
        mu1 = jnp.ones((BATCH, 4)) * 5.0
        lv = jnp.zeros((BATCH, 4))
        assert KL_loss(mu1, lv) > KL_loss(mu0, lv)
