"""
Tests for deepmms.data: Data_stream, train_test_split, powers_of_two_up_to.
No trajectory files required.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from deepmms.data import Data_stream, train_test_split, powers_of_two_up_to


class TestDataStream:
    """Data_stream must yield randomly permuted batches indefinitely."""

    def test_yields_correct_batch_size(self):
        """Each yielded batch must have exactly batch_size rows."""
        data = np.arange(100).reshape(100, 1).astype(float)
        stream = Data_stream(rng_seed=0, num_total=100, num_batches=10, batch_size=10, data=data)
        batch = next(iter(stream))
        assert batch.shape == (10, 1)

    def test_covers_full_dataset_in_one_epoch(self):
        """Indices across all batches in one pass must cover every sample."""
        n, b = 100, 10
        data = np.arange(n).reshape(n, 1).astype(float)
        stream = Data_stream(rng_seed=1, num_total=n, num_batches=n // b,
                             batch_size=b, data=data)
        seen = set()
        it = iter(stream)
        for _ in range(n // b):
            batch = next(it)
            seen.update(batch.flatten().tolist())
        assert len(seen) == n, "Not all indices were visited in one epoch"

    def test_different_seeds_give_different_order(self):
        """Two streams with different seeds should differ in order."""
        data = np.arange(50).reshape(50, 1).astype(float)
        s1 = Data_stream(0, 50, 5, 10, data)
        s2 = Data_stream(99, 50, 5, 10, data)
        b1 = next(iter(s1)).flatten().tolist()
        b2 = next(iter(s2)).flatten().tolist()
        assert b1 != b2, "Different seeds produced the same batch order"

    def test_same_seed_reproducible(self):
        """Same seed must produce identical batches."""
        data = np.arange(40).reshape(40, 1).astype(float)
        s1 = Data_stream(7, 40, 4, 10, data)
        s2 = Data_stream(7, 40, 4, 10, data)
        b1 = next(iter(s1)).flatten().tolist()
        b2 = next(iter(s2)).flatten().tolist()
        assert b1 == b2


class TestTrainTestSplit:
    """train_test_split must correctly partition frames."""

    def test_sizes_sum_to_total(self):
        """train + test frame counts must equal the total."""
        coords = jnp.ones((100, 30))
        for test_slice in [1, 2, 3, 4, 5]:
            train, test = train_test_split(coords, test_slice)
            assert train.shape[0] + test.shape[0] == 100, \
                f"test_slice={test_slice}: {train.shape[0]}+{test.shape[0]} != 100"

    def test_test_is_every_fifth_frame(self):
        """Test set must consist of every 5th frame starting at test_slice."""
        n = 50
        coords = jnp.arange(n * 3, dtype=float).reshape(n, 3)
        for ts in [1, 2, 3, 4, 5]:
            train, test = train_test_split(coords, ts)
            expected_test_indices = list(range(ts, n, 5))
            assert test.shape[0] == len(expected_test_indices), \
                f"test_slice={ts}: expected {len(expected_test_indices)} test frames"

    def test_no_overlap_between_train_and_test(self):
        """No frame should appear in both train and test."""
        n = 30
        coords = jnp.arange(n).reshape(n, 1).astype(float)
        train, test = train_test_split(coords, 1)
        train_set = set(np.array(train).flatten().tolist())
        test_set = set(np.array(test).flatten().tolist())
        assert train_set.isdisjoint(test_set), "Train and test sets overlap"

    def test_approximate_80_20_split(self):
        """Test set should be ~20% of total frames."""
        coords = jnp.ones((500, 30))
        train, test = train_test_split(coords, 1)
        ratio = test.shape[0] / 500
        assert 0.18 <= ratio <= 0.22, f"Test ratio {ratio:.2f} outside [0.18, 0.22]"


class TestPowersOfTwoUpTo:
    """powers_of_two_up_to must return the canonical latent sweep sequence."""

    def test_starts_with_1_2_3(self):
        """Sequence must always start with [1, 2, 3]."""
        result = powers_of_two_up_to(50)
        assert result[:3] == [1, 2, 3]

    def test_includes_four(self):
        """4 (= 2²) must appear for any limit >= 4."""
        assert 4 in powers_of_two_up_to(10)

    def test_does_not_exceed_limit(self):
        """No value in the sequence should exceed the limit."""
        for limit in [8, 23, 50, 327]:
            result = powers_of_two_up_to(limit)
            assert all(v <= limit for v in result), \
                f"Value exceeds limit={limit} in {result}"

    def test_sorted_and_unique(self):
        """Sequence must be strictly increasing."""
        result = powers_of_two_up_to(100)
        assert result == sorted(set(result))

    def test_known_sequence_for_n_atoms_10(self):
        """For 10 atoms: [1, 2, 3, 4, 8]."""
        result = powers_of_two_up_to(10)
        assert result == [1, 2, 3, 4, 8], f"Got {result}"
