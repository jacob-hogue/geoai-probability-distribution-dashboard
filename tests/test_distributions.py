"""Tests for probability distribution functions."""

import numpy as np
import pytest
from distributions import (
    beta_posterior,
    beta_summary,
    normal_confidence_interval,
    simulate_sensor_measurements,
    dirichlet_posterior,
    dirichlet_summary,
    dirichlet_sample,
)


class TestBeta:
    def test_posterior_update_adds_counts(self):
        """Posterior alpha = prior alpha + successes, beta = prior beta + failures."""
        post_a, post_b = beta_posterior(1, 1, observed_forest=7, observed_total=10)
        assert post_a == 8  # 1 + 7
        assert post_b == 4  # 1 + 3

    def test_posterior_with_strong_prior(self):
        """A strong prior (alpha=50, beta=50) barely moves with a few observations."""
        post_a, post_b = beta_posterior(50, 50, observed_forest=3, observed_total=5)
        summary = beta_summary(post_a, post_b)
        # Mean should still be close to 0.5 despite 3/5 observed
        assert 0.48 < summary["mean"] < 0.52

    def test_posterior_with_weak_prior(self):
        """A weak prior (alpha=1, beta=1) shifts heavily with observations."""
        post_a, post_b = beta_posterior(1, 1, observed_forest=9, observed_total=10)
        summary = beta_summary(post_a, post_b)
        # Mean should be close to 0.9
        assert summary["mean"] > 0.8

    def test_summary_confidence_interval(self):
        """95% CI should contain the mean."""
        summary = beta_summary(10, 10)
        ci_low, ci_high = summary["ci_95"]
        assert ci_low < summary["mean"] < ci_high

    def test_summary_mode_undefined_for_uniform(self):
        """Mode is undefined for Beta(1,1) since alpha and beta are both 1."""
        summary = beta_summary(1, 1)
        assert summary["mode"] is None

    def test_summary_mode_defined(self):
        """Mode should be between 0 and 1 for alpha > 1 and beta > 1."""
        summary = beta_summary(5, 2)
        assert 0 < summary["mode"] < 1


class TestNormal:
    def test_confidence_interval_narrows_with_more_samples(self):
        """More measurements should give a tighter confidence interval."""
        ci_10 = normal_confidence_interval(100, 5, n=10)
        ci_100 = normal_confidence_interval(100, 5, n=100)
        width_10 = ci_10[1] - ci_10[0]
        width_100 = ci_100[1] - ci_100[0]
        assert width_100 < width_10

    def test_confidence_interval_contains_mean(self):
        ci_low, ci_high = normal_confidence_interval(50, 10, n=30)
        assert ci_low < 50 < ci_high

    def test_simulated_measurements_have_correct_shape(self):
        measurements = simulate_sensor_measurements(100, 5, 50)
        assert measurements.shape == (50,)

    def test_simulated_measurements_centered_near_true_value(self):
        measurements = simulate_sensor_measurements(100, 1, 10000)
        assert abs(measurements.mean() - 100) < 0.1


class TestDirichlet:
    def test_posterior_adds_counts(self):
        """Posterior alpha_i = prior_i + count_i."""
        post = dirichlet_posterior([1, 1, 1], [10, 5, 3])
        assert post == [11, 6, 4]

    def test_summary_means_sum_to_one(self):
        summary = dirichlet_summary([5, 3, 2])
        assert abs(sum(summary["means"]) - 1.0) < 1e-10

    def test_samples_sum_to_one(self):
        samples = dirichlet_sample([5, 3, 2], n_samples=100)
        row_sums = samples.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_samples_shape(self):
        samples = dirichlet_sample([2, 3, 5], n_samples=50)
        assert samples.shape == (50, 3)

    def test_high_concentration_gives_low_variance(self):
        """High alpha values (strong prior) should give low variance."""
        low = dirichlet_summary([1, 1, 1])
        high = dirichlet_summary([100, 100, 100])
        assert all(h < l for h, l in zip(high["variances"], low["variances"]))
