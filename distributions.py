"""Pure functions for probability distribution computations in geospatial contexts."""

import numpy as np
from scipy import stats


# ─── Beta Distribution (binary land-cover classification) ─────────────────────

def beta_prior(alpha: float, beta_param: float, x: np.ndarray) -> np.ndarray:
    """Compute the Beta probability density function over x."""
    return stats.beta.pdf(x, alpha, beta_param)


def beta_posterior(prior_alpha: float, prior_beta: float,
                   observed_forest: int, observed_total: int) -> tuple[float, float]:
    """Update a Beta prior with binomial observation evidence.

    Returns the posterior (alpha, beta) parameters.
    In Bayesian conjugate updating for a binomial likelihood with a Beta prior:
        posterior_alpha = prior_alpha + successes
        posterior_beta  = prior_beta + failures
    """
    posterior_alpha = prior_alpha + observed_forest
    posterior_beta = prior_beta + (observed_total - observed_forest)
    return posterior_alpha, posterior_beta


def beta_summary(alpha: float, beta_param: float) -> dict:
    """Compute summary statistics for a Beta distribution."""
    mean = alpha / (alpha + beta_param)
    mode = (alpha - 1) / (alpha + beta_param - 2) if alpha > 1 and beta_param > 1 else None
    variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
    ci_low, ci_high = stats.beta.ppf([0.025, 0.975], alpha, beta_param)
    return {
        "mean": mean,
        "mode": mode,
        "variance": variance,
        "std": np.sqrt(variance),
        "ci_95": (ci_low, ci_high),
    }


# ─── Normal Distribution (measurement uncertainty) ───────────────────────────

def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute the Normal probability density function."""
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def normal_confidence_interval(mu: float, sigma: float, n: int,
                                confidence: float = 0.95) -> tuple[float, float]:
    """Compute the confidence interval for a sample mean.

    The standard error of the mean decreases as sqrt(n), so more measurements
    narrow the interval around the true value.
    """
    standard_error = sigma / np.sqrt(n)
    z = stats.norm.ppf((1 + confidence) / 2)
    return mu - z * standard_error, mu + z * standard_error


def simulate_sensor_measurements(true_value: float, noise_std: float,
                                  n_samples: int, seed: int = 42) -> np.ndarray:
    """Simulate noisy sensor measurements around a true value.

    Models the kind of measurement noise you get from elevation sensors,
    temperature probes, or spectral reflectance readings.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=true_value, scale=noise_std, size=n_samples)


# ─── Dirichlet Distribution (multi-class land cover) ─────────────────────────

def dirichlet_pdf_grid(alphas: list[float], grid_resolution: int = 100) -> dict:
    """Compute Dirichlet density on a simplex grid for 3 categories.

    Returns a dict with grid coordinates and density values for plotting
    on a triangular simplex (ternary plot).
    """
    if len(alphas) != 3:
        raise ValueError("This visualization supports exactly 3 categories")

    # Generate points on the 2-simplex
    points = []
    densities = []
    step = 1.0 / grid_resolution

    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1 - i):
            k = grid_resolution - i - j
            p1 = i * step
            p2 = j * step
            p3 = k * step

            # Avoid exact 0 or 1 (causes infinite density for some alpha values)
            p1 = max(p1, 1e-6)
            p2 = max(p2, 1e-6)
            p3 = max(p3, 1e-6)

            # Normalize to sum to 1
            total = p1 + p2 + p3
            p1, p2, p3 = p1 / total, p2 / total, p3 / total

            point = [p1, p2, p3]
            density = stats.dirichlet.pdf(point, alphas)
            points.append(point)
            densities.append(density)

    return {
        "points": np.array(points),
        "densities": np.array(densities),
    }


def dirichlet_sample(alphas: list[float], n_samples: int = 1000,
                     seed: int = 42) -> np.ndarray:
    """Draw samples from a Dirichlet distribution.

    Each row sums to 1 and represents a possible land-cover composition
    (e.g., [0.6 forest, 0.3 water, 0.1 urban]).
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alphas, size=n_samples)


def dirichlet_posterior(prior_alphas: list[float],
                        observations: list[int]) -> list[float]:
    """Update a Dirichlet prior with categorical observation counts.

    The Dirichlet is conjugate to the multinomial likelihood, so the update
    is simply: posterior_alpha_i = prior_alpha_i + observation_count_i
    """
    return [a + c for a, c in zip(prior_alphas, observations)]


def dirichlet_summary(alphas: list[float]) -> dict:
    """Compute summary statistics for a Dirichlet distribution."""
    alpha_sum = sum(alphas)
    means = [a / alpha_sum for a in alphas]
    variances = [
        (a * (alpha_sum - a)) / (alpha_sum ** 2 * (alpha_sum + 1))
        for a in alphas
    ]
    return {
        "means": means,
        "variances": variances,
        "concentration": alpha_sum,
    }
