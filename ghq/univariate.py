from typing import Callable
from functools import partial

from numpy.polynomial.hermite_e import hermegauss
from jax import numpy as jnp, jit, vmap
from jax.scipy.stats import norm


def logit(u: float) -> float:
    return jnp.log(u / (1 - u))


def inverse_logit(x: float) -> float:
    return 1 / (1 + jnp.exp(-x))


@partial(jit, static_argnums=(0, 3))
def univariate(
    integrand: Callable[[jnp.ndarray], jnp.ndarray],
    mean: jnp.ndarray,
    sd: jnp.ndarray,
    degree: int = 32,
) -> jnp.ndarray:
    """
    Gauss-Hermite integration over 1-D Gaussian(s).
    https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature

    .. math::

        \int integrand(x, extra_params) \text{N}(x | mean, sd^2) dx

    Args:
        integrand: Function to be integrated over f(x, extra_param),
            vectorised in x and extra_param if present.
        mean: Array of n means each corresponding to a 1-D Gaussian (n,).
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,).
        degree: Integer number of Gauss-Hermite points, defaults to 32.
    Returns:
        out: Array of n approximate 1D Gaussian expectations.
    """
    n = mean.size
    x, w = hermegauss(degree)
    w = w[..., jnp.newaxis]  # extend shape to (degree, 1)
    x = jnp.repeat(x[..., jnp.newaxis], n, axis=1)  # extend shape to (degree, n)
    x = sd * x + mean
    hx = vmap(integrand)(x)
    return (w * hx).sum(0) / jnp.sqrt(2 * jnp.pi)


@partial(jit, static_argnums=(0, 3))
def univariate_importance(
    integrand: Callable[[jnp.ndarray], jnp.ndarray],
    mean: jnp.ndarray = jnp.zeros(1),
    sd: jnp.ndarray = jnp.ones(1),
    degree: int = 32,
    lower: float = -jnp.inf,
    upper: float = jnp.inf,
) -> jnp.ndarray:
    """
    Numerical integration using Gauss-Hermite over a 1-D Gaussian(s)
    with importance corrected weights.

    .. math::

        \int integrand(x, extra_params)dx
            = \int integrand(x, extra_params) / N(x | mean, sd^2) * N(x | mean, sd^2) dx

    Args:
        integrand: Function to be integrated over f(x, extra_param),
            vectorised in x and extra_param if present.
        mean: Array of n means each corresponding to a 1-D Gaussian (n,), defaults to 0.
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,), defaults to 1.
        degree: Integer number of Gauss-Hermite points, defaults to 32.
        lower: Lower bound of integration, defaults to -inf.
        upper: Upper bound of integration, defaults to inf.
    Returns:
        out: Array of n approximate 1D integrations.
    """

    if lower != -jnp.inf and upper == jnp.inf:
        def transformed_integrand(y):
            ey = jnp.exp(y)
            return integrand(lower + ey) * ey
    elif lower == -jnp.inf and upper != jnp.inf:
        def transformed_integrand(y):
            ey = jnp.exp(y)
            return -integrand(upper - ey) * ey
    elif lower != -jnp.inf and upper != jnp.inf:
        def transformed_integrand(y):
            ily = inverse_logit(y)
            return integrand(lower + (upper - lower) * ily) * (upper - lower) * ily * (1 - ily)
    else:
        transformed_integrand = integrand

    def importance_integrand(x):
        return transformed_integrand(x) / norm.pdf(x, loc=mean, scale=sd)

    return univariate(mean, sd, importance_integrand, degree)
