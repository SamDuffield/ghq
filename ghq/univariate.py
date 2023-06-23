from typing import Callable
from functools import partial

from numpy.polynomial.hermite_e import hermegauss
from jax import numpy as jnp, jit
from jax.scipy.stats import norm


@partial(jit, static_argnums=(2, 4))
def univariate(
    mean: jnp.ndarray,
    sd: jnp.ndarray,
    integrand: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    extra_params: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    """
    Gauss-Hermite integration over 1-D Gaussian(s).
    https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature
    
    .. math::
    
        \int integrand(x, extra_params) \text{N}(x | mean, sd^2) dx
        
    Args:
        mean: Array of n means each corresponding to a 1-D Gaussian (n,).
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,).
        integrand: Function to be integrated over f(x, extra_param),
            vectorised in x and extra_param if present.
        extra_params: Extra params to integrand function.
        degree: Integer number of Gauss-Hermite points.
    Returns:
        out: Array of n approximate 1D Gaussian expectations.
    """
    n = mean.size
    x, w = hermegauss(degree)
    w = w[..., jnp.newaxis]  # extend shape to (degree, 1)
    x = jnp.repeat(x[..., jnp.newaxis], n, axis=1)  # extend shape to (degree, n)
    x = sd * x + mean
    hx = integrand(x, extra_params)
    return (w * hx).sum(0) / jnp.sqrt(2 * jnp.pi)


@partial(jit, static_argnums=(0, 2))
def univariate_importance(
    integrand: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    extra_params: jnp.ndarray,
    degree: int,
    mean: jnp.ndarray = jnp.zeros(1),
    sd: jnp.ndarray = jnp.ones(1),
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
        extra_params: Extra params to integrand function.
        degree: Integer number of Gauss-Hermite points.
        mean: Array of n means each corresponding to a 1-D Gaussian (n,), defaults to 0.
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,), defaults to 1.
    Returns:
        out: Array of n approximate 1D integrations.
    """

    def importance_integrand(x, eps):
        return integrand(x, eps) / norm.pdf(x, loc=mean, scale=sd)

    return univariate(mean, sd, importance_integrand, extra_params, degree)
