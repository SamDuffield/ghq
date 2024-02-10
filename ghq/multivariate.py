from typing import Callable
from itertools import product
from functools import partial

from numpy.polynomial.hermite_e import hermegauss
from jax import numpy as jnp, jit, Array, vmap


@partial(jit, static_argnums=(0, 3))
def multivariate(
    integrand: Callable[[Array], Array],
    mean: Array,
    cov: Array,
    degree: int = 32,
) -> float:
    """
    Multivariate Gauss-Hermite integration.
    https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature

    Not vectorised in mean and cov, but works with jax.vmap.

    .. math::

        \int integrand(x) \text{N}(x | mean, cov) dx

    Args:
        integrand: Function to be integrated over f(x).
        mean: Mean of multivariate Gaussian.
        cov: Covariance matrix of multivariate Gaussian.
        degree: Integer number of Gauss-Hermite points, defaults to 32.
    Returns:
        out: Approximate Gaussian expectation (scalar float).
    """
    cov_sqrt = jnp.linalg.cholesky(cov)
    dim = mean.size

    def standardized_integrand(x):
        return integrand(mean + cov_sqrt @ x)

    x, w = hermegauss(degree)

    xn = jnp.array(list(product(*(x,) * dim)))
    wn = jnp.prod(jnp.array(list(product(*(w,) * dim))), 1)

    hx = vmap(standardized_integrand)(xn)
    return (wn * hx).sum(0) / jnp.sqrt(2 * jnp.pi) ** dim
