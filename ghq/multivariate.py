from typing import Callable
from itertools import product
from functools import partial

from numpy.polynomial.hermite_e import hermegauss
from jax import numpy as jnp, jit, Array, vmap
from jax.scipy.linalg import cho_factor, solve_triangular


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
        Approximate Gaussian expectation.
    """
    cov_sqrt, _ = cho_factor(cov, lower=True)
    return _multivariate_cov_sqrt(integrand, mean, cov_sqrt, degree)


def _multivariate_cov_sqrt(
    integrand: Callable[[Array], Array],
    mean: Array,
    cov_sqrt: Array,
    degree: int,
) -> float:
    dim = mean.size

    def standardized_integrand(x):
        return integrand(mean + cov_sqrt @ x)

    x, w = hermegauss(degree)

    xn = jnp.array(list(product(*(x,) * dim)))
    wn = jnp.prod(jnp.array(list(product(*(w,) * dim))), 1)

    hx = vmap(standardized_integrand)(xn)
    wn = wn.reshape((degree**dim,) + (1,) * (hx.ndim - 1))
    return (wn * hx).sum(0) / jnp.sqrt(2 * jnp.pi) ** dim


@partial(jit, static_argnums=(0, 3))
def multivariate_importance(
    integrand: Callable[[Array], Array],
    mean: Array,
    cov: Array = None,
    degree: int = 32,
    lower: list | tuple | Array = None,
    upper: list | tuple | Array = None,
) -> Array:
    """
    Numerical integration using Gauss-Hermite over a multivariate Gaussian(s)
    with importance corrected weights.

    Not vectorised in mean and cov, but works with jax.vmap.

    .. math::

        \int integrand(x)dx
            = \int integrand(x) / N(x | mean, cov) * N(x | mean, cov) dx

    Args:
        integrand: Function to be integrated over f(x), vectorised in x.
        mean: Mean of multivariate Gaussian (dim,).
        cov: Covariance matrix of multivariate Gaussian (dim, dim).
        degree: Integer number of Gauss-Hermite points, defaults to 32.
        lower: Lower bounds of integration, defaults to -inf.
        upper: Upper bounds of integration, defaults to inf.
    Returns:
        Approximate integration value.
    """

    if lower is not None or upper is not None:
        raise NotImplementedError(
            "Constrained multivariate integration not implemented yet."
        )

    dim = mean.size

    if cov is None:
        cov = jnp.eye(dim)

    cov_sqrt, _ = cho_factor(cov, lower=True)

    def importance_integrand(x):
        pdf_evals = jnp.exp(multivariate_normal_log_pdf(x, mean, cov_sqrt))
        return jnp.where(pdf_evals > 0, integrand(x) / pdf_evals, 0)

    return _multivariate_cov_sqrt(importance_integrand, mean, cov_sqrt, degree)


def multivariate_normal_log_pdf(x, mean, cov_sqrt):
    """
    Log PDF of a multivariate Gaussian.

    Vectorized in x, not in mean and cov_sqrt. But works with jax.vmap.

    Args:
        x: Array of multivariate Gaussian samples (n, dim).
        mean: Mean of multivariate Gaussian (dim,).
        cov_sqrt: Lower triangular Cholesky decomposition
            of covariance matrix (dim, dim).

    Returns:
        Array of log PDF evaluations (n,).
    """

    dim = mean.size

    centered_x = x - mean
    y = solve_triangular(cov_sqrt, centered_x.T, lower=True).T
    maha = jnp.sum(y**2, axis=-1)
    log_det = 2 * jnp.sum(jnp.log(jnp.diag(cov_sqrt)))
    norm_const = dim * jnp.log(2 * jnp.pi) + log_det
    log_pdf = -0.5 * (norm_const + maha)
    return log_pdf
