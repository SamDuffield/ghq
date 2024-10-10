from typing import Callable
from functools import partial

from numpy.polynomial.hermite_e import hermegauss
from jax import numpy as jnp, jit, vmap, Array
from jax.lax import cond
from jax.scipy.stats import norm


def logit(u: float) -> float:
    return jnp.log(u / (1 - u))


def inverse_logit(x: float) -> float:
    return 1 / (1 + jnp.exp(-x))


@partial(jit, static_argnums=(0, 3))
def univariate(
    integrand: Callable[[Array], Array],
    mean: Array,
    sd: Array,
    degree: int = 32,
) -> Array:
    """
    Gauss-Hermite integration over 1-D Gaussian(s).
    https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature

    .. math::

        \int integrand(x) \text{N}(x | mean, sd^2) dx

    Args:
        integrand: Function to be integrated over f(x), vectorised.
        mean: Array of n means each corresponding to a 1-D Gaussian (n,).
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,).
        degree: Integer number of Gauss-Hermite points, defaults to 32.
    Returns:
        Array of n approximate 1D Gaussian expectations.
    """
    mean = jnp.array(mean)
    sd = jnp.array(sd)

    n = mean.size
    x, w = hermegauss(degree)
    x = jnp.repeat(x[..., jnp.newaxis], n, axis=1)  # extend shape to (degree, n)
    x = sd * x + mean
    hx = vmap(integrand)(x)
    w = w.reshape((degree,) + (1,) * (hx.ndim - 1))
    return jnp.squeeze((w * hx).sum(0) / jnp.sqrt(2 * jnp.pi))


def lower_bound_transform(y, integrand, lower):
    ey = jnp.exp(y)
    return integrand(lower + ey) * ey


def upper_bound_transform(y, integrand, upper):
    ey = jnp.exp(y)
    return -integrand(upper - ey) * ey


def bounded_transform(y, integrand, lower, upper):
    ily = inverse_logit(y)
    return integrand(lower + (upper - lower) * ily) * (upper - lower) * ily * (1 - ily)


@partial(jit, static_argnums=(0, 3))
def univariate_importance(
    integrand: Callable[[Array], Array],
    mean: Array = jnp.zeros(1),
    sd: Array = jnp.ones(1),
    degree: int = 32,
    lower: float = -jnp.inf,
    upper: float = jnp.inf,
) -> Array:
    """
    Numerical integration using Gauss-Hermite over a 1-D Gaussian(s)
    with importance corrected weights.

    .. math::

        \int integrand(x)dx
            = \int integrand(x) / N(x | mean, sd^2) * N(x | mean, sd^2) dx

    Args:
        integrand: Function to be integrated over f(x), vectorised in x.
        mean: Array of n means each corresponding to a 1-D Gaussian (n,), defaults to 0.
        sd: Array of n standard deviations each corresponding to a 1-D Gaussian (n,),
            defaults to 1.
        degree: Integer number of Gauss-Hermite points, defaults to 32.
        lower: Lower bound of integration, defaults to -inf.
        upper: Upper bound of integration, defaults to inf.
    Returns:
        Array of n approximate 1D integrations.
    """

    def transformed_integrand(y):
        res = cond(
            (lower != -jnp.inf) * (upper == jnp.inf),
            lambda z: lower_bound_transform(z, integrand, lower),
            lambda _: y,
            y,
        )

        res = cond(
            (lower == -jnp.inf) * (upper != jnp.inf),
            lambda z: upper_bound_transform(z, integrand, upper),
            lambda _: y,
            y,
        )

        res = cond(
            (lower != -jnp.inf) * (upper != jnp.inf),
            lambda z: bounded_transform(z, integrand, lower, upper),
            lambda _: y,
            y,
        )
        return res

    def importance_integrand(x):
        pdf_evals = norm.pdf(x, loc=mean, scale=sd)
        return jnp.where(pdf_evals > 0, transformed_integrand(x) / pdf_evals, 0)

    return univariate(importance_integrand, mean, sd, degree)
