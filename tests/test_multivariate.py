from jax import numpy as jnp, jit

import ghq


def test_zero():
    def f(x):
        return 0.0

    mean = jnp.array([0.0])
    cov = jnp.array([[1.0]])
    degree = 32
    result = jit(ghq.multivariate, static_argnums=(0, 3))(f, mean, cov, degree)
    assert jnp.isclose(result, 0.0)

    mean = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, -0.3], [-0.3, 1.0]])
    degree = 32
    result = ghq.multivariate(f, mean, cov, degree)
    assert jnp.isclose(result, 0.0)


def test_x0():
    def f(x):
        return x[0]

    mean = jnp.array([0.0])
    cov = jnp.array([[1.0]])
    degree = 32
    result = ghq.multivariate(f, mean, cov, degree)
    assert jnp.isclose(result, 0.0)

    mean = jnp.array([1.0, 0.0])
    cov = jnp.array([[1.0, -0.3], [-0.3, 1.0]])
    degree = 32
    result = ghq.multivariate(f, mean, cov, degree)
    assert jnp.isclose(result, 1.0)


def test_corr():
    def f(x):
        return x[0] * x[1]

    mean = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, -0.3], [-0.3, 1.0]])
    degree = 32
    result = ghq.multivariate(f, mean, cov, degree)
    assert jnp.isclose(result, -0.3)


def test_matrix_integrand():
    def f(x):
        return jnp.outer(x, x)

    mean = jnp.array([0.0, 0.0, 0.0])
    cov = jnp.array([[1.0, -0.3, 0.2], [-0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])
    degree = 32
    result = ghq.multivariate(f, mean, cov, degree)
    assert jnp.allclose(result, cov)
