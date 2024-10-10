from jax import numpy as jnp, jit, vmap
from jax.scipy.stats import multivariate_normal

import ghq


def test_zero():
    def f(x):
        return 0.0

    mean = jnp.array([0.0])
    cov = jnp.array([[1.0]])
    degree = 32
    result = jit(ghq.multivariate, static_argnums=(0, 3))(f, mean, cov, degree)
    assert jnp.isclose(result, 0.0)
    importance_result = jit(ghq.multivariate_importance, static_argnums=(0, 3))(
        f, mean, cov, degree
    )
    assert jnp.isclose(importance_result, 0.0)

    mean = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, -0.3], [-0.3, 1.0]])
    degree = 32
    result = ghq.multivariate(f, mean, cov, degree)
    assert jnp.isclose(result, 0.0)
    importance_result = jit(ghq.multivariate_importance, static_argnums=(0, 3))(
        f, mean, cov, degree
    )
    assert jnp.isclose(importance_result, 0.0)


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


def test_multivariate_normal_log_pdf():
    mean = jnp.array([-0.2, 3.0])
    cov = jnp.array([[1.0, -0.3], [-0.3, 1.0]])
    x = jnp.arange(8).reshape(4, 2)
    cov_sqrt = jnp.linalg.cholesky(cov)
    result = ghq.multivariate_normal_log_pdf(x, mean, cov_sqrt)
    expected = vmap(multivariate_normal.logpdf, in_axes=(0, None, None))(x, mean, cov)
    assert jnp.allclose(result, expected)


def test_importance_exp_decay():
    def f(x):
        return jnp.exp(-jnp.abs(x).sum())

    mean = jnp.array([1.0, 0.3])
    cov = jnp.array([[1.0, -0.3], [-0.3, 1.0]])
    degree = 64
    result = ghq.multivariate_importance(f, mean, cov, degree)
    assert jnp.isclose(result, 4.0, atol=1e-1)


def test_importance_gaussian():
    def f(x):
        return jnp.exp(-jnp.sum(x**2) / 2)

    mean = jnp.array([-0.2, 0.8, 0.3])
    cov = jnp.array([[0.8, 0.5, 0.1], [0.5, 1.1, -0.3], [0.1, -0.3, 1.0]])
    degree = 64
    result = ghq.multivariate_importance(f, mean, cov, degree)
    assert jnp.isclose(result, (2 * jnp.pi) ** 1.5, atol=1e-1)
