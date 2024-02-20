from jax import numpy as jnp, jit, random

import ghq


def test_zero():
    def f(x):
        return 0.0

    assert jnp.isclose(jit(ghq.univariate, static_argnums=(0,))(f, 0, 1), 0.0)
    assert jnp.isclose(
        jit(ghq.univariate_importance, static_argnums=(0,))(f, 0, 1), 0.0, atol=1e-3
    )
    assert jnp.isclose(
        jit(ghq.univariate_importance, static_argnums=(0,))(f, 0, 1, lower=0),
        0.0,
        atol=1e-3,
    )
    assert jnp.isclose(
        jit(ghq.univariate_importance, static_argnums=(0,))(f, 0, 1, upper=0),
        0.0,
        atol=1e-3,
    )
    assert jnp.isclose(
        jit(ghq.univariate_importance, static_argnums=(0,))(f, 0, 1, lower=0, upper=1),
        0.0,
        atol=1e-3,
    )


def test_x():
    def f(x):
        return x

    assert jnp.isclose(ghq.univariate(f, 0, 1), 0.0)
    assert jnp.isclose(
        ghq.univariate_importance(f, 0, 1, lower=-1, upper=1), 0.0, atol=1e-3
    )


def test_abs_x():
    def f(x):
        return jnp.abs(x)

    assert jnp.isclose(
        ghq.univariate(f, 0, 1, degree=50),
        f(random.normal(key=random.PRNGKey(0), shape=(10000,))).mean(),
        atol=1e-2,
    )
    assert jnp.isclose(
        ghq.univariate_importance(f, 0, 1, lower=-1, upper=1), 1.0, atol=1e-2
    )


def test_x2():
    def f(x):
        return jnp.square(x)

    assert jnp.isclose(ghq.univariate(f, 0, 1), 1.0)
    assert jnp.isclose(
        ghq.univariate_importance(f, 0, 1, lower=-1, upper=1), 2 / 3, atol=1e-2
    )


def test_polynomial():
    def f(x):
        return 0.1 * x**5 - 0.3 * x**3 + 2 * x**2 + 3 * x + 4

    mn = 0.3
    sd = 0.5

    assert jnp.isclose(
        ghq.univariate(f, mn, sd, degree=50),
        f(mn + sd * random.normal(key=random.PRNGKey(0), shape=(10000,))).mean(),
        atol=1e-1,
    )
    assert jnp.isclose(
        ghq.univariate_importance(f, mn, sd, degree=50, lower=-1, upper=1),
        9.33333,
        atol=1e-1,
    )


def test_matrix_integrand():
    def f(x):
        return x**2 * jnp.eye(3)

    assert jnp.allclose(jit(ghq.univariate, static_argnums=(0,))(f, 0, 1), jnp.eye(3))
