
from jax import numpy as jnp, jit

import ghq


def test_zero():

    def f(x): return 0.

    assert jnp.isclose(jit(ghq.univariate, static_argnums=(0,))(f, 0, 1), 0.)
    assert jnp.isclose(jit(ghq.univariate_importance, static_argnums=(0,))(f, 0, 1), 0., atol=1e-3)


def test_x():

    def f(x): return x

    assert jnp.isclose(ghq.univariate(f, 0, 1), 0.)
    assert jnp.isclose(ghq.univariate_importance(f, 0, 1, lower=-1, upper=1), 0., atol=1e-3)


def test_abs_x():

    def f(x): return jnp.abs(x)

    # assert jnp.isclose(ghq.univariate(f, 0, 1), 0.)
    assert jnp.isclose(ghq.univariate_importance(f, 0, 1, lower=-1, upper=1), 1., atol=1e-2)
