# ghq

Gauss-Hermite quadrature in JAX

# Installation

```
pip install git+https://github.com/SamDuffield/ghq.git
```

# Numerical Integration


## Univariate Gaussian integrals
Gauss-Hermite quadrature is a method for numerically integrating functions of the form:

$$
\int_{-\infty}^{\infty} f(x)  \mathbf{N}(x \mid \mu, \sigma^2) dx \approx \sum_{i=1}^\mathsf{degree} w_i f(x_i),
$$

where $\{x_i, w_i\}_{i=1}^\mathsf{degree}$ are chosen [deterministically](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).  The integral approximation can be executed easily with `qhq`:
```python
ghq.univariate(f, mu, sigma, degree=32)
```
where `f: Callable[[float], float]` is a JAX vectorisable function.

## Univariate unbounded integrals
More generally, we can use an importance-sampling-like approach to integrate functions of the form:

$$
\int_{-\infty}^{\infty} f(x) dx = \int_{-\infty}^{\infty} \frac{f(x)}{\mathbf{N}(x \mid \mu, \sigma^2)}  \mathbf{N}(x \mid \mu, \sigma^2)  dx \approx \sum_{i=1}^\mathsf{degree} w_i \frac{f(x_i)}{\mathbf{N}(x_i \mid \mu, \sigma^2)},
$$

and with `qhq`:
```python
ghq.univariate_importance(f, mu, sigma, degree=32)
```

## Univariate half-bounded integrals
Consider lower-bounded integrals of the form:

$$
\int_{a}^{\infty} f(x) dx =\int_{-\infty}^{\infty} f(a + e^y) e^y  dy \approx \sum_{i=1}^\mathsf{degree} w_i \frac{f(a + e^{y_i}) e^{y_i}}{\mathbf{N}(y_i \mid \mu, \sigma^2)},
$$

where we use the transformation $y = \log(x - a)$ to map the lower-bounded integral to an unbounded integral. This can be approximated with `qhq`:
```python
ghq.univariate_importance(f, mu, sigma, degree=32, lower=a)
```
or for upper-bounded integrals over $[-\infty, b)$ using transformation $y = \log(b - x)$:
```python
ghq.univariate_importance(f, mu, sigma, degree=32, upper=b)
```

## Univariate bounded integrals
For doubly-bounded integrals in $[a, b)$ we have

$$
\int_{a}^{b} f(x) dx = \int_{-\infty}^{\infty} f\left(a + (b-a)\text{logit}^{-1}(y)\right)  (b-a)  \text{logit}^{-1}(y)  \left(1-\text{logit}^{-1}(y)\right)  dy,
$$

where we use the transfomation $y=\text{logit}(\frac{x-a}{b-a})$ with $\text{logit}(u)=\log\frac{u}{1-u}$ and $\text{logit}^{-1}(v) = \frac{1}{1+e^{-v}}$.

In `qhq` we have:
```python
ghq.univariate_importance(f, mu, sigma, degree=32, lower=a, upper=b).
```

The [Stan reference manual](https://mc-stan.org/docs/reference-manual/variable-transforms.html) provides an excellent reference for transformations of variables.
