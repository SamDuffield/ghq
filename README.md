# ghq

Gauss-Hermite quadrature in JAX

Installed easily from PyPI:

```
pip install ghq
```

# Numerical Integration


## Univariate Gaussian integrals
Gauss-Hermite quadrature is a method for numerically integrating functions of the form:

$$
\int_{-\infty}^{\infty} f(x)  \mathbf{N}(x \mid \mu, \sigma^2) dx \approx \sum_{i=1}^\mathsf{degree} w_i f(x_i),
$$

where $\lbrace x_i, w_i \rbrace_{i=1}^\mathsf{degree}$ are chosen [deterministically](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).  The integral approximation can be executed easily with `ghq`:
```python
ghq.univariate(f, mu, sigma, degree=32)
```
where `f: Callable[[float], Array]` is a JAX vectorisable function and `degree` is an optional argument controlling the number of function evalutions, increasing `degree` increases the accuracy of the integral. (Note that the output of `f` can be multivariate but the input is univariate).

## Univariate unbounded integrals
More generally, we can use an importance-sampling-like approach to integrate functions of the form:

$$
\int_{-\infty}^{\infty} f(x) dx = \int_{-\infty}^{\infty} \frac{f(x)}{\mathbf{N}(x \mid \mu, \sigma^2)}  \mathbf{N}(x \mid \mu, \sigma^2)  dx \approx \sum_{i=1}^\mathsf{degree} w_i \frac{f(x_i)}{\mathbf{N}(x_i \mid \mu, \sigma^2)},
$$

and with `ghq`:
```python
ghq.univariate_importance(f, mu, sigma, degree=32)
```

## Univariate half-bounded integrals
Consider lower-bounded integrals of the form:

$$
\int_{a}^{\infty} f(x) dx =\int_{-\infty}^{\infty} f(a + e^y) e^y  dy \approx \sum_{i=1}^\mathsf{degree} w_i \frac{f(a + e^{y_i}) e^{y_i}}{\mathbf{N}(y_i \mid \mu, \sigma^2)},
$$

where we use the transformation $y = \log(x - a)$ to map the lower-bounded integral to an unbounded integral. This can be approximated with `ghq`:
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

In `ghq` we have:
```python
ghq.univariate_importance(f, mu, sigma, degree=32, lower=a, upper=b).
```

The [Stan reference manual](https://mc-stan.org/docs/reference-manual/variable-transforms.html) provides an excellent reference for transformations of variables.



## Multivariate Gaussian integrals

$$
\int f(x)  \mathbf{N}(x \mid \mu, \Sigma) dx,
$$

in `ghq` is:
```python
ghq.multivariate(f, mu, Sigma, degree=32)
```
where `f: Callable[[Array], Array]` is a function that takes a multivariate input. 
Beware though that multivariate Gauss-Hermite quadrature has complexity 
$O(\text{degree}^d)$ where $d$ is the dimension of the integral, so it is not feasible 
for high-dimensional integrals.


## Multivariate unbounded integrals

Coming soon...


# Citation

If you use `ghq` in your research, please cite it using the following BibTeX entry:

```bibtex
@software{ghq,
  author = {Duffield, Samuel},
  title = {ghq: Gauss-Hermite quadrature in JAX},
  year = {2024},
  url = {https://github.com/SamDuffield/ghq}
}
```
