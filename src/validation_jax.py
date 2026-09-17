"""JAX-based automatic differentiation: a second, independent ground truth for
src/greeks.py, alongside the finite-difference ground truth in
src/validation.py.

src/validation.py's bs_price + fd_central/fd_second approximate a derivative
by sampling the price at nearby points and trading truncation error against
floating-point cancellation to pick a step size. This module sidesteps that
tradeoff entirely: jax.grad differentiates bs_price_jax's own computational
graph and returns an exact derivative (no step size, no truncation error --
only ordinary floating-point rounding in the arithmetic itself). This is the
same technique (adjoint algorithmic differentiation, AAD) production
Greek-calculation engines use, so it's not just "another numerical check",
it's the industry's actual alternative to finite differences.

IMPORTANT -- 64-bit precision: JAX defaults to 32-bit floats. Left at that
default, jax.grad would be a *less* precise ground truth than a well-tuned
central difference (float32 gives ~1e-7 relative precision; float64 finite
differences in src/validation.py already do better than that). The line
below must run before any JAX array touches this process, which is why it's
at import time here rather than left for a notebook cell to remember.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.stats import norm

# jax.grad refuses to differentiate with respect to an integer argument, which
# numpy would silently upcast, so every parameter is annotated as a float and
# callers must pass float(S) rather than a bare int.


def bs_price_jax(S: float, K: float, r: float, q: float, sigma: float,
                 T: float) -> tuple[jax.Array, jax.Array]:
    """Black-Scholes call/put price, written with jax.numpy/jax.scipy so
    jax.grad can trace and differentiate it. Same formula as
    src.validation.bs_price -- only the array library changes."""
    d1 = (jnp.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    call = S * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    put = K * jnp.exp(-r * T) * norm.cdf(-d2) - S * jnp.exp(-q * T) * norm.cdf(-d1)
    return call, put


def call_price_jax(S: float, K: float, r: float, q: float, sigma: float,
                   T: float) -> jax.Array:
    """Scalar-output wrapper around bs_price_jax.

    jax.grad requires a function that returns a single scalar -- it can't
    differentiate a function returning a (call, put) tuple directly. Compose
    jax.grad on THIS (e.g. jax.grad(call_price_jax, argnums=0) for delta,
    jax.grad(jax.grad(call_price_jax, argnums=0), argnums=0) for gamma),
    not on bs_price_jax itself.
    """
    return bs_price_jax(S, K, r, q, sigma, T)[0]


def put_price_jax(S: float, K: float, r: float, q: float, sigma: float,
                  T: float) -> jax.Array:
    """Scalar-output wrapper around bs_price_jax -- see call_price_jax."""
    return bs_price_jax(S, K, r, q, sigma, T)[1]
