"""Independent numerical validation helpers: a from-scratch Black-Scholes pricer
and finite-difference operators.

Extracted from tests/test_greeks.py so the same ground-truth pricer and
finite-difference machinery used to validate src/greeks.py can also be
imported directly into a notebook (e.g. to plot how closely the closed-form
Greeks track finite differences across a range of inputs), instead of a third
copy of this code being written for that purpose.

bs_price() is written independently of src/greeks.py -- it shares no algebra
or code path with it. If the two disagree, it isn't because they made the
same mistake twice.
"""

import numpy as np
import scipy.stats as stats


def bs_price(S, K, r, q, sigma, T):
    """Black-Scholes call/put price, written independently of src/greeks.py."""
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    return call, put


def fd_central(f, x, h):
    """Central difference approximation of f'(x)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def fd_second(f, x, h):
    """Central difference approximation of f''(x)."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
