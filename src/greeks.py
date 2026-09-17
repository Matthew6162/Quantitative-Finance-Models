"""Black-Scholes Greeks: closed-form price sensitivities and their desk-convention
scaled forms (P&L per one vol point, one day, one percent rate move, etc.).

Extracted from notebooks/Black_Scholes_Greeks.ipynb so the functions can be
imported by both that notebook and any other notebook or test that needs them,
instead of being redefined inline in each place. See tests/test_greeks.py for
the finite-difference validation suite that checks every function here against
an independently-written pricer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

Array = npt.NDArray[np.float64]
Num = npt.ArrayLike  # a float, or an array of floats broadcast elementwise

# Every function below is vectorised: pass scalars for a single option, or arrays
# to sweep a parameter. Returns are numpy arrays in either case, 0-d for scalar
# input. Greeks that differ between call and put return a (call, put) tuple; the
# rest return a single array because the value is identical for both.


def d_1(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num) -> Array:
    """The Black-Scholes d1 term, shared by every formula below."""
    return (np.log(S / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))


def safety(x: Num) -> Array:
    """Clamp a parameter away from zero so T=0 or sigma=0 cannot divide by zero.

    The true value at those boundaries is restored by the np.where guard at the
    end of each Greek, so this only protects the intermediate arithmetic.
    """
    return np.where(x > 0, x, 1e-9)


def theta(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
          daily: bool = True) -> tuple[Array, Array]:
    """Theta, -dV/dT: the option's time decay.

    daily divides by 365 to express the value as P&L per calendar day, which is
    how a desk quotes it. Set daily=False for the raw annual derivative.
    """
    # making sure T != 0 and sigma != 0
    T_safe, sigma_safe = safety(T), safety(sigma)

    # calculating d1 and d2
    d1 = d_1(S, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    # calculating call and put
    term1 = -(S * sigma_safe * np.exp(-q * T_safe) * stats.norm.pdf(d1)) / (2 * np.sqrt(T_safe))

    call = (term1
            - r * K * np.exp(-r * T_safe) * stats.norm.cdf(d2)
            + q * S * np.exp(-q * T_safe) * stats.norm.cdf(d1))
    put = (term1
           + r * K * np.exp(-r * T_safe) * stats.norm.cdf(-d2)
           - q * S * np.exp(-q * T_safe) * stats.norm.cdf(-d1))

    # checking if T > 0 otherwise returns 0
    call = np.where(T > 0, call, 0.0)
    put = np.where(T > 0, put, 0.0)

    # returns daily if daily = True
    if daily:
        return call / 365, put / 365
    else:
        return call, put


def delta(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num) -> tuple[Array, Array]:
    """Delta, dV/dS: sensitivity of the option price to the underlying."""
    # making sure T != 0 and sigma != 0
    T_safe, sigma_safe = safety(T), safety(sigma)

    # calculating d1 and terms for call and put
    d1 = d_1(S, K, r, q, sigma_safe, T_safe)

    term1 = np.exp(-q * T_safe)
    N_d1 = stats.norm.cdf(d1)

    # calculating call and put
    call = term1 * N_d1
    put = term1 * (N_d1 - 1)

    return call, put


def gamma(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num) -> Array:
    """Gamma, d2V/dS2: the rate of change of delta.

    Identical for a call and a put, so a single array is returned.
    """
    # making sure T != 0, sigma != 0, S != 0
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating d1
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)

    # calculating and returning gamma
    numerator = (np.exp(-q * T_safe) * stats.norm.pdf(d1))
    denominator = (S_safe * sigma_safe * np.sqrt(T_safe))
    gamma_val = numerator / denominator

    return np.where(T > 0, gamma_val, 0.0)


def vega(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
         scaled: bool = True) -> Array:
    """Vega, dV/dsigma: sensitivity to volatility.

    scaled divides by 100 to express the value per one volatility point, the
    desk convention. Identical for a call and a put.
    """
    # making sure T != 0 and sigma != 0
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating vega
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    vega_val = S_safe * np.exp(-q * T_safe) * stats.norm.pdf(d1) * np.sqrt(T_safe)

    # scaling vega if scaling
    output = vega_val / 100 if scaled else vega_val

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def rho(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
        scaled: bool = True) -> tuple[Array, Array]:
    """Rho, dV/dr: sensitivity to the risk-free rate.

    scaled divides by 100 to express the value per one percent rate move.
    """
    # making sure T != 0 and sigma != 0
    T_safe, sigma_safe = safety(T), safety(sigma)

    d2 = d_1(S, K, r, q, sigma_safe, T_safe) - sigma_safe * np.sqrt(T_safe)
    term1 = K * T_safe * np.exp(-r * T_safe)

    call = term1 * stats.norm.cdf(d2)
    put = -term1 * stats.norm.cdf(-d2)

    # scaling rho if scaling
    if scaled:
        call = call / 100
        put = put / 100

    return np.where(T > 0, call, 0.0), np.where(T > 0, put, 0.0)


def vanna(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
          scaled: bool = True) -> Array:
    """Vanna, d2V/dSdsigma: how delta moves as volatility moves.

    scaled divides by 100, per one volatility point. Identical for call and put.
    """
    # controlling for edge cases
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating vanna
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    vanna_val = -np.exp(-q * T_safe) * stats.norm.pdf(d1) * d2 / sigma_safe

    # scaling vanna if scaling = True
    output = vanna_val / 100 if scaled else vanna_val

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def volga(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
          scaled: bool = True) -> Array:
    """Volga, d2V/dsigma2: the convexity of the option price in volatility.

    scaled divides by 10,000, per one volatility point squared, since the
    derivative is second order in sigma. Identical for call and put.
    """
    # controlling for edge cases
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating volga
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    vega_raw = S_safe * np.exp(-q * T_safe) * stats.norm.pdf(d1) * np.sqrt(T_safe)
    volga_val = vega_raw * d1 * d2 / sigma_safe

    # scaling volga if scaling = True
    output = volga_val / 10_000 if scaled else volga_val

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def charm(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
          scaled: bool = True) -> tuple[Array, Array]:
    """Charm, d2V/dSdT: how delta decays with the passage of time.

    scaled divides by 365, per calendar day. The call and put branches differ in
    sign convention, so both are returned.
    """
    # controlling for edge cases
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating charm
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    term2 = np.exp(-q * T_safe) * stats.norm.pdf(d1) * (((r - q) / (sigma_safe * np.sqrt(T_safe))) - (d2 / (2 * T_safe)))

    call = q * np.exp(-q * T_safe) * stats.norm.cdf(d1) - term2
    put = -q * np.exp(-q * T_safe) * stats.norm.cdf(-d1) - term2

    call = np.where((T > 0) & (sigma > 0) & (S > 0), call, 0.0)
    put = np.where((T > 0) & (sigma > 0) & (S > 0), put, 0.0)

    # scaling charm if scaling = True
    if scaled:
        return call / 365, put / 365

    return call, put


def speed(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num) -> Array:
    """Speed, d3V/dS3: the rate of change of gamma in the underlying.

    Unscaled, since there is no standard desk convention for it. Identical for
    call and put.
    """
    # controlling for edge cases
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)

    output = -(np.exp(-q * T_safe) * stats.norm.pdf(d1) / (S_safe ** 2 * sigma_safe * np.sqrt(T_safe))) * (d1 / (sigma_safe * np.sqrt(T_safe)) + 1)

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def zomma(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
          scaled: bool = True) -> Array:
    """Zomma, d3V/dS2dsigma: how gamma moves as volatility moves.

    scaled divides by 100, per one volatility point. Identical for call and put.
    """
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    numerator = (np.exp(-q * T_safe) * stats.norm.pdf(d1))
    denominator = (S_safe * sigma_safe * np.sqrt(T_safe))
    gamma_val = numerator / denominator

    output = gamma_val * (d1 * d2 - 1) / sigma_safe
    output = output / 100 if scaled else output

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def color(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
          scaled: bool = True) -> Array:
    """Color, d3V/dS2dT: how gamma decays with the passage of time.

    scaled divides by 365, per calendar day. Identical for call and put.
    """
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    numerator = np.exp(-q * T_safe) * stats.norm.pdf(d1)
    denominator = S_safe * sigma_safe * np.sqrt(T_safe)
    gamma_val = numerator / denominator

    bracket = q + d1 * (r - q) / (sigma_safe * np.sqrt(T_safe)) + (1 - d1 * d2) / (2 * T_safe)
    output = gamma_val * bracket

    if scaled:
        output /= 365
    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def ultima(S: Num, K: Num, r: Num, q: Num, sigma: Num, T: Num,
           scaled: bool = True) -> Array:
    """Ultima, d3V/dsigma3: the third-order sensitivity to volatility.

    scaled divides by 1,000,000, per one volatility point cubed, since the
    derivative is third order in sigma. Identical for call and put.
    """
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    output = vega(S, K, r, q, sigma, T, scaled=False) / (sigma_safe ** 2) * (d1 * d2 * (d1 * d2 - 1) - (d1 ** 2 + d2 ** 2))

    if scaled:
        output /= 1_000_000

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)
