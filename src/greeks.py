"""Black-Scholes Greeks: closed-form price sensitivities and their desk-convention
scaled forms (P&L per one vol point, one day, one percent rate move, etc.).

Extracted from notebooks/Black_Scholes_Greeks.ipynb so the functions can be
imported by both that notebook and any other notebook or test that needs them,
instead of being redefined inline in each place. See tests/test_greeks.py for
the finite-difference validation suite that checks every function here against
an independently-written pricer.
"""

import numpy as np
import scipy.stats as stats

d_1 = lambda S, K, r, q, sigma, T: (np.log(S / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))


def safety(x):
    return np.where(x > 0, x, 1e-9)


def theta(S, K, r, q, sigma, T, daily=True):
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


def delta(S, K, r, q, sigma, T):
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


def gamma(S, K, r, q, sigma, T):
    # making sure T != 0, sigma != 0, S != 0
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating d1
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)

    # calculating and returning gamma
    numerator = (np.exp(-q * T_safe) * stats.norm.pdf(d1))
    denominator = (S_safe * sigma_safe * np.sqrt(T_safe))
    gamma_val = numerator / denominator

    return np.where(T > 0, gamma_val, 0.0)


def vega(S, K, r, q, sigma, T, scaled=True):
    # making sure T != 0 and sigma != 0
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating vega
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    vega_val = S_safe * np.exp(-q * T_safe) * stats.norm.pdf(d1) * np.sqrt(T_safe)

    # scaling vega if scaling
    output = vega_val / 100 if scaled else vega_val

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def rho(S, K, r, q, sigma, T, scaled=True):
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


def vanna(S, K, r, q, sigma, T, scaled=True):
    # controlling for edge cases
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    # calculating vanna
    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    vanna_val = -np.exp(-q * T_safe) * stats.norm.pdf(d1) * d2 / sigma_safe

    # scaling vanna if scaling = True
    output = vanna_val / 100 if scaled else vanna_val

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def volga(S, K, r, q, sigma, T, scaled=True):
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


def charm(S, K, r, q, sigma, T, scaled=True):
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


def speed(S, K, r, q, sigma, T):
    # controlling for edge cases
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)

    output = -(np.exp(-q * T_safe) * stats.norm.pdf(d1) / (S_safe ** 2 * sigma_safe * np.sqrt(T_safe))) * (d1 / (sigma_safe * np.sqrt(T_safe)) + 1)

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def zomma(S, K, r, q, sigma, T, scaled=True):
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    numerator = (np.exp(-q * T_safe) * stats.norm.pdf(d1))
    denominator = (S_safe * sigma_safe * np.sqrt(T_safe))
    gamma_val = numerator / denominator

    output = gamma_val * (d1 * d2 - 1) / sigma_safe
    output = output / 100 if scaled else output

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)


def color(S, K, r, q, sigma, T, scaled=True):
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


def ultima(S, K, r, q, sigma, T, scaled=True):
    T_safe, sigma_safe, S_safe = safety(T), safety(sigma), safety(S)

    d1 = d_1(S_safe, K, r, q, sigma_safe, T_safe)
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    output = vega(S, K, r, q, sigma, T, scaled=False) / (sigma_safe ** 2) * (d1 * d2 * (d1 * d2 - 1) - (d1 ** 2 + d2 ** 2))

    if scaled:
        output /= 1_000_000

    return np.where((T > 0) & (sigma > 0) & (S > 0), output, 0.0)
