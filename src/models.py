"""Shared stochastic-volatility path simulators.

Extracted because the Heston variance/price update step was previously copy-pasted
across notebooks/Heston_Model.ipynb, notebooks/Bates_Model.ipynb, and
notebooks/Event_Contract_Pricer.ipynb. All three now call the functions below
instead. Random draw order within each function matches the original inline
code exactly (verified against the pre-refactor notebooks with a fixed seed)
so this is a pure move, not a behavioural change.
"""

import numpy as np


def generate_correlated_normals(n, num_paths, rho):
    """Two standard-normal shock arrays of shape (n, num_paths), correlated by rho.

    Draw order matches the original notebooks: the S-shock is drawn first, then
    the independent Z used to build the correlated v-shock.
    """
    epsilon_S = np.random.normal(0, 1, (n, num_paths))
    epsilon_v = rho * epsilon_S + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, (n, num_paths))
    return epsilon_S, epsilon_v


def simulate_cir_variance(v0, n, dt, kappa, theta, xi, epsilon_v):
    """Euler-Maruyama discretisation of the CIR variance process used by Heston/Bates.

    v0 is the starting variance -- callers pass theta (long-run mean) or sigma**2
    (an observed spot vol) depending on context, so it is a required argument
    rather than defaulted to either.
    """
    num_paths = epsilon_v.shape[1]
    v = np.zeros((n + 1, num_paths))
    v[0, :] = v0
    for i in range(n):
        v[i + 1] = v[i] + kappa * (theta - np.maximum(v[i], 0)) * dt + xi * np.sqrt(np.maximum(v[i], 0)) * np.sqrt(dt) * epsilon_v[i]
    return v


def simulate_heston(S0, v0, n, dt, kappa, theta, xi, rho, mu, num_paths):
    """Pure Heston stochastic-volatility price paths (no jumps).

    Used directly by Heston_Model.ipynb and Event_Contract_Pricer.ipynb, which
    previously duplicated this loop inline with identical update equations.
    """
    epsilon_S, epsilon_v = generate_correlated_normals(n, num_paths, rho)
    v = simulate_cir_variance(v0, n, dt, kappa, theta, xi, epsilon_v)

    S = np.zeros((n + 1, num_paths))
    S[0, :] = S0
    for i in range(n):
        S[i + 1] = S[i] + mu * S[i] * dt + np.sqrt(np.maximum(v[i], 0)) * S[i] * np.sqrt(dt) * epsilon_S[i]

    return S, v


def simulate_bates(S0, n, dt, theta, kappa, xi, rho, mu, lam, mu_j, sigma_j, num_paths):
    """Heston stochastic-volatility paths with a Merton-style jump component added.

    Reuses the same shock generation and CIR variance step as simulate_heston;
    only the price update differs (jump terms layered on top of the Heston
    diffusion). Matches the original notebooks/Bates_Model.ipynb Bates()
    function signature and return value (S only) so the call site there is
    unchanged apart from the import.
    """
    epsilon_S, epsilon_v = generate_correlated_normals(n, num_paths, rho)
    v = simulate_cir_variance(theta, n, dt, kappa, theta, xi, epsilon_v)

    k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
    dN = np.random.poisson(lam * dt, size=(n, num_paths))
    Y = np.random.normal(loc=mu_j * dN, scale=sigma_j * np.sqrt(dN), size=(n, num_paths))

    S = np.zeros((n + 1, num_paths))
    S[0, :] = S0
    for i in range(n):
        S[i + 1] = (S[i]
                    + S[i] * (mu - lam * k) * dt
                    + np.sqrt(np.maximum(v[i], 0)) * S[i] * np.sqrt(dt) * epsilon_S[i]
                    + S[i] * (np.exp(Y[i]) - 1))

    return S
