# Quantitative Finance Models

Derivatives pricing and risk models in Python: stochastic volatility, jump diffusion,
Monte Carlo VaR, and Black-Scholes Greeks cross-validated with finite differences and
automatic differentiation.

Twelve Jupyter notebooks, a shared `src/` package, and a test suite of 57 cases. Every
notebook executes end to end from a clean kernel and documents the model it implements
as a continuous-time SDE followed by the discretisation actually used in the code.

## What makes this more than a set of simulations

Closed-form formulas are easy to get subtly wrong in ways that still look plausible on a
plot, so every Greek in this repository is checked against two independently derived
numerical methods rather than one:

- **Finite differences** (`src/validation.py`), using numpy and scipy, with the step size
  chosen to balance truncation against floating-point cancellation.
- **Automatic differentiation** (`src/validation_jax.py`), using a JAX-traced pricer,
  which gives exact derivatives to floating-point precision with no step size to tune.

The two are deliberately kept as separate libraries. If they shared an implementation, a
single conceptual error could be inherited by both and neither check would catch it. The
agreement between them is the evidence, so the methods have to be independent for that
evidence to mean anything.

Rebuilding the Greeks notebook under this standard surfaced six real formula errors,
including a variable-scoping bug in gamma, a put/call branch error in charm, and a
double-scaling error in ultima.

## Contents

### `notebooks/stochastic_processes/`

| Notebook | What it covers |
|---|---|
| `Bachelier_Model` | Arithmetic Brownian motion, which permits negative prices. Sampled exactly rather than by Euler |
| `Geometric_Brownian_Motion` | The reference process for equity prices and the dynamics assumed by Black-Scholes |
| `Ornstein_Uhlenbeck_Process` | Mean reversion at rate $\kappa$, plus a seasonal variant with a time-dependent long run mean |

### `notebooks/vanilla_pricing/`

| Notebook | What it covers |
|---|---|
| `Black_Scholes_Greeks` | All twelve Greeks in closed form, first through third order, each validated two ways |
| `Cox_Ross_Rubinstein_Binomial_Tree` | Recombining lattice pricer, cross-checked against the closed-form binomial sum |
| `Event_Contract_Pricer` | A digital option priced three ways: flat volatility, Heston, and call spread replication |

### `notebooks/advanced_models/`

| Notebook | What it covers |
|---|---|
| `Heston_Model` | CIR stochastic variance correlated with the price, discretised under full truncation |
| `Bates_Model` | Heston volatility with a compound Poisson jump term layered on top |
| `Merton_Jump_Diffusion` | Geometric Brownian motion with jumps and a drift compensator |
| `SABR_Model` | Stochastic volatility on a forward, with $\beta$ controlling the elasticity |

### `notebooks/risk/`

| Notebook | What it covers |
|---|---|
| `Monte_Carlo_VaR` | VaR and Expected Shortfall under three return generators, and why the three converge over a quarter but not over a day |

### `notebooks/signal_research/`

| Notebook | What it covers |
|---|---|
| `Volatility_Signal_Research` | A volatility-scaled filter for detecting the trough of a drawdown, with the construction's own limitations documented |

## Layout

```
notebooks/          twelve notebooks, grouped by subject
src/
  models.py         shared CIR variance, Heston and Bates simulation
  greeks.py         closed-form Black-Scholes Greeks
  validation.py     finite-difference differentiation
  validation_jax.py JAX automatic differentiation
  plotting.py       error-sweep figures
  data.py           cached market data loader
tests/              pytest suite, 57 cases
data/               price cache, not tracked
```

Notebooks resolve `src/` by walking up to the directory containing it, so they run from
any depth without path configuration.

## Setup

```bash
git clone https://github.com/Matthew6162/Quantitative-Finance-Models.git
cd Quantitative-Finance-Models

python3 -m venv .venv          # Windows: python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned to the environment the notebooks were last run
in, Python 3.14.

If activation does not work on your shell, the full table of platform-specific commands
is in the [Python `venv` documentation](https://docs.python.org/3/library/venv.html). On
Windows, PowerShell blocks script execution by default, which
[Microsoft documents here](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies).

## Running the tests

```bash
pytest tests/ -q
```

Fifty-seven cases covering all twelve Greeks by finite difference and by put/call
identity, with independent JAX autodiff verification for the first-order ones.

## Reproducibility

Every simulation takes an `rng` argument and the notebooks set a seed explicitly, so a
clean run reproduces the figures in this repository exactly. The argument is normalised
through `np.random.default_rng`, which accepts `None` for a fresh non-deterministic
stream, an integer seed, or an existing generator, so several simulations can be driven
from one stream when that is what a comparison needs.

A seed fixes the sample rather than the conclusion. Results quoted in the notebooks are
reported with a standard error rather than as a single draw, and any claim made should
hold across seeds rather than only at the one chosen.

NumPy guarantees the `default_rng` stream for a given version rather than indefinitely,
which is part of why `requirements.txt` is pinned. Reproducibility here means
reproducible in this environment, not forever.

## A note on data

Ten of the twelve notebooks use synthetic parameters and run offline. `Monte_Carlo_VaR`
and `Volatility_Signal_Research` pull a price series through `src/data.py`, which caches
it as CSV under `data/` on first use and reads from that cache afterwards. The cache is
deliberately not tracked, so those two need a network connection the first time they are
run and none after that.

## Roadmap

- Calibrating a model to a real option surface rather than to chosen parameters
- A `src/pricing.py` covering European and American styles, with Longstaff-Schwartz
  cross-checked against the existing lattice
- Variance reduction and performance work, measured as standard error at a fixed compute
  budget rather than as speed and variance separately

## License

MIT, see [LICENSE](LICENSE).
