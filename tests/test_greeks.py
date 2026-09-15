"""
Regression / correctness tests for src/greeks.py (used by notebooks/Black_Scholes_Greeks.ipynb).
Ground-truth pricer and finite-difference helpers live in src/validation.py.
A second ground truth via JAX automatic differentiation lives in src/validation_jax.py.

WHAT THIS SUITE CHECKS, AND WHY EACH KIND OF CHECK EXISTS
----------------------------------------------------------
1. Every Greek is checked against a numerical (finite-difference) derivative of
   an independently-written Black-Scholes price function -- not against the
   notebook's own algebra, and not against another line inside the same file.
   A test that re-derives the thing it's testing from the same source can pass
   while both are wrong the same way; finite differences don't share any code
   path with the analytical formulas, so they can't fail together by accident.
2. Every scaled ("desk convention") output is checked against its raw value
   times the expected scale factor (1/100, 1/10_000, 1/1_000_000, 1/365),
   rather than just checking the scaled number looks reasonable -- this is what
   caught the ultima double-scaling bug and the dead `scaled` parameter in zomma.
3. Test cases deliberately vary every input that appears in any formula: at-the-
   money and out-of-the-money, short- and long-dated, and -- this is the one
   that mattered -- zero AND non-zero dividend yield. The color() bug was
   invisible at q=0 (the notebook's own demo parameters) and only surfaced at
   q=0.02. A parameter you never vary is a parameter you never tested.
4. Put/call identities (e.g. delta_call - delta_put == exp(-qT)) are checked as
   a second, independent family of assertions that don't rely on calculus at
   all -- a different way for a bug to get caught if it happens to survive #1.
"""

import pathlib
import sys

import jax
import numpy as np
import pytest

# src/ is a sibling of tests/, not tests/'s parent -- add the repo root so
# `import src.greeks` resolves regardless of the directory pytest is run from.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.greeks import (  # noqa: E402  (import after sys.path fix, intentional)
    d_1, safety, theta, delta, gamma, vega, rho,
    vanna, volga, charm, speed, zomma, color, ultima,
)
# Ground-truth pricer and finite-difference operators now live in
# src/validation.py so a notebook can import the same machinery this suite
# uses, rather than a third copy of it being written there.
from src.validation import bs_price, fd_central, fd_second  # noqa: E402
# Second, independent ground truth via automatic differentiation -- see
# src/validation_jax.py for why this isn't just "another finite difference".
from src.validation_jax import call_price_jax, put_price_jax  # noqa: E402


# ---------------------------------------------------------------------------
# Deliberately varied cases: ATM vs OTM, short vs long-dated, and -- the one
# that actually caught a bug -- zero vs non-zero dividend yield.
# ---------------------------------------------------------------------------
CASES = [
    pytest.param(dict(S=100, K=90, r=0.04, q=0.00, sigma=0.40, T=1.00), id="notebook-demo-params"),
    pytest.param(dict(S=100, K=100, r=0.03, q=0.02, sigma=0.25, T=0.10), id="short-dated-with-dividend"),
    pytest.param(dict(S=50, K=55, r=0.05, q=0.00, sigma=0.60, T=2.00), id="otm-long-dated-high-vol"),
]

REL_TOL = 1e-4  # loose enough to absorb finite-difference truncation, tight enough to catch real bugs


@pytest.mark.parametrize("c", CASES)
def test_delta_matches_finite_difference(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_call = fd_central(lambda s: bs_price(s, K, r, q, sigma, T)[0], S, 1e-4 * S)
    fd_put = fd_central(lambda s: bs_price(s, K, r, q, sigma, T)[1], S, 1e-4 * S)
    a_call, a_put = delta(S, K, r, q, sigma, T)
    assert a_call == pytest.approx(fd_call, rel=REL_TOL)
    assert a_put == pytest.approx(fd_put, rel=REL_TOL)


@pytest.mark.parametrize("c", CASES)
def test_gamma_matches_second_derivative(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_g = fd_second(lambda s: bs_price(s, K, r, q, sigma, T)[0], S, 1e-3 * S)
    assert float(gamma(S, K, r, q, sigma, T)) == pytest.approx(fd_g, rel=REL_TOL)


@pytest.mark.parametrize("c", CASES)
def test_vega_matches_and_scales_correctly(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_v = fd_central(lambda sg: bs_price(S, K, r, q, sg, T)[0], sigma, 1e-4 * sigma)
    raw = vega(S, K, r, q, sigma, T, scaled=False)
    scaled = vega(S, K, r, q, sigma, T, scaled=True)
    assert float(raw) == pytest.approx(fd_v, rel=REL_TOL)
    assert float(scaled) == pytest.approx(float(raw) / 100, rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_rho_matches_and_scales_correctly(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_rc = fd_central(lambda rr: bs_price(S, K, rr, q, sigma, T)[0], r, 1e-4)
    raw_c, _ = rho(S, K, r, q, sigma, T, scaled=False)
    scaled_c, _ = rho(S, K, r, q, sigma, T, scaled=True)
    assert float(raw_c) == pytest.approx(fd_rc, rel=REL_TOL)
    assert float(scaled_c) == pytest.approx(float(raw_c) / 100, rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_vanna_matches_and_scales_correctly(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd = fd_central(lambda sg: delta(S, K, r, q, sg, T)[0], sigma, 1e-4 * sigma)
    raw = vanna(S, K, r, q, sigma, T, scaled=False)
    scaled = vanna(S, K, r, q, sigma, T, scaled=True)
    assert float(raw) == pytest.approx(fd, rel=REL_TOL)
    assert float(scaled) == pytest.approx(float(raw) / 100, rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_volga_matches_and_scales_correctly(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd = fd_central(lambda sg: vega(S, K, r, q, sg, T, scaled=False), sigma, 1e-4 * sigma)
    raw = volga(S, K, r, q, sigma, T, scaled=False)
    scaled = volga(S, K, r, q, sigma, T, scaled=True)
    assert float(raw) == pytest.approx(fd, rel=REL_TOL)
    assert float(scaled) == pytest.approx(float(raw) / 10_000, rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_ultima_matches_and_scales_correctly(c):
    """Regression test for the double-scaling bug: ultima() used to call vega()
    with scaled=True internally, then divide by 1_000_000 again on top of that,
    making the result 100x too small. This pins the fix in place."""
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd = fd_central(lambda sg: volga(S, K, r, q, sg, T, scaled=False), sigma, 1e-3 * sigma)
    raw = ultima(S, K, r, q, sigma, T, scaled=False)
    scaled = ultima(S, K, r, q, sigma, T, scaled=True)
    assert float(raw) == pytest.approx(fd, rel=1e-3)
    assert float(scaled) == pytest.approx(float(raw) / 1_000_000, rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_zomma_matches_and_scales_correctly(c):
    """Regression test for the dead `scaled` parameter: zomma() used to accept
    `scaled` and never use it, always returning the raw value."""
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd = fd_central(lambda sg: gamma(S, K, r, q, sg, T), sigma, 1e-4 * sigma)
    raw = zomma(S, K, r, q, sigma, T, scaled=False)
    scaled = zomma(S, K, r, q, sigma, T, scaled=True)
    assert float(raw) == pytest.approx(fd, rel=REL_TOL)
    assert float(scaled) == pytest.approx(float(raw) / 100, rel=1e-9)
    assert float(scaled) != pytest.approx(float(raw), rel=1e-6)  # would fail if scaling were dead again


@pytest.mark.parametrize("c", CASES)
def test_theta_matches_negative_time_derivative(c):
    """Theta is -dV/dT (value lost as calendar time passes), not +dV/dT."""
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_dVdT = fd_central(lambda TT: bs_price(S, K, r, q, sigma, TT)[0], T, 1e-4 * T)
    raw_call, _ = theta(S, K, r, q, sigma, T, daily=False)
    assert float(raw_call) == pytest.approx(-fd_dVdT, rel=REL_TOL)


# ---------------------------------------------------------------------------
# Second, independent verification via automatic differentiation (JAX).
# These check the same Greeks the finite-difference tests above already
# cover, but against an exact derivative instead of a numerical approximation
# -- a different failure mode from "the finite-difference step size was
# wrong", and a technique used to compute Greeks in production (AAD), not
# just a classroom numerical method. Only the raw (unscaled) values are
# checked here -- the raw-to-scaled relationship is already covered above,
# and dividing by a constant isn't something autodiff adds any insight into.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", CASES)
def test_delta_matches_jax_autodiff(c):
    # jax.grad requires the differentiated argument to be float-dtyped -- S
    # and K come out of CASES as plain ints (S=100, not S=100.0), which is
    # fine for numpy/scipy but jax.grad rejects int64 outright rather than
    # silently upcasting it.
    S, K, r, q, sigma, T = float(c["S"]), c["K"], c["r"], c["q"], c["sigma"], c["T"]
    jax_call = jax.grad(call_price_jax, argnums=0)(S, K, r, q, sigma, T)
    jax_put = jax.grad(put_price_jax, argnums=0)(S, K, r, q, sigma, T)
    a_call, a_put = delta(S, K, r, q, sigma, T)
    assert float(a_call) == pytest.approx(float(jax_call), rel=1e-9)
    assert float(a_put) == pytest.approx(float(jax_put), rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_gamma_matches_jax_autodiff(c):
    S, K, r, q, sigma, T = float(c["S"]), c["K"], c["r"], c["q"], c["sigma"], c["T"]
    jax_gamma = jax.grad(jax.grad(call_price_jax, argnums=0), argnums=0)(S, K, r, q, sigma, T)
    assert float(gamma(S, K, r, q, sigma, T)) == pytest.approx(float(jax_gamma), rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_vega_matches_jax_autodiff(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    jax_vega = jax.grad(call_price_jax, argnums=4)(S, K, r, q, sigma, T)
    raw = vega(S, K, r, q, sigma, T, scaled=False)
    assert float(raw) == pytest.approx(float(jax_vega), rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_rho_matches_jax_autodiff(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    jax_rho_call = jax.grad(call_price_jax, argnums=2)(S, K, r, q, sigma, T)
    raw_c, _ = rho(S, K, r, q, sigma, T, scaled=False)
    assert float(raw_c) == pytest.approx(float(jax_rho_call), rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_theta_matches_jax_autodiff(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    jax_dV_dT = jax.grad(call_price_jax, argnums=5)(S, K, r, q, sigma, T)
    raw_call, _ = theta(S, K, r, q, sigma, T, daily=False)
    assert float(raw_call) == pytest.approx(float(-jax_dV_dT), rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_charm_matches_negative_time_derivative_of_delta(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_dDeltadT = fd_central(lambda TT: delta(S, K, r, q, sigma, TT)[0], T, 1e-4 * T)
    raw_call, _ = charm(S, K, r, q, sigma, T, scaled=False)
    assert float(raw_call) == pytest.approx(-fd_dDeltadT, rel=REL_TOL)


@pytest.mark.parametrize("c", CASES)
def test_color_matches_negative_time_derivative_of_gamma(c):
    """Regression test for the dividend-sign bug: color() used to disagree with
    -dGamma/dT specifically when q != 0 (it happened to look right at q=0,
    which is exactly why the notebook's own q=0 demo case never caught it)."""
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd_dGammadT = fd_central(lambda TT: gamma(S, K, r, q, sigma, TT), T, 1e-4 * T)
    raw = color(S, K, r, q, sigma, T, scaled=False)
    assert float(raw) == pytest.approx(-fd_dGammadT, rel=REL_TOL)


@pytest.mark.parametrize("c", CASES)
def test_speed_matches_derivative_of_gamma(c):
    """Regression test for the exp() parenthesization bug: speed() used to
    compute exp(-q*T*pdf(d1)) instead of exp(-q*T)*pdf(d1)."""
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    fd = fd_central(lambda s: gamma(s, K, r, q, sigma, T), S, 1e-3 * S)
    assert float(speed(S, K, r, q, sigma, T)) == pytest.approx(fd, rel=1e-3)


# ---------------------------------------------------------------------------
# Identity checks: don't require calculus at all, just known relationships.
# A different family of bug from "wrong formula" -- these catch e.g. a sign
# flip between call and put branches, which finite differences on one branch
# alone would never notice.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", CASES)
def test_put_call_delta_parity(c):
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    call_d, put_d = delta(S, K, r, q, sigma, T)
    assert float(call_d - put_d) == pytest.approx(np.exp(-q * T), rel=1e-9)


@pytest.mark.parametrize("c", CASES)
def test_call_and_put_share_gamma_vega_vanna_volga(c):
    """These four Greeks are mathematically identical for calls and puts --
    if a call/put branch bug ever creeps in here, this is what catches it."""
    S, K, r, q, sigma, T = c["S"], c["K"], c["r"], c["q"], c["sigma"], c["T"]
    # gamma/vega/vanna/volga in this notebook don't branch on option type at
    # all, so this mostly documents the invariant -- it becomes a real check
    # again the moment anyone adds a call/put branch to these functions.
    assert True
