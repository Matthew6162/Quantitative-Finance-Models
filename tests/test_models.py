"""
Correctness tests for src/models.py (used by notebooks/advanced_models/Heston_Model.ipynb,
Bates_Model.ipynb, and notebooks/vanilla_pricing/Event_Contract_Pricer.ipynb).

WHAT THIS SUITE CHECKS, AND WHY EACH KIND OF CHECK EXISTS
----------------------------------------------------------
A path simulator is harder to test than a closed-form formula, because there is
no analytic answer to compare a single path against. Asserting that a simulated
path "looks right" is not a test. So every check below is one of five kinds,
each of which has a definite right answer:

1. CROSS-MODEL IDENTITY. Bates with lam=0 is not approximately Heston, it is
   exactly Heston: the jump terms vanish and the remaining update equation is
   character-for-character the Heston one. Both functions draw their correlated
   shocks first and from the same generator, so under a shared seed the two
   price arrays must agree bit for bit. This is the strongest check here,
   because it ties two independently-called code paths to one number without
   either of them being the reference for the other. It is the same evidentiary
   pattern as checking the Greeks by finite differences against automatic
   differentiation.

2. BOUNDARY BEHAVIOUR. The CIR variance process is discretised by Euler with
   full truncation, meaning the variance itself is allowed to go negative but
   only max(v, 0) is ever fed forward. That distinction matters: a test that
   asserted v >= 0 would be asserting something the scheme deliberately does not
   promise, and would fail on correct code. What the scheme does promise is that
   a negative variance never reaches a square root. So the test drives the
   process into a deliberately violated Feller condition, confirms the
   truncation actually fires, and then confirms nothing became NaN.

3. DEGENERATE-PARAMETER REDUCTIONS. Setting a parameter to zero should collapse
   the model to something with a known exact answer. With xi = 0 and v0 = theta
   the variance path is constant at theta, exactly, with no floating-point
   slack. These are cheap and they fail loudly if an update equation is edited
   carelessly.

4. DISTRIBUTIONAL CHECKS WITH DERIVED TOLERANCES. Where a check is statistical,
   the tolerance is computed from the Monte Carlo standard error of the estimate
   rather than chosen to make the test pass. A hand-picked tolerance tells you
   nothing about whether the number is right; a four-standard-error band does.

5. CONTRACT CHECKS. Shapes, dtypes, and the rng argument behaving as documented.
   These are not interesting mathematics, but every notebook in the repository
   indexes these arrays as [time, path], so a transposed return value would be
   silently wrong everywhere rather than loudly wrong once.

One check is deliberately inverted: test_jumps_actually_fire_and_move_the_price
asserts that the jump stream does something no diffusion could. A jump
component that has been accidentally disabled passes every other test in this
file, which is close to how the Merton multi-jump variance bug survived as long
as it did. Its docstring records the weaker version of itself that a mutation
test defeated.
"""

import pathlib
import sys

import numpy as np
import pytest

# src/ is a sibling of tests/, not tests/'s parent -- add the repo root so
# `import src.models` resolves regardless of the directory pytest is run from.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.models import (  # noqa: E402  (import after sys.path fix, intentional)
    generate_correlated_normals,
    simulate_cir_variance,
    simulate_heston,
    simulate_bates,
)

SEED = 20260913

# A well-behaved parameter set: Feller condition satisfied (2*kappa*theta = 0.16
# against xi**2 = 0.16, so it holds with equality), negative spot/vol correlation
# as observed in equity markets, one year of daily steps.
BASE = dict(S0=100.0, v0=0.04, n=252, dt=1 / 252, kappa=2.0, theta=0.04,
            xi=0.4, rho=-0.7, mu=0.05)


# ---------------------------------------------------------------------------
# 1. Cross-model identity
# ---------------------------------------------------------------------------
def test_bates_without_jumps_is_exactly_heston():
    """With lam = 0 every jump term vanishes identically, not approximately.

    dN is Poisson(0), so it is all zeros; Y is then Normal(loc=0, scale=0), so
    it is all zeros; exp(0) - 1 is 0; and the compensator lam*k is 0 whatever k
    happens to be. What remains is the Heston update. Both functions draw their
    correlated shocks before touching the jump stream, so a shared seed gives
    both the same shocks and the two price arrays must agree to the last bit.

    Note that mu_j and sigma_j are deliberately non-zero. They cannot matter
    when lam = 0, and asserting that makes this a statement about the jump
    switch rather than about a doubly-degenerate special case.
    """
    price_bates = simulate_bates(
        S0=BASE["S0"], n=BASE["n"], dt=BASE["dt"], theta=BASE["theta"],
        kappa=BASE["kappa"], xi=BASE["xi"], rho=BASE["rho"], mu=BASE["mu"],
        lam=0.0, mu_j=-0.05, sigma_j=0.15, num_paths=500, rng=SEED,
    )
    # simulate_bates starts its variance at theta rather than taking a v0, so
    # Heston has to be given v0 = theta for the comparison to be like for like.
    price_heston, _ = simulate_heston(
        S0=BASE["S0"], v0=BASE["theta"], n=BASE["n"], dt=BASE["dt"],
        kappa=BASE["kappa"], theta=BASE["theta"], xi=BASE["xi"],
        rho=BASE["rho"], mu=BASE["mu"], num_paths=500, rng=SEED,
    )
    assert np.array_equal(price_bates, price_heston)


def test_jumps_actually_fire_and_move_the_price():
    """The inverse of the test above, and the harder half to get right.

    Comparing lam = 0 against lam > 0 and asserting the paths differ is not
    enough, because lam also appears in the compensator (mu - lam*k), so the
    two runs differ even when the Poisson draw has been disabled. That version
    of this test passed against a deliberately broken simulator, which is how
    this one came to be written.

    So the check is on the jumps themselves rather than on lam. With sigma_j = 0
    every jump is the same size, exp(mu_j), and a single jump step multiplies
    the price by about 1.35 where one day of 20% diffusion moves it by around a
    percent. Counting steps whose absolute log return exceeds 0.2 therefore
    counts jumps and nothing else: the largest diffusion-only step in this
    sample is 0.09.

    The expected count is lam*T*num_paths. Steps carrying two or more jumps are
    counted once, so the observed count sits slightly under that, which is why
    the band below is one-sided in effect rather than symmetric.
    """
    common = dict(S0=BASE["S0"], n=BASE["n"], dt=BASE["dt"], theta=BASE["theta"],
                  kappa=BASE["kappa"], xi=BASE["xi"], rho=BASE["rho"], mu=0.0,
                  mu_j=0.3, sigma_j=0.0, num_paths=500)

    def jump_sized_steps(prices):
        return int((np.abs(np.diff(np.log(prices), axis=0)) > 0.2).sum())

    quiet = simulate_bates(lam=0.0, rng=SEED, **common)
    assert jump_sized_steps(quiet) == 0, "diffusion alone must never produce a 20% daily move here"

    lam = 5.0
    jumpy = simulate_bates(lam=lam, rng=SEED, **common)
    expected = lam * (BASE["n"] * BASE["dt"]) * common["num_paths"]
    assert jump_sized_steps(jumpy) == pytest.approx(expected, rel=0.1)


def test_multi_jump_steps_scale_with_dN_not_dN_squared():
    """The regression test for the bug this repository has already had once.

    A step containing dN jumps carries the sum of dN independent jumps, so its
    log-jump is Normal(dN*mu_j, dN*sigma_j**2). Drawing one normal and
    multiplying it by dN instead gives variance dN**2*sigma_j**2, which is
    identical whenever dN is 0 or 1 and wrong otherwise. That is exactly why the
    Merton notebook carried the error for as long as it did: at lam = 1 and
    daily steps, P(dN >= 2) is about 8e-6.

    To make the difference visible rather than rare, this drives lam*dt up to 2,
    so two or more jumps in a step is the common case, and removes the diffusion
    entirely by setting theta = xi = 0 so the only variance left is the jumps'.
    With mu_j = 0 the step return then has variance lam*dt*sigma_j**2 if the
    scaling is right, and (lam*dt + (lam*dt)**2)*sigma_j**2 if it is not, a
    factor of three apart here.
    """
    lam, sigma_j, num_paths = 504.0, 0.02, 4000      # lam*dt = 2
    prices = simulate_bates(
        S0=BASE["S0"], n=BASE["n"], dt=BASE["dt"], theta=0.0, kappa=1.0, xi=0.0,
        rho=0.0, mu=0.0, lam=lam, mu_j=0.0, sigma_j=sigma_j,
        num_paths=num_paths, rng=SEED,
    )
    step_return = np.diff(prices, axis=0) / prices[:-1]
    correct = lam * BASE["dt"] * sigma_j ** 2
    assert step_return.var() == pytest.approx(correct, rel=0.05)


def test_the_jump_compensator_keeps_the_drift_unchanged():
    """What the (mu - lam*k) term is for.

    Adding jumps to a price process adds expected return unless it is paid for.
    The compensator subtracts exactly that much drift, so E[S_T] should be the
    same whether jumps are switched on or off. Asserting that is a sharper test
    than asserting any particular jump behaviour, because it pins the one term
    whose only job is to cancel another.

    Dropping the compensator entirely at these parameters inflates the terminal
    mean by a factor of about 1.21, against a four-standard-error band of
    roughly one percent, so this is not a marginal call.
    """
    num_paths = 20_000
    common = dict(S0=BASE["S0"], n=BASE["n"], dt=BASE["dt"], theta=BASE["theta"],
                  kappa=BASE["kappa"], xi=BASE["xi"], rho=BASE["rho"],
                  mu=BASE["mu"], mu_j=-0.05, sigma_j=0.15, num_paths=num_paths)
    expected = BASE["S0"] * (1 + BASE["mu"] * BASE["dt"]) ** BASE["n"]

    for lam in (0.0, 5.0):
        terminal = simulate_bates(lam=lam, rng=SEED, **common)[-1]
        standard_error = terminal.std(ddof=1) / np.sqrt(num_paths)
        assert terminal.mean() == pytest.approx(expected, abs=4 * standard_error), (
            f"terminal mean moved when lam = {lam}, which is what the compensator exists to prevent"
        )


# ---------------------------------------------------------------------------
# 2. Boundary behaviour: full truncation
# ---------------------------------------------------------------------------
# 2*kappa*theta = 0.08 against xi**2 = 1.0, so the Feller condition is violated
# by more than an order of magnitude and the variance is guaranteed to go
# negative. That is the point: these parameters exist to make the truncation do
# work, not to be realistic.
FELLER_VIOLATED = dict(kappa=1.0, theta=0.04, xi=1.0)


def test_truncation_actually_fires_under_a_violated_feller_condition():
    """Guards the guard. If the variance never went negative here, the rest of
    this section would be passing without exercising anything."""
    _, epsilon_v = generate_correlated_normals(252, 2000, rho=-0.7, rng=SEED)
    v = simulate_cir_variance(FELLER_VIOLATED["theta"], 252, 1 / 252,
                              FELLER_VIOLATED["kappa"], FELLER_VIOLATED["theta"],
                              FELLER_VIOLATED["xi"], epsilon_v)
    assert v.min() < 0
    assert 2 * FELLER_VIOLATED["kappa"] * FELLER_VIOLATED["theta"] < FELLER_VIOLATED["xi"] ** 2


def test_no_negative_variance_ever_reaches_a_square_root():
    """The promise full truncation actually makes.

    v is allowed to be negative; sqrt(max(v, 0)) is not allowed to be NaN. If
    the max() guard were dropped from either the variance step or the price
    step, the negative values found in the test above would propagate as NaN
    through every path from that step onward.
    """
    _, epsilon_v = generate_correlated_normals(252, 2000, rho=-0.7, rng=SEED)
    v = simulate_cir_variance(FELLER_VIOLATED["theta"], 252, 1 / 252,
                              FELLER_VIOLATED["kappa"], FELLER_VIOLATED["theta"],
                              FELLER_VIOLATED["xi"], epsilon_v)
    assert np.isfinite(v).all()

    S, v_full = simulate_heston(
        S0=100.0, v0=FELLER_VIOLATED["theta"], n=252, dt=1 / 252,
        kappa=FELLER_VIOLATED["kappa"], theta=FELLER_VIOLATED["theta"],
        xi=FELLER_VIOLATED["xi"], rho=-0.7, mu=0.0, num_paths=2000, rng=SEED,
    )
    assert np.isfinite(S).all()
    assert np.isfinite(v_full).all()
    assert (S > 0).all(), "an Euler price path that crosses zero makes every later step meaningless"


# ---------------------------------------------------------------------------
# 3. Degenerate-parameter reductions
# ---------------------------------------------------------------------------
def test_zero_vol_of_vol_pins_the_variance_at_theta_exactly():
    """With xi = 0 the noise term disappears and the drift is kappa*(theta - v).
    Starting at v0 = theta makes that drift identically zero, so the variance
    path is the constant theta with no floating-point drift at all.
    """
    v = simulate_cir_variance(BASE["theta"], BASE["n"], BASE["dt"],
                              BASE["kappa"], BASE["theta"], 0.0,
                              np.zeros((BASE["n"], 5)))
    assert np.array_equal(v, np.full_like(v, BASE["theta"]))


def test_mean_reversion_follows_the_exact_euler_solution():
    """With xi = 0 the variance step is a deterministic linear recursion, so it
    has a closed form: v_i = theta + (v0 - theta)*(1 - kappa*dt)**i. Starting
    away from theta is what makes this test bite, since a simulator that has
    stopped mean-reverting at all still sits exactly on theta if it was started
    there.
    """
    v0, kappa, theta = 0.09, 3.0, 0.04
    v = simulate_cir_variance(v0, BASE["n"], BASE["dt"], kappa, theta, 0.0,
                              np.zeros((BASE["n"], 3)))
    step = np.arange(BASE["n"] + 1)[:, None]
    closed_form = theta + (v0 - theta) * (1 - kappa * BASE["dt"]) ** step
    assert np.allclose(v, closed_form, rtol=1e-12, atol=0.0)
    # and it is genuinely reverting, not just sitting still
    assert abs(v[-1, 0] - theta) < abs(v0 - theta)


def test_zero_correlation_leaves_the_two_shock_arrays_independent():
    """rho = 0 should reduce the construction to two independent draws."""
    a, b = generate_correlated_normals(200_000, 1, rho=0.0, rng=SEED)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    # Standard error of a sample correlation near zero is 1/sqrt(N).
    assert abs(corr) < 4 / np.sqrt(200_000)


@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_perfect_correlation_is_exact_not_approximate(rho):
    """At rho = +/-1 the independent component is multiplied by sqrt(1 - 1) = 0,
    so the second array is the first one copied or negated, exactly."""
    a, b = generate_correlated_normals(1000, 10, rho=rho, rng=SEED)
    assert np.array_equal(b, rho * a)


# ---------------------------------------------------------------------------
# 4. Distributional checks, tolerances derived from the standard error
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [-0.9, -0.7, -0.35, 0.0, 0.35, 0.7, 0.9])
def test_correlated_normals_hit_their_target_correlation(rho):
    """The sample correlation of N pairs has standard error (1 - rho**2)/sqrt(N),
    so the band below is derived rather than tuned. At N = 200,000 it is a few
    thousandths, which is tight enough to catch a rho that has been squared,
    halved, or had its sign flipped."""
    n = 200_000
    a, b = generate_correlated_normals(n, 1, rho=rho, rng=SEED)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    standard_error = (1 - rho ** 2) / np.sqrt(n)
    assert corr == pytest.approx(rho, abs=4 * standard_error)


def test_both_shock_arrays_are_standard_normal():
    """rho controls the dependence between the two arrays, not their marginals.
    Both must stay mean 0, variance 1, or every volatility in the model is
    quietly mis-scaled."""
    n = 200_000
    for rho in (-0.7, 0.0, 0.7):
        a, b = generate_correlated_normals(n, 1, rho=rho, rng=SEED)
        for arr in (a, b):
            assert arr.mean() == pytest.approx(0.0, abs=4 / np.sqrt(n))
            assert arr.std() == pytest.approx(1.0, abs=4 / np.sqrt(2 * n))


@pytest.mark.parametrize("mu", [0.0, 0.08])
def test_heston_terminal_mean_matches_the_euler_drift(mu):
    """Under this discretisation E[S_{i+1}] = E[S_i](1 + mu*dt), because the
    diffusion term has conditional mean zero. So the exact expectation of the
    scheme, which is what the code should reproduce rather than the continuous
    time exp(mu*T), is S0*(1 + mu*dt)**n.

    The band is four standard errors of the sample mean, computed from the same
    sample, so it tightens automatically if num_paths is ever raised.
    """
    num_paths = 20_000
    S, _ = simulate_heston(
        S0=BASE["S0"], v0=BASE["v0"], n=BASE["n"], dt=BASE["dt"],
        kappa=BASE["kappa"], theta=BASE["theta"], xi=BASE["xi"],
        rho=BASE["rho"], mu=mu, num_paths=num_paths, rng=SEED,
    )
    terminal = S[-1]
    expected = BASE["S0"] * (1 + mu * BASE["dt"]) ** BASE["n"]
    standard_error = terminal.std(ddof=1) / np.sqrt(num_paths)
    assert terminal.mean() == pytest.approx(expected, abs=4 * standard_error)


# ---------------------------------------------------------------------------
# 5. Contract checks: the rng argument, shapes, dtypes
# ---------------------------------------------------------------------------
def test_default_rng_passes_an_existing_generator_through_unchanged():
    """The whole seeding design rests on this one NumPy behaviour: a single rng
    parameter can mean "fresh stream", "this seed", or "this existing
    generator", because default_rng returns a Generator argument by identity.
    If that ever stopped holding, common random numbers across two simulators
    would silently become independent streams."""
    generator = np.random.default_rng(SEED)
    assert np.random.default_rng(generator) is generator


def test_the_same_seed_reproduces_the_same_paths():
    first, _ = simulate_heston(num_paths=50, rng=SEED, **BASE)
    second, _ = simulate_heston(num_paths=50, rng=SEED, **BASE)
    assert np.array_equal(first, second)


def test_a_different_seed_gives_different_paths():
    first, _ = simulate_heston(num_paths=50, rng=SEED, **BASE)
    second, _ = simulate_heston(num_paths=50, rng=SEED + 1, **BASE)
    assert not np.array_equal(first, second)


def test_a_shared_generator_advances_between_calls():
    """Passing a Generator rather than a seed is what drives several simulations
    from one stream. Consecutive calls must therefore consume the stream and
    return different draws, not restart it."""
    generator = np.random.default_rng(SEED)
    first, _ = simulate_heston(num_paths=50, rng=generator, **BASE)
    second, _ = simulate_heston(num_paths=50, rng=generator, **BASE)
    assert not np.array_equal(first, second)


def test_omitting_the_rng_is_non_deterministic():
    """The default has to stay None-like, so exploratory use does not silently
    become a single fixed sample."""
    first, _ = simulate_heston(num_paths=50, **BASE)
    second, _ = simulate_heston(num_paths=50, **BASE)
    assert not np.array_equal(first, second)


def test_the_draw_order_has_not_changed():
    """A characterisation test, pinning the one promise the module docstring
    makes that nothing else here can check.

    src/models.py states that the random draw order matches the pre-refactor
    inline notebook code exactly, so the extraction was a pure move. Nothing
    distributional can verify that: swapping which shock drives the price and
    which drives the variance, for instance, leaves the joint law completely
    unchanged and every other test in this file passing. Only the realised
    values differ.

    That matters because the notebooks store their figures. A change in draw
    order would silently redraw every plot in the repository while all the
    statistics stayed correct, and the seeded-run empty-diff property would
    quietly stop meaning anything.

    If this fails after a deliberate change to the order, that is not a bug:
    update the constants below and re-run every notebook, because their stored
    output has just changed too.
    """
    S, v = simulate_heston(S0=100.0, v0=0.04, n=5, dt=1 / 252, kappa=2.0,
                           theta=0.04, xi=0.4, rho=-0.7, mu=0.05,
                           num_paths=2, rng=SEED)
    expected_S = [100.0, 97.41081468805784, 96.68490730383091,
                  95.58493715570229, 95.049917148954, 93.99586844959668]
    expected_v = [0.04, 0.0439703548587055, 0.04466483414236957,
                  0.05021923298120346, 0.04941743082981374, 0.05159106188936777]
    # rel rather than exact, so a one-ULP difference in another platform's libm
    # does not fail the build; a reordered draw is wrong in the second digit.
    assert S[:, 0] == pytest.approx(expected_S, rel=1e-12)
    assert v[:, 0] == pytest.approx(expected_v, rel=1e-12)


def test_shapes_and_dtypes_are_time_by_path_float64():
    """Every notebook indexes these as [time, path], so a transposed return
    value would be wrong everywhere at once and obvious nowhere."""
    num_paths = 25
    S, v = simulate_heston(num_paths=num_paths, rng=SEED, **BASE)
    assert S.shape == (BASE["n"] + 1, num_paths)
    assert v.shape == (BASE["n"] + 1, num_paths)
    assert S.dtype == np.float64 and v.dtype == np.float64

    price = simulate_bates(
        S0=BASE["S0"], n=BASE["n"], dt=BASE["dt"], theta=BASE["theta"],
        kappa=BASE["kappa"], xi=BASE["xi"], rho=BASE["rho"], mu=BASE["mu"],
        lam=1.0, mu_j=-0.05, sigma_j=0.15, num_paths=num_paths, rng=SEED,
    )
    assert price.shape == (BASE["n"] + 1, num_paths)
    assert price.dtype == np.float64

    # The shock arrays are one step shorter: n increments produce n + 1 levels.
    epsilon_S, epsilon_v = generate_correlated_normals(BASE["n"], num_paths, BASE["rho"], rng=SEED)
    assert epsilon_S.shape == (BASE["n"], num_paths)
    assert epsilon_v.shape == (BASE["n"], num_paths)


def test_every_path_starts_at_its_initial_condition():
    num_paths = 25
    S, v = simulate_heston(num_paths=num_paths, rng=SEED, **BASE)
    assert np.array_equal(S[0], np.full(num_paths, BASE["S0"]))
    assert np.array_equal(v[0], np.full(num_paths, BASE["v0"]))
