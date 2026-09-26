"""
Correctness tests for the pricing routes and the error-budget claims.

These are not coverage tests. Each one pins down a specific claim the README
makes, so that a change which quietly breaks the argument fails here first.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pricing import (
    OptionSpec,
    black_scholes_call,
    expected_payoff_analytic,
    expected_payoff_grid,
    expected_payoff_mc,
    expected_payoff_qae,
)

SPEC = OptionSpec()


def test_black_scholes_matches_published_value():
    """
    S0=100, K=105, sigma=0.20, r=0.03, T=1 prices at 7.1281.

    Verified independently against the closed form; this is the anchor every
    other number in the repository is measured against.
    """
    assert black_scholes_call(SPEC) == pytest.approx(7.128065, abs=1e-6)


def test_put_call_parity():
    """C - P = S0 - K*exp(-rT). Catches sign and discounting errors."""
    from scipy.stats import norm

    d1 = (np.log(SPEC.S0 / SPEC.K) + (SPEC.r + 0.5 * SPEC.vol**2) * SPEC.T) / (
        SPEC.vol * np.sqrt(SPEC.T))
    d2 = d1 - SPEC.vol * np.sqrt(SPEC.T)
    put = SPEC.K * np.exp(-SPEC.r * SPEC.T) * norm.cdf(-d2) - SPEC.S0 * norm.cdf(-d1)
    parity = SPEC.S0 - SPEC.K * np.exp(-SPEC.r * SPEC.T)
    assert black_scholes_call(SPEC) - put == pytest.approx(parity, abs=1e-10)


def test_discounting_relation():
    """QAE estimates the undiscounted payoff; the price is that times exp(-rT)."""
    assert expected_payoff_analytic(SPEC) * np.exp(-SPEC.r * SPEC.T) == pytest.approx(
        black_scholes_call(SPEC), abs=1e-10)


def test_monte_carlo_converges_to_analytic():
    """MC must land within ~4 standard errors of the analytic payoff."""
    est, se = expected_payoff_mc(SPEC, 2_000_000, seed=1)
    assert abs(est - expected_payoff_analytic(SPEC)) < 4 * se


def test_monte_carlo_standard_error_shrinks_as_sqrt_n():
    """Doubling samples 16x should roughly quarter the standard error."""
    _, se_small = expected_payoff_mc(SPEC, 50_000, seed=2)
    _, se_large = expected_payoff_mc(SPEC, 800_000, seed=2)
    assert se_small / se_large == pytest.approx(4.0, rel=0.25)


@pytest.mark.parametrize("nq", [3, 4, 5])
def test_grid_probabilities_are_normalised(nq):
    """The discretised lognormal must remain a probability distribution."""
    from qiskit_finance.circuit.library import LogNormalDistribution

    low, high = SPEC.bounds(5.0)
    dist = LogNormalDistribution(nq, mu=SPEC.mu, sigma=SPEC.sigma**2,
                                 bounds=(low, high))
    assert float(np.sum(dist.probabilities)) == pytest.approx(1.0, abs=1e-9)


def test_wider_window_needs_more_qubits():
    """
    The coupling claim in the README.

    At a fixed 3 uncertainty qubits, widening the truncation window makes the
    encoding error worse, because the same 8 grid points are spread thinner.
    Accuracy requires moving both together, not either alone.
    """
    narrow = expected_payoff_grid(SPEC, 3, n_std=4.0)
    wide = expected_payoff_grid(SPEC, 3, n_std=7.0)
    exact = expected_payoff_analytic(SPEC)
    assert abs(wide - exact) > abs(narrow - exact)


def test_narrow_window_does_not_converge_with_qubits():
    """
    At n_std = 3 the error plateaus: adding qubits cannot recover truncated
    tail mass. Asserts the gap stays large even at 7 qubits.
    """
    exact = expected_payoff_analytic(SPEC)
    err_7 = abs(expected_payoff_grid(SPEC, 7, n_std=3.0) - exact)
    assert err_7 > 0.3, "n_std=3 should leave a large residual regardless of nq"


def test_matched_window_and_qubits_converges():
    """Moved together, the encoding error does go to zero."""
    exact = expected_payoff_analytic(SPEC)
    assert abs(expected_payoff_grid(SPEC, 7, n_std=6.0) - exact) < 0.01


N_STD = 5.0  # must match between grid and QAE, or they estimate different things


@pytest.mark.slow
def test_grid_and_qae_use_the_same_window():
    """
    Guards the mistake this suite made on first writing: comparing a grid at
    one truncation width against QAE at another. The two then describe
    different quantities and any agreement or disagreement is meaningless.
    """
    narrow = expected_payoff_qae(SPEC, 3, epsilon_target=1e-2,
                                 rescaling_factor=0.05, n_std=3.0, seed=1000)
    wide = expected_payoff_qae(SPEC, 3, epsilon_target=1e-2,
                               rescaling_factor=0.05, n_std=7.0, seed=1000)
    assert abs(narrow["estimate"] - wide["estimate"]) > 0.1, (
        "changing n_std must change the estimated quantity")


@pytest.mark.slow
def test_smaller_rescaling_factor_reduces_bias():
    """
    The linearisation claim: at fixed grid and tight epsilon, reducing c
    reduces the systematic offset from the grid-exact value.

    Averaged over seeds, because at small c the per-seed spread exceeds the
    bias being measured - the repository's own warning about single runs.
    """
    grid = expected_payoff_grid(SPEC, 4, n_std=N_STD)
    seeds = range(1000, 1005)

    def bias(c):
        ests = [expected_payoff_qae(SPEC, 4, epsilon_target=1e-3,
                                    rescaling_factor=c, n_std=N_STD,
                                    seed=s)["estimate"] for s in seeds]
        return abs(float(np.mean(ests)) - grid)

    assert bias(0.05) < bias(0.25)


@pytest.mark.slow
def test_qae_agrees_with_grid_at_tight_epsilon():
    """
    QAE recovers the discretised expectation it is estimating.

    Averaged over five seeds: at c = 0.05 the per-seed spread (~0.22) is much
    larger than the bias (~0.01), so a single-seed tolerance would have to be
    loose enough to be meaningless. The mean over five seeds has a standard
    error near 0.1, and 0.3 is a three-sigma band around it.
    """
    grid = expected_payoff_grid(SPEC, 4, n_std=N_STD)
    ests = [expected_payoff_qae(SPEC, 4, epsilon_target=1e-3,
                                rescaling_factor=0.05, n_std=N_STD,
                                seed=s)["estimate"] for s in range(1000, 1005)]
    assert float(np.mean(ests)) == pytest.approx(grid, abs=0.3)


@pytest.mark.slow
def test_tighter_epsilon_costs_more_queries():
    """Oracle queries must grow as the target precision tightens."""
    loose = expected_payoff_qae(SPEC, 4, epsilon_target=1e-2, seed=1000)
    tight = expected_payoff_qae(SPEC, 4, epsilon_target=1e-3, seed=1000)
    assert tight["oracle_queries"] > 5 * loose["oracle_queries"]
