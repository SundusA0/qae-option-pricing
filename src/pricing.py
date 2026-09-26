"""European call option pricing: classical baselines and quantum amplitude estimation."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class OptionSpec:
    """A European call: spot, strike, volatility, risk-free rate, maturity (years)."""
    S0: float = 100.0
    K: float = 105.0
    vol: float = 0.20
    r: float = 0.03
    T: float = 1.0

    @property
    def mu(self) -> float:
        """Mean of log(S_T) under the risk-neutral measure."""
        return np.log(self.S0) + (self.r - 0.5 * self.vol**2) * self.T

    @property
    def sigma(self) -> float:
        """Std dev of log(S_T)."""
        return self.vol * np.sqrt(self.T)

    def bounds(self, n_std: float = 3.0) -> tuple[float, float]:
        """Truncation window for the lognormal, as (low, high)."""
        mean = np.exp(self.mu + self.sigma**2 / 2)
        var = (np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2)
        sd = np.sqrt(var)
        return max(0.0, mean - n_std * sd), mean + n_std * sd


def black_scholes_call(spec: OptionSpec) -> float:
    """Closed-form discounted price. This is ground truth."""
    d1 = (np.log(spec.S0 / spec.K) + (spec.r + 0.5 * spec.vol**2) * spec.T) / (
        spec.vol * np.sqrt(spec.T)
    )
    d2 = d1 - spec.vol * np.sqrt(spec.T)
    return spec.S0 * norm.cdf(d1) - spec.K * np.exp(-spec.r * spec.T) * norm.cdf(d2)


def expected_payoff_analytic(spec: OptionSpec) -> float:
    """Undiscounted E[max(S_T - K, 0)]. QAE estimates this quantity, not the price."""
    return black_scholes_call(spec) * np.exp(spec.r * spec.T)


def expected_payoff_grid(spec: OptionSpec, num_qubits: int, n_std: float = 3.0) -> float:
    """
    Exact expected payoff on the 2**num_qubits discretised grid.

    The gap between this and expected_payoff_analytic is the encoding error:
    truncation to the window, renormalisation, and discretisation onto the
    grid, taken together. It enters before any estimation error does.
    """
    from qiskit_finance.circuit.library import LogNormalDistribution

    low, high = spec.bounds(n_std)
    dist = LogNormalDistribution(
        num_qubits, mu=spec.mu, sigma=spec.sigma**2, bounds=(low, high)
    )
    return float(np.sum(dist.probabilities * np.maximum(dist.values - spec.K, 0.0)))


def expected_payoff_mc(spec: OptionSpec, n_samples: int, seed: int = 0) -> tuple[float, float]:
    """Classical Monte Carlo baseline. Returns (estimate, standard error)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    sT = np.exp(spec.mu + spec.sigma * z)
    payoff = np.maximum(sT - spec.K, 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_samples))


def expected_payoff_qae(
    spec: OptionSpec,
    num_qubits: int,
    epsilon_target: float = 1e-3,
    rescaling_factor: float = 0.05,
    alpha: float = 0.05,
    n_std: float = 3.0,
    seed: int = 42,
) -> dict:
    """
    Estimate the expected payoff with Iterative Amplitude Estimation.

    rescaling_factor (c) controls the small-angle linearisation of the payoff.
    Large c keeps the circuit cheap but leaves a systematic bias that no amount
    of sampling removes; small c lowers that bias but shrinks the signal, so a
    tighter epsilon_target is needed to resolve it. That trade-off is the point.
    """
    from qiskit_algorithms import IterativeAmplitudeEstimation
    from qiskit.primitives import StatevectorSampler
    from qiskit_finance.applications.estimation import EuropeanCallPricing
    from qiskit_finance.circuit.library import LogNormalDistribution

    low, high = spec.bounds(n_std)
    dist = LogNormalDistribution(
        num_qubits, mu=spec.mu, sigma=spec.sigma**2, bounds=(low, high)
    )
    app = EuropeanCallPricing(
        num_state_qubits=num_qubits,
        strike_price=spec.K,
        rescaling_factor=rescaling_factor,
        bounds=(low, high),
        uncertainty_model=dist,
    )
    problem = app.to_estimation_problem()
    # A Generator, not an int: with an integer seed StatevectorSampler restarts
    # the random stream on every run() call, so IQAE's successive rounds would
    # reuse the same draws. A Generator advances between calls.
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target,
        alpha=alpha,
        sampler=StatevectorSampler(seed=np.random.default_rng(seed)),
    )
    result = iae.estimate(problem)
    lo, hi = app.interpret_confidence_interval(result) if hasattr(
        app, "interpret_confidence_interval"
    ) else (np.nan, np.nan)
    return {
        "estimate": float(app.interpret(result)),
        "oracle_queries": int(result.num_oracle_queries),
        "num_qubits": num_qubits,
        "epsilon_target": epsilon_target,
        "rescaling_factor": rescaling_factor,
        "ci_low": lo,
        "ci_high": hi,
        "circuit_qubits": problem.state_preparation.num_qubits,
    }
