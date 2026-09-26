# Quantum amplitude estimation for European option pricing: a resource and error budget

Quantum amplitude estimation is the standard proposal for accelerating derivative
pricing, on the strength of a quadratic speedup over classical Monte Carlo. This
repository asks what that speedup costs in practice on the simplest possible
instrument — a vanilla European call — by measuring each error source separately
and converting the result into a hardware requirement.

**What this is and is not.** The pricing construction is not novel. Qiskit Finance
already provides `EuropeanCallPricing`, the lognormal loading, the linearised
payoff with its rescaling factor, and iterative amplitude estimation; the
[official tutorial](https://qiskit-community.github.io/qiskit-finance/tutorials/03_european_call_option_pricing.html)
is the starting point. The contribution here is the benchmarking around it:
separating the error sources, measuring how each scales, and turning the result
into a statement about circuits and devices.

**Summary of findings.** Of the four error terms between a Black–Scholes price and
what amplitude estimation returns, only one responds to spending more oracle
queries. Two of the other three are coupled, so tuning either alone makes the
answer worse. The third, the payoff linearisation, forces a trade-off that over
the configurations tested degrades the scaling from the ideal `N ~ ε⁻¹` to an
empirical `N ~ ε⁻¹·³`. Converting the IQAE schedule into circuits, the deepest
coherent computation at the tightest precision tested is 2.1 × 10⁶ two-qubit
gates, implying a tolerable error rate near 5 × 10⁻⁷ — roughly 4,000× better than
current superconducting devices. Under the depolarising noise model studied here,
at representative present-day error rates the optimal number of Grover iterations
is zero: additional amplification destroys more signal than it creates.

The option priced here has a closed-form solution that a laptop evaluates exactly.

---

## The four error sources

| # | Source | Controlled by | Shrinks with more queries? |
|---|---|---|---|
| 1 | Truncation of the lognormal | window width `n_std` | no |
| 2 | Discretisation onto `2^nq` grid points | qubit count `nq` | no |
| 3 | Small-angle linearisation of the payoff | rescaling factor `c` | no |
| 4 | Finite sampling in amplitude estimation | target precision `ε` | **yes** |

Sources 1–3 are classical modelling decisions. No improvement in quantum hardware
touches any of them. That the rescaling factor trades approximation against
estimation precision is documented behaviour in Qiskit Finance; what is measured
here is the size of that trade-off and what it does to the scaling.

### Truncation and discretisation are coupled

Encoding error `|grid exact − analytic|`, before any estimation:

| `n_std` \ `nq` | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| **3** | 0.0257 | 0.3020 | 0.4044 | 0.4078 | 0.4251 |
| **4** | 0.4548 | 0.0091 | 0.0878 | 0.0834 | 0.0915 |
| **5** | 0.7651 | 0.1317 | 0.0041 | 0.0258 | 0.0145 |
| **6** | 0.2057 | 0.1736 | 0.0134 | 0.0206 | 0.0002 |
| **7** | 1.2994 | 0.1223 | 0.0563 | 0.0107 | 0.0022 |

Accuracy lives on a diagonal. Widening the window without adding qubits spreads
the same grid points thinner; adding qubits inside a narrow window resolves a
distribution that has already been clipped. At `n_std = 3` the error plateaus near
0.42, about 5.7% of the payoff, and does not improve with qubit count over the
range tested. The reason is specific to call options: `max(S_T − K, 0)` grows
linearly in the upper tail, so truncated tail mass carries real value.

The 0.0257 at `n_std = 3, nq = 3` is accidental cancellation — a coarse grid
overshooting into a truncation deficit — not accuracy. A single row read in
isolation would suggest three qubits suffice.

### Bias and spread pull in opposite directions

Each configuration is repeated over five seeds, so a systematic offset can be
distinguished from a lucky draw. At `ε = 10⁻³`:

| `c` | bias | spread (sd) | sd × c |
|---|---|---|---|
| 0.25 | 0.9559 | 0.0253 | 0.0063 |
| 0.10 | 0.1818 | 0.0960 | 0.0096 |
| 0.05 | 0.0412 | 0.2265 | 0.0113 |

Over these three values the bias fits `c¹·⁹⁵`, consistent with a second-order
linearisation error, and the near-constant `sd × c` column indicates `sd ~ ε/c`,
as expected when the amplitude signal is itself proportional to `c`. Shrinking `c`
to suppress the bias amplifies the sampling error in the same motion.

Single-seed runs are unreliable here. At `c = 0.05, ε = 10⁻²` the spread across
seeds is 2.48, so one draw carries little information about the bias — yet it will
report a definite-looking error.

---

## What the trade-off costs

Minimising `c² + ε/c` gives `c ~ ε^(1/3)`, hence total error `~ ε^(2/3)`. Since
`N ~ 1/ε`, that predicts `N ~ error^(−3/2)`.

Efficient frontier over all 19 configurations (total error = `√(bias² + sd²)`):

| oracle queries | total error | configuration | classical MC samples | ratio |
|---|---|---|---|---|
| 3,277 | 2.8367 | nq=5, c=0.05, ε=10⁻² | 21 | 158× |
| 5,325 | 1.2519 | nq=3, c=0.25, ε=10⁻² | 106 | 50× |
| 54,682 | 0.2158 | nq=5, c=0.05, ε=10⁻³ | 3,584 | 15.3× |
| 84,173 | 0.1565 | nq=5, c=0.10, ε=10⁻³ | 6,812 | 12.4× |
| 912,589 | 0.0415 | nq=4, c=0.05, ε=10⁻⁴ | 97,063 | 9.4× |

The sampled frontier gives an empirical exponent of about **−1.32**, against −1.5
from the bias–variance model above. Ideal amplitude estimation would be −1.0 and
classical Monte Carlo is −2.0. This is an observation over seven frontier points
on one option, not a proven law, but it sits where the trade-off predicts.

Amplitude estimation is behind at every budget reachable in simulation — and an
oracle query is a nine-qubit circuit with hundreds of gates, while a Monte Carlo
sample is one exponential and one `max()`.

Extrapolating the fit, the crossover falls near **1.55 × 10⁻³ absolute error**
(2.1 basis points of the expected payoff), where both methods need roughly
7 × 10⁷ operations. This extrapolates well outside the measured region and should
be read as indicative.

---

## Converting queries into circuits

Compiled to a superconducting native basis (`cz, rz, sx, x`) at optimisation
level 3:

| `nq` | circuit qubits | 2Q gates (all-to-all) | 2Q gates (heavy-hex) | 2Q depth | routing overhead |
|---|---|---|---|---|---|
| 3 | 7 | 302 | 515 | 393 | 1.7× |
| 4 | 9 | 524 | 1,008 | 752 | 1.9× |
| 5 | 11 | 834 | 1,804 | 1,297 | 2.2× |

Routed two-qubit gates scale as (circuit width)^2.8, and the routing overhead
itself grows with width. A payoff needing 10–20 uncertainty qubits rather than 5
would cost roughly 10× more per query.

### Two costs, easily conflated

IQAE is adaptive: it raises the Grover power `k` until the confidence interval is
tight enough. Reading the schedule out of the algorithm at `nq = 5`
(`A` = 324, `Q` = 1,804 routed two-qubit gates):

| `ε` | rounds | max `k` | deepest circuit (2Q gates) | total queries |
|---|---|---|---|---|
| 10⁻² | 3 | 4 | 7,540 | 4,096 |
| 10⁻³ | 4 | 70 | 126,604 | 75,776 |
| 10⁻⁴ | 5 | 1,168 | 2,107,396 | 1,270,784 |

with `k_max ~ ε^−1.23` over this range.

Two different quantities follow, and they answer different questions:

- **Total gate executions** — every gate run across every shot of every round,
  1.24 × 10¹¹ at the extrapolated crossover. This sets wall-clock time and
  throughput. It is **not** a fidelity requirement: the shots are independent, so
  a corrupted shot adds variance to the estimate rather than invalidating the run.
- **Deepest single circuit** — `A` followed by `Q^k` at the largest `k` the
  schedule reaches. Errors accumulate coherently *within* one circuit, so this is
  what bounds the tolerable per-gate error rate.

> **Correction.** An earlier version of this analysis applied a zero-error budget
> to the *total* gate count and reported a requirement of `p ≪ 8 × 10⁻¹²`. That
> conflates the two costs above and overstates the requirement by several orders
> of magnitude. The figures below use the deepest circuit, measured from the IQAE
> schedule rather than assumed.

| | deepest circuit | tolerable `p` |
|---|---|---|
| tightest precision actually run (ε=10⁻⁴) | 2.1 × 10⁶ | ≪ 4.8 × 10⁻⁷ |
| extrapolated to the crossover | 9.3 × 10⁸ | ≪ 1.1 × 10⁻⁹ |

| device generation | 2Q error rate | vs deepest run |
|---|---|---|
| current superconducting (~2025) | 2 × 10⁻³ | 4,215× too high |
| optimistic near-term | 1 × 10⁻⁴ | 211× too high |
| early fault-tolerant logical | 1 × 10⁻⁸ | 47× headroom |

So the tightest configuration simulated here would be within reach of an early
fault-tolerant logical qubit, though still ~9× short of what the extrapolated
crossover would demand. The physical-qubit overhead error correction requires is
a separate cost not accounted for here.

---

## Noise and the useful depth limit

Each Grover iteration amplifies the signal by roughly `(2k+1)` and multiplies the
fidelity by `exp(−p(A + kQ))`. Amplification grows linearly in `k`; fidelity
decays exponentially.

### The signal collapse, measured

`P(good state)` at `nq = 3`, 20,000 shots, depolarising noise at the stated
two-qubit rate:

| `k` | 2Q gates | noiseless | p=10⁻⁵ | p=10⁻⁴ | p=10⁻³ |
|---|---|---|---|---|---|
| 0 | 125 | 0.4677 | 0.4677 | 0.4684 | 0.4679 |
| 1 | 647 | 0.6000 | 0.5995 | 0.5954 | 0.5560 |
| 2 | 1,180 | 0.3357 | 0.3368 | 0.3547 | 0.4459 |
| 3 | 1,717 | 0.7247 | 0.7219 | 0.6908 | 0.5496 |
| 4 | 2,290 | 0.2188 | 0.2254 | 0.2750 | 0.4759 |

Noiseless, the value oscillates as `sin²((2k+1)θ)` — that oscillation is the
amplification. Under noise it flattens toward 0.5, which carries no information.

### The decay follows `exp(−p · N₂Q)`

Surviving contrast, `(measured − 0.5) / (noiseless − 0.5)`:

| rate | `k` | 2Q gates | measured | predicted |
|---|---|---|---|---|
| 10⁻⁵ | 4 | 2,290 | 0.9765 | 0.9774 |
| 10⁻⁴ | 1 | 647 | 0.9540 | 0.9373 |
| 10⁻⁴ | 4 | 2,290 | 0.8003 | 0.7953 |
| 10⁻³ | 1 | 647 | 0.5595 | 0.5236 |
| 10⁻³ | 2 | 1,180 | 0.3294 | 0.3073 |
| 10⁻³ | 4 | 2,290 | 0.0857 | 0.1013 |

Mean absolute deviation **0.0133** over twelve points spanning contrast from 0.99
to 0.086.

### Where amplification stops paying

Maximising `(2k+1) · exp(−p(A + kQ))` gives `k* ≈ 1/(pQ)`. A value below 1 means
amplification destroys more signal than it creates. At `nq = 5`:

| 2Q error rate | optimal `k` | max gain | |
|---|---|---|---|
| 2 × 10⁻³ | **0** | 0.52 | amplification is net-negative |
| 1 × 10⁻⁴ | 5 | 4.3 | 4× over plain sampling |
| 1 × 10⁻⁶ | 554 | 408 | 408× over plain sampling |
| 1 × 10⁻⁸ | 55,432 | 40,785 | 40,785× over plain sampling |

Under this noise model and circuit construction, at representative current
two-qubit error rates additional Grover amplification is not beneficial: the
optimum is `k = 0`, at which point amplitude estimation reduces to ordinary
sampling. This is a statement about the configuration studied here, not a general
impossibility result.

Note that `k* ≈ 1/(pQ)` and the coherence bound `p ≪ 1/(A + k_max·Q)` are close to
algebraic restatements of one another. They are reported separately because the
two scripts reach the limit by different routes — one from the IQAE schedule, one
from a fidelity-weighted optimum — and agreeing is a useful check that neither is
mis-instrumented.

The ceiling depending on the product `pQ` also means halving the oracle cost is
worth as much as halving the error rate, which points at circuit synthesis rather
than hardware alone.

---

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

pytest tests/ -q                        # 15 tests, ~35 s
python scripts/01_validate.py --quick   # ~20 s
python scripts/01_validate.py           # ~6 min, 19 configs x 5 seeds
python scripts/02_analyse.py            # scaling fits and crossover
python scripts/03_resources.py          # circuits, schedule depth, error budget
python scripts/04_noise.py              # noise threshold, ~4 min
```

`src/pricing.py` provides four independent routes to the same quantity —
Black–Scholes closed form, exact expectation on the discretised grid, classical
Monte Carlo, and amplitude estimation — so a discrepancy can be attributed to a
specific source rather than absorbed into one number.

The tests pin the claims rather than the implementation: the Black–Scholes anchor,
put–call parity, Monte Carlo convergence and `√N` error scaling, grid
normalisation, the truncation/qubit coupling, and that reducing `c` reduces bias
when averaged over seeds.

Committed results under `results/` back every number quoted above.

**Environment.** Python 3.12, `qiskit==2.5.2`, `qiskit-aer==0.17.2`,
`qiskit-algorithms==0.4.0`, `qiskit-finance==0.4.1`. `requirements.txt` lists
direct dependencies; `requirements-lock.txt` is the full resolved environment. Run
on macOS and Linux with identical output to six decimal places, including oracle
query counts.

**Compatibility note.** The Qiskit Finance tutorials use
`qiskit_aer.primitives.Sampler`, which fails inside `IterativeAmplitudeEstimation`
on this version combination with an uninformative "job was not completed
successfully"; `qiskit.primitives.SamplerV2` fails the same way.
`qiskit.primitives.StatevectorSampler` works.

---

## Limitations

- One option (S₀=100, K=105, σ=0.20, r=0.03, T=1) at up to 5 uncertainty qubits.
  Nothing here establishes behaviour for other parameter regimes.
- The crossover extrapolates a 7-point fit far outside the measured range. It is
  indicative only, and the error-rate figures derived from it inherit that.
- The scaling exponent is an empirical fit over a coarse sweep, not a
  statistically characterised measurement with uncertainties.
- The coherence bound assumes errors accumulate independently and that one
  expected error spoils a circuit. It ignores error mitigation, algorithmic
  tolerance to modest infidelity, and, in a fault-tolerant setting, logical
  failure rates and decoding.
- Depolarising noise only, applied uniformly, with no measurement error,
  crosstalk, leakage or idle decoherence. Real devices would be worse.
- The noise measurement reaches k = 4 at 3 uncertainty qubits; larger `k` and
  wider circuits are extrapolated through the fitted fidelity model.
- Iterative amplitude estimation only. Maximum-likelihood and canonical variants
  may sit differently on the frontier.
- The argument applies to payoffs encoded through a linearised amplitude function.
  Path-dependent payoffs, where quantum methods are more plausibly interesting,
  are not covered.

## References

- [Qiskit Finance: European call option pricing tutorial](https://qiskit-community.github.io/qiskit-finance/tutorials/03_european_call_option_pricing.html) — the construction benchmarked here
- Woerner & Egger, *Quantum risk analysis*, npj Quantum Information 5, 15 (2019)
- Stamatopoulos et al., *Option pricing using quantum computers*, Quantum 4, 291 (2020)
- Grinko et al., *Iterative quantum amplitude estimation*, npj Quantum Information 7, 52 (2021)
