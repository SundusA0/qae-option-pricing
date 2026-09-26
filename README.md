# Quantum amplitude estimation for European option pricing: a resource and error budget

Quantum amplitude estimation is the standard proposal for accelerating derivative
pricing, on the strength of a quadratic speedup over classical Monte Carlo. This
repository asks what that speedup actually costs on the simplest possible
instrument — a vanilla European call — by measuring every error source separately
and converting the result into a hardware requirement.

**The headline.** Of the four error terms between a Black–Scholes price and what
amplitude estimation returns, only one responds to spending more oracle queries.
Two of the remaining three are coupled, so tuning either alone makes the answer
worse. And the third, the linearisation of the payoff function, forces a trade-off
that degrades the scaling from the advertised `N ~ ε⁻¹` to a measured `N ~ ε⁻¹·³`,
eating roughly half the quadratic advantage before a single gate error is
considered. Extrapolating to the crossover with classical Monte Carlo puts the
computation at **1.2 × 10¹¹ two-qubit gates**, requiring a two-qubit error rate
below **8 × 10⁻¹²** — about 2.5 × 10⁸ times better than current superconducting
hardware, and still three orders of magnitude beyond an early fault-tolerant
logical qubit.

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
touches any of them.

### Truncation and discretisation are coupled

Encoding error `|grid exact − analytic|`, before any estimation:

| `n_std` \ `nq` | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|
| **3** | 0.0257 | 0.3020 | 0.4044 | 0.4078 | 0.4251 |
| **4** | 0.4548 | 0.0091 | 0.0878 | 0.0834 | 0.0915 |
| **5** | 0.7651 | 0.1317 | 0.0041 | 0.0258 | 0.0145 |
| **6** | 0.2057 | 0.1736 | 0.0134 | 0.0206 | 0.0002 |
| **7** | 1.2994 | 0.1223 | 0.0563 | 0.0107 | 0.0022 |

Accuracy lives on a diagonal. Widening the window without adding qubits spreads the
same grid points thinner; adding qubits inside a narrow window resolves a
distribution that has already been clipped. At the Qiskit Finance tutorial default
of `n_std = 3`, **no qubit count converges** — the error plateaus near 0.42, about
5.7% of the payoff. The reason is specific to call options: `max(S_T − K, 0)` grows
linearly in the upper tail, so truncated tail mass carries real value.

Note the 0.0257 at `n_std = 3, nq = 3`. That is accidental cancellation — a coarse
grid overshooting into a truncation deficit — not accuracy. Reading a single row
would suggest three qubits suffice.

### Bias and spread pull in opposite directions

Each configuration is repeated over five seeds, so a systematic offset can be told
apart from a lucky draw. Measured at `ε = 10⁻³`:

| `c` | bias | spread (sd) | sd × c |
|---|---|---|---|
| 0.25 | 0.9559 | 0.0253 | 0.0063 |
| 0.10 | 0.1818 | 0.0960 | 0.0096 |
| 0.05 | 0.0412 | 0.2265 | 0.0113 |

Fitting gives **bias ~ c¹·⁹⁵**, consistent with the second-order linearisation
error, and **sd ~ ε/c**, since the amplitude signal is itself proportional to `c`.
Shrinking `c` to suppress the bias amplifies the sampling error in the same motion.

Single-seed runs are actively misleading here. At `c = 0.05, ε = 10⁻²` the spread
across seeds is 2.48 — one draw from that distribution carries no information, yet
it will happily print a small error.

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

**Measured: `N ~ error^−1.32`.** Ideal amplitude estimation is `−1.0`; classical
Monte Carlo is `−2.0`. The measured exponent sits close to the `−1.5` the
trade-off predicts.

Amplitude estimation is behind at every budget reachable in simulation — and an
oracle query is a nine-qubit circuit with hundreds of gates, while a Monte Carlo
sample is one exponential and one `max()`.

Extrapolating the fit, the crossover falls at **1.55 × 10⁻³ absolute error**
(2.1 basis points of the expected payoff), where both methods need roughly
**7 × 10⁷ operations**.

---

## Converting queries into gates

Compiled to a superconducting native basis (`cz, rz, sx, x`) at optimisation
level 3:

| `nq` | circuit qubits | 2Q gates (all-to-all) | 2Q gates (heavy-hex) | 2Q depth | routing overhead |
|---|---|---|---|---|---|
| 3 | 7 | 302 | 515 | 393 | 1.7× |
| 4 | 9 | 524 | 1,008 | 752 | 1.9× |
| 5 | 11 | 834 | 1,804 | 1,297 | 2.2× |

Routed two-qubit gates scale as (circuit width)^2.8, and the routing overhead
itself grows with width. A payoff needing 10–20 uncertainty qubits rather than 5
costs roughly 10× more per query, so the figures below are a lower bound.

At the crossover query count:

```
Oracle queries                    :     6.9 × 10⁷
Routed 2Q gates per query (nq=5)  :         1,804
2Q gates in the computation       :     1.2 × 10¹¹
Required 2Q error rate            : p ≪ 8.0 × 10⁻¹²
```

| device generation | 2Q error rate | shortfall |
|---|---|---|
| current superconducting | 2 × 10⁻³ | 2.5 × 10⁸ × |
| optimistic near-term | 1 × 10⁻⁴ | 1.2 × 10⁷ × |
| early fault-tolerant logical | 1 × 10⁻⁸ | 1.2 × 10³ × |

Error correction closes this gap. The physical-qubit overhead it demands is the
cost any serious quantum-finance proposal has to price in.

---

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/01_validate.py --quick   # ~20 s
python scripts/01_validate.py           # ~6 min, 19 configs × 5 seeds
python scripts/02_analyse.py            # scaling fits and crossover
python scripts/03_resources.py          # transpiled gate counts
```

`src/pricing.py` provides four independent routes to the same quantity —
Black–Scholes closed form, exact expectation on the discretised grid, classical
Monte Carlo, and amplitude estimation — so every discrepancy can be attributed to a
specific source rather than absorbed into one number.

Results are written to `results/` as CSV and JSON, flushed per seed so an
interrupted run leaves usable data.

**Environment.** Python 3.12, `qiskit==2.5.2`, `qiskit-aer==0.17.2`,
`qiskit-algorithms==0.4.0`, `qiskit-finance==0.4.1`. Run on macOS and Linux with
identical output to six decimal places, including oracle query counts.

**Compatibility note.** The Qiskit Finance tutorials use
`qiskit_aer.primitives.Sampler`, which fails inside
`IterativeAmplitudeEstimation` on this version combination with an uninformative
"job was not completed successfully". `qiskit.primitives.StatevectorSampler`
works.

---

## Limitations

- One option (S₀=100, K=105, σ=0.20, r=0.03, T=1) at up to 5 uncertainty qubits.
- The crossover is extrapolated from a 7-point fit spanning 3,277 to 912,589
  queries. It is indicative, not definitive.
- Oracle queries are counted, not physical gate time or shot overhead.
- Noiseless simulation throughout. Gate errors would move the frontier the wrong
  way; the conclusion is therefore optimistic.
- Iterative amplitude estimation only. Maximum-likelihood and canonical variants
  may sit differently on the frontier.
- The scaling argument applies to any payoff encoded through a linearised
  amplitude function. Path-dependent payoffs, where quantum methods are more
  plausibly interesting, are not covered.

## References

- Woerner & Egger, *Quantum risk analysis*, npj Quantum Information 5, 15 (2019)
- Stamatopoulos et al., *Option pricing using quantum computers*, Quantum 4, 291 (2020)
- Grinko et al., *Iterative quantum amplitude estimation*, npj Quantum Information 7, 52 (2021)
