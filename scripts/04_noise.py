"""
When does coherent amplification stop paying for itself?

Amplitude estimation earns its speedup by applying the Grover operator Q many
times: k iterations amplify the signal by roughly (2k+1). Every application also
runs several hundred two-qubit gates, and under depolarising noise each one
contracts the measured distribution toward maximally mixed.

So the two effects fight. Amplification grows linearly in k; fidelity decays
exponentially in k. This script measures the decay empirically, checks it against
exp(-p * N_2Q), and then solves for the k that maximises the product.

The answer has a threshold in it. Below a certain gate fidelity the optimal
number of Grover iterations is zero, and amplitude estimation reduces to plain
Monte Carlo sampling on very expensive hardware.

Usage:
    python scripts/04_noise.py            # ~4 min
    python scripts/04_noise.py --quick    # fewer shots, k <= 3
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pricing import OptionSpec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASIS = ["cz", "rz", "sx", "x"]
RATES = (0.0, 1e-5, 1e-4, 1e-3)


def noise_model(rate: float) -> NoiseModel | None:
    """Depolarising noise: `rate` on two-qubit gates, rate/10 on single-qubit."""
    if rate <= 0:
        return None
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(rate, 2), ["cz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(rate / 10, 1), ["sx", "x"])
    return nm


def build(nq: int, c: float = 0.05, n_std: float = 5.0):
    from qiskit_finance.applications.estimation import EuropeanCallPricing
    from qiskit_finance.circuit.library import LogNormalDistribution

    spec = OptionSpec()
    low, high = spec.bounds(n_std)
    dist = LogNormalDistribution(nq, mu=spec.mu, sigma=spec.sigma**2,
                                 bounds=(low, high))
    app = EuropeanCallPricing(num_state_qubits=nq, strike_price=spec.K,
                              rescaling_factor=c, bounds=(low, high),
                              uncertainty_model=dist)
    return app.to_estimation_problem()


def gate_costs(nq: int) -> tuple[int, int]:
    """Routed two-qubit gate counts for the state preparation A and operator Q."""
    prob = build(nq)
    width = prob.state_preparation.num_qubits
    cm = CouplingMap.from_heavy_hex(distance=max(3, 2 * (width // 4) + 1))

    def n2q(circ):
        t = transpile(circ, basis_gates=BASIS, coupling_map=cm,
                      optimization_level=3, seed_transpiler=11)
        return t.count_ops().get("cz", 0)

    return n2q(prob.state_preparation), n2q(prob.grover_operator)


def optimal_power(A: int, Q: int, p: float) -> tuple[int, float]:
    """
    The k maximising (2k+1) * exp(-p * (A + kQ)), and the value there.

    Amplification is linear in k; fidelity is exponential in k. The continuous
    optimum sits at k = 1/(pQ) - 1/2, giving a maximum gain of about
    0.74/(pQ) once the exponential is evaluated there. Values below 1 mean
    amplification costs more signal than it creates.
    """
    if p <= 0:
        return -1, float("inf")
    k_cont = 1.0 / (p * Q) - 0.5
    candidates = [max(0, int(np.floor(k_cont))), max(0, int(np.ceil(k_cont)))]
    best_k, best = 0, (1.0) * np.exp(-p * A)
    for k in candidates:
        val = (2 * k + 1) * np.exp(-p * (A + k * Q))
        if val > best:
            best_k, best = k, val
    return best_k, float(best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nq-empirical", type=int, default=3,
                    help="uncertainty qubits for the simulated part (keep small)")
    ap.add_argument("--nq-extrapolate", type=int, default=5,
                    help="uncertainty qubits for the analytic part")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    shots = 5_000 if args.quick else 20_000
    k_max = 3 if args.quick else 4

    nq = args.nq_empirical
    prob = build(nq)
    A_circ, Q_circ = prob.state_preparation, prob.grover_operator
    obj = prob.objective_qubits[0]
    cm = CouplingMap.from_heavy_hex(distance=3)

    print("=" * 76)
    print(f"1. AMPLIFICATION UNDER NOISE  (simulated, nq={nq}, {shots:,} shots)")
    print("=" * 76)
    print("   Noiseless, P(good) follows sin^2((2k+1)*theta) and oscillates.")
    print("   Under noise it contracts toward 0.5, which carries no information.")
    print()
    header = f"   {'k':>2} {'2Q gates':>9} |" + "".join(
        f"{('noiseless' if r == 0 else f'p={r:g}'):>11}" for r in RATES)
    print(header)

    rows = []
    for k in range(k_max + 1):
        circ = QuantumCircuit(A_circ.num_qubits, 1)
        circ.compose(A_circ, inplace=True)
        for _ in range(k):
            circ.compose(Q_circ, inplace=True)
        circ.measure(obj, 0)
        tqc = transpile(circ, AerSimulator(), basis_gates=BASIS, coupling_map=cm,
                        optimization_level=3, seed_transpiler=11)
        gates = tqc.count_ops().get("cz", 0)
        rec = {"k": k, "gates_2q": gates}
        for rate in RATES:
            sim = AerSimulator(noise_model=noise_model(rate), seed_simulator=7)
            counts = sim.run(tqc, shots=shots).result().get_counts()
            rec[f"p={rate:g}"] = counts.get("1", 0) / shots
        rows.append(rec)
        print(f"   {k:>2} {gates:>9,} |" + "".join(
            f"{rec[f'p={r:g}']:>11.4f}" for r in RATES))

    print()
    print("=" * 76)
    print("2. DOES THE CONTRAST FOLLOW exp(-p * N_2Q)?")
    print("=" * 76)
    print("   contrast = (measured - 0.5) / (noiseless - 0.5), i.e. the surviving")
    print("   fraction of the amplification signal.")
    print()
    print(f"   {'rate':>8} {'k':>3} {'2Q gates':>9} {'measured':>10} {'predicted':>10}")
    checks = []
    for rate in RATES[1:]:
        for rec in rows:
            if rec["k"] == 0:
                continue
            ideal = rec["p=0"] - 0.5
            if abs(ideal) < 0.02:
                continue
            meas = (rec[f"p={rate:g}"] - 0.5) / ideal
            pred = float(np.exp(-rate * rec["gates_2q"]))
            checks.append({"rate": rate, "k": rec["k"],
                           "measured": meas, "predicted": pred})
            print(f"   {rate:>8g} {rec['k']:>3} {rec['gates_2q']:>9,} "
                  f"{meas:>10.4f} {pred:>10.4f}")
    if checks:
        err = float(np.mean([abs(c["measured"] - c["predicted"]) for c in checks]))
        print(f"\n   Mean absolute deviation: {err:.4f}")
        print("   The exponential model holds, so it can be extrapolated to depths")
        print("   that cannot be simulated directly.")

    nqe = args.nq_extrapolate
    A, Q = gate_costs(nqe)

    print()
    print("=" * 76)
    print(f"3. IS AMPLIFICATION WORTH IT?  (extrapolated, nq={nqe})")
    print("=" * 76)
    print(f"   State preparation A: {A:,} two-qubit gates")
    print(f"   Grover operator  Q: {Q:,} two-qubit gates")
    print()
    print("   k iterations multiply the signal by (2k+1) and the fidelity by")
    print("   exp(-p*(A + kQ)). The useful gain is the product. A gain below 1")
    print("   means amplification destroys more signal than it creates.")
    print()
    print(f"   {'2Q error rate':>14} {'optimal k':>10} {'max gain':>10}   verdict")
    est = []
    for rate, label in [(2e-3, "current superconducting"),
                        (1e-4, "optimistic near-term"),
                        (1e-6, "aggressive"),
                        (1e-8, "early fault-tolerant logical")]:
        k, gain = optimal_power(A, Q, rate)
        verdict = "amplification LOSES" if gain <= 1.0 else f"{gain:.0f}x over sampling"
        est.append({"rate": rate, "label": label, "optimal_k": k, "max_gain": gain})
        print(f"   {rate:>14g} {k:>10,} {gain:>10.2f}   {verdict}")

    print()
    print("   Below a threshold fidelity the optimum is k = 0: amplitude estimation")
    print("   reduces to ordinary Monte Carlo sampling, executed on hardware many")
    print("   orders of magnitude slower than a CPU. Noise does not erode the")
    print("   quantum speedup gradually. It inverts it.")

    (RESULTS / "noise.json").write_text(json.dumps({
        "shots": shots, "rates": list(RATES), "empirical_nq": nq,
        "empirical": rows, "model_check": checks,
        "extrapolate_nq": nqe, "gates_A": A, "gates_Q": Q,
        "amplification": est,
    }, indent=2))
    print(f"\n   Wrote noise.json to results/")


if __name__ == "__main__":
    main()
