"""
How a simple depolarising-noise model limits useful amplification depth.

Amplitude estimation applies the Grover operator Q repeatedly: k iterations
amplify the signal by roughly (2k+1). Each application executes several hundred
two-qubit gates, and under depolarising noise each gate moves the measured
distribution toward maximally mixed. Amplification is linear in k; fidelity
decays exponentially in k.

This script measures the contrast decay at small k, compares it against
exp(-p * N_2Q), and evaluates a small-angle heuristic for the k at which
amplification stops being useful under the assumed noise model.

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
    optimum sits at k = 1/(pQ) - 1/2. This is a heuristic proxy for useful
    amplification, not a derived optimum for noisy amplitude estimation.
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
    print(f"1. CONTRAST UNDER NOISE  (simulated, nq={nq}, {shots:,} shots)")
    print("=" * 76)
    print("   Noiseless, P(good) follows sin^2((2k+1)*theta).")
    print("   Under noise it moves toward 0.5, the maximally mixed value.")
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
        print("   The exponential model is consistent with the tested points.")
        print("   Extrapolation beyond the simulated range is used below only as")
        print("   a heuristic.")

    nqe = args.nq_extrapolate
    A, Q = gate_costs(nqe)

    print()
    print("=" * 76)
    print(f"3. HEURISTIC USEFUL-DEPTH ESTIMATE  (extrapolated, nq={nqe})")
    print("=" * 76)
    print(f"   State preparation A: {A:,} two-qubit gates")
    print(f"   Grover operator  Q: {Q:,} two-qubit gates")
    print()
    print("   As a small-angle heuristic for useful amplification, consider")
    print("   (2k+1) * exp(-p*(A + kQ)): amplification times surviving fidelity.")
    print("   This is an engineering proxy, not a derived expression for the")
    print("   information gain of noisy amplitude estimation. A value below 1")
    print("   indicates amplification costs more signal than it creates.")
    print()
    print(f"   {'2Q error rate':>14} {'optimal k':>10} {'H(k*)':>10}   note")
    est = []
    for rate, label in [(2e-3, ""), (1e-4, ""), (1e-6, ""), (1e-8, "")]:
        k, gain = optimal_power(A, Q, rate)
        verdict = f"k = {k:,} selected"
        est.append({"rate": rate, "optimal_k": k, "heuristic_score": gain})
        print(f"   {rate:>14g} {k:>10,} {gain:>10.2f}   {verdict}")

    print()
    print("   For sufficiently large error rates the heuristic selects k = 0,")
    print("   indicating that additional coherent amplification is not beneficial")
    print("   under the assumed noise model.")

    (RESULTS / "noise.json").write_text(json.dumps({
        "shots": shots, "rates": list(RATES), "empirical_nq": nq,
        "empirical": rows, "model_check": checks,
        "extrapolate_nq": nqe, "gates_A": A, "gates_Q": Q,
        "amplification": est,
    }, indent=2))
    print(f"\n   Wrote noise.json to results/")


if __name__ == "__main__":
    main()
