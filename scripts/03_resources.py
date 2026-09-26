"""
Convert oracle queries into two-qubit gates on a realistic device.

The sweep counts oracle queries. A query is not a unit of hardware effort: it
is one application of the Grover operator Q, which on a real machine becomes
hundreds of two-qubit gates once the circuit is compiled to a native basis and
routed onto a fixed connectivity graph.

This script measures that conversion factor, then asks the only question that
matters for a fault-tolerance argument: at the query count where amplitude
estimation would finally overtake classical Monte Carlo, how many two-qubit
gates must execute without a single one failing?

Usage:
    python scripts/03_resources.py
    python scripts/03_resources.py --crossover-queries 69000000
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pricing import OptionSpec

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# A superconducting native basis. cz/rz/sx/x matches current IBM devices.
BASIS = ["cz", "rz", "sx", "x"]
TWO_QUBIT = {"cz", "ecr", "cx"}


def build_problem(nq: int, c: float, n_std: float = 5.0):
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


def count(circuit, coupling: CouplingMap | None, seed: int = 11) -> dict:
    """Transpile and report depth and two-qubit gate usage."""
    tqc = transpile(circuit, basis_gates=BASIS, coupling_map=coupling,
                    optimization_level=3, seed_transpiler=seed)
    ops = tqc.count_ops()
    two_q = sum(v for k, v in ops.items() if k in TWO_QUBIT)
    depth_2q = tqc.depth(lambda instr: instr.operation.num_qubits == 2)
    return {"qubits": tqc.num_qubits, "depth": tqc.depth(),
            "depth_2q": depth_2q, "gates_2q": two_q,
            "gates_total": sum(ops.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossover-queries", type=float, default=None,
                    help="query count at the classical crossover (from 02_analyse)")
    ap.add_argument("--configs", default="3,4,5",
                    help="uncertainty qubit counts to profile")
    args = ap.parse_args()

    nqs = [int(x) for x in args.configs.split(",")]
    RESULTS.mkdir(exist_ok=True)

    print("=" * 74)
    print("1. COST OF ONE ORACLE QUERY")
    print("=" * 74)
    print(f"   Basis {BASIS}. Two connectivity assumptions:")
    print("   all-to-all is a lower bound; heavy-hex is what IBM hardware provides.")
    print()
    print(f"   {'nq':>3} {'c':>6} {'qubits':>7} | {'all-to-all 2Q':>14} "
          f"| {'heavy-hex 2Q':>13} {'2Q depth':>9} {'routing':>8}")

    rows = []
    for nq in nqs:
        for c in (0.05,):
            prob = build_problem(nq, c)
            q_circ = prob.grover_operator
            ideal = count(q_circ, None)
            hh = CouplingMap.from_heavy_hex(distance=max(3, 2 * (ideal["qubits"] // 4) + 1))
            real = count(q_circ, hh)
            overhead = real["gates_2q"] / ideal["gates_2q"] if ideal["gates_2q"] else float("nan")
            rows.append({"num_qubits": nq, "rescaling_factor": c,
                         "circuit_qubits": ideal["qubits"],
                         "gates_2q_ideal": ideal["gates_2q"],
                         "gates_2q_routed": real["gates_2q"],
                         "depth_2q_routed": real["depth_2q"],
                         "depth_routed": real["depth"],
                         "routing_overhead": overhead})
            print(f"   {nq:>3} {c:>6} {ideal['qubits']:>7} | {ideal['gates_2q']:>14,} "
                  f"| {real['gates_2q']:>13,} {real['depth_2q']:>9,} {overhead:>7.1f}x")

    ref = rows[-1]
    per_query = ref["gates_2q_routed"]

    if len(rows) >= 2:
        w = np.log([r["circuit_qubits"] for r in rows])
        g = np.log([r["gates_2q_routed"] for r in rows])
        slope = float(np.polyfit(w, g, 1)[0])
        print()
        print(f"   Routed two-qubit gates scale as (circuit width)^{slope:.1f}, and the")
        print("   routing overhead itself grows with width. A payoff needing 10-20")
        print("   uncertainty qubits rather than 5 costs roughly "
              f"{(25 / ref['circuit_qubits']) ** slope:.0f}x more per query")
        print("   than the figure below, so treat this section as a lower bound.")

    print()
    print("=" * 74)
    print("2. TOTAL GATE COUNT AT THE CLASSICAL CROSSOVER")
    print("=" * 74)

    n_cross = args.crossover_queries
    if n_cross is None:
        summary = RESULTS / "summary.json"
        print("   No --crossover-queries given; using 69e6 from the frontier fit.")
        n_cross = 69e6

    total = n_cross * per_query
    print(f"   Oracle queries needed to overtake classical MC : {n_cross:>16,.0f}")
    print(f"   Routed two-qubit gates per query (nq={ref['num_qubits']})        : {per_query:>16,}")
    print(f"   Two-qubit gates in the whole computation       : {total:>16.3e}")
    print()
    print("   For the result to survive, the expected number of two-qubit errors")
    print("   across the entire run must be well below one, so the per-gate error")
    print("   rate must satisfy  p << 1 / (total gates).")
    print()
    required = 1.0 / total
    print(f"   Required two-qubit error rate : p << {required:.2e}")
    print()
    print(f"   {'device generation':<34} {'2Q error':>10} {'gap':>14}")
    for name, p in [("current superconducting (~2025)", 2e-3),
                    ("optimistic near-term", 1e-4),
                    ("early fault-tolerant logical", 1e-8)]:
        print(f"   {name:<34} {p:>10.0e} {p / required:>13,.0f}x")
    print()
    print("   The rightmost column is how much better the error rate must get.")
    print("   Error correction closes this, but the logical-qubit overhead is the")
    print("   cost that any serious quantum-finance proposal has to price in.")

    out = RESULTS / "resources.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "resources.json").write_text(json.dumps({
        "basis_gates": BASIS, "per_query_2q_gates": per_query,
        "crossover_queries": n_cross, "total_2q_gates": total,
        "required_2q_error_rate": required, "profiles": rows,
    }, indent=2))
    print(f"\n   Wrote resources.csv and resources.json to results/")


if __name__ == "__main__":
    main()
