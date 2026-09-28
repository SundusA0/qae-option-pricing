"""
Convert oracle queries into two-qubit gates, and into a single-circuit noise budget.

The sweep counts oracle queries. A query is not a unit of hardware effort: it is
one application of the Grover operator Q, which on a real machine becomes
hundreds of two-qubit gates once compiled to a native basis and routed onto a
fixed connectivity graph.

Two distinct costs follow, and conflating them is a common error:

  Q-operator 2Q executions - two-qubit gates contributed by Q across all
                             oracle calls. This is a workload proxy, not the
                             complete gate count and not a fidelity
                             requirement: noise accumulates within each circuit
                             execution, and across independent shots it acts on
                             the estimator (bias and variance) rather than
                             imposing a zero-fault condition on the aggregate.

  Deepest single circuit - A followed by Q^k at the largest k the IQAE schedule
                           reaches. Noise accumulates within a single circuit
                           execution, so this provides the relevant circuit
                           scale for the conservative error proxy.

Usage:
    python scripts/03_resources.py
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
from src.pricing import OptionSpec, SHOTS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Heron-like superconducting basis used for this resource proxy.
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


def powerlaw(x, y) -> tuple[float, float]:
    """Least-squares fit of y = a * x**b in log space. Returns (b, a)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, loga = np.polyfit(np.log(x), np.log(y), 1)
    return float(b), float(np.exp(loga))


def measure_schedule(nq: int, c: float = 0.05,
                     eps_levels=(1e-2, 1e-3, 1e-4)) -> list[dict]:
    """
    Run IQAE and record the Grover powers it actually chooses.

    IQAE is adaptive: it raises k until the confidence interval is tight enough.
    The largest k reached is the longest single circuit execution the algorithm
    performs, which is the quantity a fidelity budget should be built on.
    """
    from qiskit_algorithms import IterativeAmplitudeEstimation
    from qiskit.primitives import StatevectorSampler

    prob = build_problem(nq, c)
    out = []
    for eps in eps_levels:
        iae = IterativeAmplitudeEstimation(
            epsilon_target=eps, alpha=0.05,
            sampler=StatevectorSampler(default_shots=SHOTS,
                                       seed=np.random.default_rng(42)))
        res = iae.estimate(prob)
        out.append({"eps": eps, "k_max": int(max(res.powers)),
                    "n_rounds": len(res.powers),
                    "queries": int(res.num_oracle_queries)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="3,4,5",
                    help="uncertainty qubit counts to profile")
    args = ap.parse_args()

    nqs = [int(x) for x in args.configs.split(",")]
    RESULTS.mkdir(exist_ok=True)

    print("=" * 74)
    print("1. COST OF ONE ORACLE QUERY")
    print("=" * 74)
    print(f"   Basis {BASIS}. Two connectivity assumptions:")
    print("   all-to-all is a lower bound; heavy-hex is used as a Heron-like")
    print("   connectivity proxy.")
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

    # State preparation A is run once per circuit; Q is run k times.
    _ref_prob = build_problem(ref["num_qubits"], ref["rescaling_factor"])
    _hh = CouplingMap.from_heavy_hex(
        distance=max(3, 2 * (ref["circuit_qubits"] // 4) + 1))
    A = count(_ref_prob.state_preparation, _hh)["gates_2q"]

    if len(rows) >= 2:
        print()
        print("   Routing overhead increases noticeably with circuit width across")
        print("   these points. A few small circuits are not enough to infer an")
        print("   asymptotic routing law, so no exponent is fitted here.")

    print()
    print("=" * 74)
    print("2. HOW DEEP IS THE DEEPEST CIRCUIT?")
    print("=" * 74)
    print("   IQAE picks Grover powers adaptively. The largest one it reaches is")
    print("   the longest single circuit execution, which provides the scale")
    print("   for the error proxy below. Measured from the schedule, not assumed.")
    print()
    schedule = measure_schedule(ref["num_qubits"])
    print("   'deepest 2Q gates' is estimated as A + k_max*Q from independently")
    print("   transpiled blocks; the composed circuit is not transpiled whole.")
    print()
    print(f"   {'eps':>8} {'rounds':>7} {'max k':>8} {'deepest 2Q gates':>18} "
          f"{'total queries':>14}")
    for rec in schedule:
        deep = A + rec["k_max"] * per_query
        rec["deepest_2q"] = deep
        print(f"   {rec['eps']:>8g} {rec['n_rounds']:>7} {rec['k_max']:>8,} "
              f"{deep:>18,} {rec['queries']:>14,}")

    eps_exp, eps_a = powerlaw([r["eps"] for r in schedule],
                              [r["k_max"] for r in schedule])
    print(f"\n   fit: k_max ~ eps^{eps_exp:.2f}")

    print()
    print("=" * 74)
    print("3. TWO DIFFERENT COSTS, OFTEN CONFLATED")
    print("=" * 74)
    deepest = schedule[-1]
    p_req = 1.0 / deepest["deepest_2q"]
    total_gates = deepest["queries"] * per_query
    print("   Under an independent-error model, p * G_2Q << 1 is a deliberately")
    print("   conservative zero-two-qubit-fault proxy for the deepest circuit. It")
    print("   is not a fault-tolerance threshold or a prediction of algorithmic")
    print("   failure probability.")
    print()
    print(f"   At the deepest configuration actually executed (eps={deepest['eps']:g}):")
    print(f"     deepest circuit        : {deepest['deepest_2q']:>14,} 2Q gates")
    print(f"     zero-fault proxy 1/G   : {p_req:>14.2e}")
    print()
    print("   Separately, Q-operator two-qubit gate executions across all oracle")
    print(f"   calls in the tightest run: approximately {total_gates:.2e}. This")
    print("   excludes state-preparation executions and single-qubit gates. It is")
    print("   a throughput and wall-clock cost, not a fidelity requirement.")
    print("   Noise accumulates within each circuit execution; across independent")
    print("   shots it acts on the estimator, as bias and variance, rather than")
    print("   imposing a zero-fault requirement on the aggregate workload.")
    print()
    print(f"   {'2Q error rate':>14} {'vs zero-fault proxy':>22}")
    for pdev in (2e-3, 1e-3, 1e-4, 1e-6):
        ratio = pdev / p_req
        note = f"{ratio:,.0f}x above" if ratio >= 1 else f"{1 / ratio:,.0f}x below"
        print(f"   {pdev:>14.0e} {note:>22}")
    print()
    print("   Consistency with the noise study: 04_noise.py maximises the useful")
    print("   gain and finds an optimal Grover power k* = 1/(p*Q). Substituting")
    print(f"   p = {p_req:.1e} gives k* = {1.0 / (p_req * per_query):,.0f}, matching the measured schedule")
    print(f"   maximum of {deepest['k_max']:,}. Note this is close to an algebraic identity")
    print("   rather than an independent confirmation: both express the same")
    print("   statement, that useful depth is capped near 1/(p*Q). It is worth")
    print("   stating because the two scripts arrive at it by different routes -")
    print("   one from the IQAE schedule, one from a fidelity-weighted optimum.")

    out = RESULTS / "resources.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "resources.json").write_text(json.dumps({
        "basis_gates": BASIS, "per_query_2q_gates": per_query,
        "gates_A": A, "schedule": schedule,
        "k_max_exponent": eps_exp,
        "deepest_circuit_2q_measured": deepest["deepest_2q"],
        "zero_fault_proxy_measured": p_req,
        "q_operator_2q_executions_tightest_run": total_gates,
        "profiles": rows,
    }, indent=2))
    print(f"\n   Wrote resources.csv and resources.json to results/")


if __name__ == "__main__":
    main()
