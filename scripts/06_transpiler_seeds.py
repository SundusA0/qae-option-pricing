"""
Transpiler-seed variability of the resource counts, and a check of the additive
deepest-circuit estimate.

03_resources.py reports routed two-qubit gate counts from one transpiler seed
(11). Qiskit's layout and routing passes are stochastic, so a single seed is one
realisation of a heuristic compiler, not necessarily a typical one. This script
re-transpiles the state preparation A and the Grover operator Q over many seeds
on the same basis and coupling maps, and reports the spread with the reference
seed's position in it. The reference numbers used elsewhere are left unchanged.

It also checks the additive approximation behind the deepest-circuit estimate.
03_resources.py estimates the deepest circuit as A + k_max*Q from separately
transpiled blocks. Here the composed circuit A.Q^k is transpiled whole for small
k, on the same coupling map and seed, and compared with A + k*Q. The marginal
cost of one more Q inside the composed circuit is the quantity that would carry
over to large k.

Usage:
    python scripts/06_transpiler_seeds.py
    python scripts/06_transpiler_seeds.py --seeds 50 --composed-nq 3,5 --k-max 4
"""

import argparse
import importlib.util
import json
import warnings
from pathlib import Path
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# Reuse build_problem() and count() from 03_resources.py so the basis, coupling
# maps and counting rule are identical by construction.
_spec = importlib.util.spec_from_file_location("resources03", ROOT / "scripts" / "03_resources.py")
r03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r03)

REFERENCE_SEED = 11


def heavy_hex_for(width: int) -> CouplingMap:
    return CouplingMap.from_heavy_hex(distance=max(3, 2 * (width // 4) + 1))


def spread(values: list, ref_seed_index: int) -> dict:
    arr = np.asarray(values, float)
    ref = float(arr[ref_seed_index])
    return {"median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
            "q1": float(np.percentile(arr, 25)), "q3": float(np.percentile(arr, 75)),
            "reference": ref, "reference_rank": float((arr <= ref).mean())}


def fmt(s: dict) -> str:
    return (f"{s['reference']:>7,.0f} {s['median']:>8,.0f} [{s['min']:>6,.0f}, {s['max']:>6,.0f}]"
            f"  {s['reference_rank']:>5.0%}")


def composed_circuit(prob, k: int) -> QuantumCircuit:
    """A followed by Q^k with the objective qubit measured, as in 04_noise.py."""
    A, Q = prob.state_preparation, prob.grover_operator
    circ = QuantumCircuit(A.num_qubits, 1)
    circ.compose(A, inplace=True)
    for _ in range(k):
        circ.compose(Q, inplace=True)
    circ.measure(prob.objective_qubits[0], 0)
    return circ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20, help="transpiler seeds 0..N-1 (must include 11)")
    ap.add_argument("--configs", default="3,4,5", help="uncertainty qubit counts for the spread")
    ap.add_argument("--composed-nq", default="3,5", help="qubit counts for the composed-circuit check")
    ap.add_argument("--k-max", type=int, default=4, help="largest k in the composed-circuit check")
    ap.add_argument("--composed-seeds", type=int, default=5, help="seeds for the composed check")
    args = ap.parse_args()
    if not 0 <= REFERENCE_SEED < args.seeds:
        ap.error(f"--seeds must exceed the reference seed {REFERENCE_SEED}")
    seeds = list(range(args.seeds))
    RESULTS.mkdir(exist_ok=True)

    print("=" * 74)
    print(f"1. ROUTED TWO-QUBIT GATES OVER {args.seeds} TRANSPILER SEEDS  (reference seed {REFERENCE_SEED})")
    print("=" * 74)
    print("   Same basis, coupling maps and counting as 03_resources.py; only")
    print("   seed_transpiler varies. 'rank' is the fraction of seeds at or below the")
    print("   reference value.")
    print()
    print(f"   {'nq':>3} {'block':>6} {'ref':>8} {'median':>8} {'[min, max]':>17} {'rank':>6}")
    profiles = {}
    for nq in (int(x) for x in args.configs.split(",")):
        prob = r03.build_problem(nq, 0.05)
        cm = heavy_hex_for(prob.state_preparation.num_qubits)
        q_gates, q_depth, a_gates = [], [], []
        for s in seeds:
            q = r03.count(prob.grover_operator, cm, seed=s)
            a = r03.count(prob.state_preparation, cm, seed=s)
            q_gates.append(q["gates_2q"]); q_depth.append(q["depth_2q"]); a_gates.append(a["gates_2q"])
        prof = {"Q_gates_2q": spread(q_gates, REFERENCE_SEED), "Q_depth_2q": spread(q_depth, REFERENCE_SEED),
                "A_gates_2q": spread(a_gates, REFERENCE_SEED),
                "raw": {"seeds": seeds, "Q_gates_2q": q_gates, "Q_depth_2q": q_depth, "A_gates_2q": a_gates}}
        profiles[nq] = prof
        print(f"   {nq:>3} {'Q':>6} {fmt(prof['Q_gates_2q'])}")
        print(f"   {'':>3} {'Q dep':>6} {fmt(prof['Q_depth_2q'])}")
        print(f"   {'':>3} {'A':>6} {fmt(prof['A_gates_2q'])}")

    # implication for the deepest-circuit estimate at the reference configuration
    res_path = next((p for p in (RESULTS / "resources.json", RESULTS / "reference" / "resources.json")
                     if p.exists()), None)
    deepest = None
    if res_path is not None:
        res = json.loads(res_path.read_text())
        k_max = res["schedule"][-1]["k_max"]
        ref_nq = max(profiles)
        pq, pa = profiles[ref_nq]["Q_gates_2q"], profiles[ref_nq]["A_gates_2q"]
        deepest = {"nq": ref_nq, "k_max": k_max, "eps": res["schedule"][-1]["eps"],
                   "additive_reference": pa["reference"] + k_max * pq["reference"],
                   "additive_median": pa["median"] + k_max * pq["median"],
                   "additive_min": pa["min"] + k_max * pq["min"],
                   "additive_max": pa["max"] + k_max * pq["max"]}
        print()
        print(f"   Additive deepest-circuit estimate A + k_max*Q at nq={ref_nq}, k_max={k_max:,}:")
        print(f"     reference seed : {deepest['additive_reference']:>12,.0f}")
        print(f"     seed median    : {deepest['additive_median']:>12,.0f}   "
              f"[{deepest['additive_min']:,.0f}, {deepest['additive_max']:,.0f}]")

    print()
    print("=" * 74)
    print(f"2. COMPOSED CIRCUIT A.Q^k TRANSPILED WHOLE vs A + k*Q  (k <= {args.k_max})")
    print("=" * 74)
    print("   Same coupling map and seed for both. 'marginal' is G(k) - G(k-1) in the")
    print("   composed circuit, the cost of one more Q once boundaries are routed;")
    print("   its ratio to the block count Q is what would carry over to large k.")
    composed = {}
    c_seeds = list(range(args.composed_seeds))
    if REFERENCE_SEED not in c_seeds:
        c_seeds.append(REFERENCE_SEED)
    for nq in (int(x) for x in args.composed_nq.split(",")):
        prob = r03.build_problem(nq, 0.05)
        cm = heavy_hex_for(prob.state_preparation.num_qubits)
        per_seed = {}
        for s in c_seeds:
            A = r03.count(prob.state_preparation, cm, seed=s)["gates_2q"]
            Q = r03.count(prob.grover_operator, cm, seed=s)["gates_2q"]
            g = [r03.count(composed_circuit(prob, k), cm, seed=s)["gates_2q"] for k in range(args.k_max + 1)]
            per_seed[s] = {"A": A, "Q": Q, "composed": g}
        rows = []
        for k in range(1, args.k_max + 1):
            ratios = [per_seed[s]["composed"][k] / (per_seed[s]["A"] + k * per_seed[s]["Q"]) for s in c_seeds]
            marg = [(per_seed[s]["composed"][k] - per_seed[s]["composed"][k - 1]) / per_seed[s]["Q"] for s in c_seeds]
            r = per_seed[REFERENCE_SEED]
            rows.append({"k": k, "composed_ref": r["composed"][k], "additive_ref": r["A"] + k * r["Q"],
                         "ratio_ref": r["composed"][k] / (r["A"] + k * r["Q"]),
                         "ratio_median": float(np.median(ratios)), "ratio_min": float(min(ratios)),
                         "ratio_max": float(max(ratios)),
                         "marginal_over_Q_ref": (r["composed"][k] - r["composed"][k - 1]) / r["Q"],
                         "marginal_over_Q_median": float(np.median(marg))})
        composed[nq] = {"seeds": c_seeds, "per_seed": per_seed, "rows": rows}
        print()
        print(f"   nq={nq}  (reference seed; ratio spread over {len(c_seeds)} seeds)")
        print(f"   {'k':>3} {'composed':>10} {'A + kQ':>10} {'ratio':>7} {'ratio [min, max]':>18} {'marginal/Q':>11}")
        for row in rows:
            print(f"   {row['k']:>3} {row['composed_ref']:>10,} {row['additive_ref']:>10,} "
                  f"{row['ratio_ref']:>7.3f} [{row['ratio_min']:>6.3f}, {row['ratio_max']:>6.3f}]"
                  f"  {row['marginal_over_Q_ref']:>10.3f}")

    out = {"seeds": args.seeds, "reference_seed": REFERENCE_SEED,
           "profiles": {str(k): v for k, v in profiles.items()}, "deepest_additive": deepest,
           "composed": {str(k): {"seeds": v["seeds"], "rows": v["rows"],
                                 "per_seed": {str(s): d for s, d in v["per_seed"].items()}}
                        for k, v in composed.items()}}
    (RESULTS / "transpiler_seeds.json").write_text(json.dumps(out, indent=2))
    print(f"\n   Wrote transpiler_seeds.json to results/")


if __name__ == "__main__":
    main()
