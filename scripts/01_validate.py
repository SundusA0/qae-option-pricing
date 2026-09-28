"""
Validate the QAE option pricer and decompose its error budget.

Three contributions lie between the analytic expected payoff and what QAE
returns. The first is measured on its own; the second and third are measured
together as the estimator's offset and spread relative to the grid expectation:

  1. Encoding      - truncation to a window, renormalisation, and discretisation
                     onto 2**nq grid points, taken together as the gap between
                     the exact grid expectation and the analytic value
  2. Approximation - the small-angle linearisation of the payoff, set by c
  3. Estimation    - finite sampling in amplitude estimation, set by epsilon

Only the third shrinks when you spend more oracle queries. Truncation and
discretisation are not separated here: that would need a continuous truncated
reference, which this script does not compute.

Each configuration is repeated over several seeds. A single run cannot tell a
systematic offset apart from a lucky draw, so the sweep reports bias (distance
from the mean to the truth), the standard error of that mean, and the spread.

Usage:
    python scripts/01_validate.py --quick    # coarse sweep (~20 s)
    python scripts/01_validate.py            # full sweep: 18 configs x 20 seeds
                                             #   + one eps=1e-4 probe x 5 seeds
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pricing import (
    OptionSpec,
    black_scholes_call,
    expected_payoff_analytic,
    expected_payoff_grid,
    expected_payoff_mc,
    expected_payoff_qae,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def continuous_mc_baseline(spec: OptionSpec, target_se: float, seed: int = 0) -> dict:
    """
    Classical MC sample size required for a given standard error, on the
    continuous lognormal. A sanity baseline for the estimator; the matched
    comparison on the encoded grid is made in 02_analyse.py.

    The requirement is computed from a pilot variance estimate. The actual
    simulation is capped at 20M samples, so for tight targets the returned
    estimate and standard error come from fewer samples than required; both
    counts are returned so the two are not confused.
    """
    pilot_n = 100_000
    _, se_pilot = expected_payoff_mc(spec, pilot_n, seed=seed)
    sigma = se_pilot * np.sqrt(pilot_n)
    required = int(np.ceil((sigma / target_se) ** 2))
    simulated = min(required, 20_000_000)
    est, se = expected_payoff_mc(spec, simulated, seed=seed)
    return {"required_samples": required, "simulated_samples": simulated,
            "estimate": est, "std_error": se}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="coarse, fast sweep")
    parser.add_argument("--seeds", type=int, default=20,
                        help="repeats per configuration (default 20)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    spec = OptionSpec()

    bs = black_scholes_call(spec)
    analytic = expected_payoff_analytic(spec)

    print("=" * 68)
    print("REFERENCE VALUES")
    print("=" * 68)
    print(f"  Black-Scholes price (discounted) : {bs:.6f}")
    print(f"  Expected payoff (undiscounted)   : {analytic:.6f}")
    print("  QAE estimates the undiscounted payoff; multiply by exp(-rT) for price.")

    print()
    print("=" * 68)
    print("ENCODING ERROR  |grid exact - analytic|, before any estimation")
    print("=" * 68)
    print("  Rows: truncation width (n_std).  Columns: uncertainty qubits (nq).")
    print("  Window width and grid resolution must be co-tuned; increasing either")
    print("  alone does not guarantee lower encoding error. At n_std=3 no qubit")
    print("  count converges over this range.")
    print()
    nq_grid = (3, 4, 5, 6, 7)
    print("  n_std " + " ".join(f"{'nq=' + str(n):>10}" for n in nq_grid))
    disc = []
    for ns in (3, 4, 5, 6, 7):
        cells = []
        for nq in nq_grid:
            g = expected_payoff_grid(spec, nq, n_std=ns)
            err = abs(g - analytic)
            cells.append(err)
            disc.append({"n_std": ns, "num_qubits": nq,
                         "grid_exact": g, "encoding_error": err})
        print(f"  {ns:>5} " + " ".join(f"{c:>10.6f}" for c in cells))
    print()
    print("  Caution: small errors at low nq can be accidental cancellation -")
    print("  a coarse grid overshooting into a truncation deficit, not accuracy.")

    # n_std=5 is the narrowest window whose encoding error is small enough
    # that the payoff linearisation and sampling terms are visible underneath it.
    N_STD = 5

    if args.quick:
        configs = [(4, c, e) for c in (0.25, 0.05) for e in (1e-2, 1e-3)]
        n_seeds = min(args.seeds, 3)
    else:
        configs = [(nq, c, e)
                   for nq in (3, 4, 5)
                   for c in (0.25, 0.10, 0.05)
                   for e in (1e-2, 1e-3)]
        # 1e-4 is ~15x the cost of 1e-3, so probe it at one configuration only
        configs += [(4, 0.05, 1e-4)]
        n_seeds = args.seeds

    print()
    print("=" * 68)
    n_probe = sum(1 for _, _, e in configs if e < 1e-3)
    n_main = len(configs) - n_probe
    label = (f"{n_main} configs x {n_seeds} seeds"
             + (f" + {n_probe} eps<1e-3 probe x {min(n_seeds, 5)} seeds" if n_probe else ""))
    print(f"QAE SWEEP  ({label}, n_std={N_STD})")
    print("=" * 68)
    print("  bias = |mean(estimate) - grid exact|; se = sd/sqrt(n) is the standard")
    print("  error of the mean estimate. A bias smaller than se is not")
    print("  distinguishable from zero at that seed count.")
    print()
    print(f"  {'nq':>3} {'c':>6} {'eps':>7} {'n':>3} {'mean':>9} {'bias':>8} "
          f"{'se':>7} {'sd':>8} {'queries':>9} {'sec':>6}")

    csv_path = RESULTS / ("validation_quick.csv" if args.quick else "validation.csv")
    rows = []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "n_std", "num_qubits", "rescaling_factor", "epsilon_target", "seed",
            "estimate", "grid_exact", "err_vs_grid", "err_vs_analytic",
            "oracle_queries", "circuit_qubits",
        ])
        writer.writeheader()

        for nq, c, eps in configs:
            grid = expected_payoff_grid(spec, nq, n_std=N_STD)
            ests, queries = [], []
            t0 = time.perf_counter()
            # the single 1e-4 probe is ~15x the cost of the rest; cap its seeds
            n_here = min(n_seeds, 5) if eps < 1e-3 else n_seeds
            for seed in range(n_here):
                r = expected_payoff_qae(spec, nq, epsilon_target=eps,
                                        rescaling_factor=c, n_std=N_STD,
                                        seed=1000 + seed)
                ests.append(r["estimate"])
                queries.append(r["oracle_queries"])
                writer.writerow({
                    "n_std": N_STD, "num_qubits": nq, "rescaling_factor": c,
                    "epsilon_target": eps, "seed": 1000 + seed,
                    "estimate": r["estimate"], "grid_exact": grid,
                    "err_vs_grid": abs(r["estimate"] - grid),
                    "err_vs_analytic": abs(r["estimate"] - analytic),
                    "oracle_queries": r["oracle_queries"],
                    "circuit_qubits": r["circuit_qubits"],
                })
                fh.flush()  # survive an interrupted run
            dt = time.perf_counter() - t0
            mean = statistics.fmean(ests)
            sd = statistics.stdev(ests) if len(ests) > 1 else 0.0
            se = sd / (len(ests) ** 0.5) if len(ests) > 1 else float("nan")
            rows.append({
                "n_std": N_STD, "num_qubits": nq, "rescaling_factor": c,
                "epsilon_target": eps, "n_seeds": len(ests),
                "mean_estimate": mean, "grid_exact": grid,
                "bias": abs(mean - grid), "se": se, "sd": sd,
                "mean_oracle_queries": statistics.fmean(queries),
                "seconds": dt,
            })
            print(f"  {nq:>3} {c:>6} {eps:>7} {len(ests):>3} {mean:>9.5f} "
                  f"{abs(mean - grid):>8.4f} {se:>7.4f} {sd:>8.4f} "
                  f"{statistics.fmean(queries):>9.0f} {dt:>6.1f}")

    print()
    print("=" * 68)
    print("CLASSICAL MONTE CARLO BASELINE (continuous lognormal)")
    print("=" * 68)
    print("  Sanity baseline on the untruncated model. The matched-grid comparison")
    print("  that supports the README's query ratios is made in 02_analyse.py.")
    print("  'required' is the sample count for the target SE, from a pilot")
    print("  variance estimate; 'simulated' is what was actually run (capped at")
    print("  20M), and the estimate and SE columns come from that run.")
    print()
    print(f"  {'target SE':>10}  {'required':>13}  {'simulated':>11}  "
          f"{'estimate':>10}  {'SE (sim.)':>10}")
    baselines = []
    for target in (1e-1, 1e-2, 1e-3):
        b = continuous_mc_baseline(spec, target)
        b["target_se"] = target
        baselines.append(b)
        print(f"  {target:>10}  {b['required_samples']:>13,}  "
              f"{b['simulated_samples']:>11,}  {b['estimate']:>10.5f}  "
              f"{b['std_error']:>10.5f}")

    summary = {
        "black_scholes_price": bs,
        "expected_payoff_analytic": analytic,
        "option": vars(spec),
        "encoding_error": disc,
        "qae_n_std": N_STD,
        "qae_sweep": rows,
        "n_seeds": n_seeds,
        "classical_baseline": baselines,  # key kept for existing summary.json files
    }
    out = RESULTS / ("summary_quick.json" if args.quick else "summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path.name} and {out.name} to results/")


if __name__ == "__main__":
    main()
