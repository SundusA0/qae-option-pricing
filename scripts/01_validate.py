"""
Validate the QAE option pricer and decompose its error budget.

Four independent error sources stack up between the analytic expected payoff
and what QAE returns:

  1. Truncation     - the lognormal is cut off at mean +/- n_std * sd
  2. Discretisation - the survivor is placed on 2**nq grid points
  3. Approximation  - the small-angle linearisation of the payoff, set by c
  4. Estimation     - finite sampling in amplitude estimation, set by epsilon

Each configuration is repeated over several seeds. A single run cannot tell a
systematic offset apart from a lucky draw, so the sweep reports bias (distance
from the mean to the truth) and spread separately.

Only the fourth shrinks when you spend more oracle queries. The first is the
one most easily missed: a call payoff grows linearly in the upper tail, so a
window that looks generous for the distribution can still be far too narrow
for the option written on it. This script measures all four separately so the
trade-off is visible rather than assumed.

Usage:
    python scripts/01_validate.py            # full sweep (~10-20 min)
    python scripts/01_validate.py --quick    # coarse sweep (~1 min)
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


def classical_baseline(spec: OptionSpec, target_se: float, seed: int = 0) -> dict:
    """
    Smallest classical MC sample size reaching a given standard error.

    This is the number QAE has to beat. Quoting a quantum result without it
    is the most common way these comparisons mislead.
    """
    pilot_n = 100_000
    _, se_pilot = expected_payoff_mc(spec, pilot_n, seed=seed)
    sigma = se_pilot * np.sqrt(pilot_n)
    n_needed = int(np.ceil((sigma / target_se) ** 2))
    est, se = expected_payoff_mc(spec, min(n_needed, 20_000_000), seed=seed)
    return {"n_samples": n_needed, "estimate": est, "std_error": se}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="coarse, fast sweep")
    parser.add_argument("--seeds", type=int, default=5,
                        help="repeats per configuration (default 5)")
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
    print("  Read down a column, not across a row: widening the window buys more")
    print("  than adding qubits, and at n_std=3 no qubit count converges at all.")
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
    print(f"QAE SWEEP  ({len(configs)} configurations x {n_seeds} seeds, n_std={N_STD})")
    print("=" * 68)
    print("  bias = |mean(estimate) - grid exact|, the part sampling cannot remove")
    print("  sd   = spread across seeds, the part that shrinks as eps tightens")
    print()
    print(f"  {'nq':>3} {'c':>6} {'eps':>7} {'mean':>9} {'bias':>9} {'sd':>9} "
          f"{'queries':>9} {'sec':>6}")

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
            for seed in range(n_seeds):
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
            rows.append({
                "n_std": N_STD, "num_qubits": nq, "rescaling_factor": c,
                "epsilon_target": eps, "n_seeds": n_seeds,
                "mean_estimate": mean, "grid_exact": grid,
                "bias": abs(mean - grid), "sd": sd,
                "mean_oracle_queries": statistics.fmean(queries),
                "seconds": dt,
            })
            print(f"  {nq:>3} {c:>6} {eps:>7} {mean:>9.5f} {abs(mean - grid):>9.5f} "
                  f"{sd:>9.5f} {statistics.fmean(queries):>9.0f} {dt:>6.1f}")

    print()
    print("=" * 68)
    print("CLASSICAL MONTE CARLO BASELINE")
    print("=" * 68)
    print(f"  {'target SE':>10}  {'samples needed':>15}  {'estimate':>10}")
    baselines = []
    for target in (1e-1, 1e-2, 1e-3):
        b = classical_baseline(spec, target)
        b["target_se"] = target
        baselines.append(b)
        print(f"  {target:>10}  {b['n_samples']:>15,}  {b['estimate']:>10.5f}")

    summary = {
        "black_scholes_price": bs,
        "expected_payoff_analytic": analytic,
        "option": vars(spec),
        "encoding_error": disc,
        "qae_n_std": N_STD,
        "qae_sweep": rows,
        "n_seeds": n_seeds,
        "classical_baseline": baselines,
    }
    out = RESULTS / ("summary_quick.json" if args.quick else "summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path.name} and {out.name} to results/")


if __name__ == "__main__":
    main()
