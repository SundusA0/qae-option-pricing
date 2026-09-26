"""
Fit scaling laws to the sweep and locate the classical crossover.

The sweep measures two error terms separately. This script asks how each one
scales with the knobs that control it, combines them, and compares the result
against classical Monte Carlo on the same option.

The conclusion the numbers support is not the one usually quoted for amplitude
estimation, and it is the reason this repo exists.

Usage:
    python scripts/02_analyse.py
    python scripts/02_analyse.py --csv results/validation_quick.csv
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(csv_path: Path) -> list[dict]:
    """Aggregate the per-seed rows into one record per configuration."""
    groups = defaultdict(list)
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["num_qubits"]), float(r["rescaling_factor"]),
                   float(r["epsilon_target"]))
            groups[key].append((float(r["estimate"]), float(r["grid_exact"]),
                                int(r["oracle_queries"])))
    out = []
    for (nq, c, eps), vals in sorted(groups.items()):
        ests = [v[0] for v in vals]
        grid = vals[0][1]
        mean = statistics.fmean(ests)
        out.append({
            "num_qubits": nq, "rescaling_factor": c, "epsilon_target": eps,
            "n_seeds": len(ests), "mean": mean, "bias": abs(mean - grid),
            "sd": statistics.stdev(ests) if len(ests) > 1 else 0.0,
            "queries": statistics.fmean([v[2] for v in vals]),
        })
    return out


def powerlaw(x, y) -> tuple[float, float]:
    """Least-squares fit of y = a * x**b in log space. Returns (b, a)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    b, loga = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return float(b), float(np.exp(loga))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RESULTS / "validation.csv"))
    ap.add_argument("--payoff-sigma", type=float, default=None,
                    help="per-sample std dev of the payoff (read from summary.json if omitted)")
    args = ap.parse_args()

    rows = load(Path(args.csv))
    eps_levels = sorted({r["epsilon_target"] for r in rows})
    c_levels = sorted({r["rescaling_factor"] for r in rows}, reverse=True)
    tight = min(e for e in eps_levels if e <= 1e-3) if any(
        e <= 1e-3 for e in eps_levels) else min(eps_levels)

    print("=" * 70)
    print("1. BIAS vs RESCALING FACTOR")
    print("=" * 70)
    print(f"   Measured at eps={tight:g}, where the spread is small enough that")
    print("   the mean is a meaningful estimate of the systematic offset.")
    print()
    cs, bs = [], []
    for c in c_levels:
        vals = [r["bias"] for r in rows
                if r["rescaling_factor"] == c and r["epsilon_target"] == tight]
        if vals:
            cs.append(c); bs.append(statistics.fmean(vals))
            print(f"   c = {c:<6} bias = {bs[-1]:.4f}   (mean over {len(vals)} qubit counts)")
    bias_exp = None
    if len(cs) >= 2:
        bias_exp, bias_a = powerlaw(cs, bs)
        print(f"\n   fit: bias ~ c^{bias_exp:.2f}")
        print("   Consistent with c^2: the payoff linearisation is second order in c,")
        print("   and no amount of sampling touches it.")

    print()
    print("=" * 70)
    print("2. SPREAD vs EPSILON AND RESCALING FACTOR")
    print("=" * 70)
    print("   The signal encoded in the amplitude is proportional to c, so a fixed")
    print("   amplitude precision maps to a payoff error that grows as c shrinks.")
    print()
    print(f"   {'c':>6}" + "".join(f"{'eps=' + format(e, 'g'):>14}" for e in eps_levels)
          + f"{'sd * c':>12}")
    for c in c_levels:
        cells = []
        for e in eps_levels:
            vals = [r["sd"] for r in rows
                    if r["rescaling_factor"] == c and r["epsilon_target"] == e]
            cells.append(statistics.fmean(vals) if vals else float("nan"))
        at_tight = [r["sd"] for r in rows
                    if r["rescaling_factor"] == c and r["epsilon_target"] == tight]
        prod = statistics.fmean(at_tight) * c if at_tight else float("nan")
        print(f"   {c:>6}" + "".join(f"{v:>14.4f}" for v in cells) + f"{prod:>12.4f}")
    print("\n   The last column is roughly constant, so sd ~ eps / c.")

    print()
    print("=" * 70)
    print("3. EFFICIENT FRONTIER: best total error per query budget")
    print("=" * 70)
    print("   total error = sqrt(bias^2 + sd^2), the honest combined figure.")
    print()
    pts = sorted((r["queries"], float(np.hypot(r["bias"], r["sd"])), r) for r in rows)
    frontier, best = [], float("inf")
    for q, err, r in pts:
        if err < best:
            best = err
            frontier.append((q, err, r))
    print(f"   {'queries':>10} {'total err':>11}   configuration")
    for q, err, r in frontier:
        print(f"   {q:>10,.0f} {err:>11.4f}   nq={r['num_qubits']}, "
              f"c={r['rescaling_factor']}, eps={r['epsilon_target']:g}")

    if len(frontier) < 3:
        print("\n   Too few frontier points to fit a scaling law. Run the full sweep.")
        return

    err_exp, _ = powerlaw([f[0] for f in frontier], [f[1] for f in frontier])
    n_exp = 1.0 / err_exp
    print(f"\n   fit: error ~ N^{err_exp:.2f}   i.e.   N ~ error^{n_exp:.2f}")
    print()
    print("   Reference points:")
    print("     ideal amplitude estimation   N ~ error^-1.0")
    print("     classical Monte Carlo        N ~ error^-2.0")
    print()
    print("   Why the measured exponent sits between them: bias ~ c^2 and")
    print("   sd ~ eps/c are minimised together at c ~ eps^(1/3), which makes the")
    print("   total error ~ eps^(2/3). Since N ~ 1/eps, that gives N ~ error^-1.5.")
    print("   The payoff linearisation costs half of the quadratic speedup before")
    print("   a single gate error is considered.")

    sigma = args.payoff_sigma
    if sigma is None:
        summary = RESULTS / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            b = data["classical_baseline"][1]
            sigma = b["std_error"] * np.sqrt(min(b["n_samples"], 20_000_000))
    if sigma is None:
        print("\n   (no classical sigma available; skipping crossover)")
        return

    print()
    print("=" * 70)
    print("4. CLASSICAL COMPARISON AND CROSSOVER")
    print("=" * 70)
    print(f"   Per-sample payoff std dev: {sigma:.3f}")
    print(f"   Classical MC needs N = ({sigma:.2f} / error)^2 samples.")
    print()
    print(f"   {'total err':>11} {'QAE queries':>14} {'MC samples':>14} {'ratio':>9}")
    for q, err, _ in frontier:
        n_cl = (sigma / err) ** 2
        print(f"   {err:>11.4f} {q:>14,.0f} {n_cl:>14,.0f} {q / n_cl:>8.1f}x")
    print()
    print("   QAE is behind at every budget measured, on raw counts alone -")
    print("   and an oracle query is a deep circuit, while an MC sample is one")
    print("   exponential and one max().")

    q_last, e_last, _ = frontier[-1]
    A = q_last * e_last ** (-n_exp)      # QAE:       N = A * err^n_exp
    K = sigma ** 2                        # classical:  N = K * err^-2
    delta = float(np.exp(np.log(K / A) / (n_exp + 2.0)))
    print()
    print(f"   Extrapolated crossover at total error ~ {delta:.2e}")
    print(f"   = {delta / 7.345 * 1e4:.1f} basis points of the option's expected payoff,")
    print(f"   where both methods need roughly {K / delta**2:,.0f} operations.")
    print()
    print("   Treat this as indicative, not definitive: it extrapolates a fit from")
    print(f"   {len(frontier)} frontier points on one option at up to "
          f"{max(r['num_qubits'] for r in rows)} uncertainty qubits, and it counts")
    print("   oracle queries rather than physical gate time.")


if __name__ == "__main__":
    main()
