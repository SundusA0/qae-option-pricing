"""
Fit scaling laws to the sweep and compare QAE with a matched classical Monte
Carlo baseline.

The sweep measures two error terms separately. This script asks how each one
scales with the knobs that control it, combines them into conditional and
end-to-end frontiers, and costs classical Monte Carlo against the same encoded
distribution so both methods are scored on the same quantity.

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
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))


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
            "n_seeds": len(ests), "mean": mean, "grid": grid,
            "bias": abs(mean - grid),
            "sd": statistics.stdev(ests) if len(ests) > 1 else 0.0,
            "queries": statistics.fmean([v[2] for v in vals]),
        })
    return out


def first_existing(*paths: Path) -> Path:
    """Return the first path that exists; the last one if none do."""
    for p in paths:
        if p.exists():
            return p
    return paths[-1]


def grid_payoff_sigma(num_qubits: int, n_std: float = 5.0) -> float:
    """
    Exact standard deviation of the payoff ON the encoded grid.

    The grid is a finite discrete distribution, so this is a sum rather than an
    estimate. Using it means the classical comparison below estimates the same
    quantity amplitude estimation does, instead of the untruncated lognormal.
    """
    from qiskit_finance.circuit.library import LogNormalDistribution
    from src.pricing import OptionSpec

    spec = OptionSpec()
    low, high = spec.bounds(n_std)
    dist = LogNormalDistribution(num_qubits, mu=spec.mu, sigma=spec.sigma**2,
                                 bounds=(low, high))
    probs = np.asarray(dist.probabilities, float)
    vals = np.asarray(dist.values, float)
    payoff = np.maximum(vals - spec.K, 0.0)
    mean = float(probs @ payoff)
    return float(np.sqrt(probs @ (payoff - mean) ** 2))


def powerlaw(x, y) -> tuple[float, float]:
    """Least-squares fit of y = a * x**b in log space. Returns (b, a)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    b, loga = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return float(b), float(np.exp(loga))



def plot_frontiers(cond, e2e, comparison, deepest, deepest_e2e, out_path: Path) -> bool:
    """
    One log-log figure: oracle queries against RMSE, with the conditional and
    end-to-end frontiers, the matched classical Monte Carlo line, and the most
    expensive run shown at both its conditional and end-to-end error.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib not installed; skipping figure. pip install matplotlib)")
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)

    cq, ce = [f[0] for f in cond], [f[1] for f in cond]
    eq, ee = [f[0] for f in e2e], [f[1] for f in e2e]
    mq, me = [c["mc_samples"] for c in comparison], [c["error"] for c in comparison]

    ax.plot(mq, me, ls="--", color="0.45", lw=1.4, marker="s", ms=4,
            label="classical Monte Carlo, matched grid (samples)")
    ax.plot(cq, ce, color="#1f77b4", lw=1.8, marker="o", ms=5,
            label="QAE, RMSE vs encoded grid (conditional)")
    ax.plot(eq, ee, color="#d95f02", lw=1.8, marker="D", ms=5,
            label="QAE, RMSE vs analytic payoff (end-to-end)")

    # label end-to-end frontier points by qubit count: adjacent points with
    # nearly equal query cost can differ by an order of magnitude in error
    # because their encoded grids differ, and that is the point of the frontier
    for i, (q, err, r) in enumerate(e2e):
        dy = 5 if i % 2 == 0 else -11
        ax.annotate(f"nq={r['num_qubits']}", xy=(q, err), xytext=(6, dy),
                    textcoords="offset points", fontsize=7.5, color="#d95f02")

    # the most expensive run: on the conditional frontier, off the end-to-end one
    dq = deepest["queries"]
    dc = float(np.hypot(deepest["bias"], deepest["sd"]))
    ax.plot([dq, dq], [dc, deepest_e2e], ls=":", color="#d95f02", lw=1.2)
    ax.plot([dq], [deepest_e2e], marker="D", ms=7, mfc="white", mec="#d95f02",
            mew=1.6, ls="none")
    ax.annotate(f"same run vs analytic\nRMSE {deepest_e2e:.3f}",
                xy=(dq, deepest_e2e), xytext=(-12, 22), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color="#d95f02",
                arrowprops=dict(arrowstyle="-", color="#d95f02", lw=0.8))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("oracle queries (QAE)  /  samples (Monte Carlo)")
    ax.set_ylabel("RMSE of expected payoff")
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    ax.set_title("Oracle-query cost against RMSE: conditional and end-to-end frontiers",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="sweep CSV (default: results/validation.csv, falling back "
                         "to results/reference/validation.csv)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else first_existing(
        RESULTS / "validation.csv", RESULTS / "reference" / "validation.csv")
    rows = load(csv_path)
    eps_levels = sorted({r["epsilon_target"] for r in rows})
    c_levels = sorted({r["rescaling_factor"] for r in rows}, reverse=True)
    # Reference epsilon for the bias and spread fits: the tightest level that
    # is present for at least two rescaling factors. A level measured at only
    # one c cannot support a fit, however precise it is.
    coverage = {e: len({r["rescaling_factor"] for r in rows
                        if r["epsilon_target"] == e}) for e in eps_levels}
    usable = [e for e in eps_levels if coverage[e] >= 2]
    tight = min(usable) if usable else min(eps_levels)

    print("=" * 70)
    print("1. SYSTEMATIC OFFSET vs RESCALING FACTOR")
    print("=" * 70)
    print(f"   Measured at eps={tight:g}, the tightest level covering "
          f"{coverage[tight]} rescaling factors.")
    print("   At looser epsilon the spread swamps the mean; at tighter epsilon")
    print("   the sweep only probes one c, which cannot support a fit.")
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
        print("   The c-dependence is consistent with the expected quadratic payoff-")
        print("   linearisation error. This experiment does not separately identify")
        print("   finite-sampling estimator bias, which the offset may also contain.")

    print()
    print("=" * 70)
    print("2. SPREAD vs EPSILON AND RESCALING FACTOR")
    print("=" * 70)
    print("   The signal encoded in the amplitude is proportional to c, so a fixed")
    print("   amplitude precision maps to a payoff error that grows as c shrinks.")
    print()
    print(f"   {'c':>6}" + "".join(f"{'eps=' + format(e, 'g'):>14}" for e in eps_levels)
          + f"{'sd*c @' + format(tight, 'g'):>14}")
    for c in c_levels:
        cells = []
        for e in eps_levels:
            vals = [r["sd"] for r in rows
                    if r["rescaling_factor"] == c and r["epsilon_target"] == e]
            cells.append(statistics.fmean(vals) if vals else float("nan"))
        at_tight = [r["sd"] for r in rows
                    if r["rescaling_factor"] == c and r["epsilon_target"] == tight]
        prod = statistics.fmean(at_tight) * c if at_tight else float("nan")
        print(f"   {c:>6}" + "".join(f"{v:>14.4f}" for v in cells) + f"{prod:>14.4f}")
    print("\n   Spread increases as c decreases. The last column is not constant")
    print("   enough to support a clean sd ~ eps/c law from these data.")

    summary_path = first_existing(RESULTS / "summary.json",
                                  RESULTS / "reference" / "summary.json")
    analytic = None
    if summary_path.exists():
        analytic = json.loads(summary_path.read_text())["expected_payoff_analytic"]

    def build_frontier(err_fn):
        pts = sorted((r["queries"], err_fn(r), r) for r in rows)
        front, best = [], float("inf")
        for q, err, r in pts:
            if err < best:
                best = err
                front.append((q, err, r))
        return front

    def show(front, label):
        print(f"   {'queries':>10} {'RMSE':>11}   configuration")
        for q, err, r in front:
            print(f"   {q:>10,.0f} {err:>11.4f}   nq={r['num_qubits']}, "
                  f"c={r['rescaling_factor']}, eps={r['epsilon_target']:g}")
        if len(front) >= 3:
            e, _ = powerlaw([f[0] for f in front], [f[1] for f in front])
            print(f"\n   fit: error ~ N^{e:.2f}   i.e.   N ~ error^{1 / e:.2f}")
            return e
        print(f"\n   Too few points on the {label} frontier to fit.")
        return None

    print()
    print("=" * 70)
    print("3a. CONDITIONAL FRONTIER  (RMSE relative to the encoded grid)")
    print("=" * 70)
    print("   Estimated RMSE = sqrt(bias^2 + sd^2) against the exact expectation")
    print("   ON the truncated, discretised distribution. This removes each grid's")
    print("   encoding error from the metric. The mixed frontier below can still")
    print("   change nq between points; a fixed-nq fit is reported separately.")
    print()
    cond = build_frontier(lambda r: float(np.hypot(r["bias"], r["sd"])))
    err_exp = show(cond, "conditional")
    if err_exp is None:
        return
    # the tightest-precision point is the least replicated and most influential;
    # report the fit with and without it
    if len(cond) >= 4:
        e_wo, _ = powerlaw([f[0] for f in cond[:-1]], [f[1] for f in cond[:-1]])
        print(f"   without the last point:  error ~ N^{e_wo:.2f}   i.e.   N ~ error^{1 / e_wo:.2f}")
        print("   The endpoint moves the exponent noticeably; treat the fit as indicative.")

    # the mixed frontier hops between qubit counts, so the distribution grid is
    # not held fixed along it (c still varies, as the trade-off requires). Repeat
    # at the one nq that carries the eps=1e-4 probe, so the scaling argument is
    # made on a single grid.
    fixed_nq = max((r["num_qubits"] for r in rows if r["epsilon_target"] < 1e-3),
                   default=None)
    fixed_exp = None
    if fixed_nq is not None:
        pts = sorted((r["queries"], float(np.hypot(r["bias"], r["sd"])), r)
                     for r in rows if r["num_qubits"] == fixed_nq)
        fixed = []; best = float("inf")
        for q, err, r in pts:
            if err < best:
                fixed.append((q, err, r)); best = err
        if len(fixed) >= 3:
            fixed_exp, _ = powerlaw([f[0] for f in fixed], [f[1] for f in fixed])
            print()
            print(f"   Same fit at fixed nq={fixed_nq} only ({len(fixed)} frontier points),")
            print(f"   so the distribution grid does not change along the curve:")
            print(f"   error ~ N^{fixed_exp:.3f}   i.e.   N ~ error^{1 / fixed_exp:.2f}")
            print("   This is the cleaner basis for comparison with the O(M^-2/3) rate;")
            print("   the mixed-nq frontier above is a best-achieved envelope.")
    print()
    print("   Reference points: ideal amplitude estimation N ~ error^-1.0;")
    print("   classical Monte Carlo N ~ error^-2.0.")
    print()
    print("   The exponent sits between them for a known reason. Minimising")
    print("   bias ~ c^2 against sd ~ eps/c gives c ~ eps^(1/3) and total error")
    print("   ~ eps^(2/3); with N ~ 1/eps that is N ~ error^-1.5. Woerner & Egger")
    print("   (2019) derive this rate analytically as O(M^-2/3) convergence for")
    print("   the lowest-depth payoff encoding. The fit here is a small-sample")
    print("   observation qualitatively consistent with it, not a confirmation")
    print("   of the asymptotic exponent.")

    if analytic is None:
        print("\n   (summary.json missing; skipping the end-to-end frontier)")
        return

    print()
    print("=" * 70)
    print("3b. END-TO-END FRONTIER  (RMSE relative to analytic payoff)")
    print("=" * 70)
    print("   Truncation and discretisation included: the total error relative")
    print("   to the analytic expected payoff.")
    print()
    e2e = build_frontier(
        lambda r: float(np.hypot(abs(r["mean"] - analytic), r["sd"])))
    show(e2e, "end-to-end")

    cond_best = min(f[1] for f in cond)
    e2e_best = min(f[1] for f in e2e)
    deepest = max(rows, key=lambda r: r["queries"])
    deepest_e2e = float(np.hypot(abs(deepest["mean"] - analytic), deepest["sd"]))
    enc = abs(deepest["grid"] - analytic)
    print()
    print("   The two frontiers reach different minima:")
    print(f"     best conditional error : {cond_best:.4f}")
    print(f"     best end-to-end error  : {e2e_best:.4f}")
    print()
    print(f"   The most expensive configuration in the sweep "
          f"(nq={deepest['num_qubits']}, eps={deepest['epsilon_target']:g},")
    print(f"   {deepest['queries']:,.0f} queries) reaches {deepest['bias']:.4f} against its own grid")
    print(f"   but {deepest_e2e:.4f} against the analytic payoff, because that grid is itself")
    best_e2e = min(e2e, key=lambda f: f[1])
    ratio = deepest["queries"] / best_e2e[0]
    print(f"   {enc:.4f} away from the true value. It does not appear on the")
    print(f"   end-to-end frontier: a configuration using {ratio:.0f}x fewer queries has")
    print("   lower end-to-end error.")
    print()
    print(f"   For that run, encoding error at nq={deepest['num_qubits']} is the")
    print("   dominant limitation. The sweep does not include eps=1e-4 at a qubit")
    print("   count where encoding error is small, so where the end-to-end frontier")
    print("   would eventually floor is not established here.")

    print()
    print("=" * 70)
    print("4. QUERY-SCALING COMPARISON AGAINST CLASSICAL MONTE CARLO")
    print("=" * 70)
    print("   Matched target: classical Monte Carlo is costed against the SAME")
    print("   truncated, discretised distribution amplitude estimation encodes,")
    print("   using that grid's exact payoff variance. Comparing against the")
    print("   untruncated lognormal would score the two methods on different")
    print("   quantities.")
    print()
    print(f"   {'RMSE':>10} {'nq':>3} {'grid sigma':>11} {'QAE queries':>13} "
          f"{'MC samples':>12} {'ratio':>8}")
    comparison = []
    for q, err, r in cond:
        sig = grid_payoff_sigma(r["num_qubits"])
        n_cl = (sig / err) ** 2
        comparison.append({"queries": q, "error": err,
                           "num_qubits": r["num_qubits"], "grid_sigma": sig,
                           "mc_samples": n_cl, "ratio": q / n_cl})
        print(f"   {err:>10.4f} {r['num_qubits']:>3} {sig:>11.4f} {q:>13,.0f} "
              f"{n_cl:>12,.0f} {q / n_cl:>7.1f}x")
    print()
    lo, hi = min(c["ratio"] for c in comparison), max(c["ratio"] for c in comparison)
    print("   Amplitude estimation needs more oracle queries than Monte Carlo")
    print(f"   needs samples at every budget reachable in simulation, by a factor")
    print(f"   of roughly {lo:.0f}x to {hi:.0f}x across the tested range.")
    print()
    print("   No end-to-end crossover is estimated from this sweep. It does not")
    print("   extend to tight epsilon at qubit counts where encoding error is")
    print("   small, so the end-to-end frontier's eventual floor is not measured")
    print("   and extrapolating a crossing point would not be meaningful.")

    (RESULTS / "frontiers.json").write_text(json.dumps({
        "analytic": analytic,
        "conditional_frontier": [{"queries": q, "error": e,
                                  "num_qubits": r["num_qubits"],
                                  "rescaling_factor": r["rescaling_factor"],
                                  "epsilon_target": r["epsilon_target"]}
                                 for q, e, r in cond],
        "conditional_exponent": err_exp,
        "conditional_exponent_fixed_nq": fixed_exp,
        "conditional_fixed_nq": fixed_nq,
        "end_to_end_frontier": [{"queries": q, "error": e,
                                 "num_qubits": r["num_qubits"],
                                 "rescaling_factor": r["rescaling_factor"],
                                 "epsilon_target": r["epsilon_target"]}
                                for q, e, r in e2e],
        "classical_comparison": comparison,
    }, indent=2))
    print(f"\n   Wrote frontiers.json to results/")
    if plot_frontiers(cond, e2e, comparison, deepest, deepest_e2e, RESULTS / "frontier.png"):
        print("   Wrote frontier.png to results/")


if __name__ == "__main__":
    main()
