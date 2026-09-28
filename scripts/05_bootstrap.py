"""
Bootstrap the frontier exponents and frontier membership over seeds.

02_analyse.py fits RMSE ~ N^b to the Pareto frontier of the sweep. That
frontier is selected from noisy per-configuration RMSE estimates (20 seeds for
most configurations, 5 for the eps=1e-4 probe), so configurations that happen
to draw well can enter the frontier preferentially, and the fitted exponent
inherits that selection. This script resamples seeds with replacement within
each configuration, rebuilds the frontiers and refits the exponents on every
resample, and reports percentile intervals for the exponents and how often
each configuration makes each frontier.

The configuration grid itself is fixed by design and is not resampled; the
intervals describe seed-to-seed variability under this sweep, not variability
over sweeps. The aggregation, frontier rule and fit mirror 02_analyse.py and
must be kept in sync with it.

Usage:
    python scripts/05_bootstrap.py
    python scripts/05_bootstrap.py --resamples 5000 --seed 1
    python scripts/05_bootstrap.py --csv results/validation.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[-1]


def load_seeds(csv_path: Path) -> dict:
    """Per-configuration arrays of (estimate, oracle_queries) plus the exact grid value."""
    groups = defaultdict(list)
    grid = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["num_qubits"]), float(r["rescaling_factor"]),
                   float(r["epsilon_target"]))
            groups[key].append((float(r["estimate"]), float(r["oracle_queries"])))
            grid[key] = float(r["grid_exact"])
    out = {}
    for key in sorted(groups):
        arr = np.asarray(groups[key], float)
        out[key] = {"est": arr[:, 0], "queries": arr[:, 1], "grid": grid[key]}
    return out


def aggregate(cfg: dict, idx: np.ndarray | None = None) -> dict:
    """One record per configuration, as 02_analyse.load() computes it."""
    est = cfg["est"] if idx is None else cfg["est"][idx]
    q = cfg["queries"] if idx is None else cfg["queries"][idx]
    mean = float(est.mean())
    return {
        "mean": mean,
        "bias": abs(mean - cfg["grid"]),
        "sd": float(est.std(ddof=1)) if len(est) > 1 else 0.0,
        "queries": float(q.mean()),
        "grid": cfg["grid"],
    }


def powerlaw_exponent(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    if keep.sum() < 3:
        return None
    b, _ = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return float(b)


def build_frontier(records: dict, err_fn) -> list:
    """Lower-left envelope: sorted by queries, keep points whose error strictly improves."""
    pts = sorted((rec["queries"], err_fn(rec), key) for key, rec in records.items())
    front, best = [], float("inf")
    for q, err, key in pts:
        if err < best:
            best = err
            front.append((q, err, key))
    return front


def fits(records: dict, analytic: float | None, fixed_nq: int | None) -> dict:
    cond_err = lambda r: float(np.hypot(r["bias"], r["sd"]))
    cond = build_frontier(records, cond_err)
    out = {
        "conditional": powerlaw_exponent([f[0] for f in cond], [f[1] for f in cond]),
        "conditional_without_last": (
            powerlaw_exponent([f[0] for f in cond[:-1]], [f[1] for f in cond[:-1]])
            if len(cond) >= 4 else None),
        "members_conditional": [f[2] for f in cond],
    }
    if fixed_nq is not None:
        sub = {k: r for k, r in records.items() if k[0] == fixed_nq}
        fx = build_frontier(sub, cond_err)
        out["conditional_fixed_nq"] = powerlaw_exponent([f[0] for f in fx], [f[1] for f in fx])
    if analytic is not None:
        e2e = build_frontier(records, lambda r: float(np.hypot(abs(r["mean"] - analytic), r["sd"])))
        out["end_to_end"] = powerlaw_exponent([f[0] for f in e2e], [f[1] for f in e2e])
        out["members_end_to_end"] = [f[2] for f in e2e]
        out["best_end_to_end"] = min(e2e, key=lambda f: f[1])[2]
    return out


def summarise(name: str, point, samples: list, level: float) -> dict:
    vals = np.asarray([s for s in samples if s is not None], float)
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    d = {"point": point, "n_fits": int(len(vals)), "n_resamples": len(samples)}
    if len(vals):
        d.update({"ci_low": float(np.percentile(vals, lo)),
                  "ci_high": float(np.percentile(vals, hi)),
                  "median": float(np.median(vals))})
    return d


def fmt_key(k) -> str:
    return f"nq={k[0]}, c={k[1]:g}, eps={k[2]:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=95.0, help="percentile interval, in percent")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else first_existing(
        RESULTS / "validation.csv", RESULTS / "reference" / "validation.csv")
    cfgs = load_seeds(csv_path)
    summary_path = first_existing(RESULTS / "summary.json", RESULTS / "reference" / "summary.json")
    analytic = json.loads(summary_path.read_text())["expected_payoff_analytic"] if summary_path.exists() else None
    fixed_nq = max((k[0] for k in cfgs if k[2] < 1e-3), default=None)

    point = fits({k: aggregate(c) for k, c in cfgs.items()}, analytic, fixed_nq)

    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.resamples):
        recs = {}
        for k, c in cfgs.items():
            n = len(c["est"])
            recs[k] = aggregate(c, rng.integers(0, n, size=n))
        draws.append(fits(recs, analytic, fixed_nq))

    exps = [e for e in ("conditional", "conditional_without_last", "conditional_fixed_nq", "end_to_end")
            if e in point]
    report = {
        "csv": str(csv_path.relative_to(ROOT)), "resamples": args.resamples, "seed": args.seed,
        "level": args.level, "fixed_nq": fixed_nq,
        "n_seeds_per_config": {fmt_key(k): int(len(c["est"])) for k, c in cfgs.items()},
        "exponents": {e: summarise(e, point[e], [d.get(e) for d in draws], args.level) for e in exps},
    }
    for which in ("conditional", "end_to_end"):
        key = f"members_{which}"
        if key not in point:
            continue
        counts = defaultdict(int)
        for d in draws:
            for k in d[key]:
                counts[k] += 1
        report[f"membership_{which}"] = {
            fmt_key(k): {"frequency": counts[k] / args.resamples, "on_point_frontier": k in point[key]}
            for k in sorted(cfgs, key=lambda k: -counts[k])}
    if "best_end_to_end" in point:
        counts = defaultdict(int)
        for d in draws:
            counts[d["best_end_to_end"]] += 1
        report["best_end_to_end"] = {
            "point": fmt_key(point["best_end_to_end"]),
            "frequency": {fmt_key(k): v / args.resamples for k, v in
                          sorted(counts.items(), key=lambda kv: -kv[1])}}

    print("=" * 70)
    print(f"BOOTSTRAP OVER SEEDS  ({args.resamples} resamples, seed {args.seed}, {args.level:g}% percentile intervals)")
    print("=" * 70)
    print(f"   source: {report['csv']}   configurations: {len(cfgs)}   fixed nq: {fixed_nq}")
    print(f"   seeds per configuration: {sorted(set(report['n_seeds_per_config'].values()))}")
    print()
    print(f"   {'exponent':<28}{'point':>8}{'median':>9}{'interval':>20}{'fits':>8}")
    for e, s in report["exponents"].items():
        pt = f"{s['point']:.3f}" if s["point"] is not None else "  n/a"
        if "ci_low" in s:
            iv = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
            print(f"   {e:<28}{pt:>8}{s['median']:>9.3f}{iv:>20}{s['n_fits']:>5}/{s['n_resamples']}")
        else:
            print(f"   {e:<28}{pt:>8}{'':>9}{'no fit possible':>20}")
    print()
    print("   Reference exponents: ideal AE -1.0; lowest-depth encoding rate -0.667;")
    print("   classical Monte Carlo -0.5. In query terms N ~ error^(1/b).")
    for which in ("conditional", "end_to_end"):
        key = f"membership_{which}"
        if key not in report:
            continue
        print()
        print(f"   Frontier membership, {which.replace('_', '-')}: fraction of resamples in which")
        print(f"   each configuration is on the frontier (* = on the point-estimate frontier)")
        for k, v in report[key].items():
            if v["frequency"] >= 0.01 or v["on_point_frontier"]:
                mark = "*" if v["on_point_frontier"] else " "
                print(f"     {mark} {v['frequency']:>6.1%}   {k}")
    if "best_end_to_end" in report:
        print()
        print("   Lowest end-to-end error configuration across resamples:")
        for k, v in list(report["best_end_to_end"]["frequency"].items())[:5]:
            print(f"       {v:>6.1%}   {k}")

    out = RESULTS / "bootstrap.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n   written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
