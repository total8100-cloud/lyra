"""
Multi-Domain Experiment Runner.

Runs all 4 domains (financial, medical, power, cyber) with 4 conditions
(none, z3_text, cbf, lyra) and aggregates results into a cross-domain
consistency report.

RQ: "Does the same YAML-configured Lyapunov pipeline generalize across domains?"

Usage:
    python -m experiment.lyra.multi_domain_runner
    python -m experiment.lyra.multi_domain_runner --runs 10 --episodes 100
    python -m experiment.lyra.multi_domain_runner --domains financial medical
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from experiment.lyra.domain_runner import (
    run_domain_experiment,
    save_result,
    CONDITIONS,
    CONDITIONS_STRESS,
    DOMAIN_CONFIG,
)
from experiment.stats import mean

log = logging.getLogger("lyra.multi_domain")


# ---- Cross-domain consistency metrics ----------------------------------------

def compute_cross_domain_summary(all_results: dict[str, dict],
                                  scenario: str = "default") -> dict:
    """
    Aggregates violation-rate reduction across all domains.

    Key metric: cross-domain mean and std of Lyra vs None reduction.
    For stress scenario, also aggregates Lyra_ext vs None and Lyra vs Lyra_ext.
    """
    domains    = list(all_results.keys())
    reductions = []
    reductions_ext = []

    rows = []
    for domain, result in all_results.items():
        s        = result["summary"]
        none_vr  = s["none"]["violation_rate"]["mean"]
        lyra_vr  = s["lyra"]["violation_rate"]["mean"]
        cbf_vr   = s.get("cbf",  {}).get("violation_rate", {}).get("mean", None)
        z3_vr    = s.get("z3_text", {}).get("violation_rate", {}).get("mean", None)
        ext_vr   = s.get("lyra_ext", {}).get("violation_rate", {}).get("mean", None)
        reduction = (none_vr - lyra_vr) / max(none_vr, 1e-8) * 100.0

        z3_tps   = s.get("z3_text",  {}).get("tokens_per_step", {}).get("mean", 0)
        cbf_tps  = s.get("cbf",      {}).get("tokens_per_step", {}).get("mean", 0)
        lyra_tps = s["lyra"]["tokens_per_step"]["mean"]
        token_ratio_z3   = z3_tps   / max(lyra_tps, 1e-8)
        token_ratio_cbf  = cbf_tps  / max(lyra_tps, 1e-8)

        reductions.append(reduction)
        row = {
            "domain":           domain,
            "none_vrate":       round(none_vr * 100, 2),
            "z3_vrate":         round(z3_vr  * 100, 2) if z3_vr  is not None else None,
            "cbf_vrate":        round(cbf_vr * 100, 2) if cbf_vr is not None else None,
            "lyra_vrate":       round(lyra_vr * 100, 2),
            "reduction_pct":    round(reduction, 1),
            "z3_tokens":        round(z3_tps,  1),
            "cbf_tokens":       round(cbf_tps, 1),
            "lyra_tokens":      round(lyra_tps, 1),
            "token_ratio_z3":   round(token_ratio_z3,  1),
            "token_ratio_cbf":  round(token_ratio_cbf, 2),
        }
        if ext_vr is not None:
            ext_reduction = (none_vr - ext_vr) / max(none_vr, 1e-8) * 100.0
            reductions_ext.append(ext_reduction)
            row["lyra_ext_vrate"]    = round(ext_vr * 100, 2)
            row["ext_reduction_pct"] = round(ext_reduction, 1)
        rows.append(row)

    cross = {
        "domains":             domains,
        "scenario":            scenario,
        "per_domain":          rows,
        "mean_reduction_pct":  round(mean(reductions), 1),
        "min_reduction_pct":   round(min(reductions), 1),
        "max_reduction_pct":   round(max(reductions), 1),
        "all_positive":        all(r > 0 for r in reductions),
    }
    if reductions_ext:
        cross["ext_mean_reduction_pct"] = round(mean(reductions_ext), 1)
        cross["ext_all_positive"]        = all(r > 0 for r in reductions_ext)
    return cross


# ---- Report printer -----------------------------------------------------------

def print_report(all_results: dict[str, dict], cross: dict) -> None:
    scenario = cross.get("scenario", "default")
    sep = "=" * 90
    print("\n" + sep)
    print(f"Lyra Multi-Domain Experiment  [scenario={scenario}]")
    print("RQ: Same YAML->compiler pipeline -- consistent across regulatory domains?")
    print(sep)

    has_ext = any(row.get("lyra_ext_vrate") is not None for row in cross["per_domain"])

    if has_ext:
        header = (f"  {'Domain':<12}  {'None%':>6}  {'Z3%':>6}  {'CBF%':>6}  "
                  f"{'Lyra%':>6}  {'Ext%':>6}  {'Reduc%':>7}  {'ExtRed%':>8}")
    else:
        header = (f"  {'Domain':<12}  {'None%':>6}  {'Z3%':>6}  {'CBF%':>6}  "
                  f"{'Lyra%':>6}  {'Reduc%':>7}  {'Z3-tok':>7}  {'CBF-tok':>8}  {'Lyra-tok':>9}")
    print(header)
    print("-" * (90 if has_ext else 80))

    for row in cross["per_domain"]:
        z3v  = f"{row['z3_vrate']:6.2f}"  if row['z3_vrate']  is not None else "  n/a "
        cbfv = f"{row['cbf_vrate']:6.2f}" if row['cbf_vrate'] is not None else "  n/a "
        if has_ext:
            extv = f"{row['lyra_ext_vrate']:6.2f}" if row.get('lyra_ext_vrate') is not None else "  n/a "
            extred = f"{row['ext_reduction_pct']:8.1f}" if row.get('ext_reduction_pct') is not None else "     n/a"
            print(f"  {row['domain']:<12}  "
                  f"{row['none_vrate']:6.2f}  "
                  f"{z3v}  "
                  f"{cbfv}  "
                  f"{row['lyra_vrate']:6.2f}  "
                  f"{extv}  "
                  f"{row['reduction_pct']:7.1f}  "
                  f"{extred}")
        else:
            print(f"  {row['domain']:<12}  "
                  f"{row['none_vrate']:6.2f}  "
                  f"{z3v}  "
                  f"{cbfv}  "
                  f"{row['lyra_vrate']:6.2f}  "
                  f"{row['reduction_pct']:7.1f}  "
                  f"{row['z3_tokens']:7.1f}  "
                  f"{row['cbf_tokens']:8.2f}  "
                  f"{row['lyra_tokens']:9.2f}")

    print("-" * (90 if has_ext else 80))
    print(f"\n  Cross-domain violation reduction (Lyra vs None):")
    print(f"    Mean  = {cross['mean_reduction_pct']:.1f}%")
    print(f"    Range = [{cross['min_reduction_pct']:.1f}%, "
          f"{cross['max_reduction_pct']:.1f}%]")
    print(f"    All domains positive: {cross['all_positive']}")

    if "ext_mean_reduction_pct" in cross:
        print(f"\n  Cross-domain reduction (Lyra-ext/external guardrail vs None):")
        print(f"    Mean  = {cross['ext_mean_reduction_pct']:.1f}%")
        print(f"    All domains positive: {cross['ext_all_positive']}")

    print("\n" + sep)
    print("Per-domain hypothesis tests\n")

    hyp_keys = [
        "H1_lyra_vs_none",
        "H2_lyra_vs_z3",
        "H3_lyra_vs_cbf",
        "H4_cbf_vs_none",
        "H5_lyraext_vs_none",
        "H6_lyra_vs_lyraext",
    ]
    for domain, result in all_results.items():
        print(f"  [{domain.upper()}]")
        for key in hyp_keys:
            c  = result["comparisons"].get(key, {})
            if not c:
                continue
            mw = c.get("mann_whitney", {})
            print(f"    {c.get('label','')}")
            print(f"      delta={c.get('delta', 0)*100:+.2f}%  "
                  f"p={mw.get('p_str', 'n/a')}  "
                  f"d={c.get('cohen_d', 0):.3f}  "
                  f"({c.get('effect_size', 'n/a')})")
        print()


# ---- Main --------------------------------------------------------------------

def run_all(domains: list[str], n_runs: int = 30, n_episodes: int = 200,
            n_steps: int = 50, seed: int = 42,
            scenario: str = "default") -> dict:
    """
    scenario: "default" — standard random initial conditions
              "stress"  — near-violation initial conditions (adds lyra_ext condition)
    """
    conditions = CONDITIONS_STRESS if scenario == "stress" else CONDITIONS
    all_results: dict[str, dict] = {}
    for domain in domains:
        log.info("Running domain: %s  [scenario=%s]", domain, scenario)
        result = run_domain_experiment(
            domain, n_runs=n_runs, n_episodes=n_episodes,
            n_steps=n_steps, seed=seed, scenario=scenario,
        )
        all_results[domain] = result
        path = save_result(result)
        log.info("  Saved -> %s", path)

    cross = compute_cross_domain_summary(all_results, scenario=scenario)

    combined = {
        "experiment":  "lyra_multi_domain",
        "scenario":    scenario,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "config":      {"n_runs": n_runs, "n_episodes": n_episodes,
                        "n_steps": n_steps, "domains": domains,
                        "conditions": conditions},
        "cross_domain": cross,
        "domains":     all_results,
    }
    return combined


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="Lyra multi-domain experiment")
    parser.add_argument("--domains",  nargs="+",
                        default=list(DOMAIN_CONFIG.keys()),
                        choices=list(DOMAIN_CONFIG.keys()))
    parser.add_argument("--runs",     type=int, default=30)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps",    type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--scenario", choices=["default", "stress"],
                        default="default",
                        help="default: standard reset; stress: near-violation initial conditions")
    args = parser.parse_args()

    result = run_all(
        domains=args.domains,
        n_runs=args.runs,
        n_episodes=args.episodes,
        n_steps=args.steps,
        seed=args.seed,
        scenario=args.scenario,
    )

    cross = result["cross_domain"]
    print_report({d: result["domains"][d] for d in args.domains}, cross)

    out_dir = Path(__file__).parent.parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"lyra_multi_domain_{args.scenario}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved -> {path}")


if __name__ == "__main__":
    main()
