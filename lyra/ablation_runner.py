"""
Ablation study: V-only vs. ΔV-only vs. Lyra [V, ΔV].

Tests the contribution of each token in the Lyra 2-token protocol:

  Condition   Signal          WARN revision   Tokens
  ---------   ------          -------------   ------
  none        —               no              0
  binary      {0,1}           no              1
  v_only      V               no              1   ← ablation
  dv_only     ΔV              no              1   ← ablation
  lyra        [V, ΔV]         yes             2

Key questions:
  Q1: Does V alone approach Lyra performance?        (v_only vs lyra)
  Q2: Does ΔV alone approach Lyra performance?       (dv_only vs lyra)
  Q3: Is the combination strictly necessary?          (if lyra >> both → yes)

Run from backend/:
    python -m experiment.lyra.ablation_runner
    python -m experiment.lyra.ablation_runner --runs 30 --episodes 200
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats as sc

from experiment.lyra.compiler import compute_V
from experiment.lyra.verifier import verify, safety_reward, Verdict
from experiment.lyra.runner import (
    ResourceEnv, QAgent,
    N_GREEDY, N_LEARNERS, N_ACTIONS, PROSOCIAL,
    EPSILON_START, EPSILON_END, LR, GAMMA,
)
from experiment.stats import summarise, rank_biserial_r, mean

log = logging.getLogger("lyra.ablation")

CONDITIONS = ["none", "binary", "v_only", "dv_only", "lyra_no_rev", "lyra"]


# ── Episode runner (reuses runner.py logic, ablation-aware) ───────────────────

@dataclass
class EpisodeStats:
    violations:    int   = 0
    actions_taken: int   = 0
    prosocial:     int   = 0
    tokens:        int   = 0


def run_episode(env, agents, condition, epsilon,
                n_steps=50, alpha=1.0, beta=0.5) -> EpisodeStats:
    stats = EpisodeStats()
    env.reset()
    recent = [2] * N_GREEDY + [3] * N_LEARNERS

    for _ in range(n_steps):
        for peer in range(N_GREEDY):
            env.step(peer, 2)
            recent[peer] = 2

        for li, agent in enumerate(agents):
            i        = N_GREEDY + li
            pressure = sum(1 for j, a in enumerate(recent)
                           if j != i and a == 2) / max(N_GREEDY + N_LEARNERS - 1, 1)
            x = env.get_state(i, pressure)
            a = agent.choose(x, epsilon)

            result = verify(x, a, condition)
            stats.tokens += result.tokens

            # WARN revision only for full Lyra (not lyra_no_rev)
            if condition == "lyra" and result.verdict == Verdict.WARN:
                if result.delta_V > 0.0:
                    alts     = [j for j in range(N_ACTIONS) if j != a]
                    best_alt = min(alts, key=lambda j: compute_V(x, j))
                    if compute_V(x, best_alt) < result.V:
                        a = best_alt

            # BLOCK → hold for all blocking conditions
            if result.verdict == Verdict.BLOCK and condition not in ("none",):
                a = 3

            is_viol = (a == 2)
            env.compliance_history[i].append(0 if is_viol else 1)
            if is_viol:
                stats.violations += 1

            r_task = env.step(i, a)
            r_safe = safety_reward(result, condition, alpha, beta)
            agent.update(x, a, r_task + r_safe, env.get_state(i, pressure))

            stats.actions_taken += 1
            stats.prosocial     += 1 if a in PROSOCIAL else 0
            recent[i]            = a

    return stats


# ── Single run ────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    condition:      str
    run_id:         int
    violation_rate: float
    prosocial_rate: float
    tokens_per_step: float


def run_condition(condition, run_id, n_episodes=200, n_steps=50,
                  seed=None, alpha=1.0, beta=0.5) -> RunResult:
    rng    = random.Random(seed)
    env    = ResourceEnv(seed=seed)
    agents = [QAgent(seed=rng.randint(0, 10**6)) for _ in range(N_LEARNERS)]

    total_viols = total_acts = total_pro = total_tok = 0

    for ep in range(n_episodes):
        eps = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
            -3.0 * ep / n_episodes)
        s = run_episode(env, agents, condition, eps, n_steps, alpha, beta)
        total_viols += s.violations
        total_acts  += s.actions_taken
        total_pro   += s.prosocial
        total_tok   += s.tokens

    return RunResult(
        condition       = condition,
        run_id          = run_id,
        violation_rate  = total_viols / max(total_acts, 1),
        prosocial_rate  = total_pro   / max(total_acts, 1),
        tokens_per_step = total_tok   / max(total_acts, 1),
    )


# ── Full experiment ────────────────────────────────────────────────────────────

def run_ablation(n_runs=30, n_episodes=200, n_steps=50,
                 alpha=1.0, beta=0.5, seed=42) -> dict:
    import time
    rng   = random.Random(seed)
    runs  = {c: [] for c in CONDITIONS}
    total = len(CONDITIONS) * n_runs
    done  = 0
    t0    = time.time()

    for cond in CONDITIONS:
        log.info("  Condition %s ...", cond)
        for r in range(n_runs):
            run_seed = rng.randint(0, 10**9)
            result   = run_condition(cond, r, n_episodes, n_steps,
                                     seed=run_seed, alpha=alpha, beta=beta)
            runs[cond].append(result)
            done += 1
            eta = (time.time() - t0) / done * (total - done)
            if (r + 1) % 10 == 0 or n_runs <= 10:
                log.info("    run %d/%d  viol=%.1f%%  ETA %.0fs",
                         r + 1, n_runs, result.violation_rate * 100, eta)

    return runs


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(runs: dict) -> None:
    lyra_vr = [r.violation_rate for r in runs["lyra"]]

    print()
    print("=" * 72)
    print("Ablation Study: V-only vs ΔV-only vs Lyra [V,ΔV]")
    print("  (all conditions: same BLOCK redirect, no WARN revision except lyra)")
    print("=" * 72)
    header = "%-12s  %9s  %7s  %10s  %8s" % (
        "Condition", "ViolRate", "±SD", "Prosocial", "Tokens")
    print(header)
    print("-" * 72)

    LABELS = {
        "none":        "None (A)",
        "binary":      "Binary (C)",
        "v_only":      "V-only (F)",
        "dv_only":     "dV-only (G)",
        "lyra_no_rev": "Lyra no-rev (H)",
        "lyra":        "Lyra [V,dV] (E)",
    }

    for cond in CONDITIONS:
        vr  = [r.violation_rate  for r in runs[cond]]
        pro = [r.prosocial_rate  for r in runs[cond]]
        tok = [r.tokens_per_step for r in runs[cond]]
        mark = "**" if cond == "lyra" else "  "
        print("%s%-10s%s  %7.2f%%  %5.2f%%  %8.1f%%  %7.1f" % (
            mark, LABELS[cond], mark,
            mean(vr)*100, (sum((v-mean(vr))**2 for v in vr)/(len(vr)-1))**0.5*100,
            mean(pro)*100, mean(tok)))

    print()
    print("Hypothesis tests (Lyra vs. ablation conditions):")
    print("%-22s  %9s  %7s  %14s  %s" % ("Comparison", "delta", "r", "p", "sig"))
    print("-" * 72)

    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    for cond in CONDITIONS:
        if cond == "lyra":
            continue
        base_vr = [r.violation_rate for r in runs[cond]]
        _, p    = sc.mannwhitneyu(lyra_vr, base_vr, alternative="two-sided")
        r_rb    = rank_biserial_r(lyra_vr, base_vr)
        delta   = (mean(base_vr) - mean(lyra_vr)) * 100
        print("  Lyra vs %-10s  %+8.2fpp  %+6.4f  %-14s  %s" % (
            LABELS[cond], delta, r_rb, "< 0.001" if p < 0.001 else "%.4f" % p,
            stars(p)))

    # Key ablation finding
    none_vr   = [r.violation_rate for r in runs["none"]]
    vonly_vr  = [r.violation_rate for r in runs["v_only"]]
    dvonly_vr = [r.violation_rate for r in runs["dv_only"]]

    print()
    print("Ablation summary:")
    print("  None    → %.2f%%  (no signal)" % (mean(none_vr)*100))
    print("  V-only  → %.2f%%  (Δ from None: %+.2fpp)" % (
        mean(vonly_vr)*100, (mean(none_vr)-mean(vonly_vr))*100))
    print("  ΔV-only → %.2f%%  (Δ from None: %+.2fpp)" % (
        mean(dvonly_vr)*100, (mean(none_vr)-mean(dvonly_vr))*100))
    print("  Lyra    → %.2f%%  (Δ from None: %+.2fpp)" % (
        mean(lyra_vr)*100, (mean(none_vr)-mean(lyra_vr))*100))
    print()
    print("  V contribution:  %+.2fpp  (None→V-only)" % (
        (mean(none_vr)-mean(vonly_vr))*100))
    print("  ΔV contribution: %+.2fpp  (None→ΔV-only)" % (
        (mean(none_vr)-mean(dvonly_vr))*100))
    print("  Synergy:         %+.2fpp  (Lyra > V-only + ΔV-only)" % (
        (mean(none_vr)-mean(lyra_vr))*100
        - (mean(none_vr)-mean(vonly_vr))*100
        - (mean(none_vr)-mean(dvonly_vr))*100))


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(runs: dict) -> str:
    out_dir = Path(__file__).parent.parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"ablation_{ts}.json"
    data = {
        "experiment": "lyra_ablation",
        "conditions": CONDITIONS,
        "runs": {
            cond: [asdict(r) for r in rs]
            for cond, rs in runs.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Saved -> %s", path)
    return str(path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",     type=int, default=30)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps",    type=int, default=50)
    parser.add_argument("--alpha",    type=float, default=1.0)
    parser.add_argument("--beta",     type=float, default=0.5)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    log.info("Ablation  --  %d runs x %d ep x %d steps  |  %d conditions",
             args.runs, args.episodes, args.steps, len(CONDITIONS))

    runs = run_ablation(args.runs, args.episodes, args.steps,
                        args.alpha, args.beta, args.seed)
    print_report(runs)
    path = save_results(runs)
    print("Saved ->", path)


if __name__ == "__main__":
    main()
