"""
CartSafe Experiment — Lyra-DRL on CartPole with Safety Constraints.

Standard CartPole-v1 + 3-action extension (push_left / push_right / hold).
Safety constraint: don't push the pole further from vertical when it's already
in the danger zone (|angle| > 0.08 rad).

Reproduces the 5-condition comparison from the resource-sharing experiment
on a well-known benchmark, demonstrating WARN-zone generalisation.

Conditions:
  A  none     — no verification signal
  B  z3_text  — simulated Z3 trace (~487 tokens/step)
  C  binary   — 0/1 BLOCK flag
  D  pmcs     — P(violation) ∈ [0,1]
  E  lyra     — [V(x_t), ΔV_t] (2 scalars)

Metrics per condition:
  viol_rate    — fraction of steps that are directional safety violations
  episode_len  — mean episode length (task performance)
  warn_frac    — fraction of violation steps in WARN zone (τ_low < V < τ_high)
                 measures how much binary blocking would have missed

Run from backend/:
    python -m experiment.lyra.cartsafe_runner
"""
from __future__ import annotations

import math
import json
import uuid
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats as stats

from experiment.lyra.cartsafe_env import CartSafeEnv, extract_state, TAU_LOW, TAU_HIGH
from experiment.lyra.auto_compiler  import compile_yaml
from experiment.lyra.dqn_agent      import DQNAgent

# ── Config ─────────────────────────────────────────────────────────────────────

N_RUNS      = 20
N_EPISODES  = 300
MAX_STEPS   = 200      # episode length cap
EPS_START   = 1.0
EPS_END     = 0.05
EPS_DECAY   = 0.97     # per-episode multiplicative decay

ALPHA       = 1.0      # V penalty weight
BETA        = 0.5      # ΔV penalty weight

CONDITIONS  = ["none", "z3_text", "binary", "pmcs", "lyra"]

_HERE       = Path(__file__).parent
_CS_YAML    = _HERE / "constraints" / "cartsafe.yaml"


# ── Verification helper (CartSafe-specific thresholds) ─────────────────────────

def _verify_cartsafe(x: np.ndarray, a_idx: int, compiled, condition: str):
    """
    Thin wrapper around the compiled V for CartSafe.
    Returns (V, delta_V, verdict, tokens, is_blocked).
    """
    V      = compiled.compute_V(x, a_idx)
    V_hold = compiled.compute_V(x, 2)        # hold = action 2
    dV     = V - V_hold

    if V < TAU_LOW:
        verdict = "safe"
    elif V < TAU_HIGH:
        verdict = "warn"
    else:
        verdict = "block"

    blocked = False

    if condition == "none":
        r_safety = 0.0
        tokens   = 0

    elif condition == "z3_text":
        bin_sig  = 0 if verdict == "block" else 1
        r_safety = float(bin_sig - 1)        # 0 or -1
        tokens   = 487
        if verdict == "block":
            blocked = True

    elif condition == "binary":
        bin_sig  = 0 if verdict == "block" else 1
        r_safety = float(bin_sig - 1)
        tokens   = 1
        if verdict == "block":
            blocked = True

    elif condition == "pmcs":
        def _sig(z):
            return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else \
                   math.exp(z) / (1.0 + math.exp(z))
        pmcs = _sig(10.0 * (V - TAU_LOW))
        r_safety = -pmcs
        tokens   = 1

    elif condition == "lyra":
        r_safety = -ALPHA * V - BETA * dV
        tokens   = 2
        if verdict == "block":
            blocked = True

    else:
        raise ValueError(condition)

    return V, dV, verdict, tokens, r_safety, blocked


# ── Single run ─────────────────────────────────────────────────────────────────

def run_condition(condition: str, run_id: int, compiled,
                  seed: Optional[int] = None) -> dict:
    rng   = random.Random(seed)
    agent = DQNAgent(state_dim=4, n_actions=3, seed=seed)
    env   = CartSafeEnv(max_steps=MAX_STEPS)

    ep_results = []
    eps        = EPS_START

    for ep in range(N_EPISODES):
        raw_obs, _ = env.reset(seed=rng.randint(0, 2**31))
        x          = extract_state(raw_obs)
        ep_len     = 0
        ep_viols   = 0
        ep_warn    = 0    # warn-zone violation count

        for _ in range(MAX_STEPS):
            a = agent.choose(x, eps)

            # ── Verification ────────────────────────────────────────────────
            V, dV, verdict, tokens, r_safety, blocked = _verify_cartsafe(
                x, a, compiled, condition
            )

            if blocked:
                a = 2   # redirect to hold

            # ── Step ────────────────────────────────────────────────────────
            raw_next, r_task, terminated, truncated, _ = env.step(a)
            x_next = extract_state(raw_next)

            # Record violation BEFORE blocking (measures what agent INTENDED)
            is_viol = env.is_violation(x, a)
            in_warn = TAU_LOW < V < TAU_HIGH

            if is_viol:
                ep_viols += 1
                if in_warn:
                    ep_warn += 1

            r_total = r_task + r_safety
            agent.update(x, a, r_total, x_next)

            x       = x_next
            ep_len += 1

            if terminated or truncated:
                break

        ep_results.append({
            "ep_len":    ep_len,
            "viol_rate": ep_viols / ep_len if ep_len > 0 else 0.0,
            "warn_frac": ep_warn  / ep_viols if ep_viols > 0 else 0.0,
        })
        eps = max(EPS_END, eps * EPS_DECAY)

    # Aggregate last 100 episodes (post-learning)
    tail = ep_results[-100:]
    return {
        "condition": condition,
        "run_id":    run_id,
        "ep_results": ep_results,
        "viol_rate":  float(np.mean([r["viol_rate"] for r in tail])),
        "ep_len":     float(np.mean([r["ep_len"]    for r in tail])),
        "warn_frac":  float(np.mean([r["warn_frac"] for r in tail])),
    }


# ── Multi-run orchestrator ─────────────────────────────────────────────────────

def run_all(n_runs: int = N_RUNS, n_episodes: int = N_EPISODES) -> dict:
    compiled = compile_yaml(str(_CS_YAML))
    results  = {c: [] for c in CONDITIONS}

    total = len(CONDITIONS) * n_runs
    done  = 0
    t0    = time.time()

    for cond in CONDITIONS:
        print(f"\n=== Condition: {cond} ===")
        for run in range(n_runs):
            seed = 42 + run * 100
            r    = run_condition(cond, run, compiled, seed=seed)
            results[cond].append(r)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  run {run+1:2d}/{n_runs}  "
                  f"viol={r['viol_rate']:.1%}  "
                  f"eplen={r['ep_len']:.0f}  "
                  f"warn_frac={r['warn_frac']:.1%}  "
                  f"ETA {eta:.0f}s")

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    print("\n" + "=" * 72)
    print("CartSafe Results (DQN, CartPole+3actions, mean over last 100 eps)")
    print("=" * 72)
    header = f"{'Cond':8}  {'ViolRate':>9}  {'±SD':>7}  {'EpLen':>7}  {'WarnFrac':>9}  {'Tokens':>7}"
    print(header)
    print("-" * 72)

    TOKEN_MAP = {"none": 0, "z3_text": 487, "binary": 1, "pmcs": 1, "lyra": 2}

    lyra_viols = [r["viol_rate"] for r in results["lyra"]]

    for cond in CONDITIONS:
        viols = [r["viol_rate"] for r in results[cond]]
        lens  = [r["ep_len"]    for r in results[cond]]
        warns = [r["warn_frac"] for r in results[cond]]

        viol_mean = np.mean(viols) * 100
        viol_sd   = np.std(viols)  * 100
        len_mean  = np.mean(lens)
        warn_mean = np.mean(warns) * 100
        tokens    = TOKEN_MAP[cond]

        bold = "**" if cond == "lyra" else "  "
        print(f"{bold}{cond:6}{bold}  {viol_mean:8.2f}%  "
              f"{viol_sd:6.2f}%  {len_mean:7.1f}  "
              f"{warn_mean:8.1f}%  {tokens:7d}")

    print("-" * 72)

    # Hypothesis tests: Lyra vs. each baseline
    print("\nHypothesis tests (Lyra vs. each condition), Mann-Whitney U:")
    for cond in CONDITIONS:
        if cond == "lyra":
            continue
        base_viols = [r["viol_rate"] for r in results[cond]]
        u_stat, p = stats.mannwhitneyu(lyra_viols, base_viols,
                                       alternative="two-sided")
        delta = (np.mean(base_viols) - np.mean(lyra_viols)) * 100
        # rank-biserial r: r = (U_yx - U_xy) / (n1 * n2), bounded in [-1, 1]
        n1, n2 = len(lyra_viols), len(base_viols)
        u_xy = sum(1 if xi > yj else 0.5 if xi == yj else 0
                   for xi in lyra_viols for yj in base_viols)
        u_yx = n1 * n2 - u_xy
        r_rb = (u_yx - u_xy) / (n1 * n2)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  Lyra vs {cond:8s}: Δ={delta:+.2f}pp  r={r_rb:+.3f}  p={p:.4f} {sig}")

    # WARN zone analysis
    print("\nWARN zone analysis (fraction of violations that binary would have missed):")
    none_warns = [r["warn_frac"] for r in results["none"]]
    print(f"  None condition: {np.mean(none_warns)*100:.1f}% of violation steps in WARN zone")
    print(f"  (these are states τ_low < V < τ_high where binary BLOCK never fires)")


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(results: dict) -> str:
    out_dir = Path(__file__).parent.parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    run_id = str(uuid.uuid4())[:8]
    fname  = out_dir / f"cartsafe_{run_id}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {fname}")
    return str(fname)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import scipy  # noqa — verify import before long run
    print(f"CartSafe Experiment")
    print(f"  {N_RUNS} runs × {N_EPISODES} episodes × {MAX_STEPS} steps")
    print(f"  5 conditions × DQN(4→128→64→3)")
    print(f"  τ_low={TAU_LOW}  τ_high={TAU_HIGH}")
    print()

    results = run_all()
    print_report(results)
    save_results(results)
