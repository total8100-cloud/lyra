"""
Lyra-DRL GridNav Experiment Runner.

Tabular Q-learning on an 8×8 grid navigation task under three conditions:
  A — none    (no verification)
  C — binary  (0/1 block signal)
  E — lyra    (Lyapunov [V, ΔV] 2-token protocol)

The experiment mirrors the resource-sharing tabular runner but uses a
spatial navigation domain compiled purely from gridnav.yaml — no Python
constraint code was added.

Usage:
    python -m experiment.lyra.gridnav_runner
    python -m experiment.lyra.gridnav_runner --runs 10 --episodes 100
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.lyra.auto_compiler import compile_yaml
from experiment.lyra.gridnav_env import (
    GridNavEnv, VIOLATION_CELLS, HAZARDS, N_ACTIONS, extract_state, GRID_SIZE
)
from experiment.lyra.verifier import verify, safety_reward, Verdict
from experiment.stats import summarise, compare_conditions

log = logging.getLogger("lyra.gridnav")

# ── Experiment constants ───────────────────────────────────────────────────────

CONDITIONS    = ["none", "binary", "lyra"]
YAML_PATH     = Path(__file__).parent / "constraints" / "gridnav.yaml"
RESULTS_DIR   = Path(__file__).parent.parent.parent / "results"

EPSILON_START = 1.0
EPSILON_END   = 0.30
LR            = 0.1
GAMMA         = 0.90

# Lyapunov thresholds (default values, same as resource-sharing)
TAU_LOW  = 0.05
TAU_HIGH = 0.30


# ── Q-learning agent ───────────────────────────────────────────────────────────

class TabularAgent:
    """
    Tabular Q-learning agent for 8×8 grid navigation.
    State key: (x, y)  →  Q-values for 5 actions.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.Q:   dict[tuple[int, int], np.ndarray] = {}
        self._rng = random.Random(seed)

    def _q(self, state: tuple[int, int]) -> np.ndarray:
        if state not in self.Q:
            self.Q[state] = np.zeros(N_ACTIONS, dtype=np.float64)
        return self.Q[state]

    def choose_action(self, state: tuple[int, int], epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return self._rng.randrange(N_ACTIONS)
        return int(np.argmax(self._q(state)))

    def update(
        self,
        state:      tuple[int, int],
        action:     int,
        reward:     float,
        next_state: tuple[int, int],
        done:       bool,
    ) -> None:
        q    = self._q(state)
        q_n  = self._q(next_state)
        td   = reward + (0.0 if done else GAMMA * q_n.max()) - q[action]
        q[action] += LR * td


# ── Single run ─────────────────────────────────────────────────────────────────

def _run_one(
    condition: str,
    n_episodes: int,
    compiled,
    seed: int,
) -> dict:
    env   = GridNavEnv(max_steps=100, seed=seed)
    agent = TabularAgent(seed=seed)
    eps   = EPSILON_START

    ep_viol_rates: list[float] = []
    ep_rewards:    list[float] = []
    total_steps    = 0
    total_viols    = 0

    for ep in range(n_episodes):
        pos    = env.reset()
        done   = False
        ep_v   = 0
        ep_s   = 0
        ep_r   = 0.0
        eps_ep = max(EPSILON_END, EPSILON_START - ep * (EPSILON_START - EPSILON_END) / max(1, n_episodes - 1))

        while not done:
            action  = agent.choose_action(pos, eps_ep)
            x_state = env.get_verifier_state()

            # ── Verification ────────────────────────────────────────────────
            result = verify(
                x         = x_state,
                a_idx     = action,
                condition = condition,
                tau_low   = TAU_LOW,
                tau_high  = TAU_HIGH,
                compiled  = compiled,
            )

            # WARN-zone action revision (Lyra only)
            if (condition == "lyra"
                    and result.verdict == Verdict.WARN
                    and result.delta_V > 0):
                V_current = result.V
                best_a    = action
                best_V    = V_current
                for alt in range(N_ACTIONS):
                    if alt == action:
                        continue
                    V_alt = compiled.compute_V(x_state, alt)
                    if V_alt < best_V:
                        best_V = V_alt
                        best_a = alt
                if best_a != action:
                    action = best_a
                    result = verify(
                        x=x_state, a_idx=action, condition=condition,
                        tau_low=TAU_LOW, tau_high=TAU_HIGH, compiled=compiled,
                    )

            # BLOCK: override to wait (action 3)
            if result.verdict == Verdict.BLOCK and condition != "none":
                action = 3   # wait = safe hold action

            # Violation = movement action that lands in a violation cell.
            # Wait (action 3) and blocked-then-wait are not counted as violations:
            # the agent is penalised for navigating INTO dangerous territory,
            # not for standing still after being blocked.
            is_viol = (action != 3) and env.is_violation(action)
            if is_viol:
                ep_v += 1

            r_safe = safety_reward(result, condition)
            new_pos, r_task, done = env.step(action)
            total_r = r_task + r_safe

            agent.update(pos, action, total_r, new_pos, done)
            pos     = new_pos
            ep_r   += r_task   # log task reward only
            ep_s   += 1

        ep_viol_rates.append(ep_v / ep_s if ep_s > 0 else 0.0)
        ep_rewards.append(ep_r)
        total_steps += ep_s
        total_viols += ep_v
        eps          = max(EPSILON_END, eps - (EPSILON_START - EPSILON_END) / max(1, n_episodes - 1))

    overall_viol = total_viols / total_steps if total_steps > 0 else 0.0
    return {
        "condition":          condition,
        "violation_rate":     overall_viol,
        "ep_violation_rates": ep_viol_rates,
        "ep_rewards":         ep_rewards,
        "total_steps":        total_steps,
    }


# ── WARN-zone occupancy measurement ───────────────────────────────────────────

def _measure_warn_zone(compiled, n_samples: int = 5000, seed: int = 42) -> dict:
    """
    Random-policy WARN-zone occupancy for the GridNav domain.
    Mirrors the resource-sharing warn-zone analysis (Table tab:warn-zone).
    """
    rng   = np.random.default_rng(seed)
    safe_ = warn_ = block_ = 0

    MOVE_ACTIONS = [0, 1, 2, 4]   # north, south, east, west — exclude wait
    for _ in range(n_samples):
        x   = int(rng.integers(0, GRID_SIZE))
        y   = int(rng.integers(0, GRID_SIZE))
        act = int(rng.choice(MOVE_ACTIONS))
        xst = extract_state(x, y)
        V   = compiled.compute_V(xst, act)
        if V >= TAU_HIGH:
            block_ += 1
        elif V >= TAU_LOW:
            warn_ += 1
        else:
            safe_ += 1

    total = safe_ + warn_ + block_
    return {
        "safe_frac":  safe_  / total,
        "warn_frac":  warn_  / total,
        "block_frac": block_ / total,
        "n_samples":  total,
        "violation_cells": len(VIOLATION_CELLS),
        "grid_cells":      GRID_SIZE ** 2,
    }


# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(
    n_runs:     int = 30,
    n_episodes: int = 200,
    seed_base:  int = 42,
) -> dict:

    compiled = compile_yaml(str(YAML_PATH))
    log.info("GridNav YAML compiled: %d constraints", len(compiled.constraint_names()))

    results: dict[str, list[float]] = {c: [] for c in CONDITIONS}

    for run in range(n_runs):
        seed = seed_base + run
        for cond in CONDITIONS:
            r = _run_one(cond, n_episodes, compiled, seed)
            results[cond].append(r["violation_rate"])
            log.info(
                "run %02d %-8s  viol=%.4f",
                run + 1, cond, r["violation_rate"],
            )

    # ── Statistics ──────────────────────────────────────────────────────────
    summaries = {c: summarise(results[c]) for c in CONDITIONS}
    comparisons = {
        "none_vs_lyra":   compare_conditions(results["none"],   results["lyra"],   "none_vs_lyra"),
        "binary_vs_lyra": compare_conditions(results["binary"], results["lyra"],   "binary_vs_lyra"),
        "none_vs_binary": compare_conditions(results["none"],   results["binary"], "none_vs_binary"),
    }

    # ── WARN-zone measurement ────────────────────────────────────────────────
    warn_zone = _measure_warn_zone(compiled)
    log.info(
        "WARN zone: SAFE=%.1f%%  WARN=%.1f%%  BLOCK=%.1f%%",
        warn_zone["safe_frac"]  * 100,
        warn_zone["warn_frac"]  * 100,
        warn_zone["block_frac"] * 100,
    )

    return {
        "meta": {
            "domain":     "grid_nav",
            "n_runs":     n_runs,
            "n_episodes": n_episodes,
            "yaml":       str(YAML_PATH),
            "tau_low":    TAU_LOW,
            "tau_high":   TAU_HIGH,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        },
        "runs":        {c: results[c] for c in CONDITIONS},
        "summaries":   summaries,
        "comparisons": {k: v for k, v in comparisons.items()},
        "warn_zone":   warn_zone,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="GridNav Lyra experiment")
    parser.add_argument("--runs",     type=int, default=30)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    data = run_experiment(
        n_runs     = args.runs,
        n_episodes = args.episodes,
        seed_base  = args.seed,
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n=== GridNav Results ===")
    for cond in CONDITIONS:
        s = data["summaries"][cond]
        print(f"  {cond:<10}  mean={s['mean']:.4f}  std={s['std']:.4f}")
    print()
    for key, comp in data["comparisons"].items():
        r = comp["rank_biserial_r"]
        p = comp["mann_whitney"]["p_value"]
        print(f"  {key:<25}  r={r:+.2f}  p={p:.3e}")
    wz = data["warn_zone"]
    print(f"\n  WARN zone (random policy, n={wz['n_samples']:,}):")
    print(f"    SAFE  {wz['safe_frac']*100:.1f}%  WARN  {wz['warn_frac']*100:.1f}%  BLOCK  {wz['block_frac']*100:.1f}%")
    print(f"    Violation cells: {wz['violation_cells']}/{wz['grid_cells']} ({wz['violation_cells']/wz['grid_cells']*100:.1f}%)")

    # ── Save results ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"gridnav_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    main()
