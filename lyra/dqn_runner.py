"""
DQN-based Lyra-DRL Comparative Experiment Runner.

Upgrades tabular Q-learning to Deep Q-Network (PyTorch).
Same 5-condition structure as runner.py; reuses ResourceEnv and run_episode.

Usage:
    python -m experiment.lyra.dqn_runner
    python -m experiment.lyra.dqn_runner --runs 20 --episodes 150
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from experiment.lyra.dqn_agent import DQNAgent
from experiment.lyra.runner import (
    ResourceEnv, EpisodeStats, RunResult,
    CONDITIONS, N_LEARNERS, EPSILON_START, EPSILON_END,
    CONVERGENCE_WINDOW, CONVERGENCE_THRESH,
    run_episode,          # identical interface — DQNAgent is a drop-in
    save_result, print_report,
)
from experiment.stats import summarise, compare_conditions, mean

log = logging.getLogger("lyra.dqn_runner")


# ── Full condition run (DQN) ───────────────────────────────────────────────────

def run_condition_dqn(
    condition:   str,
    run_id:      int,
    n_episodes:  int   = 150,
    n_steps:     int   = 50,
    seed:        Optional[int] = None,
    alpha:       float = 1.0,
    beta:        float = 0.5,
) -> RunResult:
    rng    = random.Random(seed)
    env    = ResourceEnv(seed=seed)
    agents = [DQNAgent(seed=rng.randint(0, 10**6)) for _ in range(N_LEARNERS)]

    total_violations  = 0
    total_actions     = 0
    total_prosocial   = 0
    total_task_reward = 0.0
    total_tokens      = 0
    ep_violations     = []
    convergence_ep    = None
    rolling_vrate: list[float] = []

    for ep in range(n_episodes):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
            -3.0 * ep / n_episodes
        )
        stats = run_episode(env, agents, condition, epsilon, n_steps, alpha, beta)

        ep_vrate = stats.violations / max(stats.actions_taken, 1)
        ep_violations.append(stats.violations)
        total_violations  += stats.violations
        total_actions     += stats.actions_taken
        total_prosocial   += stats.prosocial
        total_task_reward += stats.task_reward
        total_tokens      += stats.tokens_used

        rolling_vrate.append(ep_vrate)
        if len(rolling_vrate) > CONVERGENCE_WINDOW:
            rolling_vrate.pop(0)
        if (convergence_ep is None
                and len(rolling_vrate) == CONVERGENCE_WINDOW
                and mean(rolling_vrate) < CONVERGENCE_THRESH):
            convergence_ep = ep

    return RunResult(
        condition          = condition,
        run_id             = run_id,
        violation_rate     = total_violations / max(total_actions, 1),
        prosocial_rate     = total_prosocial  / max(total_actions, 1),
        task_reward        = total_task_reward / max(n_episodes, 1),
        tokens_per_step    = total_tokens / max(total_actions, 1),
        convergence_ep     = convergence_ep,
        episode_violations = ep_violations,
    )


# ── Full experiment (DQN) ──────────────────────────────────────────────────────

def run_experiment_dqn(
    n_runs:     int   = 20,
    n_episodes: int   = 150,
    n_steps:    int   = 50,
    alpha:      float = 1.0,
    beta:       float = 0.5,
    seed:       int   = 42,
    conditions: list | None = None,
) -> dict:
    active_conds = conditions if conditions is not None else CONDITIONS
    log.info("Lyra-DRL (DQN) experiment -- %d runs x %d episodes x %d conditions",
             n_runs, n_episodes, len(active_conds))

    all_runs: dict[str, list[RunResult]] = {c: [] for c in active_conds}
    rng = random.Random(seed)

    for cond in active_conds:
        log.info("  Condition %s ...", cond)
        for r in range(n_runs):
            run_seed = rng.randint(0, 10**9)
            result   = run_condition_dqn(cond, r, n_episodes, n_steps,
                                         seed=run_seed, alpha=alpha, beta=beta)
            all_runs[cond].append(result)
            if (r + 1) % 5 == 0:
                log.info("    run %d/%d done", r + 1, n_runs)

    def metric(cond: str, key: str) -> list[float]:
        return [getattr(r, key) for r in all_runs[cond]]

    summary: dict[str, dict] = {}
    for cond in active_conds:
        summary[cond] = {
            "violation_rate":  summarise(metric(cond, "violation_rate")),
            "prosocial_rate":  summarise(metric(cond, "prosocial_rate")),
            "task_reward":     summarise(metric(cond, "task_reward")),
            "tokens_per_step": summarise(metric(cond, "tokens_per_step")),
            "convergence_ep":  summarise([
                r.convergence_ep for r in all_runs[cond]
                if r.convergence_ep is not None
            ]),
        }

    def vrate(cond):
        return metric(cond, "violation_rate")

    comparisons = {}
    if "lyra" in active_conds and "none" in active_conds:
        comparisons["H1_E_vs_A"] = compare_conditions(
            vrate("lyra"), vrate("none"),
            label="H1: Lyra reduces violations vs. no verification")
    if "lyra" in active_conds and "z3_text" in active_conds:
        comparisons["H2_E_vs_B"] = compare_conditions(
            vrate("lyra"), vrate("z3_text"),
            label="H2: Lyra outperforms Z3+Text")
    if "lyra" in active_conds and "binary" in active_conds:
        comparisons["H3_E_vs_C"] = compare_conditions(
            vrate("lyra"), vrate("binary"),
            label="H3: Lyra outperforms binary reward")
    if "lyra" in active_conds and "pmcs" in active_conds:
        comparisons["H4_E_vs_D"] = compare_conditions(
            vrate("lyra"), vrate("pmcs"),
            label="H4: Lyra outperforms PMCS (gradient advantage)")
    if "lyra" in active_conds and "z3_text" in active_conds:
        comparisons["token_E_vs_B"] = {
            "label":         "Token cost: Lyra vs Z3+Text",
            "lyra_mean":     mean(metric("lyra",    "tokens_per_step")),
            "z3text_mean":   mean(metric("z3_text", "tokens_per_step")),
            "reduction_pct": round(
                (mean(metric("z3_text", "tokens_per_step")) -
                 mean(metric("lyra",    "tokens_per_step")))
                / max(mean(metric("z3_text", "tokens_per_step")), 1) * 100, 2),
        }

    return {
        "experiment": "lyra_drl_dqn",
        "agent":      "DQN",
        "config": {
            "n_runs": n_runs, "n_episodes": n_episodes,
            "n_steps": n_steps, "n_agents": 8,
            "alpha": alpha, "beta": beta,
            "conditions": active_conds,
            "dqn": {"hidden": [128, 64], "lr": 1e-3, "gamma": 0.9,
                    "batch": 64, "buffer": 10_000, "target_update": 200},
        },
        "summary":     summary,
        "comparisons": comparisons,
        "runs": {
            cond: [asdict(r) for r in runs]
            for cond, runs in all_runs.items()
        },
    }


# ── Save & report ──────────────────────────────────────────────────────────────

def save_result_dqn(result: dict) -> str:
    out_dir = Path(__file__).parent.parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"lyra_dqn_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="Lyra-DRL DQN experiment")
    parser.add_argument("--runs",       type=int,   default=20)
    parser.add_argument("--episodes",   type=int,   default=150)
    parser.add_argument("--steps",      type=int,   default=50)
    parser.add_argument("--alpha",      type=float, default=1.0)
    parser.add_argument("--beta",       type=float, default=0.5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--conditions", type=str,   default=None,
                        help="Comma-separated subset of conditions to run, "
                             "e.g. 'none,lyra'")
    args = parser.parse_args()

    cond_subset = (
        [c.strip() for c in args.conditions.split(",")]
        if args.conditions else None
    )

    result = run_experiment_dqn(
        n_runs=args.runs, n_episodes=args.episodes,
        n_steps=args.steps, alpha=args.alpha,
        beta=args.beta, seed=args.seed,
        conditions=cond_subset,
    )
    path = save_result_dqn(result)   # save first — print_report may skip missing conds
    print(f"Saved -> {path}")
    print_report(result)


if __name__ == "__main__":
    main()
