"""
Lyra-DRL Comparative Experiment Runner.

5-condition, 30-run experiment comparing verification mechanisms:
  A — none      (no verification)
  B — z3_text   (Z3 SMT + natural-language trace, ~500 tokens)
  C — binary    (0/1 reward signal)
  D — pmcs      (P(violation) scalar)
  E — lyra      (Lyapunov [V, ΔV] 2-token protocol)

Usage:
    python -m experiment.lyra.runner
    python -m experiment.lyra.runner --runs 10 --episodes 100
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from experiment.lyra.compiler import compute_V, compute_grad_V
from experiment.lyra.verifier import verify, safety_reward, Verdict
from experiment.stats import summarise, compare_conditions, mean

log = logging.getLogger("lyra.runner")

# ── Experiment constants ───────────────────────────────────────────────────────

CONDITIONS   = ["none", "z3_text", "binary", "pmcs", "lyra"]
N_AGENTS     = 8
N_GREEDY     = 4    # hardcoded always-take "peer" agents (social pressure)
N_LEARNERS   = N_AGENTS - N_GREEDY   # Q-learning subject agents
SCARCITY     = 0.70     # total resources = SCARCITY × (N_AGENTS × 100)
ACTIONS      = ["share", "split", "take", "hold"]
N_ACTIONS    = 4
PROSOCIAL    = {0, 1}   # share, split
EPSILON_START = 1.0
EPSILON_END   = 0.30   # keep exploration high so violations stay visible
LR           = 0.1
GAMMA        = 0.9
CONVERGENCE_WINDOW   = 10
CONVERGENCE_THRESH   = 0.10   # violation rate below 10 % for convergence


# ── Resource environment ───────────────────────────────────────────────────────

class ResourceEnv:
    """
    Minimal multi-agent resource allocation environment.
    8 agents share a scarce resource pool.
    """

    def __init__(self, n_agents: int = N_AGENTS, seed: Optional[int] = None):
        self.n_agents = n_agents
        self.rng      = random.Random(seed)
        self.np_rng   = np.random.default_rng(seed)
        self.resources: np.ndarray = np.zeros(n_agents)
        self.compliance_history: list[list[int]] = [[] for _ in range(n_agents)]
        self.reset()

    def reset(self) -> None:
        total = N_AGENTS * 100 * SCARCITY
        shares = self.np_rng.dirichlet(np.ones(self.n_agents)) * total
        self.resources = np.clip(shares, 5.0, None)
        for h in self.compliance_history:
            h.clear()

    def get_state(self, agent_idx: int, pressure: float) -> np.ndarray:
        total        = self.resources.sum()
        fair_res     = total / self.n_agents if total > 0 else 1.0
        own_res      = self.resources[agent_idx]
        others       = np.delete(self.resources, agent_idx)
        peer_mean_res = others.mean() if len(others) > 0 else fair_res

        # Normalise: 0.5 = fair share, 1.0 = 2x fair share (cap)
        own_norm  = min(own_res      / fair_res, 2.0) / 2.0
        peer_norm = min(peer_mean_res / fair_res, 2.0) / 2.0

        hist   = self.compliance_history[agent_idx]
        comply = mean(hist[-10:]) if hist else 1.0

        return np.array([
            own_norm,          # own_share  ∈ [0,1]: 0.5=fair, 1.0=2× fair
            peer_norm,         # peer_share ∈ [0,1]
            float(pressure),   # social pressure [0,1]
            float(comply),     # compliance history [0,1]
        ])

    def step(self, agent_idx: int, action_idx: int) -> float:
        """
        Apply action and return INDIVIDUAL task reward.

        Reward structure deliberately creates tension between
        individual gain and collective norms (tragedy of commons):
          take  → high individual gain (+0.8)  but violates constraints
          hold  → moderate (+0.2)
          split → small loss from own perspective (-0.1)
          share → larger loss (-0.2)  but socially optimal

        Without a safety signal, greedy Q-learners will converge to 'take'.
        The verification conditions then modulate how much taking is curtailed.
        """
        own    = self.resources[agent_idx]
        others = np.delete(self.resources, agent_idx)
        r_task = 0.0

        if action_idx == 0:   # share — give 10 % to neediest
            give = min(own * 0.10, max(own - 5.0, 0.0))
            if give > 0 and len(others) > 0:
                neediest = int(np.argmin(others))
                real_idx = neediest if neediest < agent_idx else neediest + 1
                self.resources[agent_idx] -= give
                self.resources[real_idx]  += give
            r_task = -0.2   # agent loses resources → costly prosocial

        elif action_idx == 1:  # split — distribute 5 % equally
            give = min(own * 0.05, max((own - 5.0) / self.n_agents, 0.0))
            if give > 0:
                self.resources[agent_idx] -= give * (self.n_agents - 1)
                mask = np.ones(self.n_agents, dtype=bool)
                mask[agent_idx] = False
                self.resources[mask] += give
            r_task = -0.1

        elif action_idx == 2:  # take — steal 15 % from richest
            if len(others) > 0:
                richest_rel = int(np.argmax(others))
                richest     = richest_rel if richest_rel < agent_idx else richest_rel + 1
                steal = min(self.resources[richest] * 0.15,
                            max(self.resources[richest] - 5.0, 0.0))
                if steal > 0:
                    self.resources[richest]   -= steal
                    self.resources[agent_idx] += steal
            r_task = 0.8   # high individual reward → tempts greedy learner

        else:               # hold
            r_task = 0.2

        self.resources = np.clip(self.resources, 0.0, None)
        return float(r_task)

    def social_pressure(self, agent_idx: int, recent_actions: list[int]) -> float:
        """
        Fraction of OTHER agents' recent actions that are antisocial (take=2).
        """
        others = [a for i, a in enumerate(recent_actions) if i != agent_idx]
        if not others:
            return 0.0
        return sum(1 for a in others if a == 2) / len(others)


# ── Q-learning agent ───────────────────────────────────────────────────────────

class QAgent:
    """
    Tabular Q-learning with continuous state → discretised bucket key.
    One shared Q-table per condition run (agents learn together).
    """

    def __init__(self, n_actions: int = N_ACTIONS, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng       = random.Random(seed)
        self._q: dict[str, list[float]] = {}

    def _encode(self, x: np.ndarray) -> str:
        # Discretise each dim into 5 buckets
        buckets = tuple(min(4, int(xi * 5)) for xi in np.clip(x, 0, 0.9999))
        return str(buckets)

    def _qtable(self, key: str) -> list[float]:
        if key not in self._q:
            self._q[key] = [self.rng.uniform(-0.01, 0.01)
                            for _ in range(self.n_actions)]
        return self._q[key]

    def choose(self, x: np.ndarray, epsilon: float) -> int:
        key = self._encode(x)
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        return int(np.argmax(self._qtable(key)))

    def update(self, x: np.ndarray, a: int, reward: float,
               x_next: np.ndarray) -> None:
        key      = self._encode(x)
        next_key = self._encode(x_next)
        q        = self._qtable(key)
        nq       = self._qtable(next_key)
        q[a]    += LR * (reward + GAMMA * max(nq) - q[a])

    def reset(self) -> None:
        self._q.clear()


# ── Single run ─────────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    violations:   int   = 0
    actions_taken: int  = 0
    task_reward:  float = 0.0
    tokens_used:  int   = 0
    prosocial:    int   = 0
    warn_count:   int   = 0
    warn_revised: int   = 0   # WARN → agent revised to safer action


def run_episode(
    env:       ResourceEnv,
    agents:    list[QAgent],   # length = N_LEARNERS (subject agents only)
    condition: str,
    epsilon:   float,
    n_steps:   int = 50,
    alpha:     float = 1.0,
    beta:      float = 0.5,
) -> EpisodeStats:
    """
    Greedy peers (indices 0..N_GREEDY-1) always take.
    Q-learning subjects (indices N_GREEDY..N_AGENTS-1) face the dilemma.
    Violations are counted only for subject agents.
    Social pressure signal reflects the greedy majority.
    """
    stats = EpisodeStats()
    env.reset()
    # recent_actions: greedy peers start with 'take', learners with 'hold'
    recent_actions = [2] * N_GREEDY + [3] * N_LEARNERS

    for _ in range(n_steps):

        # ── Greedy peer actions (hardcoded: always take) ───────────────────
        for peer_idx in range(N_GREEDY):
            env.step(peer_idx, 2)   # take — no learning, no tracking
            recent_actions[peer_idx] = 2

        # ── Learner agent actions (Q-learning, with safety signal) ─────────
        for li, agent in enumerate(agents):
            i        = N_GREEDY + li   # actual index in environment
            pressure = env.social_pressure(i, recent_actions)
            x        = env.get_state(i, pressure)
            a        = agent.choose(x, epsilon)

            # Verify proposed action
            result = verify(x, a, condition)
            stats.tokens_used += result.tokens

            # WARN zone: Lyra agent may revise if action worsens V
            if result.verdict == Verdict.WARN and condition == "lyra":
                stats.warn_count += 1
                if result.delta_V > 0.0:
                    alts     = [j for j in range(N_ACTIONS) if j != a]
                    best_alt = min(alts, key=lambda j: compute_V(x, j))
                    if compute_V(x, best_alt) < result.V:
                        a = best_alt
                        stats.warn_revised += 1

            # BLOCK → force hold
            if result.verdict == Verdict.BLOCK:
                a = 3

            # Violation = learner executes 'take' (antisocial action)
            # V is used for reward shaping; take-rate is the primary DV
            is_violation = (a == 2)
            env.compliance_history[i].append(0 if is_violation else 1)
            if is_violation:
                stats.violations += 1

            # Execute and learn
            r_task = env.step(i, a)
            r_safe = safety_reward(result, condition, alpha, beta)
            if result.verdict in (Verdict.BLOCK, Verdict.WARN):
                r_safe = safety_reward(verify(x, a, condition),
                                       condition, alpha, beta)

            agent.update(x, a, r_task + r_safe, env.get_state(i, pressure))

            stats.task_reward   += r_task
            stats.actions_taken += 1
            stats.prosocial     += 1 if a in PROSOCIAL else 0
            recent_actions[i]    = a

    return stats


# ── Full condition run ─────────────────────────────────────────────────────────

@dataclass
class RunResult:
    condition:        str
    run_id:           int
    violation_rate:   float
    prosocial_rate:   float
    task_reward:      float
    tokens_per_step:  float
    convergence_ep:   Optional[int]     # first episode reaching safe policy
    episode_violations: list[int] = field(default_factory=list)


def run_condition(
    condition:   str,
    run_id:      int,
    n_episodes:  int = 200,
    n_steps:     int = 50,
    seed:        Optional[int] = None,
    alpha:       float = 1.0,
    beta:        float = 0.5,
) -> RunResult:
    rng    = random.Random(seed)
    env    = ResourceEnv(seed=seed)
    agents = [QAgent(seed=rng.randint(0, 10**6)) for _ in range(N_LEARNERS)]

    total_violations  = 0
    total_actions     = 0
    total_prosocial   = 0
    total_task_reward = 0.0
    total_tokens      = 0
    ep_violations     = []
    convergence_ep    = None
    rolling_vrate     = []

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

        # Convergence detection
        rolling_vrate.append(ep_vrate)
        if len(rolling_vrate) > CONVERGENCE_WINDOW:
            rolling_vrate.pop(0)
        if (convergence_ep is None and
                len(rolling_vrate) == CONVERGENCE_WINDOW and
                mean(rolling_vrate) < CONVERGENCE_THRESH):
            convergence_ep = ep

    steps_total = total_actions
    return RunResult(
        condition        = condition,
        run_id           = run_id,
        violation_rate   = total_violations / max(steps_total, 1),
        prosocial_rate   = total_prosocial  / max(steps_total, 1),
        task_reward      = total_task_reward / max(n_episodes, 1),
        tokens_per_step  = total_tokens / max(steps_total, 1),
        convergence_ep   = convergence_ep,
        episode_violations = ep_violations,
    )


# ── Full experiment ────────────────────────────────────────────────────────────

def run_experiment(
    n_runs:    int = 30,
    n_episodes: int = 200,
    n_steps:    int = 50,
    alpha:      float = 1.0,
    beta:       float = 0.5,
    seed:       int = 42,
) -> dict:
    log.info("Lyra experiment -- %d runs x %d episodes x %d conditions",
             n_runs, n_episodes, len(CONDITIONS))

    all_runs: dict[str, list[RunResult]] = {c: [] for c in CONDITIONS}
    rng = random.Random(seed)

    for cond in CONDITIONS:
        log.info("  Condition %s ...", cond)
        for r in range(n_runs):
            run_seed = rng.randint(0, 10**9)
            result   = run_condition(cond, r, n_episodes, n_steps,
                                     seed=run_seed, alpha=alpha, beta=beta)
            all_runs[cond].append(result)
            if (r + 1) % 10 == 0:
                log.info("    run %d/%d done", r + 1, n_runs)

    # ── Aggregate statistics ───────────────────────────────────────────────────
    def metric(cond: str, key: str) -> list[float]:
        return [getattr(r, key) for r in all_runs[cond]]

    summary: dict[str, dict] = {}
    for cond in CONDITIONS:
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

    # ── Hypothesis comparisons ─────────────────────────────────────────────────
    def vrate(cond):
        return metric(cond, "violation_rate")

    comparisons = {
        "H1_E_vs_A": compare_conditions(
            vrate("lyra"), vrate("none"),
            label="H1: Lyra reduces violations vs. no verification"),
        "H2_E_vs_B": compare_conditions(
            vrate("lyra"), vrate("z3_text"),
            label="H2: Lyra outperforms Z3+Text"),
        "H3_E_vs_C": compare_conditions(
            vrate("lyra"), vrate("binary"),
            label="H3: Lyra outperforms binary reward"),
        "H4_E_vs_D": compare_conditions(
            vrate("lyra"), vrate("pmcs"),
            label="H4: Lyra outperforms PMCS (gradient advantage)"),
        "token_E_vs_B": {
            "label": "Token cost: Lyra vs Z3+Text",
            "lyra_mean":    mean(metric("lyra",    "tokens_per_step")),
            "z3text_mean":  mean(metric("z3_text", "tokens_per_step")),
            "reduction_pct": round(
                (mean(metric("z3_text", "tokens_per_step")) -
                 mean(metric("lyra",    "tokens_per_step")))
                / max(mean(metric("z3_text", "tokens_per_step")), 1) * 100, 2),
        },
    }

    result = {
        "experiment": "lyra_drl",
        "config": {
            "n_runs": n_runs, "n_episodes": n_episodes,
            "n_steps": n_steps, "n_agents": N_AGENTS,
            "alpha": alpha, "beta": beta,
            "conditions": CONDITIONS,
        },
        "summary":     summary,
        "comparisons": comparisons,
        "runs": {
            cond: [asdict(r) for r in runs]
            for cond, runs in all_runs.items()
        },
    }
    return result


# ── Save & report ──────────────────────────────────────────────────────────────

def save_result(result: dict) -> str:
    out_dir = Path(__file__).parent.parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"lyra_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(path)


def print_report(result: dict) -> None:
    cfg = result["config"]
    print(f"\nLyra-DRL Experiment  |  "
          f"{cfg['n_runs']} runs x {cfg['n_episodes']} episodes x "
          f"{cfg['n_agents']} agents\n")

    cond_labels = {
        "none":    "A - None",
        "z3_text": "B - Z3+Text",
        "binary":  "C - Binary",
        "pmcs":    "D - PMCS",
        "lyra":    "E - Lyra",
    }

    print("=" * 72)
    print(f"{'Cond':<14} {'ViolRate':>9} {'+-SD':>7} "
          f"{'Prosocial':>10} {'Tokens/step':>12} {'Conv.ep':>8}")
    print("-" * 72)
    present = result["summary"]
    for cond in CONDITIONS:
        if cond not in present:
            continue
        s  = present[cond]
        vr = s["violation_rate"]
        pr = s["prosocial_rate"]
        tk = s["tokens_per_step"]
        cv = s["convergence_ep"]
        conv_str = f"{cv['mean']:.0f}" if cv.get("n", 0) > 0 else "N/A"
        print(f"  {cond_labels[cond]:<12}  "
              f"{vr['mean']*100:7.2f}%  "
              f"{vr['std']*100:5.2f}%  "
              f"{pr['mean']*100:8.1f}%  "
              f"{tk['mean']:10.1f}  "
              f"{conv_str:>8}")

    print("\n" + "=" * 72)
    print("Hypothesis Tests (Mann-Whitney U, Lyra vs. each baseline)\n")

    for key in ["H1_E_vs_A", "H2_E_vs_B", "H3_E_vs_C", "H4_E_vs_D"]:
        if key not in result["comparisons"]:
            continue
        c  = result["comparisons"][key]
        mw = c.get("mann_whitney", {})
        print(f"  {c['label']}")
        print(f"    Δ={c['delta']*100:+.2f}%  "
              f"p={mw.get('p_str','—')}  "
              f"d={c.get('cohen_d',0):.3f}  "
              f"({c.get('effect_size','—')})\n")

    tk_cmp = result["comparisons"]["token_E_vs_B"]
    print(f"Token reduction (E vs B):  "
          f"{tk_cmp['z3text_mean']:.1f} -> {tk_cmp['lyra_mean']:.1f}  "
          f"({tk_cmp['reduction_pct']:.1f}% reduction)\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(message)s")

    parser = argparse.ArgumentParser(description="Lyra-DRL experiment")
    parser.add_argument("--runs",     type=int, default=30)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps",    type=int, default=50)
    parser.add_argument("--alpha",    type=float, default=1.0)
    parser.add_argument("--beta",     type=float, default=0.5)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    result = run_experiment(
        n_runs=args.runs, n_episodes=args.episodes,
        n_steps=args.steps, alpha=args.alpha,
        beta=args.beta, seed=args.seed,
    )
    print_report(result)
    path = save_result(result)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
