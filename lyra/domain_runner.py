"""
Generic Domain Runner for Lyra multi-domain experiments.

Replaces the domain-specific runner.py with a parametric version that
works with any BaseDomainEnv + CompiledConstraints pair.

Usage:
    from experiment.lyra.domain_runner import run_domain_experiment
    result = run_domain_experiment("financial", n_runs=30)

Design mirrors runner.py exactly (same conditions, Q-agent, metrics)
so results are directly comparable across domains.
"""
from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import json

import numpy as np

from experiment.lyra.auto_compiler import compile_yaml
from experiment.lyra.verifier import verify, safety_reward, Verdict
from experiment.lyra.independent_evaluator import get_evaluator, HardEvaluatorBase
from experiment.stats import summarise, compare_conditions, mean

log = logging.getLogger("lyra.domain_runner")

CONDITIONS        = ["none", "z3_text", "hard_shield", "e_only", "lyra"]
CONDITIONS_STRESS = ["none", "z3_text", "hard_shield", "e_only", "lyra", "lyra_ext"]
# Legacy names kept for backward compatibility with saved results
CONDITIONS_LEGACY = ["none", "z3_text", "cbf", "lyra"]

EPSILON_START    = 1.0
EPSILON_END      = 0.30
LR               = 0.10
GAMMA            = 0.90
CONVERGENCE_WIN  = 10
CONVERGENCE_THR  = 0.10

DOMAIN_CONFIG = {
    "financial": {
        "env_class": "experiment.lyra.domains.financial_env.FinancialEnv",
        "yaml":      "experiment/lyra/domains/yamls/financial.yaml",
        "tau_low":  0.003,
        "tau_high": 0.020,
        "alpha": 1.0,
        "beta":  0.5,
    },
    "medical": {
        "env_class": "experiment.lyra.domains.medical_env.MedicalEnv",
        "yaml":      "experiment/lyra/domains/yamls/medical.yaml",
        "tau_low":  0.05,
        "tau_high": 0.30,
        # task_reward ~-1.3, V_typical ~0.15 → safety signal only -0.15 at alpha=1.
        # Boost to ~20% of task_reward magnitude.
        "alpha": 3.0,
        "beta":  1.5,
    },
    "power": {
        "env_class": "experiment.lyra.domains.power_env.PowerEnv",
        "yaml":      "experiment/lyra/domains/yamls/power.yaml",
        "tau_low":  0.003,
        "tau_high": 0.020,
        # task_reward ~28, V_typical ~0.01 → safety signal only -0.01 at alpha=1 (0.04% of task).
        # Boost alpha so safety ≈ 10% of task scale: alpha*0.01 ≈ 2.8 → alpha=250.
        "alpha": 250.0,
        "beta":  100.0,
    },
    "cyber": {
        "env_class": "experiment.lyra.domains.cyber_env.CyberEnv",
        "yaml":      "experiment/lyra/domains/yamls/cyber.yaml",
        "tau_low":  0.05,
        "tau_high": 0.30,
        "alpha": 1.0,
        "beta":  0.5,
    },
}


# ── Dynamic import helper ──────────────────────────────────────────────────────

def _load_env_class(dotpath: str):
    module_path, cls_name = dotpath.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _yaml_path(relative: str) -> str:
    base = Path(__file__).parent.parent.parent   # backend/
    return str(base / relative)


# ── Tabular Q-agent (identical to runner.py) ─────────────────────────────────

class QAgent:
    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self._rng      = random.Random(seed)
        self._q: dict[str, list[float]] = {}

    def _encode(self, x: np.ndarray) -> str:
        buckets = tuple(min(4, int(xi * 5)) for xi in np.clip(x, 0, 0.9999))
        return str(buckets)

    def _qtable(self, key: str) -> list[float]:
        if key not in self._q:
            self._q[key] = [self._rng.uniform(-0.01, 0.01)
                            for _ in range(self.n_actions)]
        return self._q[key]

    def choose(self, x: np.ndarray, epsilon: float) -> int:
        key = self._encode(x)
        if self._rng.random() < epsilon:
            return self._rng.randint(0, self.n_actions - 1)
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


# ── Episode stats ──────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    violations:         int   = 0   # PRIMARY: H_i^act OR H_i^out (union)
    violations_act:     int   = 0   # H_i^act only (action-triggered)
    violations_out:     int   = 0   # H_i^out only (outcome state-level)
    actions_taken:      int   = 0
    task_reward:        float = 0.0
    tokens_used:        int   = 0
    warn_count:         int   = 0
    warn_revised:       int   = 0
    intervention_count: int   = 0
    fallback_count:     int   = 0


# ── Single episode ─────────────────────────────────────────────────────────────

def run_episode(env, agent: QAgent, compiled, condition: str,
                epsilon: float, n_steps: int = 50,
                alpha: float = 1.0, beta: float = 0.5,
                tau_low: float = 0.05, tau_high: float = 0.30,
                stress: bool = False,
                hard_evaluator: HardEvaluatorBase | None = None) -> EpisodeStats:
    stats = EpisodeStats()
    x     = env.reset_stress() if stress else env.reset()
    hold  = env.hold_action

    # lyra_ext uses lyra verification but zero safety reward (external guardrail only)
    verify_cond = "lyra" if condition == "lyra_ext" else condition

    for _ in range(n_steps):
        a_proposed = agent.choose(x, epsilon)
        a          = a_proposed
        result     = verify(x, a, verify_cond, compiled=compiled,
                            tau_low=tau_low, tau_high=tau_high)
        stats.tokens_used += result.tokens

        # ── Action revision (Hard-first, then severity ranking) ──────────────
        _can_H    = hasattr(compiled, "compute_H")
        _h_revised = False

        # STEP 1: Hard-first H_i check (lyra, lyra_ext, hard_shield)
        # Catches H violations at any V level, including SAFE zone (V < τ_low)
        if condition in ("lyra", "lyra_ext", "hard_shield") and _can_H:
            if compiled.compute_H(x, a) < 0.0:
                alts      = [j for j in range(env.n_actions) if j != a]
                compliant = [j for j in alts if compiled.compute_H(x, j) >= 0.0]
                if compliant:
                    a = min(compliant, key=lambda j: compiled.compute_V(x, j))
                else:
                    a = hold
                stats.warn_count  += 1
                stats.warn_revised += 1
                _h_revised = True

        # STEP 2: WARN zone ΔE revision (lyra/lyra_ext, only if not already H-revised)
        if not _h_revised and result.verdict == Verdict.WARN and condition in ("lyra", "lyra_ext"):
            stats.warn_count += 1
            if result.delta_V > 0.0:
                alts = [j for j in range(env.n_actions) if j != a]
                pool = ([j for j in alts if compiled.compute_H(x, j) >= 0.0]
                        if _can_H else alts) or alts
                best_alt = min(pool, key=lambda j: compiled.compute_V(x, j))
                if compiled.compute_V(x, best_alt) < result.V:
                    a = best_alt
                    stats.warn_revised += 1

        # STEP 3: E-only Revision — WARN revision using E, no ΔE trigger, no H filter
        if result.verdict == Verdict.WARN and condition == "e_only":
            stats.warn_count += 1
            alts     = [j for j in range(env.n_actions) if j != a]
            best_alt = min(alts, key=lambda j: compiled.compute_V(x, j))
            if compiled.compute_V(x, best_alt) < result.V:
                a = best_alt
                stats.warn_revised += 1

        # Legacy CBF revision: pick max-H safe alternative before blocking
        if result.verdict == Verdict.BLOCK and condition == "cbf":
            stats.warn_count += 1
            alts     = [j for j in range(env.n_actions) if j != hold]
            best_alt = max(alts, key=lambda j: compiled.compute_H(x, j))
            if compiled.compute_H(x, best_alt) >= 0.0:
                a = best_alt
                stats.warn_revised += 1

        # BLOCK → force hold only if H-first revision did not already handle it
        if result.verdict == Verdict.BLOCK and not _h_revised:
            a = hold

        # Track shield interventions
        if a != a_proposed:
            stats.intervention_count += 1

        # H_i^act — action-triggered constraints on pre-step state
        if hard_evaluator is not None:
            h_act = hard_evaluator.evaluate_act(x, a)
            is_act = h_act.any_violation
        else:
            h_act = None
            is_act = (result.V >= tau_low and a != hold)

        x_next, r_task, done = env.step(a)

        # H_i^out — state-level constraints on outcome state
        if hard_evaluator is not None:
            h_out = hard_evaluator.evaluate_out(x_next)
            is_out = h_out.any_violation
        else:
            is_out = False

        if is_act:
            stats.violations_act += 1
        if is_out:
            stats.violations_out += 1
        if is_act or is_out:
            stats.violations += 1

        # Safety reward: lyra_ext gets zero (no training signal, external guardrail only)
        result_post = verify(x, a, verify_cond, compiled=compiled,
                             tau_low=tau_low, tau_high=tau_high)
        if condition == "lyra_ext":
            r_safe = 0.0
        else:
            r_safe = safety_reward(result_post, condition, alpha, beta)

        agent.update(x, a, r_task + r_safe, x_next)
        stats.task_reward   += r_task
        stats.actions_taken += 1
        x = x_next
        if done:
            break

    return stats


# ── Single condition run ───────────────────────────────────────────────────────

@dataclass
class RunResult:
    domain:              str
    condition:           str
    run_id:              int
    violation_rate:      float   # H_i^act OR H_i^out union
    hvr_act:             float   # H_i^act only (action-triggered)
    hvr_out:             float   # H_i^out only (outcome state-level)
    task_reward:         float
    tokens_per_step:     float
    convergence_ep:      Optional[int]
    episode_violations:  list[int]  = field(default_factory=list)
    intervention_rate:   float      = 0.0
    fallback_rate:       float      = 0.0


def run_condition(domain: str, condition: str, run_id: int,
                  env_class, compiled,
                  n_episodes: int = 200, n_steps: int = 50,
                  seed: Optional[int] = None,
                  alpha: float = 1.0, beta: float = 0.5,
                  tau_low: float = 0.05, tau_high: float = 0.30,
                  scenario: str = "default",
                  hard_evaluator: HardEvaluatorBase | None = None) -> RunResult:

    env   = env_class(seed=seed)
    agent = QAgent(n_actions=env.n_actions, seed=seed)
    stress = (scenario == "stress")

    total_violations     = 0
    total_violations_act = 0
    total_violations_out = 0
    total_actions        = 0
    total_reward         = 0.0
    total_tokens         = 0
    total_interventions  = 0
    total_fallbacks      = 0
    ep_violations        = []
    convergence_ep       = None
    rolling              = []

    for ep in range(n_episodes):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
            -3.0 * ep / n_episodes
        )
        stats = run_episode(env, agent, compiled, condition,
                            epsilon, n_steps, alpha, beta,
                            tau_low=tau_low, tau_high=tau_high,
                            stress=stress,
                            hard_evaluator=hard_evaluator)

        ep_vrate = stats.violations / max(stats.actions_taken, 1)
        ep_violations.append(stats.violations)
        total_violations     += stats.violations
        total_violations_act += stats.violations_act
        total_violations_out += stats.violations_out
        total_actions        += stats.actions_taken
        total_reward         += stats.task_reward
        total_tokens         += stats.tokens_used
        total_interventions  += stats.intervention_count
        total_fallbacks      += stats.fallback_count

        rolling.append(ep_vrate)
        if len(rolling) > CONVERGENCE_WIN:
            rolling.pop(0)
        if (convergence_ep is None and
                len(rolling) == CONVERGENCE_WIN and
                mean(rolling) < CONVERGENCE_THR):
            convergence_ep = ep

    n = max(total_actions, 1)
    return RunResult(
        domain             = domain,
        condition          = condition,
        run_id             = run_id,
        violation_rate     = total_violations     / n,
        hvr_act            = total_violations_act / n,
        hvr_out            = total_violations_out / n,
        task_reward        = total_reward         / max(n_episodes, 1),
        tokens_per_step    = total_tokens         / n,
        convergence_ep     = convergence_ep,
        episode_violations = ep_violations,
        intervention_rate  = total_interventions  / n,
        fallback_rate      = total_fallbacks       / n,
    )


# ── Full domain experiment ─────────────────────────────────────────────────────

def run_domain_experiment(domain: str, n_runs: int = 30, n_episodes: int = 200,
                          n_steps: int = 50, alpha: float | None = None,
                          beta: float | None = None, seed: int = 42,
                          scenario: str = "default") -> dict:
    """
    scenario: "default" — standard random initial conditions (4 conditions)
              "stress"  — near-violation initial conditions (5 conditions + lyra_ext)
    """
    cfg   = DOMAIN_CONFIG[domain]
    EnvCls  = _load_env_class(cfg["env_class"])
    compiled = compile_yaml(_yaml_path(cfg["yaml"]))

    tau_low  = cfg.get("tau_low",  0.05)
    tau_high = cfg.get("tau_high", 0.30)
    alpha = alpha if alpha is not None else cfg.get("alpha", 1.0)
    beta  = beta  if beta  is not None else cfg.get("beta",  0.5)

    conditions = CONDITIONS_STRESS if scenario == "stress" else CONDITIONS
    log.info("[%s/%s] %d runs x %d episodes x %d conditions  tau=[%.4f, %.4f]",
             domain, scenario, n_runs, n_episodes, len(conditions), tau_low, tau_high)

    # Independent hard evaluator — VR measurement decoupled from compiler/E
    try:
        hard_ev = get_evaluator(domain)
        log.info("  [%s] independent hard evaluator loaded: %s",
                 domain, type(hard_ev).__name__)
    except ValueError:
        hard_ev = None
        log.warning("  [%s] no independent evaluator found; "
                    "falling back to legacy V-based VR", domain)

    # Pre-generate shared seeds so run i gets the same environment across all conditions.
    # This enables paired Wilcoxon signed-rank tests (matched-run design).
    rng       = random.Random(seed)
    run_seeds = [rng.randint(0, 10**9) for _ in range(n_runs)]
    all_runs: dict[str, list[RunResult]] = {c: [] for c in conditions}

    for cond in conditions:
        log.info("  [%s/%s] condition %s ...", domain, scenario, cond)
        for r in range(n_runs):
            result = run_condition(domain, cond, r, EnvCls, compiled,
                                   n_episodes, n_steps, run_seeds[r], alpha, beta,
                                   tau_low=tau_low, tau_high=tau_high,
                                   scenario=scenario,
                                   hard_evaluator=hard_ev)
            all_runs[cond].append(result)

    def metric(cond: str, key: str) -> list[float]:
        return [getattr(r, key) for r in all_runs[cond]]

    summary: dict[str, dict] = {}
    for cond in conditions:
        summary[cond] = {
            "violation_rate":    summarise(metric(cond, "violation_rate")),
            "hvr_act":           summarise(metric(cond, "hvr_act")),
            "hvr_out":           summarise(metric(cond, "hvr_out")),
            "task_reward":       summarise(metric(cond, "task_reward")),
            "tokens_per_step":   summarise(metric(cond, "tokens_per_step")),
            "intervention_rate": summarise(metric(cond, "intervention_rate")),
            "fallback_rate":     summarise(metric(cond, "fallback_rate")),
            "convergence_ep":    summarise([
                r.convergence_ep for r in all_runs[cond]
                if r.convergence_ep is not None
            ]),
        }

    def vrate(cond):
        return metric(cond, "violation_rate")

    has = lambda c: c in conditions  # noqa: E731

    comparisons = {
        "H1_lyra_vs_none":  compare_conditions(vrate("lyra"), vrate("none"),
                               label="H1: Runtime Shield reduces HVR vs. None",
                               paired=True),
        "H2_lyra_vs_z3":    compare_conditions(vrate("lyra"), vrate("z3_text"),
                               label="H2: Runtime Shield vs. Z3+Text",
                               paired=True),
        "H3_lyra_vs_hard":  compare_conditions(
                               vrate("lyra"),
                               vrate("hard_shield") if has("hard_shield") else vrate("cbf"),
                               label="H3: Runtime Shield vs. Hard Shield (ablation C)",
                               paired=True),
        "H4_hard_vs_none":  compare_conditions(
                               vrate("hard_shield") if has("hard_shield") else vrate("cbf"),
                               vrate("none"),
                               label="H4: Hard Shield reduces HVR vs. None",
                               paired=True),
        # Ablation D: does ΔE add value beyond E alone?
        "H5_eonly_vs_lyra": (
            compare_conditions(vrate("e_only"), vrate("lyra"),
                               label="H5 (ablation D vs E): E-only vs Runtime Shield",
                               paired=True)
            if has("e_only") else None
        ),
        "token_reduction": {
            "z3_mean":         mean(metric("z3_text",    "tokens_per_step")),
            "hard_shield_mean": mean(metric("hard_shield", "tokens_per_step"))
                                if has("hard_shield") else None,
            "e_only_mean":     mean(metric("e_only",     "tokens_per_step"))
                                if has("e_only") else None,
            "lyra_mean":       mean(metric("lyra",       "tokens_per_step")),
        },
    }

    # Stress-only: internal vs external guardrail separation
    if scenario == "stress" and "lyra_ext" in conditions:
        comparisons["H6_lyraext_vs_none"] = compare_conditions(
            vrate("lyra_ext"), vrate("none"),
            label="H6: Lyra-ext (external guardrail) reduces HVR vs. None",
            paired=True)
        comparisons["H7_lyra_vs_lyraext"] = compare_conditions(
            vrate("lyra"), vrate("lyra_ext"),
            label="H7: Runtime Shield (training-integrated) vs Lyra-ext (external only)",
            paired=True)
        comparisons["token_reduction"]["lyra_ext_mean"] = mean(
            metric("lyra_ext", "tokens_per_step"))

    return {
        "domain":      domain,
        "scenario":    scenario,
        "config":      {"n_runs": n_runs, "n_episodes": n_episodes,
                        "n_steps": n_steps, "conditions": conditions},
        "summary":     summary,
        "comparisons": comparisons,
        "runs": {c: [asdict(r) for r in runs] for c, runs in all_runs.items()},
    }


# ── Save helper ────────────────────────────────────────────────────────────────

def save_result(result: dict, out_dir: Optional[str] = None) -> str:
    if out_dir is None:
        out_dir = str(Path(__file__).parent.parent.parent / "results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    scenario = result.get("scenario", "default")
    path     = Path(out_dir) / f"lyra_{result['domain']}_{scenario}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(path)
