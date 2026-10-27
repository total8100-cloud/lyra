"""
2-Token Verification Protocol (Algorithm 2 from Lyra paper).

For each (state, action) pair the verifier returns:
  V       — current Lyapunov energy (constraint violation level)
  delta_V — directional derivative ∇V · â  (action direction risk)
  verdict — SAFE | WARN | BLOCK
  tokens  — number of verification tokens consumed this call

Token costs per condition:
  none     — 0
  z3_text  — ~500  (SMT proof trace → natural language)
  binary   — 1     (single 0/1 flag)
  pmcs     — 1     (single probability scalar)
  v_only   — 1     (V scalar only; ablation: no ΔV)
  dv_only  — 1     (ΔV scalar only; ablation: no V)
  lyra     — 2     (V + delta_V scalars)

Pass a CompiledConstraints object from auto_compiler.compile_yaml() as
`compiled` to use the automatic symbolic compiler instead of the
hand-coded functions.  If omitted, the hand-coded compiler.py is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from experiment.lyra.compiler import (
    compute_V as _hc_compute_V,
    compute_grad_V as _hc_compute_grad_V,
    violation_breakdown as _hc_violation_breakdown,
)

# ── Verdict ────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    SAFE  = "safe"
    WARN  = "warn"
    BLOCK = "block"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    V:         float
    delta_V:   float
    verdict:   Verdict
    tokens:    int
    breakdown: dict[str, float] = field(default_factory=dict)

    # Condition-specific signal fields
    binary:    Optional[int]   = None   # conditions C
    pmcs:      Optional[float] = None   # condition D


# ── Threshold defaults ─────────────────────────────────────────────────────────

TAU_LOW  = 0.05   # SAFE / WARN boundary
TAU_HIGH = 0.30   # WARN / BLOCK boundary


# ── Sigmoid for PMCS ──────────────────────────────────────────────────────────

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ── Core verifier ──────────────────────────────────────────────────────────────

def verify(
    x:          np.ndarray,
    a_idx:      int,
    condition:  str,            # "none" | "z3_text" | "binary" | "pmcs" | "lyra"
    tau_low:    float = TAU_LOW,
    tau_high:   float = TAU_HIGH,
    compiled=None,              # CompiledConstraints from auto_compiler, or None
) -> VerifyResult:
    """
    Verify (x, a) under the given experimental condition.

    All conditions compute V and delta_V internally for bookkeeping;
    the signal actually transmitted to the agent differs per condition.
    """
    # ── Select compiler backend ───────────────────────────────────────────────
    if compiled is not None:
        _V    = compiled.compute_V
        _bd   = compiled.violation_breakdown
    else:
        _V    = _hc_compute_V
        _bd   = _hc_violation_breakdown

    # ── Always compute Lyapunov quantities ────────────────────────────────────
    V      = _V(x, a_idx)

    # Discrete-action delta_V:
    # ΔV = V(x, a_proposed) − V(x, hold)
    # Positive → proposed action worsens safety vs. doing nothing.
    # Negative → proposed action improves safety.
    V_hold  = _V(x, 3)          # action 3 = hold (neutral baseline)
    delta_V = float(V - V_hold)

    bd = _bd(x, a_idx)

    # ── Verdict from energy level ─────────────────────────────────────────────
    if V < tau_low:
        verdict = Verdict.SAFE
    elif V < tau_high:
        verdict = Verdict.WARN
    else:
        verdict = Verdict.BLOCK

    # ── Condition-specific token cost & signal ────────────────────────────────

    if condition == "none":
        return VerifyResult(V=V, delta_V=delta_V,
                            verdict=Verdict.SAFE,   # no blocking
                            tokens=0, breakdown=bd)

    if condition == "z3_text":
        # Simulates Z3 SAT/UNSAT + natural-language proof trace.
        # Verdict mapped from energy: same as lyra for comparability.
        # Token cost: empirical mean from Paper 1 measurements.
        tokens = 487
        bin_signal = 0 if verdict == Verdict.BLOCK else 1
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=tokens, breakdown=bd,
                            binary=bin_signal)

    if condition == "binary":
        # Single 0/1 flag — no gradient, no energy level.
        bin_signal = 0 if verdict == Verdict.BLOCK else 1
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=1, breakdown=bd,
                            binary=bin_signal)

    if condition == "pmcs":
        # Probabilistic Moral Confidence Score: P(violation) ∈ [0,1].
        # Derived from V via sigmoid (no gradient transmitted).
        pmcs_score = _sigmoid(10.0 * (V - tau_low))
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=1, breakdown=bd,
                            pmcs=pmcs_score)

    if condition == "cbf":
        # Control Barrier Function filter.
        # Transmits 1 token: h scalar (raw safety margin).
        # H(x,a) = min_i h_i(x,a); h_i >= 0 ↔ constraint i satisfied.
        # No WARN zone: either SAFE (H>=0) or BLOCK (H<0).
        if compiled is not None and hasattr(compiled, "compute_H"):
            H = compiled.compute_H(x, a_idx)
        else:
            # Fallback: derive from V (H < 0 iff V > tau_low)
            H = tau_low - V
        cbf_verdict = Verdict.SAFE if H >= 0.0 else Verdict.BLOCK
        return VerifyResult(V=V, delta_V=H,   # store H in delta_V slot (1 token)
                            verdict=cbf_verdict,
                            tokens=1, breakdown=bd)

    if condition == "lyra":
        # Full 2-token Lyra protocol: [V, delta_V].
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=2, breakdown=bd)

    if condition == "v_only":
        # Ablation: V only — no ΔV in reward, no WARN revision.
        # Tests contribution of instantaneous energy signal.
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=1, breakdown=bd)

    if condition == "dv_only":
        # Ablation: ΔV only — no V in reward, no WARN revision.
        # Tests contribution of marginal safety cost signal.
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=1, breakdown=bd)

    if condition == "lyra_no_rev":
        # Ablation: full [V, ΔV] signal but WARN revision disabled in runner.
        # Isolates the contribution of WARN-zone action revision.
        return VerifyResult(V=V, delta_V=delta_V, verdict=verdict,
                            tokens=2, breakdown=bd)

    if condition == "hard_shield":
        # Condition C (new ablation): Hard Constraint Shield.
        # Uses H_i(x,a) for BLOCK/PASS only — no E severity ranking in signal.
        # Action revision is done by runner (argmin_E over H_i=0 alternatives).
        # Renamed from "cbf" to better reflect the mechanism.
        if compiled is not None and hasattr(compiled, "compute_H"):
            H = compiled.compute_H(x, a_idx)
        else:
            H = tau_low - V
        hs_verdict = Verdict.SAFE if H >= 0.0 else Verdict.BLOCK
        return VerifyResult(V=V, delta_V=H,
                            verdict=hs_verdict,
                            tokens=1, breakdown=bd)

    if condition == "e_only":
        # Condition D (new ablation): E-only Revision.
        # H_i block + E-based severity ranking + revision.
        # Transmits only E (1 scalar); no ΔE in feedback.
        # Tests whether the second scalar ΔE adds value beyond E alone.
        return VerifyResult(V=V, delta_V=0.0, verdict=verdict,
                            tokens=1, breakdown=bd)

    raise ValueError(f"Unknown condition: {condition!r}")


# ── Reward modifier per condition ──────────────────────────────────────────────

def safety_reward(
    result:     VerifyResult,
    condition:  str,
    alpha:      float = 1.0,
    beta:       float = 0.5,
) -> float:
    """
    Convert verification result to a scalar safety reward addend.

    Lyra:   r_safety = -α·V  -  β·ΔV
    PMCS:   r_safety = -(pmcs score)
    Binary: r_safety = binary - 1  (0 → -1 penalty, 1 → 0)
    Z3Text: same as binary
    None:   r_safety = 0
    """
    if condition == "none":
        return 0.0

    if condition == "lyra":
        return -alpha * result.V - beta * result.delta_V

    if condition == "pmcs":
        return -(result.pmcs or 0.0)

    if condition in ("binary", "z3_text"):
        return float((result.binary or 0) - 1)

    if condition in ("cbf", "hard_shield"):
        # H stored in delta_V field; reward = min(0, H) — proportional to violation depth
        return min(0.0, result.delta_V)

    if condition == "e_only":
        # E only: penalise boundary proximity but not marginal cost ΔE
        return -alpha * result.V

    if condition == "v_only":
        return -alpha * result.V

    if condition == "dv_only":
        return -beta * result.delta_V

    if condition == "lyra_no_rev":
        # Ablation: same reward as lyra but WARN revision disabled.
        # Isolates contribution of WARN-zone action revision vs. signal alone.
        return -alpha * result.V - beta * result.delta_V

    return 0.0
