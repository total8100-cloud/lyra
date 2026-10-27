"""
Symbolic-to-Energy Compiler (Algorithm 1 from Lyra paper).

Maps normative constraints to a differentiable Lyapunov energy function.

State vector  x ∈ ℝ⁴:
  x[0]  own_share   — agent's resource share (0-1, normalised)
  x[1]  peer_share  — mean peer share (0-1)
  x[2]  pressure    — social pressure signal (0-1)
  x[3]  compliance  — recent constraint-compliance rate (0-1)

Actions (indices):
  0 = share   (prosocial — give 10 % to neediest peer)
  1 = split   (prosocial — distribute 5 % equally)
  2 = take    (antisocial — take 15 % from richest peer)
  3 = hold    (neutral)

Three normative constraints:
  C1  Consent   — may not 'take' when already ahead of peers
  C2  Survival  — may not 'take' when peers are below minimum
  C3  Fairness  — aggregate inequality must stay below threshold
"""
from __future__ import annotations

import math
import numpy as np

# ── Smooth primitives ──────────────────────────────────────────────────────────

def _softplus(z: float, k: float = 10.0) -> float:
    """Smooth max(0, z) — approaches ReLU as k → ∞."""
    if z * k > 100:   # prevent overflow
        return z
    if z * k < -100:
        return 0.0
    return math.log1p(math.exp(k * z)) / k


def _softplus_grad(z: float, k: float = 10.0) -> float:
    """d/dz softplus(z, k) = sigmoid(k·z)."""
    kz = k * z
    if kz > 100:
        return 1.0
    if kz < -100:
        return 0.0
    return 1.0 / (1.0 + math.exp(-kz))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ── Constraint C1: Consent ─────────────────────────────────────────────────────
# Violation when agent takes AND is already ahead of peers by > 0.15.
# Relaxation: v₁ = 1{a=take} · softplus(own - peer - θ)

_C1_THRESHOLD = -0.50   # negative → any 'take' has violation energy regardless of wealth


def _c1_violation(x: np.ndarray, a_idx: int) -> float:
    if a_idx != 2:   # only 'take' can violate
        return 0.0
    return _softplus(x[0] - x[1] - _C1_THRESHOLD)


def _c1_grad(x: np.ndarray, a_idx: int) -> np.ndarray:
    g = np.zeros(4)
    if a_idx != 2:
        return g
    sg = _softplus_grad(x[0] - x[1] - _C1_THRESHOLD)
    g[0] =  sg   # ∂/∂own_share
    g[1] = -sg   # ∂/∂peer_share
    return g


# ── Constraint C2: Survival ────────────────────────────────────────────────────
# Violation when agent takes AND peers are already below minimum (0.10).
# Relaxation: v₂ = 1{a=take} · softplus(min_thresh - peer_share)

_C2_MIN_PEER = 0.10


def _c2_violation(x: np.ndarray, a_idx: int) -> float:
    if a_idx != 2:
        return 0.0
    return _softplus(_C2_MIN_PEER - x[1])


def _c2_grad(x: np.ndarray, a_idx: int) -> np.ndarray:
    g = np.zeros(4)
    if a_idx != 2:
        return g
    g[1] = -_softplus_grad(_C2_MIN_PEER - x[1])   # ∂/∂peer_share
    return g


# ── Constraint C3: Fairness ────────────────────────────────────────────────────
# Violation when |own - peer| > 0.30 (Gini proxy).
# Relaxation: v₃ = softplus(|own - peer| - gini_thresh)
# Gradient via smooth abs: d/dx |f(x)| ≈ f(x)/√(f(x)²+ε) · f'(x)

_C3_GINI_THRESH = 0.30
_C3_EPS         = 1e-6


def _c3_violation(x: np.ndarray, a_idx: int) -> float:
    diff = x[0] - x[1]
    return _softplus(math.sqrt(diff * diff + _C3_EPS) - _C3_GINI_THRESH)


def _c3_grad(x: np.ndarray, a_idx: int) -> np.ndarray:
    g    = np.zeros(4)
    diff = x[0] - x[1]
    abs_diff = math.sqrt(diff * diff + _C3_EPS)
    sp_g  = _softplus_grad(abs_diff - _C3_GINI_THRESH)
    sign  = diff / abs_diff                            # smooth sign
    g[0] =  sp_g * sign
    g[1] = -sp_g * sign
    return g


# ── Constraint registry ────────────────────────────────────────────────────────
# Each entry: (name, violation_fn, grad_fn, weight)

CONSTRAINTS: list[tuple[str, callable, callable, float]] = [
    ("consent",   _c1_violation, _c1_grad,  0.5),
    ("survival",  _c2_violation, _c2_grad,  0.3),
    ("fairness",  _c3_violation, _c3_grad,  0.2),
]


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_V(x: np.ndarray, a_idx: int) -> float:
    """
    V(x, a) = Σᵢ wᵢ · vᵢ(x, a)²

    Returns the Lyapunov energy — 0.0 iff all constraints satisfied.
    """
    total = 0.0
    for _, vfn, _, w in CONSTRAINTS:
        v = vfn(x, a_idx)
        total += w * v * v
    return total


def compute_grad_V(x: np.ndarray, a_idx: int) -> np.ndarray:
    """
    ∇V(x, a) w.r.t. state vector x.

    ∇V = 2 Σᵢ wᵢ · vᵢ(x,a) · ∇vᵢ(x,a)
    """
    grad = np.zeros(4)
    for _, vfn, gfn, w in CONSTRAINTS:
        v  = vfn(x, a_idx)
        gv = gfn(x, a_idx)
        grad += 2.0 * w * v * gv
    return grad


def violation_breakdown(x: np.ndarray, a_idx: int) -> dict[str, float]:
    """Per-constraint violation values — useful for debugging and analysis."""
    return {name: vfn(x, a_idx) for name, vfn, _, _ in CONSTRAINTS}
