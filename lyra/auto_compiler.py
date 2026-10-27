"""
Symbolic-to-Energy Compiler  (Algorithm 1 — automatic version).

Given a ConstraintSchema (parsed from YAML), this compiler:
  1. Parses each constraint's violation expression symbolically (SymPy)
  2. Applies a softplus relaxation to produce a smooth violation function vᵢ(x, a)
  3. Computes the exact analytic gradient ∇vᵢ(x, a) via symbolic differentiation
  4. Assembles the Lyapunov energy  V(x,a) = Σᵢ wᵢ · vᵢ(x,a)²
     and its gradient               ∇V(x,a) = 2 Σᵢ wᵢ · vᵢ · ∇vᵢ

No manual mathematical derivation is required beyond the YAML specification.

Public API
----------
  compiled = compile_schema(schema)          -> CompiledConstraints
  V        = compiled.compute_V(x, a_idx)   -> float
  gradV    = compiled.compute_grad_V(x, a_idx) -> np.ndarray
  bd       = compiled.violation_breakdown(x, a_idx) -> dict[str, float]
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from experiment.lyra.constraint_lang import ConstraintSchema, ConstraintSpec


# ── SymPy parse helpers ────────────────────────────────────────────────────────

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _parse(expr_str: str, local_dict: dict) -> sp.Expr:
    """Parse an arithmetic expression string into a SymPy expression."""
    return parse_expr(
        expr_str,
        local_dict       = local_dict,
        transformations  = _TRANSFORMATIONS,
        evaluate         = True,
    )


# ── Softplus relaxations (symbolic) ───────────────────────────────────────────

def _softplus_sym(z: sp.Expr, k: float) -> sp.Expr:
    """Symbolic softplus: log(1 + exp(k·z)) / k."""
    return sp.log(1 + sp.exp(k * z)) / k


def _abs_smooth_sym(z: sp.Expr) -> sp.Expr:
    """Smooth |z| ≈ sqrt(z² + ε)."""
    return sp.sqrt(z**2 + sp.Float(1e-6))


def _build_violation_expr(
    spec:       ConstraintSpec,
    local_dict: dict,
) -> sp.Expr:
    """
    Build the symbolic violation expression for one constraint.

    Types:
      threshold_above  — softplus(expr - θ)
      threshold_below  — softplus(θ - expr)
      abs_threshold    — softplus(|expr| - θ)
    """
    expr  = _parse(spec.violation.expr, local_dict)
    theta = sp.Float(spec.violation.threshold)
    k     = spec.k

    vtype = spec.violation.type
    if vtype == "threshold_above":
        return _softplus_sym(expr - theta, k)
    if vtype == "threshold_below":
        return _softplus_sym(theta - expr, k)
    if vtype == "abs_threshold":
        return _softplus_sym(_abs_smooth_sym(expr) - theta, k)
    raise ValueError(f"Unknown violation type: {vtype!r}")


def _build_margin_expr(
    spec:       ConstraintSpec,
    local_dict: dict,
) -> sp.Expr:
    """
    Build the raw CBF safety margin h(x) for one constraint (no softplus).

    h(x) >= 0  ↔  constraint satisfied (safe)
    h(x) <  0  ↔  constraint violated  (unsafe)

    Types:
      threshold_above  — θ - expr        (safe when expr ≤ θ)
      threshold_below  — expr - θ        (safe when expr ≥ θ)
      abs_threshold    — θ - |expr|      (safe when |expr| ≤ θ)
    """
    expr  = _parse(spec.violation.expr, local_dict)
    theta = sp.Float(spec.violation.threshold)

    vtype = spec.violation.type
    if vtype == "threshold_above":
        return theta - expr
    if vtype == "threshold_below":
        return expr - theta
    if vtype == "abs_threshold":
        return theta - _abs_smooth_sym(expr)
    raise ValueError(f"Unknown violation type: {vtype!r}")


# ── Compiled constraint entry ──────────────────────────────────────────────────

class _CompiledConstraint:
    """
    Holds the numerically-callable violation function and gradient functions
    for a single constraint, produced by SymPy lambdify.
    """

    def __init__(
        self,
        name:         str,
        trigger_idx:  int | None,   # None → always active
        viol_fn:      Callable,     # f(*state_values) -> float
        grad_fns:     list[Callable],
        weight:       float,
        margin_fn:    Callable | None = None,  # CBF raw margin h(x)
    ) -> None:
        self.name        = name
        self.trigger_idx = trigger_idx
        self._viol       = viol_fn
        self._grads      = grad_fns
        self.weight      = weight
        self._margin     = margin_fn

    def violation(self, x: np.ndarray, a_idx: int) -> float:
        if self.trigger_idx is not None and a_idx != self.trigger_idx:
            return 0.0
        try:
            val = float(self._viol(*x))
            return val if math.isfinite(val) else 0.0
        except Exception:
            return 0.0

    def gradient(self, x: np.ndarray, a_idx: int) -> np.ndarray:
        n = len(x)
        g = np.zeros(n)
        if self.trigger_idx is not None and a_idx != self.trigger_idx:
            return g
        for i, gfn in enumerate(self._grads):
            try:
                val = float(gfn(*x))
                g[i] = val if math.isfinite(val) else 0.0
            except Exception:
                g[i] = 0.0
        return g

    def margin(self, x: np.ndarray, a_idx: int) -> float:
        """Raw CBF safety margin h(x). Positive = safe, negative = violated."""
        if self.trigger_idx is not None and a_idx != self.trigger_idx:
            return float("inf")   # constraint inactive → trivially safe
        if self._margin is None:
            return float("inf")
        try:
            val = float(self._margin(*x))
            return val if math.isfinite(val) else float("inf")
        except Exception:
            return float("inf")


# ── Public compiled container ──────────────────────────────────────────────────

class CompiledConstraints:
    """
    Result of compiling a ConstraintSchema.

    Provides the same interface as the hand-coded compiler.py so it can
    be used as a drop-in replacement in verifier.py and experiments.
    """

    def __init__(self, entries: list[_CompiledConstraint], n_state: int) -> None:
        self._entries = entries
        self._n_state = n_state

    def compute_V(self, x: np.ndarray, a_idx: int) -> float:
        """V(x, a) = Σᵢ wᵢ · vᵢ(x,a)²"""
        total = 0.0
        for c in self._entries:
            v      = c.violation(x, a_idx)
            total += c.weight * v * v
        return total

    def compute_grad_V(self, x: np.ndarray, a_idx: int) -> np.ndarray:
        """∇V(x, a) = 2 Σᵢ wᵢ · vᵢ(x,a) · ∇vᵢ(x,a)"""
        grad = np.zeros(self._n_state)
        for c in self._entries:
            v    = c.violation(x, a_idx)
            gv   = c.gradient(x, a_idx)
            grad += 2.0 * c.weight * v * gv
        return grad

    def compute_H(self, x: np.ndarray, a_idx: int) -> float:
        """
        CBF safety value H(x, a) = min_i h_i(x, a) over all active constraints.
        H >= 0 → all constraints satisfied (SAFE).
        H <  0 → at least one constraint violated (BLOCK).
        """
        margins = [c.margin(x, a_idx) for c in self._entries]
        return min(margins) if margins else float("inf")

    def violation_breakdown(self, x: np.ndarray, a_idx: int) -> dict[str, float]:
        return {c.name: c.violation(x, a_idx) for c in self._entries}

    def constraint_names(self) -> list[str]:
        return [c.name for c in self._entries]


# ── Main compilation function ──────────────────────────────────────────────────

def compile_schema(schema: ConstraintSchema) -> CompiledConstraints:
    """
    Compile a ConstraintSchema → CompiledConstraints.

    Steps per constraint:
      1. Build SymPy symbol map from state variable definitions
      2. Parse violation expression string into SymPy AST
      3. Apply softplus relaxation (symbolic)
      4. Differentiate symbolically w.r.t. each state variable
      5. Lambdify all expressions for fast numpy evaluation
    """
    # Symbol map: variable name → SymPy symbol
    state_names = schema.state_names()
    sym_map     = {name: sp.Symbol(name) for name in state_names}
    sym_list    = [sym_map[name] for name in state_names]

    # Action name → index
    action_map  = {a.name: a.index for a in schema.actions}

    entries: list[_CompiledConstraint] = []

    for spec in schema.constraints:
        # ── 1. Build symbolic violation expression ─────────────────────────
        viol_sym   = _build_violation_expr(spec, sym_map)
        margin_sym = _build_margin_expr(spec, sym_map)

        # ── 2. Symbolic gradient w.r.t. each state variable ───────────────
        grad_syms = [sp.diff(viol_sym, s) for s in sym_list]

        # ── 3. Lambdify (SymPy → Python callable) ─────────────────────────
        modules    = ["math", {"sqrt": math.sqrt, "log": math.log, "exp": math.exp}]
        viol_fn    = sp.lambdify(sym_list, viol_sym,    modules=modules)
        margin_fn  = sp.lambdify(sym_list, margin_sym,  modules=modules)
        grad_fns   = [sp.lambdify(sym_list, g, modules=modules) for g in grad_syms]

        # ── 4. Resolve trigger action index ───────────────────────────────
        trigger_idx: int | None = None
        if spec.trigger is not None:
            trigger_idx = action_map[spec.trigger]

        entries.append(_CompiledConstraint(
            name        = spec.name,
            trigger_idx = trigger_idx,
            viol_fn     = viol_fn,
            grad_fns    = grad_fns,
            weight      = spec.weight,
            margin_fn   = margin_fn,
        ))

    return CompiledConstraints(entries=entries, n_state=schema.n_state())


# ── Convenience loader ─────────────────────────────────────────────────────────

def compile_yaml(path: str) -> CompiledConstraints:
    """One-call helper: load YAML → compile → return CompiledConstraints."""
    from experiment.lyra.constraint_lang import load_schema
    return compile_schema(load_schema(path))
