"""
Constraint DSL — Schema definition and YAML parser.

Defines the intermediate representation (IR) for normative constraints
consumed by auto_compiler.py.

Supported violation types:
  threshold_above  softplus(expr - θ)       violation when expr > θ
  threshold_below  softplus(θ - expr)       violation when expr < θ
  abs_threshold    softplus(|expr| - θ)     violation when |expr| > θ
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# ── Intermediate representation ────────────────────────────────────────────────

@dataclass
class StateVarDef:
    name:        str
    index:       int
    description: str = ""


@dataclass
class ActionDef:
    index: int
    name:  str


@dataclass
class ViolationSpec:
    type:      str    # threshold_above | threshold_below | abs_threshold
    expr:      str    # arithmetic expression over state variable names
    threshold: float


@dataclass
class ConstraintSpec:
    name:        str
    description: str
    trigger:     Optional[str]    # action name that activates constraint, or None (always)
    violation:   ViolationSpec
    weight:      float
    k:           float = 10.0    # softplus sharpness


@dataclass
class ConstraintSchema:
    meta:             dict
    state_variables:  list[StateVarDef]
    actions:          list[ActionDef]
    constraints:      list[ConstraintSpec]

    def action_index(self, name: str) -> int:
        for a in self.actions:
            if a.name == name:
                return a.index
        raise KeyError(f"Unknown action: {name!r}")

    def n_state(self) -> int:
        return len(self.state_variables)

    def state_names(self) -> list[str]:
        return [v.name for v in sorted(self.state_variables, key=lambda v: v.index)]


# ── YAML loader ────────────────────────────────────────────────────────────────

def load_schema(path: str | Path) -> ConstraintSchema:
    """Parse a constraint YAML file into a ConstraintSchema IR."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    state_vars = [
        StateVarDef(
            name        = v["name"],
            index       = v["index"],
            description = v.get("description", ""),
        )
        for v in raw["state_variables"]
    ]

    actions = [
        ActionDef(index=a["index"], name=a["name"])
        for a in raw["actions"]
    ]

    constraints = []
    for c in raw["constraints"]:
        vraw = c["violation"]
        vspec = ViolationSpec(
            type      = vraw["type"],
            expr      = vraw["expr"],
            threshold = float(vraw["threshold"]),
        )
        trigger_raw = c.get("trigger")
        trigger_action = (
            trigger_raw["action"]
            if isinstance(trigger_raw, dict) and "action" in trigger_raw
            else None
        )
        constraints.append(ConstraintSpec(
            name        = c["name"],
            description = c.get("description", "").strip(),
            trigger     = trigger_action,
            violation   = vspec,
            weight      = float(c["weight"]),
            k           = float(c.get("k", 10.0)),
        ))

    return ConstraintSchema(
        meta            = raw.get("meta", {}),
        state_variables = state_vars,
        actions         = actions,
        constraints     = constraints,
    )
