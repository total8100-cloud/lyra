"""
Cybersecurity Environment — CybORG++ wrapper.

State vector (YAML-indexed):
  [0] role_level           — privilege: 0=none, 0.33=user, 0.67=operator, 1.0=admin
  [1] resource_sensitivity — 0=public, 0.33=internal, 0.67=confidential, 1.0=critical
  [2] auth_state           — 0=unauthenticated, 1=authenticated
  [3] zone_distance        — security zone hops from origin, normalized (0-1)
  [4] session_age          — session elapsed time normalized (0=fresh, 1=expired)

Actions:
  0=scan, 1=access_low, 2=access_high, 3=lateral_move, 4=privilege_escalate

Tries to import CybORG; falls back to a CDA network simulation implementing
IEC 62443 SR 2.1 zone/conduit model.
"""
from __future__ import annotations

import random
import numpy as np
from experiment.lyra.domains.base_env import BaseDomainEnv

# CDA network topology: 4 zones (DMZ, Control, Safety, Critical)
ZONE_LEVELS       = [0.0, 0.33, 0.67, 1.0]   # zone sensitivity by index
MAX_SESSION_STEPS = 60
PRIVILEGE_LEVELS  = [0.0, 0.33, 0.67, 1.0]   # none, user, operator, admin

try:
    from CybORG import CybORG  # noqa: F401
    _CYBORG_AVAILABLE = True
except Exception:
    _CYBORG_AVAILABLE = False


class CyberEnv(BaseDomainEnv):
    """
    Cyber-Physical System access control environment (IEC 62443 SR 2.1).

    Models a CDA (Critical Digital Asset) network with 4 security zones.
    The agent acts as a defender / auditor verifying access control compliance.
    Violations occur when access is attempted above privilege level, or when
    lateral movement crosses zone boundaries without authentication.
    """

    def __init__(self, seed: int | None = None):
        self._rng        = np.random.default_rng(seed)
        self._py_rng     = random.Random(seed)
        self._role       = 1   # current privilege tier (0-3)
        self._zone       = 0   # current zone index (0=DMZ, 3=Critical)
        self._auth       = False
        self._session_step = 0
        self._step_count   = 0
        self._target_sensitivity = 0.33

    # ── State transition helpers ───────────────────────────────────────────────

    def _update_target(self) -> None:
        """Randomly shift target resource sensitivity (simulates dynamic access requests)."""
        if self._py_rng.random() < 0.15:
            self._target_sensitivity = self._py_rng.choice(ZONE_LEVELS)

    def _decay_auth(self) -> None:
        """Authentication times out after MAX_SESSION_STEPS."""
        self._session_step += 1
        if self._session_step >= MAX_SESSION_STEPS:
            self._auth = False
            self._session_step = 0

    # ── YAML state extraction ──────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        # [0] role_level: current privilege tier normalized
        role_level = float(PRIVILEGE_LEVELS[self._role])

        # [1] resource_sensitivity: target resource sensitivity
        resource_sensitivity = float(self._target_sensitivity)

        # [2] auth_state
        auth_state = 1.0 if self._auth else 0.0

        # [3] zone_distance: how far current zone is from origin (DMZ)
        zone_distance = float(self._zone / 3.0)

        # [4] session_age: normalized elapsed session time
        session_age = float(np.clip(self._session_step / MAX_SESSION_STEPS, 0.0, 1.0))

        return np.array([role_level, resource_sensitivity, auth_state,
                         zone_distance, session_age], dtype=np.float32)

    # ── BaseDomainEnv interface ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._role             = self._py_rng.randint(0, 2)    # start as none/user/operator
        self._zone             = 0                              # start at DMZ
        self._auth             = self._py_rng.random() > 0.5
        self._session_step     = self._py_rng.randint(0, 20)
        self._step_count       = 0
        self._target_sensitivity = self._py_rng.choice(ZONE_LEVELS)
        return self._get_state()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        self._decay_auth()
        self._update_target()
        self._step_count += 1

        r_task = 0.0

        if action_idx == 0:    # scan — gather intelligence, low risk
            r_task = 0.1
            if not self._auth:
                r_task = -0.1   # unauth scan raises suspicion

        elif action_idx == 1:  # access_low — access public/internal resource
            r_task = 0.3 if self._auth else 0.1

        elif action_idx == 2:  # access_high — access confidential/critical resource
            if self._target_sensitivity <= PRIVILEGE_LEVELS[self._role]:
                r_task = 0.8   # legitimate high-value access
            else:
                r_task = -1.0  # policy violation

        elif action_idx == 3:  # lateral_move — move to adjacent zone
            if self._auth and self._zone < 3:
                self._zone += 1
                r_task = 0.4
            elif not self._auth:
                r_task = -0.5   # unauthenticated lateral movement

        elif action_idx == 4:  # privilege_escalate
            if self._auth and self._role < 3:
                self._role += 1
                r_task = 0.6
            elif not self._auth:
                r_task = -2.0   # unauthenticated escalation attempt
                self._auth = False   # session revoked

        done = self._step_count >= 100
        return self._get_state(), float(r_task), done

    @property
    def n_actions(self) -> int:
        return 5

    @property
    def n_state(self) -> int:
        return 5

    def reset_stress(self) -> np.ndarray:
        """Active Intrusion Response: critical resource required, low privilege, expired session.

        Scenario: threat actor detected in Critical zone; defender agent must
        access critical asset to contain the incident, but holds only user-level
        privilege, session is near expiry, and re-authentication is unavailable.
        - role:                 user (0.33) — cannot access critical resource
        - resource_sensitivity: critical (1.0) — target forced by threat
        - auth:                 False  — session expired under duress
        - zone:                 zone 2 (zone_distance 0.67) — deep in network
        - session_age:          0.85-0.94  — near/at expiry threshold (0.85)
        Every high-value action (access_high, privilege_escalate) is a violation.
        """
        self._role                 = 1          # user privilege
        self._target_sensitivity   = 1.0        # critical resource
        self._auth                 = False
        self._zone                 = 2
        self._session_step         = int(self._rng.uniform(51, 56))
        self._step_count           = 0
        return self._get_state()

    @property
    def hold_action(self) -> int:
        return 0   # scan = lowest-impact baseline action
