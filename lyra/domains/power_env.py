"""
Power Grid Environment — PowerGym wrapper.

State vector (YAML-indexed):
  [0] volt_dev      — |V - V_nom| / V_nom (0-1); violation > 0.05 (ANSI C84.1)
  [1] q_ratio       — Q / Q_rated (0-1)
  [2] loading_ratio — line loading / thermal limit (0-1)
  [3] power_factor  — 1 - pf (0=unity, 1=worst); violation > 0.10 (pf < 0.90)

Actions:
  0=no_action, 1=cap_on, 2=cap_off, 3=tap_raise, 4=tap_lower

Tries to import powergym; falls back to a linearized distribution network model.
"""
from __future__ import annotations

import random
import numpy as np
from experiment.lyra.domains.base_env import BaseDomainEnv

V_NOM       = 1.00    # per-unit nominal voltage
Q_RATED     = 1.00    # per-unit rated reactive power
CAP_STEP    = 0.05    # capacitor bank switching step (pu Q)
TAP_STEP    = 0.02    # transformer tap voltage step (pu V)
LOAD_SIGMA  = 0.03    # per-step load fluctuation std

try:
    import powergym  # noqa: F401
    _POWERGYM_AVAILABLE = True
except ImportError:
    _POWERGYM_AVAILABLE = False


class PowerEnv(BaseDomainEnv):
    """
    Distribution grid Volt-Var control environment.

    Simulates a single-feeder radial network with stochastic load variation.
    The simplified linearized model captures voltage droop characteristics
    sufficient for constraint violation studies under IEC 61850 / ANSI C84.1.
    """

    def __init__(self, seed: int | None = None):
        self._rng    = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self._voltage     = V_NOM
        self._q_inject    = 0.0    # pu reactive power injection
        self._line_load   = 0.5    # normalized line loading
        self._pf          = 0.95   # power factor
        self._step_count  = 0
        self._cap_state   = 0      # capacitor banks on (0-3)

    # ── Network physics (linearized droop) ────────────────────────────────────

    def _tick_load(self) -> None:
        """Stochastic load variation simulating daily demand pattern."""
        # Sinusoidal daily profile + noise
        t = (self._step_count % 48) / 48.0   # 48 steps = one day
        base_load = 0.5 + 0.3 * np.sin(2 * np.pi * t - np.pi / 2)
        noise = float(self._rng.normal(0.0, LOAD_SIGMA))
        self._line_load = float(np.clip(base_load + noise, 0.0, 1.0))

        # Voltage droop: higher load → lower voltage
        # V ≈ V_nom - Z * I; realistic distribution feeder: dV ≈ -0.08 * load
        v_droop = -0.08 * self._line_load
        v_cap   = self._q_inject * 0.04    # capacitor raises voltage
        v_noise = float(self._rng.normal(0.0, 0.005))
        self._voltage = float(np.clip(V_NOM + v_droop + v_cap + v_noise, 0.85, 1.15))

        # Power factor degrades with load (more reactive demand)
        self._pf = float(np.clip(0.95 - self._line_load * 0.10, 0.80, 1.00))

    # ── YAML state extraction ──────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        # [0] volt_dev: |V - V_nom| / V_nom
        volt_dev = float(abs(self._voltage - V_NOM) / V_NOM)
        volt_dev = float(np.clip(volt_dev, 0.0, 1.0))

        # [1] q_ratio: Q / Q_rated
        q_ratio = float(np.clip(abs(self._q_inject) / Q_RATED, 0.0, 1.0))

        # [2] loading_ratio
        loading_ratio = float(np.clip(self._line_load, 0.0, 1.0))

        # [3] power_factor deviation: 1 - pf
        power_factor = float(np.clip(1.0 - self._pf, 0.0, 1.0))

        return np.array([volt_dev, q_ratio, loading_ratio, power_factor],
                        dtype=np.float32)

    # ── BaseDomainEnv interface ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._voltage    = float(self._rng.uniform(0.95, 1.05))
        self._q_inject   = float(self._rng.uniform(-0.2, 0.2))
        self._line_load  = float(self._rng.uniform(0.3, 0.7))
        self._pf         = 0.95
        self._cap_state  = 0
        self._step_count = 0
        return self._get_state()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        if action_idx == 1:    # cap_on
            if self._cap_state < 3:
                self._cap_state += 1
                self._q_inject = min(self._q_inject + CAP_STEP, Q_RATED)
        elif action_idx == 2:  # cap_off
            if self._cap_state > 0:
                self._cap_state -= 1
                self._q_inject = max(self._q_inject - CAP_STEP, -Q_RATED)
        elif action_idx == 3:  # tap_raise
            self._voltage = min(self._voltage + TAP_STEP, 1.15)
        elif action_idx == 4:  # tap_lower
            self._voltage = max(self._voltage - TAP_STEP, 0.85)
        # action_idx == 0: no_action

        self._tick_load()
        self._step_count += 1

        # Task reward: minimize voltage deviation and maintain stable operation
        volt_dev = abs(self._voltage - V_NOM) / V_NOM
        r_task   = float(1.0 - 10.0 * volt_dev - 0.5 * (1.0 - self._pf))
        r_task   = float(np.clip(r_task, -2.0, 1.0))

        done = self._step_count >= 100
        return self._get_state(), r_task, done

    @property
    def n_actions(self) -> int:
        return 5

    @property
    def n_state(self) -> int:
        return 4

    def reset_stress(self) -> np.ndarray:
        """Summer Peak Demand: near-voltage-collapse + high thermal loading.

        Scenario: heatwave peak afternoon; load approaches thermal limit;
        capacitor banks near saturation; voltage below ANSI C84.1 Range A floor.
        - voltage:    0.930-0.945 pu  → volt_dev 0.055-0.070  (already violating 0.05 band)
        - line_load:  0.87-0.93       → approaching 0.90 thermal limit
        - q_inject:   0.80-0.88       → q_ratio near 0.90 cap_on limit
        - pf:         0.87-0.90       → power_factor 0.10-0.13 near 0.10 limit
        Agent must raise voltage but every control action risks a secondary violation.
        """
        self._voltage   = float(self._rng.uniform(0.930, 0.945))
        self._q_inject  = float(self._rng.uniform(0.80, 0.88))
        self._line_load = float(self._rng.uniform(0.87, 0.93))
        self._pf        = float(self._rng.uniform(0.87, 0.90))
        self._cap_state = 2
        self._step_count = 0
        return self._get_state()

    @property
    def hold_action(self) -> int:
        return 0   # no_action = neutral baseline
