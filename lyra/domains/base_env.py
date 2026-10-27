"""
Abstract base class for all Lyra multi-domain environments.

Each domain environment wraps an external simulator (FinRL, SimGlucose,
PowerGym, CybORG++) and maps its native state to the YAML-indexed
state vector expected by the compiled Lyapunov verifier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseDomainEnv(ABC):
    """
    Standard interface for all domain environments used in Lyra experiments.

    Subclasses must implement:
      - reset()     → np.ndarray  (initial state, YAML-indexed)
      - step(a_idx) → (state, reward, done)
      - n_actions   property
      - n_state     property
      - hold_action property  (index of the "do nothing" action for ΔV baseline)
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial YAML-indexed state vector."""

    @abstractmethod
    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute action_idx, return (next_state, task_reward, done).
        next_state must be YAML-indexed (same order as state_variables in YAML).
        """

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of discrete actions."""

    @property
    @abstractmethod
    def n_state(self) -> int:
        """Dimensionality of the state vector."""

    @property
    @abstractmethod
    def hold_action(self) -> int:
        """Index of the neutral/hold action used as ΔV baseline in verifier."""

    def reset_stress(self) -> np.ndarray:
        """Reset to a high-violation-pressure initial state for stress scenario.
        Default falls back to reset(); subclasses should override with
        domain-specific near-boundary initial conditions."""
        return self.reset()

    def state(self) -> np.ndarray:
        """Return current state (override if env tracks state internally)."""
        raise NotImplementedError
