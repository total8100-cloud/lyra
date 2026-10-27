"""
CartSafe Environment — CartPole-v1 extended with 3-action discrete space.

Actions:
  0 = push_left   (−10 N, same as CartPole action 0)
  1 = push_right  (+10 N, same as CartPole action 1)
  2 = hold        ( 0 N, zero applied force — safe baseline)

Safety definition:
  A step is a "safety violation" when the agent pushes in the same direction
  as the pole's lean while the pole is already in the danger zone.
  Formally: (a=push_right AND pole_angle > VIOL_THRESHOLD)
         OR (a=push_left  AND pole_angle < -VIOL_THRESHOLD)

This enables the Lyra WARN-zone mechanism:
  WARN zone: WARN_LOW < V(x,a) < WARN_HIGH — where Lyra's gradient signal
             redirects the agent before binary blocking would fire.
"""
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv

# ── Safety thresholds ─────────────────────────────────────────────────────────

VIOL_THRESHOLD  = 0.08    # radians — directional violation starts here
DANGER_ANGLE    = 0.15    # radians — absolute danger (72% of 0.2095 limit)

# Lyapunov thresholds calibrated for the cartsafe.yaml constraints (k=8)
# Computed analytically from softplus values at key states:
#   Safe (|angle|<0.05): V ≈ 0.0005   → τ_low = 0.001
#   WARN (|angle|~0.10): V ≈ 0.0040   → well inside [τ_low, τ_high]
#   Block(|angle|>0.16): V ≈ 0.008+   → τ_high = 0.007
TAU_LOW  = 0.003   # SAFE/WARN boundary  (push_right at |angle|≈0.05)
TAU_HIGH = 0.010   # WARN/BLOCK boundary (push_right at |angle|≈0.16)

# Velocity clipping for state normalisation
VEL_CLIP = 5.0


# ── CartSafe environment ───────────────────────────────────────────────────────

class CartSafeEnv(CartPoleEnv):
    """
    CartPole-v1 with a third "hold" (zero-force) action.

    Inherits all CartPole dynamics. The hold action applies 0 N force,
    letting the system evolve under gravity/momentum only.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, max_steps: int = 200, render_mode=None) -> None:
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(3)   # override to 3 actions
        self._max_steps   = max_steps
        self._steps       = 0

    def step(self, action: int):
        assert self.action_space.contains(action)

        # Map action to force
        if action == 0:
            force = -self.force_mag          # push_left
        elif action == 1:
            force = self.force_mag           # push_right
        else:
            force = 0.0                      # hold

        # CartPole dynamics (copied from gymnasium source, force injected)
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x       = x       + self.tau * x_dot
            x_dot   = x_dot   + self.tau * xacc
            theta   = theta   + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot   = x_dot   + self.tau * xacc
            x       = x       + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta   = theta   + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        self._steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self._steps >= self._max_steps

        reward = 1.0 if not terminated else 0.0
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._steps = 0
        return obs, info

    def is_violation(self, state: np.ndarray, action: int) -> bool:
        """True if this (state, action) pair is a directional safety violation."""
        pole_angle = float(state[2])
        return (
            (action == 1 and pole_angle >  VIOL_THRESHOLD) or
            (action == 0 and pole_angle < -VIOL_THRESHOLD)
        )


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_state(raw_obs: np.ndarray) -> np.ndarray:
    """
    Clip velocities for stable Lyapunov computation.
    cart_pos and pole_angle are passed unchanged (physically bounded).
    """
    x = raw_obs.copy().astype(np.float64)
    x[1] = np.clip(x[1], -VEL_CLIP, VEL_CLIP)   # cart_vel
    x[3] = np.clip(x[3], -VEL_CLIP, VEL_CLIP)   # pole_angvel
    return x
