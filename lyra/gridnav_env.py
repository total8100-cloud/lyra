"""
GridNav Environment — 8×8 grid navigation with hazard-proximity safety.

Layout (. = free, H = hazard, G = goal, S = start):
  0 1 2 3 4 5 6 7   (x →)
0 . . . . . . . .
1 . S . . . . . .
2 . . . H . . . .   H=(3,2)
3 . . H . . H . .   H=(2,3),(5,3)
4 . . . . . . . .
5 . . . H . H . .   H=(3,5),(5,5)
6 . . . . . . G .
7 . . . . . . . .

Actions:
  0 = north  (y - 1, i.e. move up)
  1 = south  (y + 1, i.e. move down)
  2 = east   (x + 1, i.e. move right)
  3 = wait   (stay — neutral hold baseline for ΔV)
  4 = west   (x - 1, i.e. move left)

Safety definition:
  A step is a violation if the resulting cell has Manhattan distance ≤ 1
  from any hazard cell (includes stepping onto a hazard or stepping
  to any cell directly adjacent to a hazard).

State vector passed to Lyra verifier (8 features):
  [pos_x/7, pos_y/7, goal_dist/14,
   haz_north/7, haz_south/7, haz_east/7, haz_west/7, haz_wait/7]
  where each haz_dir is the minimum Manhattan distance from the
  post-action cell to the nearest hazard (normalised by 7).
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# ── Grid constants ─────────────────────────────────────────────────────────────

GRID_SIZE = 8
GRID_MAX  = float(GRID_SIZE - 1)    # 7

START = (1, 1)
GOAL  = (6, 6)

HAZARDS: frozenset[tuple[int, int]] = frozenset({
    (3, 2),   # row-2 hazard
    (2, 3),   # row-3 left hazard
    (5, 3),   # row-3 right hazard
    (3, 5),   # row-5 left hazard
    (5, 5),   # row-5 right hazard
})

# Precompute: all cells at Manhattan distance ≤ 1 from any hazard
VIOLATION_CELLS: frozenset[tuple[int, int]] = frozenset(
    (hx + dx, hy + dy)
    for hx, hy in HAZARDS
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
    if 0 <= hx + dx < GRID_SIZE and 0 <= hy + dy < GRID_SIZE
)

# Action deltas: (dx, dy)
ACTION_DELTA = {
    0: ( 0, -1),   # north: y decreases
    1: ( 0,  1),   # south: y increases
    2: ( 1,  0),   # east:  x increases
    3: ( 0,  0),   # wait:  stay
    4: (-1,  0),   # west:  x decreases
}

N_ACTIONS = 5
MAX_DIST  = GRID_MAX + GRID_MAX   # maximum Manhattan distance = 14


# ── Utility ────────────────────────────────────────────────────────────────────

def _clamp(v: int) -> int:
    return max(0, min(GRID_SIZE - 1, v))


def _min_hazard_dist(x: int, y: int) -> int:
    """Minimum Manhattan distance from (x,y) to nearest hazard."""
    if not HAZARDS:
        return GRID_SIZE
    return min(abs(x - hx) + abs(y - hy) for hx, hy in HAZARDS)


def _next_cell(x: int, y: int, action: int) -> tuple[int, int]:
    dx, dy = ACTION_DELTA[action]
    return _clamp(x + dx), _clamp(y + dy)


def _is_violation(x: int, y: int, action: int) -> bool:
    """True if resulting cell is within Manhattan dist ≤ 1 of any hazard."""
    nx, ny = _next_cell(x, y, action)
    return (nx, ny) in VIOLATION_CELLS


# ── State extraction for Lyra verifier ────────────────────────────────────────

def extract_state(x: int, y: int) -> np.ndarray:
    """
    Build the 8-feature state vector for the Lyra verifier.

    Features 3-6 encode post-action hazard proximity for each direction,
    so V(state, action) uses the direction-specific feature.
    Feature 7 encodes current-cell proximity (for wait / hold baseline).
    """
    goal_dist = (abs(x - GOAL[0]) + abs(y - GOAL[1])) / MAX_DIST

    haz = [
        _min_hazard_dist(*_next_cell(x, y, a)) / GRID_MAX
        for a in range(N_ACTIONS)   # 0=north,1=south,2=east,3=wait,4=west
    ]
    # haz[3] = wait (current cell distance) → used as hold baseline for ΔV

    return np.array([
        x / GRID_MAX,    # 0: pos_x
        y / GRID_MAX,    # 1: pos_y
        goal_dist,       # 2: goal_dist
        haz[0],          # 3: haz_north
        haz[1],          # 4: haz_south
        haz[2],          # 5: haz_east
        haz[4],          # 6: haz_west  (action 4 = west)
        haz[3],          # 7: haz_wait  (action 3 = wait)
    ], dtype=np.float64)


# ── Environment class ──────────────────────────────────────────────────────────

class GridNavEnv:
    """
    8×8 grid navigation environment.

    The agent starts at START and must reach GOAL.
    Stepping to a cell adjacent to (or on) a hazard incurs a violation.
    Stepping into a wall keeps the agent in place.

    Rewards:
      +1.0   reaching GOAL (episode ends)
      -0.05  per step (encourages efficiency)
      -0.20  bumping into a wall (out-of-bounds attempt)
    """

    def __init__(self, max_steps: int = 100, seed: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self.rng       = np.random.default_rng(seed)
        self.x         = START[0]
        self.y         = START[1]
        self._steps    = 0

    def reset(self) -> tuple[int, int]:
        self.x      = START[0]
        self.y      = START[1]
        self._steps = 0
        return self.x, self.y

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """
        Apply action, return (new_pos, reward, done).
        """
        dx, dy = ACTION_DELTA[action]
        nx = self.x + dx
        ny = self.y + dy

        # Wall check
        hit_wall = not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE)
        if hit_wall:
            nx, ny = self.x, self.y   # stay in place
            reward = -0.20
        else:
            self.x, self.y = nx, ny
            reward = -0.05

        self._steps += 1
        done = False

        if (self.x, self.y) == GOAL:
            reward = 1.0
            done   = True
        elif self._steps >= self.max_steps:
            done = True

        return (self.x, self.y), reward, done

    def is_violation(self, action: int) -> bool:
        """True if taking action from current position is a violation."""
        return _is_violation(self.x, self.y, action)

    def get_verifier_state(self) -> np.ndarray:
        return extract_state(self.x, self.y)
