"""
Deep Q-Network agent for Lyra-DRL experiment.

Replaces tabular Q-learning with a neural Q-function (PyTorch).
Identical external interface to QAgent: choose(x, epsilon), update(x, a, r, x').

Architecture: FC(4→128) → ReLU → FC(128→64) → ReLU → FC(64→4)
Training:     Adam, Huber loss, experience replay, target network
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Q-network ──────────────────────────────────────────────────────────────────

class _QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ──────────────────────────────────────────────────────────────

class _ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(self, s, a: int, r: float, s2) -> None:
        self._buf.append((np.array(s, dtype=np.float32),
                          a,
                          float(r),
                          np.array(s2, dtype=np.float32)))

    def sample(self, batch: int):
        items = random.sample(self._buf, batch)
        s, a, r, s2 = zip(*items)
        return (torch.tensor(np.array(s),  dtype=torch.float32),
                torch.tensor(a,            dtype=torch.long),
                torch.tensor(r,            dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32))

    def __len__(self) -> int:
        return len(self._buf)


# ── DQN agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN with experience replay and target network.

    Drop-in replacement for QAgent in runner.py:
      choose(x, epsilon) → int
      update(x, a, reward, x_next) → None
    """

    def __init__(
        self,
        state_dim:     int   = 4,
        n_actions:     int   = 4,
        lr:            float = 1e-3,
        gamma:         float = 0.9,
        batch_size:    int   = 64,
        buffer_size:   int   = 10_000,
        target_update: int   = 200,   # gradient steps between target sync
        seed:          Optional[int] = None,
    ) -> None:
        self._state_dim    = state_dim
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self._grad_steps   = 0
        self.rng           = random.Random(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self._online = _QNet(state_dim, n_actions)
        self._target = _QNet(state_dim, n_actions)
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()

        self._opt    = optim.Adam(self._online.parameters(), lr=lr)
        self._loss   = nn.SmoothL1Loss()
        self._buffer = _ReplayBuffer(buffer_size)

    # ── public interface ───────────────────────────────────────────────────────

    def choose(self, x: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self._online(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax().item())

    def update(self, x: np.ndarray, a: int,
               reward: float, x_next: np.ndarray) -> None:
        self._buffer.push(x, a, reward, x_next)
        if len(self._buffer) < self.batch_size:
            return

        s, acts, rewards, s2 = self._buffer.sample(self.batch_size)

        with torch.no_grad():
            next_q  = self._target(s2).max(dim=1)[0]
            targets = rewards + self.gamma * next_q

        q_pred = self._online(s).gather(1, acts.unsqueeze(1)).squeeze(1)
        loss   = self._loss(q_pred, targets)

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), 1.0)
        self._opt.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update == 0:
            self._target.load_state_dict(self._online.state_dict())

    def reset(self) -> None:
        self._online = _QNet(self._state_dim, self.n_actions)
        self._target = _QNet(self._state_dim, self.n_actions)
        self._target.load_state_dict(self._online.state_dict())
        self._buffer = _ReplayBuffer(10_000)
        self._grad_steps = 0
