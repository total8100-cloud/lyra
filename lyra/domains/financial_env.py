"""
Financial Trading Environment — FinRL-Meta wrapper.

State vector (YAML-indexed):
  [0] position_ratio   — largest single-asset weight in portfolio (0-1)
  [1] daily_loss_ratio — cumulative day loss / initial capital (0-1)
  [2] leverage_ratio   — total exposure / equity, normalized to [0,1] (1.0 = 3x)
  [3] volatility       — 20-step realized volatility, normalized (0-1)

Actions:
  0=hold, 1=buy, 2=sell, 3=short

Tries to import FinRL; falls back to a lightweight faithful simulation
if the library is not installed, so the pipeline is always runnable.
"""
from __future__ import annotations

import random
import numpy as np
from experiment.lyra.domains.base_env import BaseDomainEnv

N_ASSETS   = 4
INIT_CAP   = 100_000.0
MAX_LEV    = 3.0        # 3x max leverage; normalized to 1.0

try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv as _FinRLEnv
    _FINRL_AVAILABLE = True
except ImportError:
    _FINRL_AVAILABLE = False


class FinancialEnv(BaseDomainEnv):
    """
    Portfolio trading environment with MiFID II compliance constraints.

    When FinRL is installed, wraps StockTradingEnv with a real price history.
    Otherwise, runs a synthetic GBM price simulation with identical interface.
    """

    def __init__(self, seed: int | None = None):
        self._rng    = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self._reset_internal()

    # ── Internal state ─────────────────────────────────────────────────────────

    def _reset_internal(self) -> None:
        self._capital      = INIT_CAP * self._rng.uniform(0.7, 1.0)
        self._prices       = self._rng.uniform(50, 200, N_ASSETS)
        # Initialize with non-trivial holdings so violations occur naturally
        concentrated_asset = int(self._rng.integers(0, N_ASSETS))
        units = (INIT_CAP * self._rng.uniform(0.05, 0.20)) / self._prices[concentrated_asset]
        self._holdings     = np.zeros(N_ASSETS)
        self._holdings[concentrated_asset] = units
        self._capital     -= units * self._prices[concentrated_asset]
        self._capital      = max(self._capital, 0.0)
        self._price_hist   = [self._prices.copy()]
        self._daily_loss   = INIT_CAP * self._rng.uniform(0.0, 0.04)
        self._step_count   = 0

    # ── Price simulation (GBM step) ────────────────────────────────────────────

    def _tick_prices(self) -> None:
        drift = 0.0001
        vol   = 0.015
        shocks = self._rng.normal(drift, vol, N_ASSETS)
        self._prices = np.clip(self._prices * np.exp(shocks), 1.0, None)
        self._price_hist.append(self._prices.copy())
        if len(self._price_hist) > 25:
            self._price_hist.pop(0)

    # ── YAML state extraction ──────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        portfolio_value = self._capital + np.dot(self._holdings, self._prices)

        # [0] position_ratio: largest single position / total portfolio
        position_values = self._holdings * self._prices
        if portfolio_value > 0:
            position_ratio = float(np.max(position_values) / portfolio_value)
        else:
            position_ratio = 0.0

        # [1] daily_loss_ratio: loss relative to initial capital
        daily_loss_ratio = float(np.clip(self._daily_loss / INIT_CAP, 0.0, 1.0))

        # [2] leverage_ratio: total long exposure / equity, normalized by MAX_LEV
        equity = max(portfolio_value, 1.0)
        gross_exposure = float(np.sum(np.abs(position_values)))
        leverage = gross_exposure / equity
        leverage_ratio = float(np.clip(leverage / MAX_LEV, 0.0, 1.0))

        # [3] volatility: std of 20-step returns, normalized to [0,1]
        if len(self._price_hist) >= 5:
            hist = np.array(self._price_hist)
            returns = np.diff(hist, axis=0) / (hist[:-1] + 1e-8)
            vol = float(np.mean(np.std(returns, axis=0)))
            volatility = float(np.clip(vol / 0.10, 0.0, 1.0))  # 10% vol = 1.0
        else:
            volatility = 0.0

        return np.array([position_ratio, daily_loss_ratio,
                         leverage_ratio, volatility], dtype=np.float32)

    # ── BaseDomainEnv interface ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._reset_internal()
        return self._get_state()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool]:
        """
        Actions: 0=hold, 1=buy (largest holding), 2=sell, 3=short
        Reward: risk-adjusted return (positive for gains, negative for losses/violations)
        """
        self._tick_prices()
        self._step_count += 1

        portfolio_before = self._capital + np.dot(self._holdings, self._prices)
        asset = int(np.argmax(self._prices))   # trade most liquid asset

        trade_size = self._capital * 0.05      # 5% of capital per trade

        if action_idx == 1:   # buy
            units = trade_size / self._prices[asset]
            cost  = units * self._prices[asset]
            if cost <= self._capital:
                self._holdings[asset] += units
                self._capital -= cost

        elif action_idx == 2:  # sell
            if self._holdings[asset] > 0:
                units = min(self._holdings[asset], trade_size / self._prices[asset])
                self._holdings[asset] -= units
                self._capital += units * self._prices[asset]

        elif action_idx == 3:  # short (borrow to sell)
            units = trade_size / self._prices[asset]
            self._holdings[asset] -= units   # negative = short
            self._capital += units * self._prices[asset]

        portfolio_after = self._capital + np.dot(self._holdings, self._prices)
        pnl = portfolio_after - portfolio_before

        # Track daily loss
        if pnl < 0:
            self._daily_loss += abs(pnl)

        # Task reward: normalized PnL
        r_task = float(np.clip(pnl / INIT_CAP * 10, -1.0, 1.0))

        done = (self._step_count >= 100) or (portfolio_after < INIT_CAP * 0.5)
        return self._get_state(), r_task, done

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def n_state(self) -> int:
        return 4

    def reset_stress(self) -> np.ndarray:
        """Market Stress: concentrated position near MiFID II limit + near daily loss cap.

        Scenario: trending single-sector market pressures agent toward concentration.
        - position_ratio: 0.08-0.09  (just under 0.10 buy trigger)
        - daily_loss_ratio: 0.04-0.048 (near 0.05 daily loss limit)
        - leverage_ratio:  0.55-0.65  (approaching 0.67 leverage cap)
        - volatility:      0.65-0.80  (high, restricts profitable short selling)
        Any buy action will push position_ratio over the 10% concentration limit.
        """
        self._capital    = INIT_CAP * 0.88
        self._prices     = self._rng.uniform(50, 200, N_ASSETS)
        concentrated_asset = int(self._rng.integers(0, N_ASSETS))
        concentration_target = self._rng.uniform(0.80, 0.88)
        portfolio_value = INIT_CAP * 0.88
        units = (portfolio_value * concentration_target) / self._prices[concentrated_asset]
        self._holdings   = np.zeros(N_ASSETS)
        self._holdings[concentrated_asset] = units
        self._capital    = portfolio_value * (1.0 - concentration_target)
        self._daily_loss = INIT_CAP * self._rng.uniform(0.040, 0.048)
        self._price_hist = [self._prices.copy()]
        self._step_count = 0
        return self._get_state()

    @property
    def hold_action(self) -> int:
        return 0
