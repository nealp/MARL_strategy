# ─────────────────────────────────────────────
# environments/base_env.py  —  Stage 4
#
# Shared trading environment logic for both agents.
# Handles: observation construction, action → weights, portfolio accounting.
# Subclasses only need to implement compute_reward().
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config import EPISODE_LENGTH


class BaseTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for daily portfolio rebalancing.

    Observation space:
        Flat vector of [asset_features... , current_weights...]
        Shape: (n_assets * n_features_per_asset + n_assets,)

    Action space:
        Raw logits in [-1, 1], one per asset.
        Converted to portfolio weights via softmax inside the env.
        Softmax guarantees weights are positive and sum to 1.

    Episode:
        Starts at a random date in the dataset, runs for EPISODE_LENGTH days.
        Resets peak_value each episode so the drawdown penalty is episode-relative.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,   # normalized features, MultiIndex cols (ticker, feat)
        prices:   pd.DataFrame,   # raw close prices, cols = tickers
        episode_length: int = EPISODE_LENGTH,
    ):
        super().__init__()

        # Align both DataFrames on shared dates (safety guard)
        shared = features.index.intersection(prices.index)
        self.features = features.loc[shared]
        self.prices   = prices.loc[shared]
        self.dates    = shared

        self.tickers        = prices.columns.tolist()
        self.n_assets       = len(self.tickers)
        self.episode_length = episode_length

        # Number of features per asset (same for every ticker)
        self.n_feat = len(self.features.columns.get_level_values(1).unique())

        # ── Spaces ───────────────────────────────────────────────────────────
        obs_dim = self.n_assets * self.n_feat + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Raw logits — SAC clips outputs to this range via tanh squashing
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Episode state (initialised properly in reset())
        self.current_step      = 0
        self.start_idx         = 0
        self.portfolio_weights = None
        self.portfolio_value   = None
        self.peak_value        = None
        self.drawdown          = 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _observation(self) -> np.ndarray:
        """Build flat observation: asset features (all tickers) + current weights."""
        date = self.dates[self.start_idx + self.current_step]
        row  = self.features.loc[date]   # Series with MultiIndex (ticker, feat)

        # Stack features in consistent ticker order, then flatten
        feat_vec = np.array([row[ticker].values for ticker in self.tickers]).flatten()

        return np.concatenate([feat_vec, self.portfolio_weights]).astype(np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax → valid portfolio weights (positive, sum = 1)."""
        e = np.exp(x - x.max())
        return e / e.sum()

    def _portfolio_step(
        self, new_weights: np.ndarray, old_weights: np.ndarray
    ) -> tuple[float, float]:
        """
        Advance one trading day:
          1. Compute portfolio log return using new_weights
          2. Update portfolio_value and peak_value
          3. Return (log_return, turnover)

        Turnover = Σ|w_new - w_old|, used by both agents to penalise excess trading.
        """
        idx          = self.start_idx + self.current_step
        curr_date    = self.dates[idx]
        prev_date    = self.dates[idx - 1]

        curr_prices  = self.prices.loc[curr_date,  self.tickers].values
        prev_prices  = self.prices.loc[prev_date,  self.tickers].values

        # Weighted return: each asset's price change times its portfolio weight
        asset_returns     = curr_prices / prev_prices
        portfolio_return  = float(np.dot(new_weights, asset_returns))
        log_return        = float(np.log(portfolio_return + 1e-8))

        turnover          = float(np.sum(np.abs(new_weights - old_weights)))

        # Update value and high-water mark
        self.portfolio_value *= portfolio_return
        self.peak_value       = max(self.peak_value, self.portfolio_value)

        return log_return, turnover

    # ── Subclass interface ────────────────────────────────────────────────────

    def compute_reward(
        self,
        log_return:  float,
        turnover:    float,
        new_weights: np.ndarray,
    ) -> float:
        """Override in each agent's environment to define its reward function."""
        raise NotImplementedError

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random episode start — leave room for a full episode at the end
        max_start      = len(self.dates) - self.episode_length - 1
        self.start_idx = int(self.np_random.integers(0, max_start))

        self.current_step      = 0
        self.portfolio_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.portfolio_value   = 1.0
        self.peak_value        = 1.0
        self.drawdown          = 0.0

        return self._observation(), {}

    def step(self, action: np.ndarray):
        old_weights  = self.portfolio_weights.copy()
        new_weights  = self._softmax(action).astype(np.float32)

        self.current_step += 1

        log_return, turnover = self._portfolio_step(new_weights, old_weights)
        self.drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        reward               = self.compute_reward(log_return, turnover, new_weights)

        self.portfolio_weights = new_weights

        done = self.current_step >= self.episode_length
        obs  = self._observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "log_return":       log_return,
            "turnover":         turnover,
            "portfolio_value":  self.portfolio_value,
            # Current drawdown from peak — useful for logging even for Agent 2
            "drawdown":         self.drawdown,
        }

        return obs, float(reward), done, False, info
