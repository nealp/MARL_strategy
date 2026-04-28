# ─────────────────────────────────────────────
# environments/agent1_env.py  —  Stage 5 (Agent 1)
#
# Risk-averse agent.  Assets: 12 equities + TLT.
#
# Reward = log_return
#          − λ_drawdown   * drawdown_from_peak
#          − λ_volatility * rolling_portfolio_volatility
#          − turnover_cost * turnover
#
# The drawdown and volatility penalties push the agent toward:
#   • holding TLT during equity drawdowns (safe haven rotation)
#   • maintaining diversified weights (vol penalty punishes concentration)
#   • cutting losing positions quickly (drawdown penalty makes holding them costly)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd

from config import LAMBDA_DRAWDOWN, LAMBDA_VOLATILITY, TURNOVER_COST, VOLATILITY_WINDOW
from environments.base_env import BaseTradingEnv


class Agent1Env(BaseTradingEnv):

    def __init__(
        self,
        features:         pd.DataFrame,
        prices:           pd.DataFrame,
        lambda_drawdown:  float = LAMBDA_DRAWDOWN,
        lambda_volatility: float = LAMBDA_VOLATILITY,
        turnover_cost:    float = TURNOVER_COST,
        **kwargs,
    ):
        super().__init__(features, prices, **kwargs)

        self.lambda_drawdown   = lambda_drawdown
        self.lambda_volatility = lambda_volatility
        self.turnover_cost     = turnover_cost

        # Keeps the last 20 daily log returns to compute rolling portfolio volatility
        self._return_history: list[float] = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._return_history = []
        return obs, info

    def compute_reward(self, log_return, turnover, new_weights):
        """
        Computes the reward for Agent 1 (risk-averse).
        """
        current_log_return = float(log_return)
        current_turnover = float(turnover)
        current_drawdown = float(self.drawdown)

        # Rolling portfolio volatility over recent daily log returns.
        self._return_history.append(current_log_return)
        if len(self._return_history) > VOLATILITY_WINDOW:
            self._return_history = self._return_history[-VOLATILITY_WINDOW:]

        if len(self._return_history) >= 2:
            current_port_vol = float(np.std(self._return_history))
        else:
            current_port_vol = 0.0

        # Squared drawdown makes large underwater periods much more expensive.
        drawdown_penalty = self.lambda_drawdown * (current_drawdown ** 2)
        volatility_penalty = self.lambda_volatility * current_port_vol
        turnover_penalty = self.turnover_cost * current_turnover

        reward = current_log_return - drawdown_penalty - volatility_penalty - turnover_penalty
        return float(reward)