# ─────────────────────────────────────────────
# environments/agent2_env.py  —  Stage 5 (Agent 2)
#
# Return-maximizing agent.  Equities from config.AGENT2_TICKERS (no bond ETFs); no drawdown/vol penalties.
#
# Reward = log_return − turnover_cost * turnover
#
# No drawdown or volatility penalties.  The agent is free to concentrate
# in high-momentum names, stay fully invested in equities, and take on
# as much risk as the expected return justifies.
#
# The only friction is the turnover cost, which prevents the degenerate
# strategy of rotating 100% of the portfolio every single day.
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd

from config import TURNOVER_COST
from environments.base_env import BaseTradingEnv


class Agent2Env(BaseTradingEnv):

    def __init__(
        self,
        features:      pd.DataFrame,
        prices:        pd.DataFrame,
        turnover_cost: float = TURNOVER_COST,
        **kwargs,
    ):
        super().__init__(features, prices, **kwargs)
        self.turnover_cost = turnover_cost

    def compute_reward(
        self,
        log_return:  float,
        turnover:    float,
        new_weights: np.ndarray,
    ) -> float:
        """
        r = log_return − turnover_cost * turnover

        Tuning guide:
          turnover_cost too high → agent barely rebalances (passive index-like behavior)
          turnover_cost too low  → agent over-trades on noise
          0.001 (~10 bps) is a reasonable starting point for daily rebalancing
        """
        return float(log_return - self.turnover_cost * turnover)
