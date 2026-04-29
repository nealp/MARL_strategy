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

    def compute_reward(self, log_return, turnover, new_weights):
        """
        Computes the reward for Agent 2 (Return-Maximizing).
        """
        current_log_return = log_return
        current_turnover = turnover
        
        # 1. Stop the Day-Trading: Multiply turnover cost by 2.5
        # This forces Agent 2 to hold its positions rather than chasing daily noise.
        turnover_penalty = (self.turnover_cost * 2.5) * current_turnover
        
        # 2. Asymmetric Momentum Reward
        # Amplify positive returns BEFORE fees are subtracted. 
        # This teaches the agent: "Holding stocks that go up is highly rewarded, 
        # but churning the portfolio will destroy those gains."
        if current_log_return > 0:
            reward = (current_log_return * 3.0) - turnover_penalty
        else:
            reward = current_log_return - turnover_penalty
            
        return reward
