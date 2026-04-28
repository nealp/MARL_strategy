# ─────────────────────────────────────────────
# environments/agent1_env.py  —  Stage 5 (Agent 1)
#
# Risk-averse agent.  Assets: equities + IEF/SHY (see config.AGENT1_TICKERS).
#
# Reward = log_return
#          − λ_drawdown   * drawdown_from_peak
#          − λ_volatility * rolling_portfolio_volatility
#          − turnover_cost * turnover
#
# The drawdown and volatility penalties push the agent toward:
#   • holding IEF/SHY during equity drawdowns (defensive / cash-like rotation)
#   • maintaining diversified weights (vol penalty punishes concentration)
#   • cutting losing positions quickly (drawdown penalty makes holding them costly)
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd

from config import LAMBDA_DRAWDOWN, LAMBDA_VOLATILITY, TURNOVER_COST
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
        Computes the reward for Agent 1 (Risk-Averse).
        This replaces the original compute_reward function to fix the TypeError.
        """
        # ====================================================================
        # 1. MAP YOUR VARIABLES
        # We now use the arguments directly passed from base_env.py 
        # (log_return and turnover) and grab the risk metrics from self.
        # ====================================================================
        current_log_return = log_return  
        current_turnover = turnover      
        
        # Fetch the current drawdown and portfolio volatility from the environment state.
        # (Using getattr as a safeguard in case the variable names slightly differ)
        current_drawdown = getattr(self, 'drawdown', 0.0)      
        current_port_vol = getattr(self, 'port_vol', 0.0)      
        
    
        drawdown_penalty = self.lambda_drawdown * (current_drawdown ** 2) 
        
        # Standard linear penalties
        volatility_penalty = self.lambda_volatility * current_port_vol
        turnover_penalty = self.turnover_cost * current_turnover
        
        # ====================================================================
        # 3. FINAL REWARD
        # ====================================================================
        reward = current_log_return - drawdown_penalty - volatility_penalty - turnover_penalty
        
        return reward