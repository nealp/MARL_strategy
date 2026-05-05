# Portfolio Optimization Using Multi-Agent Reinforcement Learning (MARL)

A three-agent system for adaptive, risk-aware portfolio management. Two SAC-trained trading agents compete for capital across a diversified universe of equities (and bonds), while a rule-based meta-agent dynamically allocates capital between them based on rolling risk-adjusted performance.

---

## Folder Structure

```
MARL_strategy/
│
├── config.py                     # Central config — all tunable constants live here
│
├── data/
│   ├── __init__.py
│   └── pipeline.py               # Stages 1–3: data download, feature engineering, train/val/test split
│
├── environments/
│   ├── __init__.py
│   ├── base_env.py               # Shared gymnasium environment (observation, action, portfolio accounting)
│   ├── agent1_env.py             # Agent 1 environment — risk-averse reward function
│   └── agent2_env.py             # Agent 2 environment — return-maximizing reward function
│
├── training/
│   ├── train_sac.py              # Stage 6: SAC training with Sharpe-based validation callback
│   └── save_agent_returns.py     # Stage 6b: test rollout, saves per-day returns and weights
│
├── analysis/
│   ├── backtest.py               # Stages 8–9: meta-agent, performance metrics, plots
│   ├── optimize_meta.py          # Meta-agent parameter grid search
│   └── tearsheet.py              # Stage 10b: quantstats tearsheets and heatmaps
│
├── results/                      # CSV outputs (returns, weights, metrics summary)
├── plots/                        # Generated PNG visualizations
├── models/                       # Saved SAC model checkpoints
├── test_stages.py                # Smoke test: runs stages 1–5 end-to-end with random actions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Trading Strategy

### Overview
The system consists of three components:

1. **Agent 1 (Risk-Averse)** — A SAC agent trained to maximize log returns while penalizing drawdowns and portfolio volatility. Manages 26 assets including IEF (7–10y US Treasuries) and SHY (1–3y, cash-like) as safe-haven instruments.
2. **Agent 2 (Return-Maximizing)** — A SAC agent trained with an asymmetric reward that amplifies positive returns. Manages 24 equities with no bond exposure.
3. **Meta-Agent (Capital Allocator)** — A hard-coded rule-based allocator that computes a rolling Sharpe ratio for each agent every 20 trading days and shifts capital toward the better-performing agent using fixed discrete thresholds.

### Why Two Agents?
Both agents use the same algorithm (SAC) and a similar asset universe. The behavioral divergence comes entirely from reward shaping:
- Agent 1 is structurally pushed toward **capital preservation** — it can rotate into IEF/SHY during downturns, penalizes drawdowns quadratically, and avoids concentration risk.
- Agent 2 is structurally pushed toward **return maximization** — it concentrates in momentum names, stays fully invested in equities, and accepts higher drawdowns in exchange for higher expected returns.

This creates complementary behavior across market regimes: Agent 2 outperforms in bull markets; Agent 1 outperforms (or loses less) in drawdowns. The meta-agent exploits this by dynamically shifting capital toward whoever is performing better on a risk-adjusted basis.

### Asset Universes

| Sector | Agent 1 (26 assets) | Agent 2 (24 assets) |
|---|---|---|
| Tech | AAPL, MSFT, NVDA, GOOG, NFLX, DIS | AAPL, MSFT, NVDA, GOOG, NFLX, DIS |
| Financials | GS, JPM, V | GS, JPM, V |
| Healthcare | AMGN, JNJ, UNH | AMGN, JNJ, UNH |
| Consumer | AMZN, HD, MCD, KO, COST, PG | AMZN, HD, MCD, KO, COST, PG |
| Industrial | CAT, HON, WM | CAT, HON, WM |
| Energy | COP, XOM, SLB | COP, XOM, SLB |
| Bonds | IEF, SHY | — |

### Reward Functions

**Agent 1 (Risk-Averse):**
```
reward = log_return
         − λ_drawdown   × (drawdown_from_peak)²
         − λ_volatility × rolling_portfolio_volatility
         − turnover_cost × turnover
```
- `λ_drawdown = 2.5` — squared penalty for non-linear risk aversion (small dips: minimal cost; large crashes: severe cost)
- `λ_volatility = 1.5` — penalizes rolling 20-day portfolio volatility
- `turnover_cost = 0.001` (~10 bps per unit of turnover)

**Agent 2 (Return-Maximizing):**
```
reward = (log_return × 3.0 if log_return > 0 else log_return)
         − (turnover_cost × 2.5) × turnover
```
- Positive returns are amplified 3× to encourage riding winning positions
- Turnover penalty is 2.5× that of Agent 1 to discourage short-term churn
- No drawdown or volatility penalties

### The SAC Algorithm
Soft Actor-Critic (SAC) is an off-policy, model-free RL algorithm built for continuous action spaces. It maximizes a modified objective that combines reward with a policy entropy bonus:

```
J(π) = Σ E[ r_t + α · H(π(·|s_t)) ]
```

The entropy term encourages the agent to explore diverse portfolio allocations during training rather than collapsing to a single concentrated strategy. SAC maintains two critic networks (Q1 and Q2) and always uses the more pessimistic estimate, which reduces overestimation bias — important given the noisy nature of financial returns.

### Daily Rebalancing
At each trading day, the agent:
1. Receives its observation vector (normalized price features + current portfolio weights)
2. Outputs a new portfolio weight vector via its learned policy (softmax over raw logits)
3. The environment executes the rebalance and computes the reward
4. The transition is stored in a replay buffer (size: 600,000) for off-policy learning

### Meta-Agent Capital Allocation

The meta-agent is hard-coded with discrete thresholds — no gradient or learned allocation. Every `META_REBALANCE_DAYS = 20` trading days, it computes the rolling Sharpe ratio for each agent over the past `META_LOOKBACK = 30` days, subtracts a drawdown penalty, and applies the following rules:

| Sharpe difference (Agent 1 vs Agent 2) | Allocation |
|---|---|
| diff < 0.15 | 50 / 50 |
| 0.15 ≤ diff < 0.40 | 60 / 40 toward better agent |
| diff ≥ 0.40 | 75 / 25 toward better agent |

Additional parameters:
- `META_DD_PENALTY = 3.0` — subtracts `3.0 × current_drawdown_pct` from the lagging agent's score before comparison, making the allocator more aggressive about moving capital away from a drawdown agent
- Allocation is held fixed between rebalance dates

**Why hard-coded thresholds instead of a gradient/softmax allocator?**
Earlier iterations used a softmax temperature (k=10) to produce a smooth, continuous allocation curve. After testing, this approach was too sensitive to short-term noise in Sharpe estimates — small swings in the signal caused large allocation shifts. The discrete threshold approach is more stable: it ignores noise when agents are performing similarly (diff < 0.15 → hold 50/50) and only commits capital when one agent has a clear edge.

### Feature Engineering
- **Log returns** at 1, 5, 20, 60, 120, and 252-day horizons per asset
- **Rolling 20-day volatility** per asset
- **Rolling z-score normalization** with a 252-day window (1 trading year) — applied column-by-column, using only past data to prevent lookahead bias

### Data Split

| Split | Period | Trading Days |
|---|---|---|
| Training | 2010–2018 | ~2,015 |
| Validation | 2018–2020 | ~502 |
| **Test** | **2020–2024** | **~1,007** |

The test period was held out completely during all training and hyperparameter decisions. It includes the COVID crash (2020 H1), the 2020–2021 bull run, the 2022 rate-hike bear market, and the 2023–2024 recovery.

---

## Final Results — Test Period 2020–2024

### Overall Performance

| Strategy | Total Return | Ann. Return | Sharpe | Max Drawdown |
|---|---|---|---|---|
| **Meta-Agent** | **+193.1%** | **+24.1%** | **1.064** | -32.3% |
| Agent 1 (Risk-Averse) | +171.3% | +22.2% | 1.035 | -32.2% |
| Agent 2 (Return-Max) | +177.1% | +22.7% | 0.954 | -32.4% |
| SPY (Buy & Hold) | +99.7% | +14.9% | 0.659 | -33.7% |
| Equal-Weight | +126.5% | +17.9% | 0.885 | -29.9% |
| 60/40 (SPY + IEF) | +99.7% | +14.9% | 0.659 | -33.7% |

### 6-Month Rolling Window Returns

| Window | Meta-Agent | Agent 1 | Agent 2 | SPY |
|---|---|---|---|---|
| 2020 H1 (Jan–Jun) | +0.6% | -5.6% | +3.0% | -4.1% |
| 2020 H2 (Jul–Dec) | +24.4% | +23.4% | +25.8% | +22.3% |
| 2021 H1 (Jan–Jun) | +21.9% | +23.4% | +18.8% | +15.2% |
| 2021 H2 (Jul–Dec) | +13.6% | +14.9% | +11.8% | +11.7% |
| 2022 H1 (Jan–Jun) | -11.1% | -9.1% | -18.3% | -20.0% |
| 2022 H2 (Jul–Dec) | +14.6% | +12.3% | +15.4% | +2.3% |
| 2023 H1 (Jan–Jun) | +15.7% | +15.4% | +15.4% | +16.8% |
| 2023 H2 (Jul–Dec) | +11.3% | +12.9% | +11.2% | +8.0% |
| 2024 H1 (Jan–Jun) | +18.4% | +15.1% | +21.1% | +15.2% |
| 2024 H2 (Jul–Dec) | +9.0% | +7.3% | +9.9% | +11.2% |

Notable observations:
- **2022 H1** is the best showcase of the meta-agent's value: Agent 1's defensive rotation limited losses to -9.1% while Agent 2 fell -18.3% and SPY dropped -20.0%. The meta-agent cut Agent 2 exposure and landed at -11.1%.
- **2020 H1** (COVID crash): The meta-agent stayed near break-even (+0.6%) while SPY fell -4.1%.
- **2024**: Both agents trail SPY in H2, suggesting the strategy may underperform in a strong momentum-driven market where concentration in a few names dominates.

---

## Key Parameters

All tunable constants live in `config.py`. Final values used in this run:

```python
# Reward shaping
LAMBDA_DRAWDOWN   = 2.5    # Agent 1 squared drawdown penalty weight
LAMBDA_VOLATILITY = 1.5    # Agent 1 rolling volatility penalty weight
TURNOVER_COST     = 0.001  # ~10 bps per unit of turnover (both agents; Agent 2 applies 2.5×)

# Training
EPISODE_LENGTH    = 252    # trading days per episode (~1 year)
# Replay buffer size: 600,000 transitions (set in training/train_sac.py)

# Meta-agent (set in analysis/backtest.py)
META_LOOKBACK       = 30   # rolling window for Sharpe computation (days)
META_REBALANCE_DAYS = 20   # rebalance frequency (trading days)
META_DD_PENALTY     = 3.0  # drawdown penalty weight applied to lagging agent's score
META_LO_THRESH      = 0.15 # Sharpe diff below this → 50/50 split
META_HI_THRESH      = 0.40 # Sharpe diff above this → 75/25 split
META_W_LO           = 0.60 # allocation to better agent when diff is in middle band
META_W_HI           = 0.75 # allocation to better agent when diff is large
```

---

## Full Pipeline

| Stage | Description | Tools |
|---|---|---|
| 1 | Data collection — adjusted close prices 2009–2024 | yfinance |
| 2 | Feature engineering — log returns (6 horizons) + 20-day volatility | pandas, numpy |
| 3 | Train / val / test split | pandas |
| 4 | Environment build — custom gymnasium environments | gymnasium |
| 5 | Reward implementation — per-agent reward functions | Python |
| 6 | Agent training — SAC, best model selected by validation Sharpe | stable-baselines3, PyTorch |
| 6b | Test rollout — saves per-day returns and weights to CSV | Python |
| 7 | Hyperparameter tuning — λ values and SAC params | optuna, TensorBoard |
| 8 | Meta-agent — discrete-threshold Sharpe capital allocator | pandas, numpy |
| 9 | Backtesting — metrics summary + 6-month rolling window returns | quantstats, pandas |
| 10 | Analysis & visualization — portfolio values, drawdowns, weights, rolling plots | matplotlib, seaborn |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/nealp/MARL_strategy.git
cd MARL_strategy
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> This will take a few minutes — PyTorch is a large package.

### 4. Run the smoke test
```bash
python test_stages.py
```

This verifies stages 1–5 are working. It downloads ~15 years of price data (requires internet), builds features, and runs one random-action episode per agent.

### Requirements

| Package | Version | Purpose |
|---|---|---|
| yfinance | ≥ 0.2.40 | Price data download |
| gymnasium | ≥ 0.29.1 | Trading environments |
| stable-baselines3 | ≥ 2.3.0 | SAC implementation |
| torch | ≥ 2.2.0 | Neural network backend |
| pandas | ≥ 2.2.0 | Data manipulation |
| numpy | ≥ 1.26.0 | Numerical computation |
| optuna | ≥ 3.6.0 | Hyperparameter tuning |
| tensorboard | ≥ 2.17.0 | Training visualization |
| quantstats | ≥ 0.0.62 | Portfolio performance metrics |
| matplotlib | ≥ 3.8.0 | Plotting |
| seaborn | ≥ 0.13.0 | Statistical visualization |
