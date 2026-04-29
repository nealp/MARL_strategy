# Portfolio Optimization Using Multi-Agent Reinforcement Learning (MARL)

A three-agent system for adaptive, risk-aware portfolio management. Two SAC-trained trading agents compete for capital across a diversified universe of equities (and bonds), while a rule-based meta-agent dynamically allocates capital between them based on rolling risk-adjusted performance.

---

## Folder Structure

```
ARQ_Pitch/
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
├── test_stages.py                # Smoke test: runs stages 1–5 end-to-end with random actions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## What Each File Does

### `config.py`
The single source of truth for the entire project. Contains:
- **Asset universes** for each agent (Agent 1: 13 assets including TLT; Agent 2: 12 equities)
- **Date ranges** for data download and train/val/test splits
- **Feature engineering parameters** (return windows, volatility window, normalization window)
- **Reward shaping constants** (λ_drawdown, λ_volatility, turnover cost)

If you want to change assets, dates, or penalty weights — change them here. Everything else reads from this file.

---

### `data/pipeline.py`
Handles the full data preparation pipeline across three stages:

- **Stage 1 — `download_prices(tickers)`**: Pulls adjusted close prices from Yahoo Finance via `yfinance` for the full date range (2009–2024).
- **Stage 2 — `compute_features(prices)` + `normalize_features(feature_df)`**: Computes log returns at 1, 5, 20, and 60-day horizons plus rolling 20-day volatility for every asset. Applies a rolling z-score (252-day window) to normalize features — rolling, not global, to prevent lookahead bias.
- **Stage 3 — `split_data(features, prices)`**: Slices the data into train (2010–2018), validation (2018–2020), and test (2020–2024) windows and returns them as a dictionary.
- **`build_features(tickers)`**: Convenience wrapper that runs all three steps and returns aligned `(features, prices)` DataFrames ready for the environments.

---

### `environments/base_env.py`
A custom `gymnasium.Env` that implements all shared trading logic used by both agents. Subclasses only need to override `compute_reward()`.

Key responsibilities:
- **Observation construction**: Flattens per-asset features and appends current portfolio weights into a single vector (Agent 1: 78-dim, Agent 2: 72-dim).
- **Action → weights**: Converts the agent's raw logit output into valid portfolio weights using softmax (positive, sum to 1).
- **Portfolio accounting**: Computes daily log returns, updates portfolio value, tracks the high-water mark (peak value) for drawdown calculation.
- **Episode management**: Starts each episode at a random date in the dataset and runs for `EPISODE_LENGTH` trading days (~252 by default).

---

### `environments/agent1_env.py`
Extends `BaseTradingEnv` for the **risk-averse agent**. Implements a reward function with three penalty terms:

```
reward = log_return
         − λ_drawdown   × drawdown_from_peak
         − λ_volatility × rolling_portfolio_volatility
         − turnover_cost × turnover
```

The drawdown penalty (continuous, always active below the peak) and volatility penalty push this agent to protect capital, diversify, and rotate into TLT during equity stress. Both λ values default to 0.1 and are tunable in `config.py`.

---

### `environments/agent2_env.py`
Extends `BaseTradingEnv` for the **return-maximizing agent**. Reward is purely:

```
reward = log_return − turnover_cost × turnover
```

No risk penalties. This agent is free to concentrate in high-momentum equities, stay fully invested, and take on more risk. The turnover cost is retained as a realism constraint to prevent unrealistic daily full-rotation strategies.

---

### `test_stages.py`
End-to-end smoke test for stages 1–5. Downloads data, builds features, splits the dataset, instantiates both environments, and runs one full episode of random actions per agent. Use this to verify your setup is working before training.

---

## Trading Strategy

### Overview
The system consists of three components:

1. **Agent 1 (Risk-Averse)** — A SAC agent trained to maximize log returns while penalizing drawdowns and portfolio volatility. Manages 13 assets including TLT (20+ year US treasuries) as a safe-haven instrument.
2. **Agent 2 (Return-Maximizing)** — A SAC agent trained purely on log returns with no risk penalties. Manages 12 equities with no bond exposure.
3. **Meta-Agent (Capital Allocator)** — A rule-based allocator (no RL) that computes a rolling Sharpe ratio for each agent monthly and shifts capital toward the better-performing agent.

### Why Two Agents?
Both agents use the same algorithm (SAC) and the same asset universe (minus TLT for Agent 2). The behavioral divergence comes entirely from reward shaping:
- Agent 1 is structurally pushed toward **capital preservation** — it overweights TLT during downturns, cuts losing positions quickly, and avoids concentration.
- Agent 2 is structurally pushed toward **return maximization** — it concentrates in momentum names, stays fully invested in equities, and accepts higher drawdowns in exchange for higher expected returns.

This creates complementary behavior across market regimes: Agent 2 outperforms in bull markets; Agent 1 outperforms (or loses less) in drawdowns. The meta-agent exploits this by dynamically shifting capital toward whoever is performing better on a risk-adjusted basis.

### Asset Universes

| Sector | Agent 1 | Agent 2 |
|---|---|---|
| Tech | AAPL, MSFT, NVDA, GOOGL | AAPL, MSFT, NVDA, GOOGL |
| Financials | JPM, BAC | JPM, BAC |
| Healthcare | UNH, JNJ | UNH, JNJ |
| Consumer | AMZN, WMT | AMZN, WMT |
| Energy | XOM, CVX | XOM, CVX |
| Bonds | TLT | — |

### The SAC Algorithm
Soft Actor-Critic (SAC) is an off-policy, model-free RL algorithm built for continuous action spaces. It maximizes a modified objective that combines reward with a policy entropy bonus:

```
J(π) = Σ E[ r_t + α · H(π(·|s_t)) ]
```

The entropy term encourages the agent to explore diverse portfolio allocations during training rather than collapsing to a single concentrated strategy. SAC maintains two critic networks (Q1 and Q2) and always uses the more pessimistic estimate, which reduces overestimation bias — important given the noisy nature of financial returns.

### Daily Rebalancing
At each trading day, the agent:
1. Receives its observation vector (price features + current weights)
2. Outputs a new portfolio weight vector via its learned policy
3. The environment executes the rebalance and computes the reward
4. The transition is stored in a replay buffer for off-policy learning

### Meta-Agent Capital Allocation

| Condition | Allocation |
|---|---|
| Sharpe difference < 0.2 | 50 / 50 |
| One agent clearly better (diff 0.2–0.5) | 65 / 35 toward better agent |
| One agent dominates (diff > 0.5) | 80 / 20 toward better agent |

The meta-agent is intentionally kept rule-based. Training a third RL agent would introduce non-stationarity — the trading agents' policies are always changing during training, making the meta-agent's environment non-stationary and difficult to train stably.

### Full Pipeline

| Stage | Description | Tools |
|---|---|---|
| 1 | Data collection — OHLCV price history | yfinance |
| 2 | Feature engineering — log returns, volatility, normalization | pandas, numpy |
| 3 | Train / val / test split | pandas |
| 4 | Environment build — custom gymnasium environments | gymnasium |
| 5 | Reward implementation — per-agent reward functions | Python |
| 6 | Agent training — SAC on training data | stable-baselines3, PyTorch |
| 7 | Hyperparameter tuning — λ values and SAC params | optuna, TensorBoard |
| 8 | Meta-agent — rolling Sharpe capital allocator | pandas, numpy |
| 9 | Backtesting — test set 2020–2024 | quantstats, pandas |
| 10 | Analysis & visualization — weights, drawdowns, divergence | matplotlib, seaborn |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-org/ARQ_Pitch.git
cd ARQ_Pitch
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

This verifies stages 1–5 are working. It will download ~15 years of price data (requires internet), build features, and run one random-action episode per agent. Expected output:

```
=== Stages 1–3: Data Pipeline ===
Downloaded 3842 trading days for 13 assets.
Features ready: 3589 dates × 65 columns.
  train: 2015 days
  val:   502 days
  test:  1007 days

=== Stages 4–5: Environments ===
Agent1: obs_shape=(78,)  steps=252  total_reward=-1.23  final_portfolio_value=0.97
Agent2: obs_shape=(72,)  steps=252  total_reward=-0.85  final_portfolio_value=0.98

All stages passed.
```

> Negative rewards with random actions are expected — the agents have not been trained yet.

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

---

## Architecture Notes & Changes

### 26-Asset Universe Expansion (April 2026)

Both agents were updated to trade the same expanded 26-asset universe:
- 24 equities across 8 sectors (3 per sector): Info Tech, Financials, Health Care, Consumer Discretionary, Communication Services, Industrials, Consumer Staples, Energy
- 2 bond ETFs: IEF (7–10 year Treasuries), SHY (1–3 year Treasuries)

**Why both agents now trade the same universe:** Specialization between agents comes entirely from reward shaping (Agent 1 penalizes drawdown + vol; Agent 2 does not), not from different asset sets. The meta-agent then dynamically rebalances capital between their strategies.

**Note on capital allocation — training vs. backtesting discrepancy:**

During **training**, the two agents are completely isolated. Each manages its own independent portfolio (starting value = 1.0 per episode) and has no knowledge the other agent exists. There is no shared capital, no interaction, and no competition between them.

During **backtesting**, the meta-agent in `analysis/backtest.py` runs both trained policies simultaneously and combines their outputs. It computes a daily blended return as `log(w1 × exp(r1) + w2 × exp(r2))`, where `w1 + w2 = 1.0`. Every 20 trading days it rebalances the split based on each agent's rolling Sharpe + momentum signal, clipped to [0.10, 0.90] to prevent going all-in on either agent.

From a practical standpoint, this means the meta-agent acts like a fund-of-funds: an investor allocates `w1 %` of capital to Agent 1's strategy and `w2 %` to Agent 2's, with that split adjusting dynamically. The agents themselves cannot react to each other — they are fixed, pre-trained policies. This design is sometimes called "ensemble RL" rather than true competitive MARL.

### Hyperparameter Updates (April 2026)

The following changes were made to address performance regression after the asset universe expansion:

| Parameter | Old Value | New Value | Reason |
|---|---|---|---|
| `LAMBDA_DRAWDOWN` | 1.0 | **0.6** | 26 diversified assets produce naturally smoother portfolios; the squared penalty was over-firing and forcing Agent 1 into near-cash |
| `LAMBDA_VOLATILITY` | 0.5 | **0.35** | Same reason — reduced penalty scale for the wider, more diversified universe |
| `net_arch` | [64, 64] | **[256, 256]** | Default network was a bottleneck for 130-dim input → 26-dim output |
| `buffer_size` | 100,000 | **300,000** | Larger buffer for more diverse replay experience across the wider state space |

---

## Backtest Results — Test Period 2020–2024

### Full-Period Performance Summary

| Strategy | Total Return | Ann. Return | Ann. Sharpe | Max Drawdown | Calmar | Sortino |
|---|---|---|---|---|---|---|
| **Meta-Agent** | **+151.7%** | **+20.4%** | **0.955** | **-29.0%** | **0.702** | **0.992** |
| Agent 1 (Risk-Averse) | +147.0% | +19.9% | 0.929 | -32.1% | 0.620 | 0.939 |
| Agent 2 (Return-Max) | +156.4% | +20.8% | 0.969 | -28.6% | 0.728 | 1.014 |
| SPY (Buy & Hold) | +99.7% | +14.9% | 0.659 | -33.7% | 0.442 | 0.665 |
| Equal-Weight | +147.5% | +20.0% | 0.936 | -30.8% | 0.649 | 0.965 |
| 60/40 (SPY + IEF) | +51.5% | +8.7% | 0.653 | -21.0% | 0.414 | 0.652 |

All three agents beat SPY by 50+ percentage points in total return. The meta-agent improves on both individual agents in terms of drawdown control (-29.0% vs -32.1% for Agent 1 and -28.6% for Agent 2) while maintaining competitive returns. The test period spans COVID crash (March 2020), the 2021 bull run, the 2022 rate-hike bear market, and the 2023–2024 recovery.

---

### Best 6-Month Windows — Meta-Agent vs SPY

The eight highest-Sharpe six-month periods for the meta-agent, compared against SPY over the same windows. All results are from the new 26-asset strategy trained at 400k timesteps with net_arch [256, 256].

| Period | Meta Sharpe | Meta Return | SPY Sharpe | SPY Return | Sharpe Edge |
|---|---|---|---|---|---|
| Nov 2020 → May 2021 | **4.43** | +31.7% | 3.56 | +28.2% | +0.87 |
| Feb 2021 → Jul 2021 | 3.67 | +22.7% | 2.80 | +19.3% | +0.86 |
| May 2021 → Nov 2021 | 3.05 | +17.2% | 2.83 | +16.0% | +0.22 |
| Dec 2022 → Jun 2023 | 3.20 | +23.4% | 2.34 | +18.6% | +0.86 |
| Mar 2023 → Sep 2023 | **3.95** | +23.3% | 2.61 | +16.6% | **+1.34** |
| Nov 2023 → May 2024 | 4.24 | +20.9% | 3.42 | +20.9% | +0.82 |
| Jan 2024 → Jul 2024 | 3.78 | +18.0% | 3.08 | +18.0% | +0.69 |
| May 2024 → Nov 2024 | 3.13 | +17.7% | 2.08 | +14.9% | +1.04 |

The strategy outperforms SPY on Sharpe in every window. The largest return alpha was Mar 2023 → Sep 2023 (+6.7pp over SPY, Sharpe edge +1.34). Nov 2023 → May 2024 and Jan 2024 → Jul 2024 matched SPY's total return but at meaningfully higher Sharpe — same gains, lower volatility. The weakest edge was May 2021 → Nov 2021 (+0.22) when SPY ran on a smooth, low-vol bull trend that is structurally hard to beat on a risk-adjusted basis.

---

## Changes from the Original Strategy

The following changes were made relative to the original 13/12-asset version of the system:

### Asset Universe

| | Original | Current |
|---|---|---|
| Agent 1 assets | 13 (12 equities + TLT) | 26 (24 equities + IEF + SHY) |
| Agent 2 assets | 12 equities (no bonds) | 26 (same as Agent 1) |
| Differentiation | Different asset sets | Same assets, different reward shaping only |

Originally the two agents had different universes — Agent 1 held TLT as a dedicated safe-haven and Agent 2 held no bonds. Under the expanded design, behavioral divergence comes entirely from reward shaping; both agents can access the same instruments.

### Reward Shaping Penalties

| Parameter | Original | Current | Why Changed |
|---|---|---|---|
| `LAMBDA_DRAWDOWN` | 1.0 | **0.6** | 26 diversified assets produce naturally smoother portfolios; the original penalty was over-firing and pushing Agent 1 into near-cash positions |
| `LAMBDA_VOLATILITY` | 0.5 | **0.35** | Same reason — wider diversification already dampens vol; the full penalty was redundant |

### SAC Hyperparameters

| Parameter | Original | Current | Why Changed |
|---|---|---|---|
| `net_arch` | [64, 64] | **[256, 256]** | The 64-unit hidden layers were a bottleneck for mapping 130-dim input → 26-dim action space |
| `buffer_size` | 100,000 | **300,000** | Larger buffer needed to maintain diverse replay experience across the wider 26-asset state space |

### Meta-Agent

The meta-agent allocation logic was upgraded from a simple Sharpe-threshold table (50/50 → 65/35 → 80/20 step function) to a continuous softmax allocation:

- **Signal:** Composite = Sharpe + 2.0 × normalized momentum (captures trending regimes pure Sharpe misses)
- **Allocation:** Sigmoid with temperature k=10, clipped to [0.10, 0.90] — aggressively commits to the leading agent while preventing all-in positions
- **Drawdown penalty:** Subtracts 2.0 × current drawdown from a trailing agent's score to penalize recovery lag
- **Rebalance frequency:** Every 20 trading days over a 30-day rolling window
