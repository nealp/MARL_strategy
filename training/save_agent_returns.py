"""
Stage 6b — Run a trained SAC agent over the test period and save results to CSV.

Usage
-----
    # Use best model (recommended):
    python -m training.save_agent_returns --agent 1 --model-path ./models/agent1_best/best_model.zip
    python -m training.save_agent_returns --agent 2 --model-path ./models/agent2_best/best_model.zip

    # Use final model:
    python -m training.save_agent_returns --agent 1 --model-path ./models/agent1_final.zip

Output files (written to ./results/)
-------------------------------------
    agent{N}_test_returns.csv  — columns: date, log_return, portfolio_value
    agent{N}_test_weights.csv  — columns: date, <ticker1>, <ticker2>, …

These CSVs are the input to the meta-agent (Stage 8).  Keep them on the test
split (2020-01-01 to 2024-12-31) so there is no lookahead into training data.

Sequential rollout design
--------------------------
The base environment is episodic: reset() picks a random start date and runs
for EPISODE_LENGTH steps.  For backtesting we need a single sequential pass
through the entire test period instead.

We achieve this by setting episode_length = len(test_dates) - 2 at env
construction time.  With this value:

    max_start = len(dates) - episode_length - 1 = 1

so np_random.integers(0, 1) always returns 0, forcing start_idx = 0.

TODO: This relies on undocumented internal env behavior (the max_start
formula and the exclusive upper bound of np_random.integers).  If BaseTradingEnv
adds a max_start clamp or changes the RNG call, this breaks silently.  A cleaner
fix would be to add a `start_from=0` parameter to BaseTradingEnv.reset().
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import SAC
except ImportError:
    print(
        "ERROR: stable_baselines3 is not installed.\n"
        "Run:   pip install -r requirements.txt"
    )
    raise SystemExit(1)

from config import AGENT1_TICKERS, AGENT2_TICKERS
from data.pipeline import build_features, split_data
from environments.agent1_env import Agent1Env
from environments.agent2_env import Agent2Env


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_agent_config(agent_num: int) -> dict:
    if agent_num == 1:
        return {"tickers": AGENT1_TICKERS, "env_cls": Agent1Env}
    if agent_num == 2:
        return {"tickers": AGENT2_TICKERS, "env_cls": Agent2Env}
    raise ValueError(f"--agent must be 1 or 2, got {agent_num}")


def make_sequential_env(env_cls, features: pd.DataFrame, prices: pd.DataFrame):
    """
    Construct an environment that walks the given dataset from start to finish
    in a single episode rather than starting at a random date.

    Sets episode_length = len(dates) - 2 so that:
        max_start = len(dates) - episode_length - 1 = 1
        np_random.integers(0, 1) == 0  (always)

    The resulting episode covers dates[1] through dates[len(dates)-2], capturing
    len(dates)-2 daily returns.  The first and last dates are dropped because
    _portfolio_step needs both a current and a previous price row.

    See module docstring for the known fragility of this approach.
    """
    n_dates = len(features)
    if n_dates < 4:
        raise ValueError(
            f"Test dataset has only {n_dates} rows — too small for a sequential rollout."
        )
    episode_length = n_dates - 2
    return env_cls(features, prices, episode_length=episode_length)


# ── Core rollout ──────────────────────────────────────────────────────────────

def run_agent_on_test(
    model: SAC,
    env,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roll out the trained agent deterministically through the entire test period.

    Returns
    -------
    returns_df : DataFrame with columns [date, log_return, portfolio_value]
    weights_df : DataFrame with columns [date, <ticker1>, …, <tickerN>]

    Date convention
    ---------------
    The date recorded for each row is the arrival date of the price used in that
    step.  Concretely, if the env computes the return from dates[t-1] to dates[t],
    the row is labelled dates[t].  This matches the convention used by quantstats
    and most backtest frameworks (value as of market close on the recorded date).

    Portfolio weights convention
    ----------------------------
    Weights saved for date t are the weights chosen by the agent at date t-1
    (after observing features up to t-1) that drove the return to date t.
    These are the weights held during the trading day ending at t's close.

    TODO: The env does not expose the current date in its info dict or step
    return.  We infer it from env.dates[env.current_step] (since start_idx=0).
    If the sequential-env assumption breaks (start_idx != 0), dates will be wrong
    without any error being raised.  Consider adding a sanity assertion after
    reset() to verify env.start_idx == 0.
    """
    obs, _ = env.reset(seed=seed)

    # Sanity check: the sequential env trick only works when start_idx lands at 0.
    # TODO: replace with a proper start_from parameter on BaseTradingEnv.reset().
    assert env.start_idx == 0, (
        f"Sequential rollout requires start_idx=0 but got {env.start_idx}. "
        "The make_sequential_env trick may have broken — check episode_length."
    )

    dates: list = []
    log_returns: list[float] = []
    portfolio_values: list[float] = []
    weights_rows: list[np.ndarray] = []

    while True:
        # model.predict handles both (obs_dim,) and (1, obs_dim) shapes.
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, _truncated, info = env.step(action)

        # env.current_step has already been incremented inside step().
        # Since start_idx=0, the date for this step is dates[current_step].
        step_date = env.dates[env.current_step]

        dates.append(step_date)
        log_returns.append(info["log_return"])
        portfolio_values.append(info["portfolio_value"])
        # portfolio_weights is set to new_weights inside step(), so these are
        # the weights that DROVE the return recorded above.
        weights_rows.append(env.portfolio_weights.copy())

        if done:
            break

    returns_df = pd.DataFrame(
        {"date": dates, "log_return": log_returns, "portfolio_value": portfolio_values}
    )
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    returns_df = returns_df.set_index("date")

    weights_df = pd.DataFrame(weights_rows, index=pd.to_datetime(dates), columns=env.tickers)
    weights_df.index.name = "date"

    return returns_df, weights_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained SAC agent on the test split and save results to CSV."
    )
    parser.add_argument(
        "--agent", type=int, required=True, choices=[1, 2],
        help="Which agent (1 = risk-averse, 2 = return-maximising)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help=(
            "Path to a .zip model file produced by train_sac.py. "
            "Defaults to ./models/agent{N}_best/best_model.zip, "
            "falling back to ./models/agent{N}_final.zip if not found."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed passed to env.reset() for the sequential rollout (default: 0)",
    )
    args = parser.parse_args()

    n = args.agent
    cfg = get_agent_config(n)

    # ── Resolve model path ────────────────────────────────────────────────────
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        best = Path(f"./models/agent{n}_best/best_model.zip")
        final = Path(f"./models/agent{n}_final.zip")
        if best.exists():
            model_path = best
            print(f"No --model-path given; using best model: {best}")
        elif final.exists():
            model_path = final
            print(f"No --model-path given; using final model: {final}")
        else:
            print(
                f"ERROR: No trained model found.\n"
                f"  Looked for: {best}\n"
                f"              {final}\n"
                f"Run training first:  python -m training.train_sac --agent {n}"
            )
            raise SystemExit(1)

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        raise SystemExit(1)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n=== Agent {n} — Test Rollout ===")
    print("Loading data …")
    features, prices = build_features(cfg["tickers"])
    splits = split_data(features, prices)

    test_features = splits["test_features"]
    test_prices   = splits["test_prices"]

    print(f"Test period: {test_features.index[0].date()} → {test_features.index[-1].date()} "
          f"({len(test_features)} rows)")

    # ── Environment ───────────────────────────────────────────────────────────
    env = make_sequential_env(cfg["env_cls"], test_features, test_prices)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} …")
    model = SAC.load(str(model_path), env=None)

    # ── Rollout ───────────────────────────────────────────────────────────────
    print("Running deterministic rollout …")
    returns_df, weights_df = run_agent_on_test(model, env, seed=args.seed)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = Path("./results")
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_path = out_dir / f"agent{n}_test_returns.csv"
    weights_path = out_dir / f"agent{n}_test_weights.csv"

    returns_df.to_csv(returns_path)
    weights_df.to_csv(weights_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_return = float(returns_df["portfolio_value"].iloc[-1] - 1.0)
    daily = returns_df["log_return"].values
    sharpe = float(np.mean(daily) / (np.std(daily) + 1e-8) * np.sqrt(252))
    max_dd = float(
        1.0
        - (returns_df["portfolio_value"] / returns_df["portfolio_value"].cummax()).min()
    )

    print(f"\n=== Test Period Summary ===")
    print(f"Steps recorded    : {len(returns_df)}")
    print(f"Total return      : {total_return:+.2%}")
    print(f"Annualised Sharpe : {sharpe:.4f}")
    print(f"Max drawdown      : {max_dd:.2%}")
    print(f"Returns saved to  : {returns_path}")
    print(f"Weights saved to  : {weights_path}")


if __name__ == "__main__":
    main()
