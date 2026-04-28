"""
Stage 6 — SAC training for Agent 1 (risk-averse) and Agent 2 (return-maximising).

Usage
-----
    python -m training.train_sac --agent 1
    python -m training.train_sac --agent 2
    python -m training.train_sac --agent 1 --timesteps 200000 --seed 42 --eval-freq 10000

Best model is selected by annualised Sharpe ratio computed from daily log returns
across validation episodes, not by mean episode reward.  Both metrics are logged
to TensorBoard under the "eval/" prefix.

Portfolio return timing assumption
----------------------------------
At each step the agent observes features derived from yesterday's closing prices
and chooses new portfolio weights w_new.  The environment then computes:

    portfolio_return = dot(w_new, today_close / yesterday_close)
    portfolio_value *= portfolio_return

This models end-of-day rebalancing: the decision is made using yesterday's
information and earns today's close-to-close return.  The 10-bps turnover cost
partially offsets the optimistic assumption of instantaneous, zero-cost
rebalancing.  See environments/base_env.py:_portfolio_step for the exact
implementation.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np

try:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print(
        "ERROR: stable_baselines3 or torch is not installed.\n"
        "Run:   pip install -r requirements.txt"
    )
    raise SystemExit(1)

from config import AGENT1_TICKERS, AGENT2_TICKERS
from data.pipeline import build_features, split_data
from environments.agent1_env import Agent1Env
from environments.agent2_env import Agent2Env


# ── Custom evaluation callback ────────────────────────────────────────────────

class SharpeEvalCallback(BaseCallback):
    """
    Evaluates the agent on the validation environment every eval_freq steps.

    Collects daily log_return values from each step's info dict across
    n_eval_episodes random validation episodes, then computes annualised Sharpe:

        Sharpe = mean(daily_log_returns) / std(daily_log_returns) * sqrt(252)

    Note: this uses log returns with no risk-free-rate deduction, which is
    standard in RL portfolio papers and acceptable for a pitch context.

    The model is saved to save_path/best_model.zip whenever Sharpe improves.
    Both Sharpe and mean episode reward are logged to TensorBoard under "eval/".
    """

    def __init__(
        self,
        val_env,
        eval_freq: int,
        save_path: str,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.val_env = val_env          # DummyVecEnv wrapping the validation env
        self.eval_freq = eval_freq
        self.save_path = Path(save_path)
        self.n_eval_episodes = n_eval_episodes
        self.best_sharpe = -np.inf
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        all_daily_returns: list[float] = []
        all_episode_rewards: list[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.val_env.reset()
            episode_reward = 0.0
            # DummyVecEnv returns done as a (n_envs,) bool array
            done = np.array([False])

            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, infos = self.val_env.step(action)
                episode_reward += float(rewards[0])
                # info dict is always populated, including on the terminal step
                all_daily_returns.append(float(infos[0].get("log_return", 0.0)))

            all_episode_rewards.append(episode_reward)

        mean_reward = float(np.mean(all_episode_rewards))

        daily = np.array(all_daily_returns)
        sharpe = (
            float(np.mean(daily) / (np.std(daily) + 1e-8) * np.sqrt(252))
            if len(daily) > 1
            else 0.0
        )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/sharpe", sharpe)
        self.logger.dump(self.n_calls)

        if self.verbose >= 1:
            print(
                f"\n[Eval] step={self.n_calls:>7d}  "
                f"mean_reward={mean_reward:+.4f}  sharpe={sharpe:+.4f}"
            )

        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.model.save(str(self.save_path / "best_model"))
            if self.verbose >= 1:
                print(f"         → New best Sharpe={sharpe:.4f}, model saved.")

        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_agent_config(agent_num: int) -> dict:
    """Return the ticker list and environment class for the requested agent."""
    if agent_num == 1:
        return {"tickers": AGENT1_TICKERS, "env_cls": Agent1Env}
    if agent_num == 2:
        return {"tickers": AGENT2_TICKERS, "env_cls": Agent2Env}
    raise ValueError(f"--agent must be 1 or 2, got {agent_num}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an SAC agent for portfolio allocation (Stage 6)."
    )
    parser.add_argument(
        "--agent", type=int, required=True, choices=[1, 2],
        help="Which agent to train (1 = risk-averse, 2 = return-maximising)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training steps (default: 200000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Evaluate on val env every N steps (default: 10000)",
    )
    args = parser.parse_args()

    n = args.agent
    set_seeds(args.seed)

    # ── Output paths ──────────────────────────────────────────────────────────
    log_dir    = Path(f"./logs/agent{n}")
    model_dir  = Path(f"./models/agent{n}_best")
    final_path = Path(f"./models/agent{n}_final.zip")
    for d in (log_dir, model_dir, final_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\n=== Agent {n} — Stage 6: SAC Training ===")
    cfg = get_agent_config(n)
    print("Downloading and building features …")
    features, prices = build_features(cfg["tickers"])

    if len(features) == 0:
        print(
            "\nERROR: Data download returned 0 rows.\n"
            "  This is usually a transient yfinance cache lock.  Fix:\n"
            "    python -c \"import shutil, os; "
            "path = os.path.expanduser('~/.cache/py-yfinance'); "
            "shutil.rmtree(path, ignore_errors=True); print('cache cleared')\"\n"
            "  Then rerun this script."
        )
        raise SystemExit(1)

    splits = split_data(features, prices)

    # ── Environments ──────────────────────────────────────────────────────────
    env_cls = cfg["env_cls"]

    # DummyVecEnv expects a list of zero-argument callables (thunks).
    # Default-argument capture is used to avoid the classic Python late-binding
    # closure bug (all lambdas would otherwise share the same final values of f/p).
    def make_train_env(
        f=splits["train_features"],
        p=splits["train_prices"],
    ):
        return env_cls(f, p)

    def make_val_env(
        f=splits["val_features"],
        p=splits["val_prices"],
    ):
        return env_cls(f, p)

    train_vec = DummyVecEnv([make_train_env])
    val_vec   = DummyVecEnv([make_val_env])

    # ── SAC model ─────────────────────────────────────────────────────────────
    model = SAC(
        "MlpPolicy",
        train_vec,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_dir),
    )

    # ── Callback ──────────────────────────────────────────────────────────────
    eval_cb = SharpeEvalCallback(
        val_env=val_vec,
        eval_freq=args.eval_freq,
        save_path=str(model_dir),
        n_eval_episodes=5,
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training for {args.timesteps:,} steps (seed={args.seed}) …")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)
    elapsed = time.time() - t0

    # ── Save final checkpoint ─────────────────────────────────────────────────
    model.save(str(final_path))

    # ── Summary ───────────────────────────────────────────────────────────────
    best_model_path = model_dir / "best_model.zip"
    print(f"\n=== Training Complete ===")
    print(f"Duration          : {elapsed / 60:.1f} min")
    print(f"Best val Sharpe   : {eval_cb.best_sharpe:.4f}")
    print(f"Best model path   : {best_model_path}")
    print(f"Final model path  : {final_path}")
    print(f"TensorBoard logs  : {log_dir}")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir ./logs/")


if __name__ == "__main__":
    main()


# ── Quick-start commands (copy-paste, do not uncomment to run from here) ──────
#
# Train Agent 1 (risk-averse; see AGENT1_TICKERS in config.py):
#   python -m training.train_sac --agent 1 --timesteps 200000 --seed 42
#
# Train Agent 2 (return-maximising; equities only per config.AGENT2_TICKERS):
#   python -m training.train_sac --agent 2 --timesteps 200000 --seed 42
#
# Monitor both agents in TensorBoard:
#   tensorboard --logdir ./logs/
