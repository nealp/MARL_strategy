"""
Stages 8, 9, 10 — Meta-agent, backtesting, and visualization.

Reads the CSVs produced by training/save_agent_returns.py and:
  1. Loads meta-agent parameters from results/meta_params.json
     (produced by analysis/optimize_meta — run that first)
  2. Runs the meta-agent with those parameters
  3. Downloads benchmark prices (SPY, IEF, full ticker universe)
  4. Computes benchmarks: SPY buy-and-hold, equal-weight, 60/40
  5. Prints a performance metrics table
  6. Generates three plots: portfolio values, meta-agent allocation, drawdowns

Usage:
    python -m analysis.optimize_meta   # find best params (run once after retraining)
    python -m analysis.backtest        # backtest + plots
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from config import ALL_TICKERS
from analysis.optimize_meta import SIGNAL_FNS

RESULTS_DIR    = Path("./results")
BENCHMARKS_DIR = Path("./results/benchmarks")
PLOTS_DIR      = Path("./plots")


# ── 1. Load agent data ────────────────────────────────────────────────────────

def _load_csv(n: int, split: str = "test") -> pd.DataFrame:
    path = RESULTS_DIR / f"agent{n}_{split}_returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run first:  python -m training.save_agent_returns --agent {n} --split {split}"
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)


# ── 2. Load meta-agent parameters ────────────────────────────────────────────

def _load_meta_params() -> dict:
    """
    Read the best meta-agent parameters saved by analysis/optimize_meta.

    Raises a clear error if the file doesn't exist so the user knows
    to run optimize_meta first.
    """
    path = RESULTS_DIR / "meta_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run first:  python -m analysis.optimize_meta"
        )
    with open(path) as f:
        params = json.load(f)
    ap = params["alloc_params"]
    print(f"  Loaded params from {path}")
    print(f"  signal={params['signal']}  LB={params['lookback']}  "
          f"RF={params['rebalance_freq']}  k={ap.get('k', 'N/A')}  "
          f"DDP={params['dd_penalty']}  "
          f"(selected on {params.get('selected_on', 'unknown')})")
    return params


# ── 3. Meta-agent simulation ──────────────────────────────────────────────────

def run_meta_agent(
    log1:   pd.Series,
    log2:   pd.Series,
    params: dict,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate the meta-agent over the test period using params found by _find_meta_params.

    params keys used: lookback, rebalance_freq, signal, alloc_params, dd_penalty
    """
    shared = log1.index.intersection(log2.index)
    r1 = log1.loc[shared].values
    r2 = log2.loc[shared].values
    n  = len(shared)

    lookback      = params["lookback"]
    rebalance_freq = params["rebalance_freq"]
    sig_fn        = SIGNAL_FNS[params["signal"]]
    ap            = params["alloc_params"]
    dd_penalty    = params["dd_penalty"]
    k             = ap.get("k", 10.0)
    clip_lo       = ap.get("clip_lo", 0.10)
    clip_hi       = ap.get("clip_hi", 0.90)

    val1 = np.ones(n + 1)
    val2 = np.ones(n + 1)
    for i in range(n):
        val1[i + 1] = val1[i] * np.exp(r1[i])
        val2[i + 1] = val2[i] * np.exp(r2[i])

    meta_log      = np.empty(n)
    alloc_records = []
    w1, w2        = 0.5, 0.5

    for i in range(n):
        if i >= lookback and i % rebalance_freq == 0:
            window1 = r1[i - lookback : i]
            window2 = r2[i - lookback : i]

            s1 = sig_fn(window1)
            s2 = sig_fn(window2)

            if dd_penalty > 0.0:
                v1_win = val1[i - lookback : i + 1]
                v2_win = val2[i - lookback : i + 1]
                dd1 = float(val1[i] / np.max(v1_win) - 1)
                dd2 = float(val2[i] / np.max(v2_win) - 1)
                s1 += dd_penalty * dd1
                s2 += dd_penalty * dd2

            diff = s1 - s2
            w1   = float(np.clip(1.0 / (1.0 + np.exp(-k * diff)), clip_lo, clip_hi))
            w2   = 1.0 - w1
            alloc_records.append(
                {"date": shared[i], "w1": w1, "w2": w2, "signal1": s1, "signal2": s2}
            )

        meta_log[i] = np.log(w1 * np.exp(r1[i]) + w2 * np.exp(r2[i]) + 1e-10)

    meta_series = pd.Series(meta_log, index=shared, name="meta_log_return")
    alloc_df    = (pd.DataFrame(alloc_records).set_index("date")
                   if alloc_records
                   else pd.DataFrame(columns=["w1", "w2", "signal1", "signal2"]))

    return meta_series, alloc_df


# ── 4. Benchmarks ─────────────────────────────────────────────────────────────

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df[tickers].dropna()


def _daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).iloc[1:]


def benchmark_spy(prices: pd.DataFrame) -> pd.Series:
    return _daily_log_returns(prices[["SPY"]])["SPY"].rename("SPY")


def benchmark_equal_weight(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    p = prices[tickers].dropna()
    gross = (p / p.shift(1)).iloc[1:]
    return np.log(gross.mean(axis=1)).rename("EqualWeight")


def benchmark_sixty_forty(prices: pd.DataFrame) -> pd.Series:
    gross_spy = prices["SPY"] / prices["SPY"].shift(1)
    gross_ief = prices["IEF"] / prices["IEF"].shift(1)
    return np.log((0.6 * gross_spy + 0.4 * gross_ief).iloc[1:]).rename("60/40")


# ── 5. Metrics ────────────────────────────────────────────────────────────────

def _metrics(log_ret: pd.Series, label: str) -> dict:
    r      = log_ret.dropna()
    simple = np.exp(r) - 1

    n_days    = len(r)
    total_ret = float((1 + simple).prod() - 1)
    ann_ret   = float((1 + total_ret) ** (252 / n_days) - 1)
    sharpe    = float(r.mean() / (r.std() + 1e-8) * np.sqrt(252))

    cum       = (1 + simple).cumprod()
    max_dd    = float((cum / cum.cummax() - 1).min())
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")

    downside      = simple[simple < 0]
    sortino_denom = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252)) if len(downside) > 0 else 1e-8
    sortino       = ann_ret / sortino_denom

    return {
        "Strategy":     label,
        "Total Return": f"{total_ret:+.1%}",
        "Ann. Return":  f"{ann_ret:+.1%}",
        "Ann. Sharpe":  f"{sharpe:.3f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Ann. Vol":     f"{float(r.std() * np.sqrt(252)):.1%}",
        "Win Rate":     f"{float((r > 0).mean()):.1%}",
        "Calmar":       f"{calmar:.3f}",
        "Sortino":      f"{sortino:.3f}",
    }


# ── 6. Plots ──────────────────────────────────────────────────────────────────

_PALETTE = {
    "Meta-Agent":            "#1f77b4",
    "Agent 1 (Risk-Averse)": "#2ca02c",
    "Agent 2 (Return-Max)":  "#ff7f0e",
    "SPY":                   "#d62728",
    "Equal-Weight":          "#9467bd",
    "60/40 (SPY+IEF)":       "#8c564b",
}

_YEAR_FMT = mdates.DateFormatter("%Y")


def _cum_value(log_ret: pd.Series) -> pd.Series:
    return (np.exp(log_ret) - 1 + 1).cumprod()


def plot_portfolio_values(strategies: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    line_styles = {
        "Meta-Agent":            dict(linewidth=2.8, zorder=5),
        "Agent 1 (Risk-Averse)": dict(linewidth=1.4, linestyle="--", alpha=0.85),
        "Agent 2 (Return-Max)":  dict(linewidth=1.4, linestyle="--", alpha=0.85),
        "SPY":                   dict(linewidth=1.6, alpha=0.9),
        "Equal-Weight":          dict(linewidth=1.2, linestyle=":", alpha=0.85),
        "60/40 (SPY+IEF)":       dict(linewidth=1.2, linestyle=":", alpha=0.85),
    }
    for name, log_ret in strategies.items():
        cv = _cum_value(log_ret)
        ax.plot(cv.index, cv.values, label=name,
                color=_PALETTE.get(name), **line_styles.get(name, {}))
    ax.axvline(pd.Timestamp("2020-03-23"), color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(pd.Timestamp("2020-03-25"), ax.get_ylim()[0] * 1.02, "COVID\ncrash", fontsize=7, color="grey")
    ax.axvline(pd.Timestamp("2022-03-16"), color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(pd.Timestamp("2022-03-18"), ax.get_ylim()[0] * 1.02, "Rate\nhikes", fontsize=7, color="grey")
    ax.axhline(1.0, color="black", linewidth=0.4, linestyle="--", alpha=0.3)
    ax.set_ylabel("Portfolio Value  (start = $1.00)", fontsize=11)
    ax.set_title("Portfolio Performance  —  Test Period 2020–2024", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_formatter(_YEAR_FMT)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_meta_allocation(alloc_df: pd.DataFrame, rebalance_freq: int, out: Path) -> None:
    if alloc_df.empty:
        print("  Skipping allocation plot (insufficient history).")
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.fill_between(alloc_df.index, alloc_df["w1"],
                    alpha=0.65, label="Agent 1 (Risk-Averse)", color=_PALETTE["Agent 1 (Risk-Averse)"])
    ax.fill_between(alloc_df.index, alloc_df["w1"], 1.0,
                    alpha=0.65, label="Agent 2 (Return-Max)", color=_PALETTE["Agent 2 (Return-Max)"])
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.35)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Capital Weight", fontsize=10)
    ax.set_title(f"Meta-Agent Capital Allocation  (rebalanced every {rebalance_freq} trading days)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.18)
    ax2 = axes[1]
    ax2.plot(alloc_df.index, alloc_df["signal1"], color=_PALETTE["Agent 1 (Risk-Averse)"],
             linewidth=1.3, label="Signal 1")
    ax2.plot(alloc_df.index, alloc_df["signal2"], color=_PALETTE["Agent 2 (Return-Max)"],
             linewidth=1.3, label="Signal 2")
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Composite Signal", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.18)
    ax2.xaxis.set_major_formatter(_YEAR_FMT)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_drawdowns(strategies: dict, out: Path) -> None:
    highlight = ["Meta-Agent", "SPY", "60/40 (SPY+IEF)"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, log_ret in strategies.items():
        if name not in highlight:
            continue
        simple = np.exp(log_ret) - 1
        cum    = (1 + simple).cumprod()
        dd     = (cum / cum.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, label=name, color=_PALETTE.get(name), linewidth=1.8)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.07, color=_PALETTE.get(name))
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.set_title("Drawdown Comparison  —  Meta-Agent vs Benchmarks", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(_YEAR_FMT)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load agent test returns ───────────────────────────────────────────────
    print("Loading agent test returns …")
    df1 = _load_csv(1)
    df2 = _load_csv(2)
    log1 = df1["log_return"]
    log2 = df2["log_return"]

    # ── Load meta-agent parameters ────────────────────────────────────────────
    print("\nLoading meta-agent parameters …")
    params = _load_meta_params()

    # ── Run meta-agent ────────────────────────────────────────────────────────
    print("\nRunning meta-agent …")
    meta_log, alloc_df = run_meta_agent(log1, log2, params)

    pd.DataFrame({
        "log_return":      meta_log,
        "portfolio_value": np.exp(meta_log.cumsum()),
    }).to_csv(RESULTS_DIR / "meta_agent_test_returns.csv")
    alloc_df.to_csv(RESULTS_DIR / "meta_agent_alloc.csv")

    # ── Download benchmarks ───────────────────────────────────────────────────
    dl_start = str((log1.index[0] - pd.Timedelta(days=5)).date())
    dl_end   = str(log1.index[-1].date())
    bench_tickers = ["SPY"] + ALL_TICKERS

    print(f"Downloading benchmark prices ({dl_start} → {dl_end}) …")
    bench_prices = _download_prices(bench_tickers, dl_start, dl_end)

    spy_log = benchmark_spy(bench_prices)
    eq_log  = benchmark_equal_weight(bench_prices, ALL_TICKERS)
    s40_log = benchmark_sixty_forty(bench_prices)

    # ── Align on shared trading days ──────────────────────────────────────────
    shared = (
        meta_log.dropna().index
        .intersection(log1.index)
        .intersection(log2.index)
        .intersection(spy_log.index)
        .intersection(eq_log.index)
        .intersection(s40_log.index)
    )

    strategies = {
        "Meta-Agent":            meta_log.loc[shared],
        "Agent 1 (Risk-Averse)": log1.loc[shared],
        "Agent 2 (Return-Max)":  log2.loc[shared],
        "SPY":                   spy_log.loc[shared],
        "Equal-Weight":          eq_log.loc[shared],
        "60/40 (SPY+IEF)":       s40_log.loc[shared],
    }

    # ── Metrics table ─────────────────────────────────────────────────────────
    rows = [_metrics(ret, name) for name, ret in strategies.items()]
    metrics_df = pd.DataFrame(rows).set_index("Strategy")
    metrics_df.to_csv(RESULTS_DIR / "metrics_summary.csv")

    spy_log.loc[shared].to_csv(BENCHMARKS_DIR / "spy.csv", header=True)
    eq_log.loc[shared].to_csv(BENCHMARKS_DIR / "equal_weight.csv", header=True)
    s40_log.loc[shared].to_csv(BENCHMARKS_DIR / "sixty_forty.csv", header=True)

    sep = "=" * 78
    print(f"\n{sep}")
    print("  PERFORMANCE SUMMARY  —  Test Period 2020–2024")
    print(sep)
    print(metrics_df.to_string())
    print(sep)
    print(f"\nMetrics saved → {RESULTS_DIR / 'metrics_summary.csv'}")
    print(f"\nMeta params used: signal={params['signal']}  "
          f"LB={params['lookback']}  RF={params['rebalance_freq']}  "
          f"k={params['alloc_params'].get('k')}  DDP={params['dd_penalty']}  "
          f"(selected on {params.get('selected_on', 'unknown')})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_portfolio_values(strategies,   PLOTS_DIR / "portfolio_values.png")
    plot_meta_allocation(alloc_df, params["rebalance_freq"], PLOTS_DIR / "meta_allocation.png")
    plot_drawdowns(strategies,          PLOTS_DIR / "drawdowns.png")

    print(f"\nAll done.  Open ./plots/ to review the charts.")


if __name__ == "__main__":
    main()
