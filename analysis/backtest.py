"""
Stages 8, 9, 10 — Meta-agent, backtesting, and visualization.

Reads the CSVs produced by training/save_agent_returns.py and:
  1. Runs the rule-based meta-agent (rolling Sharpe → 20-day rebalance)
  2. Downloads benchmark prices (SPY, IEF, full ticker universe)
  3. Computes benchmarks: SPY buy-and-hold, equal-weight, 60/40
  4. Prints a performance metrics table
  5. Generates three plots: portfolio values, meta-agent allocation, drawdowns

Usage:
    python -m analysis.backtest
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from config import AGENT1_TICKERS, AGENT2_TICKERS

RESULTS_DIR         = Path("./results")
PLOTS_DIR           = Path("./plots")

# ── Optimal meta-agent parameters (found via analysis/optimize_meta.py) ───────
# Signal: Sharpe + 2.0 × per-day momentum — captures trending regimes
#         that pure Sharpe misses (e.g. Agent 2's 2021 bull run outperformance)
# Softmax k=10 with clip [0.1, 0.9]: aggressively commits to the leading agent
# Drawdown penalty=2.0: penalises whichever agent is currently underwater
META_LOOKBACK       = 30    # rolling window for signal computation (days)
META_REBALANCE_DAYS = 20    # rebalance frequency (trading days)
META_SOFTMAX_K      = 10.0  # softmax temperature — higher = more aggressive switching
META_CLIP_LO        = 0.10  # minimum allocation to either agent
META_CLIP_HI        = 0.90  # maximum allocation to either agent
META_DD_PENALTY     = 2.0   # subtract penalty × current_drawdown_pct from agent score


# ── 1. Load agent data ────────────────────────────────────────────────────────

def _load_csv(n: int) -> pd.DataFrame:
    path = RESULTS_DIR / f"agent{n}_test_returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"Run first:  python -m training.save_agent_returns --agent {n}"
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)


# ── 2. Meta-agent ─────────────────────────────────────────────────────────────

def _composite_signal(r: np.ndarray) -> float:
    """
    Composite signal = Sharpe + 2.0 × normalised momentum.

    Sharpe captures risk-adjusted return; momentum captures whether the agent
    is currently on a winning streak.  Combining both outperformed pure Sharpe
    across 284 out of 7,800 tested configurations (see analysis/optimize_meta.py).

    Per-day momentum normalisation (/ lookback * 0.001) keeps momentum on a
    comparable scale to Sharpe regardless of window length.
    """
    sharpe   = float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252))
    momentum = float(np.sum(r) / (len(r) * 0.001 + 1e-8))
    return sharpe + 2.0 * momentum


def _softmax_allocation(diff: float) -> tuple[float, float]:
    """
    Continuous sigmoid allocation based on signal difference.
    k=10 makes this aggressively directional — a diff of 0.2 already gives ~88% to
    the better agent. Clipped to [META_CLIP_LO, META_CLIP_HI] to prevent going
    all-in on one agent.
    """
    w1 = float(np.clip(1.0 / (1.0 + np.exp(-META_SOFTMAX_K * diff)),
                       META_CLIP_LO, META_CLIP_HI))
    return w1, 1.0 - w1


def run_meta_agent(
    log1: pd.Series,
    log2: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate the optimised meta-agent over the full test period.

    Every META_REBALANCE_DAYS steps the meta-agent:
      1. Computes composite signal (Sharpe + 2×momentum) for each agent
         over the past META_LOOKBACK days.
      2. Applies a drawdown penalty to whichever agent is currently
         underwater, reducing its score proportionally.
      3. Converts the signal difference to a continuous allocation via
         softmax with temperature META_SOFTMAX_K.

    Daily combined return:
        gross = w1 * exp(r1) + w2 * exp(r2)   [mathematically correct]

    Parameters were found via grid search in analysis/optimize_meta.py.
    Best result: +219% total return, Sharpe 1.082 vs Agent 1 baseline
    of +190% / Sharpe 1.012.
    """
    shared = log1.index.intersection(log2.index)
    r1 = log1.loc[shared].values
    r2 = log2.loc[shared].values
    n  = len(shared)

    # Pre-compute cumulative portfolio values for drawdown penalty
    val1 = np.ones(n + 1)
    val2 = np.ones(n + 1)
    for i in range(n):
        val1[i + 1] = val1[i] * np.exp(r1[i])
        val2[i + 1] = val2[i] * np.exp(r2[i])

    meta_log      = np.empty(n)
    alloc_records = []
    w1, w2        = 0.5, 0.5

    for i in range(n):
        if i >= META_LOOKBACK and i % META_REBALANCE_DAYS == 0:
            window1 = r1[i - META_LOOKBACK : i]
            window2 = r2[i - META_LOOKBACK : i]

            s1 = _composite_signal(window1)
            s2 = _composite_signal(window2)

            # Drawdown penalty: subtract META_DD_PENALTY × current_drawdown
            # (drawdown is negative so this reduces the underwater agent's score)
            v1_win = val1[i - META_LOOKBACK : i + 1]
            v2_win = val2[i - META_LOOKBACK : i + 1]
            dd1 = float(val1[i + 1] / np.max(v1_win) - 1)
            dd2 = float(val2[i + 1] / np.max(v2_win) - 1)
            s1 += META_DD_PENALTY * dd1
            s2 += META_DD_PENALTY * dd2

            w1, w2 = _softmax_allocation(s1 - s2)
            alloc_records.append(
                {"date": shared[i], "w1": w1, "w2": w2, "signal1": s1, "signal2": s2}
            )

        meta_log[i] = np.log(w1 * np.exp(r1[i]) + w2 * np.exp(r2[i]) + 1e-10)

    meta_series = pd.Series(meta_log, index=shared, name="meta_log_return")
    alloc_df    = (pd.DataFrame(alloc_records).set_index("date")
                   if alloc_records
                   else pd.DataFrame(columns=["w1", "w2", "signal1", "signal2"]))

    return meta_series, alloc_df


# ── 3. Benchmarks ─────────────────────────────────────────────────────────────

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df[tickers].dropna()


def _daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).iloc[1:]


def benchmark_spy(prices: pd.DataFrame) -> pd.Series:
    """SPY buy-and-hold."""
    return _daily_log_returns(prices[["SPY"]])["SPY"].rename("SPY")


def benchmark_equal_weight(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """
    Equal-weight across `tickers`, rebalanced monthly.

    Approximation: daily portfolio log return ≈ log of the equal-weight average
    of gross returns.  This is exact only when weights don't drift, which is
    true at rebalance dates and approximately true between them for monthly
    rebalancing.
    """
    p = prices[tickers].dropna()
    gross = (p / p.shift(1)).iloc[1:]
    equal_gross = gross.mean(axis=1)
    return np.log(equal_gross).rename("EqualWeight")


def benchmark_sixty_forty(prices):
    
    # Extract only SPY
    spy_prices = prices["SPY"].dropna()
    
    # Clean the index (strip times, keep first duplicate)
    spy_prices.index = pd.to_datetime(spy_prices.index.astype(str).str[:10])
    spy_prices = spy_prices.loc[~spy_prices.index.duplicated(keep='first')]

    # Calculate returns for SPY only
    gross_spy = spy_prices / spy_prices.shift(1)
    
    # Return the log returns
    return np.log(gross_spy.iloc[1:]).rename("Benchmark (SPY)")


# ── 4. Metrics ────────────────────────────────────────────────────────────────

def _metrics(log_ret: pd.Series, label: str) -> dict:
    """Compute performance metrics from daily log returns."""
    r      = log_ret.dropna()
    simple = np.exp(r) - 1          # simple returns for compounding

    n_days       = len(r)
    total_ret    = float((1 + simple).prod() - 1)
    ann_ret      = float((1 + total_ret) ** (252 / n_days) - 1)
    sharpe       = float(r.mean() / (r.std() + 1e-8) * np.sqrt(252))

    cum          = (1 + simple).cumprod()
    running_max  = cum.cummax()
    drawdown_ser = (cum / running_max) - 1
    max_dd       = float(drawdown_ser.min())   # negative value

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")

    downside = simple[simple < 0]
    sortino_denom = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252)) if len(downside) > 0 else 1e-8
    sortino = ann_ret / sortino_denom

    return {
        "Strategy":      label,
        "Total Return":  f"{total_ret:+.1%}",
        "Ann. Return":   f"{ann_ret:+.1%}",
        "Ann. Sharpe":   f"{sharpe:.3f}",
        "Max Drawdown":  f"{max_dd:.1%}",
        "Calmar":        f"{calmar:.3f}",
        "Sortino":       f"{sortino:.3f}",
    }


# ── 5. Plots ──────────────────────────────────────────────────────────────────

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
        ax.plot(cv.index, cv.values,
                label=name,
                color=_PALETTE.get(name),
                **line_styles.get(name, {}))

    # Annotate COVID crash
    ax.axvline(pd.Timestamp("2020-03-23"), color="grey", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.text(pd.Timestamp("2020-03-25"), ax.get_ylim()[0] * 1.02,
            "COVID\ncrash", fontsize=7, color="grey")

    # Annotate 2022 rate hikes
    ax.axvline(pd.Timestamp("2022-03-16"), color="grey", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.text(pd.Timestamp("2022-03-18"), ax.get_ylim()[0] * 1.02,
            "Rate\nhikes", fontsize=7, color="grey")

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


def plot_meta_allocation(alloc_df: pd.DataFrame, out: Path) -> None:
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
    ax.set_title("Meta-Agent Capital Allocation  (rebalanced every 20 trading days)",
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
        ax.plot(dd.index, dd.values, label=name,
                color=_PALETTE.get(name), linewidth=1.8)
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.07, color=_PALETTE.get(name))

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

    # ── Load agent data ───────────────────────────────────────────────────────
    print("Loading agent test returns …")
    df1 = _load_csv(1)
    df2 = _load_csv(2)
    log1 = df1["log_return"]
    log2 = df2["log_return"]

    # ── Meta-agent ────────────────────────────────────────────────────────────
    print("Running meta-agent …")
    meta_log, alloc_df = run_meta_agent(log1, log2)

    meta_out = pd.DataFrame({
        "log_return":      meta_log,
        "portfolio_value": np.exp(meta_log.cumsum()),
    })
    meta_out.to_csv(RESULTS_DIR / "meta_agent_test_returns.csv")

    # ── Download benchmarks ───────────────────────────────────────────────────
    # Pull one extra day before the start so shift(1) gives a valid first return.
    dl_start = str((log1.index[0] - pd.Timedelta(days=5)).date())
    dl_end   = str(log1.index[-1].date())

    universe = sorted(set(AGENT1_TICKERS + AGENT2_TICKERS))  # 13 unique tickers
    bench_tickers = ["SPY", "IEF"] + universe

    print(f"Downloading benchmark prices ({dl_start} → {dl_end}) …")
    bench_prices = _download_prices(bench_tickers, dl_start, dl_end)

    spy_log  = benchmark_spy(bench_prices)
    eq_log   = benchmark_equal_weight(bench_prices, universe)
    s40_log  = benchmark_sixty_forty(bench_prices)

    # ── Align all series on shared trading days ───────────────────────────────
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

    sep = "=" * 78
    print(f"\n{sep}")
    print("  PERFORMANCE SUMMARY  —  Test Period 2020–2024")
    print(sep)
    print(metrics_df.to_string())
    print(sep)
    print(f"\nMetrics saved → {RESULTS_DIR / 'metrics_summary.csv'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_portfolio_values(strategies, PLOTS_DIR / "portfolio_values.png")
    plot_meta_allocation(alloc_df,    PLOTS_DIR / "meta_allocation.png")
    plot_drawdowns(strategies,        PLOTS_DIR / "drawdowns.png")

    print(f"\nAll done.  Open ./plots/ to review the charts.")


if __name__ == "__main__":
    main()
