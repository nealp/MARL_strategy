"""
Stage 10b — Quantstats tearsheet and pitch-ready plots.

Generates:
  plots/tearsheet_meta.html       — full HTML tearsheet for meta-agent vs SPY
  plots/tearsheet_agent1.html     — tearsheet for Agent 1 vs SPY
  plots/tearsheet_agent2.html     — tearsheet for Agent 2 vs SPY
  plots/monthly_heatmap_meta.png  — monthly returns heatmap (great for slides)
  plots/monthly_heatmap_agent1.png
  plots/monthly_heatmap_agent2.png
  plots/yearly_returns.png        — year-by-year bar chart vs SPY
  plots/weights_agent1.png        — agent 1 portfolio weight heatmap over time
  plots/weights_agent2.png        — agent 2 portfolio weight heatmap over time

Usage:
    python -m analysis.tearsheet
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # headless — must be set before importing pyplot or quantstats

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
import quantstats as qs

from config import AGENT1_TICKERS, AGENT2_TICKERS

RESULTS_DIR = Path("./results")
PLOTS_DIR   = Path("./plots")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_simple_returns(name: str) -> pd.Series:
    """Load log returns CSV and convert to simple returns for quantstats."""
    path = RESULTS_DIR / f"{name}_test_returns.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    simple = np.exp(df["log_return"]) - 1
    simple.name = name
    return simple


def download_spy_simple(start: str, end: str) -> pd.Series:
    p = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    r = (p / p.shift(1) - 1).dropna()
    r.name = "SPY"
    return r


def align(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    shared = s1.index.intersection(s2.index)
    return s1.loc[shared], s2.loc[shared]


# ── 1. HTML tearsheets ────────────────────────────────────────────────────────

def generate_tearsheets(strategies: dict[str, pd.Series], spy: pd.Series) -> None:
    """Generate one HTML tearsheet per strategy using quantstats."""
    for name, ret in strategies.items():
        r, b = align(ret, spy)
        out = PLOTS_DIR / f"tearsheet_{name.lower().replace(' ', '_')}.html"
        try:
            qs.reports.html(
                r, b,
                output=str(out),
                title=f"{name} — Test Period 2020–2024",
                download_filename=str(out),
            )
            print(f"  Saved: {out}")
        except Exception as e:
            print(f"  WARNING: tearsheet for {name} failed: {e}")


# ── 2. Monthly returns heatmaps ───────────────────────────────────────────────

def generate_monthly_heatmaps(strategies: dict[str, pd.Series]) -> None:
    """Save a monthly returns heatmap PNG for each strategy."""
    for name, ret in strategies.items():
        out = PLOTS_DIR / f"monthly_heatmap_{name.lower().replace(' ', '_')}.png"
        try:
            fig = qs.plots.monthly_heatmap(
                ret,
                show=False,
                figsize=(11, 5),
            )
            if fig is not None:
                fig.savefig(out, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                # quantstats sometimes calls plt.show() internally; grab current figure
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.close("all")
            print(f"  Saved: {out}")
        except Exception as e:
            print(f"  WARNING: heatmap for {name} failed: {e}")
            plt.close("all")


# ── 3. Yearly returns bar chart ───────────────────────────────────────────────

def plot_yearly_returns(strategies: dict[str, pd.Series], out: Path) -> None:
    """Year-by-year returns bar chart — easy to read in a pitch deck."""
    annual: dict[str, pd.Series] = {}
    for name, ret in strategies.items():
        # Compound daily returns within each calendar year
        yearly = (1 + ret).groupby(ret.index.year).prod() - 1
        annual[name] = yearly * 100   # as percentage

    years = sorted({y for s in annual.values() for y in s.index})
    n_strategies = len(strategies)
    width  = 0.13
    x      = np.arange(len(years))

    colors = {
        "Meta-Agent":            "#1f77b4",
        "Agent 1 (Risk-Averse)": "#2ca02c",
        "Agent 2 (Return-Max)":  "#ff7f0e",
        "SPY":                   "#d62728",
        "Equal-Weight":          "#9467bd",
        "60/40 (SPY+IEF)":       "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    offsets = np.linspace(-(n_strategies - 1) / 2, (n_strategies - 1) / 2, n_strategies)

    for (name, yearly), offset in zip(annual.items(), offsets):
        vals = [yearly.get(y, 0.0) for y in years]
        bars = ax.bar(x + offset * width, vals, width=width * 0.9,
                      label=name, color=colors.get(name), alpha=0.85)
        # Label bars that are notably large or small
        for bar, v in zip(bars, vals):
            if abs(v) > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1 if v >= 0 else -4),
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5,
                        color="black")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11)
    ax.set_ylabel("Annual Return (%)", fontsize=11)
    ax.set_title("Year-by-Year Returns  —  2020–2024", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 4. Portfolio weight heatmaps ──────────────────────────────────────────────

def plot_weight_heatmap(agent_num: int, out: Path) -> None:
    """
    Heatmap of daily portfolio weights over the test period.
    Rows = assets, columns = trading days (sampled weekly to keep it readable).
    """
    weights_path = RESULTS_DIR / f"agent{agent_num}_test_weights.csv"
    if not weights_path.exists():
        print(f"  WARNING: {weights_path} not found, skipping weight heatmap.")
        return

    df = pd.read_csv(weights_path, index_col="date", parse_dates=True)

    # Sample every 5 trading days to keep the heatmap readable
    df_sampled = df.iloc[::5]

    tickers = df.columns.tolist()
    tickers_label = AGENT1_TICKERS if agent_num == 1 else AGENT2_TICKERS

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(
        df_sampled.T.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=df_sampled.values.max(),
        interpolation="nearest",
    )

    # X-axis: dates (show yearly ticks)
    date_ticks = [i for i, d in enumerate(df_sampled.index) if d.month == 1 and d.day <= 7]
    date_labels = [str(df_sampled.index[i].year) for i in date_ticks]
    ax.set_xticks(date_ticks)
    ax.set_xticklabels(date_labels, fontsize=10)

    # Y-axis: tickers
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=9)

    plt.colorbar(im, ax=ax, label="Portfolio Weight", shrink=0.8)
    ax.set_title(
        f"Agent {agent_num} Portfolio Weights Over Time  (2020–2024)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Year", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load returns ──────────────────────────────────────────────────────────
    print("Loading returns …")
    meta   = load_simple_returns("meta_agent")
    agent1 = load_simple_returns("agent1")
    agent2 = load_simple_returns("agent2")

    dl_start = str((meta.index[0] - pd.Timedelta(days=5)).date())
    dl_end   = str(meta.index[-1].date())
    print(f"Downloading SPY ({dl_start} → {dl_end}) …")
    spy = download_spy_simple(dl_start, dl_end)

    # Align everything to shared dates
    shared = meta.index.intersection(agent1.index).intersection(agent2.index).intersection(spy.index)
    meta, agent1, agent2, spy = (s.loc[shared] for s in [meta, agent1, agent2, spy])

    strategies = {
        "Meta-Agent":            meta,
        "Agent 1 (Risk-Averse)": agent1,
        "Agent 2 (Return-Max)":  agent2,
        "SPY":                   spy,
    }

    pitch_strategies = {
        "Meta-Agent":            meta,
        "Agent 1 (Risk-Averse)": agent1,
        "Agent 2 (Return-Max)":  agent2,
    }

    # ── Tearsheets ────────────────────────────────────────────────────────────
    print("\nGenerating HTML tearsheets …")
    generate_tearsheets(pitch_strategies, spy)

    # ── Monthly heatmaps ──────────────────────────────────────────────────────
    print("\nGenerating monthly return heatmaps …")
    generate_monthly_heatmaps(pitch_strategies)

    # ── Yearly returns ────────────────────────────────────────────────────────
    print("\nGenerating yearly returns chart …")

    # Build 60/40 from cached meta CSV date range for the yearly chart
    bench_prices = yf.download(
        ["SPY", "IEF"], start=dl_start, end=dl_end,
        auto_adjust=True, progress=False
    )["Close"].dropna()
    sixty40_simple = (0.6 * (bench_prices["SPY"] / bench_prices["SPY"].shift(1) - 1)
                      + 0.4 * (bench_prices["IEF"] / bench_prices["IEF"].shift(1) - 1)).dropna()
    sixty40_simple.name = "60/40 (SPY+IEF)"
    sixty40_simple = sixty40_simple.loc[shared]

    yearly_strategies = {**strategies, "60/40 (SPY+IEF)": sixty40_simple}
    plot_yearly_returns(yearly_strategies, PLOTS_DIR / "yearly_returns.png")

    # ── Weight heatmaps ───────────────────────────────────────────────────────
    print("\nGenerating portfolio weight heatmaps …")
    plot_weight_heatmap(1, PLOTS_DIR / "weights_agent1.png")
    plot_weight_heatmap(2, PLOTS_DIR / "weights_agent2.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
All outputs saved to ./plots/

For your slide deck, the most impactful visuals are:
  1. plots/portfolio_values.png      — cumulative return curves (already built)
  2. plots/monthly_heatmap_meta.png  — monthly returns grid (green/red, very visual)
  3. plots/yearly_returns.png        — year-by-year bar chart (easy to explain)
  4. plots/meta_allocation.png       — shows the meta-agent switching between agents
  5. plots/tearsheet_meta.html       — open in browser for the full professional report
  6. plots/weights_agent1.png        — shows agent 1 rotating toward bonds (e.g. IEF/SHY) during stress
""")


if __name__ == "__main__":
    main()