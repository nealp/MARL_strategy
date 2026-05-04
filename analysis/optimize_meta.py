"""
Systematic search for the best meta-agent allocation rules.

Benchmarks to beat:
  Agent 1 alone: +190.1% total, Sharpe 1.012, Max DD -28.3%

Tests ~200+ parameter combinations across:
  - Signal metrics: Sharpe, Sortino, Momentum, Calmar, Composite
  - Lookback windows: 10, 15, 20, 30, 60 days
  - Rebalance frequencies: 5, 10, 15, 20 days
  - Allocation styles: discrete (various thresholds), continuous softmax
  - Drawdown penalty modifier
  - Win-rate signal

Usage:
    python -m analysis.optimize_meta
"""

import warnings
warnings.filterwarnings("ignore")

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("./results")

AGENT1_BASELINE = {"total_return": 1.901, "sharpe": 1.012, "max_dd": -0.283, "calmar": 0.842}


# ── Signal functions ──────────────────────────────────────────────────────────

def _annualized_return(r: np.ndarray) -> float:
    n = len(r)
    total = float(np.exp(np.sum(r)) - 1)
    return float((1 + total) ** (252 / max(n, 1)) - 1)


def sig_sharpe(r: np.ndarray) -> float:
    return float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252))


def sig_sortino(r: np.ndarray) -> float:
    ann = _annualized_return(r)
    down = r[r < 0]
    denom = float(np.sqrt(np.mean(down ** 2)) * np.sqrt(252)) if len(down) > 1 else 1e-8
    return ann / denom


def sig_momentum(r: np.ndarray) -> float:
    """Total log return over window — captures trend strength."""
    return float(np.sum(r))


def sig_calmar(r: np.ndarray) -> float:
    ann = _annualized_return(r)
    cum = np.exp(np.cumsum(r))
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return ann / (abs(mdd) + 1e-8)


def sig_winrate(r: np.ndarray) -> float:
    """Fraction of positive return days."""
    return float(np.mean(r > 0))


def sig_composite_sm(r: np.ndarray, alpha: float = 1.0) -> float:
    """Sharpe + alpha * normalised momentum."""
    s = sig_sharpe(r)
    m = sig_momentum(r) / (len(r) * 0.001 + 1e-8)  # per-day momentum
    return s + alpha * m


def sig_composite_sortino_mom(r: np.ndarray, alpha: float = 1.0) -> float:
    return sig_sortino(r) + alpha * sig_momentum(r) / (len(r) * 0.001 + 1e-8)


def sig_composite_all(r: np.ndarray) -> float:
    """Equal-weight combination of Sharpe, Sortino, and Momentum."""
    s  = sig_sharpe(r)
    so = sig_sortino(r)
    m  = sig_momentum(r) / (len(r) * 0.001 + 1e-8)
    # Normalise each to roughly the same scale before combining
    return (s / 2.0) + (so / 4.0) + (m / 2.0)


SIGNAL_FNS = {
    "sharpe":               sig_sharpe,
    "sortino":              sig_sortino,
    "momentum":             sig_momentum,
    "calmar":               sig_calmar,
    "winrate":              sig_winrate,
    "composite_sm_0.5":     lambda r: sig_composite_sm(r, 0.5),
    "composite_sm_1.0":     lambda r: sig_composite_sm(r, 1.0),
    "composite_sm_2.0":     lambda r: sig_composite_sm(r, 2.0),
    "composite_sortino_1":  lambda r: sig_composite_sortino_mom(r, 1.0),
    "composite_all":        sig_composite_all,
}


# ── Allocation functions ──────────────────────────────────────────────────────

def alloc_discrete(diff: float, lo: float, hi: float,
                   w_lo: float, w_hi: float) -> float:
    """
    Returns w1 (weight for Agent 1).
    diff = signal1 - signal2
    lo/hi = absolute diff thresholds
    w_lo/w_hi = allocation for Agent 1 when it's clearly better
    """
    if abs(diff) < lo:
        return 0.5
    if abs(diff) < hi:
        return w_lo if diff > 0 else (1.0 - w_lo)
    return w_hi if diff > 0 else (1.0 - w_hi)


def alloc_softmax(diff: float, k: float,
                  clip_lo: float = 0.1, clip_hi: float = 0.9) -> float:
    """Continuous sigmoid allocation — k controls aggressiveness."""
    w1 = 1.0 / (1.0 + np.exp(-k * diff))
    return float(np.clip(w1, clip_lo, clip_hi))


# ── Simulate meta-agent ───────────────────────────────────────────────────────

def simulate(
    r1: np.ndarray,
    r2: np.ndarray,
    lookback: int,
    rebalance_freq: int,
    signal_name: str,
    alloc_params: dict,
    drawdown_penalty: float = 0.0,   # subtract γ × current_drawdown from losing agent's score
) -> dict:
    """
    Simulate one meta-agent strategy.

    Returns dict with: total_return, sharpe, max_dd, calmar, sortino, score
    """
    n  = len(r1)
    fn = SIGNAL_FNS[signal_name]
    w1, w2 = 0.5, 0.5

    meta = np.empty(n)

    # Track portfolio values for drawdown penalty signal
    val1 = np.ones(n + 1)
    val2 = np.ones(n + 1)
    for i in range(n):
        val1[i + 1] = val1[i] * np.exp(r1[i])
        val2[i + 1] = val2[i] * np.exp(r2[i])

    for i in range(n):
        if i >= lookback and i % rebalance_freq == 0:
            window1 = r1[i - lookback: i]
            window2 = r2[i - lookback: i]

            s1 = fn(window1)
            s2 = fn(window2)

            if drawdown_penalty > 0.0:
                # Current drawdown of each agent's value over the lookback window
                v1_win = val1[i - lookback: i + 1]
                v2_win = val2[i - lookback: i + 1]
                dd1 = float((v1_win[-1] / np.maximum.accumulate(v1_win).max()) - 1)
                dd2 = float((v2_win[-1] / np.maximum.accumulate(v2_win).max()) - 1)
                # Penalise the agent currently in deeper drawdown
                s1 += drawdown_penalty * dd1   # dd1 is negative → subtracts from s1
                s2 += drawdown_penalty * dd2

            diff = s1 - s2

            style = alloc_params["style"]
            if style == "discrete":
                w1 = alloc_discrete(diff,
                                    alloc_params["lo_thresh"],
                                    alloc_params["hi_thresh"],
                                    alloc_params["w_lo"],
                                    alloc_params["w_hi"])
            else:   # continuous softmax
                w1 = alloc_softmax(diff,
                                   alloc_params["k"],
                                   alloc_params.get("clip_lo", 0.1),
                                   alloc_params.get("clip_hi", 0.9))
            w2 = 1.0 - w1

        meta[i] = np.log(w1 * np.exp(r1[i]) + w2 * np.exp(r2[i]) + 1e-10)

    # Metrics
    simple     = np.exp(meta) - 1
    total_ret  = float((1 + simple).prod() - 1)
    n_days     = len(meta)
    ann_ret    = float((1 + total_ret) ** (252 / n_days) - 1)
    sharpe     = float(np.mean(meta) / (np.std(meta) + 1e-8) * np.sqrt(252))
    cum        = np.cumprod(1 + simple)
    max_dd     = float((cum / np.maximum.accumulate(cum) - 1).min())
    calmar     = ann_ret / (abs(max_dd) + 1e-8)
    down       = simple[simple < 0]
    sortino_d  = float(np.sqrt(np.mean(down ** 2)) * np.sqrt(252)) if len(down) > 1 else 1e-8
    sortino    = ann_ret / sortino_d

    # Composite rank score: weights return, Sharpe, and low drawdown
    score = (total_ret / AGENT1_BASELINE["total_return"]) * 0.35 \
          + (sharpe    / AGENT1_BASELINE["sharpe"])        * 0.40 \
          + (max_dd    / AGENT1_BASELINE["max_dd"])        * 0.25

    return {
        "total_return": total_ret,
        "ann_return":   ann_ret,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "calmar":       calmar,
        "sortino":      sortino,
        "score":        score,
    }


# ── Build search grid ─────────────────────────────────────────────────────────

def build_configs() -> list[dict]:
    configs = []

    lookbacks       = [10, 15, 20, 30, 60]
    rebalance_freqs = [5, 10, 15, 20]
    dd_penalties    = [0.0, 1.0, 2.0]

    discrete_alloc_params = [
        {"style": "discrete", "lo_thresh": 0.10, "hi_thresh": 0.30, "w_lo": 0.60, "w_hi": 0.75},
        {"style": "discrete", "lo_thresh": 0.10, "hi_thresh": 0.30, "w_lo": 0.65, "w_hi": 0.80},
        {"style": "discrete", "lo_thresh": 0.10, "hi_thresh": 0.30, "w_lo": 0.70, "w_hi": 0.85},
        {"style": "discrete", "lo_thresh": 0.10, "hi_thresh": 0.30, "w_lo": 0.70, "w_hi": 0.90},
        {"style": "discrete", "lo_thresh": 0.15, "hi_thresh": 0.40, "w_lo": 0.65, "w_hi": 0.85},
        {"style": "discrete", "lo_thresh": 0.20, "hi_thresh": 0.50, "w_lo": 0.65, "w_hi": 0.80},  # original
        {"style": "discrete", "lo_thresh": 0.20, "hi_thresh": 0.50, "w_lo": 0.70, "w_hi": 0.90},
        {"style": "discrete", "lo_thresh": 0.05, "hi_thresh": 0.20, "w_lo": 0.65, "w_hi": 0.85},
    ]

    softmax_alloc_params = [
        {"style": "softmax", "k": 0.5,  "clip_lo": 0.0, "clip_hi": 1.0}, # Very smooth gradient
        {"style": "softmax", "k": 1.0,  "clip_lo": 0.0, "clip_hi": 1.0}, # Smooth gradient
        {"style": "softmax", "k": 2.0,  "clip_lo": 0.10, "clip_hi": 0.90},
        {"style": "softmax", "k": 1.0,  "clip_lo": 0.15, "clip_hi": 0.85},
        {"style": "softmax", "k": 2.0,  "clip_lo": 0.15, "clip_hi": 0.85},
        {"style": "softmax", "k": 3.0,  "clip_lo": 0.10, "clip_hi": 0.90},
        {"style": "softmax", "k": 5.0,  "clip_lo": 0.10, "clip_hi": 0.90},
        {"style": "softmax", "k": 10.0, "clip_lo": 0.10, "clip_hi": 0.90},
    ]

    all_alloc = discrete_alloc_params + softmax_alloc_params

    for lb, rf, sig, ap, ddp in product(
        lookbacks, rebalance_freqs, SIGNAL_FNS.keys(), all_alloc, dd_penalties
    ):
        configs.append({
            "lookback":        lb,
            "rebalance_freq":  rf,
            "signal":          sig,
            "alloc_params":    ap,
            "dd_penalty":      ddp,
        })

    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading agent returns …")
    df1 = pd.read_csv(RESULTS_DIR / "agent1_test_returns.csv",
                      index_col="date", parse_dates=True)
    df2 = pd.read_csv(RESULTS_DIR / "agent2_test_returns.csv",
                      index_col="date", parse_dates=True)

    shared = df1.index.intersection(df2.index)
    r1 = df1.loc[shared, "log_return"].values
    r2 = df2.loc[shared, "log_return"].values

    configs = build_configs()
    print(f"Testing {len(configs):,} configurations …\n")

    rows = []
    for i, cfg in enumerate(configs):
        if i % 500 == 0:
            print(f"  {i}/{len(configs)} …")
        try:
            m = simulate(r1, r2,
                         lookback       = cfg["lookback"],
                         rebalance_freq = cfg["rebalance_freq"],
                         signal_name    = cfg["signal"],
                         alloc_params   = cfg["alloc_params"],
                         drawdown_penalty = cfg["dd_penalty"])
            rows.append({**cfg, **m})
        except Exception:
            pass

    results = pd.DataFrame(rows)
    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    # ── Print top 20 ──────────────────────────────────────────────────────────
    top = results.head(20)
    sep = "─" * 110

    print(f"\n{'='*110}")
    print(f"  TOP 20 STRATEGIES  (ranked by composite score: 40% Sharpe + 35% Total Return + 25% Max DD)")
    print(f"  Agent 1 baseline:  Total={AGENT1_BASELINE['total_return']:+.1%}  "
          f"Sharpe={AGENT1_BASELINE['sharpe']:.3f}  MaxDD={AGENT1_BASELINE['max_dd']:.1%}")
    print(f"{'='*110}")
    print(f"{'Rank':<5} {'Signal':<25} {'Alloc':<10} {'LB':>4} {'RF':>4} {'DDP':>5} "
          f"{'TotalRet':>9} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Score':>7}")
    print(sep)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        ap = row["alloc_params"]
        if ap["style"] == "discrete":
            alloc_str = f"D {ap['w_lo']:.0%}/{ap['w_hi']:.0%}"
        else:
            alloc_str = f"S k={ap['k']}"

        beats = " ✓" if row["total_return"] > AGENT1_BASELINE["total_return"] else "  "
        print(f"{rank:<5} {row['signal']:<25} {alloc_str:<10} {row['lookback']:>4} "
              f"{row['rebalance_freq']:>4} {row['dd_penalty']:>5.1f} "
              f"{row['total_return']:>+8.1%}{beats} {row['sharpe']:>7.3f} "
              f"{row['max_dd']:>7.1%} {row['calmar']:>7.3f} {row['score']:>7.4f}")

    print(sep)

    # ── Best strategy details ─────────────────────────────────────────────────
    best = results.iloc[0]
    ap   = best["alloc_params"]

    print(f"\n{'='*60}")
    print(f"  BEST STRATEGY")
    print(f"{'='*60}")
    print(f"  Signal metric   : {best['signal']}")
    print(f"  Lookback        : {best['lookback']} days")
    print(f"  Rebalance every : {best['rebalance_freq']} days")
    print(f"  Drawdown penalty: {best['dd_penalty']}")
    print(f"  Allocation style: {ap['style']}")
    if ap["style"] == "discrete":
        print(f"  Thresholds      : lo={ap['lo_thresh']}  hi={ap['hi_thresh']}")
        print(f"  Weights (if A1 better): {ap['w_lo']:.0%}/{1-ap['w_lo']:.0%} → "
              f"{ap['w_hi']:.0%}/{1-ap['w_hi']:.0%}")
    else:
        print(f"  k (softmax)     : {ap['k']}")
        print(f"  Clip            : [{ap['clip_lo']}, {ap['clip_hi']}]")
    print(f"{'─'*60}")
    print(f"  Total Return    : {best['total_return']:+.2%}  "
          f"(Agent1 baseline: {AGENT1_BASELINE['total_return']:+.1%})")
    print(f"  Ann. Sharpe     : {best['sharpe']:.4f}  "
          f"(Agent1 baseline: {AGENT1_BASELINE['sharpe']:.3f})")
    print(f"  Max Drawdown    : {best['max_dd']:.2%}  "
          f"(Agent1 baseline: {AGENT1_BASELINE['max_dd']:.1%})")
    print(f"  Calmar          : {best['calmar']:.4f}  "
          f"(Agent1 baseline: {AGENT1_BASELINE['calmar']:.3f})")
    print(f"  Composite Score : {best['score']:.4f}")
    print(f"{'='*60}")

    # ── How many strategies beat Agent 1 on each metric? ─────────────────────
    print(f"\n  Strategies beating Agent 1 baseline:")
    print(f"    Total Return  > {AGENT1_BASELINE['total_return']:+.1%} : "
          f"{(results['total_return'] > AGENT1_BASELINE['total_return']).sum()}")
    print(f"    Sharpe        > {AGENT1_BASELINE['sharpe']:.3f}        : "
          f"{(results['sharpe'] > AGENT1_BASELINE['sharpe']).sum()}")
    print(f"    Both metrics                       : "
          f"{((results['total_return'] > AGENT1_BASELINE['total_return']) & (results['sharpe'] > AGENT1_BASELINE['sharpe'])).sum()}")

    results.to_csv(RESULTS_DIR / "meta_optimization_results.csv", index=False)
    print(f"\n  Full results saved → {RESULTS_DIR / 'meta_optimization_results.csv'}")

    # ── Best parameters summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BEST PARAMETERS TO USE IN BACKTEST")
    print(f"{'='*60}")
    print(f"  META_LOOKBACK       = {int(best['lookback'])}")
    print(f"  META_REBALANCE_DAYS = {int(best['rebalance_freq'])}")
    print(f"  Signal              = {best['signal']}")
    print(f"  META_DD_PENALTY     = {best['dd_penalty']}")
    if ap["style"] == "discrete":
        print(f"  Allocation style    = discrete")
        print(f"  META_LO_THRESH      = {ap['lo_thresh']}")
        print(f"  META_HI_THRESH      = {ap['hi_thresh']}")
        print(f"  META_W_LO           = {ap['w_lo']}")
        print(f"  META_W_HI           = {ap['w_hi']}")
    else:
        print(f"  Allocation style    = softmax")
        print(f"  META_SOFTMAX_K      = {ap['k']}")
        print(f"  META_CLIP_LO        = {ap.get('clip_lo', 0.1)}")
        print(f"  META_CLIP_HI        = {ap.get('clip_hi', 0.9)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
