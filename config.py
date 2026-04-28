# ─────────────────────────────────────────────
# config.py  —  central config for the whole pipeline
# Change values here; everything else imports from here
# ─────────────────────────────────────────────

# ── Asset universes ──────────────────────────
# Both agents use the expanded 26-asset universe (24 equities + 2 bond ETFs).
AGENT1_TICKERS = [
    # Information Technology
    "AAPL", "MSFT", "NVDA",
    # Financials
    "GS", "JPM", "V",
    # Health Care
    "AMGN", "JNJ", "UNH",
    # Consumer Discretionary
    "AMZN", "HD", "MCD",
    # Communication Services
    "GOOG", "META", "NFLX",
    # Industrials
    "CAT", "HON", "WM",
    # Consumer Staples
    "KO", "COST", "PG",
    # Energy
    "COP", "XOM", "SLB",
    # Bonds
    "IEF", "SHY",
]

AGENT2_TICKERS = [
    # Information Technology
    "AAPL", "MSFT", "NVDA",
    # Financials
    "GS", "JPM", "V",
    # Health Care
    "AMGN", "JNJ", "UNH",
    # Consumer Discretionary
    "AMZN", "HD", "MCD",
    # Communication Services
    "GOOG", "META", "NFLX",
    # Industrials
    "CAT", "HON", "WM",
    # Consumer Staples
    "KO", "COST", "PG",
    # Energy
    "COP", "XOM", "SLB",
    # Bonds
    "IEF", "SHY",
]

# ── Date ranges ──────────────────────────────
# Download starts in 2009 so the 60-day + 252-day lookbacks are valid by 2010
DOWNLOAD_START  = "2009-01-01"
DOWNLOAD_END    = "2024-12-31"

TRAIN_START = "2010-01-01"
TRAIN_END   = "2018-01-01"

VAL_START   = "2018-01-01"
VAL_END     = "2020-01-01"

TEST_START  = "2020-01-01"   # includes COVID crash + 2022 rate-hike regime
TEST_END    = "2024-12-31"

# ── Feature engineering ──────────────────────
RETURN_WINDOWS    = [1, 5, 20, 60]   # lookback horizons for log returns (days)
VOLATILITY_WINDOW = 20               # rolling std window for vol feature
NORM_WINDOW       = 252              # rolling z-score window (1 trading year)

# ── Environment ──────────────────────────────
EPISODE_LENGTH = 252    # trading days per training episode (~1 year)

# ── Reward shaping ───────────────────────────
# Only Agent 1 uses drawdown and volatility penalties
# Tune these if Agent 1 is too passive (raise) or indistinguishable from Agent 2 (lower)
LAMBDA_DRAWDOWN   = 0.6    # weight on drawdown penalty (reduced from 1.0 for 26-asset universe)
LAMBDA_VOLATILITY = 0.35   # weight on rolling portfolio-volatility penalty (reduced from 0.5)

# Both agents pay this friction cost to prevent unrealistic high-turnover strategies
TURNOVER_COST = 0.001      # ~10 bps per unit of turnover
