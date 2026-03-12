"""CoinGecko crypto data source — provides OHLCV, indicators, and fundamentals for crypto assets."""

import os
import time
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated
from stockstats import wrap

from .config import get_config

# --- Ticker mapping ---
CRYPTO_TICKER_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "LTC": "litecoin",
    "NEAR": "near",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "APT": "aptos",
    "FIL": "filecoin",
    "AAVE": "aave",
}


def is_crypto_ticker(symbol: str) -> bool:
    """Check if a symbol is a known crypto ticker."""
    return symbol.upper().replace("-USD", "") in CRYPTO_TICKER_MAP


def _resolve_coin_id(ticker: str) -> str:
    """Convert ticker symbol to CoinGecko coin ID."""
    clean = ticker.upper().replace("-USD", "")
    if clean not in CRYPTO_TICKER_MAP:
        raise ValueError(f"Unknown crypto ticker: {ticker}. Known: {list(CRYPTO_TICKER_MAP.keys())}")
    return CRYPTO_TICKER_MAP[clean]


# --- Rate limiter (free tier: ~10-30 calls/min) ---
class _RateLimiter:
    def __init__(self, max_calls: int = 10, period: float = 60.0):
        self._max_calls = max_calls
        self._period = period
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < self._period]
            if len(self._calls) >= self._max_calls:
                sleep_time = self._period - (now - self._calls[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._calls.append(time.time())


_rate_limiter = _RateLimiter(max_calls=10, period=60.0)


class CoinGeckoRateLimitError(Exception):
    """Raised when CoinGecko API returns a rate limit error."""
    pass


def _make_request(endpoint: str, params: dict = None) -> dict:
    """HTTP GET to CoinGecko API with optional API key and rate limiting."""
    _rate_limiter.wait()

    base_url = "https://api.coingecko.com/api/v3"
    api_key = os.getenv("COINGECKO_API_KEY")

    headers = {}
    if api_key:
        # Pro API uses different base and header
        base_url = "https://pro-api.coingecko.com/api/v3"
        headers["x-cg-pro-api-key"] = api_key

    url = f"{base_url}/{endpoint.lstrip('/')}"
    params = params or {}

    resp = requests.get(url, params=params, headers=headers, timeout=30)

    if resp.status_code == 429:
        raise CoinGeckoRateLimitError(f"CoinGecko rate limit exceeded: {resp.text}")
    resp.raise_for_status()
    return resp.json()


# --- Indicator descriptions (reused from y_finance.py) ---
best_ind_params = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters in fast-moving markets."
    ),
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
    ),
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
        "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
    ),
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    ),
    "mfi": (
        "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
        "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
    ),
}


# --- Vendor functions ---

def get_stock_data(
    symbol: Annotated[str, "ticker symbol (e.g. BTC)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Get OHLCV data for a cryptocurrency from CoinGecko."""
    coin_id = _resolve_coin_id(symbol)
    config = get_config()

    # Check cache
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    cache_file = os.path.join(
        config["data_cache_dir"],
        f"{coin_id}-CoinGecko-data-{start_date}-{end_date}.csv",
    )

    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Use days-based endpoint (free tier doesn't support market_chart/range)
        days_diff = (end_dt - start_dt).days
        if days_diff <= 0:
            days_diff = 1
        # CoinGecko free tier caps at 365 days
        days_diff = min(days_diff, 365)

        result = _make_request(f"coins/{coin_id}/market_chart", {
            "vs_currency": "usd",
            "days": days_diff,
        })

        # Build OHLCV DataFrame from market_chart data
        # market_chart gives prices, market_caps, total_volumes as [timestamp, value] pairs
        prices = result.get("prices", [])
        volumes = result.get("total_volumes", [])

        if not prices:
            return f"No data found for '{symbol}' between {start_date} and {end_date}"

        # Group prices by day to construct OHLCV
        daily = {}
        for ts_ms, price in prices:
            day = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"open": price, "high": price, "low": price, "close": price, "prices": []}
            daily[day]["prices"].append(price)
            daily[day]["high"] = max(daily[day]["high"], price)
            daily[day]["low"] = min(daily[day]["low"], price)
            daily[day]["close"] = price  # last price of the day

        # Add volume data
        vol_by_day = {}
        for ts_ms, vol in volumes:
            day = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            vol_by_day[day] = vol  # last volume reading for the day

        rows = []
        for day in sorted(daily.keys()):
            d = daily[day]
            rows.append({
                "Date": day,
                "Open": round(d["open"], 2),
                "High": round(d["high"], 2),
                "Low": round(d["low"], 2),
                "Close": round(d["close"], 2),
                "Volume": round(vol_by_day.get(day, 0), 0),
            })

        data = pd.DataFrame(rows)
        data.to_csv(cache_file, index=False)

    csv_string = data.to_csv(index=False)
    header = f"# Crypto data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Source: CoinGecko ({_resolve_coin_id(symbol)})\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_indicators(
    symbol: Annotated[str, "ticker symbol (e.g. BTC)"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Calculate technical indicators for a cryptocurrency using stockstats."""
    # Defensive parsing: if the LLM passed multiple indicators as a comma-separated string,
    # split and recursively call for each, returning combined results.
    if "," in indicator:
        indicators = [item.strip() for item in indicator.split(",") if item.strip()]
        if indicators:
            return "\n".join(
                get_indicators(symbol, item, curr_date, look_back_days)
                for item in indicators
            )

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    coin_id = _resolve_coin_id(symbol)
    config = get_config()

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - timedelta(days=look_back_days)

    # Fetch enough historical data for indicator calculation (need extra for warmup)
    warmup_days = 365
    fetch_start = curr_date_dt - timedelta(days=warmup_days)
    fetch_end = curr_date_dt + timedelta(days=1)

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    cache_file = os.path.join(
        config["data_cache_dir"],
        f"{coin_id}-CoinGecko-ohlc-{fetch_start.strftime('%Y-%m-%d')}-{fetch_end.strftime('%Y-%m-%d')}.csv",
    )

    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file)
    else:
        # Use days-based endpoint (free tier doesn't support market_chart/range)
        days_diff = (fetch_end - fetch_start).days
        days_diff = min(days_diff, 365)

        result = _make_request(f"coins/{coin_id}/market_chart", {
            "vs_currency": "usd",
            "days": days_diff,
        })

        prices = result.get("prices", [])
        volumes = result.get("total_volumes", [])

        if not prices:
            return f"No data available for {symbol} to calculate indicators"

        # Build daily OHLCV
        daily = {}
        for ts_ms, price in prices:
            day = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"open": price, "high": price, "low": price, "close": price}
            daily[day]["high"] = max(daily[day]["high"], price)
            daily[day]["low"] = min(daily[day]["low"], price)
            daily[day]["close"] = price

        vol_by_day = {}
        for ts_ms, vol in volumes:
            day = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            vol_by_day[day] = vol

        rows = []
        for day in sorted(daily.keys()):
            d = daily[day]
            rows.append({
                "Date": day,
                "Open": d["open"],
                "High": d["high"],
                "Low": d["low"],
                "Close": d["close"],
                "Volume": vol_by_day.get(day, 0),
            })

        data = pd.DataFrame(rows)
        data.to_csv(cache_file, index=False)

    # Calculate indicator using stockstats
    data["Date"] = pd.to_datetime(data["Date"])
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df[indicator]  # trigger stockstats calculation

    # Build result for the requested date range
    ind_string = ""
    current_dt = curr_date_dt
    while current_dt >= before:
        date_str = current_dt.strftime("%Y-%m-%d")
        matching = df[df["Date"] == date_str]
        if not matching.empty:
            val = matching[indicator].values[0]
            if pd.isna(val):
                ind_string += f"{date_str}: N/A\n"
            else:
                ind_string += f"{date_str}: {val}\n"
        else:
            ind_string += f"{date_str}: N/A: No data for this date\n"
        current_dt -= timedelta(days=1)

    result_str = (
        f"## {indicator} values for {symbol.upper()} from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )
    return result_str


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol (e.g. BTC)"],
    curr_date: Annotated[str, "current date (optional)"] = None,
) -> str:
    """Get cryptocurrency fundamentals from CoinGecko."""
    coin_id = _resolve_coin_id(ticker)

    data = _make_request(f"coins/{coin_id}", {
        "localization": "false",
        "tickers": "false",
        "community_data": "true",
        "developer_data": "true",
    })

    market = data.get("market_data", {})

    fields = [
        ("Name", data.get("name")),
        ("Symbol", data.get("symbol", "").upper()),
        ("Current Price (USD)", market.get("current_price", {}).get("usd")),
        ("Market Cap (USD)", market.get("market_cap", {}).get("usd")),
        ("Market Cap Rank", data.get("market_cap_rank")),
        ("Total Volume (USD)", market.get("total_volume", {}).get("usd")),
        ("24h High (USD)", market.get("high_24h", {}).get("usd")),
        ("24h Low (USD)", market.get("low_24h", {}).get("usd")),
        ("Price Change 24h (%)", market.get("price_change_percentage_24h")),
        ("Price Change 7d (%)", market.get("price_change_percentage_7d")),
        ("Price Change 30d (%)", market.get("price_change_percentage_30d")),
        ("Price Change 1y (%)", market.get("price_change_percentage_1y")),
        ("All-Time High (USD)", market.get("ath", {}).get("usd")),
        ("ATH Change (%)", market.get("ath_change_percentage", {}).get("usd")),
        ("ATH Date", market.get("ath_date", {}).get("usd")),
        ("All-Time Low (USD)", market.get("atl", {}).get("usd")),
        ("ATL Change (%)", market.get("atl_change_percentage", {}).get("usd")),
        ("Circulating Supply", market.get("circulating_supply")),
        ("Total Supply", market.get("total_supply")),
        ("Max Supply", market.get("max_supply")),
        ("Fully Diluted Valuation (USD)", market.get("fully_diluted_valuation", {}).get("usd")),
    ]

    # Community data
    community = data.get("community_data", {})
    if community:
        fields.extend([
            ("Twitter Followers", community.get("twitter_followers")),
            ("Reddit Subscribers", community.get("reddit_subscribers")),
        ])

    # Developer data
    dev = data.get("developer_data", {})
    if dev:
        fields.extend([
            ("GitHub Stars", dev.get("stars")),
            ("GitHub Forks", dev.get("forks")),
            ("GitHub Subscribers", dev.get("subscribers")),
            ("Commit Count (4 weeks)", dev.get("commit_count_4_weeks")),
        ])

    lines = []
    for label, value in fields:
        if value is not None:
            lines.append(f"{label}: {value}")

    header = f"# Cryptocurrency Fundamentals for {ticker.upper()} ({data.get('name', '')})\n"
    header += f"# Source: CoinGecko\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + "\n".join(lines)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Balance sheet is not applicable for cryptocurrencies."""
    return (
        f"# Balance Sheet for {ticker.upper()}\n\n"
        "Balance sheet data is not applicable for cryptocurrencies. "
        "Cryptocurrencies are decentralized digital assets without traditional corporate financial statements. "
        "See get_fundamentals for on-chain and market data instead."
    )


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Cash flow is not applicable for cryptocurrencies."""
    return (
        f"# Cash Flow for {ticker.upper()}\n\n"
        "Cash flow data is not applicable for cryptocurrencies. "
        "Cryptocurrencies are decentralized digital assets without traditional corporate financial statements. "
        "See get_fundamentals for on-chain and market data instead."
    )


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "frequency of data"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
) -> str:
    """Income statement is not applicable for cryptocurrencies."""
    return (
        f"# Income Statement for {ticker.upper()}\n\n"
        "Income statement data is not applicable for cryptocurrencies. "
        "Cryptocurrencies are decentralized digital assets without traditional corporate financial statements. "
        "See get_fundamentals for on-chain and market data instead."
    )


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """Insider transactions are not applicable for cryptocurrencies."""
    return (
        f"# Insider Transactions for {ticker.upper()}\n\n"
        "Insider transaction data is not applicable for cryptocurrencies. "
        "Cryptocurrencies are decentralized — there are no corporate insiders. "
        "See get_fundamentals for community and developer activity data instead."
    )


def get_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Get crypto news via yfinance using {TICKER}-USD format."""
    from .yfinance_news import get_news_yfinance

    # yfinance supports crypto news via BTC-USD format
    clean = ticker.upper().replace("-USD", "")
    yf_ticker = f"{clean}-USD"
    return get_news_yfinance(yf_ticker, start_date, end_date)


def get_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 5,
) -> str:
    """Global news is asset-agnostic — passthrough to yfinance."""
    from .yfinance_news import get_global_news_yfinance
    return get_global_news_yfinance(curr_date, look_back_days, limit)
