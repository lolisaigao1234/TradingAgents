from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .coingecko import (
    get_stock_data as get_coingecko_stock_data,
    get_indicators as get_coingecko_indicators,
    get_fundamentals as get_coingecko_fundamentals,
    get_balance_sheet as get_coingecko_balance_sheet,
    get_cashflow as get_coingecko_cashflow,
    get_income_statement as get_coingecko_income_statement,
    get_insider_transactions as get_coingecko_insider_transactions,
    get_news as get_coingecko_news,
    get_global_news as get_coingecko_global_news,
    is_crypto_ticker,
    CoinGeckoRateLimitError,
)
from .wsj_news import get_news_wsj, get_global_news_wsj

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "coingecko",
    "wsj",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "coingecko": get_coingecko_stock_data,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "coingecko": get_coingecko_indicators,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "coingecko": get_coingecko_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "coingecko": get_coingecko_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "coingecko": get_coingecko_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "coingecko": get_coingecko_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "coingecko": get_coingecko_news,
        "wsj": get_news_wsj,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "coingecko": get_coingecko_global_news,
        "wsj": get_global_news_wsj,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "coingecko": get_coingecko_insider_transactions,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def _extract_ticker(method: str, args: tuple, kwargs: dict) -> str | None:
    """Extract ticker/symbol from positional args based on method signature.

    Most data methods take ticker/symbol as the first positional arg.
    get_global_news does not have a ticker arg.
    """
    # Methods where the first arg is a ticker/symbol
    ticker_first_methods = {
        "get_stock_data", "get_indicators", "get_fundamentals",
        "get_balance_sheet", "get_cashflow", "get_income_statement",
        "get_insider_transactions", "get_news",
    }
    if method in ticker_first_methods and args:
        return str(args[0]).upper().replace("-USD", "")
    return kwargs.get("ticker", kwargs.get("symbol", None))


def route_to_vendor_direct(method: str, vendor: str, *args, **kwargs):
    """Call a specific vendor implementation directly (no routing/fallback)."""
    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")
    if vendor not in VENDOR_METHODS[method]:
        raise ValueError(f"Vendor '{vendor}' not available for '{method}'")

    vendor_impl = VENDOR_METHODS[method][vendor]
    impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl
    return impl_func(*args, **kwargs)


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""

    # Auto-detect crypto tickers and route to coingecko
    ticker = _extract_ticker(method, args, kwargs)
    if ticker and is_crypto_ticker(ticker) and method in VENDOR_METHODS:
        if "coingecko" in VENDOR_METHODS[method]:
            try:
                return route_to_vendor_direct(method, "coingecko", *args, **kwargs)
            except CoinGeckoRateLimitError:
                pass  # Fall through to normal routing

    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except (AlphaVantageRateLimitError, CoinGeckoRateLimitError):
            continue  # Only rate limits trigger fallback

    raise RuntimeError(f"No available vendor for '{method}'")