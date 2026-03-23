#!/usr/bin/env python3
"""Backtest evaluator for TradingAgents.

Runs propagate() across historical dates, compares decisions against actual
price movements, and outputs directional accuracy metrics.

Usage:
    python scripts/backtest-eval.py --ticker NVDA --dates 2026-03-10
    python scripts/backtest-eval.py --ticker NVDA --dates 2026-03-10,2026-03-14,2026-03-17 --verbose
"""

import argparse
import os
import re
import signal
import sys
import json
import subprocess
import traceback
from datetime import datetime, timedelta

# Ensure project root is on sys.path so `tradingagents` is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Credential injection (reused from run_portfolio.py)
# ---------------------------------------------------------------------------

def _inject_vertex_credentials():
    """Retrieve Vertex AI SA JSON via secret-resolver and set env var."""
    if os.environ.get("GOOGLE_VERTEX_SA_JSON"):
        return

    try:
        request = json.dumps({
            "protocolVersion": 1,
            "ids": ["vertex-embed/oauth/serviceAccountJson"],
        })
        result = subprocess.run(
            ["sudo", "-n", "-u", "credproxy", "/usr/bin/node",
             "/opt/openclaw-security/secret-resolver.mjs"],
            input=request, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout)
            sa_json = payload["values"]["vertex-embed/oauth/serviceAccountJson"]
            os.environ["GOOGLE_VERTEX_SA_JSON"] = sa_json
            print("Vertex AI credentials loaded via secret-resolver", file=sys.stderr)
        else:
            print(f"Warning: secret-resolver failed (rc={result.returncode}): {result.stderr.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: could not load Vertex AI credentials: {e}", file=sys.stderr)


def _make_config():
    """Build a backtest-specific config for TradingAgentsGraph.

    Data leakage safety for backtesting:
      SAFE (honor trade_date): market_analyst (get_stock_data, get_indicators)
      UNSAFE (return live/current data, ignoring trade_date):
        - fundamentals_analyst: get_fundamentals, get_balance_sheet, get_cashflow,
          get_income_statement — yfinance returns latest-filed fundamentals regardless
          of date.
        - news_analyst: get_news, get_global_news — returns current headlines, not
          headlines as-of trade_date.
        - social_media_analyst: get_news — same issue as news.

    Phase 1: only enable market analyst (historical price/indicator data is date-safe).
    """
    from tradingagents.default_config import DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "google"
    config["deep_think_llm"] = "gemini-3.1-pro-preview"
    config["quick_think_llm"] = "gemini-3.1-flash-lite-preview"
    config["google_vertexai"] = True
    config["google_cloud_project"] = os.getenv("GOOGLE_CLOUD_PROJECT", config.get("google_cloud_project"))
    config["google_cloud_location"] = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    sa_json = os.getenv("GOOGLE_VERTEX_SA_JSON")
    if sa_json:
        config["google_service_account_json"] = sa_json
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["llm_timeout"] = 300       # backtest needs more time per request
    config["llm_max_retries"] = 2     # retry once on transient failures
    return config

# Analysts safe for backtesting (no data leakage). See _make_config() docstring.
_BACKTEST_SAFE_ANALYSTS = ["market"]


# ---------------------------------------------------------------------------
# Signal parser
# ---------------------------------------------------------------------------

_VALID_SIGNALS = {"BUY", "SELL", "HOLD"}
_SIGNAL_RE = re.compile(r'\b(BUY|SELL|HOLD)\b', re.IGNORECASE)


def parse_signal(raw_signal: str) -> str:
    """Extract BUY/SELL/HOLD from LLM prose. Returns INCONCLUSIVE if no match.

    Strategy: first try exact match on stripped/uppercased text (handles clean
    propagate() output). If that fails, use case-insensitive word-boundary regex.
    """
    # Fast path: propagate() may return a clean signal directly
    stripped = raw_signal.strip().upper()
    if stripped in _VALID_SIGNALS:
        return stripped

    # Fallback: search for signal word in prose
    match = _SIGNAL_RE.search(raw_signal)
    return match.group(1).upper() if match else "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Returns calculator
# ---------------------------------------------------------------------------

def get_close_price(ticker: str, target_date: str, search_forward_days: int = 10):
    """Get adjusted close price on target_date (or next available trading day).

    Returns (actual_date_str, price) or (None, None) if unavailable.
    """
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    # Fetch a window to account for weekends/holidays
    start = dt
    end = dt + timedelta(days=search_forward_days)

    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )

    if data.empty:
        return None, None

    # Use Close (auto_adjust=True makes Close equivalent to Adj Close)
    first_row = data.iloc[0]
    actual_date = data.index[0]
    if hasattr(actual_date, "strftime"):
        actual_date = actual_date.strftime("%Y-%m-%d")
    return str(actual_date), float(first_row["Close"])


def compute_return(ticker: str, trade_date: str, horizon_days: int = 5):
    """Compute percentage return from trade_date to horizon_days trading days later.

    Returns (entry_date, exit_date, pct_change, status) where status is one of:
      "OK" - full horizon data available
      "PARTIAL_HORIZON" - insufficient trading days, excluded from accuracy
      "NO_DATA" - no price data at all
    """
    entry_date, entry_price = get_close_price(ticker, trade_date)
    if entry_price is None:
        return None, None, None, "NO_DATA"

    dt = datetime.strptime(entry_date, "%Y-%m-%d")
    future_start = dt + timedelta(days=1)
    future_end = dt + timedelta(days=horizon_days * 3)

    data = yf.download(
        ticker,
        start=future_start.strftime("%Y-%m-%d"),
        end=future_end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )

    if data.empty:
        return entry_date, None, None, "NO_DATA"

    if len(data) < horizon_days:
        # Not enough trading days — do NOT silently fall back to shorter horizon
        return entry_date, None, None, "PARTIAL_HORIZON"

    exit_row = data.iloc[horizon_days - 1]
    exit_date = data.index[horizon_days - 1]
    if hasattr(exit_date, "strftime"):
        exit_date = exit_date.strftime("%Y-%m-%d")

    exit_price = float(exit_row["Close"])
    pct_change = ((exit_price - entry_price) / entry_price) * 100.0
    return entry_date, str(exit_date), pct_change, "OK"


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate_decision(decision: str, pct_change: float, threshold: float = 1.0) -> bool:
    """Check if a decision was directionally correct given the price change.

    Boundaries are consistent: BUY >= threshold, SELL <= -threshold, HOLD inside.
    """
    if decision == "BUY":
        return pct_change >= threshold
    elif decision == "SELL":
        return pct_change <= -threshold
    elif decision == "HOLD":
        return abs(pct_change) < threshold
    return False


# ---------------------------------------------------------------------------
# Date-range runner
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("propagate() timed out")


def run_single_date(ticker: str, date: str, config: dict, verbose: bool = False,
                    selected_analysts: list = None):
    """Run propagate() for a single (ticker, date). Returns a result dict."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    if selected_analysts is None:
        selected_analysts = _BACKTEST_SAFE_ANALYSTS

    result = {
        "date": date,
        "raw_signal": None,
        "decision": None,
        "status": "ok",
        "error": None,
    }

    # Set 600s timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(600)

    try:
        if verbose:
            print(f"  [{date}] Running propagate({ticker}, {date})...", file=sys.stderr)

        ta = TradingAgentsGraph(debug=verbose, config=config,
                                selected_analysts=selected_analysts)
        _, raw_signal = ta.propagate(ticker, date)

        result["raw_signal"] = raw_signal
        result["decision"] = parse_signal(raw_signal)

        if verbose:
            print(f"  [{date}] Raw signal: {raw_signal[:200]}...", file=sys.stderr)
            print(f"  [{date}] Parsed decision: {result['decision']}", file=sys.stderr)

    except TimeoutError:
        result["status"] = "CRASH"
        result["error"] = "timeout (600s)"
        if verbose:
            print(f"  [{date}] TIMEOUT after 600s", file=sys.stderr)
    except Exception as e:
        result["status"] = "CRASH"
        result["error"] = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"  [{date}] ERROR: {result['error']}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _validate_date(date_str: str) -> str:
    """Validate YYYY-MM-DD format. Raises ValueError if invalid."""
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def main():
    parser = argparse.ArgumentParser(description="Backtest evaluator for TradingAgents")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--dates", required=True, help="Comma-separated dates (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress to stderr")
    parser.add_argument("--horizon", type=int, default=5, help="Trading days to measure return (default: 5)")
    args = parser.parse_args()

    # --- CLI validation (#6) ---
    ticker = args.ticker.strip().upper()
    if not ticker:
        parser.error("--ticker must not be empty")

    if args.horizon < 1:
        parser.error("--horizon must be >= 1")

    # Parse dates: drop empty tokens, validate format
    raw_tokens = [d.strip() for d in args.dates.split(",")]
    dates = []
    for tok in raw_tokens:
        if not tok:
            continue  # skip empty tokens from trailing commas etc.
        try:
            _validate_date(tok)
        except ValueError:
            parser.error(f"Invalid date format '{tok}', expected YYYY-MM-DD")
        dates.append(tok)

    if not dates:
        parser.error("--dates must contain at least one valid date")

    verbose = args.verbose

    # Inject credentials
    _inject_vertex_credentials()

    config = _make_config()

    if verbose:
        print(f"Backtest: {ticker} | Dates: {dates} | Horizon: {args.horizon}d", file=sys.stderr)
        print(f"  Safe analysts: {_BACKTEST_SAFE_ANALYSTS}", file=sys.stderr)

    # Run propagate for each date
    run_results = []
    for date in dates:
        r = run_single_date(ticker, date, config, verbose)
        run_results.append(r)

    # Compute returns and evaluate
    details = []
    correct = 0
    total = 0
    weighted_correct = 0.0
    total_weight = 0.0

    for r in run_results:
        decision = r["decision"]

        if r["status"] == "CRASH":
            details.append(f"CRASH({r['date']})")
            continue

        if decision == "INCONCLUSIVE":
            details.append(f"INCONCLUSIVE({r['date']})")
            continue

        # Per-date try/except for price data (#3) — one date's yfinance failure
        # must not kill the whole run.
        try:
            entry_date, exit_date, pct_change, price_status = compute_return(
                ticker, r["date"], args.horizon
            )
        except Exception as e:
            details.append(f"{decision}({r['date']}):PRICE_ERROR")
            if verbose:
                print(f"  [{r['date']}] PRICE_ERROR: {e}", file=sys.stderr)
            continue

        if price_status == "PARTIAL_HORIZON":
            details.append(f"{decision}({r['date']}):PARTIAL_HORIZON")
            continue

        if pct_change is None:
            details.append(f"{decision}({r['date']}):no_price_data")
            continue

        is_correct = evaluate_decision(decision, pct_change)
        label = "correct" if is_correct else "wrong"
        details.append(f"{decision}({pct_change:+.1f}%):{label}")

        total += 1
        if is_correct:
            correct += 1
            weighted_correct += abs(pct_change)
        total_weight += abs(pct_change)

    accuracy = correct / total if total > 0 else 0.0
    weighted_accuracy = weighted_correct / total_weight if total_weight > 0 else 0.0

    # TSV output
    details_str = " ".join(details) if details else "none"
    print("\t".join([
        ticker,
        str(len(dates)),
        str(total),
        f"{accuracy:.3f}",
        f"{weighted_accuracy:.3f}",
        details_str,
    ]))


if __name__ == "__main__":
    main()
