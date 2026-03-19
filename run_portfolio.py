#!/usr/bin/env python3
"""Run TradingAgents analysis on full portfolio.

Usage:
    python run_portfolio.py                    # all tickers, today's date (2 batches)
    python run_portfolio.py NVDA COIN          # specific tickers only
    python run_portfolio.py --date 2026-03-10  # specific date
    python run_portfolio.py --batch-size 4     # custom batch size (default: 3)
    python run_portfolio.py --batch 1          # run only batch 1 (1-indexed)
    python run_portfolio.py --batch 2          # run only batch 2
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# --- Portfolio ---
PORTFOLIO = {
    "BTC": 70620.64,
    "NVDA": 183.41,
    "VOO": 629.47,
    "STUB": 23.50,
    "COIN": 373.48,
    "DEFT": 3.39,
    "CRSP": 68.68,
}

# BTC now supported via CoinGecko integration
SKIP_BY_DEFAULT = set()

BATCH_SIZE = 3  # default: 3 tickers per batch

def get_et_date():
    """Get current date in US Eastern Time."""
    et = timezone(timedelta(hours=-4))  # EDT (Mar-Nov)
    return datetime.now(et).strftime("%Y-%m-%d")

def split_batches(tickers, batch_size):
    """Split tickers into batches of batch_size."""
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

MAX_CONCURRENT_TICKERS = 3  # Vertex AI rate limit safety cap

def _make_config():
    """Build a shared config dict for TradingAgentsGraph."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "google"
    config["deep_think_llm"] = "gemini-3.1-pro-preview"
    config["quick_think_llm"] = "gemini-3.1-flash-lite-preview"
    config["google_vertexai"] = True
    config["google_cloud_project"] = os.getenv("GOOGLE_CLOUD_PROJECT", config.get("google_cloud_project"))
    config["google_cloud_location"] = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    return config

def _analyze_one(ticker, analysis_date, config):
    """Analyze a single ticker (designed to run in a thread)."""
    print(f"\n{'='*60}")
    print(f"  Analyzing {ticker} | Date: {analysis_date}")
    print(f"  Avg Cost: ${PORTFOLIO.get(ticker, '?')}")
    print(f"{'='*60}\n")

    try:
        ta = TradingAgentsGraph(debug=True, config=config)
        _, decision = ta.propagate(ticker, analysis_date)
        print(f"\n>>> {ticker} DECISION: {decision[:200]}...")
        return ticker, {
            "decision": decision,
            "avg_cost": PORTFOLIO.get(ticker),
            "status": "ok",
        }
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"\n>>> {ticker} ERROR: {err}")
        return ticker, {"decision": None, "error": err, "status": "error"}

def run_analysis(tickers, analysis_date):
    config = _make_config()
    results = {}

    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_TICKERS, len(tickers))) as pool:
        futures = {
            pool.submit(_analyze_one, ticker, analysis_date, config): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker, result = future.result()
            results[ticker] = result

    return results

def main():
    # Parse args
    args = sys.argv[1:]
    analysis_date = get_et_date()
    tickers = []
    batch_size = BATCH_SIZE
    batch_num = None  # None = run all batches sequentially

    i = 0
    while i < len(args):
        if args[i] == "--date" and i + 1 < len(args):
            analysis_date = args[i + 1]
            i += 2
        elif args[i] == "--batch-size" and i + 1 < len(args):
            batch_size = int(args[i + 1])
            i += 2
        elif args[i] == "--batch" and i + 1 < len(args):
            batch_num = int(args[i + 1])
            i += 2
        else:
            tickers.append(args[i].upper())
            i += 1

    if not tickers:
        tickers = [t for t in PORTFOLIO if t not in SKIP_BY_DEFAULT]

    batches = split_batches(tickers, batch_size)

    # If --batch specified, run only that batch
    if batch_num is not None:
        if batch_num < 1 or batch_num > len(batches):
            print(f"Error: --batch {batch_num} out of range (1-{len(batches)})")
            sys.exit(1)
        batches = [batches[batch_num - 1]]
        print(f"Portfolio Analysis | Date: {analysis_date} | Batch {batch_num}/{len(split_batches(tickers, batch_size))}")
    else:
        print(f"Portfolio Analysis | Date: {analysis_date} | {len(batches)} batches of {batch_size}")

    print(f"Tickers: {', '.join(tickers)}")

    all_results = {}

    for batch_idx, batch in enumerate(batches, 1):
        actual_batch_num = batch_num if batch_num else batch_idx
        total_batches = len(split_batches(tickers, batch_size))
        print(f"\n{'#'*60}")
        print(f"  BATCH {actual_batch_num}/{total_batches}: {', '.join(batch)}")
        print(f"{'#'*60}")

        results = run_analysis(batch, analysis_date)
        all_results.update(results)

    # Summary
    print(f"\n{'='*60}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'='*60}")
    for ticker, r in all_results.items():
        if r["status"] == "ok":
            dec_text = r["decision"][:100] if r["decision"] else "N/A"
            print(f"  {ticker:6s} | Cost ${r['avg_cost']:>10.2f} | {dec_text}")
        else:
            print(f"  {ticker:6s} | ERROR: {r['error']}")

    # Save results (merge with existing file for same date if running single batch)
    out_file = f"portfolio_{analysis_date}.json"
    existing = {}
    if batch_num is not None and os.path.exists(out_file):
        with open(out_file) as f:
            existing = json.load(f).get("results", {})
    existing.update(all_results)

    with open(out_file, "w") as f:
        json.dump({"date": analysis_date, "results": existing}, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
