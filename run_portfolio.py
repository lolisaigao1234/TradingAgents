#!/usr/bin/env python3
"""Run TradingAgents analysis on full portfolio.

Usage:
    python run_portfolio.py              # all tickers, today's date
    python run_portfolio.py NVDA COIN    # specific tickers only
    python run_portfolio.py --date 2026-03-10  # specific date
"""

import os
import sys
import json
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

def get_et_date():
    """Get current date in US Eastern Time."""
    et = timezone(timedelta(hours=-4))  # EDT (Mar-Nov)
    return datetime.now(et).strftime("%Y-%m-%d")

def run_analysis(tickers, analysis_date):
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "google"
    config["deep_think_llm"] = "gemini-2.5-pro"
    config["quick_think_llm"] = "gemini-2.5-pro"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1

    ta = TradingAgentsGraph(debug=True, config=config)
    results = {}

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"  Analyzing {ticker} | Date: {analysis_date}")
        print(f"  Avg Cost: ${PORTFOLIO.get(ticker, '?')}")
        print(f"{'='*60}\n")

        try:
            _, decision = ta.propagate(ticker, analysis_date)
            results[ticker] = {
                "decision": decision,
                "avg_cost": PORTFOLIO.get(ticker),
                "status": "ok",
            }
            print(f"\n>>> {ticker} DECISION: {decision[:200]}...")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            results[ticker] = {"decision": None, "error": err, "status": "error"}
            print(f"\n>>> {ticker} ERROR: {err}")

    return results

def main():
    # Parse args
    args = sys.argv[1:]
    analysis_date = get_et_date()
    tickers = []

    i = 0
    while i < len(args):
        if args[i] == "--date" and i + 1 < len(args):
            analysis_date = args[i + 1]
            i += 2
        else:
            tickers.append(args[i].upper())
            i += 1

    if not tickers:
        tickers = [t for t in PORTFOLIO if t not in SKIP_BY_DEFAULT]

    print(f"Portfolio Analysis | Date: {analysis_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Skipped (no data source): {', '.join(SKIP_BY_DEFAULT - set(tickers))}")

    results = run_analysis(tickers, analysis_date)

    # Summary
    print(f"\n{'='*60}")
    print("  PORTFOLIO SUMMARY")
    print(f"{'='*60}")
    for ticker, r in results.items():
        if r["status"] == "ok":
            # Extract just the action (BUY/SELL/HOLD) from decision text
            dec_text = r["decision"][:100] if r["decision"] else "N/A"
            print(f"  {ticker:6s} | Cost ${r['avg_cost']:>10.2f} | {dec_text}")
        else:
            print(f"  {ticker:6s} | ERROR: {r['error']}")

    # Save results
    out_file = f"portfolio_{analysis_date}.json"
    with open(out_file, "w") as f:
        json.dump({"date": analysis_date, "results": results}, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
