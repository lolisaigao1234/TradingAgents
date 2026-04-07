#!/usr/bin/env python3
"""Resumable portfolio runner — orchestrates serial single-ticker runs.

Spawns `python run_portfolio.py <TICKER> --date <DATE>` one at a time,
using the daily portfolio JSON as a checkpoint so interrupted runs can
be safely resumed.

Usage:
    python run_portfolio_resumable.py                          # all tickers, today (ET)
    python run_portfolio_resumable.py NVDA COIN                # specific tickers
    python run_portfolio_resumable.py --date 2026-04-06        # specific date
    python run_portfolio_resumable.py --dry-run                # show plan, don't run
    python run_portfolio_resumable.py --delay-seconds 5        # pause between tickers
    python run_portfolio_resumable.py --no-retry-errors        # skip status=error tickers
    python run_portfolio_resumable.py --force-all              # rerun even status=ok tickers
    python run_portfolio_resumable.py --force NVDA COIN        # rerun specific completed tickers
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

# Mirror the portfolio and date helper from run_portfolio.py so we can
# import without triggering its heavy side-effects (dotenv, tradingagents,
# Vertex credential injection).  Keep in sync manually.
PORTFOLIO = {
    "BTC": 70620.64,
    "NVDA": 183.41,
    "VOO": 629.47,
    "STUB": 23.50,
    "COIN": 373.48,
    "DEFT": 3.39,
    "CRSP": 68.68,
}
SKIP_BY_DEFAULT: set[str] = set()


def get_et_date() -> str:
    """Get current date in US Eastern Time (matches run_portfolio.py)."""
    et = timezone(timedelta(hours=-4))
    return datetime.now(et).strftime("%Y-%m-%d")


def _checkpoint_path(analysis_date: str) -> str:
    return f"portfolio_{analysis_date}.json"


def _read_checkpoint(path: str) -> dict:
    """Read checkpoint JSON, returning {} on missing/malformed file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("results", {})
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _classify_tickers(target_tickers, checkpoint, retry_errors, force_tickers, force_all):
    """Classify each ticker as skip/pending/retry.

    Returns (to_run, skipped) where to_run is an ordered list and
    skipped is a dict of {ticker: reason}.
    """
    to_run = []
    skipped = {}

    for ticker in target_tickers:
        entry = checkpoint.get(ticker)

        if force_all or ticker in force_tickers:
            to_run.append(ticker)
            continue

        if entry is None:
            to_run.append(ticker)
            continue

        status = entry.get("status")
        if status == "ok":
            skipped[ticker] = "already completed"
            continue

        if status == "error":
            if retry_errors:
                to_run.append(ticker)
            else:
                skipped[ticker] = "previous error (skipped, use without --no-retry-errors to retry)"
            continue

        # Unknown status — treat as pending
        to_run.append(ticker)

    return to_run, skipped


def _run_ticker(ticker: str, analysis_date: str) -> int:
    """Spawn run_portfolio.py for one ticker. Returns exit code."""
    cmd = [sys.executable, "run_portfolio.py", ticker, "--date", analysis_date]
    print(f"\n  >> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def _print_progress(idx, total, ticker, checkpoint_path):
    """Print progress after a ticker finishes."""
    cp = _read_checkpoint(checkpoint_path)
    entry = cp.get(ticker, {})
    status = entry.get("status", "unknown")
    marker = "OK" if status == "ok" else "FAIL" if status == "error" else "???"
    print(f"\n  [{idx}/{total}] {ticker}: {marker}")

    done = sum(1 for v in cp.values() if v.get("status") == "ok")
    errs = sum(1 for v in cp.values() if v.get("status") == "error")
    print(f"  Checkpoint: {done} ok, {errs} error, {len(cp)} total in file")


def main():
    parser = argparse.ArgumentParser(
        description="Resumable portfolio runner — serial single-ticker subprocess runs.",
    )
    parser.add_argument(
        "tickers", nargs="*", type=str.upper,
        help="Tickers to run (default: full portfolio)",
    )
    parser.add_argument("--date", default=None, help="Analysis date (default: today ET)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--delay-seconds", type=int, default=0, help="Pause between tickers (default: 0)")
    parser.add_argument(
        "--no-retry-errors", action="store_true",
        help="Skip tickers with status=error instead of retrying them",
    )
    parser.add_argument("--force", nargs="*", type=str.upper, default=[], help="Force rerun of specific tickers")
    parser.add_argument("--force-all", action="store_true", help="Force rerun of all tickers (ignore checkpoint)")

    args = parser.parse_args()

    analysis_date = args.date or get_et_date()
    target_tickers = args.tickers or [t for t in PORTFOLIO if t not in SKIP_BY_DEFAULT]
    retry_errors = not args.no_retry_errors
    force_tickers = set(args.force) if args.force else set()

    cp_path = _checkpoint_path(analysis_date)
    checkpoint = _read_checkpoint(cp_path)

    to_run, skipped = _classify_tickers(
        target_tickers, checkpoint, retry_errors, force_tickers, args.force_all,
    )

    # --- Plan summary ---
    print(f"{'='*60}")
    print(f"  Resumable Portfolio Runner")
    print(f"  Date:    {analysis_date}")
    print(f"  Target:  {len(target_tickers)} tickers — {', '.join(target_tickers)}")
    print(f"  Checkpoint file: {cp_path}")
    if checkpoint:
        ok = sum(1 for v in checkpoint.values() if v.get("status") == "ok")
        err = sum(1 for v in checkpoint.values() if v.get("status") == "error")
        print(f"  Checkpoint state: {ok} ok, {err} error, {len(checkpoint)} total")
    else:
        print(f"  Checkpoint state: no existing file")
    print(f"{'='*60}")

    if skipped:
        print(f"\n  Skipping {len(skipped)} tickers:")
        for t, reason in skipped.items():
            print(f"    {t:6s} — {reason}")

    if to_run:
        print(f"\n  Will run {len(to_run)} tickers: {', '.join(to_run)}")
    else:
        print(f"\n  Nothing to run — all target tickers are already handled.")

    if args.dry_run:
        print(f"\n  [DRY RUN] Exiting without running.\n")
        return

    if not to_run:
        return

    # --- Execute ---
    completed = 0
    failed = 0

    for i, ticker in enumerate(to_run, 1):
        print(f"\n{'─'*60}")
        print(f"  Starting ticker {i}/{len(to_run)}: {ticker}")
        print(f"{'─'*60}")

        exit_code = _run_ticker(ticker, analysis_date)

        # Re-read checkpoint after subprocess writes it
        _print_progress(i, len(to_run), ticker, cp_path)

        cp_after = _read_checkpoint(cp_path)
        entry = cp_after.get(ticker, {})
        if entry.get("status") == "ok":
            completed += 1
        else:
            failed += 1

        if i < len(to_run) and args.delay_seconds > 0:
            print(f"  Waiting {args.delay_seconds}s before next ticker...")
            time.sleep(args.delay_seconds)

    # --- Final summary ---
    final_cp = _read_checkpoint(cp_path)
    total_ok = sum(1 for v in final_cp.values() if v.get("status") == "ok")
    total_err = sum(1 for v in final_cp.values() if v.get("status") == "error")

    print(f"\n{'='*60}")
    print(f"  RESUMABLE RUN COMPLETE")
    print(f"{'='*60}")
    print(f"  This run:  {completed} completed, {failed} failed out of {len(to_run)} attempted")
    print(f"  Overall:   {total_ok} ok, {total_err} error in checkpoint")
    print(f"  File:      {cp_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
