# TASK: Build Backtest Evaluator (Phase 1A)

## Goal
Build `scripts/backtest-eval.py` — a standalone script that runs TradingAgents `propagate()` across historical dates for a ticker, compares decisions against actual price movements, and outputs directional accuracy metrics.

## Context
- TradingAgents has NO existing backtest mode
- `propagate(ticker, date)` returns `(final_state, decision_string)` with BUY/SELL/HOLD
- `process_signal()` returns LLM prose, need regex parser to extract BUY/SELL/HOLD
- Vertex AI credentials: reuse `_inject_vertex_credentials()` from `run_portfolio.py`
- Config: reuse `_make_config()` from `run_portfolio.py`

## Requirements

### 1. Signal Parser
- Extract BUY/SELL/HOLD from `process_signal()` LLM output
- Regex: `r'\b(BUY|SELL|HOLD)\b'` — first match wins
- If no match → mark as INCONCLUSIVE, don't count toward accuracy

### 2. Date-Range Runner
- Accept: ticker, list of historical dates
- For EACH date: create a FRESH `TradingAgentsGraph` instance (avoid memory contamination from `FinancialSituationMemory`)
- Call `propagate(ticker, date)` → capture decision
- Timeout: 600s per run, mark as CRASH if exceeded
- Error handling: wrap in try/except, continue on failure

### 3. Returns Calculator
- For each date with a decision, fetch actual price data from yfinance
- Compare: close price on trade_date vs close price 5 trading days later
- BUY correct if price went UP (>1%)
- SELL correct if price went DOWN (<-1%)  
- HOLD correct if price stayed FLAT (abs change ≤1%)
- Use Adjusted Close prices (handles splits/dividends)
- Handle weekends/holidays: find next available trading day

### 4. Metric Aggregator
- Directional accuracy: correct_decisions / total_decisions (exclude INCONCLUSIVE and CRASH)
- Magnitude-weighted accuracy: weight each correct decision by abs(price_change_pct)
- Output both metrics

### 5. Output Format
TSV to stdout:
```
ticker  dates_tested  decisions_made  accuracy  weighted_accuracy  details
NVDA    3             3               0.667     0.723              BUY(+3.2%):correct SELL(-1.5%):correct HOLD(+4.1%):wrong
```

### 6. CLI Interface
```bash
python scripts/backtest-eval.py --ticker NVDA --dates 2026-03-10,2026-03-14,2026-03-17
python scripts/backtest-eval.py --ticker NVDA --dates 2026-03-10,2026-03-14,2026-03-17 --verbose
```

## Technical Constraints
- Reuse `_inject_vertex_credentials()` and `_make_config()` from `run_portfolio.py`
- Create FRESH `TradingAgentsGraph` per (ticker, date) — do NOT reuse instances
- Use `source ~/anaconda3/bin/activate tradingagents` environment
- yfinance data: use Adjusted Close, handle non-trading days
- Exclude BTC for Phase 1 (no traditional close price)
- No new dependencies beyond what's in the existing environment

## Validation
After building, run these tests:
1. `python scripts/backtest-eval.py --ticker NVDA --dates 2026-03-10 --verbose` — single date smoke test
2. Verify signal parser handles various `process_signal()` output formats
3. Verify yfinance correctly returns prices and handles weekends

## Files to Create
- `scripts/backtest-eval.py` — main evaluator script

## Files to Read (for context)
- `run_portfolio.py` — credential injection + config pattern
- `tradingagents/graph/trading_graph.py` — propagate() API, process_signal()
- `tradingagents/agents/analysts/fundamentals_analyst.py` — prompt structure
- `tradingagents/default_config.py` — default configuration
- `tradingagents/dataflows/y_finance.py` — yfinance data access

## Do NOT Modify
- Any existing files. This task is CREATE ONLY.
