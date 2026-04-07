# TradingAgents × Autoresearch Blueprint

> Compiled from research session — April 2026
> Purpose: Foundation document for Claude Code build sessions
> Fork target: `https://github.com/lolisaigao1234/TradingAgents.git`
> Upstream: `https://github.com/TauricResearch/TradingAgents` (v0.2.0, Apache-2.0)

---

## 1. Core Thesis

Combine two Karpathy patterns into a self-evolving trading intelligence system:

1. **Autoresearch pattern** (`github.com/karpathy/autoresearch`) — an autonomous evolution loop where an LLM agent modifies code/config, runs an experiment against a measurable metric, keeps or discards the result, and repeats. The human writes `program.md` (the guardrails), the agent writes everything else.

2. **LLM Knowledge Base pattern** (Karpathy blog post) — raw data is ingested into `raw/`, an LLM "compiles" it into a structured markdown wiki with summaries, backlinks, cross-references, and index files. The wiki is the LLM's domain; the human rarely touches it. Queries against the wiki file results back in, creating a compounding flywheel.

**The synthesis**: TradingAgents already has multi-agent infrastructure (analysts, researchers, traders, risk managers on LangGraph). What it lacks is a feedback loop. We add the autoresearch evolution loop so the system improves itself over time, and the knowledge base wiki so it remembers what it learned.

---

## 2. Background: What TradingAgents Is

### Architecture (from the paper arXiv:2412.20138)

TradingAgents mirrors a real trading firm with specialized LLM-powered agents:

- **Analyst Team** (4 agents, run concurrently):
  - Fundamentals Analyst — company financials, intrinsic value, red flags
  - Sentiment Analyst — social media sentiment scoring, short-term market mood
  - News Analyst — global news, macroeconomic indicators, event impact
  - Technical Analyst — MACD, RSI, pattern detection, price forecasting

- **Researcher Team** (2 agents, structured debate):
  - Bull Researcher — argues for upside potential
  - Bear Researcher — argues for downside risk
  - They debate to produce a balanced assessment

- **Trader Agent** — synthesizes analyst + researcher reports into a trading decision (buy/sell/hold with magnitude and timing)

- **Risk Management Team** — evaluates portfolio exposure, volatility, liquidity; provides assessment reports

- **Portfolio Manager (Fund Manager)** — final approval/rejection gate; executes approved orders on simulated exchange

### Technical Stack

- Built on **LangGraph** for agent orchestration (flexible, modular graph)
- Supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, OpenRouter, Ollama
- Uses "deep think" models for complex reasoning and "quick think" models for data retrieval
- Data from **FinnHub API** (free tier) and **Alpha Vantage**
- No GPU required — pure LLM inference via API

### Key Entry Point

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

### Configurable Parameters

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"       # openai, google, anthropic, xai, openrouter, ollama
config["deep_think_llm"] = "claude-sonnet-4-20250514"
config["quick_think_llm"] = "claude-haiku-4-5-20251001"
config["max_debate_rounds"] = 2            # bull vs bear debate iterations
```

Full config in `tradingagents/default_config.py`.

### Repo Structure (Upstream)

```
TradingAgents/
├── assets/              # images, diagrams
├── cli/                 # CLI interface (main.py)
├── tradingagents/       # core package
│   ├── graph/           # LangGraph trading graph
│   ├── agents/          # individual agent definitions
│   ├── default_config.py
│   └── ...
├── main.py              # entry point
├── test.py              # basic tests
├── requirements.txt
├── pyproject.toml
└── uv.lock
```

---

## 3. Background: What Autoresearch Is

### Core Loop

```
while True:
    1. Agent reads program.md (human-written guardrails)
    2. Agent modifies train.py (the code under evolution)
    3. Run experiment (fixed 5-min time budget)
    4. Measure val_bpb (the metric)
    5. If improved → keep changes
       If worse   → revert changes
    6. Log results
    7. Repeat
```

### Key Design Principles

- **Single file to modify** — agent only touches `train.py`, keeps scope manageable
- **Fixed time budget** — every experiment is directly comparable regardless of what changed
- **Single metric** — `val_bpb` (validation bits per byte), lower is better
- **Human writes strategy, agent writes code** — the human iterates on `program.md`, not on `train.py`
- **Self-contained** — no complex configs, one GPU, one file, one metric

### The program.md Pattern

This is essentially a "skill file" — project-specific instructions that tell the agent what the project is, what the goals are, what constraints to respect, and how to run experiments. It is the single most transferable piece of autoresearch to other domains.

---

## 4. Background: Karpathy's LLM Knowledge Base Pattern

### Workflow

1. **Data Ingest**: Index source documents (articles, papers, repos, images) into `raw/` directory
2. **Compilation**: LLM incrementally "compiles" a wiki — collection of `.md` files in a directory structure with summaries, backlinks, categories, concept articles, cross-links
3. **IDE**: Obsidian as the frontend for viewing raw data, compiled wiki, and derived visualizations
4. **Q&A**: Once wiki is big enough (~100 articles, ~400K words), ask LLM complex questions against it — the LLM auto-maintains index files and brief summaries, reads related data easily at this scale
5. **Output**: Results rendered as markdown, slideshows (Marp), matplotlib images — all viewable in Obsidian. Outputs get filed back into the wiki (flywheel effect)
6. **Linting**: LLM health checks find inconsistencies, impute missing data, suggest new connections
7. **Extra tools**: Custom search engines, CLI tools handed to the LLM for larger queries

### Key Insight

At ~400K words, you don't need fancy RAG. The LLM maintains its own retrieval system via index files and summaries. No vector embeddings needed at this scale.

### Flywheel Effect

Every query enriches the wiki. Every linting pass improves data integrity. The knowledge base compounds in quality over time, unlike a static vector store.

---

## 5. The Combined Architecture

### Layer 1: program.md (Human Layer)

The human writes and iterates on this file. It defines:

- Evolution goals (maximize Sharpe ratio, minimize max drawdown, target cumulative returns)
- Constraints (max position size, sector exposure limits, rebalancing frequency)
- Backtest configuration (tickers, date ranges, initial capital, transaction costs)
- What the agent is allowed to modify (agent prompts, config params, graph structure)
- What the agent must NOT modify (backtest harness, data pipeline, metrics calculation)
- Experiment protocol (how many runs per config for statistical significance, how to handle LLM non-determinism)

### Layer 2: Evolution Agent

Reads wiki + metrics → proposes changes → runs backtest → evaluates:

```
while True:
    1. Read program.md for current goals and constraints
    2. Read wiki/ for past experiment results and institutional knowledge
    3. Propose a specific change (e.g., "increase debate rounds from 2 to 3")
    4. Apply the change to TradingAgents config/prompts/code
    5. Run backtest harness across defined universe
    6. Compute metrics (returns, Sharpe, max drawdown)
    7. If improved → commit change, update wiki with results
       If worse   → revert, log failure reason in wiki
    8. Repeat
```

### Layer 3: TradingAgents Core (LangGraph)

The existing multi-agent system, untouched structurally but with tunable parameters:

- Analyst Team → Researcher Team (debate) → Trader → Risk Management → Portfolio Manager
- The evolution agent modifies: agent prompts, debate rounds, risk thresholds, indicator weights, model selection, graph structure

### Layer 4: Backtest Engine

The "val_bpb" equivalent — provides the objective scoring surface:

- **Universe**: S&P 500 (VOO) + Magnificent 7 (AAPL, MSFT, TSLA, NVDA, META, AMZN, GOOGL)
- **Period**: 20 years of historical data
- **Data splits** (CRITICAL to avoid overfitting):
  - Train: 2005–2018
  - Validation: 2019–2022
  - Test: 2023–2025 (held out, never used during evolution)
- **Metrics**:
  - Cumulative returns (primary)
  - Sharpe ratio (risk-adjusted returns)
  - Maximum drawdown (worst peak-to-trough)
  - Win rate (% of profitable trades)
  - Compared against baselines: Buy & Hold, simple moving average crossover

### Layer 5: wiki/ (LLM-Maintained Knowledge Base)

The institutional memory. The LLM writes and maintains all of this:

```
wiki/
├── _index.md                    # master index of all articles
├── architecture/
│   ├── overview.md              # system architecture summary
│   ├── agent-graph.md           # LangGraph flow documentation
│   ├── analyst-team.md          # how analysts work
│   ├── researcher-debate.md     # bull vs bear debate mechanics
│   ├── risk-management.md       # risk evaluation process
│   └── backtest-harness.md      # how backtesting works
├── experiments/
│   ├── _experiment-log.md       # chronological log of all experiments
│   ├── exp-001-baseline.md      # baseline results
│   ├── exp-002-debate-rounds.md # effect of increasing debate rounds
│   ├── exp-003-risk-threshold.md
│   └── ...
├── strategies/
│   ├── momentum.md              # what we learned about momentum
│   ├── mean-reversion.md
│   ├── sentiment-driven.md
│   └── ...
├── market-insights/
│   ├── voo-patterns.md          # S&P 500 behavioral patterns
│   ├── tsla-volatility.md       # TSLA-specific findings
│   ├── crisis-periods.md        # 2008, 2020 crash behaviors
│   └── ...
├── decisions/
│   ├── why-3-debate-rounds.md   # decision log: why we settled on 3 rounds
│   ├── why-anthropic-models.md
│   └── ...
└── linting/
    ├── inconsistencies.md       # flagged data issues
    ├── missing-data.md          # gaps identified
    └── connection-candidates.md # suggested new articles/research
```

### Layer 6: Outputs (Obsidian-Viewable)

- Performance dashboards (matplotlib → PNG)
- Strategy comparison slides (Marp format)
- Experiment reports (markdown)
- All viewable in Obsidian alongside the wiki

---

## 6. What the Evolution Agent Can Modify

These are the "levers" the agent pulls between experiments:

### Agent Prompts
- Modify the system prompts for each agent (analyst, researcher, trader, risk manager)
- Adjust what data each analyst prioritizes
- Change debate framing for bull/bear researchers

### LangGraph Configuration
- `max_debate_rounds` — how many rounds of bull vs bear debate
- `llm_provider` and model selection per agent role
- Model temperature per agent (lower for risk management, higher for creative strategy)

### Strategy Parameters
- Risk tolerance thresholds
- Position sizing rules
- Rebalancing frequency
- Which technical indicators the technical analyst uses
- Sentiment scoring weights

### Graph Structure (Advanced)
- Add new specialist agents (e.g., a macro analyst, a volatility specialist)
- Change the agent communication flow
- Add/remove debate stages

### What Must NOT Be Modified
- The backtest harness itself (equivalent to autoresearch's `prepare.py`)
- Historical data pipeline
- Metrics calculation
- The wiki structure conventions

---

## 7. The Metric: Trading's "val_bpb"

### Primary Metric: Sharpe Ratio

The Sharpe ratio is the most natural single metric because it captures both returns AND risk:

```
Sharpe = (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
```

Higher is better. This is analogous to lower `val_bpb` in autoresearch.

### Secondary Metrics (Constraints, Not Optimization Targets)

- **Max Drawdown** < 25% (hard constraint — reject any config that exceeds this)
- **Win Rate** > 50% (soft constraint — prefer higher)
- **Cumulative Return** vs Buy & Hold (must beat the baseline)

### Handling Non-Determinism

Unlike autoresearch where `val_bpb` is deterministic, LLM trading decisions vary across runs. Protocol:

- Run each configuration **3 times minimum**
- Use **median Sharpe ratio** as the score (robust to outliers)
- Track **variance across runs** — high variance = unreliable config, penalize it
- Log all runs in wiki for transparency

---

## 8. Backtest Universe

### Tickers

| Ticker | Name | Why |
|--------|------|-----|
| VOO | Vanguard S&P 500 ETF | Broad market benchmark |
| AAPL | Apple | Mega-cap tech, steady growth |
| MSFT | Microsoft | Enterprise + cloud |
| TSLA | Tesla | High volatility stress test |
| NVDA | NVIDIA | AI boom / sector momentum |
| META | Meta Platforms | Sentiment-driven swings |
| AMZN | Amazon | E-commerce + cloud |
| GOOGL | Alphabet | Search + AI |

### Time Periods

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2005–2018 | Evolution agent optimizes against this |
| Validation | 2019–2022 | Check for overfitting during evolution |
| Test | 2023–2025 | Final held-out evaluation (touch ONCE) |

### Key Stress Periods Captured

- 2008–2009: Financial crisis
- 2011: European debt crisis
- 2015–2016: China slowdown fears
- 2018 Q4: Fed tightening selloff
- 2020 Q1: COVID crash
- 2022: Rate hiking bear market

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1–2)

- [ ] Fork TradingAgents, set up dev environment
- [ ] Create `program.md` with initial evolution rules
- [ ] Create `wiki/` directory structure with initial architecture docs
- [ ] Build `backtest_harness.py` wrapping `TradingAgentsGraph.propagate()` in a historical date loop
- [ ] Establish baseline metrics (run current TradingAgents unchanged against the universe)
- [ ] Log baseline results in `wiki/experiments/exp-001-baseline.md`

### Phase 2: Evolution Loop (Week 3–4)

- [ ] Build the evolution agent script that reads program.md + wiki, proposes changes, runs backtest
- [ ] Implement keep/revert logic based on Sharpe ratio improvement
- [ ] Implement experiment logging to wiki (auto-generated markdown)
- [ ] Run first 10–20 autonomous experiments
- [ ] Review results, refine program.md constraints

### Phase 3: Knowledge Base Enrichment (Week 5–6)

- [ ] Implement wiki linting (find inconsistencies, missing data, suggest new articles)
- [ ] Build wiki index auto-maintenance (the LLM updates `_index.md` and cross-links)
- [ ] Implement the flywheel — experiment outputs get filed back into wiki
- [ ] Build simple search tool over wiki (CLI, for LLM to use as a tool)

### Phase 4: Advanced Evolution (Week 7–8)

- [ ] Allow agent to modify LangGraph structure (add/remove agents)
- [ ] Allow agent to modify agent prompts (not just config params)
- [ ] Implement multi-ticker optimization (portfolio-level, not just per-ticker)
- [ ] Run validation split check — ensure train performance generalizes
- [ ] Expand to overnight autonomous runs (like autoresearch's "sleep and wake up to results")

### Phase 5: Evaluation & Hardening (Week 9–10)

- [ ] Run test split evaluation (one-time, final score)
- [ ] Compare against baselines: Buy & Hold, SMA crossover, original TradingAgents
- [ ] Document findings in wiki
- [ ] Generate final performance report (Marp slides, matplotlib charts)
- [ ] Consider: synthetic data generation + finetuning (Karpathy's "further explorations")

---

## 10. Technical Considerations

### API Cost Management

Each backtest day requires multiple LLM calls (4 analysts + 2 researchers + 1 trader + risk team). Over 20 years × 252 trading days × 8 tickers = ~40,000+ propagation calls per full backtest.

**Mitigation strategies:**
- Start narrow: 2020–2024 on VOO only for initial experiments
- Use cheaper models for quick-think roles (Haiku for analysts, Sonnet for debate/trading)
- Cache analyst outputs for identical market conditions
- Sample dates instead of running every trading day (weekly decisions reduce calls by 5x)

### LLM Non-Determinism

- Set temperature=0 for reproducibility where possible
- Run each config 3x minimum, use median
- Track variance — high variance configs are unreliable regardless of median score

### Data Pipeline

- Historical price data: Yahoo Finance API (free) or Alpha Vantage
- Fundamentals: FinnHub (free tier)
- Sentiment: may need to simulate or use archived data for historical periods
- News: archived sources or synthetic summaries for backtesting

### Overfitting Risk

This is the biggest risk. A system that perfectly trades the 2008 crash in hindsight is useless.

**Defenses:**
- Strict train/validation/test splits
- Monitor validation metrics during evolution — stop if validation diverges from train
- Prefer simpler configurations over complex ones (Occam's razor)
- Log overfitting warnings in wiki when train >> validation performance

---

## 11. File Structure for the Fork

```
TradingAgents/                    # forked repo
├── program.md                    # THE human-written guardrail (autoresearch pattern)
├── wiki/                         # LLM-maintained knowledge base
│   ├── _index.md
│   ├── architecture/
│   ├── experiments/
│   ├── strategies/
│   ├── market-insights/
│   ├── decisions/
│   └── linting/
├── evolution/                    # new: evolution loop code
│   ├── evolve.py                 # main evolution agent loop
│   ├── backtest_harness.py       # wraps TradingAgentsGraph in historical loop
│   ├── metrics.py                # Sharpe, MDD, returns calculation
│   └── experiment_logger.py      # auto-writes experiment results to wiki
├── tools/                        # new: CLI tools for wiki operations
│   ├── wiki_search.py            # search engine over wiki
│   ├── wiki_lint.py              # health check / linting
│   └── wiki_compile.py           # recompile index and cross-links
├── raw/                          # raw source data (Karpathy pattern)
│   ├── papers/
│   ├── market-data/
│   └── articles/
├── outputs/                      # generated outputs (viewable in Obsidian)
│   ├── reports/
│   ├── slides/
│   └── charts/
├── tradingagents/                # existing: core package (agent modifies this)
├── cli/                          # existing: CLI interface
├── main.py                       # existing: entry point
├── test.py                       # existing: tests
├── requirements.txt
├── pyproject.toml
└── .obsidian/                    # Obsidian vault config (for wiki viewing)
```

---

## 12. Key References

| Resource | URL | Relevance |
|----------|-----|-----------|
| TradingAgents (upstream) | `github.com/TauricResearch/TradingAgents` | Base framework |
| TradingAgents paper | `arxiv.org/abs/2412.20138` | Architecture details, experiment methodology |
| Autoresearch | `github.com/karpathy/autoresearch` | Evolution loop pattern, program.md pattern |
| Karpathy LLM Knowledge Bases | Blog post, March 2026 | Wiki compilation pattern, flywheel concept |
| LangGraph docs | `langchain-ai.github.io/langgraph/` | Agent orchestration framework |
| FinnHub API | `finnhub.io` | Financial data (free tier) |

---

## 13. Open Questions for Build Sessions

1. **Granularity of trading decisions**: Daily? Weekly? How often does `propagate()` run per backtest?
2. **Portfolio vs single-stock**: Does the evolution agent optimize per-ticker or across the whole portfolio?
3. **Agent prompt versioning**: How do we track prompt changes across experiments? Git commits? Wiki entries? Both?
4. **Obsidian plugins**: Which plugins for best wiki viewing? Marp for slides, what else?
5. **Cost ceiling**: What's the monthly API budget? This constrains how many experiments we can run.
6. **gstack integration**: How does gstack fit into the build workflow? Need to understand its capabilities for the Claude Code sessions.
7. **Live trading path**: Is the eventual goal paper trading → live trading, or purely research?

---

*This document is the starting point. The wiki will grow from here. The human writes program.md. The LLM writes everything else.*
