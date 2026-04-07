# LLM Wiki

> Source: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
> Author: Andrej Karpathy
> Retrieved: 2026-04-04

A pattern for building personal knowledge bases using LLMs.

This is an idea file, designed to be copy pasted to your own LLM Agent (e.g. OpenAI Codex, Claude Code, OpenCode / Pi, or etc.). Its goal is to communicate the high level idea, but your agent will build out the specifics in collaboration with you.

## The Core Idea

Most people's experience with LLMs and documents resembles RAG: upload files, retrieve relevant chunks at query time, generate answers. This works, but the LLM rediscovers knowledge from scratch on every question. There's no accumulation.

The alternative: the LLM **incrementally builds and maintains a persistent wiki** — a structured, interlinked collection of markdown files between you and raw sources. When adding a new source, the LLM reads it, extracts key information, and integrates it into the existing wiki — updating entity pages, revising topic summaries, noting contradictions, strengthening synthesis. Knowledge compiles once and stays current, not re-derived on every query.

**The wiki is a persistent, compounding artifact.** Cross-references exist already. Contradictions are flagged. Synthesis reflects everything read. The wiki gets richer with every source and question.

You source, explore, and ask questions. The LLM maintains everything — summarizing, cross-referencing, filing, bookkeeping. With one side showing the LLM agent and Obsidian open on the other: the LLM makes edits, you browse results in real time, following links and checking the graph view. Obsidian functions as the IDE; the LLM as programmer; the wiki as codebase.

**Applications:**
- Personal: tracking goals, health, psychology, self-improvement
- Research: going deep on topics over weeks, building comprehensive wikis
- Reading books: filing chapters, building pages for characters, themes, plot threads
- Business/team: internal wikis fed by Slack, transcripts, documents
- Competitive analysis, due diligence, trip planning, courses, hobbies

## Architecture

Three layers exist:

**Raw sources** — curated source documents: articles, papers, images, data files. Immutable; the LLM reads but never modifies. Source of truth.

**The wiki** — LLM-generated markdown files: summaries, entity pages, concept pages, comparisons, overviews, synthesis. The LLM owns this entirely, creating pages, updating them, maintaining cross-references, ensuring consistency. You read; the LLM writes.

**The schema** — document (CLAUDE.md for Claude Code or AGENTS.md for Codex) telling the LLM how the wiki is structured, conventions, workflows for ingesting sources, answering questions, maintaining the wiki. This configuration file makes the LLM a disciplined maintainer rather than generic chatbot. Co-evolve this over time.

## Operations

**Ingest.** Drop a new source and tell the LLM to process it. The LLM reads the source, discusses takeaways, writes a summary page, updates the index, updates relevant entity and concept pages, appends a log entry. A single source might touch 10-15 wiki pages. Prefer ingesting sources one at a time with involvement — read summaries, check updates, guide emphasis. Or batch-ingest with less supervision.

**Query.** Ask questions against the wiki. The LLM searches relevant pages, reads them, synthesizes answers with citations. Answers vary — markdown pages, comparison tables, slide decks (Marp), charts (matplotlib), canvas. **Good answers file back into the wiki as new pages.** Comparisons, analyses, discovered connections become valuable artifacts rather than disappearing into chat history. Explorations compound in the knowledge base like ingested sources.

**Lint.** Periodically ask the LLM to health-check the wiki. Look for: contradictions between pages, stale claims superseded by newer sources, orphan pages with no inbound links, important concepts lacking own pages, missing cross-references, data gaps. The LLM suggests new questions to investigate and sources to find. This maintains wiki health as it grows.

## Indexing and Logging

Two special files help navigation as the wiki grows, serving different purposes:

**index.md** is content-oriented. A catalog of everything — each page listed with link, one-line summary, optionally metadata like date or source count. Organized by category (entities, concepts, sources, etc.). The LLM updates it on every ingest. When answering queries, the LLM reads the index first to find relevant pages, then drills in. This works surprisingly well at moderate scale (~100 sources, ~hundreds of pages) avoiding embedding-based RAG infrastructure need.

**log.md** is chronological. Append-only record of what happened and when — ingests, queries, lint passes. Useful tip: if each entry starts with consistent prefix (e.g. `## [2026-04-02] ingest | Article Title`), the log becomes parseable with simple unix tools — `grep "^## \[" log.md | tail -5` gives the last 5 entries. The log shows the wiki's evolution timeline and helps the LLM understand recent actions.

## Optional: CLI Tools

At some point you may want building small tools helping the LLM operate on the wiki more efficiently. A search engine over wiki pages is most obvious — at small scale the index file suffices, but as the wiki grows you want proper search. [qmd](https://github.com/tobi/qmd) is good: a local search engine for markdown files with hybrid BM25/vector search and LLM re-ranking, all on-device. Has both CLI (so the LLM can shell out) and MCP server (so the LLM can use as native tool). You could build something simpler yourself — the LLM can help vibe-code a naive search script as need arises.

## Tips and Tricks

- **Obsidian Web Clipper** is a browser extension converting web articles to markdown. Very useful for quickly getting sources into your raw collection.

- **Download images locally.** In Obsidian Settings > Files and links, set "Attachment folder path" to fixed directory (e.g. `raw/assets/`). Then in Settings > Hotkeys, search for "Download" to find "Download attachments for current file" and bind to hotkey (e.g. Ctrl+Shift+D). After clipping an article, hit the hotkey and all images download to local disk. Optional but useful — lets the LLM view and reference images directly instead of relying on URLs that may break. Note that LLMs can't natively read markdown with inline images in one pass — the workaround is having the LLM read the text first, then view some or all referenced images separately for additional context. A bit clunky but works well enough.

- **Obsidian's graph view** is the best way to see the wiki's shape — what connects to what, which pages are hubs, which are orphans.

- **Marp** is a markdown-based slide deck format. Obsidian has a plugin for it. Useful for generating presentations directly from wiki content.

- **Dataview** is an Obsidian plugin running queries over page frontmatter. If your LLM adds YAML frontmatter to wiki pages (tags, dates, source counts), Dataview can generate dynamic tables and lists.

- The wiki is just a git repo of markdown files. You get version history, branching, and collaboration for free.

## Why This Works

The tedious part of maintaining a knowledge base isn't reading or thinking — it's bookkeeping. Updating cross-references, keeping summaries current, noting when new data contradicts old claims, maintaining consistency across dozens of pages. Humans abandon wikis because maintenance burden grows faster than value. LLMs don't get bored, don't forget to update cross-references, and can touch 15 files in one pass. The wiki stays maintained because maintenance cost is near zero.

The human's job: curate sources, direct analysis, ask good questions, think about what it all means. The LLM's job: everything else.

The idea relates in spirit to Vannevar Bush's Memex (1945) — a personal, curated knowledge store with associative trails between documents. Bush's vision was closer to this than what the web became: private, actively curated, with connections between documents as valuable as documents themselves. The part he couldn't solve: who does maintenance. The LLM handles that.

## Note

This document is intentionally abstract. It describes the idea, not a specific implementation. The exact directory structure, schema conventions, page formats, tooling — all depend on your domain, preferences, and LLM choice. Everything mentioned is optional and modular — pick what's useful, ignore what isn't. For example: your sources might be text-only, so you don't need image handling. Your wiki might be small enough that the index file suffices, no search engine required. You might not care about slide decks and want only markdown pages. You might want completely different output formats. The right way: share this with your LLM agent and work together to instantiate a version fitting your needs. The document's only job: communicate the pattern. Your LLM figures out the rest.
