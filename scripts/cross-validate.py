#!/usr/bin/env python3
"""Cross-validation orchestrator for dual-agent review.

Takes top-K experiment results from experiment-harness.py, dispatches both
Claude Code and Codex to independently review each candidate's diff, compares
their reviews, and outputs a structured verdict.

Usage:
    python scripts/cross-validate.py \\
      --worktree /tmp/evolve-market-001-12345 \\
      --baseline-sha abc1234 \\
      --candidates "var-001:def5678,var-003:ghi9012" \\
      --manifest experiments/nvda-market.yaml \\
      --baseline-metric 0.667

    python scripts/cross-validate.py --dry-run \\
      --worktree /tmp/evolve-market-001-12345 \\
      --baseline-sha abc1234 \\
      --candidates "var-001:def5678,var-003:ghi9012" \\
      --manifest experiments/nvda-market.yaml \\
      --baseline-metric 0.667
"""

import argparse
import os
import re
import subprocess
import sys
import yaml


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Agent review timeout in seconds
_REVIEW_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Manifest loader (minimal — only need allowed_files)
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    """Load experiment manifest YAML and return the experiment dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "experiment" not in raw:
        print("Error: manifest must have top-level 'experiment' key", file=sys.stderr)
        sys.exit(1)

    return raw["experiment"]


# ---------------------------------------------------------------------------
# Candidate parsing
# ---------------------------------------------------------------------------

def parse_candidates(candidates_str: str) -> list[dict]:
    """Parse 'var-001:sha1,var-003:sha2' into list of dicts."""
    candidates = []
    for entry in candidates_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            print(f"Error: candidate entry must be 'name:sha', got '{entry}'", file=sys.stderr)
            sys.exit(1)
        name, sha = entry.split(":", 1)
        candidates.append({"name": name.strip(), "sha": sha.strip()})
    return candidates


# ---------------------------------------------------------------------------
# Diff generation
# ---------------------------------------------------------------------------

def generate_diff(worktree_path: str, baseline_sha: str, candidate_sha: str) -> str:
    """Generate git diff between baseline and candidate."""
    result = subprocess.run(
        ["git", "diff", f"{baseline_sha}..{candidate_sha}"],
        capture_output=True, text=True, cwd=worktree_path,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")
    return result.stdout


# ---------------------------------------------------------------------------
# Review prompt builder
# ---------------------------------------------------------------------------

def build_review_prompt(
    diff: str,
    allowed_files: list[str],
    baseline_metric: float,
    candidate_metric: float | None,
    direction: str,
) -> str:
    """Build the review prompt sent to both agents."""
    if candidate_metric is not None:
        metric_line = (
            f"Metric change: baseline {baseline_metric:.3f} → "
            f"candidate {candidate_metric:.3f} ({direction})"
        )
    else:
        metric_line = f"Baseline metric: {baseline_metric:.3f} ({direction})"

    return (
        "Review this diff for a trading analyst prompt optimization experiment.\n"
        "\n"
        f"The diff modifies: {', '.join(allowed_files)}\n"
        f"{metric_line}\n"
        "\n"
        "Check for:\n"
        "1. Data leakage: Does the prompt encode specific dates, prices, or future-looking information?\n"
        "2. Overfitting: Is the prompt change too narrow/specific to work only on the test dates?\n"
        "3. Prompt quality: Is the new prompt well-structured, clear, and generalizable?\n"
        "4. Regression risk: Could this change break behavior on dates/tickers not tested?\n"
        "5. Code safety: Any unintended changes outside the system_message string?\n"
        "\n"
        'Output a numbered list of issues found. For each: severity (CRITICAL/HIGH/MEDIUM/LOW), description.\n'
        'If no issues: output "PASS: No issues found."\n'
        "\n"
        "Diff:\n"
        "```diff\n"
        f"{diff}\n"
        "```"
    )


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def call_codex(review_prompt: str, worktree_path: str) -> str:
    """Call Codex to review the diff. Returns raw output."""
    result = subprocess.run(
        ["codex", "exec", "--full-auto", review_prompt],
        capture_output=True, text=True,
        timeout=_REVIEW_TIMEOUT, cwd=worktree_path,
    )
    if result.returncode != 0:
        return f"AGENT_ERROR: Codex returned rc={result.returncode}: {result.stderr[:200]}"
    return result.stdout.strip()


def call_claude(review_prompt: str, worktree_path: str) -> str:
    """Call Claude Code to review the diff. Returns raw output."""
    result = subprocess.run(
        ["claude", "--permission-mode", "bypassPermissions", "--print", review_prompt],
        capture_output=True, text=True,
        timeout=_REVIEW_TIMEOUT, cwd=worktree_path,
    )
    if result.returncode != 0:
        return f"AGENT_ERROR: Claude returned rc={result.returncode}: {result.stderr[:200]}"
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Issue parsing
# ---------------------------------------------------------------------------

_ISSUE_RE = re.compile(
    r"^\s*\d+[\.\)]\s*[`*]?(CRITICAL|HIGH|MEDIUM|LOW)[`*]?\s*[:\-–—]\s*(.+)",
    re.IGNORECASE | re.MULTILINE,
)

_PASS_RE = re.compile(r"PASS\s*[:\-–—]?\s*[Nn]o issues", re.IGNORECASE)


def parse_issues(raw_output: str) -> list[dict]:
    """Parse agent output into structured issue list.

    Returns list of {'severity': str, 'description': str}.
    If output indicates PASS, returns empty list.
    If output indicates agent error, returns a single CRITICAL issue.
    """
    if raw_output.startswith("AGENT_ERROR:"):
        return [{"severity": "CRITICAL", "description": raw_output}]

    if _PASS_RE.search(raw_output):
        return []

    issues = []
    for match in _ISSUE_RE.finditer(raw_output):
        severity = match.group(1).upper()
        description = match.group(2).strip()
        issues.append({"severity": severity, "description": description})

    return issues


# ---------------------------------------------------------------------------
# Issue comparison (Jaccard similarity on words)
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    """Extract lowercase words from text."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_SIMILARITY_THRESHOLD = 0.3


def compare_issues(
    issues_a: list[dict],
    issues_b: list[dict],
) -> dict:
    """Compare issue lists from two agents.

    Returns:
        {
            'agreed': [{'issue_a': ..., 'issue_b': ..., 'match': 'AGREED'|'PARTIAL_AGREE'}],
            'only_a': [issue, ...],
            'only_b': [issue, ...],
        }
    """
    matched_b = set()
    agreed = []
    only_a = []

    for ia in issues_a:
        words_a = _word_set(ia["description"])
        best_sim = 0.0
        best_idx = -1
        for j, ib in enumerate(issues_b):
            if j in matched_b:
                continue
            sim = _jaccard(words_a, _word_set(ib["description"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = j

        if best_sim >= _SIMILARITY_THRESHOLD and best_idx >= 0:
            matched_b.add(best_idx)
            match_type = (
                "AGREED" if ia["severity"] == issues_b[best_idx]["severity"]
                else "PARTIAL_AGREE"
            )
            agreed.append({
                "issue_a": ia,
                "issue_b": issues_b[best_idx],
                "match": match_type,
            })
        else:
            only_a.append(ia)

    only_b = [ib for j, ib in enumerate(issues_b) if j not in matched_b]

    return {"agreed": agreed, "only_a": only_a, "only_b": only_b}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def compute_verdict(comparison: dict, issues_codex: list, issues_claude: list) -> str:
    """Determine overall verdict: APPROVE / NEEDS_ATTENTION / REJECT."""
    # Both agents found no issues
    if not issues_codex and not issues_claude:
        return "APPROVE"

    # Any CRITICAL issue agreed by both agents → REJECT
    for a in comparison["agreed"]:
        if a["issue_a"]["severity"] == "CRITICAL" or a["issue_b"]["severity"] == "CRITICAL":
            return "REJECT"

    # Any disagreements or partial agreements → NEEDS_ATTENTION
    if comparison["only_a"] or comparison["only_b"]:
        return "NEEDS_ATTENTION"

    # All issues agreed, none critical
    has_high = any(
        a["issue_a"]["severity"] == "HIGH" or a["issue_b"]["severity"] == "HIGH"
        for a in comparison["agreed"]
    )
    if has_high:
        return "NEEDS_ATTENTION"

    return "APPROVE"


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def format_report(
    candidate_name: str,
    candidate_metric: float | None,
    raw_codex: str,
    raw_claude: str,
    issues_codex: list[dict],
    issues_claude: list[dict],
    comparison: dict,
    verdict: str,
) -> str:
    """Format a human-readable report for one candidate."""
    metric_str = f" (metric: {candidate_metric:.3f})" if candidate_metric is not None else ""
    lines = []
    lines.append(f"=== Candidate: {candidate_name}{metric_str} ===")

    codex_count = len(issues_codex)
    claude_count = len(issues_claude)

    # Codex summary
    if codex_count == 0:
        lines.append("Codex verdict: PASS (0 issues)")
    else:
        sev_list = ", ".join(f"{i['severity']}: {i['description'][:60]}" for i in issues_codex)
        lines.append(f"Codex verdict: {codex_count} issue(s) ({sev_list})")

    # Claude summary
    if claude_count == 0:
        lines.append("Claude verdict: PASS (0 issues)")
    else:
        sev_list = ", ".join(f"{i['severity']}: {i['description'][:60]}" for i in issues_claude)
        lines.append(f"Claude verdict: {claude_count} issue(s) ({sev_list})")

    # Agreement
    n_agreed = len(comparison["agreed"])
    n_disagree = len(comparison["only_a"]) + len(comparison["only_b"])
    if n_disagree == 0 and n_agreed == 0:
        lines.append("Agreement: FULL — both agents found no issues")
    elif n_disagree == 0:
        lines.append(f"Agreement: FULL — {n_agreed} issue(s) agreed")
    else:
        lines.append(f"Agreement: PARTIAL — {n_disagree} disagreement(s)")

    lines.append(f"Overall: {verdict}")

    # Disagreements detail
    if comparison["only_a"] or comparison["only_b"]:
        lines.append("")
        lines.append("Disagreements:")
        for issue in comparison["only_a"]:
            lines.append(f"  - Codex found {issue['severity']} issue not flagged by Claude: \"{issue['description']}\"")
        for issue in comparison["only_b"]:
            lines.append(f"  - Claude found {issue['severity']} issue not flagged by Codex: \"{issue['description']}\"")

    # Recommendation
    lines.append("")
    if verdict == "APPROVE":
        lines.append("Recommendation: Safe to merge.")
    elif verdict == "REJECT":
        lines.append("Recommendation: Do NOT merge. Critical issues found by both agents.")
    else:
        severity_tags = set()
        for issue in comparison["only_a"] + comparison["only_b"]:
            severity_tags.add(issue["severity"])
        for a in comparison["agreed"]:
            if a["match"] == "PARTIAL_AGREE":
                severity_tags.add(a["issue_a"]["severity"])
        tag_str = "/".join(sorted(severity_tags)) if severity_tags else "flagged"
        lines.append(f"Recommendation: Review the {tag_str} issue(s) manually before merging.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_cross_validation(
    worktree_path: str,
    baseline_sha: str,
    candidates: list[dict],
    manifest: dict,
    baseline_metric: float,
    dry_run: bool = False,
) -> list[dict]:
    """Run cross-validation for all candidates.

    Returns list of per-candidate result dicts with keys:
        name, verdict, report, comparison
    """
    allowed_files = manifest["allowed_files"]
    direction = manifest.get("direction", "maximize")
    results = []

    print(f"\n{'#' * 60}", file=sys.stderr)
    print("  Cross-Validation Orchestrator", file=sys.stderr)
    print(f"  Worktree: {worktree_path}", file=sys.stderr)
    print(f"  Baseline SHA: {baseline_sha[:12]}", file=sys.stderr)
    print(f"  Baseline metric: {baseline_metric:.3f}", file=sys.stderr)
    print(f"  Candidates: {len(candidates)}", file=sys.stderr)
    print(f"  Allowed files: {allowed_files}", file=sys.stderr)
    print(f"  Dry run: {dry_run}", file=sys.stderr)
    print(f"{'#' * 60}\n", file=sys.stderr)

    for candidate in candidates:
        name = candidate["name"]
        sha = candidate["sha"]
        candidate_metric = candidate.get("metric")

        print(f"  --- Reviewing {name} (sha: {sha[:12]}) ---", file=sys.stderr)

        # Step 1: Generate diff
        try:
            diff = generate_diff(worktree_path, baseline_sha, sha)
        except RuntimeError as e:
            print(f"  Error generating diff for {name}: {e}", file=sys.stderr)
            results.append({
                "name": name,
                "verdict": "REJECT",
                "report": f"=== Candidate: {name} ===\nError: could not generate diff: {e}",
                "comparison": None,
            })
            continue

        if not diff.strip():
            print(f"  {name}: empty diff, skipping", file=sys.stderr)
            results.append({
                "name": name,
                "verdict": "APPROVE",
                "report": f"=== Candidate: {name} ===\nEmpty diff — no changes to review.",
                "comparison": None,
            })
            continue

        # Step 2: Build review prompt
        review_prompt = build_review_prompt(
            diff, allowed_files, baseline_metric, candidate_metric, direction,
        )

        if dry_run:
            print(f"  [DRY RUN] Would send review prompt ({len(review_prompt)} chars) to both agents", file=sys.stderr)
            print(f"  [DRY RUN] Diff size: {len(diff)} chars", file=sys.stderr)
            results.append({
                "name": name,
                "verdict": "DRY_RUN",
                "report": (
                    f"=== Candidate: {name} ===\n"
                    f"[DRY RUN] Diff: {len(diff)} chars\n"
                    f"[DRY RUN] Review prompt: {len(review_prompt)} chars\n"
                    f"[DRY RUN] Would dispatch to Codex + Claude Code"
                ),
                "comparison": None,
            })
            continue

        # Step 3: Dispatch to both agents
        print(f"  Calling Codex...", file=sys.stderr)
        try:
            raw_codex = call_codex(review_prompt, worktree_path)
        except subprocess.TimeoutExpired:
            raw_codex = "AGENT_ERROR: Codex timed out after 120s"
        print(f"  Calling Claude Code...", file=sys.stderr)
        try:
            raw_claude = call_claude(review_prompt, worktree_path)
        except subprocess.TimeoutExpired:
            raw_claude = "AGENT_ERROR: Claude Code timed out after 120s"

        # Step 4: Parse issues
        issues_codex = parse_issues(raw_codex)
        issues_claude = parse_issues(raw_claude)

        print(f"  Codex: {len(issues_codex)} issue(s)", file=sys.stderr)
        print(f"  Claude: {len(issues_claude)} issue(s)", file=sys.stderr)

        # Step 5: Compare
        comparison = compare_issues(issues_codex, issues_claude)

        # Step 6: Verdict
        verdict = compute_verdict(comparison, issues_codex, issues_claude)
        print(f"  Verdict: {verdict}", file=sys.stderr)

        # Step 7: Format report
        report = format_report(
            name, candidate_metric,
            raw_codex, raw_claude,
            issues_codex, issues_claude,
            comparison, verdict,
        )

        results.append({
            "name": name,
            "verdict": verdict,
            "report": report,
            "comparison": comparison,
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation orchestrator for dual-agent review",
    )
    parser.add_argument(
        "--worktree", required=True,
        help="Path to worktree with experiment results",
    )
    parser.add_argument(
        "--baseline-sha", required=True,
        help="Baseline commit SHA (before variations)",
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Comma-separated candidate list: 'var-001:sha1,var-003:sha2'",
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to experiment manifest YAML file",
    )
    parser.add_argument(
        "--baseline-metric", required=True, type=float,
        help="Baseline metric value for context in review prompt",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be reviewed without calling agents",
    )
    args = parser.parse_args()

    # Validate worktree exists
    if not os.path.isdir(args.worktree):
        print(f"Error: worktree path does not exist: {args.worktree}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    candidates = parse_candidates(args.candidates)

    if not candidates:
        print("Error: no candidates provided", file=sys.stderr)
        sys.exit(1)

    results = run_cross_validation(
        worktree_path=args.worktree,
        baseline_sha=args.baseline_sha,
        candidates=candidates,
        manifest=manifest,
        baseline_metric=args.baseline_metric,
        dry_run=args.dry_run,
    )

    # Print all reports to stdout
    print()
    for r in results:
        print(r["report"])
        print()

    # Summary
    verdicts = [r["verdict"] for r in results]
    n_approve = verdicts.count("APPROVE")
    n_attention = verdicts.count("NEEDS_ATTENTION")
    n_reject = verdicts.count("REJECT")

    print("=" * 60)
    print(f"  SUMMARY: {len(results)} candidate(s) reviewed")
    print(f"  APPROVE: {n_approve}  |  NEEDS_ATTENTION: {n_attention}  |  REJECT: {n_reject}")
    print("=" * 60)

    # Exit non-zero if any rejections
    if n_reject > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
