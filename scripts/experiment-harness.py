#!/usr/bin/env python3
"""Experiment harness for auto-evolve prompt optimization.

Orchestrates the experiment loop:
  1. Parse experiment manifest YAML
  2. Create git worktree for isolation
  3. Call Claude Code to generate N prompt variations
  4. For each variation: run backtest-eval.py, record result
  5. Rank results, present to human
  6. Log to results.tsv
  7. Clean up worktree

Usage:
    python scripts/experiment-harness.py --manifest experiments/nvda-fundamentals.yaml
    python scripts/experiment-harness.py --manifest experiments/nvda-fundamentals.yaml --dry-run
    python scripts/experiment-harness.py --manifest experiments/nvda-fundamentals.yaml --max-variations 3
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = [
    "id", "domain", "allowed_files", "eval_script", "eval_args",
    "metric", "direction", "max_variations",
]


def load_manifest(path: str) -> dict:
    """Load and validate an experiment manifest YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "experiment" not in raw:
        print("Error: manifest must have top-level 'experiment' key", file=sys.stderr)
        sys.exit(1)

    manifest = raw["experiment"]
    missing = [k for k in _REQUIRED_KEYS if k not in manifest]
    if missing:
        print(f"Error: manifest missing required keys: {missing}", file=sys.stderr)
        sys.exit(1)

    # Defaults
    manifest.setdefault("timeout_per_run", 600)
    manifest.setdefault("early_stop_after", 2)

    return manifest


# ---------------------------------------------------------------------------
# Git worktree helpers
# ---------------------------------------------------------------------------

def _run_git(*args, cwd=None):
    """Run a git command, return stdout. Raises on failure."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=cwd or _PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def create_worktree(experiment_id: str) -> str:
    """Create an isolated git worktree. Returns the worktree path."""
    worktree_path = f"/tmp/evolve-{experiment_id}"
    branch_name = f"evolve/{experiment_id}"

    # Clean up stale worktree if it exists
    if os.path.exists(worktree_path):
        print(f"  Removing stale worktree at {worktree_path}", file=sys.stderr)
        try:
            _run_git("worktree", "remove", "--force", worktree_path)
        except RuntimeError:
            shutil.rmtree(worktree_path, ignore_errors=True)

    # Delete stale branch if it exists
    try:
        _run_git("branch", "-D", branch_name)
    except RuntimeError:
        pass  # branch didn't exist

    _run_git("worktree", "add", worktree_path, "-b", branch_name)
    print(f"  Created worktree: {worktree_path} (branch: {branch_name})", file=sys.stderr)
    return worktree_path


def cleanup_worktree(experiment_id: str, worktree_path: str):
    """Remove worktree and its branch."""
    branch_name = f"evolve/{experiment_id}"
    try:
        _run_git("worktree", "remove", "--force", worktree_path)
    except RuntimeError:
        shutil.rmtree(worktree_path, ignore_errors=True)
    try:
        _run_git("branch", "-D", branch_name)
    except RuntimeError:
        pass
    print(f"  Cleaned up worktree: {worktree_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Whitelist enforcer
# ---------------------------------------------------------------------------

def check_whitelist(allowed_files: list[str], cwd: str) -> list[str]:
    """Return list of changed files that are NOT in the allowed list."""
    output = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True, text=True, cwd=cwd,
    ).stdout.strip()
    if not output:
        return []
    changed = output.split("\n")
    return [f for f in changed if f not in allowed_files]


# ---------------------------------------------------------------------------
# Backtest evaluation
# ---------------------------------------------------------------------------

def run_eval(eval_script: str, eval_args: str, cwd: str, timeout: int) -> dict:
    """Run backtest-eval.py and parse TSV output.

    Returns dict with keys: directional_accuracy, weighted_accuracy, details, raw.
    """
    cmd = f"python {eval_script} {eval_args}"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd, timeout=timeout,
    )

    if result.returncode != 0:
        return {
            "directional_accuracy": 0.0,
            "weighted_accuracy": 0.0,
            "details": f"EVAL_FAILED: {result.stderr.strip()[:200]}",
            "raw": result.stderr.strip(),
        }

    # Parse TSV: ticker  n_dates  n_scored  accuracy  weighted_accuracy  details
    line = result.stdout.strip().split("\n")[-1]  # last line is the TSV
    parts = line.split("\t")
    if len(parts) < 5:
        return {
            "directional_accuracy": 0.0,
            "weighted_accuracy": 0.0,
            "details": f"PARSE_ERROR: {line[:200]}",
            "raw": result.stdout.strip(),
        }

    return {
        "directional_accuracy": float(parts[3]),
        "weighted_accuracy": float(parts[4]),
        "details": parts[5] if len(parts) > 5 else "",
        "raw": result.stdout.strip(),
    }


# ---------------------------------------------------------------------------
# Claude Code variation generator
# ---------------------------------------------------------------------------

_VARIATION_HINTS = [
    "Add structured reasoning steps (chain-of-thought) to improve analytical depth",
    "Emphasize quantitative metrics and ratio analysis over qualitative descriptions",
    "Add explicit risk assessment framework with bull/bear case analysis",
    "Improve signal clarity by requiring explicit confidence levels and price targets",
    "Add comparative analysis instructions (sector peers, historical benchmarks)",
    "Focus on cash flow quality and earnings sustainability indicators",
    "Add momentum and trend analysis to fundamental evaluation",
    "Require explicit identification of key assumptions and potential catalysts",
    "Emphasize margin of safety calculations and valuation anchoring",
    "Add contrarian analysis — require consideration of consensus view and potential surprises",
]


def generate_variation(
    worktree_path: str,
    allowed_files: list[str],
    variation_num: int,
    baseline_metric: float,
    eval_args: str,
    dry_run: bool = False,
) -> str | None:
    """Call Claude Code to generate a single prompt variation.

    Returns a brief description of the change, or None on failure.
    """
    hint = _VARIATION_HINTS[variation_num % len(_VARIATION_HINTS)]

    # Read current prompt content from the first allowed file
    target_file = allowed_files[0]
    target_path = os.path.join(worktree_path, target_file)
    try:
        with open(target_path) as f:
            current_content = f.read()
    except FileNotFoundError:
        print(f"  Warning: {target_file} not found in worktree", file=sys.stderr)
        return None

    # Extract dates from eval_args for context
    dates = ""
    for part in eval_args.split():
        if "," in part or (len(part) == 10 and "-" in part):
            dates = part
            break

    prompt = textwrap.dedent(f"""\
        You are optimizing the system_message prompt in {target_file} for a trading analyst agent.

        Current file content:
        ```python
        {current_content}
        ```

        Current metric: {baseline_metric:.3f} directional accuracy on dates {dates}

        Generate a SINGLE improved variation. Focus on: {hint}

        Rules:
        - Only modify the system_message string
        - Keep the function signature and tool bindings unchanged
        - Do not hardcode specific dates, prices, or ticker symbols
        - Focus on improving the analyst's reasoning quality
        - Do not add any imports or change the function structure

        Make the change directly to the file {target_file}.
    """)

    if dry_run:
        print(f"  [DRY RUN] Would call Claude Code with hint: {hint}", file=sys.stderr)
        return f"[dry-run] {hint}"

    result = subprocess.run(
        [
            "claude",
            "--permission-mode", "bypassPermissions",
            "--print",
            "-p", prompt,
        ],
        capture_output=True, text=True, cwd=worktree_path, timeout=120,
    )

    if result.returncode != 0:
        print(f"  Claude Code failed (rc={result.returncode}): {result.stderr[:200]}", file=sys.stderr)
        return None

    return hint[:80]


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def write_results_tsv(results: list[dict], output_path: str):
    """Write results to a TSV file."""
    with open(output_path, "w") as f:
        f.write("variation\tmetric\tstatus\tdescription\n")
        for r in results:
            f.write(f"{r['variation']}\t{r['metric']:.3f}\t{r['status']}\t{r['description']}\n")


def print_results_table(results: list[dict]):
    """Print a ranked results table to stderr."""
    # Sort by metric descending
    ranked = sorted(results, key=lambda r: r["metric"], reverse=True)

    print("\n" + "=" * 72, file=sys.stderr)
    print("  EXPERIMENT RESULTS (ranked by metric)", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(f"  {'Rank':<6}{'Variation':<14}{'Metric':<10}{'Status':<10}{'Description'}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)

    for i, r in enumerate(ranked, 1):
        marker = " *" if r["status"] == "candidate" else ""
        print(
            f"  {i:<6}{r['variation']:<14}{r['metric']:<10.3f}{r['status']:<10}{r['description'][:40]}{marker}",
            file=sys.stderr,
        )

    print("=" * 72, file=sys.stderr)
    print("  * = improved over baseline", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(manifest: dict, max_variations: int | None = None, dry_run: bool = False):
    """Run the full experiment loop."""
    experiment_id = manifest["id"]
    allowed_files = manifest["allowed_files"]
    eval_script = manifest["eval_script"]
    eval_args = manifest["eval_args"]
    metric_key = manifest["metric"]
    direction = manifest.get("direction", "maximize")
    n_variations = max_variations or manifest["max_variations"]
    timeout = manifest["timeout_per_run"]
    early_stop_after = manifest["early_stop_after"]

    results = []

    print(f"\n{'#' * 60}", file=sys.stderr)
    print(f"  Experiment: {experiment_id}", file=sys.stderr)
    print(f"  Allowed files: {allowed_files}", file=sys.stderr)
    print(f"  Eval: {eval_script} {eval_args}", file=sys.stderr)
    print(f"  Metric: {metric_key} ({direction})", file=sys.stderr)
    print(f"  Variations: {n_variations} | Timeout: {timeout}s", file=sys.stderr)
    print(f"{'#' * 60}\n", file=sys.stderr)

    # Step 1: Create worktree
    worktree_path = create_worktree(experiment_id)

    try:
        # Step 2: Run baseline
        print("  Running baseline evaluation...", file=sys.stderr)
        if dry_run:
            baseline_result = {"directional_accuracy": 0.667, "weighted_accuracy": 0.5, "details": "[dry-run]"}
        else:
            baseline_result = run_eval(eval_script, eval_args, worktree_path, timeout)

        baseline_metric = baseline_result[metric_key]
        print(f"  Baseline {metric_key}: {baseline_metric:.3f}", file=sys.stderr)

        results.append({
            "variation": "baseline",
            "metric": baseline_metric,
            "status": "keep",
            "description": "unchanged prompts",
        })

        # Step 3: Generate and evaluate variations
        for var_num in range(1, n_variations + 1):
            var_id = f"var-{var_num:03d}"
            print(f"\n  --- Variation {var_num}/{n_variations} ({var_id}) ---", file=sys.stderr)

            # Reset worktree to clean state
            subprocess.run(
                ["git", "checkout", "--", "."],
                capture_output=True, cwd=worktree_path,
            )

            # Generate variation via Claude Code
            description = generate_variation(
                worktree_path, allowed_files, var_num - 1,
                baseline_metric, eval_args, dry_run,
            )
            if description is None:
                results.append({
                    "variation": var_id,
                    "metric": 0.0,
                    "status": "discard",
                    "description": "GENERATION_FAILED",
                })
                continue

            # Whitelist check
            violations = check_whitelist(allowed_files, worktree_path)
            if violations:
                print(f"  WHITELIST_VIOLATION: {violations}", file=sys.stderr)
                subprocess.run(
                    ["git", "checkout", "--", "."],
                    capture_output=True, cwd=worktree_path,
                )
                results.append({
                    "variation": var_id,
                    "metric": 0.0,
                    "status": "discard",
                    "description": f"WHITELIST_VIOLATION: {violations}",
                })
                continue

            # Commit the variation
            subprocess.run(
                ["git", "add"] + allowed_files,
                capture_output=True, cwd=worktree_path,
            )
            subprocess.run(
                ["git", "commit", "-m", f"{var_id}: {description}"],
                capture_output=True, cwd=worktree_path,
            )

            # Evaluate
            print(f"  Evaluating {var_id}...", file=sys.stderr)
            if dry_run:
                var_result = {"directional_accuracy": 0.0, "weighted_accuracy": 0.0, "details": "[dry-run]"}
            else:
                var_result = run_eval(eval_script, eval_args, worktree_path, timeout)

            var_metric = var_result[metric_key]
            print(f"  {var_id} {metric_key}: {var_metric:.3f}", file=sys.stderr)

            # Determine status
            is_better = (
                var_metric > baseline_metric if direction == "maximize"
                else var_metric < baseline_metric
            )
            status = "candidate" if is_better else "discard"

            results.append({
                "variation": var_id,
                "metric": var_metric,
                "status": status,
                "description": description,
            })

            if is_better:
                print(f"  {var_id} IMPROVED over baseline ({var_metric:.3f} vs {baseline_metric:.3f})", file=sys.stderr)

    finally:
        # Step 4: Clean up worktree
        cleanup_worktree(experiment_id, worktree_path)

    # Step 5: Write results
    results_dir = os.path.join(_PROJECT_ROOT, "experiments")
    os.makedirs(results_dir, exist_ok=True)
    tsv_path = os.path.join(results_dir, f"{experiment_id}-results.tsv")
    write_results_tsv(results, tsv_path)
    print(f"\n  Results written to {tsv_path}", file=sys.stderr)

    # Step 6: Print ranked table
    print_results_table(results)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment harness for auto-evolve prompt optimization",
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to experiment manifest YAML file",
    )
    parser.add_argument(
        "--max-variations", type=int, default=None,
        help="Override max_variations from manifest",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running evals or calling Claude Code",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    results = run_experiment(manifest, args.max_variations, args.dry_run)

    # Exit with non-zero if no candidates found
    candidates = [r for r in results if r["status"] == "candidate"]
    if not candidates:
        print("  No improvements found.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(candidates)} candidate(s) found.", file=sys.stderr)


if __name__ == "__main__":
    main()
