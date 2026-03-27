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
    python scripts/experiment-harness.py --manifest experiments/nvda-market.yaml
    python scripts/experiment-harness.py --manifest experiments/nvda-market.yaml --dry-run
    python scripts/experiment-harness.py --manifest experiments/nvda-market.yaml --max-variations 3
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
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

    # --- Issue #8: Strong manifest validation ---
    if manifest["direction"] not in ("maximize", "minimize"):
        print(f"Error: direction must be 'maximize' or 'minimize', got '{manifest['direction']}'", file=sys.stderr)
        sys.exit(1)

    if not manifest["allowed_files"]:
        print("Error: allowed_files must not be empty", file=sys.stderr)
        sys.exit(1)

    for af in manifest["allowed_files"]:
        # Reject symlinks in allowed_files paths
        full_path = os.path.join(_PROJECT_ROOT, af)
        if os.path.islink(full_path):
            print(f"Error: allowed_files entry is a symlink: {af}", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(full_path):
            print(f"Error: allowed_files entry does not exist: {af}", file=sys.stderr)
            sys.exit(1)

    timeout = manifest.get("timeout_per_run", 600)
    if timeout <= 0:
        print(f"Error: timeout_per_run must be > 0, got {timeout}", file=sys.stderr)
        sys.exit(1)

    if manifest["max_variations"] <= 0:
        print(f"Error: max_variations must be > 0, got {manifest['max_variations']}", file=sys.stderr)
        sys.exit(1)

    # Defaults
    manifest.setdefault("timeout_per_run", 600)
    manifest.setdefault("early_stop_after", 2)
    manifest.setdefault("n_dates", None)
    manifest.setdefault("date_range", None)
    manifest.setdefault("workers", 1)
    manifest.setdefault("significance_alpha", 0.05)
    manifest.setdefault("kill_gates", None)

    return manifest


# ---------------------------------------------------------------------------
# Git helpers  (Issue #6: all git ops go through _run_git, checks returncode)
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


# ---------------------------------------------------------------------------
# Git worktree helpers
# ---------------------------------------------------------------------------

def create_worktree(experiment_id: str) -> str:
    """Create an isolated git worktree. Returns the worktree path."""
    # Issue #7: Add PID to worktree path to avoid concurrent run conflicts
    worktree_path = f"/tmp/evolve-{experiment_id}-{os.getpid()}"
    branch_name = f"evolve/{experiment_id}-{os.getpid()}"

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


def cleanup_worktree(experiment_id: str, worktree_path: str, preserve_branch: bool = False):
    """Remove worktree and optionally its branch.

    If preserve_branch is True, the worktree directory is removed but the
    branch is kept so that cross-validate.py can still reference it.
    """
    branch_name = f"evolve/{experiment_id}-{os.getpid()}"
    try:
        _run_git("worktree", "remove", "--force", worktree_path)
    except RuntimeError:
        shutil.rmtree(worktree_path, ignore_errors=True)
    if preserve_branch:
        print(f"  Cleaned up worktree: {worktree_path}", file=sys.stderr)
        print(f"  Branch preserved for cross-validation: {branch_name}", file=sys.stderr)
    else:
        try:
            _run_git("branch", "-D", branch_name)
        except RuntimeError:
            pass
        print(f"  Cleaned up worktree and branch: {worktree_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Whitelist enforcer  (Issue #1: staged+unstaged+untracked, reject symlinks)
# ---------------------------------------------------------------------------

def check_whitelist(allowed_files: list[str], cwd: str) -> list[str]:
    """Return list of changed files that are NOT in the allowed list.

    Checks staged+unstaged changes (vs HEAD) and untracked files.
    Also rejects symlinks in allowed_files paths within the worktree.
    """
    violations = []

    # Check for symlinks in allowed_files within the worktree
    for af in allowed_files:
        full_path = os.path.join(cwd, af)
        if os.path.islink(full_path):
            violations.append(f"{af} (symlink)")

    # Staged + unstaged changes vs HEAD
    tracked_output = _run_git("diff", "--name-only", "HEAD", cwd=cwd)
    tracked_changed = tracked_output.split("\n") if tracked_output else []

    # Untracked files
    untracked_output = _run_git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    untracked_files = untracked_output.split("\n") if untracked_output else []

    all_changed = tracked_changed + untracked_files
    for f in all_changed:
        if f and f not in allowed_files:
            violations.append(f)

    return violations


# ---------------------------------------------------------------------------
# Baseline caching
# ---------------------------------------------------------------------------

def _manifest_hash(manifest: dict) -> str:
    """Compute a stable hash of the manifest + relevant source files."""
    h = hashlib.sha256()
    # Hash manifest content (excluding internal keys)
    clean = {k: v for k, v in sorted(manifest.items()) if not k.startswith("_")}
    h.update(json.dumps(clean, sort_keys=True, default=str).encode())
    # Hash content of allowed_files
    for af in manifest.get("allowed_files", []):
        full_path = os.path.join(_PROJECT_ROOT, af)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def _cache_path(manifest: dict) -> str:
    ticker = manifest.get("eval_args", "").split("--ticker")[-1].split()[0].strip() if "--ticker" in manifest.get("eval_args", "") else "unknown"
    n_dates = manifest.get("n_dates", "manual")
    # Include manifest hash + eval script content hash so changes to
    # evaluation logic invalidate the cache.
    h = hashlib.sha256()
    h.update(_manifest_hash(manifest).encode())
    eval_script_path = os.path.join(_PROJECT_ROOT, manifest.get("eval_script", ""))
    if os.path.exists(eval_script_path):
        with open(eval_script_path, "rb") as f:
            h.update(f.read())
    combined_hash = h.hexdigest()[:16]
    # Cache in the MAIN repo, not the worktree (worktrees are ephemeral).
    # Use git to find the main worktree root regardless of where we're called from.
    try:
        git_common = subprocess.run(
            ["git", "-C", _PROJECT_ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if git_common and os.path.isdir(git_common):
            cache_dir = os.path.join(os.path.dirname(git_common), ".cache")
        else:
            cache_dir = os.path.join(_PROJECT_ROOT, ".cache")
    except Exception:
        cache_dir = os.path.join(_PROJECT_ROOT, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"baseline-{ticker}-{n_dates}-{combined_hash}.json")


def load_cached_baseline(manifest: dict):
    """Load cached baseline result if it exists and config matches."""
    path = _cache_path(manifest)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_baseline_cache(manifest: dict, result: dict):
    """Save baseline evaluation result to cache."""
    path = _cache_path(manifest)
    try:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        print(f"  Warning: could not write baseline cache: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Backtest evaluation  (Issue #10: no shell=True, use list args)
# ---------------------------------------------------------------------------

def run_backtest(script: str, args_str: str, cwd: str, timeout: int) -> dict:
    """Run backtest-eval.py and parse output (JSON or TSV).

    Returns dict with keys: directional_accuracy, weighted_accuracy, details, raw,
    and optionally p_value, ci_lower, ci_upper, significant.
    On failure returns None.

    Note: timeout is per-DATE (each propagate() call), not for the whole run.
    The subprocess timeout is scaled by the number of dates to allow the full
    evaluation to complete. backtest-eval.py handles per-date timeouts internally
    via signal.alarm.
    """
    cmd = [sys.executable, script] + shlex.split(args_str)
    # Scale subprocess timeout by number of dates (+ buffer for startup/eval overhead)
    n_dates = 50  # default
    parts = shlex.split(args_str)
    for i, p in enumerate(parts):
        if p == "--n-dates" and i + 1 < len(parts):
            try:
                n_dates = int(parts[i + 1])
            except ValueError:
                pass
        elif p == "--dates" and i + 1 < len(parts):
            n_dates = len(parts[i + 1].split(","))
    subprocess_timeout = timeout * n_dates + 300  # per-date timeout × dates + 5 min buffer
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=cwd, timeout=subprocess_timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    stdout = result.stdout.strip()
    if not stdout:
        return None

    # Try JSON first (if --output-json was used)
    if stdout.startswith("{"):
        try:
            data = json.loads(stdout)
            return {
                "directional_accuracy": data.get("accuracy", 0.0),
                "weighted_accuracy": data.get("weighted_accuracy", 0.0),
                "n_scored": data.get("n_scored", 0),
                "p_value": data.get("p_value", 1.0),
                "ci_lower": data.get("ci_lower"),
                "ci_upper": data.get("ci_upper"),
                "significant": data.get("significant", False),
                "details": " ".join(data.get("details", [])),
                "raw": stdout,
            }
        except json.JSONDecodeError:
            pass

    # Fall back to TSV parsing
    line = stdout.split("\n")[-1]
    parts = line.split("\t")
    if len(parts) < 5:
        return None

    parsed = {
        "directional_accuracy": float(parts[3]),
        "weighted_accuracy": float(parts[4]),
        "details": parts[5] if len(parts) > 5 else "",
        "raw": stdout,
    }
    # Propagate n_scored from TSV column 2 (n_total)
    try:
        parsed["n_scored"] = int(parts[2])
    except (ValueError, IndexError):
        pass
    # Parse extended TSV columns (p_value, CI, significant)
    if len(parts) >= 8:
        try:
            parsed["p_value"] = float(parts[5])
            ci_parts = parts[6].split("-")
            # Handle 'NA' values in CI (emitted when n_scored==0)
            ci_lo = ci_parts[0].strip()
            ci_hi = ci_parts[1].strip() if len(ci_parts) > 1 else ci_lo
            parsed["ci_lower"] = None if ci_lo == "NA" else float(ci_lo)
            parsed["ci_upper"] = None if ci_hi == "NA" else float(ci_hi)
            parsed["significant"] = parts[7].strip().lower() == "yes"
            parsed["details"] = parts[8] if len(parts) > 8 else ""
        except (ValueError, IndexError):
            pass
    return parsed


# Backward-compatible alias
run_eval = run_backtest


# ---------------------------------------------------------------------------
# Early-stop date helpers  (Issue #9)
# ---------------------------------------------------------------------------

def _extract_dates(eval_args: str) -> list[str]:
    """Extract the comma-separated dates list from eval_args string."""
    parts = shlex.split(eval_args)
    for i, p in enumerate(parts):
        if p == "--dates" and i + 1 < len(parts):
            return [d.strip() for d in parts[i + 1].split(",") if d.strip()]
    return []


def _resolve_dates(eval_script: str, eval_args: str, cwd: str) -> list[str]:
    """Resolve the actual date list by calling backtest-eval.py --list-dates.

    Works for both --dates and --n-dates manifests. Returns [] on failure.
    """
    cmd = [sys.executable, eval_script] + shlex.split(eval_args) + ["--list-dates"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return []


def _replace_dates(eval_args: str, dates: list[str]) -> str:
    """Return eval_args with the --dates value replaced."""
    parts = shlex.split(eval_args)
    result = []
    i = 0
    while i < len(parts):
        if parts[i] == "--dates" and i + 1 < len(parts):
            result.append("--dates")
            result.append(",".join(dates))
            i += 2
        else:
            result.append(parts[i])
            i += 1
    return " ".join(shlex.quote(p) for p in result)


def _build_dates_args(eval_args: str, dates: list[str]) -> str:
    """Build eval_args using explicit --dates, stripping --n-dates/--date-range.

    For early-stop we need to run on a specific subset of resolved dates,
    so we replace any date-generation args with a concrete --dates list.
    """
    parts = shlex.split(eval_args)
    result = []
    i = 0
    skip_next = {"--dates", "--n-dates", "--date-range"}
    while i < len(parts):
        if parts[i] in skip_next and i + 1 < len(parts):
            i += 2  # skip flag + value
        else:
            result.append(parts[i])
            i += 1
    result.extend(["--dates", ",".join(dates)])
    return " ".join(shlex.quote(p) for p in result)


# ---------------------------------------------------------------------------
# Claude Code variation generator  (Issue #10: no shell=True)
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
    for part in shlex.split(eval_args):
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
# Results logging  (Issue #10: standardized statuses)
# ---------------------------------------------------------------------------

# Valid statuses: 'baseline', 'keep', 'discard', 'crash', 'whitelist_violation'

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
    print(f"  {'Rank':<6}{'Variation':<14}{'Metric':<10}{'Status':<20}{'Description'}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)

    for i, r in enumerate(ranked, 1):
        marker = " *" if r["status"] == "keep" and r["variation"] != "baseline" else ""
        print(
            f"  {i:<6}{r['variation']:<14}{r['metric']:<10.3f}{r['status']:<20}{r['description'][:40]}{marker}",
            file=sys.stderr,
        )

    print("=" * 72, file=sys.stderr)
    print("  * = improved over baseline", file=sys.stderr)


# ---------------------------------------------------------------------------
# Kill gates
# ---------------------------------------------------------------------------

def check_kill_gates(kill_gates: dict, var_result: dict, baseline_metric: float) -> str | None:
    """Check if any kill gate is triggered. Returns reason string or None."""
    if not kill_gates:
        return None

    n_scored = var_result.get('n_scored', var_result.get('n_total', -1))
    if n_scored == 0:
        return None  # No scored data — cannot evaluate kill gates

    accuracy = var_result.get("directional_accuracy", 0.0)

    max_acc = kill_gates.get("max_accuracy_for_kill")
    if max_acc is not None and accuracy <= max_acc and baseline_metric <= max_acc:
        return f"kill_gate: both baseline ({baseline_metric:.3f}) and variation ({accuracy:.3f}) <= max_accuracy_for_kill ({max_acc})"

    min_ceiling = kill_gates.get("min_ceiling")
    if min_ceiling is not None:
        ci_upper = var_result.get("ci_upper")
        # Skip min_ceiling check when CI is unavailable (e.g. n_scored==0)
        if ci_upper is not None and ci_upper < min_ceiling:
            return f"kill_gate: CI upper bound ({ci_upper:.3f}) < min_ceiling ({min_ceiling})"

    return None


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
    significance_alpha = manifest.get("significance_alpha", 0.05)
    kill_gates = manifest.get("kill_gates")

    results = []

    print(f"\n{'#' * 60}", file=sys.stderr)
    print(f"  Experiment: {experiment_id}", file=sys.stderr)
    print(f"  Allowed files: {allowed_files}", file=sys.stderr)
    print(f"  Eval: {eval_script} {eval_args}", file=sys.stderr)
    print(f"  Metric: {metric_key} ({direction})", file=sys.stderr)
    print(f"  Variations: {n_variations} | Timeout: {timeout}s", file=sys.stderr)
    print(f"  Early stop after: {early_stop_after} dates", file=sys.stderr)
    print(f"  Significance alpha: {significance_alpha}", file=sys.stderr)
    if kill_gates:
        print(f"  Kill gates: {kill_gates}", file=sys.stderr)
    print(f"{'#' * 60}\n", file=sys.stderr)

    # Step 1: Create worktree
    worktree_path = create_worktree(experiment_id)

    try:
        # Issue #2: Capture baseline SHA for resetting between variations
        baseline_sha = _run_git("rev-parse", "HEAD", cwd=worktree_path)
        print(f"  Baseline SHA: {baseline_sha[:12]}", file=sys.stderr)

        # Step 2: Run baseline (with caching)
        cached_baseline = load_cached_baseline(manifest)
        if cached_baseline and not dry_run:
            print("  Using cached baseline evaluation.", file=sys.stderr)
            baseline_result = cached_baseline
        else:
            print("  Running baseline evaluation...", file=sys.stderr)
            if dry_run:
                baseline_result = {"directional_accuracy": 0.667, "weighted_accuracy": 0.5, "details": "[dry-run]"}
            else:
                baseline_result = run_eval(eval_script, eval_args, worktree_path, timeout)

        # Issue #5: If baseline eval fails, abort entire experiment
        if baseline_result is None:
            print("  FATAL: Baseline evaluation failed. Aborting experiment.", file=sys.stderr)
            sys.exit(1)

        # Cache the baseline result
        if not dry_run and not cached_baseline:
            save_baseline_cache(manifest, baseline_result)

        baseline_metric = baseline_result[metric_key]
        print(f"  Baseline {metric_key}: {baseline_metric:.3f}", file=sys.stderr)
        if "p_value" in baseline_result:
            print(f"  Baseline p-value: {baseline_result['p_value']:.4f}", file=sys.stderr)

        results.append({
            "variation": "baseline",
            "metric": baseline_metric,
            "status": "baseline",
            "description": "unchanged prompts",
        })

        # Resolve 'auto' kill gate: set max_accuracy_for_kill = baseline - 0.15
        if kill_gates and kill_gates.get("max_accuracy_for_kill") == "auto":
            kill_gates["max_accuracy_for_kill"] = max(baseline_metric - 0.15, 0.0)
            print(f"  Kill gate auto-resolved: max_accuracy_for_kill = {kill_gates['max_accuracy_for_kill']:.3f}", file=sys.stderr)

        # Resolve dates for early-stop (Issue #9)
        all_dates = _resolve_dates(eval_script, eval_args, worktree_path)
        if not all_dates:
            # Fallback to parsing --dates from args directly
            all_dates = _extract_dates(eval_args)
        # Early-stop threshold: baseline minus tolerance margin
        early_threshold = baseline_metric - 0.10
        print(f"  Early-stop threshold: {early_threshold:.3f} (baseline - 0.10)", file=sys.stderr)

        # Step 3: Generate and evaluate variations
        killed = False
        kill_reason = None
        for var_num in range(1, n_variations + 1):
            if killed:
                print(f"\n  KILLED: skipping remaining variations. Reason: {kill_reason}", file=sys.stderr)
                break

            var_id = f"var-{var_num:03d}"
            print(f"\n  --- Variation {var_num}/{n_variations} ({var_id}) ---", file=sys.stderr)

            # Issue #2/#3: Hard reset to baseline SHA + clean untracked files
            _run_git("reset", "--hard", baseline_sha, cwd=worktree_path)
            _run_git("clean", "-fdx", cwd=worktree_path)

            # Generate variation via Claude Code
            description = generate_variation(
                worktree_path, allowed_files, var_num - 1,
                baseline_metric, eval_args, dry_run,
            )
            if description is None:
                results.append({
                    "variation": var_id,
                    "metric": 0.0,
                    "status": "crash",
                    "description": "GENERATION_FAILED",
                })
                continue

            # Whitelist check (Issue #1)
            violations = check_whitelist(allowed_files, worktree_path)
            if violations:
                print(f"  WHITELIST_VIOLATION: {violations}", file=sys.stderr)
                # Reset to abort this variation
                _run_git("reset", "--hard", baseline_sha, cwd=worktree_path)
                _run_git("clean", "-fdx", cwd=worktree_path)
                results.append({
                    "variation": var_id,
                    "metric": 0.0,
                    "status": "whitelist_violation",
                    "description": f"WHITELIST_VIOLATION: {violations}",
                })
                continue

            # Commit the variation (Issue #6: use _run_git)
            _run_git("add", *allowed_files, cwd=worktree_path)
            # Check if there are staged changes before committing
            has_staged = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=worktree_path,
            ).returncode != 0
            if not has_staged:
                print(f"  {var_id} no changes made, skipping", file=sys.stderr)
                results.append({
                    "variation": var_id,
                    "metric": 0.0,
                    "status": "discard",
                    "description": "no changes made",
                })
                continue
            _run_git("commit", "-m", f"{var_id}: {description}", cwd=worktree_path)

            # Evaluate
            print(f"  Evaluating {var_id}...", file=sys.stderr)
            if dry_run:
                var_result = {"directional_accuracy": 0.0, "weighted_accuracy": 0.0, "details": "[dry-run]"}
                var_metric = 0.0
            else:
                # Issue #9: early_stop_after — run first N dates, bail if below threshold
                if early_stop_after and 0 < early_stop_after < len(all_dates):
                    early_dates = all_dates[:early_stop_after]
                    early_eval_args = _build_dates_args(eval_args, early_dates)
                    print(f"  Early-stop check: evaluating first {early_stop_after} dates...", file=sys.stderr)
                    early_result = run_eval(eval_script, early_eval_args, worktree_path, timeout)

                    if early_result is None:
                        print(f"  {var_id} EVAL CRASHED during early-stop check", file=sys.stderr)
                        results.append({
                            "variation": var_id,
                            "metric": 0.0,
                            "status": "crash",
                            "description": f"EVAL_CRASHED: {description}",
                        })
                        continue

                    early_metric = early_result[metric_key]
                    early_is_worse = (
                        early_metric < early_threshold if direction == "maximize"
                        else early_metric > early_threshold
                    )
                    if early_is_worse:
                        print(f"  {var_id} early-stop: {early_metric:.3f} below threshold {early_threshold:.3f}, skipping remaining dates", file=sys.stderr)
                        results.append({
                            "variation": var_id,
                            "metric": early_metric,
                            "status": "discard",
                            "description": f"early_stopped: {description}",
                        })
                        continue

                # Full evaluation
                var_result = run_eval(eval_script, eval_args, worktree_path, timeout)

                # Issue #5: Variation failures → status='crash' with error message
                if var_result is None:
                    print(f"  {var_id} EVAL CRASHED", file=sys.stderr)
                    results.append({
                        "variation": var_id,
                        "metric": 0.0,
                        "status": "crash",
                        "description": f"EVAL_CRASHED: {description}",
                    })
                    continue

                var_metric = var_result[metric_key]

            print(f"  {var_id} {metric_key}: {var_metric:.3f}", file=sys.stderr)

            # Mark as invalid if no dates were actually scored — check BEFORE
            # kill gates so zero-scored evals don't falsely trigger kills.
            n_scored = var_result.get("n_scored", var_result.get("n_total", -1))
            if not dry_run and n_scored == 0:
                results.append({
                    "variation": var_id,
                    "metric": var_metric,
                    "status": "invalid_eval",
                    "description": f"n_scored=0: {description}",
                })
                print(f"  {var_id} no dates scored (n_scored=0), marking invalid_eval", file=sys.stderr)
                continue

            # Kill gate check (after n_scored==0 invalidation)
            if not dry_run and kill_gates:
                kill_reason = check_kill_gates(kill_gates, var_result, baseline_metric)
                if kill_reason:
                    killed = True
                    print(f"  KILL GATE TRIGGERED: {kill_reason}", file=sys.stderr)
                    results.append({
                        "variation": var_id,
                        "metric": var_metric,
                        "status": "killed",
                        "description": kill_reason,
                    })
                    continue

            # Significance-based keep/discard
            is_better = (
                var_metric > baseline_metric if direction == "maximize"
                else var_metric < baseline_metric
            )
            p_value = var_result.get("p_value") if not dry_run else None
            is_significant = p_value is not None and p_value < significance_alpha

            if is_better and (is_significant or p_value is None):
                status = "keep"
            else:
                status = "discard"

            results.append({
                "variation": var_id,
                "metric": var_metric,
                "status": status,
                "description": description,
            })

            if status == "keep":
                sig_str = f", p={p_value:.4f}" if p_value is not None else ""
                print(f"  {var_id} IMPROVED over baseline ({var_metric:.3f} vs {baseline_metric:.3f}{sig_str})", file=sys.stderr)
            elif is_better and not is_significant:
                print(f"  {var_id} better but NOT significant (p={p_value:.4f} >= {significance_alpha}), discarding", file=sys.stderr)

    finally:
        # Determine if any candidates should be kept
        has_keep = any(r["status"] == "keep" for r in results if r["variation"] != "baseline")

        # Step 4a: Write results before cleanup (needed for hint)
        results_dir = os.path.join(_PROJECT_ROOT, "experiments")
        os.makedirs(results_dir, exist_ok=True)
        tsv_path = os.path.join(results_dir, f"{experiment_id}-results.tsv")
        write_results_tsv(results, tsv_path)
        print(f"\n  Results written to {tsv_path}", file=sys.stderr)

        # Step 4b: Print ranked table
        print_results_table(results)

        # Step 4c: Print cross-validate hint BEFORE cleanup so branch still exists
        branch_name = f"evolve/{experiment_id}-{os.getpid()}"
        manifest_path = manifest.get("_manifest_path", "experiments/nvda-market.yaml")
        if has_keep:
            print(f"\n  To cross-validate results:", file=sys.stderr)
            print(f"    python scripts/cross-validate.py \\", file=sys.stderr)
            print(f"      --results-tsv {tsv_path} \\", file=sys.stderr)
            print(f"      --branch {branch_name} \\", file=sys.stderr)
            print(f"      --manifest {manifest_path}", file=sys.stderr)

        # Step 4d: Clean up worktree; preserve branch if there are 'keep' candidates
        cleanup_worktree(experiment_id, worktree_path, preserve_branch=has_keep)

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
    manifest["_manifest_path"] = args.manifest
    results = run_experiment(manifest, args.max_variations, args.dry_run)

    # Exit with non-zero if no candidates found
    candidates = [r for r in results if r["status"] == "keep"]
    if not candidates:
        print("  No improvements found.", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(candidates)} candidate(s) found.", file=sys.stderr)


if __name__ == "__main__":
    main()
