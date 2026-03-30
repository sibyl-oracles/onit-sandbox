#!/usr/bin/env python3
"""
Analyze cyclomatic complexity of Python files in the project.

Flags files with cyclomatic complexity > 20.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Dict keys for consistency
_KEY_FLAGGED = "flagged"
_KEY_ALL = "all"
_KEY_FILE = "file"
_KEY_COMPLEXITY = "complexity"

# Complexity threshold for flagging
_CC_THRESHOLD = 20


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def _check_radon_available() -> bool:
    """Check if radon is available on the system."""
    return shutil.which("radon") is not None


def analyze_complexity() -> dict[str, Any]:
    """Analyze cyclomatic complexity using radon.

    Returns a dict with file metrics.
    """
    if not _check_radon_available():
        print(
            "Error: radon command not found. Install with: pip install radon",
            file=sys.stderr,
        )
        sys.exit(1)

    root = get_project_root()
    src_dir = root / "src"
    tests_dir = root / "tests"

    if not src_dir.exists() or not tests_dir.exists():
        print("Error: src/ and tests/ directories not found", file=sys.stderr)
        sys.exit(1)

    results: dict[str, Any] = {_KEY_FLAGGED: [], _KEY_ALL: []}

    py_files = list(src_dir.rglob("*.py")) + list(tests_dir.rglob("*.py"))

    for py_file in sorted(py_files):
        try:
            py_file_str = str(py_file)
            cmd = ["radon", "cc", py_file_str, "-j"]
            output = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
            )
            data = json.loads(output.stdout)

            if py_file_str in data:
                metrics = data[py_file_str]

                # Sum complexity across all functions in file
                total_cc = 0
                for func_metrics in metrics:
                    if isinstance(func_metrics, dict) and _KEY_COMPLEXITY in func_metrics:
                        total_cc += func_metrics[_KEY_COMPLEXITY]

                is_flagged = total_cc > _CC_THRESHOLD
                file_info = {
                    _KEY_FILE: str(py_file.relative_to(root)),
                    _KEY_COMPLEXITY: total_cc,
                    _KEY_FLAGGED: is_flagged,
                }
                results[_KEY_ALL].append(file_info)

                if is_flagged:
                    results[_KEY_FLAGGED].append(file_info)

        except subprocess.TimeoutExpired:
            print(f"Warning: Timeout analyzing {py_file}", file=sys.stderr)
        except json.JSONDecodeError:
            print(
                f"Warning: Failed to parse radon output for {py_file}",
                file=sys.stderr,
            )

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print analysis results in a clear format."""
    print("=" * 70)
    print("CYCLOMATIC COMPLEXITY ANALYSIS")
    print("=" * 70)
    print()

    # Print all files
    print("ALL FILES:")
    print("-" * 70)
    for file_info in results[_KEY_ALL]:
        status = f"FLAGGED (> {_CC_THRESHOLD})" if file_info[_KEY_FLAGGED] else "OK"
        print(f"File: {file_info[_KEY_FILE]}")
        print(f"Cyclomatic Complexity: {file_info[_KEY_COMPLEXITY]}")
        print(f"Status: {status}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_files = len(results[_KEY_ALL])
    flagged_files = len(results[_KEY_FLAGGED])
    print(f"Total files analyzed: {total_files}")
    print(f"Files flagged (CC > {_CC_THRESHOLD}): {flagged_files}")

    if flagged_files > 0:
        print()
        print("Flagged files:")
        for file_info in results[_KEY_FLAGGED]:
            print(f"  - {file_info[_KEY_FILE]} (CC: {file_info[_KEY_COMPLEXITY]})")

    print()


def main() -> None:
    """Main entry point."""
    results = analyze_complexity()
    print_results(results)

    # Exit with error if any files are flagged
    if results[_KEY_FLAGGED]:
        sys.exit(1)


if __name__ == "__main__":
    main()
