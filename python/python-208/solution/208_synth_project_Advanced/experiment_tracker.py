"""
experiment_tracker.py
=====================

Experiment-iteration tracking via sequential filename generation.
Adapted from `filename_inc` of project 429.

Purpose
-------
The multi-fidelity UQ pipeline runs many iterations of the adaptive-
sampling loop, each producing a batch of samples, a GP surrogate, and
a confidence-band analysis.  We track each iteration with a sequential
identifier encoded in a filename-like string, e.g.

    mf_run_000 -> mf_run_001 -> mf_run_002 -> ...

The `filename_inc` function increments the rightmost digit group with
carry; we generalize it to support multiple independent counters
(iteration, batch, fold) encoded in a single string.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# filename_inc (following project 429 verbatim).
# ----------------------------------------------------------------------
def filename_inc(filename: str) -> str:
    """Generate the next filename in a series by incrementing the
    rightmost digit group with carry.  Non-digit characters are
    unaffected.  If all digits are 9, they wrap to 0.
    """
    if not filename:
        raise ValueError("filename_inc: empty filename.")
    chars = list(filename)
    n = len(chars)
    change = 0
    for i in range(n - 1, -1, -1):
        c = chars[i]
        if '0' <= c <= '8':
            chars[i] = chr(ord(c) + 1)
            return "".join(chars)
        elif c == '9':
            chars[i] = '0'
            change += 1
    if change == 0:
        return ""
    return "".join(chars)


def filename_sequence(prefix: str, n: int) -> List[str]:
    """Generate a sequence of n filenames starting from `prefix`."""
    out: List[str] = [prefix]
    current = prefix
    for _ in range(n - 1):
        nxt = filename_inc(current)
        if not nxt:
            raise ValueError(
                "filename_sequence: all digits wrapped around to 0."
            )
        out.append(nxt)
        current = nxt
    return out


# ----------------------------------------------------------------------
# Experiment registry.
# ----------------------------------------------------------------------
class ExperimentTracker:
    """Track a sequence of multi-fidelity UQ experiments."""

    def __init__(self, base_name: str = "mf_run_000") -> None:
        self.base_name = base_name
        self.history: List[Dict] = []
        self.current_name = base_name

    def next(self, metadata: Optional[Dict] = None) -> str:
        """Advance to the next experiment; return its name."""
        name = self.current_name
        record = {"name": name}
        if metadata:
            record.update(metadata)
        self.history.append(record)
        nxt = filename_inc(self.current_name)
        if not nxt:
            raise RuntimeError("ExperimentTracker: counter overflow.")
        self.current_name = nxt
        return name

    def summary(self) -> str:
        n = len(self.history)
        if n == 0:
            return "ExperimentTracker: no experiments recorded."
        return (
            f"ExperimentTracker: {n} experiments, from "
            f"'{self.history[0]['name']}' to '{self.history[-1]['name']}'."
        )

    def dump(self) -> List[Dict]:
        return list(self.history)


# ----------------------------------------------------------------------
# Counter helpers.
# ----------------------------------------------------------------------
def integer_to_padded(i: int, width: int = 3) -> str:
    """Convert integer i to a zero-padded string of given width."""
    if width < 1:
        raise ValueError("integer_to_padded: width must be >= 1.")
    return str(i).zfill(width)


def padded_to_integer(s: str) -> int:
    """Convert a zero-padded string back to an integer."""
    return int(s)


# ----------------------------------------------------------------------
# Sample-log filename generator.
# ----------------------------------------------------------------------
def sample_log_filename(run_name: str, level: int) -> str:
    """Construct a sample-log filename for a given run and fidelity level."""
    return f"{run_name}_L{level}.log"


def confidence_band_filename(run_name: str, alpha: float) -> str:
    """Construct a confidence-band output filename."""
    alpha_pct = int(round(alpha * 100))
    return f"{run_name}_pi{alpha_pct:02d}.dat"
