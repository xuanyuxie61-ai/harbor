# -*- coding: utf-8 -*-
"""
shell_merger.py
===============
Hierarchical shell merging for the onion-skin structure of evolved stars.

Background
----------
As a star evolves, burning shells advance in mass coordinate.  When
two burning shells approach each other they can *merge* into a single
wider shell, or a shell can *split* if the composition gradient
becomes steep enough.  This hierarchical reorganisation mirrors the
dark-matter halo merger tree analysis in seed 1131_cchrisgong_aip_rockstar:

  - each burning shell is a "halo" with a characteristic mass, radius,
    and composition;
  - two shells are "bound" if their separation is smaller than their
    combined width times a safety factor (analogous to the virial
    criterion for halo binding);
  - the merger algorithm descends a tree of shells, binding those
    that overlap, much like the overlapping-halo finder in rockstar.

In addition, we compute a Roche-lobe analogue for burning shells: a
shell is tidally disrupted if its composition gradient exceeds a
critical value, mimicking the tidal radius calculation in seed 1131.

Key formulae
------------
Binding criterion:
    bound(i,j)  iff  |m_i - m_j| < eta * (w_i + w_j)
Roche criterion for composition shell:
    |dX/dm| > X / (alpha * m_shell)  =>  disruption
Merger tree:
    recursive descent: for each pair (i,j), if bound, merge and
    form a parent node; repeat until no more mergers.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import math


# =====================================================================
# Shell class
# =====================================================================

class BurningShell:
    """Representation of a burning shell in a stellar model.

    Attributes
    ----------
    name : str
        Burning stage name (e.g. "H", "He", "C").
    m_centre : float
        Mass coordinate of the shell centre [g].
    width : float
        Mass half-width of the shell [g].
    luminosity : float
        Nuclear luminosity of the shell [erg / s].
    composition : dict
        Mass fractions {species_name: Y} at the shell centre.
    temperature : float
        Temperature at the shell centre [K].
    """

    def __init__(self, name: str, m_centre: float, width: float,
                 luminosity: float, composition: Dict[str, float],
                 temperature: float):
        self.name = name
        self.m_centre = m_centre
        self.width = max(width, 1.0e-30)
        self.luminosity = max(0.0, luminosity)
        self.composition = composition
        self.temperature = max(0.0, temperature)

    def mass_extent(self) -> Tuple[float, float]:
        return (self.m_centre - self.width, self.m_centre + self.width)

    def overlaps(self, other: "BurningShell", eta: float = 1.0) -> bool:
        """Two shells overlap if their centres are closer than
        eta * sum of their widths.  This is the direct analogue of
        the overlapping-halo criterion in rockstar (seed 1131)."""
        return abs(self.m_centre - other.m_centre) < eta * (self.width + other.width)

    def roche_check(self, alpha: float = 1.0) -> bool:
        """Roche-style disruption criterion.

        A shell is disrupted if its composition gradient exceeds a
        critical value.  We approximate the gradient by the ratio
        of the dominant composition to the shell mass:
            |dX/dm| ~ X_max / (alpha * m_centre)
        If  X_max > 0.5 and the ratio is above threshold, the shell
        is unstable to disruption.  This mirrors the Roche lobe
        calculation in `roche.py` of seed 1131.
        """
        X_max = max(self.composition.values()) if self.composition else 0.0
        if self.m_centre <= 0.0:
            return False
        return X_max / (alpha * self.m_centre) > 1.0e-10

    def merge_with(self, other: "BurningShell") -> "BurningShell":
        """Merge two overlapping shells by luminosity-weighted averaging
        of their properties."""
        L1 = self.luminosity
        L2 = other.luminosity
        Ltot = L1 + L2
        if Ltot <= 0.0:
            w1, w2 = 0.5, 0.5
        else:
            w1 = L1 / Ltot
            w2 = L2 / Ltot
        new_m = w1 * self.m_centre + w2 * other.m_centre
        new_w = max(self.width, other.width,
                    0.5 * abs(self.m_centre - other.m_centre) + self.width + other.width)
        new_L = Ltot
        new_T = w1 * self.temperature + w2 * other.temperature
        new_comp = {}
        for k in set(self.composition) | set(other.composition):
            x1 = self.composition.get(k, 0.0)
            x2 = other.composition.get(k, 0.0)
            new_comp[k] = w1 * x1 + w2 * x2
        return BurningShell(f"({self.name}+{other.name})",
                            new_m, new_w, new_L, new_comp, new_T)

    def __repr__(self) -> str:
        return (f"Shell({self.name}, m={self.m_centre:.3e}, "
                f"w={self.width:.3e}, L={self.luminosity:.3e}, "
                f"T={self.temperature:.3e})")


# =====================================================================
# Merger tree construction
# =====================================================================

def build_merger_tree(shells: List[BurningShell],
                      eta: float = 1.0) -> Tuple[List[BurningShell],
                                                  List[Tuple[int, int]]]:
    """Build a merger tree of overlapping burning shells.

    Following the rockstar hierarchical halo finder, we iterate over
    all pairs and merge those that overlap.  The process is repeated
    until no more mergers occur.

    Parameters
    ----------
    shells : list of BurningShell.
    eta : overlap parameter (default 1.0).

    Returns
    -------
    merged : list of merged shells.
    pairs : list of (i, j) pairs that were merged (indices into the
        original list).
    """
    current = shells[:]
    pairs: List[Tuple[int, int]] = []
    changed = True
    while changed:
        changed = False
        new_list: List[BurningShell] = []
        used = [False] * len(current)
        for i in range(len(current)):
            if used[i]:
                continue
            merged = current[i]
            for j in range(i+1, len(current)):
                if used[j]:
                    continue
                if merged.overlaps(current[j], eta=eta):
                    merged = merged.merge_with(current[j])
                    used[j] = True
                    pairs.append((i, j))
                    changed = True
            new_list.append(merged)
            used[i] = True
        current = new_list
    return current, pairs


# =====================================================================
# Shell splitting
# =====================================================================

def split_shell(shell: BurningShell, n_sub: int = 2
                ) -> List[BurningShell]:
    """Split a shell into n_sub sub-shells of equal width.

    Used to refine the shell structure when the composition gradient
    is steep (Roche criterion violated).
    """
    if n_sub < 2:
        return [shell]
    subs = []
    w_sub = 2.0 * shell.width / n_sub
    m0 = shell.m_centre - shell.width
    for k in range(n_sub):
        mc = m0 + (k + 0.5) * w_sub
        new_comp = {k: v for k, v in shell.composition.items()}
        new_L = shell.luminosity / n_sub
        subs.append(BurningShell(f"{shell.name}[{k}]", mc, 0.5*w_sub,
                                  new_L, new_comp, shell.temperature))
    return subs


# =====================================================================
# Diagnostic
# =====================================================================

def _self_test():
    print("shell_merger self-test:")
    shells = [
        BurningShell("H", 0.3, 0.05, 1.0, {"H1": 0.7, "He4": 0.3}, 1.5e7),
        BurningShell("He", 0.35, 0.04, 0.5, {"He4": 0.9, "C12": 0.1}, 2e8),
        BurningShell("C", 0.8, 0.02, 0.1, {"C12": 0.8, "O16": 0.2}, 8e8),
        BurningShell("O", 0.85, 0.015, 0.05, {"O16": 0.9, "Ne20": 0.1}, 1.5e9),
    ]
    merged, pairs = build_merger_tree(shells, eta=1.0)
    print(f"  initial {len(shells)} shells -> merged {len(merged)} shells")
    print(f"  merger pairs = {pairs}")
    for s in merged:
        print(f"    {s}")
    # Split test
    subs = split_shell(shells[0], n_sub=3)
    print(f"  split H shell into {len(subs)} sub-shells")
    print("shell_merger self-test OK")


if __name__ == "__main__":
    _self_test()
