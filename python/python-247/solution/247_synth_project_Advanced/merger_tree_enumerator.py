"""
merger_tree_enumerator.py
=========================
Enumeration, simplification and resilience analysis of dark matter
halo merger trees.

Merger trees as balanced parenthesis sequences
----------------------------------------------
A merger tree with N branching events can be encoded as a balanced
parenthesis (Dyck / ballot) sequence of length 2N.  The number of
such trees is the Catalan number C_N = binom(2N, N)/(N+1).  We use
the combinatorial tools of Kreher & Simpson (1998):
  * bal_seq_enum   : count trees of size N,
  * bal_seq_unrank : recover the k-th tree in lexicographic order,
  * bal_seq_random : sample a tree uniformly at random.

Topology simplification (yogurt-ybn strategy 1)
-----------------------------------------------
A full merger tree produced by e.g. Rockstar/Consistent-Trees contains
many low-mass branches that do not contribute significantly to the
main progenitor line.  We simplify the tree by removing nodes whose
progenitor mass ratio falls below a threshold eta_min, following the
"strategy1_simplify" approach.

Resilience analysis (yogurt-ybn strategy 2)
-------------------------------------------
We measure the resilience of the main branch against random removal
of progenitor subhaloes.  The resilience index R is defined as the
fraction of total accreted mass that remains on the main branch after
removing a fraction p of the nodes.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple

import numpy as np


# ---------- Catalan / balanced sequences --------------------------------------

def bal_seq_enum(n: int) -> int:
    """Number of balanced parenthesis sequences of length 2n: Catalan C_n."""
    if n < 0:
        return 0
    return math.comb(2 * n, n) // (n + 1)


def bal_seq_unrank(rank: int, n: int) -> List[int]:
    """Unrank the balanced sequence of length 2n with the given
    1-based lexicographic rank.  Returns a list of 0/1 values
    (0 = '(', 1 = ')')."""
    if rank < 1 or rank > bal_seq_enum(n):
        raise ValueError("rank out of range")
    seq = []
    x = y = 0
    remaining = rank
    for pos in range(2 * n):
        # Number of sequences starting with '(' at this position
        # is C(n - x - 1 + y, y) ... here we use the ballot-number
        # recursion directly.
        a = n - x - 1
        b = y
        if a < 0:
            ways_open = 0
        else:
            ways_open = _ballot(a, b)
        if remaining <= ways_open:
            seq.append(0)
            x += 1
        else:
            remaining -= ways_open
            seq.append(1)
            y += 1
    return seq


def _ballot(a: int, b: int) -> int:
    """Number of paths from (0,0) to (a,b) that never cross the diagonal.
    Equivalent to binom(a+b, a) - binom(a+b, a+1)."""
    if a < 0 or b < 0 or b > a + 1:
        return 0
    return math.comb(a + b, a) - math.comb(a + b, a + 1)


def bal_seq_random(n: int, rng: random.Random | None = None) -> List[int]:
    """Sample a balanced sequence of length 2n uniformly at random."""
    rng = rng or random.Random()
    total = bal_seq_enum(n)
    rank = rng.randint(1, total)
    return bal_seq_unrank(rank, n)


# ---------- Merger tree construction from Dyck word ---------------------------

class HaloNode:
    """A node in the merger tree."""
    __slots__ = ("id", "mass", "redshift", "children", "parent",
                 "is_main", "removed")

    def __init__(self, id_: int, mass: float, redshift: float):
        self.id = id_
        self.mass = mass
        self.redshift = redshift
        self.children: List[HaloNode] = []
        self.parent: HaloNode | None = None
        self.is_main = False
        self.removed = False


def dyck_to_tree(seq: List[int],
                 mass_law: str = "power_law",
                 rng: random.Random | None = None) -> HaloNode:
    """Convert a balanced parenthesis sequence to a rooted merger tree.

    Each '(' opens a new child branch; each ')' returns to the parent.
    Masses are assigned by a power-law distribution of progenitor
    mass ratios, mimicking the Press-Schechter mass function."""
    rng = rng or random.Random()
    root = HaloNode(0, mass=1.0, redshift=0.0)
    stack: List[HaloNode] = [root]
    counter = 1
    for token in seq:
        if token == 0:
            parent = stack[-1]
            q = rng.betavariate(2.0, 5.0)          # mass ratio in (0,1)
            dz = rng.expovariate(5.0)              # redshift step
            child = HaloNode(counter,
                             mass=parent.mass * q,
                             redshift=parent.redshift + dz)
            child.parent = parent
            parent.children.append(child)
            stack.append(child)
            counter += 1
        else:
            if len(stack) > 1:
                stack.pop()
    return root


# ---------- Topology simplification ------------------------------------------

def simplify_tree(root: HaloNode, eta_min: float = 0.01) -> int:
    """Remove branches whose mass ratio to their parent falls below
    eta_min.  Returns the number of removed nodes."""
    removed = 0

    def _visit(node: HaloNode):
        nonlocal removed
        keep_children = []
        for c in node.children:
            ratio = c.mass / max(node.mass, 1e-30)
            if ratio < eta_min:
                c.removed = True
                removed += 1
            else:
                keep_children.append(c)
                _visit(c)
        node.children = keep_children

    _visit(root)
    return removed


# ---------- Resilience analysis ----------------------------------------------

def resilience_index(root: HaloNode, removal_fraction: float,
                     rng: random.Random | None = None) -> float:
    """Fraction of total mass remaining on the main branch after
    removing a random subset of nodes."""
    rng = rng or random.Random()
    nodes = _flatten(root)
    k = max(1, int(removal_fraction * len(nodes)))
    victims = set(rng.sample(range(1, len(nodes)), min(k, len(nodes) - 1)))
    surviving_mass = root.mass
    for i, n in enumerate(nodes[1:], start=1):
        if i not in victims:
            surviving_mass += n.mass
    total_mass = sum(n.mass for n in nodes)
    return surviving_mass / max(total_mass, 1e-30)


def _flatten(root: HaloNode) -> List[HaloNode]:
    out: List[HaloNode] = []

    def _visit(n: HaloNode):
        out.append(n)
        for c in n.children:
            _visit(c)

    _visit(root)
    return out


# ---------- Main-branch identification ---------------------------------------

def mark_main_branch(root: HaloNode) -> None:
    """Depth-first marking of the most massive progenitor line."""

    def _visit(node: HaloNode):
        node.is_main = True
        if not node.children:
            return
        main_child = max(node.children, key=lambda c: c.mass)
        _visit(main_child)

    _visit(root)


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    rng = random.Random(42)
    seq = bal_seq_random(6, rng=rng)
    root = dyck_to_tree(seq, rng=rng)
    mark_main_branch(root)
    removed = simplify_tree(root, eta_min=0.02)
    R = resilience_index(root, removal_fraction=0.3, rng=rng)
    return dict(catalan_C6=bal_seq_enum(6),
                seq_len=len(seq),
                n_removed=removed,
                resilience=R)
