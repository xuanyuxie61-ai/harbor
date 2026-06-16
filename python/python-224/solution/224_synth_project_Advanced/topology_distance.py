"""
topology_distance.py
====================
Edit distance and topological similarity between Higgs decay topologies.

Each Higgs decay channel is encoded as a string representing the
production x decay chain:
    "ggF_H_bb"        -> gluon-gluon fusion, Higgs, b-quark pair
    "VBF_H_ww_lep"    -> vector boson fusion, Higgs, W pair, leptonic
    "VH_H_zz_4l"      -> associated VH production, Higgs, ZZ, 4 leptons
    "ttH_H_gammagamma" -> ttH production, Higgs, diphoton

The Levenshtein edit distance (seed 668_levenshtein_distance) between
two such strings provides a measure of topological similarity:
    D(s1, s2) = min #insertions, deletions, substitutions to convert s1 to s2.

The normalized distance:
    d(s1, s2) = D(s1, s2) / max(|s1|, |s2|)

A similarity kernel can be built from this:
    K(s1, s2) = exp(-d(s1, s2)^2 / (2 sigma^2))

This kernel is used for topology-aware regularization in combined fits.
"""
from __future__ import annotations
import numpy as np


class TopologyDistance:
    """
    Edit distance between Higgs decay topology strings.
    (seed 668_levenshtein_distance)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #                   Classic Levenshtein (DP)                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """
        Levenshtein edit distance via dynamic programming.
        Time O(nm), space O(min(n, m)).
        (seed 668_levenshtein_distance)
        """
        if len(s1) < len(s2):
            return TopologyDistance.levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = np.arange(len(s2) + 1, dtype=int)
        for i, c1 in enumerate(s1):
            curr_row = np.zeros(len(s2) + 1, dtype=int)
            curr_row[0] = i + 1
            for j, c2 in enumerate(s2):
                # Insertion, deletion, substitution costs
                insert = prev_row[j + 1] + 1
                delete = curr_row[j] + 1
                sub = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row[j + 1] = min(insert, delete, sub)
            prev_row = curr_row
        return int(prev_row[-1])

    # ------------------------------------------------------------------ #
    #                Damerau-Levenshtein (transpositions)                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def damerau_levenshtein(s1: str, s2: str) -> int:
        """
        Damerau-Levenshtein: allows adjacent transpositions.
        Useful for channels where order of tokens is swapped.
        """
        len1, len2 = len(s1), len(s2)
        d = np.zeros((len1 + 1, len2 + 1), dtype=int)
        for i in range(len1 + 1):
            d[i, 0] = i
        for j in range(len2 + 1):
            d[0, j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i, j] = min(
                    d[i - 1, j] + 1,         # deletion
                    d[i, j - 1] + 1,         # insertion
                    d[i - 1, j - 1] + cost,  # substitution
                )
                if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    d[i, j] = min(d[i, j], d[i - 2, j - 2] + cost)
        return int(d[len1, len2])

    # ------------------------------------------------------------------ #
    #                  Normalized distance and kernel                    #
    # ------------------------------------------------------------------ #
    def normalized_distance(self, s1: str, s2: str) -> float:
        """Normalized edit distance in [0, 1]."""
        if len(s1) == 0 and len(s2) == 0:
            return 0.0
        D = self.levenshtein(s1, s2)
        return D / max(len(s1), len(s2))

    @staticmethod
    def similarity_kernel(d: float, sigma: float = 0.3) -> float:
        """Gaussian similarity kernel K = exp(-d^2 / (2 sigma^2))."""
        return float(np.exp(-d ** 2 / (2.0 * sigma ** 2)))

    # ------------------------------------------------------------------ #
    #                  Pairwise matrix                                   #
    # ------------------------------------------------------------------ #
    def pairwise_matrix(self, channels: list, sigma: float = 0.3) -> dict:
        """
        Compute pairwise distance and similarity matrices for a list of
        channel strings.
        """
        n = len(channels)
        D = np.zeros((n, n))
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                d = self.normalized_distance(channels[i], channels[j])
                D[i, j] = D[j, i] = d
                K[i, j] = K[j, i] = self.similarity_kernel(d, sigma)
        return {"distance": D, "similarity": K, "channels": channels}

    # ------------------------------------------------------------------ #
    #                  Topological complexity                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def complexity(channel: str) -> int:
        """
        Topological complexity of a channel = number of '_' separators + 1
        (number of tokens in the topology string).
        """
        return channel.count("_") + 1

    @staticmethod
    def token_set(channel: str) -> set:
        """Set of tokens in a channel string."""
        return set(channel.split("_"))

    def jaccard_similarity(self, s1: str, s2: str) -> float:
        """Jaccard index between token sets."""
        A = self.token_set(s1)
        B = self.token_set(s2)
        if not A and not B:
            return 1.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union > 0 else 0.0
