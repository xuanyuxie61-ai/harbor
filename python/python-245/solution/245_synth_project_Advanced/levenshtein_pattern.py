"""
levenshtein_pattern.py
========================
Levenshtein edit-distance based classification of fission fragment
yield patterns.

Maps from: 669_levenshtein_matrix (dynamic programming edit distance)

Physical context
----------------
Different fissioning systems produce characteristic fragment mass yield
patterns (yield curves).  We can classify unknown fissioning systems by
comparing their yield patterns to known reference patterns using the
Levenshtein edit distance.

The yield curve is discretised into a string of symbols:
  "H" = high yield (above threshold)
  "M" = medium yield
  "L" = low yield

The Levenshtein distance between two yield-pattern strings measures
how many insertions, deletions, or substitutions are needed to
transform one pattern into another.

The DP matrix computation (from 669_levenshtein_matrix):
  d[i+1, j+1] = min(d[i, j+1] + 1, d[i+1, j] + 1, d[i, j] + cost)
  where cost = 0 if s[i] == t[j], else 1.
"""

import math
from typing import List, Dict, Tuple, Optional


def levenshtein_distance(s: str, t: str) -> int:
    """
    Compute the Levenshtein edit distance between strings s and t.

    Uses the standard DP algorithm from 669_levenshtein_matrix:
      d[i+1, j+1] = min(d[i, j+1]+1, d[i+1, j]+1, d[i, j]+cost)
      where cost = 0 if s[i]==t[j], else 1.

    Returns the edit distance d[m+1, n+1].
    """
    m = len(s)
    n = len(t)

    # DP matrix (m+1) x (n+1)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialisation
    for i in range(m + 1):
        d[i][0] = i  # deletions
    for j in range(n + 1):
        d[0][j] = j  # insertions

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost  # substitution
            )

    return d[m][n]


def levenshtein_matrix(s: str, t: str) -> List[List[int]]:
    """
    Return the full Levenshtein DP matrix (from 669_levenshtein_matrix).
    """
    m = len(s)
    n = len(t)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )

    return d


def normalised_levenshtein(s: str, t: str) -> float:
    """
    Normalised edit distance: d_norm = d / max(|s|, |t|)
    Returns value in [0, 1].
    """
    d = levenshtein_distance(s, t)
    max_len = max(len(s), len(t))
    if max_len == 0:
        return 0.0
    return d / max_len


def yield_to_pattern(yields: List[float], n_levels: int = 3) -> str:
    """
    Convert a yield curve to a string pattern.

    Discretise yields into n_levels levels:
      Level 0 (L): yield < percentile 33
      Level 1 (M): 33 <= yield < percentile 67
      Level 2 (H): yield >= percentile 67

    Returns string of 'L', 'M', 'H' characters.
    """
    if not yields:
        return ""

    sorted_y = sorted(yields)
    n = len(sorted_y)
    p33 = sorted_y[n // 3] if n >= 3 else sorted_y[0]
    p67 = sorted_y[2 * n // 3] if n >= 3 else sorted_y[-1]

    symbols = {0: 'L', 1: 'M', 2: 'H'}
    pattern = []
    for y in yields:
        if y < p33:
            pattern.append('L')
        elif y < p67:
            pattern.append('M')
        else:
            pattern.append('H')

    return ''.join(pattern)


def compress_pattern(pattern: str, block_size: int = 3) -> str:
    """
    Compress yield pattern by taking the most common symbol in each block.
    This reduces sensitivity to local fluctuations.
    """
    compressed = []
    for i in range(0, len(pattern), block_size):
        block = pattern[i:i + block_size]
        if not block:
            break
        # Most common character
        counts = {}
        for c in block:
            counts[c] = counts.get(c, 0) + 1
        most_common = max(counts, key=counts.get)
        compressed.append(most_common)

    return ''.join(compressed)


class FissionPatternClassifier:
    """
    Classify fission systems by comparing their yield patterns
    using Levenshtein distance.
    """

    def __init__(self):
        self.reference_patterns: Dict[str, str] = {}
        self.reference_systems: Dict[str, Dict] = {}

    def add_reference(self, system_name: str, yields: List[float],
                      metadata: Optional[Dict] = None) -> None:
        """
        Add a reference fission system with its yield pattern.
        """
        pattern = yield_to_pattern(yields)
        compressed = compress_pattern(pattern)

        self.reference_patterns[system_name] = compressed
        self.reference_systems[system_name] = {
            'yields': yields,
            'pattern': pattern,
            'compressed': compressed,
            'metadata': metadata or {},
        }

    def classify(self, yields: List[float],
                 top_k: int = 3) -> List[Dict[str, any]]:
        """
        Classify an unknown yield pattern against reference patterns.

        Returns top_k closest matches with distances.
        """
        query_pattern = compress_pattern(yield_to_pattern(yields))

        distances = []
        for name, ref_pattern in self.reference_patterns.items():
            d = levenshtein_distance(query_pattern, ref_pattern)
            d_norm = normalised_levenshtein(query_pattern, ref_pattern)
            distances.append({
                'system': name,
                'edit_distance': d,
                'normalised_distance': d_norm,
                'query_pattern': query_pattern,
                'reference_pattern': ref_pattern,
            })

        distances.sort(key=lambda x: x['edit_distance'])
        return distances[:top_k]

    def build_distance_matrix(self) -> List[List[float]]:
        """
        Compute pairwise distance matrix between all reference systems.
        """
        names = list(self.reference_patterns.keys())
        n = len(names)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                d = normalised_levenshtein(
                    self.reference_patterns[names[i]],
                    self.reference_patterns[names[j]]
                )
                matrix[i][j] = d
                matrix[j][i] = d

        return matrix


def generate_reference_patterns() -> Dict[str, List[float]]:
    """
    Generate reference yield patterns for common fissioning systems.

    Based on experimental systematics (ENDF/B-VIII.0):
    - U-235(n_th, f): asymmetric, peaks at A~95 and A~140
    - U-238(n, f) at 14 MeV: more symmetric component
    - Cf-252(sf): asymmetric, peaks at A~106 and A~142
    - Pu-239(n_th, f): similar to U-235 but shifted
    """
    from fragment_mass_distribution import FragmentMassDistribution

    patterns = {}

    # U-235 thermal fission
    fmd_u235 = FragmentMassDistribution(92, 236, 1.5)
    masses_u, yields_u = fmd_u235.compute_yield_curve(70, 170)
    patterns['U235_nth'] = yields_u

    # U-238 fast fission (higher T -> more symmetric)
    fmd_u238 = FragmentMassDistribution(92, 239, 2.5)
    masses_u8, yields_u8 = fmd_u238.compute_yield_curve(70, 170)
    patterns['U238_fast'] = yields_u8

    # Cf-252 spontaneous fission
    fmd_cf = FragmentMassDistribution(98, 252, 1.0)
    masses_cf, yields_cf = fmd_cf.compute_yield_curve(70, 170)
    patterns['Cf252_sf'] = yields_cf

    # Pu-239 thermal fission
    fmd_pu = FragmentMassDistribution(94, 240, 1.5)
    masses_pu, yields_pu = fmd_pu.compute_yield_curve(70, 170)
    patterns['Pu239_nth'] = yields_pu

    return patterns
