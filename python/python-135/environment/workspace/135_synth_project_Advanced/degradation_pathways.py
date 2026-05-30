
import numpy as np
from itertools import combinations


class SubsetEnumerator:

    def __init__(self, items):
        self.items = list(items)
        self.n = len(items)

    def all_subsets(self):
        subsets = []
        for mask in range(1 << self.n):
            subset = [self.items[i] for i in range(self.n) if (mask >> i) & 1]
            subsets.append(subset)
        return subsets

    def subsets_of_size(self, k):
        return [list(c) for c in combinations(self.items, k)]

    def next_subset_lex(self, current):
        s = np.array(current, dtype=int).copy()
        n = len(s)

        for i in range(n - 1, -1, -1):
            if s[i] == 0 and np.any(s[i + 1:] == 1):
                s[i] = 1
                s[i + 1:] = 0
                return s.tolist()

        if np.sum(s) < n:
            idx = np.where(s == 0)[0]
            if len(idx) > 0:
                s[idx[0]] = 1
                s[idx[0] + 1:] = 0
                return s.tolist()
        return None

    def rank_subset(self, subset):
        rank = 0
        for i, val in enumerate(subset):
            if val == 1:
                rank += 2 ** i
        return rank

    def unrank_subset(self, rank, n):
        s = []
        for i in range(n):
            s.append((rank >> i) & 1)
        return s


class DegradationMechanismAnalyzer:

    def __init__(self, base_amine="MEA"):
        self.base_amine = base_amine
        if base_amine == "MEA":
            self.products = [
                "HEIA", "HEEDA", "OZD", "HEIA-OZD",
                "N-(2-hydroxyethyl)imidazolidinone",
                "NH3", "formate", "acetate", "glycolate"
            ]
        else:
            self.products = ["product_1", "product_2", "product_3"]

    def enumerate_possible_mechanisms(self, max_products=4):
        enumerator = SubsetEnumerator(self.products)
        mechanisms = []

        for k in range(1, min(max_products + 1, len(self.products) + 1)):
            for subset in enumerator.subsets_of_size(k):


                score = self._stoichiometric_score(subset)
                mechanisms.append({
                    "products": subset,
                    "num_products": k,
                    "stoichiometric_score": score,
                    "feasible": score > 0.5
                })

        return mechanisms

    def _stoichiometric_score(self, products):

        base_atoms = {"C": 2, "H": 7, "N": 1, "O": 1}
        product_atoms = {
            "HEIA": {"C": 4, "H": 10, "N": 2, "O": 2},
            "HEEDA": {"C": 4, "H": 12, "N": 2, "O": 2},
            "OZD": {"C": 3, "H": 5, "N": 1, "O": 2},
            "HEIA-OZD": {"C": 5, "H": 9, "N": 2, "O": 3},
            "N-(2-hydroxyethyl)imidazolidinone": {"C": 5, "H": 10, "N": 2, "O": 2},
            "NH3": {"C": 0, "H": 3, "N": 1, "O": 0},
            "formate": {"C": 1, "H": 1, "N": 0, "O": 2},
            "acetate": {"C": 2, "H": 3, "N": 0, "O": 2},
            "glycolate": {"C": 2, "H": 3, "N": 0, "O": 3},
        }

        total_atoms = {"C": 0, "H": 0, "N": 0, "O": 0}
        for p in products:
            if p in product_atoms:
                for atom, count in product_atoms[p].items():
                    total_atoms[atom] += count


        score = 0.0
        for atom in ["C", "N"]:
            if total_atoms[atom] >= base_atoms[atom]:
                score += 0.25

            ratio = total_atoms[atom] / max(base_atoms[atom], 1)
            score += 0.125 * np.exp(-abs(np.log(ratio + 1e-10)))

        return np.clip(score, 0.0, 1.0)

    def partition_analysis(self, n_molecules=4):
        partitions = self._generate_partitions(n_molecules)
        analyses = []
        for p in partitions:
            analyses.append({
                "partition": p,
                "sum": sum(p),
                "length": len(p),
                "max_part": max(p) if p else 0
            })
        return analyses

    def _generate_partitions(self, n):
        result = []

        def _partition(remaining, max_val, current):
            if remaining == 0:
                result.append(current[:])
                return
            for i in range(min(max_val, remaining), 0, -1):
                current.append(i)
                _partition(remaining - i, i, current)
                current.pop()

        _partition(n, n, [])
        return result

    def count_distinct_pathways(self, max_steps=5):

        n = min(max_steps, len(self.products))
        bell = self._bell_number(n)
        return bell

    def _bell_number(self, n):
        if n <= 0:
            return 1
        bell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
        bell[0][0] = 1
        for i in range(1, n + 1):
            bell[i][0] = bell[i - 1][i - 1]
            for j in range(1, i + 1):
                bell[i][j] = bell[i][j - 1] + bell[i - 1][j - 1]
        return bell[n][0]


def knapsack_additive_selection(additives, costs, benefits, budget):
    n = len(additives)
    best_value = 0.0
    best_cost = 0.0
    best_selection = []

    for mask in range(1, 1 << n):
        total_cost = 0.0
        total_benefit = 0.0
        selection = []
        for i in range(n):
            if (mask >> i) & 1:
                total_cost += costs[i]
                total_benefit += benefits[i]
                selection.append(additives[i])

        if total_cost <= budget and total_benefit > best_value:
            best_value = total_benefit
            best_cost = total_cost
            best_selection = selection

    return {
        "additives": best_selection,
        "total_cost": best_cost,
        "total_benefit": best_value,
        "cost_efficiency": best_value / best_cost if best_cost > 0 else 0.0
    }
