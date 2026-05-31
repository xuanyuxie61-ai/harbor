"""
Reaction Network Path Optimization for Amine Degradation
Integrates Bellman-Ford shortest path algorithm from bellman_ford.m.

Amine degradation involves complex reaction networks:
- Thermal degradation: carbamate polymerization, oxazolidinone formation
- Oxidative degradation: radical chain reactions, amino acid formation
- Each path has associated energy cost (activation energy sum)

Bellman-Ford is used to find the minimum-energy degradation pathway,
which is critical for predicting solvent lifetime.
"""

import numpy as np
from utils import validate_positive


class DegradationNetwork:
    """
    Model amine degradation as a weighted directed graph.
    Nodes = chemical species, edges = degradation reactions.
    Edge weights = activation energy barriers (kJ/mol).
    """

    def __init__(self):
        self.species = []
        self.reactions = []  # list of (from_idx, to_idx, Ea, rate_constant_A)

    def add_species(self, name):
        if name not in self.species:
            self.species.append(name)
        return self.species.index(name)

    def add_reaction(self, from_species, to_species, Ea_kJ, A_factor=1e10):
        """
        Add a degradation reaction with activation energy barrier.
        Weight = Ea (lower = faster degradation path).
        """
        i = self.add_species(from_species)
        j = self.add_species(to_species)
        self.reactions.append((i, j, Ea_kJ, A_factor))

    def build_mea_degradation_network(self):
        """
        Build MEA thermal degradation network based on literature:
        Lepaumier et al. (2009), Voice (2013).
        """
        species_list = [
            "MEA", "MEACOO-", "HEIA", "HEEDA", "HEIA-OZD", "OZD",
            "N-(2-hydroxyethyl)imidazolidinone",
            " polymers", "NH3", "CO2_loss"
        ]
        for s in species_list:
            self.add_species(s)

        # Thermal degradation pathways (activation energies in kJ/mol)
        self.add_reaction("MEA", "MEACOO-", 35.0, 4.4e11)
        self.add_reaction("MEACOO-", "HEIA", 95.0, 1.0e12)  # Cyclization
        self.add_reaction("HEIA", "HEEDA", 110.0, 5.0e11)   # Ring opening + MEA
        self.add_reaction("HEIA", "HEIA-OZD", 85.0, 2.0e11)  # Oxazolidinone
        self.add_reaction("HEIA-OZD", "OZD", 75.0, 3.0e11)
        self.add_reaction("HEEDA", " polymers", 120.0, 1.0e12)
        self.add_reaction("MEACOO-", "NH3", 140.0, 8.0e11)  # Deamination
        self.add_reaction("MEA", "NH3", 150.0, 1.0e12)      # Direct decomposition
        self.add_reaction("OZD", "CO2_loss", 60.0, 5.0e10)   # CO2 release
        self.add_reaction("HEIA", "N-(2-hydroxyethyl)imidazolidinone", 90.0, 1.5e11)

    def bellman_ford_shortest_path(self, source_idx):
        """
        Bellman-Ford algorithm for shortest paths from source.
        Returns minimum activation energy paths and predecessors.
        Based on bellman_ford.m.
        """
        V = len(self.species)
        E = len(self.reactions)
        r8_big = 1.0e30

        distances = np.full(V, r8_big)
        distances[source_idx] = 0.0
        predecessors = np.full(V, -1, dtype=int)

        # Relax edges V-1 times
        for _ in range(V - 1):
            updated = False
            for j in range(E):
                u, v, weight, _ = self.reactions[j]
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                    updated = True
            if not updated:
                break

        # Check for negative cycles
        for j in range(E):
            u, v, weight, _ = self.reactions[j]
            if distances[u] + weight < distances[v] - 1e-9:
                raise ValueError("Negative cycle detected in degradation network")

        return distances, predecessors

    def reconstruct_path(self, predecessors, target_idx):
        """Reconstruct shortest path from source to target."""
        path = []
        curr = target_idx
        while curr != -1:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()
        return [self.species[i] for i in path]

    def find_minimum_energy_pathway(self, source="MEA", target="CO2_loss"):
        """Find the minimum-energy degradation pathway."""
        src_idx = self.species.index(source)
        tgt_idx = self.species.index(target)
        distances, predecessors = self.bellman_ford_shortest_path(src_idx)
        path = self.reconstruct_path(predecessors, tgt_idx)
        return {
            "path": path,
            "total_Ea": distances[tgt_idx],
            "path_length": len(path) - 1
        }

    def compute_rate_along_path(self, T, source="MEA", target="CO2_loss"):
        """
        Compute effective rate constant along minimum-energy path.
        Uses the rate-limiting step approximation.
        """
        pathway = self.find_minimum_energy_pathway(source, target)
        total_Ea = pathway["total_Ea"] * 1000.0  # Convert to J/mol
        # Rate-limiting step: use highest barrier
        # For simplicity, use total path barrier
        from utils import arrhenius_rate
        k_eff = arrhenius_rate(1.0e12, total_Ea, T)
        return k_eff, pathway

    def network_statistics(self):
        """Compute network topology statistics."""
        V = len(self.species)
        E = len(self.reactions)
        # Average degree
        in_degree = np.zeros(V)
        out_degree = np.zeros(V)
        for u, v, _, _ in self.reactions:
            out_degree[u] += 1
            in_degree[v] += 1

        return {
            "num_species": V,
            "num_reactions": E,
            "avg_out_degree": np.mean(out_degree),
            "max_out_degree": np.max(out_degree),
            "source_nodes": sum(1 for d in in_degree if d == 0),
            "sink_nodes": sum(1 for d in out_degree if d == 0)
        }
