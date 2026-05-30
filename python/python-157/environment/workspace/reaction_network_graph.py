import numpy as np
from combustion_utils import check_positive, check_nonnegative


class ReactionNetwork:

    def __init__(self):
        self.node_names = []
        self.node_map = {}
        self.edge_list = []
        self.adjacency_built = False
        self.edge_pointer = None
        self.edge_data = None
        self.edge_weights = None

    def add_node(self, name):
        if name not in self.node_map:
            self.node_map[name] = len(self.node_names)
            self.node_names.append(name)
        return self.node_map[name]

    def add_edge(self, name_i, name_j, rate_coeff=1.0):
        i = self.add_node(name_i)
        j = self.add_node(name_j)
        self.edge_list.append((i, j, float(rate_coeff)))
        self.adjacency_built = False

    def build_adjacency(self):
        n_nodes = len(self.node_names)
        n_edges = len(self.edge_list)


        sorted_edges = sorted(self.edge_list, key=lambda e: e[0])

        self.edge_pointer = np.zeros(n_nodes + 1, dtype=int)
        self.edge_data = np.zeros(n_edges, dtype=int)
        self.edge_weights = np.zeros(n_edges, dtype=float)

        for idx, (i, j, w) in enumerate(sorted_edges):
            self.edge_data[idx] = j
            self.edge_weights[idx] = w


        pos = 0
        self.edge_pointer[0] = 0
        for i in range(n_nodes):
            count = sum(1 for e in sorted_edges if e[0] == i)
            pos += count
            self.edge_pointer[i + 1] = pos

        self.adjacency_built = True

    def neighbors(self, i):
        if not self.adjacency_built:
            self.build_adjacency()
        start = self.edge_pointer[i]
        end = self.edge_pointer[i + 1]
        return self.edge_data[start:end]

    def degree(self, i):
        if not self.adjacency_built:
            self.build_adjacency()
        return self.edge_pointer[i + 1] - self.edge_pointer[i]

    def bfs_shortest_path(self, start_name, target_name):
        if start_name not in self.node_map or target_name not in self.node_map:
            return None
        start = self.node_map[start_name]
        target = self.node_map[target_name]

        visited = [False] * len(self.node_names)
        parent = [-1] * len(self.node_names)
        queue = [start]
        visited[start] = True

        while queue:
            u = queue.pop(0)
            if u == target:

                path = []
                cur = target
                while cur != -1:
                    path.append(self.node_names[cur])
                    cur = parent[cur]
                return path[::-1]
            for v in self.neighbors(u):
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        return None

    def find_cycles(self, max_length=6):
        n = len(self.node_names)
        cycles = []

        def dfs(u, start, depth, path, visited_set):
            if depth > max_length:
                return
            for v in self.neighbors(u):
                if v == start and depth >= 2:
                    cycle = tuple(path + [v])

                    min_idx = cycle.index(min(cycle))
                    norm = tuple(cycle[min_idx:] + cycle[:min_idx])
                    if norm not in cycles:
                        cycles.append(norm)
                elif v not in visited_set and v > start:
                    visited_set.add(v)
                    path.append(v)
                    dfs(v, start, depth + 1, path, visited_set)
                    path.pop()
                    visited_set.remove(v)

        for i in range(n):
            dfs(i, i, 0, [i], {i})

        return cycles

    def network_statistics(self):
        if not self.adjacency_built:
            self.build_adjacency()
        n = len(self.node_names)
        if n == 0:
            return {}

        degrees = [self.degree(i) for i in range(n)]
        avg_degree = np.mean(degrees)
        density = len(self.edge_list) / (n * (n - 1.0)) if n > 1 else 0.0


        clustering = []
        for i in range(n):
            neighbors_i = set(self.neighbors(i))
            ki = len(neighbors_i)
            if ki < 2:
                clustering.append(0.0)
                continue
            edges_between = 0
            for u in neighbors_i:
                for v in neighbors_i:
                    if u < v and v in set(self.neighbors(u)):
                        edges_between += 1
            clustering.append(2.0 * edges_between / (ki * (ki - 1.0)))
        avg_clustering = np.mean(clustering)

        return {
            'n_nodes': n,
            'n_edges': len(self.edge_list),
            'avg_degree': avg_degree,
            'density': density,
            'avg_clustering': avg_clustering
        }


def build_hydrogen_oxygen_network():
    net = ReactionNetwork()


    net.add_edge("H2", "H", 1.0e-3)
    net.add_edge("O2", "O", 5.0e-4)


    net.add_edge("H", "O2", 2.0e10)
    net.add_edge("O", "H2", 5.0e4)
    net.add_edge("OH", "H2", 1.0e8)


    net.add_edge("H", "OH", 1.0e10)
    net.add_edge("O", "OH", 3.0e9)
    net.add_edge("OH", "H2O", 5.0e9)


    net.add_edge("H", "HO2", 1.0e10)
    net.add_edge("O", "HO2", 5.0e9)
    net.add_edge("OH", "HO2", 2.0e10)


    net.add_edge("HO2", "H2O2", 1.0e6)
    net.add_edge("H2O2", "OH", 2.0e7)

    net.build_adjacency()
    return net
