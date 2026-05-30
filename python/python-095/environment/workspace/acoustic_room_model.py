
import numpy as np
from collections import deque


class AcousticRoomFEM:

    def __init__(self, nodes, elements):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.Nn = self.nodes.shape[0]
        self.Ne = self.elements.shape[0]
        self.perm = np.arange(self.Nn)
        self.perm_inv = np.arange(self.Nn)

    def build_adjacency(self):
        adj = [set() for _ in range(self.Nn)]
        for elem in self.elements:
            for i in range(4):
                for j in range(i + 1, 4):
                    n1 = elem[i]
                    n2 = elem[j]
                    adj[n1].add(n2)
                    adj[n2].add(n1)
        return adj

    def rcm_reorder(self):
        adj = self.build_adjacency()
        degrees = [len(adj[i]) for i in range(self.Nn)]
        visited = [False] * self.Nn
        perm = []

        while len(perm) < self.Nn:

            root = -1
            min_deg = self.Nn + 1
            for i in range(self.Nn):
                if not visited[i] and degrees[i] < min_deg:
                    min_deg = degrees[i]
                    root = i

            if root < 0:
                break


            level = [root]
            visited[root] = True
            while level:

                level.sort(key=lambda x: degrees[x])
                next_level = []
                for node in level:
                    perm.append(node)
                    for nbr in sorted(adj[node], key=lambda x: degrees[x]):
                        if not visited[nbr]:
                            visited[nbr] = True
                            next_level.append(nbr)
                level = next_level


        perm = perm[::-1]
        self.perm = np.array(perm, dtype=int)
        self.perm_inv = np.zeros(self.Nn, dtype=int)
        for new_idx, old_idx in enumerate(self.perm):
            self.perm_inv[old_idx] = new_idx


        self.nodes = self.nodes[self.perm, :]
        for e in range(self.Ne):
            for i in range(4):
                self.elements[e, i] = self.perm_inv[self.elements[e, i]]

        return self.perm, self.perm_inv

    def compute_bandwidth(self):
        adj = self.build_adjacency()
        max_bw = 0
        for i in range(self.Nn):
            for j in adj[i]:
                bw = abs(i - j)
                if bw > max_bw:
                    max_bw = bw
        return max_bw

    def assemble_system(self, k):
        A = np.zeros((self.Nn, self.Nn), dtype=float)
        b = np.zeros(self.Nn, dtype=float)

        for elem in self.elements:
            idx = elem
            coords = self.nodes[idx, :]

            for i in range(4):
                for j in range(4):
                    if i == j:
                        A[idx[i], idx[j]] += 1.0 - (k ** 2) * 0.025
                    else:
                        A[idx[i], idx[j]] += -0.25

        return A, b


def generate_box_mesh(Lx, Ly, Lz, nx, ny, nz):

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    nodes = []
    node_idx = {}
    idx = 0
    for k_ in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k_]])
                node_idx[(i, j, k_)] = idx
                idx += 1

    nodes = np.array(nodes, dtype=float)


    rng = np.random.default_rng(7)
    shuffle_idx = rng.permutation(nodes.shape[0])
    inv_shuffle = np.zeros_like(shuffle_idx)
    inv_shuffle[shuffle_idx] = np.arange(len(shuffle_idx))
    nodes = nodes[shuffle_idx, :]

    new_node_idx = {}
    for (i, j, k_), old in node_idx.items():
        new_node_idx[(i, j, k_)] = inv_shuffle[old]
    node_idx = new_node_idx


    elements = []
    for k_ in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n000 = node_idx[(i, j, k_)]
                n100 = node_idx[(i + 1, j, k_)]
                n010 = node_idx[(i, j + 1, k_)]
                n110 = node_idx[(i + 1, j + 1, k_)]
                n001 = node_idx[(i, j, k_ + 1)]
                n101 = node_idx[(i + 1, j, k_ + 1)]
                n011 = node_idx[(i, j + 1, k_ + 1)]
                n111 = node_idx[(i + 1, j + 1, k_ + 1)]


                tets = [
                    [n000, n100, n110, n111],
                    [n000, n100, n111, n101],
                    [n000, n101, n111, n001],
                    [n000, n111, n011, n001],
                    [n000, n110, n111, n011],
                    [n000, n010, n110, n011],
                ]
                elements.extend(tets)

    elements = np.array(elements, dtype=int)
    return nodes, elements
