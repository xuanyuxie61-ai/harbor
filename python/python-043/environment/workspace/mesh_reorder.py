
import numpy as np
from typing import Tuple, List


class MeshReorderRCM:

    def __init__(self, nr: int, ntheta: int):
        self.nr = nr
        self.ntheta = ntheta
        self.n_total = nr * ntheta

    def _idx(self, i: int, j: int) -> int:
        return i * self.ntheta + j

    def _ij(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.ntheta)

    def build_adjacency(self) -> Tuple[np.ndarray, np.ndarray]:
        adj_lists = [[] for _ in range(self.n_total)]

        for i in range(self.nr):
            for j in range(self.ntheta):
                idx = self._idx(i, j)
                neighbors = []

                if i > 0:
                    neighbors.append(self._idx(i - 1, j))
                if i < self.nr - 1:
                    neighbors.append(self._idx(i + 1, j))

                if j > 0:
                    neighbors.append(self._idx(i, j - 1))
                if j < self.ntheta - 1:
                    neighbors.append(self._idx(i, j + 1))

                adj_lists[idx] = sorted(neighbors)


        adj_row = np.zeros(self.n_total + 1, dtype=int)
        for idx in range(self.n_total):
            adj_row[idx + 1] = adj_row[idx] + len(adj_lists[idx])

        adj = np.zeros(adj_row[-1], dtype=int)
        pos = 0
        for idx in range(self.n_total):
            for neighbor in adj_lists[idx]:
                adj[pos] = neighbor
                pos += 1

        return adj_row, adj

    def compute_bandwidth(self, adj_row: np.ndarray, adj: np.ndarray, perm: np.ndarray = None) -> int:
        if perm is None:
            perm = np.arange(self.n_total)
        perm_inv = np.zeros(self.n_total, dtype=int)
        perm_inv[perm] = np.arange(self.n_total)

        band_lo = 0
        band_hi = 0
        for i in range(self.n_total):
            pi = perm[i]
            for j in range(adj_row[pi], adj_row[pi + 1]):
                col = perm_inv[adj[j]]
                band_lo = max(band_lo, i - col)
                band_hi = max(band_hi, col - i)
        return band_lo + 1 + band_hi

    def root_find(
        self,
        root: int,
        adj_row: np.ndarray,
        adj: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        node_num = self.n_total
        level = np.zeros(node_num, dtype=int)
        level_row = np.zeros(node_num + 1, dtype=int)

        while True:

            ls = np.zeros(node_num, dtype=int)
            ls[0] = root
            mask_copy = mask.copy()
            mask_copy[root] = 0
            iccsze = 1
            lvlend = 0
            level_num = 0

            while True:
                lbegin = lvlend + 1
                lvlend = iccsze
                level_row[level_num] = lbegin
                level_num += 1
                for k in range(lbegin - 1, lvlend):
                    node = ls[k]
                    for j in range(adj_row[node], adj_row[node + 1]):
                        nbr = adj[j]
                        if mask_copy[nbr] != 0:
                            mask_copy[nbr] = 0
                            ls[iccsze] = nbr
                            iccsze += 1
                if iccsze == lvlend:
                    break

            level_row[level_num] = iccsze + 1


            min_deg = node_num + 1
            new_root = root
            for k in range(level_row[level_num - 1] - 1, iccsze):
                node = ls[k]
                deg = adj_row[node + 1] - adj_row[node]
                if deg < min_deg:
                    min_deg = deg
                    new_root = node

            if new_root == root:
                return root, level_num, level_row[: level_num + 1], ls[:iccsze]
            root = new_root

    def rcm(
        self,
        root: int,
        adj_row: np.ndarray,
        adj: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        node_num = self.n_total
        deg = np.zeros(node_num, dtype=int)
        level = np.zeros(node_num, dtype=int)
        ls = np.zeros(node_num, dtype=int)

        ls[0] = root
        adj_row[root] = -adj_row[root]
        lvlend = 0
        iccsze = 1

        while True:
            lbegin = lvlend + 1
            lvlend = iccsze
            for k in range(lbegin - 1, lvlend):
                node = ls[k]
                jstrt = -adj_row[node]
                jstop = abs(adj_row[node + 1]) - 1
                ideg = 0
                for j in range(jstrt, jstop + 1):
                    nbr = adj[j]
                    if mask[nbr] != 0:
                        ideg += 1
                        if 0 <= adj_row[nbr]:
                            adj_row[nbr] = -adj_row[nbr]
                            ls[iccsze] = nbr
                            iccsze += 1
                deg[node] = ideg

            lvsize = iccsze - lvlend
            if lvsize == 0:
                break


        for k in range(iccsze):
            node = ls[k]
            adj_row[node] = -adj_row[node]


        mask_out = mask.copy()
        mask_out[ls[:iccsze]] = 0



        level[:iccsze] = ls[:iccsze]

        order = np.argsort(deg[level[:iccsze]])
        level[:iccsze] = level[:iccsze][order]

        return mask_out, level, iccsze

    def genrcm(self, adj_row: np.ndarray, adj: np.ndarray) -> np.ndarray:
        node_num = self.n_total
        mask = np.ones(node_num, dtype=int)
        perm = np.zeros(node_num, dtype=int)
        num = 0

        adj_row_copy = adj_row.copy()

        for i in range(node_num):
            if mask[i] != 0:
                root = i
                root, level_num, level_row, level_arr = self.root_find(
                    root, adj_row_copy, adj, mask
                )
                mask, level, iccsze = self.rcm(root, adj_row_copy, adj, mask)
                perm[num : num + iccsze] = level[:iccsze]
                num += iccsze
                if num >= node_num:
                    break


        perm = perm[::-1]
        return perm

    def reorder(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        adj_row, adj = self.build_adjacency()
        bw_before = self.compute_bandwidth(adj_row, adj)

        perm = self.genrcm(adj_row, adj)
        perm_inv = np.zeros(self.n_total, dtype=int)
        perm_inv[perm] = np.arange(self.n_total)

        bw_after = self.compute_bandwidth(adj_row, adj, perm)
        return perm, perm_inv, bw_before, bw_after
