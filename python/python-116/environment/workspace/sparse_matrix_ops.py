
import numpy as np


class SparseMatrixOps:

    @staticmethod
    def adjacency_to_csr(adj_dict, n_nodes):
        adj_row = np.zeros(n_nodes + 1, dtype=int)
        adj_col = []
        for i in range(n_nodes):
            neighbors = sorted(set(adj_dict.get(i, [])))
            adj_row[i + 1] = adj_row[i] + len(neighbors)
            adj_col.extend(neighbors)
        return adj_row, np.array(adj_col, dtype=int)

    @staticmethod
    def degree(root, adj_row, adj_col, mask, n_nodes):
        deg = np.zeros(n_nodes, dtype=int)
        iccsze = 0
        perm = np.zeros(n_nodes, dtype=int)

        if n_nodes < 1 or root < 0 or root >= n_nodes or mask[root] == 0:
            return deg, iccsze, perm


        queue = [root]
        mask_copy = mask.copy()
        mask_copy[root] = 0
        perm[0] = root
        iccsze = 1
        front = 0

        while front < len(queue):
            node = queue[front]
            front += 1
            jstrt = adj_row[node]
            jstop = adj_row[node + 1] - 1 if node + 1 < len(adj_row) else len(adj_col) - 1
            if jstop < jstrt:
                deg[node] = 0
                continue
            deg[node] = 0
            for j in range(jstrt, min(jstop + 1, len(adj_col))):
                nbr = adj_col[j]
                if 0 <= nbr < n_nodes and mask_copy[nbr] != 0:
                    mask_copy[nbr] = 0
                    queue.append(nbr)
                    perm[iccsze] = nbr
                    iccsze += 1
                    deg[node] += 1
                elif 0 <= nbr < n_nodes:
                    deg[node] += 1

        return deg, iccsze, perm

    @staticmethod
    def rcm_reorder(root, adj_row, adj_col, n_nodes):
        if n_nodes < 1:
            return np.array([], dtype=int)
        if root < 0 or root >= n_nodes:
            raise ValueError("root 超出范围。")

        mask = np.ones(n_nodes, dtype=int)
        deg, iccsze, perm = SparseMatrixOps.degree(root, adj_row, adj_col, mask, n_nodes)

        if iccsze <= 1:
            return perm[:iccsze]

        mask[root] = 0
        lvlend = 0
        lnbr = 1

        while lvlend < lnbr:
            lbegin = lvlend + 1
            lvlend = lnbr
            for i in range(lbegin - 1, lvlend):
                node = perm[i]
                jstrt = adj_row[node]
                jstop = adj_row[node + 1] - 1 if node + 1 < len(adj_row) else len(adj_col) - 1
                fnbr = lnbr + 1
                for j in range(jstrt, min(jstop + 1, len(adj_col))):
                    nbr = adj_col[j]
                    if 0 <= nbr < n_nodes and mask[nbr] != 0:
                        lnbr += 1
                        mask[nbr] = 0
                        perm[lnbr - 1] = nbr
                if lnbr <= fnbr:
                    continue

                k = fnbr - 1
                while k < lnbr - 1:
                    l = k
                    k += 1
                    nbr = perm[k]
                    while l >= fnbr - 1:
                        lperm = perm[l]
                        if deg[lperm] <= deg[nbr]:
                            break
                        perm[l + 1] = lperm
                        l -= 1
                    perm[l + 1] = nbr


        perm[:iccsze] = perm[:iccsze][::-1]
        return perm[:iccsze]

    @staticmethod
    def bandwidth(adj_dict, perm):
        bw = 0
        inv_perm = {old: new for new, old in enumerate(perm)}
        for i, neighbors in adj_dict.items():
            if i not in inv_perm:
                continue
            pi = inv_perm[i]
            for j in neighbors:
                if j not in inv_perm:
                    continue
                pj = inv_perm[j]
                bw = max(bw, abs(pi - pj))
        return bw


class MarkovStateModel:

    def __init__(self, transition_matrix):
        P = np.asarray(transition_matrix, dtype=float)
        n = P.shape[0]
        if P.shape[0] != P.shape[1]:
            raise ValueError("转移矩阵必须是方阵。")
        row_sums = P.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("转移矩阵每行和必须为正。")
        self.P = P / row_sums[:, None]
        self.n_states = n

    def power_method_steady_state(self, max_iter=200, tol=1e-10):
        x = np.ones(self.n_states) / self.n_states
        for _ in range(max_iter):
            x_new = self.P.T @ x
            diff = np.max(np.abs(x_new - x))
            x = x_new
            if diff < tol:
                break

        s = np.sum(x)
        if s > 0:
            x = x / s
        return x

    def pagerank_style_rank(self, damping=0.85, max_iter=200, tol=1e-10):
        if not (0.0 < damping < 1.0):
            raise ValueError("damping 必须在 (0,1) 内。")
        r = np.ones(self.n_states) / self.n_states
        v = np.ones(self.n_states) / self.n_states
        for _ in range(max_iter):
            r_new = (1.0 - damping) * v + damping * (self.P.T @ r)
            diff = np.max(np.abs(r_new - r))
            r = r_new
            if diff < tol:
                break
        return r

    def free_energy_landscape(self, temperature=300.0, kb=0.008314):
        pi = self.power_method_steady_state()
        pi = np.clip(pi, 1e-12, 1.0)
        F = -kb * temperature * np.log(pi)
        return F

    def implied_timescales(self, n_eigen=5):
        eigenvalues = np.linalg.eigvals(self.P)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[1:n_eigen]

        eigenvalues = np.clip(eigenvalues, 1e-12, 1.0 - 1e-12)
        timescales = -1.0 / np.log(eigenvalues)
        return timescales


def build_lipid_adjacency(nx, ny, interaction_range=1):
    adj = {}
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            neighbors = []
            for di in range(-interaction_range, interaction_range + 1):
                for dj in range(-interaction_range, interaction_range + 1):
                    if di == 0 and dj == 0:
                        continue
                    ii = (i + di) % nx
                    jj = (j + dj) % ny
                    nidx = ii * ny + jj
                    neighbors.append(nidx)
            adj[idx] = neighbors
    return adj


def build_diffusion_matrix_from_adjacency(adj_dict, n_nodes, D=1.0, dt=0.001):
    L = np.zeros((n_nodes, n_nodes))
    for i, neighbors in adj_dict.items():
        if len(neighbors) == 0:
            continue
        rate = D * len(neighbors)
        L[i, i] = -rate
        for j in neighbors:
            L[i, j] = D
    return L
