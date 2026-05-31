"""
sparse_matrix_ops.py
稀疏矩阵运算与特征分析模块

本模块实现适用于脂质双分子层大规模模拟的稀疏矩阵操作，包括:
  - Reverse Cuthill-McKee (RCM) 重排序，降低稀疏矩阵带宽
  - PageRank 幂法求主导特征向量（用于 Markov 态模型）

参考种子项目: 1016_rcm (Reverse Cuthill-McKee 重排序)
                844_pagerank (PageRank 幂法)

物理背景:
    在 MD 模拟中，邻接图描述脂质分子间的相互作用网络。
    当求解 Poisson-Boltzmann 方程或扩散方程时，相应的离散化矩阵
    是大型稀疏矩阵。RCM 重排序可显著减少矩阵带宽，提高直接求解器效率。

    Markov 态模型 (MSM) 将构象空间离散化为若干态，转移概率矩阵 P
    的主导特征向量（对应特征值 1）给出稳态分布。PageRank 幂法是
    求解该特征向量的高效方法。
"""

import numpy as np


class SparseMatrixOps:
    """
    稀疏矩阵操作集合。
    """

    @staticmethod
    def adjacency_to_csr(adj_dict, n_nodes):
        """
        将邻接字典转换为 CSR 风格表示 (adj_row, adj_col)。

        Parameters
        ----------
        adj_dict : dict
            adj_dict[i] = 节点 i 的邻居索引列表。
        n_nodes : int
            节点总数。

        Returns
        -------
        adj_row : ndarray, shape (n_nodes+1,)
            行指针。
        adj_col : ndarray
            列索引。
        """
        adj_row = np.zeros(n_nodes + 1, dtype=int)
        adj_col = []
        for i in range(n_nodes):
            neighbors = sorted(set(adj_dict.get(i, [])))
            adj_row[i + 1] = adj_row[i] + len(neighbors)
            adj_col.extend(neighbors)
        return adj_row, np.array(adj_col, dtype=int)

    @staticmethod
    def degree(root, adj_row, adj_col, mask, n_nodes):
        """
        计算 mask 指定子图中各节点的度（RCM 辅助函数）。

        受种子项目 1016_rcm 的 degree.m 启发。
        """
        deg = np.zeros(n_nodes, dtype=int)
        iccsze = 0
        perm = np.zeros(n_nodes, dtype=int)

        if n_nodes < 1 or root < 0 or root >= n_nodes or mask[root] == 0:
            return deg, iccsze, perm

        # BFS 找到连通分量
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
        """
        Reverse Cuthill-McKee 重排序。

        算法步骤（George & Liu, 1981）:
          1. 从根节点 root 开始 BFS，找到连通分量。
          2. 按度递增顺序为每层邻居编号（Cuthill-McKee 序）。
          3. 将该序反转得到 RCM 序。

        Returns
        -------
        perm : ndarray
            RCM 排列: perm[new_index] = old_index。
        """
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
                # 按度递增排序新发现的邻居
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

        # 反转得到 RCM 序
        perm[:iccsze] = perm[:iccsze][::-1]
        return perm[:iccsze]

    @staticmethod
    def bandwidth(adj_dict, perm):
        """
        计算排列后的矩阵带宽。

        bandwidth = max_{i,j∈E} |perm(i) - perm(j)|
        """
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
    """
    基于转移矩阵的 Markov 态模型（MSM）。

    受种子项目 844_pagerank (power_rank) 启发，使用幂法计算稳态分布。

    转移矩阵 P 满足:
        P_{ij} = 从态 i 转移到态 j 的概率
        Σ_j P_{ij} = 1  （行随机）

    稳态分布 π 满足:
        π^T = π^T P   或   P^T π = π
    即 π 是 P^T 对应特征值 1 的左特征向量。
    """

    def __init__(self, transition_matrix):
        """
        Parameters
        ----------
        transition_matrix : ndarray, shape (n_states, n_states)
            行随机转移矩阵。
        """
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
        """
        幂法求稳态分布 π。

        迭代:
            x_{k+1} = P^T x_k
        由于 P^T 是列随机矩阵，其最大特征值为 1，
        对应特征向量即为稳态分布（归一化后）。
        """
        x = np.ones(self.n_states) / self.n_states
        for _ in range(max_iter):
            x_new = self.P.T @ x
            diff = np.max(np.abs(x_new - x))
            x = x_new
            if diff < tol:
                break
        # 归一化
        s = np.sum(x)
        if s > 0:
            x = x / s
        return x

    def pagerank_style_rank(self, damping=0.85, max_iter=200, tol=1e-10):
        """
        PageRank 风格的状态重要性排序。

        求解:
            r = (1-d)/N * 1 + d * P^T r
        其中 d 为阻尼因子（typical 0.85）。
        """
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
        """
        由稳态分布计算自由能景观:
            F_i = -k_B T ln(π_i)
        """
        pi = self.power_method_steady_state()
        pi = np.clip(pi, 1e-12, 1.0)
        F = -kb * temperature * np.log(pi)
        return F

    def implied_timescales(self, n_eigen=5):
        """
        计算隐含的弛豫时间尺度:
            τ_k = -dt / ln(λ_k)
        其中 λ_k 为转移矩阵的特征值（|λ_k| < 1）。
        """
        eigenvalues = np.linalg.eigvals(self.P)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[1:n_eigen]  # 去掉 1
        # 假设 dt=1（无量纲），τ = -1/ln(λ)
        eigenvalues = np.clip(eigenvalues, 1e-12, 1.0 - 1e-12)
        timescales = -1.0 / np.log(eigenvalues)
        return timescales


def build_lipid_adjacency(nx, ny, interaction_range=1):
    """
    为 nx×ny 的脂质格点构建邻接图。

    节点编号: idx = i * ny + j
    """
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
    """
    由邻接图构建离散扩散算子矩阵 L。

    L_{ii} = -Σ_j D/dx²  (对角)
    L_{ij} = D/dx²       (i,j 为邻居)

    稳态方程 L ρ = 0 的解为均匀密度。
    """
    L = np.zeros((n_nodes, n_nodes))
    for i, neighbors in adj_dict.items():
        if len(neighbors) == 0:
            continue
        rate = D * len(neighbors)
        L[i, i] = -rate
        for j in neighbors:
            L[i, j] = D
    return L
