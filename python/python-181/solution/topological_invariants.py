"""
topological_invariants.py
六边形对称群与拓扑不变量计算
融合原项目: 340_eternity_hexity, 672_lights_out

核心科学思想:
利用六边形(hexagonal)晶格的对称群 D_6 (二面体群 of order 12)
以及Lights Out问题的模2线性代数结构，
提取数据流形的拓扑不变量与离散对称性特征。

数学模型:
六边形对称群 D_6:
    生成元: r (旋转 π/3), s (反射)
    关系: r^6 = s^2 = e, srs = r^{-1}

Lights Out矩阵:
    5x5 网格的邻接矩阵 A (mod 2)
    A_{ij} = 1 若格子i与j相邻或i=j
    配置空间: F_2^{25}

Betti数估计:
    通过图Laplacian的零空间维数估计0阶Betti数 (连通分支数)
"""

import numpy as np
from typing import Tuple, List


def hexity_rotate(hex_state: np.ndarray, k: int = 1) -> np.ndarray:
    """
    六边形状态旋转 k * 60 度
    hex_state: 12维向量，表示六边形的12个三角片类型
    """
    rotated = np.roll(hex_state, k)
    return rotated


def hexity_reflect(hex_state: np.ndarray) -> np.ndarray:
    """
    六边形状态反射
    """
    reflected = hex_state.copy()
    # 反转顺序并调整符号
    reflected = np.flip(reflected)
    return reflected


def dihedral_group_d6_action(state: np.ndarray, operation: str) -> np.ndarray:
    """
    D_6 群作用
    operation: 'r0'..'r5' (旋转), 's0'..'s5' (反射)
    """
    if operation.startswith('r'):
        k = int(operation[1])
        return hexity_rotate(state, k)
    elif operation.startswith('s'):
        k = int(operation[1])
        s = hexity_reflect(state)
        return hexity_rotate(s, k)
    else:
        return state.copy()


def orbit_under_d6(state: np.ndarray) -> List[np.ndarray]:
    """
    计算D_6群作用下状态的轨道
    返回轨道中所有不同状态
    """
    orbit = []
    seen = set()
    for op in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5',
               's0', 's1', 's2', 's3', 's4', 's5']:
        new_state = dihedral_group_d6_action(state, op)
        key = tuple(new_state.tolist())
        if key not in seen:
            seen.add(key)
            orbit.append(new_state)
    return orbit


def symmetry_order(state: np.ndarray) -> int:
    """
    状态的对称阶数 (稳定子群的阶)
    """
    stabilizer_size = 0
    for op in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5',
               's0', 's1', 's2', 's3', 's4', 's5']:
        new_state = dihedral_group_d6_action(state, op)
        if np.allclose(new_state, state):
            stabilizer_size += 1
    return stabilizer_size


def lights_out_matrix(mrow: int = 5, ncol: int = 5) -> np.ndarray:
    """
    构建 mrow x ncol 的Lights Out矩阵 (mod 2)
    A_{c,neighbor} = 1 若相邻或自身
    """
    n = mrow * ncol
    A = np.zeros((n, n), dtype=int)
    def index(i, j):
        if i < 0 or i >= mrow or j < 0 or j >= ncol:
            return -1
        return i * ncol + j
    for i in range(mrow):
        for j in range(ncol):
            c = index(i, j)
            neighbors = [index(i, j), index(i - 1, j), index(i + 1, j),
                         index(i, j - 1), index(i, j + 1)]
            for nbr in neighbors:
                if nbr >= 0:
                    A[nbr, c] = 1
    return A


def lights_out_solve(initial: np.ndarray, mrow: int = 5, ncol: int = 5) -> np.ndarray:
    """
    求解Lights Out问题: A p = initial (mod 2)
    使用高斯消元法在 F_2 上求解
    """
    A = lights_out_matrix(mrow, ncol)
    n = mrow * ncol
    b = initial.copy() % 2
    # 增广矩阵
    aug = np.hstack([A.astype(int), b.reshape(-1, 1)])
    # F_2 高斯消元
    for col in range(n):
        # 找主元
        pivot = -1
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue
        # 交换行
        aug[[col, pivot]] = aug[[pivot, col]]
        # 消去
        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[col]) % 2
    # 回代
    p = aug[:, n].copy()
    return p % 2


def betti_number_estimate(edges: np.ndarray, n_vertices: int) -> int:
    """
    通过图Laplacian的零空间维数估计0阶Betti数
    β_0 = dim ker(L) = 连通分支数
    """
    # 构建邻接矩阵
    W = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for (i, j) in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    # 计算零特征值数量
    eigvals = np.linalg.eigvalsh(L)
    # 判断接近0的特征值
    threshold = 1e-10
    beta_0 = int(np.sum(eigvals < threshold))
    return max(beta_0, 1)


def persistence_homology_filtration(data: np.ndarray, radii: np.ndarray) -> dict:
    """
    简单的持久同调 filtration
    追踪随着半径增大，连通分支数 (β_0) 的变化
    """
    N = len(data)
    persistence = {}
    for r in radii:
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(data[i] - data[j]) <= r:
                    edges.append([i, j])
        edges = np.array(edges, dtype=int) if len(edges) > 0 else np.zeros((0, 2), dtype=int)
        beta_0 = betti_number_estimate(edges, N)
        persistence[r] = beta_0
    return persistence


def discrete_topological_features(data: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    将数据离散化为拓扑特征向量
    使用六边形铺砌对2D投影进行编码
    """
    from linear_algebra_core import jacobi_eigenvalue
    # PCA到2维
    cov = np.cov(data.T)
    _, vecs = jacobi_eigenvalue(cov)
    proj = data @ vecs[:, :2]
    # 离散化为网格
    xmin, xmax = np.min(proj[:, 0]), np.max(proj[:, 0])
    ymin, ymax = np.min(proj[:, 1]), np.max(proj[:, 1])
    grid = np.zeros((n_bins, n_bins), dtype=int)
    for p in proj:
        ix = int((p[0] - xmin) / (xmax - xmin + 1e-10) * n_bins)
        iy = int((p[1] - ymin) / (ymax - ymin + 1e-10) * n_bins)
        ix = min(ix, n_bins - 1)
        iy = min(iy, n_bins - 1)
        grid[ix, iy] = 1
    # 将网格展平为特征向量
    return grid.ravel()
