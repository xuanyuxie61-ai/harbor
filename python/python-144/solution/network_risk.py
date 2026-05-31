"""
network_risk.py
资产网络构建、系统性风险度量与风险传播模拟模块。

融入的原项目核心算法：
- 1205_test_digraph_arc: 有向图弧列表与PageRank网络结构
- 1170_stochastic_heat2d: 二维随机热方程（稀疏矩阵有限差分）
- 1352_triangulation_svg: 三角剖分数据结构与Delaunay剖分思想

科学背景：
现代金融系统的风险不仅来自单个资产，更来自资产间的复杂关联网络。
通过构建资产依赖图，应用PageRank-like算法识别系统性重要资产；
利用随机热方程模拟风险冲击在网络中的扩散过程；
结合Delaunay三角剖分构建资产相似性网络，识别潜在的风险传染路径。
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay


def build_asset_digraph(n: int, threshold: float, corr: np.ndarray) -> np.ndarray:
    """
    基于相关性阈值构建资产依赖有向图的邻接矩阵。

    构造规则：
    对每对资产 (i, j)，若 |C_{ij}| ≥ threshold 且 i ≠ j，
    则添加有向边 i → j，边权重为 |C_{ij}|。

    参数
    ----------
    n : int
        资产数量。
    threshold : float
        相关性阈值，默认 0.3。
    corr : np.ndarray
        相关性矩阵。

    返回
    -------
    np.ndarray, shape (n, n)
        有向图邻接矩阵（行随机化前）。
    """
    if corr.shape != (n, n):
        raise ValueError("build_asset_digraph: 相关性矩阵维度不匹配。")
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr[i, j]) >= threshold:
                adj[i, j] = abs(corr[i, j])
    # 处理孤立节点：自环
    row_sums = adj.sum(axis=1)
    isolated = row_sums == 0
    adj[np.diag_indices(n)] = np.where(isolated, 1.0, adj[np.diag_indices(n)])
    return adj


def pagerank_systemic_risk(adj: np.ndarray, damping: float = 0.85,
                            max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """
    计算资产网络的PageRank-like系统性风险得分。

    数学模型（Moler, Experiments with MATLAB）：
    设 A 为邻接矩阵，P 为行随机矩阵：
        P_{ij} = A_{ij} / Σ_k A_{ik}。
    Google矩阵定义为
        G = α P + (1-α) (1/n) 1 1^T，
    其中 α 为阻尼因子（默认 0.85）。
    PageRank向量 x 满足
        x = G^T x，   Σ_i x_i = 1。

    在金融语境下，PageRank值越高的资产越"系统重要"，
    其风险事件越可能通过网络传染至整个系统。

    参数
    ----------
    adj : np.ndarray
        邻接矩阵。
    damping : float
        阻尼因子。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容差。

    返回
    -------
    np.ndarray
        每个资产的系统性风险得分（和为1）。
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    # 行随机化
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    P = adj / row_sums
    # 幂迭代
    x = np.ones(n) / n
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        x_new = damping * (P.T @ x) + teleport
        x_new = x_new / np.sum(x_new)
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x


def delaunay_similarity_triangulation(positions: np.ndarray) -> np.ndarray:
    """
    对资产在二维空间中的嵌入进行Delaunay三角剖分，构建相似性网络。

    Delaunay三角剖分满足空圆性质：
    对任意三角形，其外接圆内不含其他点。
    该剖分最大化最小角，避免狭长三角形，在几何上最优。

    返回三角剖分的边列表（无向图邻接矩阵）。
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("delaunay_similarity_triangulation: 输入必须是 N×2 数组。")
    if positions.shape[0] < 3:
        return np.zeros((positions.shape[0], positions.shape[0]))
    tri = Delaunay(positions)
    n = positions.shape[0]
    adj = np.zeros((n, n))
    for simplex in tri.simplices:
        i, j, k = simplex
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        adj[i, k] = 1.0
        adj[k, i] = 1.0
        adj[j, k] = 1.0
        adj[k, j] = 1.0
    return adj


def stochastic_risk_diffusion(network_adj: np.ndarray,
                               initial_risk: np.ndarray,
                               omega: np.ndarray,
                               nx: int = 20, ny: int = 20) -> np.ndarray:
    """
    基于二维随机热方程模拟风险在金融网络中的扩散。

    偏微分方程模型：
    将资产网络映射到二维网格，风险场 u(x,y) 满足稳态随机热方程：
        -∇·(a(x,y;ω) ∇u) = f(x,y),    (x,y) ∈ Ω,
        u = g(x,y),                    (x,y) ∈ ∂Ω。
    其中扩散系数 a(x,y;ω) 受随机参数 ω 控制，体现市场环境的不确定性；
    f(x,y) 为外部风险冲击源项；g(x,y) 为边界上的基准风险水平。

    离散化：
    采用五点有限差分格式，在 nx×ny 网格上建立稀疏线性系统 AU = F，
    通过稀疏直接求解器计算风险分布。

    参数
    ----------
    network_adj : np.ndarray
        网络邻接矩阵（用于确定风险传播结构）。
    initial_risk : np.ndarray
        初始风险冲击向量（长度 nx*ny）。
    omega : np.ndarray, shape (4,)
        随机参数 [ω1, ω2, ω3, ω4]，控制扩散系数。
    nx, ny : int
        网格尺寸。

    返回
    -------
    np.ndarray, shape (nx, ny)
        稳态风险分布场。
    """
    n = nx * ny
    if len(initial_risk) != n:
        raise ValueError("stochastic_risk_diffusion: initial_risk 长度必须等于 nx*ny。")
    if len(omega) != 4:
        raise ValueError("stochastic_risk_diffusion: omega 必须包含4个参数。")

    # 构造网格坐标
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # 扩散系数 a(x,y;ω) = ω1 + ω2 * sin(πx) * sin(πy)
    def diffusivity(xi, yj):
        return omega[0] + omega[1] * np.sin(np.pi * xi) * np.sin(np.pi * yj)

    # 稀疏矩阵构造（五点格式）
    row_ind = []
    col_ind = []
    data = []
    rhs = np.zeros(n)

    def idx(i, j):
        return i * nx + j

    for i in range(ny):
        for j in range(nx):
            k = idx(i, j)
            if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
                # Dirichlet 边界
                row_ind.append(k)
                col_ind.append(k)
                data.append(1.0)
                rhs[k] = omega[2]  # 边界风险水平
            else:
                a_ip = diffusivity(x[j], 0.5 * (y[i] + y[i + 1]))
                a_im = diffusivity(x[j], 0.5 * (y[i] + y[i - 1]))
                a_jp = diffusivity(0.5 * (x[j] + x[j + 1]), y[i])
                a_jm = diffusivity(0.5 * (x[j] + x[j - 1]), y[i])

                coeff = 0.0
                # 中心点
                center = (a_ip + a_im) / dy ** 2 + (a_jp + a_jm) / dx ** 2
                row_ind.append(k)
                col_ind.append(k)
                data.append(center)
                coeff += center

                # 上邻居
                row_ind.append(k)
                col_ind.append(idx(i + 1, j))
                data.append(-a_ip / dy ** 2)

                # 下邻居
                row_ind.append(k)
                col_ind.append(idx(i - 1, j))
                data.append(-a_im / dy ** 2)

                # 右邻居
                row_ind.append(k)
                col_ind.append(idx(i, j + 1))
                data.append(-a_jp / dx ** 2)

                # 左邻居
                row_ind.append(k)
                col_ind.append(idx(i, j - 1))
                data.append(-a_jm / dx ** 2)

                # 源项：将 initial_risk 映射为源项
                rhs[k] = omega[3] * initial_risk[k]

    A = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    u = spsolve(A, rhs)
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        raise RuntimeError("stochastic_risk_diffusion: 求解失败，矩阵可能奇异。")
    return u.reshape((ny, nx))


def network_risk_contribution(adj: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    """
    计算每个资产的网络风险贡献度。

    定义：
        RC_i = σ_i * Σ_j A_{ij} * β_{ij}
    其中 σ_i 为资产 i 的波动率，β_{ij} = Cov(r_i, r_j) / Var(r_j)。
    """
    n = adj.shape[0]
    cov = np.cov(asset_returns.T)
    vol = np.sqrt(np.diag(cov))
    beta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if cov[j, j] > 1e-12:
                beta[i, j] = cov[i, j] / cov[j, j]
    rc = vol * (adj * beta).sum(axis=1)
    return rc
