"""
p-version 一维有限元求解器
基于 fem1d_pmethod 和 fem1d_pack 的核心算法：
- 正交多项式基函数构造
- Gauss-Legendre 数值积分
- Galerkin 投影求解扩散方程

应用：求解一维 Smoluchowski 方程 (Fokker-Planck 方程的漂移-扩散形式):
    ∂p/∂t = D * ∂²p/∂x² + (D/(k_B T)) * ∂/∂x( p(x) * ∂F/∂x )

稳态形式 (时间导数为0):
    D * d²p/dx² + (D/(k_B T)) * d/dx( p * dF/dx ) = 0
    等价于: d/dx[ D * ( dp/dx + (1/(k_B T)) * p * dF/dx ) ] = 0
    即概率流 J = -D * (dp/dx + (1/(k_B T)) * p * dF/dx) = const

在自由能景观分析中，用于计算沿反应坐标的稳态概率密度分布 p(x)，
进而得到自由能 F(x) = -k_B T ln p(x) + const。
"""

import numpy as np
from typing import Callable, Tuple


def legendre_com(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Legendre 积分的节点和权重。
    
    积分公式:
        ∫_{-1}^{1} f(x) dx ≈ sum_{i=1}^{n} w_i * f(x_i)
    
    节点 x_i 为 n 阶 Legendre 多项式 P_n(x) 的根，权重:
        w_i = 2 / [ (1 - x_i^2) * (P_n'(x_i))^2 ]
    
    本实现采用 Newton 迭代结合渐近公式初猜定位根。
    
    Parameters
    ----------
    n : int
        积分点数，要求 n >= 1。
    
    Returns
    -------
    x : np.ndarray
        积分节点。
    w : np.ndarray
        积分权重。
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    
    # 使用 numpy 的多项式 Legendre 根作为初猜
    # 实际上用 numpy 的 legendre.roots 更高效，但为了展示算法:
    # 这里使用标准实现
    x = np.zeros(n)
    w = np.zeros(n)
    
    m = (n + 1) // 2
    eps = 1e-14
    
    for i in range(1, m + 1):
        # 根的初始猜测 (渐近公式)
        z = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        z1 = 0.0
        while abs(z - z1) > eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        x[i - 1] = -z
        x[n - i] = z
        w[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp)
        w[n - i] = w[i - 1]
    return x, w


def local_basis_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    """
    计算一维 Lagrange 插值基函数在点 x 处的值。
    
    基函数定义:
        φ_i(x) = prod_{j≠i} (x - x_j) / (x_i - x_j)
    
    Parameters
    ----------
    order : int
        基函数个数 (多项式阶数+1)。
    node_x : np.ndarray, shape (order,)
        节点坐标。
    x : float
        求值点。
    
    Returns
    -------
    phi : np.ndarray, shape (order,)
        基函数值。
    """
    if len(node_x) != order:
        raise ValueError("node_x length must equal order")
    phi = np.ones(order)
    for i in range(order):
        for j in range(order):
            if i != j:
                denom = node_x[i] - node_x[j]
                if abs(denom) < 1e-14:
                    raise ValueError("Nodes too close")
                phi[i] *= (x - node_x[j]) / denom
    return phi


def local_basis_prime_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    """
    计算一维 Lagrange 基函数的一阶导数在点 x 处的值。
    
    导数公式:
        dφ_i/dx = sum_{j≠i} [ 1/(x_i - x_j) * prod_{k≠i,j} (x - x_k)/(x_j - x_k) ]
    
    Parameters
    ----------
    order : int
        基函数个数。
    node_x : np.ndarray, shape (order,)
        节点坐标。
    x : float
        求值点。
    
    Returns
    -------
    dphi : np.ndarray, shape (order,)
        基函数导数值。
    """
    dphi = np.zeros(order)
    for i in range(order):
        for j in range(order):
            if i != j:
                denom_ij = node_x[i] - node_x[j]
                if abs(denom_ij) < 1e-14:
                    raise ValueError("Nodes too close")
                term = 1.0 / denom_ij
                for k in range(order):
                    if k != i and k != j:
                        denom_jk = node_x[j] - node_x[k]
                        if abs(denom_jk) < 1e-14:
                            raise ValueError("Nodes too close")
                        term *= (x - node_x[k]) / denom_jk
                dphi[i] += term
    return dphi


def solve_steady_smoluchowski_1d(x_nodes: np.ndarray, free_energy: np.ndarray,
                                 D: float = 1.0, kT: float = 1.0,
                                 p_left: float = 1.0, p_right: float = 1.0) -> np.ndarray:
    """
    用有限元方法求解一维稳态 Smoluchowski 方程。
    
    控制方程:
        d/dx[ D * dp/dx + (D/kT) * p * dF/dx ] = 0,  x in [x_min, x_max]
        p(x_min) = p_left,  p(x_max) = p_right
    
    采用线性有限元 (P1) 离散，单元刚度矩阵通过 Gauss-Legendre 积分计算。
    
    Parameters
    ----------
    x_nodes : np.ndarray, shape (N,)
        一维网格节点（已排序）。
    free_energy : np.ndarray, shape (N,)
        节点上的自由能值 F(x)。
    D : float
        扩散系数。
    kT : float
        热能量 k_B T。
    p_left, p_right : float
        Dirichlet 边界条件。
    
    Returns
    -------
    p : np.ndarray, shape (N,)
        稳态概率密度分布。
    """
    N = len(x_nodes)
    if len(free_energy) != N:
        raise ValueError("x_nodes and free_energy must have the same length")
    if N < 3:
        raise ValueError("Need at least 3 nodes")
    
    # 构造三对角线性系统 A * p = b
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # Gauss-Legendre 2点规则足够精确处理 P1 元
    gauss_xi, gauss_w = legendre_com(2)
    
    for e in range(N - 1):
        xL, xR = x_nodes[e], x_nodes[e + 1]
        h = xR - xL
        if h <= 0:
            raise ValueError("x_nodes must be strictly increasing")
        
        # 单元内的自由能导数近似 (常数)
        dF_dx = (free_energy[e + 1] - free_energy[e]) / h
        
        # 单元刚度矩阵 (2x2)
        Ke = np.zeros((2, 2))
        for q in range(len(gauss_xi)):
            xi = gauss_xi[q]
            w = gauss_w[q]
            # 坐标变换: x = xL + 0.5*h*(1+xi), dx = 0.5*h*dxi
            # P1 基函数: N1 = (1-xi)/2, N2 = (1+xi)/2
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            dN1 = -0.5
            dN2 = 0.5
            
            # TODO: Hole 2 - 实现单元刚度矩阵的漂移-扩散算子组装
            # 要求: 组装扩散项和对流项到单元刚度矩阵 Ke
            # 提示: jac = 0.5 * h, factor = D * jac
            #       扩散项: D * ∫ dN_i/dx * dN_j/dx dx
            #       对流项: (D/kT) * dF/dx * ∫ N_i * dN_j/dx dx
            #       注意坐标变换导致 dN/dx = dN/dxi * (1/jac)
            raise NotImplementedError("Hole 2: 请补全单元刚度矩阵漂移-扩散算子组装")
        
        # 组装到全局矩阵
        A[e, e] += Ke[0, 0]
        A[e, e + 1] += Ke[0, 1]
        A[e + 1, e] += Ke[1, 0]
        A[e + 1, e + 1] += Ke[1, 1]
    
    # 施加 Dirichlet 边界条件
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = p_left
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = p_right
    
    p = np.linalg.solve(A, b)
    # 确保概率密度非负
    p = np.maximum(p, 1e-12)
    return p


def solve_fokker_planck_eigenvalue_1d(x_nodes: np.ndarray, potential: np.ndarray,
                                      D: float = 1.0, kT: float = 1.0,
                                      n_modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解一维 Fokker-Planck 算子的前 n_modes 个特征值和特征函数。
    
    在蛋白质折叠中，最小非零特征值对应折叠/解折叠的速率 (Kramers rate)。
    
    控制方程 (Fokker-Planck / Smoluchowski 算子):
        L[p] = D * d²p/dx² + (D/kT) * d/dx( p * dV/dx )
    
    求解广义特征值问题:
        K * u = λ * M * u
    
    其中 K 为刚度矩阵，M 为质量矩阵。
    
    Parameters
    ----------
    x_nodes : np.ndarray
        网格节点。
    potential : np.ndarray
        势能 V(x)（或自由能 F(x)）。
    D : float
        扩散系数。
    kT : float
        热能量。
    n_modes : int
        需要计算的特征模态数。
    
    Returns
    -------
    eigenvalues : np.ndarray
        特征值（按升序排列）。
    eigenvectors : np.ndarray, shape (N, n_modes)
        特征函数（每列为一个特征向量）。
    """
    N = len(x_nodes)
    K = np.zeros((N, N))
    M = np.zeros((N, N))
    
    gauss_xi, gauss_w = legendre_com(3)
    
    for e in range(N - 1):
        xL, xR = x_nodes[e], x_nodes[e + 1]
        h = xR - xL
        jac = 0.5 * h
        
        dV_dx = (potential[e + 1] - potential[e]) / h
        
        Ke = np.zeros((2, 2))
        Me = np.zeros((2, 2))
        for q in range(len(gauss_xi)):
            xi = gauss_xi[q]
            w = gauss_w[q]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            dN1 = -0.5
            dN2 = 0.5
            
            # 刚度: D * ∫ dN_i * dN_j dx
            factor = D * w / jac
            Ke[0, 0] += factor * dN1 * dN1
            Ke[0, 1] += factor * dN1 * dN2
            Ke[1, 0] += factor * dN2 * dN1
            Ke[1, 1] += factor * dN2 * dN2
            
            # 质量: ∫ N_i * N_j dx
            factor_mass = jac * w
            Me[0, 0] += factor_mass * N1 * N1
            Me[0, 1] += factor_mass * N1 * N2
            Me[1, 0] += factor_mass * N2 * N1
            Me[1, 1] += factor_mass * N2 * N2
            
            # 对流项 (非对称)
            drift = (D / kT) * dV_dx * w
            Ke[0, 0] += drift * N1 * dN1
            Ke[0, 1] += drift * N1 * dN2
            Ke[1, 0] += drift * N2 * dN1
            Ke[1, 1] += drift * N2 * dN2
        
        idx = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K[idx[i], idx[j]] += Ke[i, j]
                M[idx[i], idx[j]] += Me[i, j]
    
    # Dirichlet 边界条件 (零边界)
    K[0, :] = 0.0
    K[0, 0] = 1.0
    M[0, :] = 0.0
    M[0, 0] = 1e-10  # 避免奇异
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    M[-1, :] = 0.0
    M[-1, -1] = 1e-10
    
    # 广义特征值问题
    eigvals, eigvecs = np.linalg.eig(np.linalg.solve(M + 1e-12 * np.eye(N), K))
    # 排序并取实部
    idx_sorted = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[idx_sorted])
    eigvecs = np.real(eigvecs[:, idx_sorted])
    return eigvals[:n_modes], eigvecs[:, :n_modes]
