"""
spectral_dg.py
谱方法与间断Galerkin离散化模块

融合种子项目:
- 161_chebyshev_matrix: Chebyshev谱微分矩阵
- 273_dg1d_heat: 1D DG方法（Jacobi多项式、Vandermonde矩阵、 lift算子、低存储RK）

科学应用: Cosserat杆方程的空间高阶离散
"""

import numpy as np
from typing import Tuple
from scipy.special import jacobi, roots_jacobi


def chebyshev_matrix(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chebyshev谱微分矩阵 — 基于种子项目161_chebyshev_matrix
    在n+1个Chebyshev节点上构造(n+1)x(n+1)微分矩阵D

    节点: x_j = cos(j*pi/n), j=0,...,n
    权重: c = [2, 1, ..., 1, 2] .* (-1)^j
    非对角元: D_ij = (c_i/c_j) / (x_i - x_j), i!=j
    对角元: D_ii = -sum_{j!=i} D_ji  (负和技巧)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))

    X = np.tile(x.reshape(-1, 1), (1, n + 1))
    dX = X - X.T

    D = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                D[i, j] = c[i] / (c[j] * (x[i] - x[j]))

    # 对角元负和技巧
    for i in range(n + 1):
        D[i, i] = -np.sum(D[:, i])
        # 更精确的计算（端点特殊处理）
        if i == 0:
            D[i, i] = (2.0 * n * n + 1.0) / 6.0
        elif i == n:
            D[i, i] = -(2.0 * n * n + 1.0) / 6.0
        else:
            D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))

    return x, D


def jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    计算Jacobi多项式 P_n^{(alpha,beta)}(x) 的值 — 基于种子项目273_dg1d_heat
    使用三项递推关系:
        P_0 = 1
        P_1 = 0.5*(alpha-beta) + 0.5*(alpha+beta+2)*x
        a_n P_{n+1} = (b_n + c_n x) P_n - d_n P_{n-1}
    """
    x = np.atleast_1d(x)
    if N == 0:
        return np.ones_like(x)
    if N == 1:
        return 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    # 存储所有阶数
    P = np.zeros((N + 1, len(x)))
    P[0] = 1.0
    P[1] = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    for n in range(1, N):
        a1 = 2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * (2.0 * n + alpha + beta)
        a2 = (2.0 * n + alpha + beta + 1.0) * (alpha ** 2 - beta ** 2)
        a3 = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta + 1.0) * (2.0 * n + alpha + beta + 2.0)
        a4 = 2.0 * (n + alpha) * (n + beta) * (2.0 * n + alpha + beta + 2.0)

        P[n + 1] = ((a2 + a3 * x) * P[n] - a4 * P[n - 1]) / a1

    return P[N]


def vandermonde_1d(N: int, r: np.ndarray) -> np.ndarray:
    """
    1D Vandermonde矩阵 — 基于种子项目273_dg1d_heat
    V_ij = P_j(r_i), i=0,...,len(r)-1, j=0,...,N
    """
    r = np.atleast_1d(r)
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def dmatrix_1d(N: int, r: np.ndarray, V: np.ndarray = None) -> np.ndarray:
    """
    1D谱微分矩阵 — 基于种子项目273_dg1d_heat
    Dr = Vr / V, 其中 Vr_ij = dP_j/dr(r_i)
    """
    if V is None:
        V = vandermonde_1d(N, r)

    # 计算Jacobi多项式导数
    Vr = np.zeros_like(V)
    for j in range(N + 1):
        if j == 0:
            Vr[:, j] = 0.0
        else:
            # d/dr P_j^{(0,0)} = sqrt(j(j+1)) * P_{j-1}^{(1,1)} (归一化版本)
            # 这里使用非归一化版本的导数公式
            Vr[:, j] = 0.5 * (j + 1.0) * jacobi_polynomial(r, 1.0, 1.0, j - 1)

    Dr = Vr @ np.linalg.inv(V)
    return Dr


def jacobi_gauss_lobatto(alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Jacobi-Gauss-Lobatto点 — 基于种子项目273_dg1d_heat
    在[-1,1]上包含端点的N+1个点
    内部点为 P_{N-1}^{(alpha+1,beta+1)} 的零点
    """
    if N == 0:
        return np.array([-1.0])
    if N == 1:
        return np.array([-1.0, 1.0])

    # 内部点
    x_int, _ = roots_jacobi(N - 1, alpha + 1.0, beta + 1.0)
    x = np.sort(np.concatenate([[-1.0], x_int, [1.0]]))
    return x


def lift_1d(N: int, V: np.ndarray = None) -> np.ndarray:
    """
    DG lift算子 — 基于种子项目273_dg1d_heat
    LIFT = V * V^T * Emat
    用于将面通量提升入单元内部
    """
    if V is None:
        # 使用Gauss-Lobatto点
        r = jacobi_gauss_lobatto(0.0, 0.0, N)
        V = vandermonde_1d(N, r)

    Np = N + 1
    Nfaces = 2
    Nfp = 1

    Emat = np.zeros((Np, Nfaces * Nfp))
    Emat[0, 0] = 1.0   # 左端点
    Emat[Np - 1, 1] = 1.0  # 右端点

    invV = np.linalg.inv(V)
    LIFT = V @ (V.T @ Emat)
    return LIFT


def dg_derivative_1d(u: np.ndarray, Dr: np.ndarray, rx: np.ndarray,
                     LIFT: np.ndarray, nx: np.ndarray,
                     vmapM: np.ndarray, vmapP: np.ndarray) -> np.ndarray:
    """
    1D DG空间导数计算 — 基于种子项目273_dg1d_heat的HeatCRHS1D思想
    计算 du/dx 的DG近似

    参数:
        u: (Np, K) 单元内节点值
        Dr: (Np, Np) 参考单元微分矩阵
        rx: (Np, K) 几何因子 1/J
        LIFT: (Np, 2) lift算子
        nx: (2, K) 面法向
        vmapM, vmapP: 面节点映射
    """
    Np, K = u.shape

    # 体积项
    dudr = Dr @ u

    # 面通量（central flux）
    uM = u.flatten()[vmapM]
    uP = u.flatten()[vmapP]
    flux = 0.5 * (uM + uP)  # central flux

    # 边界条件（Dirichlet零边界）
    # 简化处理：如果vmapP指向边界（超出范围），设为0
    # 这里假设调用者已经处理好vmapP

    # 面跳跃
    du = uM - uP
    flux = nx.flatten() * du * 0.5

    # 提升
    Nfaces = 2
    fluxmat = flux.reshape(Nfaces, K, order='F')
    lifted = LIFT @ fluxmat

    # 总导数
    dudx = rx * dudr - rx * lifted
    return dudx


def build_maps_1d(Np: int, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建1D DG面节点映射 — 基于种子项目273_dg1d_heat
    vmapM: 当前单元面节点索引
    vmapP: 相邻单元面节点索引
    nx: 面法向
    """
    Nfaces = 2
    Nfp = 1
    NfacesK = Nfaces * K

    # 面节点全局索引
    vmapM = np.zeros(NfacesK, dtype=int)
    vmapP = np.zeros(NfacesK, dtype=int)
    nx = np.zeros((Nfaces, K))

    for k in range(K):
        # 左面
        vmapM[0 * K + k] = k * Np
        # 右面
        vmapM[1 * K + k] = k * Np + Np - 1

        # 邻居映射
        if k == 0:
            vmapP[0 * K + k] = vmapM[1 * K + k]  # 自反射（简化边界）
        else:
            vmapP[0 * K + k] = vmapM[1 * K + (k - 1)]

        if k == K - 1:
            vmapP[1 * K + k] = vmapM[0 * K + k]
        else:
            vmapP[1 * K + k] = vmapM[0 * K + (k + 1)]

        nx[0, k] = -1.0
        nx[1, k] = 1.0

    return vmapM, vmapP, nx


def meshgen_1d(xmin: float, xmax: float, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D均匀网格生成 — 基于种子项目273_dg1d_heat
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    VX = np.linspace(xmin, xmax, K + 1)
    EToV = np.zeros((K, 2), dtype=int)
    for k in range(K):
        EToV[k] = [k, k + 1]
    return VX, EToV


def geometric_factors_1d(x: np.ndarray, Dr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D几何因子计算 — 基于种子项目273_dg1d_heat
    J = Dr @ x,  rx = 1/J
    """
    J = Dr @ x
    if np.any(np.abs(J) < 1e-14):
        # 保护小值
        J = np.where(np.abs(J) < 1e-14, np.sign(J + 1e-14) * 1e-14, J)
    rx = 1.0 / J
    return rx, J
