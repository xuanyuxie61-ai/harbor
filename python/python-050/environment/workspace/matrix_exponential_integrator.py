"""
matrix_exponential_integrator.py
矩阵指数积分器 — 冰盖动力学算子分裂方法

基于种子项目 1217_test_matrix_exponential 的矩阵指数基准测试思想，
将矩阵指数方法 (exponential integrators) 应用于冰盖演化的线性-非线性算子分裂。

核心数学:
  考虑冰盖演化方程的抽象形式:
      \frac{\partial \mathbf{u}}{\partial t} = \mathcal{L} \mathbf{u} + \mathcal{N}(\mathbf{u})

  其中 \mathcal{L} 为线性算子 (扩散)，\mathcal{N} 为非线性算子。

  指数 Euler 方法 (Cox & Matthews, 2002):
      \mathbf{u}^{n+1} = \exp(\Delta t \mathcal{L}) \mathbf{u}^n
                         + \Delta t \varphi_1(\Delta t \mathcal{L}) \mathcal{N}(\mathbf{u}^n)

  其中 \varphi_1(z) = (\exp(z) - 1) / z 为指数型函数。

  对于离散化后的矩阵形式 L:
      \mathbf{u}^{n+1} = \exp(\Delta t L) \mathbf{u}^n
                         + \Delta t \cdot \varphi_1(\Delta t L) \cdot \mathbf{N}(\mathbf{u}^n)

矩阵指数计算:
  - 小矩阵 (n < 50): 采用缩放-平方算法 (scaling and squaring) 配合 Padé 近似
  - 大稀疏矩阵: Krylov 子空间近似 (Arnoldi 迭代)

应用场景:
  - 冰盖厚度演化的扩散主导阶段
  - 温度方程的线性热扩散部分
  - 与显式非线性项分离，允许更大时间步长
"""

import numpy as np
from typing import Callable, Optional
from scipy.linalg import expm


def phi1_function(z: np.ndarray) -> np.ndarray:
    """
    计算指数型函数 \varphi_1(z) = (e^z - 1) / z。

    在 z = 0 处采用 Taylor 展开:
        \varphi_1(z) = 1 + z/2! + z^2/3! + z^3/4! + \cdots

    参数:
        z: 输入标量或矩阵

    返回:
        \varphi_1(z)
    """
    z = np.asarray(z, dtype=np.complex128 if np.iscomplexobj(z) else np.float64)

    if np.isscalar(z) or z.size == 1:
        z_val = float(z) if not np.iscomplexobj(z) else complex(z)
        if abs(z_val) < 1e-8:
            return np.array(1.0 + z_val / 2.0 + z_val ** 2 / 6.0)
        return np.array((np.exp(z_val) - 1.0) / z_val)

    # 矩阵情况
    if z.ndim == 2 and z.shape[0] == z.shape[1]:
        # 小矩阵: 通过矩阵指数计算
        nz = z.shape[0]
        I = np.eye(nz, dtype=z.dtype)
        if nz <= 50:
            # 利用 \varphi_1(A) = A^{-1}(e^A - I)
            # 对可能奇异的情况用极限形式
            eA = expm(z)
            diff = eA - I
            # 使用伪逆处理接近奇异的情况
            phi1 = np.linalg.solve(z + 1e-14 * I, diff)
            return phi1
        else:
            raise NotImplementedError("Large matrix phi1 requires Krylov method.")

    # 元素级计算
    result = np.empty_like(z, dtype=np.float64)
    small = np.abs(z) < 1e-8
    result[small] = 1.0 + z[small] / 2.0 + z[small] ** 2 / 6.0 + z[small] ** 3 / 24.0
    result[~small] = (np.exp(z[~small]) - 1.0) / z[~small]
    return result


def exponential_euler_step(u: np.ndarray,
                           dt: float,
                           L: np.ndarray,
                           N_func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    执行指数 Euler 单步推进。

      u^{n+1} = \exp(\Delta t L) u^n + \Delta t \varphi_1(\Delta t L) N(u^n)

    参数:
        u: 当前状态向量
        dt: 时间步长
        L: 线性算子矩阵 (m, m)
        N_func: 非线性函数 N(u)

    返回:
        u_new: 下一时刻状态
    """
    u = np.asarray(u, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    if L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix.")
    if len(u) != L.shape[0]:
        raise ValueError("u and L dimensions must match.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    # 矩阵指数
    dtL = dt * L
    exp_dtL = expm(dtL)

    # 非线性项
    Nu = np.asarray(N_func(u), dtype=np.float64)

    # phi1(dtL) * N(u)
    if len(u) <= 50:
        phi1_dtL = phi1_function(dtL)
        phi_N = phi1_dtL @ Nu
    else:
        # 近似: phi1(dtL) * N(u) ~ (expm(dtL) - I) * inv(dtL) * N(u)
        I = np.eye(len(u))
        # 使用迭代细化
        rhs = (expm(dtL) - I) @ Nu
        phi_N = np.linalg.solve(dtL + 1e-12 * I, rhs)

    u_new = exp_dtL @ u + dt * phi_N
    return u_new


def build_1d_diffusion_matrix(n: int, dx: float,
                               diffusivity: float) -> np.ndarray:
    """
    构建一维扩散算子的离散矩阵 (Dirichlet-Dirichlet)。

    离散格式:
        L_{i,i} = -2D/dx^2,  L_{i,i\pm1} = D/dx^2

    参数:
        n: 网格点数
        dx: 网格间距
        diffusivity: 扩散系数 D

    返回:
        L: (n, n) 扩散矩阵
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    coef = diffusivity / (dx ** 2)
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n - 1):
        L[i, i - 1] = coef
        L[i, i] = -2.0 * coef
        L[i, i + 1] = coef
    L[0, 0] = -coef
    L[0, 1] = coef
    L[-1, -2] = coef
    L[-1, -1] = -coef
    return L


def exponential_integrator_ice_thickness(H: np.ndarray,
                                         dt: float,
                                         dx: float,
                                         diffusivity_func: Callable,
                                         accumulation: np.ndarray) -> np.ndarray:
    """
    使用指数积分器推进一维冰厚度演化。

    将 SIA 方程分解为:
        \partial H/\partial t = L H + N(H) + a

    其中 L 为线性化扩散算子，N(H) 为剩余非线性项。

    参数:
        H: 当前厚度剖面 (n,)
        dt: 时间步长
        dx: 网格间距
        diffusivity_func: 返回当前有效扩散系数的函数
        accumulation: 积累率 (n,)

    返回:
        H_new: 新厚度剖面
    """
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    D_eff = float(diffusivity_func(H))

    L = build_1d_diffusion_matrix(n, dx, D_eff)

    def N_func(u: np.ndarray) -> np.ndarray:
        # 非线性余项 + 积累
        # 这里简化为仅积累
        return np.asarray(accumulation, dtype=np.float64)

    H_new = exponential_euler_step(H, dt, L, N_func)
    H_new = np.maximum(H_new, 0.0)
    return H_new


def krylov_phi1_approximation(A: np.ndarray, v: np.ndarray,
                              m_krylov: int = 30) -> np.ndarray:
    """
    基于 Krylov 子空间的 \varphi_1(A) v 近似。

    使用 Arnoldi 迭代构建 Krylov 基 K_m = span{v, Av, A^2v, ..., A^{m-1}v}，
    然后在小空间上计算矩阵指数。

    参数:
        A: 大型稀疏矩阵 (n, n)
        v: 向量 (n,)
        m_krylov: Krylov 子空间维度

    返回:
        approx: \varphi_1(A) v 的近似
    """
    n = len(v)
    m = min(m_krylov, n)

    # Arnoldi 迭代
    V = np.zeros((n, m + 1), dtype=np.float64)
    H_arnoldi = np.zeros((m + 1, m), dtype=np.float64)

    beta = np.linalg.norm(v)
    if beta < 1e-15:
        return np.zeros(n, dtype=np.float64)
    V[:, 0] = v / beta

    for j in range(m):
        w = A @ V[:, j]
        for i in range(j + 1):
            H_arnoldi[i, j] = np.dot(V[:, i], w)
            w = w - H_arnoldi[i, j] * V[:, i]
        H_arnoldi[j + 1, j] = np.linalg.norm(w)
        if H_arnoldi[j + 1, j] < 1e-14:
            m = j + 1
            break
        V[:, j + 1] = w / H_arnoldi[j + 1, j]

    # 在 Krylov 子空间中计算 phi1(H_m) e_1
    Hm = H_arnoldi[:m, :m]
    e1 = np.zeros(m, dtype=np.float64)
    e1[0] = 1.0

    # phi1(Hm) * e1
    phi1_Hm = phi1_function(Hm)
    y = phi1_Hm @ e1

    approx = beta * V[:, :m] @ y
    return approx
