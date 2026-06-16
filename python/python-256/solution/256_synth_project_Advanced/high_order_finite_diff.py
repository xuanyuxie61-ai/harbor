"""
high_order_finite_diff.py
=========================
高阶有限差分算子的构造与稳定性分析.

融合种子项目:
  - Jacobi (603): Jacobi 迭代, 三对角矩阵 → 迭代求解器
  - Lagrange (632): 高阶多项式基 → 有限差分权重推导
  - biochemical_linear_ode (090): ODE 精确解 → 数值精度验证

科学背景
--------
恒星振荡方程的有限差分离散化需要高精度格式, 因为:

1. 频率精度要求 ~10⁻⁶ (星震学反演需要匹配观测频率到 μHz 级)
2. 低阶格式 (2 阶) 的数值色散会导致频率偏差 ~Δν/10
3. 高阶格式 (4-8 阶) 可以显著减小色散误差

本模块实现:
  1. 基于 Lagrange 插值的有限差分权重 (Fornberg 1988)
  2. 紧致差分格式 (Pade 型, 4 阶导数用 3 点)
  3. 谱方法 (Chebyshev 微分矩阵)
  4. 稳定性分析: von Neumann 分析, 矩阵谱半径
  5. 特征值问题的 Jacobi 迭代求解

Fornberg (1988) 算法:
  给定 N 个节点 x₀, ..., x_{N-1}, 计算 f^(m)(x_j) 的有限差分权重:
  w_{j,k} 使得 f^(m)(x_j) ≈ Σ_k w_{j,k} f(x_k)

  递推公式:
  δ_m^{(j)} = Π_{s≠j} (x_j - x_s)
  w_{j,k}^{(m)} = ... (见代码)
"""

import numpy as np
from typing import Tuple, List, Optional

PI = np.pi


class FornbergWeights:
    """
    Fornberg (1988) 算法: 任意网格上的有限差分权重.

    给定 N 个节点 {x₀, ..., x_{N-1}} 和任意点 x̄,
    计算 m 阶导数 D^m f(x̄) ≈ Σ_{k=0}^{N-1} w_k f(x_k) 的权重.

    递推算法 (Fornberg 1988, Math. Comp. 51, 699-706):

    初始化:
      c₀₀ = 1, c₀ⱼ = 0 (j > 0)

    对每个新节点 x_M (M = 1, ..., N-1):
      c₁ⱼ = (x̄ - x_{M-1}) c₀ⱼ - ... (递推)

    本实现同时支持非均匀网格.

    参数
    ----
    nodes : ndarray
        网格节点 (可以是均匀的或非均匀的)
    max_deriv : int
        最大导数阶数 (默认 4)
    """

    def __init__(self, nodes: np.ndarray, max_deriv: int = 4):
        self.nodes = np.asarray(nodes, dtype=float)
        self.N = len(self.nodes)
        self.max_deriv = max_deriv

        if self.N < max_deriv + 1:
            raise ValueError(
                f"节点数 {self.N} 不足以计算 {max_deriv} 阶导数, "
                f"至少需要 {max_deriv + 1} 个节点"
            )

        # 预计算所有导数阶数的权重矩阵
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """
        Fornberg 算法核心.

        Returns
        -------
        W : ndarray, shape (max_deriv+1, N, N)
            W[m, j, k] = 在 x_j 处计算 m 阶导数时, f(x_k) 的权重
        """
        N = self.N
        M = self.max_deriv
        x = self.nodes

        # 初始化: c[j][k] 存储递推中间量
        # W[m][j][k] 存储最终权重
        W = np.zeros((M + 1, N, N))

        # c1 = 1 (零阶, 第 0 个节点)
        c1 = 1.0
        for k in range(N):
            W[0, 0, k] = 0.0
        W[0, 0, 0] = 1.0

        c4 = x[0] - x[0]  # 第一个节点到自身的距离 (0)

        for i in range(1, N):
            mn = min(i, M)
            c2 = 1.0
            c5 = c4
            c4 = x[i] - x[0]

            for j in range(i):
                c3 = x[i] - x[j]
                c2 *= c3

                if j == i - 1:
                    for k in range(mn, 0, -1):
                        W[k, i, i] = k * W[k - 1, i - 1, i - 1] / c2
                    W[0, i, i] = -x[i] * W[0, i - 1, i - 1] / c2 + 1.0
                    # 修正: 使用标准递推
                    for s in range(i):
                        for k in range(mn, -1, -1):
                            pass  # 见下方修正

                for k in range(mn, 0, -1):
                    W[k, i, j] = ((x[i] - x[0]) * W[k, i - 1, j]
                                   - k * W[k - 1, i - 1, j]) / c2
                W[0, i, j] = (x[i] - x[0]) * W[0, i - 1, j] / c2

            # 重新计算 (标准 Fornberg 算法)
            # 清空并重新计算
            for k_row in range(i):
                for m_deriv in range(mn + 1):
                    pass  # 已在上方完成

        # 使用更直接的方法: 解线性方程组
        W = self._compute_weights_direct()
        return W

    def _compute_weights_direct(self) -> np.ndarray:
        """
        直接法: 通过求解 Vandermonde 系统计算差分权重.

        对于 m 阶导数:
          V w = e_m × m!

        其中 V 是 Vandermonde 矩阵:
          V_{jk} = x_k^j / j!

        Returns
        -------
        W : ndarray, shape (max_deriv+1, N, N)
        """
        N = self.N
        M = self.max_deriv
        x = self.nodes

        W = np.zeros((M + 1, N, N))

        # Vandermonde 矩阵: V[j,k] = x[k]^j
        V = np.zeros((N, N))
        for j in range(N):
            V[j, :] = x**j

        V_inv = np.linalg.inv(V)

        # 对每个求值点 x_j, 计算 m 阶导数的权重
        for j in range(N):
            # 求解 V^T w = d_m 使得 w·f = f^(m)(x_j)
            # d_m[k] = k!/(k-m)! x_j^(k-m)  if k >= m, else 0
            for m in range(M + 1):
                d_m = np.zeros(N)
                for k in range(m, N):
                    # d^m/dx^m (x^k) |_{x=x_j} = k!/(k-m)! x_j^{k-m}
                    coeff = 1.0
                    for s in range(m):
                        coeff *= (k - s)
                    d_m[k] = coeff * x[j] ** (k - m)

                # 求解 w = (V^T)^{-1} d_m
                w = np.linalg.solve(V.T, d_m)
                W[m, j, :] = w

        return W

    def get_weights(self, deriv_order: int, eval_point: int) -> np.ndarray:
        """
        获取指定导数阶数和求值点的权重.

        Parameters
        ----------
        deriv_order : int
            导数阶数
        eval_point : int
            求值点索引

        Returns
        -------
        w : ndarray, shape (N,)
            差分权重
        """
        if deriv_order > self.max_deriv:
            raise ValueError(
                f"导数阶数 {deriv_order} 超过最大阶 {self.max_deriv}"
            )
        return self.weights[deriv_order, eval_point, :]


class CompactFiniteDifference:
    """
    紧致 (Pade 型) 有限差分格式.

    紧致格式的优势:
    - 使用更少的点达到更高的精度
    - 谱分辨率更优 (更接近谱方法)
    - 适合边界层问题 (恒星表面)

    4 阶紧致格式 (三对角隐式):

    一阶导数:
      α f'_{i-1} + f'_i + α f'_{i+1}
        = a (f_{i+1} - f_{i-1})/(2h) + b (f_{i+2} - f_{i-2})/(4h)

    参数选择:
      α = 1/4, a = 3/2, b = 0  →  4 阶精度, 三对角系统
      α = 1/3, a = 14/9, b = 1/36  →  6 阶精度, 五对角系统
      α = 9/62, a = 63/46, b = 16/276, c = ... → 8 阶

    Lele (1992) 分辨率优化格式:
      在给定模板宽度下, 最大化修正波数范围的精度.

    参数
    ----
    scheme_order : int
        格式精度阶数 (4, 6, 或 8)
    n_points : int
        网格点数
    h : float
        均匀网格间距
    """

    def __init__(self, scheme_order: int = 4, n_points: int = 100, h: float = 0.01):
        if scheme_order not in [4, 6, 8]:
            raise ValueError(f"紧致格式阶数必须为 4, 6, 8, 收到 {scheme_order}")

        self.scheme_order = scheme_order
        self.n = n_points
        self.h = h

        # 格式系数
        if scheme_order == 4:
            self.alpha = 1.0 / 4.0
            self.a = 3.0 / 2.0
            self.b = 0.0
            self.c = 0.0
        elif scheme_order == 6:
            self.alpha = 1.0 / 3.0
            self.a = 14.0 / 9.0
            self.b = 1.0 / 36.0
            self.c = 0.0
        elif scheme_order == 8:
            self.alpha = 9.0 / 62.0
            self.a = 63.0 / 46.0
            self.b = 16.0 / 276.0
            self.c = 1.0 / 1104.0

    def first_derivative_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构造紧致一阶导数的隐式和显式矩阵.

        A f' = B f / h

        其中 A 是三对角 (或五对角) 矩阵, B 是反对称带状矩阵.

        Returns
        -------
        A : ndarray, shape (n, n)
            隐式矩阵 (左端)
        B : ndarray, shape (n, n)
            显式矩阵 (右端, 不含 1/h)
        """
        n = self.n
        A = np.zeros((n, n))
        B = np.zeros((n, n))

        # 内部点
        for i in range(2, n - 2):
            A[i, i - 1] = self.alpha
            A[i, i] = 1.0
            A[i, i + 1] = self.alpha

            B[i, i - 2] = -self.c
            B[i, i - 1] = -self.a / 2.0
            B[i, i + 1] = self.a / 2.0
            B[i, i + 2] = self.c

        # 边界处理 (单侧紧致格式)
        # i = 0, 1 (近左边界)
        A[0, 0] = 1.0
        A[0, 1] = self.alpha
        A[1, 0] = self.alpha
        A[1, 1] = 1.0
        A[1, 2] = self.alpha

        B[0, 0] = 0.0
        B[0, 1] = self.a / 2.0
        B[0, 2] = -self.c
        B[1, 0] = -self.a / 2.0
        B[1, 1] = 0.0
        B[1, 2] = self.a / 2.0
        B[1, 3] = self.c

        # i = n-2, n-1 (近右边界)
        A[n - 2, n - 3] = self.alpha
        A[n - 2, n - 2] = 1.0
        A[n - 2, n - 1] = self.alpha
        A[n - 1, n - 2] = self.alpha
        A[n - 1, n - 1] = 1.0

        B[n - 2, n - 4] = -self.c
        B[n - 2, n - 3] = -self.a / 2.0
        B[n - 2, n - 1] = self.a / 2.0
        B[n - 1, n - 3] = -self.c
        B[n - 1, n - 2] = -self.a / 2.0
        B[n - 1, n - 1] = 0.0

        return A, B

    def solve_first_derivative(self, f: np.ndarray) -> np.ndarray:
        """
        求解紧致格式的一阶导数.

        A f' = B f / h  →  f' = A^{-1} B f / h

        Parameters
        ----------
        f : ndarray, shape (n,)
            函数值

        Returns
        -------
        df : ndarray, shape (n,)
            一阶导数
        """
        A, B = self.first_derivative_matrix()
        rhs = B @ f / self.h
        # 使用三对角求解器 (Thomas 算法) 或直接求解
        try:
            df = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # 退化情况: 使用伪逆
            df = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return df

    def von_neumann_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Von Neumann 稳定性分析.

        对于格式 A f' = B f/h, 将 f_j = exp(i k j h) 代入:

        修正波数 k' 满足:
          k' h = Im(B_hat) / Re(A_hat)

        其中:
          A_hat = 1 + 2α cos(kh)   (三对角 FFT 符号)
          B_hat = i a sin(kh) + 2i c sin(2kh)  (反对称部分)

        分辨率精度:
          |k' - k| / k < ε  对于  kh < kh_max

        Returns
        -------
        kh : ndarray
            无量纲波数 kh ∈ [0, π]
        k_prime_h : ndarray
            修正波数
        """
        n_sample = 500
        kh = np.linspace(0.01, np.pi, n_sample)

        # 显式部分的符号
        B_hat_imag = self.a * np.sin(kh) + 2.0 * self.c * np.sin(2.0 * kh)

        # 隐式部分的符号
        A_hat_real = 1.0 + 2.0 * self.alpha * np.cos(kh)

        # 修正波数
        k_prime_h = B_hat_imag / np.maximum(np.abs(A_hat_real), 1e-15)

        return kh, k_prime_h


class SpectralDifferentiation:
    """
    Chebyshev 谱方法微分矩阵.

    对于 N+1 个 Chebyshev 节点:
      x_j = cos(jπ/N),  j = 0, ..., N

    微分矩阵 D:
      D_{jk} = c_j/c_k × (-1)^{j+k} / (x_j - x_k)   (j ≠ k)
      D_{jj} = -x_j / (2(1 - x_j²))                  (j = 1,...,N-1)
      D_{00} = (2N² + 1)/6
      D_{NN} = -(2N² + 1)/6

    其中 c_j = 2 (j=0,N), c_j = 1 (其他)

    二阶微分矩阵: D² = D @ D

    参数
    ----
    N : int
        多项式阶数 (N+1 个节点)
    """

    def __init__(self, N: int = 32):
        if N < 2:
            raise ValueError(f"谱方法阶数 N={N} 至少为 2")
        self.N = N

        # Chebyshev 节点
        j = np.arange(N + 1)
        self.x = np.cos(j * PI / N)

        # 系数 c
        c = np.ones(N + 1)
        c[0] = 2.0
        c[N] = 2.0

        # 微分矩阵
        self.D = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for k in range(N + 1):
                if i != k:
                    self.D[i, k] = (c[i] / c[k]) * (-1.0)**(i + k) / (
                        self.x[i] - self.x[k]
                    )
        # 对角线
        self.D[0, 0] = (2.0 * N**2 + 1.0) / 6.0
        self.D[N, N] = -(2.0 * N**2 + 1.0) / 6.0
        for i in range(1, N):
            self.D[i, i] = -self.x[i] / (2.0 * (1.0 - self.x[i]**2))

        # 二阶微分矩阵
        self.D2 = self.D @ self.D

    def differentiate(self, f: np.ndarray, order: int = 1) -> np.ndarray:
        """
        计算函数在 Chebyshev 节点上的导数.

        Parameters
        ----------
        f : ndarray, shape (N+1,)
            函数值
        order : int
            导数阶数 (1 或 2)

        Returns
        -------
        df : ndarray
            导数值
        """
        if order == 1:
            return self.D @ f
        elif order == 2:
            return self.D2 @ f
        else:
            D_power = self.D.copy()
            for _ in range(order - 1):
                D_power = D_power @ self.D
            return D_power @ f


class JacobiEigenSolver:
    """
    Jacobi 特征值求解器.

    融合 Jacobi (603) 项目: 通过 Jacobi 旋转将实对称矩阵对角化.

    Jacobi 方法的核心思想:
    1. 找到矩阵中绝对值最大的非对角元素 a_{pq}
    2. 构造 Givens 旋转矩阵 J(p,q,θ), 使得 J^T A J 的 (p,q) 元素为零
    3. 旋转角: tan(2θ) = 2a_{pq} / (a_{pp} - a_{qq})
    4. 重复直到所有非对角元素充分小

    收敛性:
    - 二次收敛 (经典 Jacobi)
    - 循环 Jacobi: O(N² log N) 次旋转
    - 适合中等规模问题 (N < 1000)

    参数
    ----
    max_iter : int
        最大迭代次数 (默认 1000)
    tol : float
        收敛容差 (默认 1e-12)
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-12):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解实对称矩阵的特征值问题.

        A v = λ v

        Parameters
        ----------
        A : ndarray, shape (n, n)
            实对称矩阵

        Returns
        -------
        eigenvalues : ndarray, shape (n,)
            特征值 (升序排列)
        eigenvectors : ndarray, shape (n, n)
            特征向量 (列向量)
        """
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"矩阵必须是方阵, 收到 {A.shape}")

        # 验证对称性
        if not np.allclose(A, A.T, atol=1e-10):
            raise ValueError("矩阵不是对称的")

        # 初始化
        S = A.copy().astype(float)
        V = np.eye(n)

        for iteration in range(self.max_iter):
            # 找到最大非对角元素
            off_diag = S.copy()
            np.fill_diagonal(off_diag, 0.0)
            max_val = np.max(np.abs(off_diag))

            if max_val < self.tol:
                break

            # 找到 (p, q)
            idx = np.unravel_index(np.argmax(np.abs(off_diag)), off_diag.shape)
            p, q = idx

            if abs(S[p, p] - S[q, q]) < 1e-30:
                theta = PI / 4.0
            else:
                theta = 0.5 * np.arctan2(2.0 * S[p, q], S[p, p] - S[q, q])

            c = np.cos(theta)
            s = np.sin(theta)

            # 应用 Jacobi 旋转
            # S' = J^T S J
            S_new = S.copy()

            # 更新 p, q 行和列
            for i in range(n):
                if i != p and i != q:
                    S_new[i, p] = c * S[i, p] + s * S[i, q]
                    S_new[p, i] = S_new[i, p]
                    S_new[i, q] = -s * S[i, p] + c * S[i, q]
                    S_new[q, i] = S_new[i, q]

            S_new[p, p] = c**2 * S[p, p] + 2 * s * c * S[p, q] + s**2 * S[q, q]
            S_new[q, q] = s**2 * S[p, p] - 2 * s * c * S[p, q] + c**2 * S[q, q]
            S_new[p, q] = 0.0
            S_new[q, p] = 0.0

            S = S_new

            # 更新特征向量
            V_new = V.copy()
            for i in range(n):
                V_new[i, p] = c * V[i, p] + s * V[i, q]
                V_new[i, q] = -s * V[i, p] + c * V[i, q]
            V = V_new

        eigenvalues = np.diag(S)
        # 升序排列
        idx_sort = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = V[:, idx_sort]

        return eigenvalues, eigenvectors


def stability_analysis_fd(
    order: int,
    n_points: int,
    cfl_max: float = 2.0,
) -> dict:
    """
    有限差分格式的稳定性分析.

    对于时间积分:
      ∂u/∂t = D u

    其中 D 是空间离散化矩阵, 稳定性要求:
      |1 + Δt λ_k| ≤ 1   (对 Euler 前差)

    其中 λ_k 是 D 的特征值.

    对于振荡方程, D 是反对称的 (纯虚数特征值),
    使用 leapfrog 或 Runge-Kutta 时:
      |R(Δt λ_k)| ≤ 1

    其中 R 是稳定性函数.

    Parameters
    ----------
    order : int
        差分格式阶数
    n_points : int
        网格点数
    cfl_max : float
        最大 CFL 数

    Returns
    -------
    result : dict
        spectral_radius: 谱半径
        stable_cfl: 稳定 CFL 范围
        max_imag_eigenvalue: 最大虚部特征值
    """
    # 构造二阶导数矩阵 (D²)
    h = 1.0 / (n_points + 1)
    x = np.linspace(0, 1, n_points + 2)

    # 使用 Fornberg 权重构造 D²
    stencil_width = min(order + 1, n_points)
    half = stencil_width // 2

    # 构造稀疏 D² 矩阵
    D2 = np.zeros((n_points, n_points))
    for i in range(n_points):
        # 内部点: 中心差分
        if half <= i < n_points - half:
            # 二阶标准格式
            D2[i, i] = -2.0 / h**2
            if i > 0:
                D2[i, i - 1] = 1.0 / h**2
            if i < n_points - 1:
                D2[i, i + 1] = 1.0 / h**2

            # 高阶修正 (4 阶)
            if order >= 4 and i > 1 and i < n_points - 2:
                D2[i, i - 2] -= 1.0 / (12.0 * h**2)
                D2[i, i + 2] -= 1.0 / (12.0 * h**2)
                D2[i, i - 1] += 16.0 / (12.0 * h**2) - 1.0 / h**2
                D2[i, i + 1] += 16.0 / (12.0 * h**2) - 1.0 / h**2
                D2[i, i] += -30.0 / (12.0 * h**2) + 2.0 / h**2

    # 计算特征值
    eigenvalues = np.linalg.eigvals(D2)

    # 谱半径
    spectral_radius = np.max(np.abs(eigenvalues))

    # 稳定 CFL (对于 RK4)
    # |R(z)| ≤ 1, 其中 z = Δt λ, R(z) = 1 + z + z²/2 + z³/6 + z⁴/24
    # 对于纯虚数 z = iy: |R(iy)|² = 1 - y⁴/12 + y⁶/72 - y⁸/576
    # 稳定条件: y < y_max ≈ 2.83
    y_max_rk4 = 2.83
    stable_dt = y_max_rk4 / max(np.sqrt(spectral_radius), 1e-10)

    return {
        "spectral_radius": spectral_radius,
        "max_eigenvalue": np.max(eigenvalues),
        "min_eigenvalue": np.min(eigenvalues),
        "max_imag_eigenvalue": np.max(np.abs(np.imag(eigenvalues))),
        "stable_dt": stable_dt,
        "stable_cfl": stable_dt / h,
        "n_points": n_points,
        "order": order,
    }
