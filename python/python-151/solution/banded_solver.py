"""
banded_solver.py
================
带状矩阵线性求解器与抛物型偏微分方程离散化工具

原项目映射:
- 987_r8pbl: 对称正定带状矩阵存储格式 (R8PBL) 与矩阵-向量乘法
- 1368_tumor_pde: 肿瘤PDE参数管理、边界条件、初始条件、系数函数定义

科学功能:
本模块实现了VQE中用于预处理矩阵元的带状矩阵运算，以及将
连续空间场论（如Gross-Pitaevskii型非线性薛定谔方程）离散化
为有限差分形式的工具，为后续变分量子本征求解提供经典基准。
"""

import numpy as np
from typing import Tuple, Optional, Callable


class PDEParameters:
    """
    管理PDE物理参数的持久化默认值，基于 tumor_parameters 思想。
    用于Gross-Pitaevskii方程或扩散-反应系统的参数存储。
    """
    _defaults = {
        'alpha': 10.0,      # 非线性反应系数
        'beta': 4.0,        # 衰减系数
        'cstar': 0.2,       # 临界阈值
        'delta': 1.0,       # 扩散系数
        'epsilon': 0.001,   # 小参数（奇异摄动）
        'gamma': 1.0,       # 饱和系数
        'k': 0.75,          # 对流耦合系数
        'lambda_': 1.0,     # 特征值偏移
        'mu': 100.0,        # 非线性强度
        't0': 0.0,
        'tstop': 0.7,
        'xmin': 0.0,
        'xmax': 1.0,
    }

    @classmethod
    def get_defaults(cls) -> dict:
        return cls._defaults.copy()

    @classmethod
    def update_defaults(cls, **kwargs):
        for k, v in kwargs.items():
            if k in cls._defaults:
                cls._defaults[k] = float(v)


def pde_coefficients(x: float, t: float, u: np.ndarray, dudx: np.ndarray,
                     params: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    定义抛物型PDE的系数函数 c, f, s，对应 tumor_fun。

    PDE形式: c(x,t,u,dudx) * du/dt = x^{-m} d/dx (x^m f) + s

    这里将其推广为耦合的Gross-Pitaevskii型方程组：
    c[0] = 1,  c[1] = 1
    f[0] = delta * dudx[0]                         (扩散)
    f[1] = epsilon * dudx[1] - k * u[1] * dudx[0]  (交叉扩散+对流)
    s[0] = -alpha*u[0]*u[1]/(gamma+u[0]) - lambda*u[1]
    s[1] = mu*u[1]*(1-u[1])*max(u[0]-cstar,0) - beta*u[1]

    参数:
        x: 空间坐标
        t: 时间坐标
        u: 解向量 (2,)
        dudx: 空间导数 (2,)
        params: 参数字典
    返回:
        c, f, s: 系数向量 (2,)
    """
    if params is None:
        params = PDEParameters.get_defaults()

    alpha = params.get('alpha', 10.0)
    beta = params.get('beta', 4.0)
    cstar = params.get('cstar', 0.2)
    delta = params.get('delta', 1.0)
    epsilon = params.get('epsilon', 0.001)
    gamma = params.get('gamma', 1.0)
    k = params.get('k', 0.75)
    lambda_ = params.get('lambda_', 1.0)
    mu = params.get('mu', 100.0)

    c = np.ones(2)
    f = np.array([delta * dudx[0],
                  epsilon * dudx[1] - k * u[1] * dudx[0]])
    g = max(u[0] - cstar, 0.0)
    s = np.array([
        -alpha * u[0] * u[1] / (gamma + u[0] + 1e-12) - lambda_ * u[1],
        mu * u[1] * (1.0 - u[1]) * g - beta * u[1]
    ])
    return c, f, s


def pde_boundary_conditions(xl: float, ul: np.ndarray,
                            xr: float, ur: np.ndarray,
                            t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    边界条件，对应 tumor_bc。
    Dirichlet-Neumann 混合边界:
        左边界: u[0]=1,  du[1]/dx=0  (即 flux[1]=0)
        右边界: u[0]=0,  u[1]=1
    """
    pl = np.array([ul[0] - 1.0, 0.0])
    ql = np.array([0.0, 1.0])
    pr = np.array([ur[0], ur[1] - 1.0])
    qr = np.array([0.0, 0.0])
    return pl, ql, pr, qr


def pde_initial_condition(x: float) -> np.ndarray:
    """
    初始条件，对应 tumor_ic。
    u0[0] = cos(pi*x/2),  u0[1] = 0 (x<1) else 1
    """
    c0 = np.cos(0.5 * np.pi * x)
    n0 = 0.0 if x < 1.0 else 1.0
    return np.array([c0, n0])


class BandedMatrix:
    """
    对称正定带状矩阵 (R8PBL) 的Python实现，基于 987_r8pbl。
    存储格式: a[ml+1, n]，其中 a[0,:] 为对角线，a[k+1, 0:n-k-1] 为第k次对角线。
    """
    def __init__(self, n: int, ml: int, data: Optional[np.ndarray] = None):
        if n <= 0:
            raise ValueError("矩阵阶数 n 必须为正整数")
        if ml < 0 or ml > n - 1:
            raise ValueError("半带宽 ml 必须在 [0, n-1] 范围内")
        self.n = n
        self.ml = ml
        if data is None:
            self.a = np.zeros((ml + 1, n))
        else:
            if data.shape != (ml + 1, n):
                raise ValueError(f"data形状应为({ml+1},{n})，实际为{data.shape}")
            self.a = data.astype(float).copy()

    @classmethod
    def dif2(cls, n: int, ml: int = 1) -> 'BandedMatrix':
        """
        构造DIF2三对角矩阵（有限差分拉普拉斯离散化）。
        特征值: lambda_i = 2 + 2*cos(i*pi/(n+1)) = 4*sin^2(i*pi/(2n+2))
        对应特征向量: X_i(j) = sqrt(2/(n+1)) * sin(i*j*pi/(n+1))
        """
        bm = cls(n, ml)
        bm.a[0, :] = 2.0
        if ml >= 1 and n > 1:
            bm.a[1, :n - 1] = -1.0
        return bm

    def mv(self, x: np.ndarray) -> np.ndarray:
        """
        矩阵-向量乘法 y = A @ x，利用带状结构优化，O(n*ml)。
        对应 r8pbl_mv。
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != self.n:
            raise ValueError(f"向量维度{x.shape[0]}不等于矩阵阶数{self.n}")
        b = self.a[0, :] * x
        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                i = j + k
                aij = self.a[k, j]
                b[i] += aij * x[j]
                b[j] += aij * x[i]
        return b

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵，用于验证或小规模计算。"""
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            A[i, i] = self.a[0, i]
        for k in range(1, self.ml + 1):
            for j in range(self.n - k):
                i = j + k
                A[i, j] = self.a[k, j]
                A[j, i] = self.a[k, j]
        return A

    def cholesky_band(self) -> np.ndarray:
        """
        带状Cholesky分解 A = L @ L.T。
        DIF2矩阵的Cholesky因子满足:
            L(i,i)   = sqrt((i+1)/i)
            L(i,i-1) = -sqrt((i-1)/i)
        """
        L = np.zeros((self.ml + 1, self.n))
        for j in range(self.n):
            # 计算L[j,j]
            sum_sq = self.a[0, j]
            for k in range(1, self.ml + 1):
                if j - k >= 0:
                    sum_sq -= L[k, j - k] ** 2
            if sum_sq <= 0:
                raise ValueError("矩阵非正定，Cholesky分解失败")
            L[0, j] = np.sqrt(sum_sq)
            # 计算次对角线元素
            for i in range(j + 1, min(j + self.ml + 1, self.n)):
                k = i - j
                s = self.a[k, j]
                for m in range(1, self.ml + 1):
                    if j - m >= 0 and k - m >= 0:
                        s -= L[m, j - m] * L[k - m, j - m + k - m]
                # 简化: 仅考虑直接相邻贡献
                if k == 1 and j + 1 < self.n:
                    s = self.a[1, j]
                    if j > 0:
                        s -= L[1, j - 1] * L[0, j - 1]
                L[k, j] = s / L[0, j]
        return L

    def solve_cholesky(self, b: np.ndarray) -> np.ndarray:
        """使用带状Cholesky分解求解 A x = b。"""
        L = self.cholesky_band()
        y = np.zeros(self.n)
        # 前向替代 L y = b
        for i in range(self.n):
            y[i] = b[i]
            for k in range(1, self.ml + 1):
                if i - k >= 0:
                    y[i] -= L[k, i - k] * y[i - k]
            y[i] /= L[0, i]
        # 后向替代 L.T x = y
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = y[i]
            for k in range(1, self.ml + 1):
                if i + k < self.n:
                    x[i] -= L[k, i] * x[i + k]
            x[i] /= L[0, i]
        return x

    def eigenvalues_dif2(self) -> np.ndarray:
        """
        DIF2矩阵的解析特征值:
            lambda_i = 4 * sin^2(i*pi / (2*(n+1)))
        """
        i = np.arange(1, self.n + 1)
        return 4.0 * np.sin(i * np.pi / (2.0 * (self.n + 1))) ** 2

    def eigenvector_dif2(self, idx: int) -> np.ndarray:
        """
        DIF2矩阵的第idx个解析特征向量（1-based）:
            X_j = sqrt(2/(n+1)) * sin(idx * j * pi / (n+1))
        """
        j = np.arange(1, self.n + 1)
        return np.sqrt(2.0 / (self.n + 1)) * np.sin(idx * j * np.pi / (self.n + 1))


def finite_difference_discretize(n: int, xl: float, xr: float,
                                  pde_func: Optional[Callable] = None,
                                  params: Optional[dict] = None) -> Tuple[BandedMatrix, np.ndarray]:
    """
    将一维抛物型PDE离散化为带状线性系统 A u = f。
    使用中心差分格式，空间步长 h = (xr - xl) / (n + 1)。

    对于稳态问题: -delta * d^2u/dx^2 + V(x) u = f(x)
    离散化后: -(delta/h^2) * (u_{i-1} - 2u_i + u_{i+1}) + V_i u_i = f_i

    返回:
        A: BandedMatrix (ml=1)
        f: 右端项
    """
    if n <= 2:
        raise ValueError("离散化点数n必须大于2")
    h = (xr - xl) / (n + 1)
    if h <= 0:
        raise ValueError("区间长度必须为正")

    A = BandedMatrix.dif2(n, ml=1)
    delta = 1.0 if params is None else params.get('delta', 1.0)
    coeff = delta / (h ** 2)
    A.a[0, :] *= coeff
    A.a[1, :] *= coeff

    # 添加势能项（对角质量矩阵）
    x_grid = np.linspace(xl + h, xr - h, n)
    if pde_func is not None:
        for i, xi in enumerate(x_grid):
            _, _, s = pde_func(xi, 0.0, np.zeros(2), np.zeros(2), params)
            # 简化为标量势能: V = |s[0]| 作为正则化势能
            V = abs(s[0]) * 0.01
            A.a[0, i] += V
    else:
        # 默认谐振子势能 V(x) = 0.5 * x^2
        V = 0.5 * x_grid ** 2
        A.a[0, :] += V

    # 右端项
    f = np.ones(n) * h  # 简单源项
    return A, f


def solve_steady_pde(n: int = 128, xl: float = 0.0, xr: float = 1.0) -> np.ndarray:
    """
    求解稳态PDE作为VQE的经典基准验证。
    边界条件通过修改右端项实现（简支）。
    """
    params = PDEParameters.get_defaults()
    A, f = finite_difference_discretize(n, xl, xr, pde_coefficients, params)
    # 修正边界条件（Dirichlet）
    pl, ql, pr, qr = pde_boundary_conditions(xl, np.array([1.0, 0.0]),
                                             xr, np.array([0.0, 1.0]), 0.0)
    # 将边界值移入右端项
    h = (xr - xl) / (n + 1)
    coeff = params['delta'] / (h ** 2)
    f[0] -= coeff * 1.0   # 左边界 u=1
    f[-1] -= coeff * 0.0  # 右边界 u=0
    u = A.solve_cholesky(f)
    return u
