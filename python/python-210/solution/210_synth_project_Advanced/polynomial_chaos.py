"""
polynomial_chaos.py - 多项式混沌展开 (PCE)

基于Chebyshev逼近思想构建随机响应的多项式混沌代理模型。
PCE将随机输出表示为输入随机变量的正交多项式级数:

  Y(u) ≈ sum_{|alpha|<=p} c_alpha * Phi_alpha(u)

其中 Phi_alpha 为多变量正交多项式基（本模块采用张量积Legendre基），
c_alpha 为展开系数，通过非侵入配点法确定。

Chebyshev节点用于配点，以避免Runge现象：
  x_k = cos((2k+1)*pi / (2N)),  k=0,...,N-1

核心公式 (PCE矩):
  E[Y] = c_0
  Var[Y] = sum_{alpha != 0} c_alpha^2 * <Phi_alpha, Phi_alpha>
  Sobol指数: S_i = Var_i / Var[Y]

种子项目映射:
  014_approx_chebyshev → Chebyshev配点 + 差商插值 → PCE基函数构建
"""

import numpy as np
from itertools import product as iterproduct
from math import comb


def chebyshev_nodes(a, b, n):
    """在 [a,b] 上生成 n 个 Chebyshev 节点 (第一类)

    x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)*pi/(2n))
    这些节点最小化插值多项式的 Lebesgue 常数。
    """
    if n < 1:
        return np.array([(a + b) / 2.0])
    k = np.arange(n)
    theta = (2.0 * k + 1.0) * np.pi / (2.0 * n)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos(theta)


def chebyshev_weights(n):
    """Chebyshev-Gauss 求积权重

    w_k = pi / n
    """
    return np.full(n, np.pi / n)


def divided_differences(x, y):
    """计算差商表 (Divided Difference Table)

    f[x_0,...,x_k] = (f[x_1,...,x_k] - f[x_0,...,x_{k-1}]) / (x_k - x_0)

    返回差商向量 (Newton插值系数).
    种子项目 014_approx_chebyshev 中 divdif() 的直接Python翻译。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    yd = y.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            denom = x[j] - x[j - i]
            if abs(denom) < 1e-15:
                yd[j] = 0.0
            else:
                yd[j] = (yd[j] - yd[j - 1]) / denom
    return yd


def newton_interp_eval(xd, yd, xp):
    """Newton 差商形式插值多项式求值

    P(x) = d_0 + d_1*(x-x_0) + d_2*(x-x_0)*(x-x_1) + ...
    使用 Horner 式从后向前递推。
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    xp = np.asarray(xp, dtype=float)
    nd = len(xd)
    yp = np.full_like(xp, yd[nd - 1], dtype=float)
    for i in range(nd - 2, -1, -1):
        yp = yd[i] + (xp - xd[i]) * yp
    return yp


def multi_index_set(n_dim, degree):
    """生成全阶数 <= degree 的多重指标集合

    {alpha in N^n : |alpha| = sum(alpha_i) <= degree}

    这是PCE展开的多项式基指标集。
    总项数 = C(n_dim + degree, degree)
    """
    if n_dim == 0:
        return [()]
    indices = []
    for alpha in iterproduct(range(degree + 1), repeat=n_dim):
        if sum(alpha) <= degree:
            indices.append(alpha)
    return indices


def legendre_basis_1d(x, max_degree):
    """计算一维Legendre多项式 T_0(x), T_1(x), ..., T_{max_degree}(x)

    三点递推:
      T_0(x) = 1
      T_1(x) = x
      (k+1)*T_{k+1}(x) = (2k+1)*x*T_k(x) - k*T_{k-1}(x)

    正交性: int_{-1}^{1} T_i(x)*T_j(x) dx = 2/(2i+1) * delta_{ij}
    """
    x = np.asarray(x, dtype=float)
    T = [np.ones_like(x)]
    if max_degree >= 1:
        T.append(x.copy())
    for k in range(1, max_degree):
        T_next = ((2.0 * k + 1.0) * x * T[k] - k * T[k - 1]) / (k + 1.0)
        T.append(T_next)
    return T


def legendre_eval_multi(alpha, u):
    """计算多变量 Legendre 基函数 Phi_alpha(u)

    Phi_alpha(u) = prod_{i=1}^{d} T_{alpha_i}(u_i)
    其中 T_k 为 k 阶 Legendre 多项式。
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n_samples = u.shape[0]
    n_dim = u.shape[1]
    result = np.ones(n_samples)
    for i in range(n_dim):
        if alpha[i] > 0:
            T = legendre_basis_1d(u[:, i], alpha[i])
            result *= T[alpha[i]]
    return result


class PolynomialChaosExpansion:
    """多项式混沌展开 (PCE) 代理模型

    通过非侵入配点法构建:
    1. 在 Chebyshev 节点上采样目标函数
    2. 用最小二乘拟合 PCE 系数
    3. 代理模型可用于快速 Monte Carlo 和灵敏度分析

    核心公式:
      Y_hat(u) = sum_{alpha} c_alpha * Phi_alpha(u)
      E[Y] ≈ c_0 (对于正交归一基)
      Var[Y] ≈ sum_{alpha!=0} c_alpha^2 * ||Phi_alpha||^2
    """

    def __init__(self, n_dim, degree):
        self.n_dim = n_dim
        self.degree = degree
        self.indices = multi_index_set(n_dim, degree)
        self.n_terms = len(self.indices)
        self.coefficients = None

    def _build_basis_matrix(self, u):
        """构建基函数矩阵 B[i,j] = Phi_{alpha_j}(u_i)"""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        n_samples = u.shape[0]
        B = np.zeros((n_samples, self.n_terms))
        for j, alpha in enumerate(self.indices):
            B[:, j] = legendre_eval_multi(alpha, u)
        return B

    def fit_collocation(self, func, n_quad=None):
        """通过 Chebyshev 配点 + 最小二乘拟合 PCE 系数

        Parameters
        ----------
        func : callable, 接受 (n_samples, n_dim) 的数组
        n_quad : int, 每个维度的配点数 (默认 = degree + 2)
        """
        if n_quad is None:
            n_quad = max(self.degree + 2, 3)

        # 每个维度生成 Chebyshev 节点 (在 [-3, 3] 上，覆盖 ±3 sigma)
        nodes_1d = chebyshev_nodes(-3.0, 3.0, n_quad)

        # 张量积节点
        grids = [nodes_1d for _ in range(self.n_dim)]
        u_quad = np.array(list(iterproduct(*grids)))
        n_total = u_quad.shape[0]

        if n_total < self.n_terms:
            n_quad = max(n_quad, int(np.ceil(self.n_terms ** (1.0 / self.n_dim))) + 1)
            nodes_1d = chebyshev_nodes(-3.0, 3.0, n_quad)
            grids = [nodes_1d for _ in range(self.n_dim)]
            u_quad = np.array(list(iterproduct(*grids)))
            n_total = u_quad.shape[0]

        f_vals = np.array([func(u_quad[i]) for i in range(n_total)]).ravel()

        # 去除 NaN/Inf
        mask = np.isfinite(f_vals)
        if np.sum(mask) < self.n_terms:
            # 退化情况：使用零系数
            self.coefficients = np.zeros(self.n_terms)
            if np.sum(mask) > 0:
                self.coefficients[0] = np.mean(f_vals[mask])
            return

        B = self._build_basis_matrix(u_quad[mask])
        try:
            self.coefficients, _, _, _ = np.linalg.lstsq(B, f_vals[mask], rcond=None)
        except np.linalg.LinAlgError:
            self.coefficients = np.zeros(self.n_terms)
            self.coefficients[0] = np.mean(f_vals[mask])

        # 数值清理
        self.coefficients[~np.isfinite(self.coefficients)] = 0.0

    def predict(self, u):
        """代理模型预测: Y_hat(u) = sum c_alpha * Phi_alpha(u)"""
        if self.coefficients is None:
            raise RuntimeError("PCE 尚未拟合，请先调用 fit_collocation()")
        u = np.atleast_2d(np.asarray(u, dtype=float))
        B = self._build_basis_matrix(u)
        return B @ self.coefficients

    def mean(self):
        """PCE 均值: E[Y] ≈ c_0"""
        if self.coefficients is None:
            return 0.0
        return float(self.coefficients[0])

    def variance(self):
        """PCE 方差 (解析)

        Var[Y] = sum_{alpha != 0} c_alpha^2 * prod_i (1/(2*alpha_i+1))
        其中 ||Phi_alpha||^2 = prod_i 2/(2*alpha_i+1),  归一化后为 prod_i 1/(2*alpha_i+1)
        """
        if self.coefficients is None:
            return 0.0
        var = 0.0
        for j, alpha in enumerate(self.indices):
            if j == 0:
                continue
            norm_sq = 1.0
            for ai in alpha:
                norm_sq *= 1.0 / (2.0 * ai + 1.0)
            var += self.coefficients[j] ** 2 * norm_sq
        return max(var, 0.0)

    def sobol_indices(self):
        """计算一阶 Sobol 灵敏度指数 (解析, 从PCE系数)

        S_i = Var_i / Var[Y]
        其中 Var_i = sum_{alpha: alpha_i>0, alpha_j=0 for j!=i} c_alpha^2 * ||Phi_alpha||^2
        """
        total_var = self.variance()
        if total_var < 1e-30:
            return np.zeros(self.n_dim)

        S = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            var_i = 0.0
            for j, alpha in enumerate(self.indices):
                if j == 0:
                    continue
                if alpha[i] > 0 and all(alpha[k] == 0 for k in range(self.n_dim) if k != i):
                    norm_sq = 1.0
                    for ai in alpha:
                        norm_sq *= 1.0 / (2.0 * ai + 1.0)
                    var_i += self.coefficients[j] ** 2 * norm_sq
            S[i] = var_i / total_var
        return np.clip(S, 0.0, 1.0)

    def total_sobol_indices(self):
        """计算全阶 Sobol 指数 (解析)

        S_Ti = 1 - Var_{~i} / Var[Y]
        Var_{~i} = sum_{alpha: alpha_i=0} c_alpha^2 * ||Phi_alpha||^2
        """
        total_var = self.variance()
        if total_var < 1e-30:
            return np.zeros(self.n_dim)

        ST = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            var_not_i = 0.0
            for j, alpha in enumerate(self.indices):
                if j == 0:
                    continue
                if alpha[i] == 0:
                    norm_sq = 1.0
                    for ai in alpha:
                        norm_sq *= 1.0 / (2.0 * ai + 1.0)
                    var_not_i += self.coefficients[j] ** 2 * norm_sq
            ST[i] = 1.0 - var_not_i / total_var
        return np.clip(ST, 0.0, 1.0)
