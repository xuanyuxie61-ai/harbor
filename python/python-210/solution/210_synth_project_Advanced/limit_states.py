"""
limit_states.py - 极限状态函数库：结构失效边界的数学描述

在可靠性分析中，极限状态函数 g(u) 将随机空间划分为安全域 (g > 0)
与失效域 (g <= 0)。失效概率 P_f = P(g(u) <= 0)。

本模块实现多种经典与前沿的极限状态函数，来自结构可靠性文献，
每个函数均支持梯度与Hessian的解析计算（用于FORM/SORM）。

种子项目映射:
  345_exm (predprey) → 非线性动力学失效面
  468_geometry         → 几何距离类失效面
  855_pdflib           → 应力-强度干涉失效面
"""

import numpy as np


class LimitStateFunction:
    """极限状态函数基类（标准正态空间 u ∈ R^n）

    约定:
      g(u) > 0  → 安全域
      g(u) = 0  → 极限状态面 (失效边界)
      g(u) < 0  → 失效域

    所有函数接受 u: np.ndarray, shape (..., n_dim) 并返回 shape (...) 的值。
    """

    def __init__(self, n_dim):
        self.n_dim = n_dim

    def evaluate(self, u):
        raise NotImplementedError

    def gradient(self, u):
        """返回梯度向量, shape (n_dim,) 对单个点"""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        n = len(u)
        g0 = self.evaluate(u)
        grad = np.zeros(n)
        h = 1e-6
        for i in range(n):
            u_p = u.copy(); u_p[i] += h
            u_m = u.copy(); u_m[i] -= h
            grad[i] = (self.evaluate(u_p) - self.evaluate(u_m)) / (2.0 * h)
        return grad

    def hessian(self, u):
        """返回Hessian矩阵, shape (n_dim, n_dim) 对单个点"""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        n = len(u)
        h = 1e-5
        H = np.zeros((n, n))
        g0 = self.evaluate(u)
        for i in range(n):
            for j in range(i, n):
                u_pp = u.copy(); u_pp[i] += h; u_pp[j] += h
                u_pm = u.copy(); u_pm[i] += h; u_pm[j] -= h
                u_mp = u.copy(); u_mp[i] -= h; u_mp[j] += h
                u_mm = u.copy(); u_mm[i] -= h; u_mm[j] -= h
                H[i, j] = (self.evaluate(u_pp) - self.evaluate(u_pm)
                           - self.evaluate(u_mp) + self.evaluate(u_mm)) / (4.0 * h * h)
                H[j, i] = H[i, j]
        return H

    def __call__(self, u):
        return self.evaluate(u)


class LinearLimitState(LimitStateFunction):
    """线性极限状态: g(u) = c_0 - sum(a_i * u_i)

    这是最基础的可靠性基准问题。
    精确解: beta = c_0 / ||a||_2
    """

    def __init__(self, a, c0=3.0):
        a = np.atleast_1d(np.asarray(a, dtype=float))
        super().__init__(len(a))
        self.a = a
        self.c0 = float(c0)

    def evaluate(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return self.c0 - u @ self.a

    def gradient(self, u):
        return -self.a.copy()

    def hessian(self, u):
        return np.zeros((self.n_dim, self.n_dim))

    def exact_beta(self):
        return self.c0 / np.linalg.norm(self.a)


class NonlinearLimitState(LimitStateFunction):
    """非线性极限状态 (Der Kiureghian & Stefano, 1996)

    g(u) = c - u_2 - kappa * (u_1 - lambda)^2

    精确FORM解:
    beta = (c - kappa*lambda^2) / sqrt(1 + (2*kappa*lambda)^2) ... 需迭代

    这是一个经典的非线性FORM测试问题。
    """

    def __init__(self, c0=5.0, lam=2.0, kappa=0.5):
        super().__init__(2)
        self.c0 = float(c0)
        self.lam = float(lam)
        self.kappa = float(kappa)

    def evaluate(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return self.c0 - u[:, 1] - self.kappa * (u[:, 0] - self.lam) ** 2

    def gradient(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        return np.array([
            -2.0 * self.kappa * (u[0] - self.lam),
            -1.0
        ])

    def hessian(self, u):
        H = np.zeros((2, 2))
        H[0, 0] = -2.0 * self.kappa
        return H


class QuarticLimitState(LimitStateFunction):
    """四次极限状态: g(u) = c - u_1^2 - u_2^2 - u_3^2 (3D)

    失效域为球外区域。精确 beta = sqrt(c).
    """

    def __init__(self, c0=5.0):
        super().__init__(3)
        self.c0 = float(c0)

    def evaluate(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return self.c0 - np.sum(u ** 2, axis=-1)

    def gradient(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        return -2.0 * u

    def hessian(self, u):
        return -2.0 * np.eye(self.n_dim)

    def exact_beta(self):
        return np.sqrt(self.c0)


class SeriesSystemLimitState(LimitStateFunction):
    """串联系统失效: g(u) = min(g_1(u), g_2(u))

    系统失效 = 任一分支失效 (weakest link).
    近似为 P-F 平滑: g ≈ -log(exp(-beta1*g1) + exp(-beta1*g2)) / beta1
    """

    def __init__(self, n_dim=2, a1=None, a2=None, c1=3.0, c2=4.0, beta_smooth=5.0):
        super().__init__(n_dim)
        self.a1 = np.atleast_1d(np.asarray(a1 or np.ones(n_dim), dtype=float))
        self.a2 = np.atleast_1d(np.asarray(a2 or np.array([-1.0] + [1.0] * (n_dim - 1)), dtype=float))
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.beta_smooth = float(beta_smooth)

    def _g1(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return self.c1 - u @ self.a1

    def _g2(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return self.c2 - u @ self.a2

    def evaluate(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        g1 = self._g1(u)
        g2 = self._g2(u)
        # P-F 平滑近似 min: -log(exp(-b*g1)+exp(-b*g2))/b
        b = self.beta_smooth
        mg1 = -b * g1
        mg2 = -b * g2
        m_max = np.maximum(mg1, mg2)
        return -(m_max + np.log(np.exp(mg1 - m_max) + np.exp(mg2 - m_max))) / b


class StressStrengthLimitState(LimitStateFunction):
    """应力-强度干涉模型 (2D):

    g(u) = R(u) - S(u)
    R = mu_R + sigma_R * u_1    (强度, resistance)
    S = mu_S + sigma_S * u_2    (应力, load)

    精确: beta = (mu_R - mu_S) / sqrt(sigma_R^2 + sigma_S^2)
    """

    def __init__(self, mu_R=10.0, sigma_R=2.0, mu_S=6.0, sigma_S=1.5):
        super().__init__(2)
        self.mu_R = float(mu_R)
        self.sigma_R = float(sigma_R)
        self.mu_S = float(mu_S)
        self.sigma_S = float(sigma_S)

    def evaluate(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        R = self.mu_R + self.sigma_R * u[:, 0]
        S = self.mu_S + self.sigma_S * u[:, 1]
        return R - S

    def gradient(self, u):
        return np.array([self.sigma_R, -self.sigma_S])

    def hessian(self, u):
        return np.zeros((2, 2))

    def exact_beta(self):
        return (self.mu_R - self.mu_S) / np.sqrt(self.sigma_R ** 2 + self.sigma_S ** 2)


class ExponentialCosLimitState(LimitStateFunction):
    """指数-余弦极限状态 (高非线性, 可扩展至任意维):

    g(u) = exp(-0.5*(u_1+u_2)) + 0.5*cos(2*u_1) - 0.3*sin(3*u_2)
           - sum_{i>=3} c_i * u_i + c

    用于测试FORM在高维非线性问题上的表现。
    """

    def __init__(self, n_dim=4, c0=1.5):
        super().__init__(n_dim)
        self.c0 = float(c0)
        # 低维系数
        self._c = [0.1, 0.05, 0.02, 0.01, 0.005]

    def evaluate(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        val = (np.exp(-0.5 * (u[:, 0] + u[:, 1]))
               + 0.5 * np.cos(2.0 * u[:, 0])
               - 0.3 * np.sin(3.0 * u[:, 1])
               + self.c0)
        for i in range(2, self.n_dim):
            ci = self._c[min(i - 2, len(self._c) - 1)]
            val = val - ci * u[:, i]
        return val

    def gradient(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        e_term = np.exp(-0.5 * (u[0] + u[1]))
        grad = np.zeros(self.n_dim)
        grad[0] = -0.5 * e_term - np.sin(2.0 * u[0])
        grad[1] = -0.5 * e_term - 0.9 * np.cos(3.0 * u[1])
        for i in range(2, self.n_dim):
            grad[i] = -self._c[min(i - 2, len(self._c) - 1)]
        return grad

    def hessian(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        e_term = np.exp(-0.5 * (u[0] + u[1]))
        H = np.zeros((self.n_dim, self.n_dim))
        H[0, 0] = 0.25 * e_term - 2.0 * np.cos(2.0 * u[0])
        H[1, 1] = 0.25 * e_term + 2.7 * np.sin(3.0 * u[1])
        H[0, 1] = 0.25 * e_term
        H[1, 0] = H[0, 1]
        return H
