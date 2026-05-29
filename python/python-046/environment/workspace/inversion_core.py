"""
inversion_core.py
断层滑动分布反演核心模块：优化与参数估计。

融合种子项目:
  - 797_nelder_mead: Nelder-Mead 单纯形优化算法

在 InSAR 形变反演中的应用:
  1. 构造目标泛函（数据拟合 + 正则化）；
  2. 使用 Nelder-Mead 方法优化非线性目标函数（如 L1 正则化、摩擦参数联合反演）；
  3. 与线性反演（Tikhonov）形成互补，用于验证全局最优性。

目标泛函:
    J(m) = 1/2 || W^{1/2} (G m - d) ||_2^2 + λ R(m) + γ ||m||_1
    R(m): Tikhonov 平滑项 = 1/2 ||L m||_2^2
"""

import numpy as np
from regularization import tikhonov_solve, build_laplacian_2d
from utils import check_finite


class NelderMeadOptimizer:
    """
    Nelder-Mead 单纯形优化器（基于种子项目 797_nelder_mead 的算法）。

    算法参数:
        ρ (rho):   反射系数 = 1.0
        χ (chi):   扩展系数 = 2.0
        γ (gamma): 收缩系数 = 0.5
        σ (sigma): 缩小系数 = 0.5
    """

    def __init__(self, rho=1.0, chi=2.0, gamma=0.5, sigma=0.5,
                 tol=1e-6, max_iter=500):
        self.rho = rho
        self.chi = chi
        self.gamma = gamma
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, objective_func, x0):
        """
        对目标函数 objective_func 进行 Nelder-Mead 优化。

        参数:
            objective_func: callable, f(x) -> float
            x0: (n_dim,) 或 (n_dim+1, n_dim) 初始单纯形

        返回:
            x_opt: 最优解
            f_opt: 最优函数值
            n_eval: 函数评估次数
        """
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 1:
            n_dim = len(x0)
            # 构造初始单纯形
            simplex = np.zeros((n_dim + 1, n_dim))
            simplex[0] = x0
            for i in range(n_dim):
                simplex[i + 1] = x0.copy()
                if x0[i] != 0:
                    simplex[i + 1, i] *= 1.05
                else:
                    simplex[i + 1, i] = 0.05
        else:
            simplex = x0.copy()
            n_dim = simplex.shape[1]

        # 评估初始单纯形
        f_vals = np.array([objective_func(simplex[i]) for i in range(n_dim + 1)])
        n_eval = n_dim + 1

        for iteration in range(self.max_iter):
            # 排序
            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]

            # 收敛检查
            if f_vals[-1] - f_vals[0] < self.tol:
                break

            # 形心（排除最差点的其余点的平均）
            x_bar = np.mean(simplex[:-1], axis=0)

            # 反射点
            x_r = (1.0 + self.rho) * x_bar - self.rho * simplex[-1]
            f_r = objective_func(x_r)
            n_eval += 1

            if f_vals[0] <= f_r < f_vals[-2]:
                # 反射点可接受
                simplex[-1] = x_r
                f_vals[-1] = f_r
            elif f_r < f_vals[0]:
                # 扩展
                x_e = (1.0 + self.rho * self.chi) * x_bar - self.rho * self.chi * simplex[-1]
                f_e = objective_func(x_e)
                n_eval += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_vals[-2] <= f_r < f_vals[-1]:
                # 外收缩
                x_c = (1.0 + self.rho * self.gamma) * x_bar - self.rho * self.gamma * simplex[-1]
                f_c = objective_func(x_c)
                n_eval += 1
                if f_c <= f_r:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    simplex, f_vals = self._shrink(simplex, f_vals, objective_func)
                    n_eval += n_dim
            else:
                # 内收缩
                x_c = (1.0 - self.gamma) * x_bar + self.gamma * simplex[-1]
                f_c = objective_func(x_c)
                n_eval += 1
                if f_c < f_vals[-1]:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    simplex, f_vals = self._shrink(simplex, f_vals, objective_func)
                    n_eval += n_dim

        return simplex[0], f_vals[0], n_eval

    def _shrink(self, simplex, f_vals, objective_func):
        """
        向最优点收缩整个单纯形。
        """
        n_dim = simplex.shape[1]
        for i in range(1, n_dim + 1):
            simplex[i] = self.sigma * simplex[i] + (1.0 - self.sigma) * simplex[0]
            f_vals[i] = objective_func(simplex[i])
        return simplex, f_vals


class FaultSlipInversion:
    """
    断层滑动分布反演器。
    支持线性 Tikhonov 反演和非线性 Nelder-Mead 联合优化。
    """

    def __init__(self, G, W, d, lam, L=None):
        """
        参数:
            G: 格林函数矩阵 (M × N)
            W: 数据权重矩阵 (M × M)
            d: 观测数据 (M,)
            lam: 正则化参数
            L: 正则化算子 (K × N)，若为 None 则构造二维 Laplacian
        """
        self.G = np.asarray(G, dtype=float)
        self.W = np.asarray(W, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.lam = lam
        self.M, self.N = self.G.shape

        if L is None:
            # 自动构造 Laplacian（假设 N 是 nx * ny）
            nx = int(np.sqrt(self.N))
            ny = nx
            if nx * ny != self.N:
                # 若 N 不是完全平方，构造一维 Laplacian
                from regularization import build_laplacian_1d
                self.L = build_laplacian_1d(self.N)
            else:
                self.L = build_laplacian_2d(nx, ny)
        else:
            self.L = np.asarray(L, dtype=float)

    def linear_inversion(self):
        """
        线性 Tikhonov 正则化反演。
        """
        # HOLE 2: 需调用 regularization 模块的 tikhonov_solve 完成线性反演。
        # 注意：必须正确传递 G, W, d, lam, L 五个参数，并返回 (m, cov)。
        # 仅修复此文件而不修复 regularization.py 中的 tikhonov_solve，
        # 反演流程仍无法完成。
        raise NotImplementedError("linear_inversion: 待实现线性 Tikhonov 反演调用")

    def nonlinear_l1_inversion(self, m0=None, gamma=0.01):
        """
        L1 正则化非线性反演，使用 Nelder-Mead 优化。
        目标函数:
            J(m) = 1/2 ||W^{1/2}(Gm - d)||_2^2 + λ^2/2 ||Lm||_2^2 + γ ||m||_1

        参数:
            m0: 初始猜测
            gamma: L1 权重
        """
        if m0 is None:
            m0 = np.zeros(self.N)

        W_sqrt = np.sqrt(np.diag(self.W)) if self.W.ndim == 2 else np.sqrt(self.W)

        def objective(m):
            residual = self.G @ m - self.d
            data_fit = 0.5 * np.sum((W_sqrt * residual) ** 2)
            reg_tik = 0.5 * (self.lam ** 2) * np.sum((self.L @ m) ** 2)
            reg_l1 = gamma * np.sum(np.abs(m))
            return data_fit + reg_tik + reg_l1

        optimizer = NelderMeadOptimizer(tol=1e-5, max_iter=800)
        m_opt, f_opt, n_eval = optimizer.optimize(objective, m0)
        check_finite(m_opt, "nonlinear_l1_inversion m_opt")
        return m_opt, f_opt, n_eval

    def compute_misfit(self, m):
        """
        计算数据拟合残差。
        """
        residual = self.G @ m - self.d
        misfit = np.sqrt(np.mean(residual ** 2))
        return misfit

    def compute_model_norm(self, m):
        """
        计算模型范数 ||L m||。
        """
        return np.linalg.norm(self.L @ m)

    def jackknife_uncertainty(self, m):
        """
        刀切法 (Jackknife) 估计滑动分布的不确定性。
        每次删除一个观测点，重新反演，计算滑动分布的标准差。
        """
        m_jack = np.zeros((self.M, self.N))
        for i in range(self.M):
            # 删除第 i 个观测
            G_i = np.delete(self.G, i, axis=0)
            d_i = np.delete(self.d, i)
            W_i = np.delete(np.delete(self.W, i, axis=0), i, axis=1)
            inv_i = FaultSlipInversion(G_i, W_i, d_i, self.lam, self.L)
            m_i, _ = inv_i.linear_inversion()
            m_jack[i] = m_i

        m_mean = np.mean(m_jack, axis=0)
        m_std = np.sqrt((self.M - 1) / self.M * np.sum((m_jack - m_mean) ** 2, axis=0))
        return m_std
