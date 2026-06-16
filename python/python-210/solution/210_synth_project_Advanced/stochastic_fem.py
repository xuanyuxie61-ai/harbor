"""
stochastic_fem.py - 随机有限元：Karhunen-Loève展开与随机反应扩散

本模块实现空间随机场的离散化与随机偏微分方程的求解，
用于结构可靠性分析中材料属性的不确定性传播。

Karhunen-Loève (KL) 展开:
  E[x, theta] = mu(x) + sum_{i=1}^M sqrt(lambda_i) * phi_i(x) * xi_i(theta)
  其中 lambda_i, phi_i 为协方差核的特征值/特征函数,
  xi_i 为不相关随机变量。

指数型协方差核:
  C(x1, x2) = sigma^2 * exp(-|x1 - x2| / l_c)
  其中 l_c 为相关长度。

FEM 离散化 (种子项目 377_fem_neumann):
  M * dw/dt = -K * w + NL(w, c)
  其中 M 为质量矩阵, K 为刚度矩阵, NL 为非线性反应项。
  质量矩阵: M_ij = int phi_i * phi_j dx
  刚度矩阵: K_ij = int grad(phi_i) . grad(phi_j) dx

种子项目映射:
  377_fem_neumann → FEM 质量/刚度矩阵 + 反应扩散 ODE
  820_numgrid     → 空间网格生成
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal


class KarhunenLoeveExpansion:
    """Karhunen-Loève 展开: 随机场的低维参数化

    对均匀一维域 [0, L] 上的平稳随机场,
    指数协方差核 C(x1,x2) = sigma^2 * exp(-|x1-x2|/l_c)
    的 KL 展开可通过求解积分方程的特征值问题得到。

    近似解: 使用 Galerkin 截断, 保留前 M 项。
    特征值近似: lambda_i ≈ 2*sigma^2*l_c / (1 + (i*pi*l_c/L)^2)
    """

    def __init__(self, n_elements, domain_length, sigma_field, correlation_length, n_terms=None):
        self.n_elements = int(n_elements)
        self.L = float(domain_length)
        self.sigma = float(sigma_field)
        self.lc = float(correlation_length)
        self.n_terms = n_terms or min(20, n_elements)
        self.h = self.L / self.n_elements
        self.x_nodes = np.linspace(0, self.L, self.n_elements + 1)

        # 计算 KL 特征对
        self._compute_eigenpairs()

    def _compute_eigenpairs(self):
        """计算 KL 特征值和特征函数

        对指数协方差核, 积分方程:
          int_0^L C(x,s) * phi(s) ds = lambda * phi(x)
        等价于微分方程:
          -l_c^2 * phi''(x) + phi(x) = (2*l_c/lambda) * phi(x)  不对

        这里使用离散近似:
        C_ij = C(x_i, x_j) * h
        求解 C * phi = lambda * phi
        """
        n = self.n_elements + 1
        x = self.x_nodes

        # 构建协方差矩阵
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                C[i, j] = self.sigma ** 2 * np.exp(-abs(x[i] - x[j]) / self.lc)
        C *= self.h

        # 对称化以确保数值稳定
        C = 0.5 * (C + C.T)

        # 特征分解
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            eigvals = np.zeros(n)
            eigvecs = np.eye(n)

        # 按降序排列
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 截断
        M = min(self.n_terms, n)
        self.eigenvalues = np.maximum(eigvals[:M], 0.0)
        self.eigenfunctions = eigvecs[:, :M]

        # 归一化特征函数: int phi_i^2 dx ≈ 1
        for i in range(M):
            norm = np.sqrt(np.sum(self.eigenfunctions[:, i] ** 2) * self.h)
            if norm > 1e-15:
                self.eigenfunctions[:, i] /= norm

    def evaluate(self, xi):
        """给定随机变量 xi = (xi_1,...,xi_M), 计算随机场样本

        E(x) = mu + sum_i sqrt(lambda_i) * phi_i(x) * xi_i

        Parameters
        ----------
        xi : np.ndarray (M,), KL 随机变量 (标准正态)

        Returns
        -------
        field : np.ndarray (n_elements+1,), 随机场值
        """
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        M = min(len(xi), len(self.eigenvalues))
        field = np.zeros(self.n_elements + 1)
        for i in range(M):
            field += np.sqrt(self.eigenvalues[i]) * self.eigenfunctions[:, i] * xi[i]
        return field

    def explained_variance_ratio(self):
        """KL 展开的方差解释比例"""
        total = np.sum(self.eigenvalues)
        if total < 1e-30:
            return np.zeros(len(self.eigenvalues))
        cumulative = np.cumsum(self.eigenvalues) / total
        return cumulative


class StochasticReactionDiffusion:
    """随机反应扩散方程 (FEM 求解)

    种子项目 377_fem_neumann 的随机扩展:
      dw/dt = D(x,theta) * d²w/dx² + NL(w, c)

    其中 D(x,theta) 为由 KL 展开参数化的随机扩散系数,
    NL(w,c) = c[0] + c[1]*w + c[2]*w^2 + c[3]*w^3 为非线性反应项。

    边界条件: Neumann 齐次  dw/dx(0) = dw/dx(1) = 0
    初始条件: w(0,x) = sin(pi*x)

    FEM 离散化 (线性 hat 函数):
      M * dw/dt = -K(D) * w + NL(w, c)
    """

    def __init__(self, n_elements=30):
        self.n = int(n_elements)
        self.h = 1.0 / self.n

        # 质量矩阵 (集中质量)
        diag = np.ones(self.n + 1)
        diag[0] = 0.5
        diag[-1] = 0.5
        self.M_diag = diag * self.h

        # 刚度矩阵 (确定性, D=1)
        self.K_base = self._assemble_stiffness(np.ones(self.n + 1))

    def _assemble_stiffness(self, D_values):
        """组装刚度矩阵 K_ij = int D(x) * phi_i' * phi_j' dx

        对线性 hat 函数:
        K 为三对角矩阵, K[i,i] = (D[i]+D[i+1])/(2h), K[i,i+1] = -(D[i]+D[i+1])/(4h)
        """
        n = self.n
        h = self.h
        K = np.zeros((n + 1, n + 1))

        for e in range(n):
            x_left = e * h
            x_right = (e + 1) * h
            D_e = 0.5 * (D_values[e] + D_values[e + 1])
            ke = D_e / h * np.array([[1.0, -1.0], [-1.0, 1.0]])
            K[e:e + 2, e:e + 2] += ke

        return K

    def _nl_reaction(self, w, c):
        """非线性反应项 NL(w,c) = c[0] + c[1]*w + c[2]*w^2 + c[3]*w^3

        种子项目 377_fem_neumann 中 NL(), Nq(), Nc() 的统一实现。
        """
        n = self.n
        h = self.h
        # 常数项贡献
        val = c[0] * h * np.ones(n + 1)
        val[0] *= 0.5
        val[-1] *= 0.5

        # 线性项
        if len(c) > 1:
            val += c[1] * w * self.M_diag

        # 二次项: int w^2 * phi_i dx
        if len(c) > 2:
            w2 = w ** 2
            wx = 0.25 * (w[:-1] + w[1:]) ** 2
            nq = np.zeros(n + 1)
            nq[0] = 2.0 * w2[0] + wx[0]
            if n > 1:
                nq[1:-1] = wx[:-1] + 4.0 * w2[1:-1] + wx[1:]
            nq[-1] = wx[-1] + 2.0 * w2[-1]
            val += c[2] * nq * h / 6.0

        # 三次项: int w^3 * phi_i dx
        if len(c) > 3:
            w2 = w ** 2
            w3 = w * w2
            wx = 0.125 * (w[:-1] + w[1:]) ** 3
            nc = np.zeros(n + 1)
            nc[0] = 3.0 * w3[0] + wx[0] - w[0] * w[min(1, n)] ** 2
            if n > 1:
                for i in range(1, n):
                    nc[i] = (wx[i - 1] + 6.0 * w3[i] + wx[min(i, n - 1)]
                             - w[i] * (w2[max(0, i - 1)] + w2[min(i + 1, n)]) * 0.5)
            nc[-1] = wx[-1] + 3.0 * w3[-1] - w[-1] * w[max(0, n - 1)] ** 2
            val += c[3] * nc * h / 10.0

        return val

    def solve_deterministic(self, c_array, t_final=1.0, n_steps=100):
        """确定性反应扩散求解 (D=1)

        使用向后 Euler 时间推进:
        (M + dt*K) * w^{n+1} = M * w^n + dt * NL(w^n, c)
        """
        n = self.n
        dt = t_final / n_steps
        w = np.sin(np.pi * np.linspace(0, 1, n + 1))

        A = np.diag(self.M_diag) + dt * self.K_base

        t_vals = [0.0]
        w_vals = [w.copy()]

        for step in range(n_steps):
            rhs = self.M_diag * w + dt * self._nl_reaction(w, c_array)
            try:
                w = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                break

            # 数值稳定性: 限制 w 的范围
            w = np.clip(w, -100.0, 100.0)

            t_vals.append((step + 1) * dt)
            w_vals.append(w.copy())

        return np.array(t_vals), np.array(w_vals)

    def solve_stochastic_mc(self, kl_expansion, c_array, n_mc=50,
                            t_final=1.0, n_steps=50):
        """Monte Carlo 随机反应扩散求解

        对每个 MC 样本:
        1. 从标准正态分布采样 KL 变量 xi
        2. 用 KL 展开计算随机扩散场 D(x, theta)
        3. 组装对应的刚度矩阵并求解

        Returns
        -------
        mean_w : np.ndarray (n_steps+1, n+1)
        std_w : np.ndarray (n_steps+1, n+1)
        """
        n = self.n
        dt = t_final / n_steps
        w0 = np.sin(np.pi * np.linspace(0, 1, n + 1))

        all_w = []

        for mc in range(n_mc):
            xi = np.random.default_rng(mc + 100).standard_normal(kl_expansion.n_terms)
            D_field = 1.0 + kl_expansion.evaluate(xi)
            D_field = np.maximum(D_field, 0.01)  # 保证正定性

            K = self._assemble_stiffness(D_field)
            A = np.diag(self.M_diag) + dt * K
            w = w0.copy()
            w_snap = [w.copy()]

            for step in range(n_steps):
                rhs = self.M_diag * w + dt * self._nl_reaction(w, c_array)
                try:
                    w = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    break
                w = np.clip(w, -100.0, 100.0)
                w_snap.append(w.copy())

            all_w.append(np.array(w_snap))

        all_w = np.array(all_w)
        return np.mean(all_w, axis=0), np.std(all_w, axis=0)
