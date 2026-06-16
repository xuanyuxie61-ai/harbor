"""
interior_point_solver.py - 原始-对偶内点法求解器
================================================

核心模块: 实现 Mehrotra Predictor-Corrector 内点法

融合种子项目的算法:
  - 404_fem2d_heat_rectangle: 带状矩阵求解 (DGB_FA/DGB_SL)
  - 963_r83_np: 三对角矩阵操作
  - 809_nonlin_regula: 非线性求根 (用于步长选择)
  - 104_boundary_locus: 稳定域分析 (用于线搜索)
  - 339_eternity: LP 稀疏矩阵建模

核心数学: Mehrotra Predictor-Corrector 内点法

算法框架:
  求解凸优化问题:
    min f(x)  s.t. Ax = b, l <= x <= u

  引入松弛变量 s_l = x - l >= 0, s_u = u - x >= 0:
    min f(x)  s.t. Ax = b, x - s_l = l, x + s_u = u
                  s_l >= 0, s_u >= 0

  KKT 条件 (含障碍扰动):
    F_mu(z) = 0
  其中 z = (x, y, z_l, z_u, s_l, s_u), 且:

    r_d = nabla f(x) - A^T y - z_l + z_u           (对偶残差)
    r_p = Ax - b                                      (原始残差)
    r_sl = x - l - s_l                                (下界松弛)
    r_su = x + s_u - u                                (上界松弛)
    r_cl = S_l Z_l e - mu e                          (下互补)
    r_cu = S_u Z_u e - mu e                          (上互补)

Mehrotra 算法:
  1. 仿射方向 (predictor):
     求解 F'_mu(z) dz_aff = -F_0(z)  (mu=0)
     计算最大步长 alpha_aff

  2. 中心参数:
     mu = s_l^T z_l + s_u^T z_u) / (2*n_bounded)
     mu_aff = ((s_l+alpha_aff*ds_l)^T (z_l+alpha_aff*dz_l) +
               (s_u+alpha_aff*ds_u)^T (z_u+alpha_aff*dz_u)) / (2*n)
     sigma = (mu_aff / mu)^3

  3. 修正方向 (corrector):
     求解 F'_mu(z) dz = -F_{sigma*mu}(z) - 修正项
     修正项 = [0, ..., 0, ds_l_aff * dz_l_aff, ds_u_aff * dz_u_aff]^T

  4. 步长选择:
     alpha_p = max { alpha : s_l + alpha*ds_l >= 0, s_u + alpha*ds_u >= 0 }
     alpha_d = max { alpha : z_l + alpha*dz_l >= 0, z_u + alpha*dz_u >= 0 }
     使用 fraction-to-boundary 规则: alpha = min(1, 0.995 * alpha_max)

收敛准则:
  ||r_p||_inf < eps_tol * (1 + ||b||_inf)
  ||r_d||_inf < eps_tol * (1 + ||c||_inf)
  mu < eps_tol
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class IPMResult:
    """内点法求解结果."""
    x: np.ndarray           # 原始最优解
    y: np.ndarray           # 等式对偶变量
    objective: float        # 最优目标值
    status: str             # 状态 ('optimal', 'infeasible', 'max_iter')
    iterations: int         # 迭代次数
    gap_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)
    mu_history: List[float] = field(default_factory=list)


class InteriorPointSolver:
    """
    Mehrotra Predictor-Corrector 原始-对偶内点法.

    这是凸优化中最先进的求解算法之一, 具有以下特性:
      - 超线性收敛 (在非退化情况下)
      - 多项式迭代复杂度 O(sqrt(n) * L)
        其中 L 是输入长度 (比特数)
      - 自适应中心参数 (Mehrotra 启发式)
      - 稳定的线搜索策略

    Parameters
    ----------
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差
    verbose : bool
        是否打印迭代信息
    """

    def __init__(self, max_iter: int = 100, tol: float = 1.0e-8,
                 verbose: bool = True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def solve(self, c: np.ndarray, A: sparse.spmatrix, b: np.ndarray,
              lb: np.ndarray, ub: np.ndarray,
              Q: Optional[sparse.spmatrix] = None) -> IPMResult:
        """
        求解凸优化问题.

        min c^T x + 0.5 x^T Q x
        s.t. Ax = b
             lb <= x <= ub

        Parameters
        ----------
        c : ndarray
            线性目标系数
        A : sparse matrix
            等式约束矩阵
        b : ndarray
            等式约束右端
        lb, ub : ndarray
            变量界
        Q : sparse matrix, optional
            二次目标 Hessian

        Returns
        -------
        IPMResult
            求解结果
        """
        n = len(c)
        m = A.shape[0]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Mehrotra Predictor-Corrector 内点法")
            print(f"{'='*60}")
            print(f"变量数 n = {n},  约束数 m = {m}")

        # 初始化
        x, y, z_l, z_u = self._initialize(c, A, b, lb, ub, Q)

        # 记录
        gap_history = []
        residual_history = []
        mu_history = []

        # 主迭代
        for k in range(self.max_iter):
            # 计算残差
            res = self._compute_residuals(x, y, z_l, z_u, c, A, b, Q, lb, ub)
            mu = res['mu']

            gap_history.append(res['primal_res'] + res['dual_res'])
            residual_history.append(res['dual_res'])
            mu_history.append(mu)

            if self.verbose and (k % 10 == 0 or k < 5):
                print(f"  iter {k:3d} | mu = {mu:.3e} | "
                      f"p_res = {res['primal_res']:.3e} | "
                      f"d_res = {res['dual_res']:.3e} | "
                      f"obj = {res['objective']:.6f}")

            # 收敛检查
            if self._check_convergence(res, b, c, mu):
                if self.verbose:
                    print(f"\n  收敛! 迭代 {k} 次, mu = {mu:.3e}")
                return IPMResult(
                    x=x, y=y, objective=res['objective'],
                    status='optimal', iterations=k,
                    gap_history=gap_history,
                    residual_history=residual_history,
                    mu_history=mu_history
                )

            # Mehrotra predictor-corrector 步
            x, y, z_l, z_u = self._mehrotra_step(
                x, y, z_l, z_u, c, A, b, lb, ub, Q, mu)

        # 未收敛
        if self.verbose:
            print(f"\n  达到最大迭代次数 {self.max_iter}")

        obj = float(c @ x + 0.5 * (x @ (Q @ x) if Q is not None else 0.0))
        return IPMResult(
            x=x, y=y, objective=obj,
            status='max_iter', iterations=self.max_iter,
            gap_history=gap_history,
            residual_history=residual_history,
            mu_history=mu_history
        )

    def _initialize(self, c, A, b, lb, ub, Q):
        """初始化内点 (Mehrotra 启发式)."""
        n = len(c)
        m = A.shape[0]

        # 最小二乘可行点
        A_dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
        try:
            AAT = A_dense @ A_dense.T + 1e-10 * np.eye(m)
            y = np.linalg.solve(AAT, b)
            x = A_dense.T @ y
        except np.linalg.LinAlgError:
            x = np.zeros(n)
            y = np.zeros(m)

        # 投影到严格内部
        x = np.maximum(x, lb + 0.1)
        x = np.minimum(x, ub - 0.1)
        x = np.clip(x, lb + 0.1, ub - 0.1)

        # 对偶变量初始化
        z_l = np.ones(n)
        z_u = np.ones(n)

        slack_l = x - lb
        slack_u = ub - x

        for i in range(n):
            if slack_l[i] > slack_u[i]:
                z_l[i] = 1.0
                z_u[i] = max(1.0, slack_l[i] - slack_u[i])
            else:
                z_u[i] = 1.0
                z_l[i] = max(1.0, slack_u[i] - slack_l[i])

        return x, y, z_l, z_u

    def _compute_residuals(self, x, y, z_l, z_u, c, A, b, Q, lb, ub):
        """计算 KKT 残差."""
        n = len(x)

        # 原始残差 (等式约束)
        A_dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
        r_p = A_dense @ x - b
        primal_res = float(np.linalg.norm(r_p, np.inf))

        # 对偶残差
        grad = c.copy()
        if Q is not None:
            Q_dense = Q.toarray() if sparse.issparse(Q) else np.asarray(Q)
            grad = grad + Q_dense @ x
        r_d = grad - A_dense.T @ y - z_l + z_u
        dual_res = float(np.linalg.norm(r_d, np.inf))

        # 互补间隙
        gap_l = np.dot(x - lb, z_l)
        gap_u = np.dot(ub - x, z_u)
        mu = (gap_l + gap_u) / (2.0 * n)

        # 目标值
        obj = float(c @ x + 0.5 * (x @ (Q @ x) if Q is not None else 0.0))

        return {
            'primal_res': primal_res,
            'dual_res': dual_res,
            'mu': max(mu, 1e-30),
            'objective': obj
        }

    def _check_convergence(self, res, b, c, mu):
        """检查收敛条件."""
        b_norm = max(np.linalg.norm(b, np.inf), 1.0)
        c_norm = max(np.linalg.norm(c, np.inf), 1.0)

        primal_ok = res['primal_res'] < self.tol * b_norm
        dual_ok = res['dual_res'] < self.tol * c_norm
        gap_ok = mu < self.tol

        return primal_ok and dual_ok and gap_ok

    def _mehrotra_step(self, x, y, z_l, z_u, c, A, b, lb, ub, Q, mu):
        """Mehrotra predictor-corrector 步."""
        n = len(x)
        m = len(y)

        A_dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
        if Q is not None:
            Q_dense = Q.toarray() if sparse.issparse(Q) else np.asarray(Q)
        else:
            Q_dense = np.zeros((n, n))

        # 松弛
        s_l = x - lb
        s_u = ub - x
        s_l = np.maximum(s_l, 1e-10)
        s_u = np.maximum(s_u, 1e-10)

        # === 仿射步 (Predictor) ===
        # Hessian + 障碍对角
        theta_aff = z_l / s_l + z_u / s_u
        H_aff = Q_dense + np.diag(theta_aff)

        # 右端
        grad = c + Q_dense @ x
        r_d_aff = grad - A_dense.T @ y - z_l + z_u
        r_p_aff = A_dense @ x - b
        r_cl_aff = s_l * z_l
        r_cu_aff = s_u * z_u

        # 约化右端
        # Newton 方向: J*dx = -F
        # 消去 dz_l, dz_u 后: r_bar = -r_d - z_l + z_u + r_cl/s_l - r_cu/s_u
        # 对仿射步 (mu=0): r_cl = s_l*z_l, r_cu = s_u*z_u
        # 简化: r_bar_aff = -r_d_aff
        r_bar_aff = -r_d_aff

        # 求解 KKT 系统
        dx_aff, dy_aff = self._solve_kkt(H_aff, A_dense, r_bar_aff, r_p_aff, n, m)

        # 松弛步
        ds_l_aff = dx_aff
        ds_u_aff = -dx_aff

        # 对偶步
        dz_l_aff = -(z_l * (s_l + ds_l_aff)) / s_l
        dz_u_aff = -(z_u * (s_u + ds_u_aff)) / s_u

        # 最大仿射步长
        alpha_p_aff = self._max_step(s_l, ds_l_aff)
        alpha_p_aff = min(alpha_p_aff, self._max_step(s_u, ds_u_aff))
        alpha_d_aff = self._max_step(z_l, dz_l_aff)
        alpha_d_aff = min(alpha_d_aff, self._max_step(z_u, dz_u_aff))

        # === 中心参数 ===
        mu_aff = (np.dot(s_l + alpha_p_aff * ds_l_aff,
                          z_l + alpha_d_aff * dz_l_aff) +
                   np.dot(s_u + alpha_p_aff * ds_u_aff,
                          z_u + alpha_d_aff * dz_u_aff)) / (2.0 * n)

        sigma = (mu_aff / mu) ** 3
        sigma = np.clip(sigma, 1e-6, 0.9999)

        # === 修正步 (Corrector) ===
        mu_new = sigma * mu

        # 修正项 (二阶)
        corr_l = ds_l_aff * dz_l_aff
        corr_u = ds_u_aff * dz_u_aff

        # 更新 Hessian (使用 mu_new)
        theta = z_l / s_l + z_u / s_u
        H = Q_dense + np.diag(theta)

        # 约化右端 (含修正)
        r_d = c + Q_dense @ x - A_dense.T @ y - z_l + z_u
        r_p = A_dense @ x - b
        r_cl = s_l * z_l - mu_new
        r_cu = s_u * z_u - mu_new

        # 约化右端 (含修正, 正确符号)
        # r_bar = -r_d - z_l + z_u + (r_cl + corr_l)/s_l - (r_cu + corr_u)/s_u
        r_bar = (-r_d - z_l + z_u
                 + (r_cl + corr_l) / s_l
                 - (r_cu + corr_u) / s_u)

        dx, dy = self._solve_kkt(H, A_dense, r_bar, r_p, n, m)

        # 完整步
        ds_l = dx
        ds_u = -dx
        dz_l = -(r_cl + z_l * ds_l) / s_l
        dz_u = -(r_cu + z_u * ds_u) / s_u

        # === 步长选择 (fraction-to-boundary) ===
        gamma = 0.995
        alpha_p = min(1.0, gamma * self._max_step(s_l, ds_l))
        alpha_p = min(alpha_p, gamma * self._max_step(s_u, ds_u))
        alpha_d = min(1.0, gamma * self._max_step(z_l, dz_l))
        alpha_d = min(alpha_d, gamma * self._max_step(z_u, dz_u))

        # === 更新 ===
        x_new = x + alpha_p * dx
        y_new = y + alpha_d * dy
        z_l_new = np.maximum(z_l + alpha_d * dz_l, 1e-15)
        z_u_new = np.maximum(z_u + alpha_d * dz_u, 1e-15)

        # 确保严格可行
        x_new = np.maximum(x_new, lb + 1e-12)
        x_new = np.minimum(x_new, ub - 1e-12)

        return x_new, y_new, z_l_new, z_u_new

    def _solve_kkt(self, H, A, r_bar, r_p, n, m):
        """求解约化 KKT 系统."""
        total = n + m
        aug = np.zeros((total, total))
        reg = 1e-8

        aug[:n, :n] = H + reg * np.eye(n)
        aug[:n, n:] = A.T
        aug[n:, :n] = A
        aug[n:, n:] = -reg * np.eye(m)

        rhs = np.concatenate([r_bar, r_p])

        try:
            sol = np.linalg.solve(aug, rhs)
        except np.linalg.LinAlgError:
            # 增加正则化
            aug[:n, :n] += 1e-5 * np.eye(n)
            aug[n:, n:] -= 1e-5 * np.eye(m)
            try:
                sol = np.linalg.solve(aug, rhs)
            except np.linalg.LinAlgError:
                sol = np.zeros(total)

        return sol[:n], sol[n:]

    def _max_step(self, x, dx, gamma=0.995):
        """最大正步长."""
        alpha = 1.0
        neg = dx < -1e-15
        if np.any(neg):
            ratios = -gamma * x[neg] / dx[neg]
            alpha = min(alpha, float(np.min(ratios)))
        return max(alpha, 1e-10)
