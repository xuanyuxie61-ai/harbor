"""
linear_solver.py — 地核发电机隐式离散化稀疏线性系统求解模块

原项目映射: 152_cg_rc — 反向通信共轭梯度法 (Reverse Communication CG)

改造思路:
  将MATLAB的cg_rc()改写为Python实现，用于求解发电机模拟中每个时间步的
  隐式扩散方程。磁感应方程的径向离散化产生大型稀疏对称正定矩阵系统:
    (I - θ·Δt·D) · S^{n+1} = RHS
  其中 D 是离散化的扩散-对流算子，θ 为隐式因子 (Crank-Nicolson: θ=0.5)。

科学背景:
  地核发电机方程在球坐标下的离散化产生带宽较大的稀疏矩阵。
  对于 N_r × N_θ ≈ 1500 的网格，直接求解代价为 O(N³) ~ 3×10⁹ 运算量，
  而预处理共轭梯度法 (PCG) 仅需 O(N) 每步迭代，配合RCM重排序后
  带宽压缩显著，迭代次数大幅降低。

  离散化矩阵 A 的结构:
    A = I - θ·Δt·(D_r + D_θ + C)
  其中 D_r 为径向二阶差分 (三对角), D_θ 为角度方向差分 (块三对角),
  C 为对流项 (非对称小量)。在 θ 方向周期性边界条件下矩阵具有
  近似的块三对角结构。
"""

import numpy as np
from typing import Tuple, Optional, Callable


class ConjugateGradientSolver:
    """
    共轭梯度求解器，支持显式矩阵和矩阵-向量乘积回调两种模式。
    融合 cg_rc.m 的反向通信思想，实现为直接迭代调用。
    """

    def __init__(
        self,
        max_iter: int = 2000,
        tol: float = 1e-10,
        preconditioner: Optional[str] = "jacobi",
    ):
        """
        初始化CG求解器。

        参数:
            max_iter: 最大迭代次数
            tol: 残差收敛容差 (‖r‖₂ / ‖b‖₂ < tol)
            preconditioner: 预处理器类型 ("jacobi", "identity", or callable)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.preconditioner = preconditioner

    def _apply_preconditioner(self, M_diag: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        应用Jacobi预处理器: z = M^{-1} r, 其中 M = diag(A)。
        """
        if self.preconditioner == "jacobi":
            # 安全除法，避免除零
            diag_safe = np.where(np.abs(M_diag) > 1e-30, M_diag, 1.0)
            return r / diag_safe
        elif self.preconditioner == "identity":
            return r.copy()
        elif callable(self.preconditioner):
            return self.preconditioner(r)
        else:
            return r.copy()

    def solve(
        self,
        A_matvec: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        M_diag: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        使用共轭梯度法求解 A·x = b。

        参数:
            A_matvec: 矩阵-向量乘积函数，接受向量v返回A·v
            b: 右端项向量
            x0: 初始猜测 (默认零向量)
            M_diag: 矩阵对角线，用于Jacobi预处理
        返回:
            x: 解向量
            info: 包含迭代次数、残差、收敛标志的字典
        """
        b = np.asarray(b, dtype=float)
        n = b.size

        if x0 is None:
            x = np.zeros(n)
        else:
            x = np.asarray(x0, dtype=float).copy()

        if M_diag is None:
            M_diag = np.ones(n)

        # 初始残差 r = b - A·x
        r = b - A_matvec(x)

        # 预处理: z = M^{-1} r
        z = self._apply_preconditioner(M_diag, r)
        p = z.copy()

        rho = np.dot(r, z)
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-30:
            b_norm = 1.0

        residual_history = []

        for k in range(self.max_iter):
            # q = A·p
            q = A_matvec(p)
            pdotq = np.dot(p, q)

            if abs(pdotq) < 1e-30:
                info = {
                    "iter": k,
                    "residual": np.linalg.norm(r),
                    "rel_residual": np.linalg.norm(r) / b_norm,
                    "converged": False,
                    "reason": "pdotq too small",
                }
                return x, info

            alpha = rho / pdotq
            x += alpha * p
            r -= alpha * q

            res_norm = np.linalg.norm(r)
            rel_res = res_norm / b_norm
            residual_history.append(rel_res)

            if rel_res < self.tol:
                info = {
                    "iter": k + 1,
                    "residual": res_norm,
                    "rel_residual": rel_res,
                    "converged": True,
                    "reason": "tolerance reached",
                }
                return x, info

            z = self._apply_preconditioner(M_diag, r)
            rho_new = np.dot(r, z)
            beta = rho_new / rho
            p = z + beta * p
            rho = rho_new

        info = {
            "iter": self.max_iter,
            "residual": np.linalg.norm(r),
            "rel_residual": np.linalg.norm(r) / b_norm,
            "converged": False,
            "reason": "max_iter reached",
        }
        return x, info

    def solve_dense(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        对显式稠密/稀疏矩阵使用CG求解。
        """
        A = np.asarray(A, dtype=float)
        M_diag = np.diag(A)

        def matvec(v):
            return A.dot(v)

        return self.solve(matvec, b, x0, M_diag)


class SparseMatrixOperator:
    """
    稀疏矩阵算子封装，用于发电机离散化系统的矩阵-向量乘积。
    基于有限差分离散化的块三对角结构，避免显式存储完整矩阵。
    """

    def __init__(
        self,
        nr: int,
        ntheta: int,
        dr: float,
        dtheta: float,
        r_grid: np.ndarray,
        theta_grid: np.ndarray,
        dt: float,
        theta_implicit: float = 0.5,
        eta: float = 1.0,
    ):
        """
        初始化稀疏算子参数。

        离散化格式 (Crank-Nicolson):
          (I - θ·Δt·D) S^{n+1} = (I + (1-θ)·Δt·D) S^n + Δt·source

        D算子在 (r,θ) 网格上的中心差分:
          D[S]_{i,j} = (S_{i+1,j} - 2S_{i,j} + S_{i-1,j}) / Δr²
                     + (S_{i,j+1} - 2S_{i,j} + S_{i,j-1}) / (r_i² Δθ²)
                     - cot(θ_j)·(S_{i,j+1} - S_{i,j-1}) / (2 r_i² Δθ)
        """
        self.nr = nr
        self.ntheta = ntheta
        self.dr = dr
        self.dtheta = dtheta
        self.r_grid = r_grid
        self.theta_grid = theta_grid
        self.dt = dt
        self.theta_imp = theta_implicit
        self.eta = eta
        self.n_total = nr * ntheta

        # 预计算系数
        self._build_coefficients()

    def _build_coefficients(self):
        """
        预计算有限差分系数，提升matvec效率。
        """
        self.c_r_center = -2.0 * self.eta / (self.dr ** 2)
        self.c_r_plus = self.eta / (self.dr ** 2)
        self.c_r_minus = self.eta / (self.dr ** 2)

        self.c_t_center = np.zeros((self.nr, self.ntheta))
        self.c_t_plus = np.zeros((self.nr, self.ntheta))
        self.c_t_minus = np.zeros((self.nr, self.ntheta))

        for i in range(self.nr):
            r = self.r_grid[i]
            r2 = r * r
            for j in range(self.ntheta):
                theta = self.theta_grid[j]
                # 角度扩散项
                self.c_t_center[i, j] = -2.0 * self.eta / (r2 * self.dtheta ** 2)
                self.c_t_plus[i, j] = self.eta / (r2 * self.dtheta ** 2)
                self.c_t_minus[i, j] = self.eta / (r2 * self.dtheta ** 2)
                # 角度对流/一阶项
                if abs(theta) > 1e-10 and abs(theta - np.pi) > 1e-10:
                    cot_term = self.eta * np.cos(theta) / (r2 * np.sin(theta) * 2.0 * self.dtheta)
                    self.c_t_plus[i, j] -= cot_term
                    self.c_t_minus[i, j] += cot_term

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        计算 (I - θ·Δt·D) · v 的矩阵-向量乘积。
        边界条件已嵌入: 边界上的 v 被强制置零 (Dirichlet)。
        """
        v2d = v.reshape((self.nr, self.ntheta))
        result = np.zeros_like(v2d)

        for i in range(1, self.nr - 1):
            for j in range(1, self.ntheta - 1):
                val = v2d[i, j]
                # 径向
                val += self.theta_imp * self.dt * (
                    self.c_r_center * v2d[i, j]
                    + self.c_r_plus * v2d[i + 1, j]
                    + self.c_r_minus * v2d[i - 1, j]
                )
                # 角度
                val += self.theta_imp * self.dt * (
                    self.c_t_center[i, j] * v2d[i, j]
                    + self.c_t_plus[i, j] * v2d[i, j + 1]
                    + self.c_t_minus[i, j] * v2d[i, j - 1]
                )
                result[i, j] = val

        # 边界保持零
        return result.reshape(-1)

    def apply_rhs(self, v: np.ndarray) -> np.ndarray:
        """
        计算右端项: (I + (1-θ)·Δt·D) · v
        """
        v2d = v.reshape((self.nr, self.ntheta))
        result = np.zeros_like(v2d)
        theta_exp = 1.0 - self.theta_imp

        for i in range(1, self.nr - 1):
            for j in range(1, self.ntheta - 1):
                val = v2d[i, j]
                val += theta_exp * self.dt * (
                    self.c_r_center * v2d[i, j]
                    + self.c_r_plus * v2d[i + 1, j]
                    + self.c_r_minus * v2d[i - 1, j]
                )
                val += theta_exp * self.dt * (
                    self.c_t_center[i, j] * v2d[i, j]
                    + self.c_t_plus[i, j] * v2d[i, j + 1]
                    + self.c_t_minus[i, j] * v2d[i, j - 1]
                )
                result[i, j] = val

        return result.reshape(-1)

    def diagonal(self) -> np.ndarray:
        """
        返回矩阵对角线，用于Jacobi预处理。
        """
        diag = np.ones(self.n_total)
        for i in range(1, self.nr - 1):
            for j in range(1, self.ntheta - 1):
                idx = i * self.ntheta + j
                diag[idx] = 1.0 + self.theta_imp * self.dt * (
                    self.c_r_center + self.c_t_center[i, j]
                )
        return diag
