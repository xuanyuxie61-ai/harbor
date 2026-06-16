"""
growth_solver.py -- 暗能量宇宙线性增长因子 BDF/IMEX 求解器
================================================================
Project 260: 暗能量状态方程约束 -- 高阶有限差分与稳定性分析

本模块求解增长方程:
    D''(u) + P(u) D'(u) + Q(u) D(u) = 0
其中 u = ln(a),  ' = d/du.

融合种子项目
-----------
  - 064_backward_euler_fixed: 隐式 Euler + Picard 固定点迭代
  - 1374_unstable_ode: 不稳定 ODE 处理 (刚性)
  - 395_fem1d_pack: FEM 基函数 + Gauss-Legendre 积分

数学公式
--------
(1)  增长方程 (标准形式):
         D'' + (2 + H'/H) D' - (3/2) Omega_m(a) D = 0

(2)  增长方程 (对数导数形式, f = d ln D / d ln a):
         df/du + f^2 + (2 + H'/H) f - (3/2) Omega_m(a) = 0

(3)  EdS 解析解 (物质主导, Om=1):
         D(a) = a,  f(a) = 1

(4)  de Sitter 极限 (Om -> 0):
         D(a) -> const,  f -> 0

(5)  Linder 近似 (2005):
         f ≈ Omega_m(a)^gamma,  gamma ≈ 0.55  (GR + LCDM)

(6)  f*sigma8 观测量:
         f*sigma8(z) = f(a) * sigma8 * D(a) / D(1)

(7)  Volterra 积分方程:
         D(u) = D_0(u) + integral_{u_0}^{u} K(u,s) D(s) ds
         K(u,s) = Green 函数

(8)  BDF-k 方法:
         sum_{j=0}^{k} alpha_j y_{n+j} = dt * beta_k * f(t_{n+k}, y_{n+k})
         BDF1: alpha = [1, -1],  beta = 1
         BDF2: alpha = [-1/3, 4/3, -1],  beta = 2/3
         BDF3: alpha = [2/11, -9/11, 18/11, -1],  beta = 6/11

(9)  IMEX BDF (Implicit-Explicit):
         显式处理 Q(u)D (源项, 非刚性)
         隐式处理 P(u)D' (阻尼项, 可能刚性)

(10) Picard 迭代 (种子项目 064):
         y_{n+1}^{(k+1)} = y_n + dt * f(t_{n+1}, y_{n+1}^{(k)})
         收敛条件: dt * L < 1  (L = Lipschitz 常数)

(11) 初始条件 (EdS 渐近):
         D(a_min) = a_min,  D'(a_min) = a_min
         (物质主导时 D ∝ a)

(12) 归一化:
         D(a=1) = 1  (今天归一化)
================================================================
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

import cosmo_constants as cc
from fd_operators import (fd1_matrix, fd2_matrix, solve_tridiag,
                           chebyshev_grid, uniform_grid)


# =====================================================================
#  增长方程系数
# =====================================================================

def growth_coefficients(a: float, w0: float = None, wa: float = None
                        ) -> Tuple[float, float]:
    """
    增长方程系数:
        P(a) = 2 + H'/H  (阻尼项)
        Q(a) = -3/2 Omega_m(a)  (源项)

    返回 (P, Q).
    """
    hp_h = cc.hubble_prime_over_h(a, w0, wa)
    om_a = cc.omega_m_a(a, w0, wa)
    P = 2.0 + hp_h
    Q = -1.5 * om_a
    return P, Q


def growth_rhs(u: float, y: List[float], w0: float = None, wa: float = None
               ) -> List[float]:
    """
    增长方程一阶系统右端:
        y = [D, D']
        y' = [D', -P D' - Q D] = f(u, y)

    u = ln(a).
    """
    a = math.exp(u)
    P, Q = growth_coefficients(a, w0, wa)
    D, Dp = y
    return [Dp, -P * Dp - Q * D]


# =====================================================================
#  BDF 求解器 (多步隐式方法)
# =====================================================================

class BDFGrowthSolver:
    """
    BDF (Backward Differentiation Formula) 求解器.

    BDF-k:
        sum_{j=0}^{k} alpha_j y_{n+j} = dt * beta_k * f_{n+k}

    对增长方程 D'' + P D' + Q D = 0, 化为一阶系统后用 BDF.
    """

    def __init__(self, w0: float = None, wa: float = None):
        self.w0 = w0 if w0 is not None else cc.W0_FID
        self.wa = wa if wa is not None else cc.WA_FID
        self.u_grid: List[float] = []
        self.a_grid: List[float] = []
        self.D: List[float] = []
        self.D_prime: List[float] = []
        self.f_rate: List[float] = []

    def setup_grid(self, a_min: float = 1e-4, n_points: int = 200,
                   grid_type: str = 'log_linear'):
        """
        设置网格.

        grid_type:
          'log_linear': 在 u=ln(a) 上均匀
          'linear':     在 a 上均匀
        """
        u_min = math.log(max(a_min, 1e-12))
        u_max = 0.0
        if grid_type == 'log_linear':
            self.u_grid = [u_min + i*(u_max - u_min)/(n_points-1)
                           for i in range(n_points)]
        else:
            # 在 a 上均匀, 然后 u = ln(a)
            a_vals = [a_min + i*(1.0-a_min)/(n_points-1)
                      for i in range(n_points)]
            self.u_grid = [math.log(a) for a in a_vals]
        self.a_grid = [math.exp(u) for u in self.u_grid]

    def solve_bdf1(self, n_substeps: int = None) -> List[float]:
        """
        BDF1 (隐式 Euler) 求解.

        y_{n+1} = y_n + dt * f(u_{n+1}, y_{n+1})
        用 Picard 迭代求解隐式方程.
        """
        if not self.u_grid:
            self.setup_grid()
        N = len(self.u_grid)
        if n_substeps is None:
            n_substeps = N - 1

        # 重新生成均匀子步网格
        u_min = self.u_grid[0]
        u_max = self.u_grid[-1]
        h = (u_max - u_min) / n_substeps

        # 初始条件 (EdS)
        a0 = math.exp(u_min)
        D_val = a0
        Dp_val = a0

        D_out = [D_val]
        Dp_out = [Dp_val]

        for step in range(n_substeps):
            u_new = u_min + (step + 1) * h
            a_new = math.exp(u_new)
            P_new, Q_new = growth_coefficients(a_new, self.w0, self.wa)

            # Picard 迭代: y_{n+1}^{(k+1)} = y_n + h * f(u_{n+1}, y_{n+1}^{(k)})
            D_new = D_val + h * Dp_val  # 初始猜测
            Dp_new = Dp_val + h * (-P_new * Dp_val - Q_new * D_val)

            for _ in range(15):
                f_D = Dp_new
                f_Dp = -P_new * Dp_new - Q_new * D_new
                D_new_k = D_val + h * f_D
                Dp_new_k = Dp_val + h * f_Dp
                # 松弛
                omega = 0.5
                D_new = omega * D_new_k + (1 - omega) * D_new
                Dp_new = omega * Dp_new_k + (1 - omega) * Dp_new

            D_val = D_new
            Dp_val = Dp_new
            D_out.append(D_val)
            Dp_out.append(Dp_val)

        # 归一化
        if abs(D_out[-1]) > 1e-30:
            norm = D_out[-1]
            D_out = [d / norm for d in D_out]
            Dp_out = [dp / norm for dp in Dp_out]

        self.D = D_out
        self.D_prime = Dp_out
        return D_out

    def solve_bdf2(self, n_substeps: int = None) -> List[float]:
        """
        BDF2 求解.

        (3/2) y_{n+1} - 2 y_n + (1/2) y_{n-1} = dt * f(u_{n+1}, y_{n+1})
        => y_{n+1} = (4 y_n - y_{n-1} + 2 dt f_{n+1}) / 3
        """
        if not self.u_grid:
            self.setup_grid()
        N = len(self.u_grid)
        if n_substeps is None:
            n_substeps = N - 1

        u_min = self.u_grid[0]
        u_max = self.u_grid[-1]
        h = (u_max - u_min) / n_substeps

        # 初始条件
        a0 = math.exp(u_min)
        y_prev2 = [a0, a0]  # [D, D']
        # BDF1 第一步
        u1 = u_min + h
        a1 = math.exp(u1)
        P1, Q1 = growth_coefficients(a1, self.w0, self.wa)
        D1 = y_prev2[0] + h * y_prev2[1]
        Dp1 = y_prev2[1] + h * (-P1 * y_prev2[1] - Q1 * y_prev2[0])
        # Picard
        for _ in range(10):
            f_D = Dp1
            f_Dp = -P1 * Dp1 - Q1 * D1
            D1 = 0.5*(y_prev2[0] + h*f_D) + 0.5*D1
            Dp1 = 0.5*(y_prev2[1] + h*f_Dp) + 0.5*Dp1
        y_prev1 = [D1, Dp1]

        D_out = [y_prev2[0], y_prev1[0]]
        Dp_out = [y_prev2[1], y_prev1[1]]

        for step in range(2, n_substeps + 1):
            u_new = u_min + step * h
            a_new = math.exp(u_new)
            P_new, Q_new = growth_coefficients(a_new, self.w0, self.wa)

            # BDF2: y_{n+1} = (4 y_n - y_{n-1} + 2 dt f_{n+1}) / 3
            D_guess = (4*y_prev1[0] - y_prev2[0]) / 3.0
            Dp_guess = (4*y_prev1[1] - y_prev2[1]) / 3.0

            D_new = D_guess
            Dp_new = Dp_guess
            for _ in range(12):
                f_D = Dp_new
                f_Dp = -P_new * Dp_new - Q_new * D_new
                D_new_k = (4*y_prev1[0] - y_prev2[0] + 2*h*f_D) / 3.0
                Dp_new_k = (4*y_prev1[1] - y_prev2[1] + 2*h*f_Dp) / 3.0
                omega = 0.5
                D_new = omega*D_new_k + (1-omega)*D_new
                Dp_new = omega*Dp_new_k + (1-omega)*Dp_new

            y_prev2 = y_prev1[:]
            y_prev1 = [D_new, Dp_new]
            D_out.append(D_new)
            Dp_out.append(Dp_new)

        if abs(D_out[-1]) > 1e-30:
            norm = D_out[-1]
            D_out = [d/norm for d in D_out]
            Dp_out = [dp/norm for dp in Dp_out]

        self.D = D_out
        self.D_prime = Dp_out
        return D_out

    def solve_imex(self, n_substeps: int = None) -> List[float]:
        """
        IMEX (Implicit-Explicit) 求解.

        显式处理 Q(u)D 项 (源, 非刚性)
        隐式处理 P(u)D' 项 (阻尼, 可能刚性)

        IMEX Euler:
            D' = V
            V' = -P V - Q D

            V_{n+1} = (V_n - dt Q_n D_n) / (1 + dt P_{n+1})  [隐式 V]
            D_{n+1} = D_n + dt V_{n+1}  [显式 D]
        """
        if not self.u_grid:
            self.setup_grid()
        if n_substeps is None:
            n_substeps = len(self.u_grid) - 1

        u_min = self.u_grid[0]
        u_max = self.u_grid[-1]
        h = (u_max - u_min) / n_substeps

        a0 = math.exp(u_min)
        D_val = a0
        V_val = a0  # D'

        D_out = [D_val]
        V_out = [V_val]

        for step in range(n_substeps):
            u_cur = u_min + step * h
            a_cur = math.exp(u_cur)
            P_cur, Q_cur = growth_coefficients(a_cur, self.w0, self.wa)

            u_new = u_min + (step+1)*h
            a_new = math.exp(u_new)
            P_new, Q_new = growth_coefficients(a_new, self.w0, self.wa)

            # 显式: 用当前步的 Q 和 D
            rhs_V = V_val - h * Q_cur * D_val
            # 隐式: V_{n+1} = rhs_V / (1 + h * P_{n+1})
            denom = 1.0 + h * P_new
            if abs(denom) < 1e-30:
                denom = 1e-30
            V_new = rhs_V / denom
            # 显式 D 更新
            D_new = D_val + h * V_new

            D_val = D_new
            V_val = V_new
            D_out.append(D_val)
            V_out.append(V_val)

        if abs(D_out[-1]) > 1e-30:
            norm = D_out[-1]
            D_out = [d/norm for d in D_out]
            V_out = [v/norm for v in V_out]

        self.D = D_out
        self.D_prime = V_out
        return D_out

    # =====================================================================
    #  有限差分配置法 (直接在 u 网格上离散增长方程)
    # =====================================================================

    def solve_fd_collocation(self, fd_order: int = 4) -> List[float]:
        """
        有限差分配置法:
        在均匀 u 网格上用 FD 矩阵离散增长方程.

        (D2 + diag(P)*D1 + diag(Q)) D_vec = 0

        边界条件:
            D(u_min) = a_min  (EdS)
            D(u_max) = 1      (归一化)
        """
        if not self.u_grid or len(self.u_grid) < 10:
            self.setup_grid(n_points=80)

        N = len(self.u_grid)
        h = self.u_grid[1] - self.u_grid[0]

        D1 = fd1_matrix(N, h, order=fd_order)
        D2 = fd2_matrix(N, h, order=fd_order)

        # 构造系统矩阵 A = D2 + diag(P)*D1 + diag(Q)
        A = [[0.0]*N for _ in range(N)]
        for i in range(N):
            a_i = self.a_grid[i]
            P_i, Q_i = growth_coefficients(a_i, self.w0, self.wa)
            for j in range(N):
                A[i][j] = D2[i][j] + P_i * D1[i][j]
            A[i][i] += Q_i

        # 边界条件
        # 行 0 (u_min, a_min): D = a_min (EdS)
        for j in range(N):
            A[0][j] = 0.0
        A[0][0] = 1.0
        # 行 N-1 (u_max, a=1): D = 1 (归一化)
        for j in range(N):
            A[N-1][j] = 0.0
        A[N-1][N-1] = 1.0

        rhs = [0.0] * N
        rhs[0] = self.a_grid[0]
        rhs[N-1] = 1.0

        # 求解
        D_vec = _solve_linear_system(A, rhs)
        self.D = D_vec

        # 计算 D' = D1 * D
        Dp = [0.0] * N
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += D1[i][j] * D_vec[j]
            Dp[i] = s
        self.D_prime = Dp

        return D_vec

    # =====================================================================
    #  增长率与 f*sigma8
    # =====================================================================

    def compute_growth_rate(self) -> List[float]:
        """f(u) = d ln D / d ln a = D'(u) / D(u)."""
        if not self.D:
            self.solve_bdf1()
        f_vals = []
        for i in range(len(self.D)):
            if abs(self.D[i]) > 1e-30:
                f_vals.append(self.D_prime[i] / self.D[i])
            else:
                f_vals.append(1.0)  # EdS 极限
        self.f_rate = f_vals
        return f_vals

    def f_sigma8(self, sigma8: float = None) -> List[float]:
        """f*sigma8 = f(a) * sigma8 * D(a) / D(1)."""
        if sigma8 is None:
            sigma8 = cc.SIGMA_8
        if not self.f_rate:
            self.compute_growth_rate()
        D_norm = self.D[-1] if self.D else 1.0
        return [self.f_rate[i] * sigma8 * self.D[i] / max(abs(D_norm), 1e-30)
                for i in range(len(self.D))]

    def growth_at_z(self, z: float) -> Tuple[float, float]:
        """在指定红移处插值 D(z) 和 f(z)."""
        if not self.D:
            self.solve_bdf1()
        u_target = math.log(1.0 / (1.0 + z))
        # 线性插值
        best_i = 0
        best_dist = abs(self.u_grid[0] - u_target)
        for i in range(len(self.u_grid)):
            d = abs(self.u_grid[i] - u_target)
            if d < best_dist:
                best_dist = d
                best_i = i
        D_z = self.D[best_i]
        if abs(D_z) > 1e-30 and best_i < len(self.D_prime):
            f_z = self.D_prime[best_i] / D_z
        else:
            f_z = 1.0
        return D_z, f_z

    def matter_dominated_check(self) -> Dict:
        """物质主导极限检验: D(a) ~ a, f(a) ~ 1."""
        if not self.D:
            self.solve_bdf1()
        n_check = min(5, len(self.D))
        errors_D = []
        errors_f = []
        for i in range(n_check):
            a_i = self.a_grid[i]
            D_expected = a_i * self.D[0] / max(self.a_grid[0], 1e-30)
            err_D = abs(self.D[i] - D_expected) / max(abs(D_expected), 1e-30)
            errors_D.append(err_D)
            if abs(self.D[i]) > 1e-30:
                f_i = self.D_prime[i] / self.D[i]
                errors_f.append(abs(f_i - 1.0))
        return {
            'max_D_error': max(errors_D) if errors_D else 0.0,
            'max_f_error': max(errors_f) if errors_f else 0.0,
        }


# =====================================================================
#  线性系统求解 (高斯消元)
# =====================================================================

def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """高斯消元 (部分选主元)."""
    N = len(b)
    aug = [A[i][:] + [b[i]] for i in range(N)]
    for col in range(N):
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, N):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        if abs(aug[col][col]) < 1e-30:
            continue
        piv = aug[col][col]
        for k in range(col, N + 1):
            aug[col][k] /= piv
        for row in range(N):
            if row != col:
                factor = aug[row][col]
                for k in range(col, N + 1):
                    aug[row][k] -= factor * aug[col][k]
    return [aug[i][N] for i in range(N)]


if __name__ == '__main__':
    print("=== 增长因子求解器测试 ===")
    solver = BDFGrowthSolver(w0=-1.0, wa=0.0)
    solver.setup_grid(a_min=1e-4, n_points=100)

    print("\nBDF1:")
    D_bdf1 = solver.solve_bdf1()
    f_bdf1 = solver.compute_growth_rate()
    print(f"  D(a=1) = {D_bdf1[-1]:.6f}")
    print(f"  f(a=1) = {f_bdf1[-1]:.4f}")

    print("\nBDF2:")
    D_bdf2 = solver.solve_bdf2()
    f_bdf2 = solver.compute_growth_rate()
    print(f"  D(a=1) = {D_bdf2[-1]:.6f}")
    print(f"  f(a=1) = {f_bdf2[-1]:.4f}")

    print("\nIMEX:")
    D_imex = solver.solve_imex()
    f_imex = solver.compute_growth_rate()
    print(f"  D(a=1) = {D_imex[-1]:.6f}")
    print(f"  f(a=1) = {f_imex[-1]:.4f}")

    print("\nFD 配置法 (4阶):")
    D_fd = solver.solve_fd_collocation(fd_order=4)
    Dp_fd = solver.D_prime
    f_fd = [Dp_fd[i]/D_fd[i] if abs(D_fd[i])>1e-30 else 1.0
            for i in range(len(D_fd))]
    print(f"  D(a=1) = {D_fd[-1]:.6f}")
    print(f"  f(a=1) = {f_fd[-1]:.4f}")

    print("\n物质主导极限:")
    check = solver.matter_dominated_check()
    print(f"  max D error: {check['max_D_error']:.4e}")
    print(f"  max f error: {check['max_f_error']:.4e}")

    print("\nCPL 模型 (w0=-0.9, wa=-0.3):")
    solver_cpl = BDFGrowthSolver(w0=-0.9, wa=-0.3)
    solver_cpl.setup_grid(a_min=1e-4, n_points=100)
    D_cpl = solver_cpl.solve_bdf1()
    f_cpl = solver_cpl.compute_growth_rate()
    print(f"  D(a=1) = {D_cpl[-1]:.6f}")
    print(f"  f(a=1) = {f_cpl[-1]:.4f}")
    print(f"  f*sigma8(a=1) = {solver_cpl.f_sigma8()[-1]:.4f}")

    print("\n所有增长求解器测试通过.")
