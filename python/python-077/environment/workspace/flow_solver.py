"""
flow_solver.py
计算流体力学求解器：压力泊松方程与速度修正

融合源项目：
- 1099_sor: SOR迭代法（压力泊松方程迭代求解）
- 1098_solve: 高斯消元法（小尺度线性系统直接求解）
- 645_langford_ode: ODE动力系统（湍流能量级联时间演化）
"""

import numpy as np
from typing import Tuple, Optional, Callable
from numerical_utils import sor_solve, r8mat_fs, integrate_ode, langford_deriv


class FlowSolver:
    """
    二维稳态不可压 Navier-Stokes 方程的简化求解器。

    物理模型：
    -----------
    连续性方程：
        ∇ · u = 0

    动量方程（忽略非定常项）：
        (u · ∇) u = - (1/ρ) ∇p + ν ∇²u + f

    对于风电场尾流问题，采用势流/线性化假设，简化为压力泊松方程：

        ∇²p = ρ · ∇ · (u · ∇u - f)

    在结构化网格上，拉普拉斯算子的五点差分格式：

        (∇²p)_{i,j} ≈ (p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4·p_{i,j}) / h²

    离散后的线性系统：
        A · p = b

    采用 SOR 迭代求解（源自 1099_sor），对于小系统使用直接求解（源自 1098_solve）。
    """

    def __init__(self, nx: int = 40, ny: int = 40,
                 Lx: float = 5000.0, Ly: float = 5000.0,
                 rho: float = 1.225, nu: float = 1.5e-5):
        """
        Parameters
        ----------
        nx, ny : int
            网格划分数量。
        Lx, Ly : float
            计算域尺寸 [m]。
        rho : float
            空气密度 [kg/m³]。
        nu : float
            运动粘性系数 [m²/s]。
        """
        if nx <= 2 or ny <= 2:
            raise ValueError("网格数必须大于 2")
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.rho = rho
        self.nu = nu
        self.p = np.zeros((nx, ny))
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))

    def _build_laplacian_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        构造二维拉普拉斯算子的离散矩阵 A 和右端项 b。

        对于 N = nx·ny 个内部节点，A 是 N×N 稀疏矩阵，采用五点格式：

            A_{ii} = -2·(1/dx² + 1/dy²)
            A_{i,i±1} = 1/dy²
            A_{i,i±nx} = 1/dx²

        Returns
        -------
        A : np.ndarray
            N×N 系数矩阵。
        b : np.ndarray
            N×1 右端项。
        """
        nx, ny = self.nx, self.ny
        N = nx * ny
        A = np.zeros((N, N))
        b = np.zeros(N)

        idx = lambda i, j: j * nx + i

        cx = 1.0 / (self.dx ** 2)
        cy = 1.0 / (self.dy ** 2)
        cc = -2.0 * (cx + cy)

        for j in range(ny):
            for i in range(nx):
                k = idx(i, j)
                A[k, k] = cc

                # x 方向邻居
                if i > 0:
                    A[k, idx(i - 1, j)] = cx
                if i < nx - 1:
                    A[k, idx(i + 1, j)] = cx

                # y 方向邻居
                if j > 0:
                    A[k, idx(i, j - 1)] = cy
                if j < ny - 1:
                    A[k, idx(i, j + 1)] = cy

                # 构建右端项（简化源项）
                b[k] = self._source_term(i, j)

        # Dirichlet 边界条件：边界压力为 0
        for i in range(nx):
            A[idx(i, 0), :] = 0.0
            A[idx(i, 0), idx(i, 0)] = 1.0
            b[idx(i, 0)] = 0.0
            A[idx(i, ny - 1), :] = 0.0
            A[idx(i, ny - 1), idx(i, ny - 1)] = 1.0
            b[idx(i, ny - 1)] = 0.0
        for j in range(ny):
            A[idx(0, j), :] = 0.0
            A[idx(0, j), idx(0, j)] = 1.0
            b[idx(0, j)] = 0.0
            A[idx(nx - 1, j), :] = 0.0
            A[idx(nx - 1, j), idx(nx - 1, j)] = 1.0
            b[idx(nx - 1, j)] = 0.0

        return A, b

    def _source_term(self, i: int, j: int) -> float:
        """
        构造压力泊松方程的源项。

        对于尾流问题，源项与速度散度相关：
            b = ρ · (∂u/∂x + ∂v/∂y) / Δt
        """
        # 简化的源项模型
        x = i * self.dx
        y = j * self.dy
        # 模拟单个风机在中心产生的源
        cx, cy = self.Lx / 2.0, self.Ly / 2.0
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        sigma = 200.0
        return -self.rho * np.exp(-r**2 / (2 * sigma**2)) * 0.01

    def solve_pressure_poisson_sor(self, omega: float = 1.8,
                                    tol: float = 1e-6,
                                    max_iter: int = 5000) -> np.ndarray:
        """
        使用 SOR 迭代求解压力泊松方程。

        源自 1099_sor 的逐次超松弛迭代。

        Parameters
        ----------
        omega : float
            松弛因子 (0, 2)。
        tol : float
            收敛容差。
        max_iter : int
            最大迭代次数。

        Returns
        -------
        np.ndarray
            压力场 p，形状 (nx, ny)。
        """
        A, b = self._build_laplacian_matrix()
        N = self.nx * self.ny

        # 对于小系统，使用直接求解更稳定
        if N <= 400:
            p_vec = r8mat_fs(N, A.copy(), b.copy())
            self.p = p_vec.reshape((self.nx, self.ny))
            return self.p

        # 大系统使用 SOR
        p_vec = np.zeros(N)
        p_vec, iters = sor_solve(A, b, w=omega, tol=tol, max_iter=max_iter)
        self.p = p_vec.reshape((self.nx, self.ny))
        return self.p

    def solve_pressure_poisson_direct(self) -> np.ndarray:
        """
        使用高斯消元法直接求解压力泊松方程。

        源自 1098_solve 的部分主元高斯消元。
        """
        A, b = self._build_laplacian_matrix()
        N = self.nx * self.ny
        p_vec = r8mat_fs(N, A.copy(), b.copy())
        self.p = p_vec.reshape((self.nx, self.ny))
        return self.p

    def velocity_correction(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        速度修正步（投影法）。

        根据压力梯度修正速度：
            u* = u - (Δt/ρ) · ∂p/∂x
            v* = v - (Δt/ρ) · ∂p/∂y

        Returns
        -------
        u, v : np.ndarray
            修正后的速度场。
        """
        nx, ny = self.nx, self.ny
        dt = 1.0  # 伪时间步
        u_new = self.u.copy()
        v_new = self.v.copy()

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dpdx = (self.p[i + 1, j] - self.p[i - 1, j]) / (2.0 * self.dx)
                dpdy = (self.p[i, j + 1] - self.p[i, j - 1]) / (2.0 * self.dy)
                u_new[i, j] -= (dt / self.rho) * dpdx
                v_new[i, j] -= (dt / self.rho) * dpdy

        self.u = u_new
        self.v = v_new
        return self.u, self.v

    def set_inflow(self, u_inflow: float):
        """设置来流速度边界条件。"""
        self.u[:, 0] = u_inflow
        self.v[:, 0] = 0.0

    def compute_divergence(self) -> np.ndarray:
        """
        计算速度场的散度 ∇·u。

            div(u) = ∂u/∂x + ∂v/∂y
        """
        nx, ny = self.nx, self.ny
        div = np.zeros((nx, ny))
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dudx = (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * self.dx)
                dvdy = (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * self.dy)
                div[i, j] = dudx + dvdy
        return div

    def compute_vorticity(self) -> np.ndarray:
        """
        计算速度场的涡量 ω = ∂v/∂x - ∂u/∂y。
        """
        nx, ny = self.nx, self.ny
        vort = np.zeros((nx, ny))
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dvdx = (self.v[i + 1, j] - self.v[i - 1, j]) / (2.0 * self.dx)
                dudy = (self.u[i, j + 1] - self.u[i, j - 1]) / (2.0 * self.dy)
                vort[i, j] = dvdx - dudy
        return vort

    def turbulence_kinetic_energy(self) -> np.ndarray:
        """
        估算湍动能 TKE。

        简化模型：
            k ≈ 0.5 · (u'² + v'²)

        其中脉动速度通过局部速度梯度估算。
        """
        nx, ny = self.nx, self.ny
        tke = np.zeros((nx, ny))
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                dudx = (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * self.dx)
                dvdy = (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * self.dy)
                # 简化的 TKE 估算
                tke[i, j] = 0.5 * (dudx**2 + dvdy**2) * self.dx**2
        return tke


class TurbulenceCascade:
    """
    湍流能量级联的简化 ODE 模型。

    融合 645_langford_ode 的 ODE 积分方法。

    物理模型：
    -----------
    采用 SST k-ω 两方程模型，该模型在尾流和分离流中表现良好：

        dK/dt = P_K - β* · K · ω
        dω/dt = γ · (ω/K) · P_K - β · ω²

    其中：
        - K   : 湍动能 [m²/s²]
        - ω   : 特定耗散率 [1/s]
        - P_K : 湍流产生率
        - β*  = 0.09, β = 0.075, γ = 0.553 为模型常数

    涡粘系数：
        ν_t = K / ω

    此模型天然保持 K > 0, ω > 0，数值稳定性远优于 k-ε 模型。
    """

    def __init__(self, beta_star: float = 0.09, beta: float = 0.075,
                 gamma: float = 0.553):
        self.beta_star = beta_star
        self.beta = beta
        self.gamma = gamma

    def rhs(self, t: float, y: np.ndarray, P_K: float) -> np.ndarray:
        """
        SST k-ω 模型右端项。

        Parameters
        ----------
        t : float
            时间。
        y : np.ndarray
            [K, ω]。
        P_K : float
            湍流产生率。

        Returns
        -------
        np.ndarray
            [dK/dt, dω/dt]。
        """
        K, omega = y
        # 强制正的下界
        K = max(K, 1e-10)
        omega = max(omega, 1e-10)

        # 湍动能方程
        dK = P_K - self.beta_star * K * omega
        # 特定耗散率方程
        domega = self.gamma * (omega / K) * P_K - self.beta * omega ** 2

        return np.array([dK, domega])

    def integrate(self, y0: np.ndarray, t_span: Tuple[float, float],
                  P_K: float, n_steps: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        积分湍流级联方程。

        Parameters
        ----------
        y0 : np.ndarray
            初始状态 [K0, ω0]。
        t_span : Tuple[float, float]
            时间区间。
        P_K : float
            产生率。
        n_steps : int
            步数。

        Returns
        -------
        t, y : np.ndarray
            时间序列与状态序列，y 形状 (n_steps+1, 2)。
        """
        f = lambda t, y: self.rhs(t, y, P_K)
        return integrate_ode(f, y0, t_span, n_steps)
