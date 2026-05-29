"""
electrostatic_solver.py
=======================
静电泊松-玻尔兹曼求解模块（融合 seed 606_jacobi_poisson_1d 与 seed 085_bicg）

在纳米颗粒-生物膜相互作用中，静电效应（如带电金纳米颗粒与带负电磷脂头部
之间的库仑相互作用）起决定性作用。本模块实现**一维球对称泊松-玻尔兹曼方程**
的数值求解，用于计算德拜屏蔽层内的电势分布：

    d^2 phi / dr^2 + (2/r) d phi / dr = (8*pi*e*n_0 / epsilon) * sinh(e*phi/(k_B*T))

在弱电势近似下（|e*phi| << k_B*T），线性化为德拜-休克尔方程：

    d^2 phi / dr^2 + (2/r) d phi / dr = kappa_D^2 * phi

其中德拜长度 kappa_D = sqrt(8*pi*e^2*n_0 / (epsilon*k_B*T))。

离散化：在球坐标径向网格 r_i 上采用中心差分，得到三对角线性系统：

    (1/h^2 - 1/(r_i*h)) * phi_{i-1}
    + (-2/h^2 - kappa_D^2) * phi_i
    + (1/h^2 + 1/(r_i*h)) * phi_{i+1} = 0

边界条件：
    - 内边界（r = R_np）：phi = z_np * e / (4*pi*epsilon*R_np)   （颗粒表面电势）
    - 外边界（r = R_max）：phi = 0   （本体溶液电中性）

求解器：
    1. Jacobi 迭代（seed 606 核心）：适用于教学演示与对角占优系统；
    2. BiCG（seed 085 核心）：双共轭梯度法，适用于非对称稀疏系统，收敛更快。
"""

import numpy as np
from typing import Tuple


class PoissonBoltzmannSolver:
    """
    球对称泊松-玻尔兹曼方程求解器。
    """

    def __init__(self, R_np: float = 2.5, R_max: float = 25.0,
                 z_np: float = +10.0, n_0: float = 0.1,
                 epsilon: float = 80.0 * 8.854e-12,
                 T: float = 300.0):
        """
        Parameters
        ----------
        R_np : float
            纳米颗粒半径（nm）。
        R_max : float
            计算域外边界（nm）。
        z_np : float
            纳米颗粒净电荷（以基本电荷 e 为单位）。
        n_0 : float
            本体离子浓度（mol/L）。
        epsilon : float
            溶液介电常数（F/m）。
        T : float
            温度（K）。
        """
        self.R_np = float(R_np)
        self.R_max = float(R_max)
        self.z_np = float(z_np)
        self.n_0 = float(n_0)  # mol/L
        self.epsilon = float(epsilon)
        self.T = float(T)
        # 物理常数（SI）
        self.k_B = 1.380649e-23  # J/K
        self.e_charge = 1.602176634e-19  # C
        self.N_A = 6.02214076e23  # /mol
        # 德拜长度（m -> nm）
        ionic_strength = n_0 * 1000.0 * self.N_A  # 转换为 /m^3（假设 1:1 电解质）
        self.kappa_D = np.sqrt(2 * self.e_charge ** 2 * ionic_strength /
                               (self.epsilon * self.k_B * self.T))  # 1/m
        self.kappa_D_nm = self.kappa_D * 1e-9  # 1/nm

    def _build_system(self, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建离散线性系统 A * u = b（包含边界条件的修正）。

        采用变量替换 u(r) = r * phi(r)，将球对称 Poisson-Boltzmann 方程
        转化为对称的 1D Helmholtz 方程：
            d^2u / dr^2 = kappa_D^2 * u
        离散化后得到严格对称三对角系统：
            u_{i-1} - (2 + h^2*kappa_D^2) * u_i + u_{i+1} = 0

        Returns
        -------
        A : ndarray, shape (n_grid, n_grid)
            对称系数矩阵。
        b : ndarray, shape (n_grid,)
            右端项（u 的边界值）。
        r : ndarray, shape (n_grid,)
            径向网格坐标。
        """
        # [HOLE 1] 请补全 Poisson-Boltzmann 方程的离散线性系统构建：
        # 1. 在 [R_np, R_max] 上生成 n_grid 个均匀网格点 r
        # 2. 计算步长 h 和 kappa_D^2
        # 3. 构造对称三对角矩阵 A：
        #    内部点：A[i,i-1] = -1, A[i,i] = 2 + h^2*kappa_D^2, A[i,i+1] = -1
        # 4. 设置 Dirichlet 边界条件：
        #    - 内边界：phi_surf = z_np*e / (4*pi*epsilon*R_np*1e-9), u_surf = R_np*phi_surf
        #    - 外边界：phi = 0, 即 u = 0
        # 5. 为保持矩阵对称，将边界耦合项移到右端项 b
        # TODO: 实现上述离散化过程
        raise NotImplementedError("HOLE 1: 请补全 _build_system 的离散化实现")

    def _u_to_phi(self, u: np.ndarray, r: np.ndarray) -> np.ndarray:
        """将 u = r*phi 转换回 phi，并在 r=0 附近做数值保护。"""
        phi = np.zeros_like(u)
        # r > 0 时 phi = u / r
        mask = r > 1e-12
        phi[mask] = u[mask] / r[mask]
        # r = 0 时用极限值
        phi[~mask] = u[~mask] / (r[~mask] + 1e-12)
        return phi

    def solve_jacobi(self, n_grid: int = 257, it_max: int = 50000,
                     tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        使用 Jacobi 迭代求解 u = r*phi（源自 seed 606_jacobi_poisson_1d 核心算法）。

        迭代格式：
            u^{new}_i = (b_i - sum_{j!=i} A_{ij} * u^{old}_j) / A_{ii}

        Parameters
        ----------
        n_grid : int
            径向格点数。
        it_max : int
            最大迭代次数。
        tol : float
            RMS 残差收敛阈值。

        Returns
        -------
        phi : ndarray
            电势分布（V）。
        r : ndarray
            径向坐标（nm）。
        it : int
            实际迭代次数。
        residual : float
            最终 RMS 残差。
        """
        A, b, r = self._build_system(n_grid)
        u = np.zeros(n_grid, dtype=np.float64)
        u[0] = b[0]
        u[-1] = b[-1]
        diag = np.diag(A).copy()
        diag[diag == 0] = 1.0
        for it in range(1, it_max + 1):
            u_new = (b - A @ u + diag * u) / diag
            u_new[0] = b[0]
            u_new[-1] = b[-1]
            res = np.linalg.norm(A @ u_new - b) / np.sqrt(n_grid)
            u = u_new
            if res < tol:
                phi = self._u_to_phi(u, r)
                return phi, r, it, res
        phi = self._u_to_phi(u, r)
        return phi, r, it_max, res

    def solve_bicg(self, n_grid: int = 257, max_it: int = 2000,
                   tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        使用带 Jacobi 对角预处理的 BiCG（双共轭梯度法）求解对称系统
        （融合 seed 085_bicg 的 bicg 与 bicg_pre 核心算法）。

        由于采用 u = r*phi 变换后矩阵对称，BiCG 退化为预处理 CG，
        数值稳定性大幅提高。

        预处理矩阵 M = diag(A)，其逆为 M^{-1} = 1/diag(A)。

        Returns
        -------
        phi, r, iter, residual
        """
        A, b, r = self._build_system(n_grid)
        x = np.zeros(n_grid, dtype=np.float64)
        # Jacobi 预处理矩阵（对角）
        diag_A = np.diag(A).copy()
        diag_A[diag_A == 0] = 1.0
        M_inv = 1.0 / diag_A
        # 初始残差
        r_vec = b - A @ x
        r_tld = r_vec.copy()
        if np.linalg.norm(r_vec) < tol:
            phi = self._u_to_phi(x, r)
            return phi, r, 0, np.linalg.norm(r_vec)
        z = M_inv * r_vec
        z_tld = M_inv * r_tld
        p = z.copy()
        p_tld = z_tld.copy()
        rho_old = np.dot(z, r_tld)
        for it in range(1, max_it + 1):
            q = A @ p
            q_tld = A.T @ p_tld
            denom = np.dot(p_tld, q)
            if abs(denom) < 1e-30:
                break
            alpha = rho_old / denom
            x = x + alpha * p
            r_vec = r_vec - alpha * q
            r_tld = r_tld - alpha * q_tld
            residual = np.linalg.norm(r_vec)
            if residual < tol:
                phi = self._u_to_phi(x, r)
                return phi, r, it, residual
            z = M_inv * r_vec
            z_tld = M_inv * r_tld
            rho = np.dot(z, r_tld)
            if abs(rho_old) < 1e-30:
                break
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld
            rho_old = rho
        # 最后强制边界精确
        x[0] = b[0]
        x[-1] = b[-1]
        phi = self._u_to_phi(x, r)
        return phi, r, it, np.linalg.norm(b - A @ x)

    def debye_length(self) -> float:
        """返回德拜长度（nm）。"""
        return 1.0 / self.kappa_D_nm if self.kappa_D_nm > 0 else float('inf')

    def electrostatic_force(self, n_grid: int = 257) -> float:
        """
        计算纳米颗粒表面由于电荷-电势自相互作用产生的有效残余力。
        在球对称假设下，表面电场的麦克斯韦应力为：
            sigma = 0.5 * epsilon * E^2
        其中 E = -d phi / dr |_{R_np} = -(du/dr * r - u) / r^2 |_{R_np}。
        本函数返回一个与表面电场梯度成正比的无量纲标度因子。
        """
        A, b, r = self._build_system(n_grid)
        x = np.zeros(n_grid, dtype=np.float64)
        diag_A = np.diag(A).copy()
        diag_A[diag_A == 0] = 1.0
        M_inv = 1.0 / diag_A
        r_vec = b - A @ x
        r_tld = r_vec.copy()
        z = M_inv * r_vec
        z_tld = M_inv * r_tld
        p = z.copy()
        p_tld = z_tld.copy()
        rho_old = np.dot(z, r_tld)
        for it in range(1, 2000):
            q = A @ p
            q_tld = A.T @ p_tld
            denom = np.dot(p_tld, q)
            if abs(denom) < 1e-30:
                break
            alpha = rho_old / denom
            x = x + alpha * p
            r_vec = r_vec - alpha * q
            r_tld = r_tld - alpha * q_tld
            if np.linalg.norm(r_vec) < 1e-10:
                break
            z = M_inv * r_vec
            z_tld = M_inv * r_tld
            rho = np.dot(z, r_tld)
            if abs(rho_old) < 1e-30:
                break
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld
            rho_old = rho
        x[0] = b[0]
        x[-1] = b[-1]
        # 计算 du/dr 在表面
        h = r[1] - r[0]
        dudr = (-3.0 * x[0] + 4.0 * x[1] - x[2]) / (2.0 * h)
        u0 = x[0]
        R = self.R_np
        # E = -(du/dr * R - u0) / R^2 = -(dudr - phi_surf) / R
        # 因为 u0 = R * phi_surf
        dphi_dr = -(dudr * R - u0) / (R ** 2)
        return float(dphi_dr)
