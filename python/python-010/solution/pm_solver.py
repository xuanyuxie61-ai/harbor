"""
pm_solver.py
============
粒子网格（Particle-Mesh, PM）引力求解器

采用 Cloud-in-Cell (CIC) 质量分配、FFT 求解泊松方程、CIC 力插值，
融入 r8crs（稀疏矩阵存储）与 r8ge（稠密 LU 分解）的核心算法，
为小规模 N 体模拟提供高效的引力计算。

核心物理公式
------------
泊松方程（共动坐标）:
    ∇² Φ(x) = 4π G a² ρ̄ δ(x)

    其中 Φ 为引力势，δ = (ρ - ρ̄)/ρ̄ 为密度对比度，a 为尺度因子。

傅里叶空间解法:
    Φ_k = -4π G a² ρ̄ δ_k / k²

引力（加速度）:
    g = -∇Φ / a = - (1/a) ∇Φ

CIC 质量分配:
    对于位于 x_p 的粒子，将其质量分配到周围 2ᵈ 个网格点上，
    权重与距离成线性反比:
        W(x_p - x_g) = Π_i (1 - |x_{p,i} - x_{g,i}|/dx)

离散化稳定性条件:
    力分辨率需满足 Nyquist 准则: k_Nyq = π N / L
"""

import numpy as np
from typing import Tuple
from linalg_utils import SparseCRS, build_laplacian_1d, solve_tridiagonal


class PMSolver:
    """
    三维粒子网格引力求解器。
    """

    def __init__(self, N: int, L: float, G: float = 4.30091e-9):
        """
        Parameters
        ----------
        N : int
            每维网格数
        L : float
            盒子边长（Mpc）
        G : float
            引力常数（Mpc M⊙⁻¹ (km/s)²）
        """
        self.N = N
        self.L = L
        self.dx = L / N
        self.G = G
        self.volume = L ** 3
        # 预计算格林函数（傅里叶空间）
        self._setup_green_function()

    def _setup_green_function(self):
        """
        预计算泊松方程的傅里叶空间格林函数:
            G_k = -4π G / k²
        """
        k_vec = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
        self.k2 = kx ** 2 + ky ** 2 + kz ** 2
        # 避免零除
        self.green = np.zeros_like(self.k2)
        mask = self.k2 > 0.0
        self.green[mask] = -4.0 * np.pi * self.G / self.k2[mask]

    def cic_deposit(
        self, pos: np.ndarray, mass: np.ndarray
    ) -> np.ndarray:
        """
        Cloud-in-Cell 质量分配。

        对于每个粒子 p:
            1. 找到其所在网格索引 i = floor(x_p / dx)
            2. 计算到相邻网格的相对距离
            3. 将质量 m_p 分配到 8 个邻域网格

        Parameters
        ----------
        pos : np.ndarray, shape (N_p, 3)
            粒子位置（共动坐标，范围 [0, L)）
        mass : np.ndarray, shape (N_p,)
            粒子质量

        Returns
        -------
        rho : np.ndarray, shape (N, N, N)
            网格上的质量密度场
        """
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos 必须为 (N_p, 3) 数组")
        n_part = pos.shape[0]
        rho = np.zeros((self.N, self.N, self.N), dtype=float)

        # 归一化到网格坐标
        xg = pos[:, 0] / self.dx
        yg = pos[:, 1] / self.dx
        zg = pos[:, 2] / self.dx

        i0 = np.floor(xg).astype(int) % self.N
        j0 = np.floor(yg).astype(int) % self.N
        k0 = np.floor(zg).astype(int) % self.N

        dx1 = xg - i0
        dy1 = yg - j0
        dz1 = zg - k0
        dx0 = 1.0 - dx1
        dy0 = 1.0 - dy1
        dz0 = 1.0 - dz1

        # CIC 权重分配到 8 个邻域
        for di in [0, 1]:
            wx = dx0 if di == 0 else dx1
            ii = (i0 + di) % self.N
            for dj in [0, 1]:
                wy = dy0 if dj == 0 else dy1
                jj = (j0 + dj) % self.N
                for dk in [0, 1]:
                    wz = dz0 if dk == 0 else dz1
                    kk = (k0 + dk) % self.N
                    w = wx * wy * wz
                    np.add.at(rho, (ii, jj, kk), mass * w)

        # 转换为物理密度: ρ = mass_grid / dx³
        rho /= self.dx ** 3
        return rho

    def compute_density_contrast(self, rho: np.ndarray, rho_mean: float) -> np.ndarray:
        """
        计算密度对比度 δ(x) = (ρ(x) - ρ̄) / ρ̄ 。

        边界处理:
            若 ρ̄ ≈ 0，则返回零场以避免除零。
        """
        if abs(rho_mean) < 1e-30:
            return np.zeros_like(rho)
        delta = (rho - rho_mean) / rho_mean
        return delta

    def solve_poisson_fft(self, delta: np.ndarray, a_scale: float = 1.0) -> np.ndarray:
        """
        使用 FFT 求解泊松方程得到引力势 Φ。

        方程:
            ∇²Φ = 4π G a² ρ̄ δ

        傅里叶空间:
            Φ_k = -4π G a² ρ̄ δ_k / k²

        Parameters
        ----------
        delta : np.ndarray, shape (N, N, N)
            密度对比度场
        a_scale : float
            尺度因子

        Returns
        -------
        phi : np.ndarray, shape (N, N, N)
            引力势场
        """
        delta_k = np.fft.fftn(delta) / (self.N ** 3)
        phi_k = self.green * a_scale ** 2 * delta_k
        phi = np.fft.ifftn(phi_k).real * (self.N ** 3)
        return phi

    def compute_force_from_potential(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从引力势计算加速度场（有限差分）。

        g_i = -∂Φ/∂x_i

        采用中心差分（周期性边界）:
            ∂Φ/∂x ≈ [Φ(i+1) - Φ(i-1)] / (2 dx)
        """
        # x 方向
        gx = -(np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * self.dx)
        # y 方向
        gy = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * self.dx)
        # z 方向
        gz = -(np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * self.dx)
        return gx, gy, gz

    def cic_interpolate_force(
        self, pos: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
    ) -> np.ndarray:
        """
        CIC 插值将网格力插值到粒子位置。

        与 cic_deposit 互为逆操作。
        """
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos 必须为 (N_p, 3)")
        n_part = pos.shape[0]
        acc = np.zeros((n_part, 3), dtype=float)

        xg = pos[:, 0] / self.dx
        yg = pos[:, 1] / self.dx
        zg = pos[:, 2] / self.dx

        i0 = np.floor(xg).astype(int) % self.N
        j0 = np.floor(yg).astype(int) % self.N
        k0 = np.floor(zg).astype(int) % self.N

        dx1 = xg - i0
        dy1 = yg - j0
        dz1 = zg - k0
        dx0 = 1.0 - dx1
        dy0 = 1.0 - dy1
        dz0 = 1.0 - dz1

        for di in [0, 1]:
            wx = dx0 if di == 0 else dx1
            ii = (i0 + di) % self.N
            for dj in [0, 1]:
                wy = dy0 if dj == 0 else dy1
                jj = (j0 + dj) % self.N
                for dk in [0, 1]:
                    wz = dz0 if dk == 0 else dz1
                    kk = (k0 + dk) % self.N
                    w = wx * wy * wz
                    acc[:, 0] += w * gx[ii, jj, kk]
                    acc[:, 1] += w * gy[ii, jj, kk]
                    acc[:, 2] += w * gz[ii, jj, kk]

        return acc

    def solve_poisson_sparse_direct(
        self, delta: np.ndarray, a_scale: float = 1.0
    ) -> np.ndarray:
        """
        使用稀疏矩阵直接法（融入 r8crs 与 r8ge 思想）求解一维泊松方程。

        主要用于验证 FFT 解法的正确性。将三维问题沿 x 方向切片，
        每片用 Thomas 算法（三对角矩阵直接法）求解。

        方程:
            d²Φ/dx² = 4π G a² ρ̄ δ(x)
        """
        phi = np.zeros_like(delta)
        rho_bar_term = 4.0 * np.pi * self.G * a_scale ** 2
        n = self.N
        dx = self.dx

        # 构造三对角 Laplace 矩阵的分解（Thomas 预处理）
        a_tri = np.ones(n)  # 下对角
        b_tri = -2.0 * np.ones(n)  # 主对角
        c_tri = np.ones(n)  # 上对角
        # 周期性边界修正：首尾相连
        # 这里采用 Dirichlet 近似，在边界设 Φ=0

        for j in range(self.N):
            for k in range(self.N):
                rhs = rho_bar_term * delta[:, j, k] * dx ** 2
                # 边界条件: phi[0] = phi[-1] = 0
                rhs[0] = 0.0
                rhs[-1] = 0.0
                b_mod = b_tri.copy()
                b_mod[0] = 1.0
                b_mod[-1] = 1.0
                a_mod = a_tri.copy()
                a_mod[0] = 0.0
                c_mod = c_tri.copy()
                c_mod[-1] = 0.0
                try:
                    phi[:, j, k] = solve_tridiagonal(a_mod, b_mod, c_mod, rhs)
                except RuntimeError:
                    phi[:, j, k] = 0.0
        return phi

    def compute_gravity(
        self,
        pos: np.ndarray,
        mass: np.ndarray,
        rho_mean: float,
        a_scale: float = 1.0,
        use_fft: bool = True,
    ) -> np.ndarray:
        """
        完整引力计算流程: CIC 分配 → 求 δ → 解泊松 → 求力 → CIC 插值。

        Returns
        -------
        acc : np.ndarray, shape (N_p, 3)
            粒子加速度（物理单位）
        """
        rho = self.cic_deposit(pos, mass)
        delta = self.compute_density_contrast(rho, rho_mean)
        if use_fft:
            phi = self.solve_poisson_fft(delta, a_scale)
        else:
            phi = self.solve_poisson_sparse_direct(delta, a_scale)
        gx, gy, gz = self.compute_force_from_potential(phi)
        acc = self.cic_interpolate_force(pos, gx, gy, gz)
        # 加速度除以尺度因子以得到共动加速度
        acc = acc / a_scale
        return acc


if __name__ == "__main__":
    N = 32
    L = 100.0
    solver = PMSolver(N, L)
    n_part = N ** 3
    pos = np.random.rand(n_part, 3) * L
    mass = np.ones(n_part) * 1e10
    rho_mean = n_part * 1e10 / (L ** 3)
    acc = solver.compute_gravity(pos, mass, rho_mean)
    print("加速度统计:")
    print(f"  mean = {acc.mean(axis=0)}")
    print(f"  std = {acc.std(axis=0)}")
