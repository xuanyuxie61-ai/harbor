"""
electrostatic_solver.py
=======================
分子静电势求解器

融合种子项目:
  - 869_pic : 粒子云网格 (PIC) 电荷沉积、Gauss-Seidel Poisson 求解、
              双线性插值、Maxwell-Boltzmann 分布

科学背景:
  分子内电子云产生静电势 φ，满足 Poisson 方程:
      ∇²φ = -ρ / ε₀
  其中 ρ 为电荷密度。在离散网格上采用有限差分法，
  结合 Gauss-Seidel 迭代求解。电场 E = -∇φ 用于计算原子受力。

  本模块将原子电荷沉积到三维网格（类似 PIC 的 Cloud-in-Cell），
  然后求解 Poisson 方程，最后将电场插值回原子位置。
"""

import numpy as np
from typing import Tuple


class ElectrostaticSolver:
    """
    基于有限差分与 Gauss-Seidel 的分子静电势求解器。
    """

    def __init__(self, box: Tuple[float, float, float], grid: Tuple[int, int, int],
                 epsilon: float = 1.0, max_iter: int = 2000, tol: float = 1e-6):
        """
        Parameters
        ----------
        box : (Lx, Ly, Lz)
            模拟盒子尺寸 (Å)。
        grid : (nx, ny, nz)
            网格分辨率。
        epsilon : float
            介电常数 (相对真空)。
        max_iter : int
            Gauss-Seidel 最大迭代次数。
        tol : float
            残差收敛阈值。
        """
        self.Lx, self.Ly, self.Lz = box
        self.nx, self.ny, self.nz = grid
        self.dx = self.Lx / max(self.nx - 1, 1)
        self.dy = self.Ly / max(self.ny - 1, 1)
        self.dz = self.Lz / max(self.nz - 1, 1)
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------
    # 1. 电荷沉积 (Cloud-in-Cell, 源自 PIC)
    # ------------------------------------------------------------------
    def deposit_charge(self, atoms: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """
        将原子电荷通过三线性权重沉积到三维网格。

        Parameters
        ----------
        atoms : np.ndarray, shape (n, 3)
            原子坐标 (Å)。
        charges : np.ndarray, shape (n,)
            原子部分电荷 (e)。

        Returns
        -------
        rho : np.ndarray, shape (nx, ny, nz)
            网格电荷密度 (e/Å³)。
        """
        rho = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)
        n_atoms = atoms.shape[0]
        for a in range(n_atoms):
            x, y, z = atoms[a]
            # 映射到网格索引 (包含边界处理)
            ix = int(np.floor(x / self.dx))
            iy = int(np.floor(y / self.dy))
            iz = int(np.floor(z / self.dz))
            # 限制在有效范围
            ix = max(0, min(ix, self.nx - 2))
            iy = max(0, min(iy, self.ny - 2))
            iz = max(0, min(iz, self.nz - 2))

            fx = (x - ix * self.dx) / self.dx
            fy = (y - iy * self.dy) / self.dy
            fz = (z - iz * self.dz) / self.dz
            fx = max(0.0, min(1.0, fx))
            fy = max(0.0, min(1.0, fy))
            fz = max(0.0, min(1.0, fz))

            wx0, wx1 = 1.0 - fx, fx
            wy0, wy1 = 1.0 - fy, fy
            wz0, wz1 = 1.0 - fz, fz

            q = charges[a]
            vol = self.dx * self.dy * self.dz
            if vol < 1e-18:
                vol = 1e-18
            qv = q / vol

            for dx_ in (0, 1):
                for dy_ in (0, 1):
                    for dz_ in (0, 1):
                        w = (wx0 if dx_ == 0 else wx1) * \
                            (wy0 if dy_ == 0 else wy1) * \
                            (wz0 if dz_ == 0 else wz1)
                        rho[ix + dx_, iy + dy_, iz + dz_] += w * qv
        return rho

    # ------------------------------------------------------------------
    # 2. 有限差分 Laplacian 与 Gauss-Seidel 求解 (源自 PIC 的 eval_2dpot_gs)
    # ------------------------------------------------------------------
    def solve_poisson(self, rho: np.ndarray) -> np.ndarray:
        """
        求解 ∇²φ = -ρ/ε₀，边界条件为 Dirichlet φ=0 在盒子外缘。
        采用 Gauss-Seidel 逐点迭代，含 SOR 加速 (omega=1.5)。

        离散 Laplacian (7点模板):
            (∇²φ)_{i,j,k} ≈
                (φ_{i+1,j,k} - 2φ_{i,j,k} + φ_{i-1,j,k}) / dx²
              + (φ_{i,j+1,k} - 2φ_{i,j,k} + φ_{i,j-1,k}) / dy²
              + (φ_{i,j,k+1} - 2φ_{i,j,k} + φ_{i,j,k-1}) / dz²
        """
        phi = np.zeros_like(rho)
        omega = 1.5
        nx, ny, nz = self.nx, self.ny, self.nz
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        dz2 = self.dz ** 2
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2 + 1.0 / dz2)
        if denom < 1e-18:
            return phi

        rhs = -rho / self.epsilon

        for it in range(self.max_iter):
            phi_old = phi.copy()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        residual = (
                            (phi[i + 1, j, k] + phi[i - 1, j, k]) / dx2 +
                            (phi[i, j + 1, k] + phi[i, j - 1, k]) / dy2 +
                            (phi[i, j, k + 1] + phi[i, j, k - 1]) / dz2 -
                            denom * phi[i, j, k] - rhs[i, j, k]
                        )
                        phi[i, j, k] += omega * residual / denom
            diff = np.max(np.abs(phi - phi_old))
            if diff < self.tol:
                break
        return phi

    # ------------------------------------------------------------------
    # 3. 电场计算 (中心差分) 与插值回原子 (源自 PIC)
    # ------------------------------------------------------------------
    def compute_electric_field(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        由 φ 通过中心差分计算电场分量 E = -∇φ。
        边界处采用单侧差分。
        """
        Ex = np.zeros_like(phi)
        Ey = np.zeros_like(phi)
        Ez = np.zeros_like(phi)
        nx, ny, nz = self.nx, self.ny, self.nz

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if i == 0:
                        dx_ = phi[i + 1, j, k] - phi[i, j, k] if nx > 1 else 0.0
                        dx_d = self.dx
                    elif i == nx - 1:
                        dx_ = phi[i, j, k] - phi[i - 1, j, k]
                        dx_d = self.dx
                    else:
                        dx_ = phi[i + 1, j, k] - phi[i - 1, j, k]
                        dx_d = 2.0 * self.dx

                    if j == 0:
                        dy_ = phi[i, j + 1, k] - phi[i, j, k] if ny > 1 else 0.0
                        dy_d = self.dy
                    elif j == ny - 1:
                        dy_ = phi[i, j, k] - phi[i, j - 1, k]
                        dy_d = self.dy
                    else:
                        dy_ = phi[i, j + 1, k] - phi[i, j - 1, k]
                        dy_d = 2.0 * self.dy

                    if k == 0:
                        dz_ = phi[i, j, k + 1] - phi[i, j, k] if nz > 1 else 0.0
                        dz_d = self.dz
                    elif k == nz - 1:
                        dz_ = phi[i, j, k] - phi[i, j, k - 1]
                        dz_d = self.dz
                    else:
                        dz_ = phi[i, j, k + 1] - phi[i, j, k - 1]
                        dz_d = 2.0 * self.dz

                    Ex[i, j, k] = -dx_ / max(dx_d, 1e-12)
                    Ey[i, j, k] = -dy_ / max(dy_d, 1e-12)
                    Ez[i, j, k] = -dz_ / max(dz_d, 1e-12)
        return Ex, Ey, Ez

    def interpolate_field_to_atoms(self, atoms: np.ndarray,
                                   Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray) -> np.ndarray:
        """
        将网格电场通过三线性插值映射回原子位置。
        """
        n_atoms = atoms.shape[0]
        E_atoms = np.zeros((n_atoms, 3), dtype=np.float64)
        for a in range(n_atoms):
            x, y, z = atoms[a]
            ix = int(np.floor(x / self.dx))
            iy = int(np.floor(y / self.dy))
            iz = int(np.floor(z / self.dz))
            ix = max(0, min(ix, self.nx - 2))
            iy = max(0, min(iy, self.ny - 2))
            iz = max(0, min(iz, self.nz - 2))

            fx = (x - ix * self.dx) / self.dx
            fy = (y - iy * self.dy) / self.dy
            fz = (z - iz * self.dz) / self.dz
            fx = max(0.0, min(1.0, fx))
            fy = max(0.0, min(1.0, fy))
            fz = max(0.0, min(1.0, fz))

            wx0, wx1 = 1.0 - fx, fx
            wy0, wy1 = 1.0 - fy, fy
            wz0, wz1 = 1.0 - fz, fz

            ex_val = 0.0
            ey_val = 0.0
            ez_val = 0.0
            for dx_ in (0, 1):
                for dy_ in (0, 1):
                    for dz_ in (0, 1):
                        w = (wx0 if dx_ == 0 else wx1) * \
                            (wy0 if dy_ == 0 else wy1) * \
                            (wz0 if dz_ == 0 else wz1)
                        ex_val += w * Ex[ix + dx_, iy + dy_, iz + dz_]
                        ey_val += w * Ey[ix + dx_, iy + dy_, iz + dz_]
                        ez_val += w * Ez[ix + dx_, iy + dy_, iz + dz_]
            E_atoms[a] = [ex_val, ey_val, ez_val]
        return E_atoms

    def compute_electrostatic_energy(self, phi: np.ndarray, rho: np.ndarray) -> float:
        """
        静电能: E = 0.5 * ε₀ ∫ φ ρ dV ≈ 0.5 * ε₀ Σ φ_i ρ_i dV。
        """
        dV = self.dx * self.dy * self.dz
        energy = 0.5 * self.epsilon * np.sum(phi * rho) * dV
        return float(energy)


def maxwell_boltzmann_velocity(temperature: float, mass_amu: float, n_samples: int) -> np.ndarray:
    """
    Maxwell-Boltzmann 速度采样 (源自 PIC)。
    一维速度分布:
        p(v) = sqrt(m / (2π k_B T)) * exp(-m v^2 / (2 k_B T))
    返回 n_samples 个速度 (Å/ps)。
    """
    k_B = 0.831446  # Boltzmann constant in amu·Å²/(ps²·K)
    sigma = np.sqrt(k_B * temperature / mass_amu)
    return np.random.normal(0.0, sigma, n_samples)
