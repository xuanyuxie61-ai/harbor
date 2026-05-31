"""
tight_binding.py
紧束缚 (Tight-Binding) 电子结构计算模块

整合原项目:
  - 974_r8cbb: 压缩边界带状矩阵 (Compressed Border-Banded) 分解与求解
  - 1088_slap_io: 稀疏矩阵 SLAP Triad 格式 I/O
  - 358_fd1d_bvp: 一维有限差分边界值问题

科学背景:
  表面催化反应的电子结构决定吸附能和反应势垒。
  紧束缚模型将电子波函数表示为原子轨道的线性组合:
    |ψ_k⟩ = Σ_j c_{kj} |φ_j⟩
  
  久期方程 (Secular equation):
    H * c = E * S * c
  
  其中 H 为 Hamiltonian 矩阵，S 为重叠矩阵。
  在正交基近似下 S = I，简化为标准本征值问题。
"""

import numpy as np
from typing import Tuple, Optional


class TightBindingSolver:
    """
    表面催化体系的紧束缚 Hamiltonian 求解器
    
    采用 Slater-Koster 参数化方法构建 Hamiltonian:
      H_{ij} = 
        ε_i               (i = j, onsite)
        V_{ssσ}, V_{spσ}, V_{ppσ}, V_{ppπ}  (i ≠ j, hopping)
    
    对于表面催化体系，引入有效介质近似 (Effective Medium Approximation, EMA):
      ε_i^{eff} = ε_i^{atom} + α * n_{d, i}
    
    其中 n_{d, i} 为 d 带填充数，α 为耦合参数
    """

    def __init__(self, n_atoms: int, n_orbitals_per_atom: int = 1):
        self.n_atoms = n_atoms
        self.n_orbitals = n_orbitals_per_atom
        self.n_basis = n_atoms * n_orbitals_per_atom
        self.H = np.zeros((self.n_basis, self.n_basis))
        self.S = np.eye(self.n_basis)
        self.eigenvalues = None
        self.eigenvectors = None

    def build_hamiltonian_sk(self, positions: np.ndarray,
                             onsite_energies: np.ndarray,
                             v_ss_sigma: float = -1.0,
                             v_sp_sigma: float = 1.5,
                             r_cutoff: float = 3.5e-10,
                             decay_length: float = 0.5e-10):
        """
        使用 Slater-Koster 双中心近似构建 Hamiltonian
        
        参数:
          positions: (n_atoms, 3) 原子坐标 (m)
          onsite_energies: (n_atoms,)  onsite 能量 (eV)
          v_ss_sigma: s-s σ 键 hopping 积分 (eV)
          r_cutoff: 截断半径 (m)
        
        公式:
          H_{ij} = V_{ssσ} * exp(-(r_{ij} - r_0) / d_0)  (r_{ij} <= r_cutoff)
        """
        if positions.shape[0] != self.n_atoms:
            raise ValueError("positions 行数必须等于 n_atoms")
        if onsite_energies.shape[0] != self.n_atoms:
            raise ValueError("onsite_energies 长度必须等于 n_atoms")

        n = self.n_basis
        self.H = np.zeros((n, n))
        for i in range(self.n_atoms):
            self.H[i, i] = onsite_energies[i]
            for j in range(i + 1, self.n_atoms):
                r_vec = positions[i] - positions[j]
                r_ij = np.linalg.norm(r_vec)
                if r_ij > 1e-12 and r_ij <= r_cutoff:
                    # 指数衰减 hopping 积分
                    hopping = v_ss_sigma * np.exp(-(r_ij - 2.5e-10) / decay_length)
                    self.H[i, j] = hopping
                    self.H[j, i] = hopping

    def apply_border_banded_factorization(self, n1: int, n2: int,
                                          ml: int, mu: int) -> Tuple[np.ndarray, int]:
        """
        压缩边界带状矩阵分解 (R8CBB_FA 的 Python 实现)
        
        整合原项目 974_r8cbb:
        将 Hamiltonian 分解为:
            [ A1  A2 ]
            [ A3  A4 ]
        
        其中 A1 为 (n1, n1) 带状矩阵，带宽 (ml, mu)
        A2, A3, A4 为稠密边界块
        
        分解算法:
          1. 分解 A1 = L1 * U1 (无选主元带状分解)
          2. 计算 A2' = -A1^{-1} * A2
          3. 计算 Schur 补: A4' = A4 + A3 * A2'
          4. 分解 A4' = L4 * U4
        """
        n = self.n_basis
        if n1 + n2 != n:
            raise ValueError("n1 + n2 必须等于矩阵阶数")
        if ml < 0 or mu < 0 or ml >= n1 or mu >= n1:
            raise ValueError("带宽参数非法")

        # 构建完整矩阵用于分解
        A_full = self.H.copy()
        A_lu = A_full.copy()

        # A1 分解 (带状部分，简化实现使用 dense LU)
        if n1 > 0:
            A1 = A_lu[:n1, :n1]
            try:
                # 使用 scipy 不可用，使用 numpy LU
                P, L, U = self._lu_decomposition(A1)
                A_lu[:n1, :n1] = U
                # 存储 L 在严格下三角
                for i in range(n1):
                    for j in range(i):
                        A_lu[i, j] = L[i, j]
            except np.linalg.LinAlgError:
                return A_lu, 1

        # A2' = -A1^{-1} * A2
        if n1 > 0 and n2 > 0:
            A2 = A_lu[:n1, n1:]
            for j in range(n2):
                b = -A2[:, j]
                # 前代回代求解
                x = self._band_solve(A_lu[:n1, :n1], b, ml, mu)
                A_lu[:n1, n1 + j] = x

        # Schur 补: A4' = A4 + A3 * A2'
        if n1 > 0 and n2 > 0:
            A3 = A_full[n1:, :n1]
            A2_prime = A_lu[:n1, n1:]
            A4 = A_full[n1:, n1:]
            A4_schur = A4 + A3 @ A2_prime
            A_lu[n1:, n1:] = A4_schur

        # A4 分解
        if n2 > 0:
            try:
                P, L, U = self._lu_decomposition(A_lu[n1:, n1:])
                A_lu[n1:, n1:] = U
                for i in range(n2):
                    for j in range(i):
                        A_lu[n1 + i, n1 + j] = L[i, j]
            except np.linalg.LinAlgError:
                return A_lu, n1 + 1

        return A_lu, 0

    def _lu_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Doolittle LU 分解 (无选主元)"""
        n = A.shape[0]
        L = np.eye(n)
        U = A.copy()
        for k in range(n - 1):
            if abs(U[k, k]) < 1e-15:
                raise np.linalg.LinAlgError("Zero pivot")
            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]
        return np.eye(n), L, U

    def _band_solve(self, A_band: np.ndarray, b: np.ndarray,
                    ml: int, mu: int) -> np.ndarray:
        """带状矩阵求解 (简化版)"""
        # 使用 numpy 直接求解
        return np.linalg.solve(A_band, b)

    def solve_eigenvalues_dense(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解标准本征值问题 H * c = E * c
        
        使用 numpy.linalg.eigh (对称矩阵)
        """
        eigvals, eigvecs = np.linalg.eigh(self.H)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs
        return eigvals, eigvecs

    def compute_dos(self, energies: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """
        计算电子态密度 (Density of States, DOS)
        
        公式 (Gaussian 展宽):
          ρ(E) = Σ_i (1 / sqrt(2π)σ) * exp(-(E - ε_i)^2 / (2σ^2))
        
        参数:
          energies: 能量网格 (eV)
          sigma: 展宽宽度 (eV)
        """
        if self.eigenvalues is None:
            raise RuntimeError("必须先调用 solve_eigenvalues_dense()")
        energies = np.asarray(energies, dtype=float)
        dos = np.zeros_like(energies)
        prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        for ei in self.eigenvalues:
            dos += prefactor * np.exp(-0.5 * ((energies - ei) / sigma) ** 2)
        return dos

    def compute_band_energy(self, n_electrons: int,
                           temperature_k: float = 300.0) -> float:
        """
        计算电子能带能量 (Fermi-Dirac 占据)
        
        公式:
          E_band = Σ_i f_i * ε_i
        
        其中 Fermi-Dirac 分布:
          f_i = 1 / (exp((ε_i - ε_F) / (k_B T)) + 1)
        
        化学势 ε_F 由总电子数约束确定:
          Σ_i f_i = n_electrons
        """
        if self.eigenvalues is None:
            raise RuntimeError("必须先调用 solve_eigenvalues_dense()")
        from utils import kb_t_ev
        kb_t = kb_t_ev(temperature_k)
        if kb_t < 1e-12:
            # T=0 近似
            occ = np.zeros_like(self.eigenvalues)
            idx = np.argsort(self.eigenvalues)
            occ[idx[:n_electrons]] = 1.0
            return float(np.sum(occ * self.eigenvalues))

        # 二分法求 Fermi 能级
        e_min = np.min(self.eigenvalues) - 10.0
        e_max = np.max(self.eigenvalues) + 10.0
        for _ in range(100):
            e_fermi = 0.5 * (e_min + e_max)
            fd = 1.0 / (np.exp((self.eigenvalues - e_fermi) / kb_t) + 1.0)
            ne = np.sum(fd)
            if abs(ne - n_electrons) < 1e-10:
                break
            if ne > n_electrons:
                e_max = e_fermi
            else:
                e_min = e_fermi

        fd = 1.0 / (np.exp((self.eigenvalues - e_fermi) / kb_t) + 1.0)
        return float(np.sum(fd * self.eigenvalues))

    def compute_adsorption_energy(self, e_isolated: float,
                                  e_surface: float,
                                  e_complex: float) -> float:
        """
        计算吸附能
        
        公式:
          E_ads = E_{adsorbate+surface} - E_{surface} - E_{adsorbate}
        
        负值表示放热吸附 (稳定)
        """
        return e_complex - e_surface - e_isolated

    def write_slap_format(self, filename: str):
        """
        将 Hamiltonian 矩阵写入 SLAP Triad 格式文件
        
        整合原项目 1088_slap_io:
        SLAP Triad 格式:
          首行: N, NELT, ISYM, IRHS, ISOLN
          后续: IA(I), JA(I), A(I)
          可选: RHS, SOLN
        """
        n = self.n_basis
        # 只写非零元
        rows, cols = np.nonzero(np.abs(self.H) > 1e-15)
        nelt = len(rows)
        isym = 1 if np.allclose(self.H, self.H.T, atol=1e-12) else 0
        irhs = 0
        isoln = 0

        with open(filename, 'w') as f:
            f.write(f"{n:10d}{nelt:10d}{isym:10d}{irhs:10d}{isoln:10d}\n")
            for idx in range(nelt):
                f.write(f" {rows[idx]:5d} {cols[idx]:5d} {self.H[rows[idx], cols[idx]]:16.7e}\n")

    @staticmethod
    def read_slap_format(filename: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        从 SLAP Triad 格式文件读取稀疏矩阵
        
        返回:
          n: 矩阵阶数
          ia, ja: 行列索引
          a: 非零元值
        """
        with open(filename, 'r') as f:
            header = f.readline().strip()
            n, nelt, isym, irhs, isoln = [int(x) for x in header.split()]
            ia = np.zeros(nelt, dtype=int)
            ja = np.zeros(nelt, dtype=int)
            a = np.zeros(nelt, dtype=float)
            for i in range(nelt):
                line = f.readline()
                parts = line.split()
                ia[i] = int(parts[0])
                ja[i] = int(parts[1])
                a[i] = float(parts[2])
        return n, ia, ja, a

    def solve_poisson_fd1d(self, charge_density: np.ndarray,
                           x_grid: np.ndarray,
                           epsilon_r: float = 1.0) -> np.ndarray:
        """
        一维有限差分求解 Poisson 方程
        
        整合原项目 358_fd1d_bvp:
        方程:
          -d/dx (ε(x) dφ/dx) = ρ(x) / ε_0
        
        边界条件:
          φ(x_0) = 0, φ(x_N) = 0
        
        离散化 (非均匀网格):
          -ε_M * (φ_{i-1} - 2φ_i + φ_{i+1}) / (Δx_L * Δx_R)
          -ε'_M * (φ_{i+1} - φ_{i-1}) / (2 * Δx)
          = ρ_i / ε_0
        """
        n = len(x_grid)
        if n < 3:
            raise ValueError("网格点数量必须 >= 3")
        if len(charge_density) != n:
            raise ValueError("charge_density 长度必须等于网格点数")

        eps0 = 8.854187817e-12  # F/m
        A = np.zeros((n, n))
        rhs = np.zeros(n)

        # Dirichlet 边界
        A[0, 0] = 1.0
        rhs[0] = 0.0
        A[n - 1, n - 1] = 1.0
        rhs[n - 1] = 0.0

        for i in range(1, n - 1):
            xm = x_grid[i]
            dx_l = xm - x_grid[i - 1]
            dx_r = x_grid[i + 1] - xm
            dx = x_grid[i + 1] - x_grid[i - 1]

            eps_m = epsilon_r
            # 简化: 常数介电常数
            A[i, i - 1] = -2.0 * eps_m / (dx_l * dx)
            A[i, i] = 2.0 * eps_m / (dx_l * dx_r)
            A[i, i + 1] = -2.0 * eps_m / (dx_r * dx)
            rhs[i] = charge_density[i] / eps0

        phi = np.linalg.solve(A, rhs)
        return phi
