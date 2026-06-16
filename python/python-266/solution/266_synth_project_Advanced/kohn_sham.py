"""
kohn_sham.py — Kohn-Sham 哈密顿量与特征值求解
================================================

本模块实现 1D Kohn-Sham 方程的离散化和特征值求解。
融合种子项目:
  - 362_fd1d_heat_steady: 稀疏三对角系统求解
  - 271_dg1d_advection: 谱方法特征值分析
  - 302_disk01_rule: Jacobi 矩阵特征值求积节点

核心物理:
  Kohn-Sham 方程 (原子单位 ℏ = m_e = 1):
    [-½ d²/dx² + V_eff(x)] φ_i(x) = ε_i φ_i(x)

  其中有效势:
    V_eff(x) = V_ext(x) + V_H(x) + V_xc(x)

  在 FD 网格上离散:
    H = T_FD + diag(V_eff)

  其中 T_FD 为 FD 动能矩阵 (fd_operators.py),
  diag(V_eff) 为有效势的对角矩阵。

  特征值问题:
    H · c_i = ε_i · c_i  (矩阵已是正交基)

  本征值 ε_i 为能带能量, 本征向量 c_i 为 FD 网格上的波函数。
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, Dict, Optional
from physical_constants import fd_coefficients_2nd
from fd_operators import build_laplacian_matrix, build_kinetic_matrix


# ============================================================
# Kohn-Sham 哈密顿量构建
# ============================================================

class KohnShamHamiltonian:
    """
    Kohn-Sham 哈密顿量 H = T + V_eff。

    对于 1D 周期系统, 在每个 k 点上:
      H(k) = T_FD(k) + diag(V_eff)

    其中 T_FD(k) 为包含 Bloch 相位的 FD 动能矩阵,
    V_eff 为实空间有效势向量。

    矩阵性质:
    - Hermitian (实对称当 k=0 或 k=π/a)
    - 带状 (半带宽 p+1)
    - 条件数 κ(H) ~ O(N²/dx²) ~ O(N^{2+2p})
    """

    def __init__(self, n_grid: int, dx: float, fd_order: int,
                 lattice_constant: float):
        self.n_grid = n_grid
        self.dx = dx
        self.fd_order = fd_order
        self.a = lattice_constant

    def build_hamiltonian(self, v_eff: np.ndarray,
                           k_point: float) -> np.ndarray:
        """
        构建 KS 哈密顿矩阵。

        H(k) = T(k) + diag(V_eff)

        其中:
          T(k) = -½ L(k)   (L 为 Bloch 边界条件的 Laplacian)
          V_eff 为实空间有效势

        Parameters
        ----------
        v_eff : np.ndarray, shape (N,)
            有效势 V_eff(x_i)
        k_point : float
            k 点坐标

        Returns
        -------
        H : np.ndarray, shape (N, N), complex
            KS 哈密顿矩阵
        """
        if len(v_eff) != self.n_grid:
            raise ValueError(
                f"V_eff 长度 {len(v_eff)} ≠ 网格点数 {self.n_grid}")

        # 动能矩阵
        T = build_kinetic_matrix(self.n_grid, self.dx, self.fd_order,
                                  k_point, self.a)

        # 势能 (对角)
        V = np.diag(v_eff.astype(complex))

        return T + V

    def solve_eigenproblem(self, H: np.ndarray,
                            n_eigenvalues: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 Hermitian 特征值问题 H·c = ε·c。

        使用 scipy.linalg.eigh (LAPACK 的 zheevd),
        时间复杂度 O(N³) 对于稠密矩阵。

        对于大规模问题, 可以使用:
        - 迭代方法 (Lanczos/Arnoldi): O(kN²) 求 k 个最低本征值
        -  Davidson 方法: DFT 中标准的大规模特征值求解器

        Parameters
        ----------
        H : np.ndarray, shape (N, N)
            Hermitian 矩阵
        n_eigenvalues : int, optional
            需要的本征值数量 (默认全部)

        Returns
        -------
        eigenvalues : np.ndarray
            本征值 (实数, 已排序)
        eigenvectors : np.ndarray, shape (N, n_eigenvalues)
            本征向量 (列向量)
        """
        if n_eigenvalues is None:
            n_eigenvalues = self.n_grid

        try:
            eigenvalues, eigenvectors = la.eigh(
                H, subset_by_index=[0, n_eigenvalues - 1])
        except la.LinAlgError:
            # 退化情况: 使用 general 求解器
            eigenvalues, eigenvectors = la.eig(H)
            idx = np.argsort(eigenvalues.real)
            eigenvalues = eigenvalues[idx].real[:n_eigenvalues]
            eigenvectors = eigenvectors[:, idx][:, :n_eigenvalues]

        # 确保本征值排序
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def check_hermiticity(self, H: np.ndarray) -> float:
        """
        检查矩阵的 Hermiticity 偏差。

        ||H - H†||_F / ||H||_F

        应该为 0 (或机器精度)。非零值表明 Bloch BC 实现有误。
        """
        diff = H - H.conj().T
        return np.linalg.norm(diff) / max(np.linalg.norm(H), 1e-30)


# ============================================================
# 完整 KS 求解 (单 k 点)
# ============================================================

def solve_kohn_sham_at_kpoint(v_eff: np.ndarray, k_point: float,
                                n_grid: int, dx: float,
                                fd_order: int, a: float,
                                n_bands: int
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个 k 点上求解 KS 方程。

    流程:
    1. 构建 H(k) = T_FD(k) + V_eff
    2. 检查 Hermiticity
    3. 求解特征值问题
    4. 返回最低 n_bands 个本征值和本征向量

    Parameters
    ----------
    v_eff : np.ndarray
        有效势
    k_point : float
        k 点
    n_grid : int
        网格点数
    dx : float
        网格间距
    fd_order : int
        FD 半带宽
    a : float
        晶格常数
    n_bands : int
        计算的能带数

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_bands,)
        能带能量
    eigenvectors : np.ndarray, shape (n_grid, n_bands)
        波函数 (FD 网格上)
    """
    if n_bands > n_grid:
        raise ValueError(
            f"能带数 {n_bands} 超过网格点数 {n_grid}")

    ks = KohnShamHamiltonian(n_grid, dx, fd_order, a)
    H = ks.build_hamiltonian(v_eff, k_point)

    # 验证 Hermiticity
    herm_error = ks.check_hermiticity(H)
    if herm_error > 1e-10:
        # 强制 Hermitian (对称化)
        H = 0.5 * (H + H.conj().T)

    eigenvalues, eigenvectors = ks.solve_eigenproblem(H, n_bands)

    return eigenvalues, eigenvectors


# ============================================================
# 全 BZ 能带求解
# ============================================================

def solve_all_bands(v_eff: np.ndarray, kpoints: np.ndarray,
                     n_grid: int, dx: float, fd_order: int,
                     a: float, n_bands: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    在所有 k 点上求解 KS 方程。

    Parameters
    ----------
    v_eff : np.ndarray
        有效势
    kpoints : np.ndarray, shape (n_kpoints,)
        k 点列表
    n_grid : int
        网格点数
    dx : float
        网格间距
    fd_order : int
        FD 半带宽
    a : float
        晶格常数
    n_bands : int
        能带数

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_kpoints, n_bands)
        能带能量 ε_{n,k}
    eigenvectors : np.ndarray, shape (n_kpoints, n_grid, n_bands)
        波函数 ψ_{n,k}(x)
    """
    n_kpoints = len(kpoints)
    all_eigenvalues = np.zeros((n_kpoints, n_bands))
    all_eigenvectors = np.zeros((n_kpoints, n_grid, n_bands), dtype=complex)

    for ik, k in enumerate(kpoints):
        evals, evecs = solve_kohn_sham_at_kpoint(
            v_eff, k, n_grid, dx, fd_order, a, n_bands)
        all_eigenvalues[ik, :] = evals
        all_eigenvectors[ik, :, :] = evecs

    return all_eigenvalues, all_eigenvectors


# ============================================================
# 电子密度计算
# ============================================================

def compute_electron_density(eigenvalues: np.ndarray,
                              eigenvectors: np.ndarray,
                              weights: np.ndarray,
                              mu: float,
                              temperature: float,
                              dx: float
                              ) -> Tuple[np.ndarray, float]:
    """
    从 KS 本征态计算电子密度。

    n(x) = 2 · Σ_{n,k} w_k · f(ε_{nk}, μ, T) · |ψ_{nk}(x)|²

    其中:
    - 因子 2 来自自旋简并
    - f 为 Fermi-Dirac 分布
    - w_k 为 k 点权重

    同时检查密度归一化:
      ∫₀ᵃ n(x) dx = N_electrons

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_kpoints, n_bands)
        本征值
    eigenvectors : np.ndarray, shape (n_kpoints, n_grid, n_bands)
        本征向量
    weights : np.ndarray, shape (n_kpoints,)
        k 点权重
    mu : float
        化学势
    temperature : float
        电子温度
    dx : float
        网格间距

    Returns
    -------
    density : np.ndarray, shape (n_grid,)
        电子密度
    n_electrons : float
        总电子数 (用于检验归一化)
    """
    from xc_functionals import fermi_dirac_distribution

    n_kpoints, n_grid, n_bands = eigenvectors.shape
    density = np.zeros(n_grid)

    for ik in range(n_kpoints):
        for ib in range(n_bands):
            # Fermi 占据数
            f = fermi_dirac_distribution(
                np.array([eigenvalues[ik, ib]]), mu, temperature)[0]

            # |ψ|² 贡献
            psi_sq = np.abs(eigenvectors[ik, :, ib]) ** 2
            density += 2.0 * weights[ik] * f * psi_sq

    n_electrons = np.sum(density) * dx
    return density, n_electrons


# ============================================================
# 总能计算
# ============================================================

def compute_total_energy(eigenvalues: np.ndarray,
                           weights: np.ndarray,
                           density: np.ndarray,
                           v_ext: np.ndarray,
                           v_hartree: np.ndarray,
                           v_xc: np.ndarray,
                           e_xc: float,
                           dx: float) -> Dict[str, float]:
    """
    计算 DFT 总能。

    总能表达:
      E_total = E_band - E_dc + E_H + E_xc + E_ion

    其中:
      E_band = Σ_{n,k} w_k · f_{nk} · ε_{nk}  (能带能量)
      E_dc = ∫ n(x) [V_H(x)/2 + V_xc(x)] dx  (双计数修正)
      E_H = ½ ∫ n(x) V_H(x) dx               (Hartree 能量)
      E_xc = ∫ n(x) ε_xc(n(x)) dx            (XC 能量)
      E_ion = ∫ n(x) V_ext(x) dx              (外势能)

    简化 (V_ext 包含离子势):
      E_total = E_band - ½ ∫ n·V_H dx - ∫ n·V_xc dx + E_xc

    Parameters
    ----------
    eigenvalues : np.ndarray
        本征值 (n_kpoints, n_bands)
    weights : np.ndarray
        k 点权重
    density : np.ndarray
        电子密度
    v_ext : np.ndarray
        外势
    v_hartree : np.ndarray
        Hartree 势
    v_xc : np.ndarray
        XC 势
    e_xc : float
        XC 能量
    dx : float
        网格间距

    Returns
    -------
    energies : dict
        各能量分量
    """
    from xc_functionals import fermi_dirac_distribution

    # 能带能量 (需要 Fermi 权重, 这里用简化版本)
    e_band = np.sum(weights[:, np.newaxis] * eigenvalues) * 2.0

    # 外势能
    e_ext = np.sum(density * v_ext) * dx

    # Hartree 能量
    e_hartree = 0.5 * np.sum(density * v_hartree) * dx

    # XC 双计数修正
    e_xc_dc = np.sum(density * v_xc) * dx

    # 总能
    e_total = e_band - e_hartree - e_xc_dc + e_xc

    return {
        'total': e_total,
        'band': e_band,
        'external': e_ext,
        'hartree': e_hartree,
        'xc': e_xc,
        'xc_dc': e_xc_dc,
    }
