# -*- coding: utf-8 -*-
"""
coronal_energy.py
-----------------
日冕磁流体力学变分能量原理.

物理背景
--------
理想 MHD 中, 日冕磁环的平衡态使总势能泛函 W 取极值:

    W = W_mag + W_grav + W_thermal

其中:
    W_mag = (1/2 mu_0) int |B|^2 dV
    W_grav = - int rho (G M_sun / r) dV
    W_thermal = (1/(gamma-1)) int p dV

线性稳定性判据 (Bernstein et al. 1958):
    delta^2 W > 0   => 稳定
    delta^2 W < 0   => 不稳定 (kink/sausage/torus 不稳定性)

变分公式 (Tayler 1973):
    delta^2 W = (1/2) int [
        (1/mu_0) |curl (xi x B)|^2
        + gamma p |div xi|^2
        + (xi . grad p) div xi*
        - xi . grad(xi . grad p)
        + rho |g . xi|^2 (Rayleigh-Taylor)
    ] dV

本模块:
1) 实现一维环方向上的 delta^2 W 泛函离散
2) 通过变分最小化寻找最不稳定本征模
3) 判定 kink 不稳定性阈值 (Kruskal-Shafranov 条件推广)
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from solar_constants import VACUUM_PERMEABILITY, BOLTZMANN, PROTON_MASS


def magnetic_energy_density(b_field: np.ndarray) -> np.ndarray:
    """u_B = |B|^2 / (2 mu_0)."""
    return np.sum(b_field**2, axis=-1) / (2.0 * VACUUM_PERMEABILITY)


def thermal_energy_density(p: np.ndarray, gamma: float = 5.0 / 3.0) -> np.ndarray:
    """u_th = p / (gamma - 1)."""
    return p / (gamma - 1.0)


def gravitational_potential_density(rho: np.ndarray, r: np.ndarray,
                                    solar_mass: float = 1.989e30,
                                    grav_const: float = 6.674e-11) -> np.ndarray:
    """u_g = - rho G M_sun / r."""
    return -rho * grav_const * solar_mass / (r + 1.0e-12)


def total_potential_energy(rho: np.ndarray, p: np.ndarray,
                           b_field: np.ndarray, r: np.ndarray,
                           z_grid: np.ndarray,
                           gamma: float = 5.0 / 3.0) -> float:
    """integrate W over the loop axis (1D approximation)."""
    u_b = magnetic_energy_density(b_field)
    u_th = thermal_energy_density(p, gamma)
    u_g = gravitational_potential_density(rho, r)
    integrand = u_b + u_th + u_g
    return float(np.trapz(integrand, z_grid))


def second_variation_operator(z: np.ndarray, b0: np.ndarray,
                              p0: np.ndarray, rho0: np.ndarray,
                              gamma: float = 5.0 / 3.0) -> sparse.csr_matrix:
    """构造 delta^2 W 的二次型矩阵 (简化: 仅磁张力 + 热压):

    delta^2 W ~ (1/2) xi^T K xi

    K ~ - (B^2/mu_0) d^2/ds^2 + gamma p d^2/ds^2
      = (B^2/mu_0 - gamma p) * (-d^2/ds^2)
    """
    n = z.size
    h = z[1:] - z[:-1]

    # 等效刚度
    kappa_mag = b0**2 / VACUUM_PERMEABILITY
    kappa_th = gamma * p0
    kappa_total = kappa_mag + kappa_th   # 磁张力 + 热压恢复力

    # 构造 -d/ds [ kappa(s) d/ds ] 矩阵
    rows, cols, vals = [], [], []
    for i in range(1, n - 1):
        h_im1, h_i = h[i - 1], h[i]
        k_im1 = 0.5 * (kappa_total[i - 1] + kappa_total[i])
        k_ip = 0.5 * (kappa_total[i] + kappa_total[i + 1])
        coeff = 2.0 / (h_im1 + h_i)
        c_im1 = -coeff * k_im1 / h_im1
        c_i = coeff * (k_im1 / h_im1 + k_ip / h_i)
        c_ip1 = -coeff * k_ip / h_i
        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        vals += [c_im1, c_i, c_ip1]

    # 夹紧边界 xi=0 at endpoints
    rows += [0, n - 1]; cols += [0, n - 1]; vals += [1.0, 1.0]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def kink_instability_threshold(b_axial: float, b_azimuthal: float,
                               loop_length: float, radius: float) -> float:
    """Kruskal-Shafranov 条件推广:

    q = (r B_z) / (R_0 B_theta) > 1  => 稳定
    q < 1                              => 发生 kink (m=1)

    返回 q 值 (安全因子).
    """
    if abs(b_azimuthal) < 1.0e-20:
        return 1.0e10
    q = (radius * b_axial) / (loop_length * abs(b_azimuthal) / (2.0 * np.pi))
    return float(q)


def torus_instability_index(b_ext_power: float, current_height: float,
                            loop_radius: float) -> float:
    """Torus 不稳定性判据 (Kliem & Torok 2006):
    衰减指数 n = - d ln B_ext / d ln h
    n > n_crit ~ 1.5 => 发生 torus 不稳定性 (CME 触发).
    """
    n = b_ext_power
    return float(n)


def compute_unstable_mode(z: np.ndarray, b0: np.ndarray,
                          p0: np.ndarray, rho0: np.ndarray,
                          gamma: float = 5.0 / 3.0,
                          n_modes: int = 3) -> tuple:
    """求解最不稳定本征模 (最小本征值).

    若 lambda_min < 0, 则系统不稳定; 对应本征矢为最危险扰动.
    """
    K = second_variation_operator(z, b0, p0, rho0, gamma)
    # 对称化 (本问题 K 近似对称)
    K_sym = 0.5 * (K + K.T)
    n = K_sym.shape[0]
    try:
        evals, evecs = eigsh(K_sym, k=min(n_modes, n - 2), which="SA")
    except Exception:
        evals, evecs = np.linalg.eigh(K_sym.toarray())
        evals = evals[:n_modes]
        evecs = evecs[:, :n_modes]
    return evals, evecs


def krksll_shaf_local(q_safety: np.ndarray) -> dict:
    """沿环轴的局部 Kruskal-Shafranov 稳定性诊断."""
    return dict(
        q_min=float(q_safety.min()),
        q_max=float(q_safety.max()),
        q_mean=float(q_safety.mean()),
        unstable_fraction=float((q_safety < 1.0).mean()),
    )
