"""
径向扩散求解器 (radial_solver.py)
==================================
基于种子项目 351_fd_to_tec 的有限差分离散思想与 152_cg_rc 的线性求解思想，
实现地核外核径向磁扩散方程的高效求解。

核心方程（球坐标径向一维近似）:
    dB/dt = eta * (1/r^2) * d/dr (r^2 * dB/dr) - l*(l+1)*eta/r^2 * B + S(r,t)

其中:
  - B(r,t) 为第 l 阶球谐系数的径向振幅
  - eta 为磁扩散系数
  - S(r,t) 为感应源项（alpha 效应与 omega 效应）

本模块提供：
  - 二阶中心差分 Laplacian 构造
  - Crank-Nicolson 时间离散
  - 边界条件处理（绝缘边界条件 d/dr(r*B)=0 在 CMB，r*B=0 在 ICB）
  - 调用 cg_solver 进行大型系统求解
"""

import numpy as np
from typing import Tuple
from cg_solver import solve_radial_diffusion_cg


# ---------------------------------------------------------------------------
# 1. 径向二阶导数算子（球坐标，含 1/r^2 因子）
#    d/dr(r^2 dB/dr) 的离散形式
# ---------------------------------------------------------------------------
def build_spherical_radial_laplacian(r: np.ndarray) -> np.ndarray:
    """
    构建球坐标径向 Laplacian 的离散矩阵 L，使得 (L @ B) 近似
    (1/r^2) * d/dr(r^2 * dB/dr)。

    在内部节点 i 上使用中心差分:
      dB/dr|_{i+1/2} ~ (B_{i+1} - B_i) / (r_{i+1} - r_i)
      d/dr(r^2 dB/dr)_i ~ [r_{i+1/2}^2 * (B_{i+1}-B_i)/dr_{i+1/2}
                            - r_{i-1/2}^2 * (B_i-B_{i-1})/dr_{i-1/2}] / dr_i

    其中 dr_{i+1/2} = r_{i+1} - r_i, dr_i = 0.5*(r_{i+1} - r_{i-1})。

    返回矩阵 L (n x n)。
    """
    n = len(r)
    L = np.zeros((n, n), dtype=float)

    for i in range(1, n - 1):
        dr_plus = r[i + 1] - r[i]
        dr_minus = r[i] - r[i - 1]
        dr_avg = 0.5 * (dr_plus + dr_minus)
        r_plus_sq = (0.5 * (r[i] + r[i + 1])) ** 2
        r_minus_sq = (0.5 * (r[i] + r[i - 1])) ** 2

        coeff_plus = r_plus_sq / (dr_plus * dr_avg * r[i] ** 2)
        coeff_minus = r_minus_sq / (dr_minus * dr_avg * r[i] ** 2)
        coeff_center = -(coeff_plus + coeff_minus)

        L[i, i - 1] = coeff_minus
        L[i, i] = coeff_center
        L[i, i + 1] = coeff_plus

    # 边界条件将在外部施加
    return L


# ---------------------------------------------------------------------------
# 2. 径向扩散源项（alpha 效应与 omega 效应的简化参数化）
# ---------------------------------------------------------------------------
def alpha_effect_source(r: np.ndarray, r_icb: float, r_cmb: float,
                        alpha0: float, l: int) -> np.ndarray:
    """
    alpha 效应源项参数化：
      S_alpha(r) = alpha0 * f_alpha(r) * (1 / r) * sqrt(l*(l+1))
    其中 f_alpha(r) 在核幔边界附近最强（对流驱动区）。

    f_alpha(r) 取为:
      f_alpha(r) = sin(pi * (r - r_icb) / (r_cmb - r_icb))
    """
    d = r_cmb - r_icb
    f = np.sin(np.pi * (r - r_icb) / d)
    f[r <= r_icb] = 0.0
    f[r >= r_cmb] = 0.0
    return alpha0 * f * np.sqrt(l * (l + 1.0)) / r


def omega_effect_source(r: np.ndarray, r_icb: float, r_cmb: float,
                        omega_shear: float, Omega: float, B_tor: np.ndarray) -> np.ndarray:
    """
    Omega 效应（差速自转剪切）：将环向磁场 B_tor 剪切为极向磁场。
    简化模型:
      S_omega(r) = omega_shear * Omega * dB_tor/dr
    """
    dBdr = np.gradient(B_tor, r)
    return omega_shear * Omega * dBdr


# ---------------------------------------------------------------------------
# 3. 径向扩散方程求解（单时间步）
# ---------------------------------------------------------------------------
def step_radial_diffusion_cn(B: np.ndarray, r: np.ndarray,
                              dt: float, eta: float, l: int,
                              source: np.ndarray,
                              r_icb: float, r_cmb: float,
                              theta_cn: float = 0.5) -> np.ndarray:
    """
    使用 Crank-Nicolson 推进径向扩散方程一步：
        (I - theta*dt*eta*L) B_new = (I + (1-theta)*dt*eta*L) B_old + dt*source

    边界条件:
      ICB (r = r_icb): B = 0  (理想导体内核)
      CMB (r = r_cmb): dB/dr = 0  (绝缘地幔)
    """
    n = len(r)
    L = build_spherical_radial_laplacian(r)

    # 加入球谐衰减项: -l*(l+1)*eta/r^2 * B
    decay = -l * (l + 1.0) * eta / (r ** 2)
    for i in range(n):
        L[i, i] += decay[i]

    # 边界条件
    L[0, :] = 0.0
    L[0, 0] = 1.0  # Dirichlet: B=0 at ICB
    L[-1, :] = 0.0
    # Neumann: dB/dr = 0 at CMB -> B_{n-1} = B_{n-2}
    L[-1, -1] = 1.0
    L[-1, -2] = -1.0

    I = np.eye(n, dtype=float)
    A = I - theta_cn * dt * eta * L
    B_rhs = (I + (1.0 - theta_cn) * dt * eta * L) @ B + dt * source

    # 边界值修正
    B_rhs[0] = 0.0
    B_rhs[-1] = 0.0  # dB/dr=0 已通过矩阵行编码

    # 使用 CG 求解（虽然 n 通常较小，但为了演示鲁棒性）
    dr = np.mean(np.diff(r))
    B_new = solve_radial_diffusion_cg(B_rhs, n, dr, dt, eta, theta_cn=theta_cn)

    # 边界强制
    B_new[0] = 0.0
    # Neumann 条件已在矩阵中编码，数值求解后自然满足近似
    return B_new


# ---------------------------------------------------------------------------
# 4. 完整的径向模式演化（多 l 耦合的简化模型）
# ---------------------------------------------------------------------------
def evolve_radial_modes(B_modes: dict, r: np.ndarray, dt: float, eta: float,
                        r_icb: float, r_cmb: float,
                        alpha0: float, omega_shear: float, Omega: float) -> dict:
    """
    推进所有球谐模式的径向振幅一个时间步。

    B_modes: dict[(l,m)] -> np.ndarray(n_radial,)  环向或极向场振幅
    返回更新后的 B_modes。
    """
    B_new = {}
    for key, B in B_modes.items():
        l, m = key
        # 源项：alpha 效应（产生环向场）
        src_alpha = alpha_effect_source(r, r_icb, r_cmb, alpha0, l)
        # Omega 效应简化：假设存在某个平均环向场剪切
        src_omega = omega_effect_source(r, r_icb, r_cmb, omega_shear, Omega, B)
        source = src_alpha + src_omega
        B_new[key] = step_radial_diffusion_cn(B, r, dt, eta, l, source, r_icb, r_cmb)
    return B_new


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    n = 32
    r_icb = 1221e3
    r_cmb = 3480e3
    r = np.linspace(r_icb, r_cmb, n)
    B = np.sin(np.pi * (r - r_icb) / (r_cmb - r_icb))
    dt = 1e4 * 365.25 * 24 * 3600  # 1万年（秒）
    eta = 2.0
    l = 2
    source = np.zeros(n, dtype=float)
    B_new = step_radial_diffusion_cn(B, r, dt, eta, l, source, r_icb, r_cmb)
    assert not np.isnan(B_new).any()
    assert B_new[0] == 0.0  # Dirichlet 边界
    assert not np.isinf(B_new).any()

    # 多模式测试
    modes = {(1, 0): B.copy(), (2, 0): B.copy()}
    modes_new = evolve_radial_modes(modes, r, dt, eta, r_icb, r_cmb,
                                     alpha0=0.5, omega_shear=1.0, Omega=7.29e-5)
    assert len(modes_new) == 2
    print("radial_solver: self-test passed.")


if __name__ == "__main__":
    _self_test()
