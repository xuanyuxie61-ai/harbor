"""
stability_analysis.py — Von Neumann 稳定性分析
================================================
种子项目映射:
  1097_catniplab (吸引子/Lyapunov) → 放大因子与Lyapunov指数
  034_asa082 (行列式) → 稳定性矩阵性质分析

Von Neumann分析:
  对误差做Fourier分解 e_j^n = G^n exp(ikjh)
  G(k) 为放大因子, 稳定性要求 |G(k)| <= 1 + O(dt)

  对 KG 方程 + Newmark-β:
    s = dt² * (k̃² + m²)
    (1+βs)G² - 2(1-(½-β)s)G + (1+(β-½)s) = 0
"""
import numpy as np


def dispersion_sigma(kh, fd_order=4):
    """散谱函数 σ(kh): D² exp(ikx) = -σ/h² exp(ikx)."""
    if fd_order == 2:
        return 2.0 * (1.0 - np.cos(kh))
    elif fd_order == 4:
        return (30.0 - 32.0 * np.cos(kh) + 2.0 * np.cos(2 * kh)) / 12.0
    elif fd_order == 6:
        return (210.0 - 270.0 * np.cos(kh) + 75.0 * np.cos(2 * kh)
                - 10.0 * np.cos(3 * kh)) / 180.0
    else:
        raise ValueError(f"fd_order={fd_order}")


def amplification_factor(k, h, dt, mass=0.0, fd_order=4, beta=0.0, gamma=0.5):
    """
    计算放大因子 G(k).

    Returns: (|G|, phase)
    """
    kh = k * h
    sigma = dispersion_sigma(kh, fd_order)
    s = dt**2 * (sigma / h**2 + mass**2)

    if abs(beta) < 1e-15:
        # 显式: G = (1 - s/2) ± sqrt((1-s/2)² - 1)
        a = 1.0 - s / 2.0
        disc = a**2 - 1.0
        G = np.zeros_like(k, dtype=complex)
        stable = np.abs(a) <= 1.0
        G[stable] = a[stable] + 1j * np.sqrt(np.maximum(-disc[stable], 0.0))
        unstable = ~stable
        d = np.maximum(disc[unstable], 0.0)
        G1 = a[unstable] + np.sqrt(d)
        G2 = a[unstable] - np.sqrt(d)
        G[unstable] = np.where(np.abs(G1) > np.abs(G2), G1, G2)
    else:
        # Newmark-β
        a2 = 1.0 + beta * s
        a1 = -2.0 * (1.0 - (0.5 - beta) * s)
        a0 = 1.0 + (beta - 0.5) * s
        disc = a1**2 - 4 * a2 * a0
        G = np.zeros_like(k, dtype=complex)
        stable = disc < 0
        r = -a1[stable] / (2 * a2[stable])
        im = np.sqrt(np.abs(disc[stable])) / (2 * np.abs(a2[stable]))
        G[stable] = r + 1j * im
        unstable = ~stable
        sd = np.maximum(disc[unstable], 0.0)
        G1 = (-a1[unstable] + np.sqrt(sd)) / (2 * a2[unstable])
        G2 = (-a1[unstable] - np.sqrt(sd)) / (2 * a2[unstable])
        G[unstable] = np.where(np.abs(G1) > np.abs(G2), G1, G2)

    return np.abs(G), np.angle(G)


def stability_limit(h, mass=0.0, fd_order=4):
    """计算最大稳定时间步长 dt_max."""
    if fd_order == 2:
        sigma_max = 4.0
    elif fd_order == 4:
        sigma_max = 64.0 / 12.0
    elif fd_order == 6:
        sigma_max = 565.0 / 180.0
    else:
        sigma_max = 4.0
    return 2.0 * h / np.sqrt(sigma_max + (mass * h)**2)


def lyapunov_exponent(k, h, dt, mass=0.0, fd_order=4):
    """Lyapunov指数: λ(k) = ln|G(k)|/dt."""
    G_mag, _ = amplification_factor(k, h, dt, mass, fd_order)
    return np.log(np.maximum(G_mag, 1e-30)) / dt


def cfl_number(h, dt, fd_order=4):
    """CFL数 = dt / dt_max."""
    dt_max = stability_limit(h, 0.0, fd_order)
    return dt / dt_max
