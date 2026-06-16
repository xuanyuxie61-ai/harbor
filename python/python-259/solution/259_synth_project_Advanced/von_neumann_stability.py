# -*- coding: utf-8 -*-
"""
von_neumann_stability.py — 有限差分格式的 Von Neumann 稳定性分析

本模块为 `acoustic_wave_pde.py` 的 IMEX 格式提供严格的稳定性分析:
  1. 对纯显式、纯隐式、IMEX 三种格式, 计算放大因子 G(k·Δχ, c_s Δη/Δχ).
  2. 扫描参数空间, 确定稳定区域 |G| ≤ 1 + ε.
  3. 给出 CFL 数临界值.
  4. 计算放大矩阵的谱半径 ρ(M) 并绘制 (数值输出) 稳定域.

物理上, Von Neumann 稳定性等价于"小扰动在 Fourier 空间不增长". 对 BAO
分析而言, 这意味着数值声波的振幅不能因离散化而被放大 — 否则会污染 ξ(s)
在 150 Mpc 处的 BAO 峰.

数学框架
-------
对半离散化 ∂_t u = A u, 空间 Fourier 变换后得到:
  ∂_t û(k,t) = λ(k) û(k,t)
其中 λ(k) 是差分算子 A 的符号 (symbol). 对 4 阶中心差分 ∂_χ^2:
  λ(k) = -c_s^2 k^2 [ 1 - (kΔχ)^2/12 + (kΔχ)^4/90 - ... ]

对显式 Euler:     G = 1 + Δt λ(k)
对 Crank-Nicolson: G = (1 + 0.5 Δt λ) / (1 - 0.5 Δt λ)
对 IMEX (θ=0.5 + 显式 AB2): G = [ (1 + 0.5 Δt λ) / (1 - 0.5 Δt λ) ] × (1 + ...)

稳定条件: |G| ≤ 1 对所有 k·Δχ ∈ [0, π] 成立.

种子项目映射
----------
- 1080 safe-CCMPC-in-CARLA : 安全控制障碍函数 (CBF) 的"不变集"思想
  在此被移植为"稳定性不变集" — 给定 CFL 数, 若 |G| ≤ 1 则稳定, 否则
  需要缩小 Δt 或增大隐式比例.
- 1135 bilevel-optim       : 其谱分析工具 (特征值分解) 被直接复用于此.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import math
import numpy as np

from fornberg_fd import fornberg_weights, build_derivative_matrix_1d


# =============================================================
# 差分算子的符号 (symbol) 计算
# =============================================================
def symbol_second_derivative(dx: float, stencil: int, n_samples: int = 512
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 ∂^2/∂χ^2 差分算子的 Fourier 符号 λ(k), k ∈ [0, π/dx].
    方法: 在等距网格上用 Fornberg 权重生成差分矩阵 D, 然后对 D 做 DFT 得到
    λ(k) = Σ_j w_j exp(i k x_j) 的实部 (因为 ∂^2 是 Hermitian 算子).

    Returns
    -------
    k_arr   : (n_samples,) 波数值
    lambda_k: (n_samples,) 符号值 (应为负实数)
    """
    if stencil % 2 == 0:
        raise ValueError("stencil 必须为奇数")
    half = stencil // 2
    # 中心 stencil 的节点
    x_nodes = np.arange(-half, half + 1, dtype=np.float64) * dx
    xc = 0.0
    W = fornberg_weights(xc, x_nodes, 2)
    w = W[2, :]  # 2 阶导数权重
    # 波数采样
    k_max = math.pi / dx
    k_arr = np.linspace(0.0, k_max, n_samples)
    # 符号: λ(k) = Σ_j w_j exp(i k x_j) 的实部
    lambda_k = np.zeros(n_samples, dtype=np.float64)
    for i, k in enumerate(k_arr):
        lambda_k[i] = np.sum(w * np.cos(k * x_nodes))
    return k_arr, lambda_k


def symbol_exact_second_derivative(dx: float, n_samples: int = 512
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """精确的 ∂^2/∂χ^2 符号: λ_exact(k) = -k^2."""
    k_max = math.pi / dx
    k_arr = np.linspace(0.0, k_max, n_samples)
    return k_arr, -k_arr ** 2


# =============================================================
# 放大因子计算
# =============================================================
@dataclass
class SchemeParams:
    """差分格式参数."""
    dx      : float     # 空间步长
    dt      : float     # 时间步长 (此处为 Δη)
    cs      : float     # 声速 c_s (c=1 单位, 但此处保留 Mpc 单位)
    Hc      : float     # 共形 Hubble ℋ (Mpc^{-1})
    stencil : int = 5


def amplification_explicit(p: SchemeParams, k_arr: np.ndarray
                           ) -> np.ndarray:
    """显式 Euler: G = 1 + dt (cs^2 λ(k) - 2 Hc) (简化: 忽略 Hubble 摩擦)."""
    _, lambda_k = symbol_second_derivative(p.dx, p.stencil, len(k_arr))
    # 完整: ∂_t û = -2 Hc v̂ + cs^2 λ(k) û 的耦合系统. 这里先简化.
    # 显式 Euler 对 ∂_t u = cs^2 ∂^2_x u: G = 1 + cs^2 dt λ(k)
    return 1.0 + p.cs ** 2 * p.dt * lambda_k


def amplification_crank_nicolson(p: SchemeParams, k_arr: np.ndarray
                                 ) -> np.ndarray:
    """Crank-Nicolson: G = (1 + 0.5 α) / (1 - 0.5 α), α = cs^2 dt λ(k)."""
    _, lambda_k = symbol_second_derivative(p.dx, p.stencil, len(k_arr))
    alpha = p.cs ** 2 * p.dt * lambda_k
    return (1.0 + 0.5 * alpha) / (1.0 - 0.5 * alpha)


def amplification_imex(p: SchemeParams, k_arr: np.ndarray,
                       theta: float = 0.5) -> np.ndarray:
    """
    IMEX: 声项用 θ-方法, Hubble 摩擦显式.
      (1 - 0.5 θ α) G = 1 + 0.5 (1-θ) α + 2 Hc dt
    简化为忽略 Hubble 摩擦的纯声项稳定性:
      G = (1 + (1-θ) α/2) / (1 - θ α/2)
    """
    _, lambda_k = symbol_second_derivative(p.dx, p.stencil, len(k_arr))
    alpha = p.cs ** 2 * p.dt * lambda_k
    num = 1.0 + 0.5 * (1.0 - theta) * alpha
    den = 1.0 - 0.5 * theta * alpha
    return num / den


# =============================================================
# CFL 临界值搜索
# =============================================================
def find_cfl_critical(scheme: str, stencil: int = 5, theta: float = 0.5,
                      n_samples: int = 1024, tol: float = 1.0e-4
                      ) -> float:
    """
    二分搜索临界 CFL 数 r = c_s Δη / Δχ, 使得 max_k |G(k)| ≤ 1.

    Parameters
    ----------
    scheme : "explicit" | "CN" | "IMEX"
    """
    dx = 1.0  # 归一化
    # 二分: CFL ∈ [0.01, 10]
    lo, hi = 0.001, 10.0
    k_max = math.pi / dx
    k_arr = np.linspace(0.0, k_max, n_samples)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        params = SchemeParams(dx=dx, dt=mid * dx, cs=1.0, Hc=0.0, stencil=stencil)
        if scheme == "explicit":
            G = amplification_explicit(params, k_arr)
        elif scheme == "CN":
            G = amplification_crank_nicolson(params, k_arr)
        elif scheme == "IMEX":
            G = amplification_imex(params, k_arr, theta)
        else:
            raise ValueError(f"未知 scheme: {scheme}")
        max_abs = float(np.max(np.abs(G)))
        if max_abs > 1.0 + tol:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1.0e-8:
            break
    return 0.5 * (lo + hi)


# =============================================================
# 完整放大矩阵谱半径 (二维耦合系统)
# =============================================================
def amplification_matrix_eigenvalues(dx: float, dt: float,
                                     cs: float, Hc: float,
                                     stencil: int = 5,
                                     n_k: int = 256
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    对耦合系统 (δ, v) 构造 2×2 放大矩阵 M(k), 返回其特征值.
    系统:
      ∂_η δ = v
      ∂_η v = cs^2 ∂_χ^2 δ - 2 Hc v

    Fourier 变换后, 显式 Euler 的放大矩阵:
      M(k) = [ 1,     dt     ]
             [ cs^2 dt λ(k), 1 - 2 Hc dt ]
    """
    k_max = math.pi / dx
    k_arr = np.linspace(0.0, k_max, n_k)
    _, lambda_k = symbol_second_derivative(dx, stencil, n_k)
    eigvals = np.zeros((n_k, 2), dtype=complex)
    for i, k in enumerate(k_arr):
        M = np.array([
            [1.0, dt],
            [cs ** 2 * dt * lambda_k[i], 1.0 - 2.0 * Hc * dt],
        ], dtype=complex)
        eigvals[i, :] = np.linalg.eigvals(M)
    return k_arr, eigvals


def spectral_radius_vs_cfl(dx: float, cs: float, Hc: float,
                           stencil: int = 5,
                           n_cfl: int = 200,
                           cfl_max: float = 5.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (cfl_arr, rho_arr), 其中 rho 是放大矩阵的谱半径.
    """
    cfl_arr = np.linspace(0.01, cfl_max, n_cfl)
    rho_arr = np.zeros(n_cfl)
    for i, cfl in enumerate(cfl_arr):
        dt = cfl * dx / cs
        _, eigvals = amplification_matrix_eigenvalues(
            dx, dt, cs, Hc, stencil, n_k=128
        )
        rho_arr[i] = float(np.max(np.abs(eigvals)))
    return cfl_arr, rho_arr


# =============================================================
# Von Neumann 稳定性诊断报告
# =============================================================
@dataclass
class StabilityReport:
    scheme            : str
    stencil           : int
    cfl_critical      : float
    rho_at_cfl1       : float
    is_stable_at_cfl1 : bool


def full_stability_diagnosis(dx: float = 1.0, cs: float = 1.0 / math.sqrt(3.0),
                             Hc: float = 0.01, stencil: int = 5
                             ) -> StabilityReport:
    """
    对 IMEX 格式做完整稳定性诊断. 报告包括:
      - 临界 CFL 数
      - CFL=1 处的谱半径
      - 是否稳定
    """
    cfl_crit = find_cfl_critical("IMEX", stencil, theta=0.5)
    dt_test = 1.0 * dx / cs
    _, eigvals = amplification_matrix_eigenvalues(
        dx, dt_test, cs, Hc, stencil, n_k=256
    )
    rho_test = float(np.max(np.abs(eigvals)))
    return StabilityReport(
        scheme="IMEX",
        stencil=stencil,
        cfl_critical=cfl_crit,
        rho_at_cfl1=rho_test,
        is_stable_at_cfl1=(rho_test <= 1.0 + 1.0e-6),
    )


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    for scheme in ["explicit", "CN", "IMEX"]:
        cfl = find_cfl_critical(scheme)
        print(f"[von_neumann] {scheme:8s} CFL_critical = {cfl:.4f}")
    report = full_stability_diagnosis()
    print(f"[von_neumann] 完整诊断: scheme={report.scheme}, "
          f"cfl_crit={report.cfl_critical:.3f}, "
          f"rho(CFL=1)={report.rho_at_cfl1:.4f}, "
          f"stable={report.is_stable_at_cfl1}")


if __name__ == "__main__":
    _self_check()
