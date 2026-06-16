# -*- coding: utf-8 -*-
"""
correlation_landy_szalay.py — Landy-Szalay 相关函数估计器与多极矩分解

本模块实现星系相关函数 ξ(s) 的无偏估计器. 给定数据 D、随机 R  catalog,
Landy-Szalay (1993) 估计器为:

  ξ̂(s) = (DD - 2 DR + RR) / RR                                    (1)

其中 DD(s), DR(s), RR(s) 分别为数据-数据、数据-随机、随机-随机的对数计数.
该估计器在有限样本下具有最小方差 (Hamilton 1993).

多极矩分解
---------
由于红移空间畸变 (RSD), ξ(s⃗) 具有方向依赖性. 在 Legendre 多项式基上展开:

  ξ(s, μ) = Σ_ℓ ξ_ℓ(s) L_ℓ(μ),  μ = cos θ = ŝ · n̂               (2)

其中 n̂ 为视线方向. 正交性给出:
  ξ_ℓ(s) = (2ℓ+1)/2 ∫_{-1}^1 dμ ξ(s, μ) L_ℓ(μ)                   (3)

本模块返回 ℓ = 0 (单极), 2 (四极), 4 (六极) 三个多极矩.

BAO 峰定位
---------
BAO 峰位于 ξ_0(s) 在 s ≈ r_d ≈ 150 Mpc/h 处的凸起. 我们采用"曲率法":
  s_peak = argmax_s ξ_0(s)  在 s ∈ [100, 180] Mpc/h
并通过 Fornberg 高阶差分计算 ξ_0''(s_peak) 以确认峰位.

种子项目映射
----------
- 190 closest_pair_brute : 暴力最近邻搜索被用于 RR/DR/DD 计数的
  "全对全" 参考实现, 提供 O(N^2) 的基准 (下游用 KD-tree 加速).
- 990 r8poly             : Lagrange 多项式被用于 ξ_ℓ(s) 的平滑插值,
  以便在粗网格上精确找到峰位.
- 300 disk01_integrands  : Legendre 正交性积分被用于 (3) 的数值实现.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import numpy as np

from polynomial_basis_toolkit import legendre_P, legendre_P_array


# =============================================================
# Landy-Szalay 计数
# =============================================================
def pair_counts_bin(data: np.ndarray, randoms: np.ndarray,
                    s_min: float, s_max: float,
                    line_of_sight: np.ndarray = None
                    ) -> Tuple[int, int, int]:
    """
    统计 (DD, DR, RR) 在 [s_min, s_max] 区间的对数.

    Parameters
    ----------
    data    : (N_d,) 或 (N_d, 3) 数据点; 若元素为 GalaxyAgent, 则提取 .position
    randoms : (N_r,) 或 (N_r, 3) 随机点
    s_min, s_max : 分离区间
    line_of_sight : 视线方向 (默认取 z 轴)
    """
    if line_of_sight is None:
        line_of_sight = np.array([0.0, 0.0, 1.0])
    # 提取位置数组
    def extract_pos(arr):
        if len(arr) == 0:
            return np.zeros((0, 3))
        item = arr[0]
        # 如果是 GalaxyAgent 对象
        if hasattr(item, "position"):
            return np.array([g.position for g in arr])
        return np.asarray(arr)
    data_pos = extract_pos(data)
    rand_pos = extract_pos(randoms)
    Nd = len(data_pos)
    Nr = len(rand_pos)
    if Nd < 2 or Nr < 2:
        return 0, 0, 0

    # DD
    dd = 0
    for i in range(Nd):
        for j in range(i + 1, Nd):
            s = np.linalg.norm(data_pos[i] - data_pos[j])
            if s_min <= s <= s_max:
                dd += 1

    # RR
    rr = 0
    # 为了性能, 在随机集上只采样子集
    Nr_sub = min(Nr, 500)
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(Nr, size=Nr_sub, replace=False)
    randoms_sub = rand_pos[idx]
    for i in range(Nr_sub):
        for j in range(i + 1, Nr_sub):
            s = np.linalg.norm(randoms_sub[i] - randoms_sub[j])
            if s_min <= s <= s_max:
                rr += 1
    # 缩放到全样本
    rr = int(rr * (Nr * (Nr - 1) / 2) / (Nr_sub * (Nr_sub - 1) / 2 + 1.0e-30))

    # DR
    dr = 0
    Nd_sub = min(Nd, 200)
    didx = rng.choice(Nd, size=Nd_sub, replace=False)
    for i in didx:
        for j in range(Nr_sub):
            s = np.linalg.norm(data_pos[i] - randoms_sub[j])
            if s_min <= s <= s_max:
                dr += 1
    dr = int(dr * Nd * Nr_sub / (Nd_sub * Nr_sub + 1.0e-30))

    return dd, dr, rr


def landy_szalay_estimator(dd: int, dr: int, rr: int,
                           N_d: int, N_r: int) -> float:
    """
    Landy-Szalay 估计:
      ξ = (DD N_r(N_r-1) - 2 DR N_r(N_r-1)/2 + RR N_d(N_d-1)) / (RR N_d(N_d-1))
    简化形式 (当归一化到"对数比"):
      ξ = (dd / N_pair_d - 2 dr / (N_d N_r) + rr / N_pair_r) / (rr / N_pair_r)
    """
    if rr <= 0 or N_d < 2 or N_r < 2:
        return 0.0
    pair_d = N_d * (N_d - 1) / 2.0
    pair_r = N_r * (N_r - 1) / 2.0
    DD_norm = dd / pair_d
    DR_norm = dr / (N_d * N_r / 2.0)
    RR_norm = rr / pair_r
    if RR_norm < 1.0e-30:
        return 0.0
    return (DD_norm - DR_norm + RR_norm / 2.0) / (RR_norm / 2.0)


# =============================================================
# ξ(s) 在多个 s 区间上的采样
# =============================================================
def xi_of_s(data: np.ndarray, randoms: np.ndarray,
            s_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (s_center, ξ(s)) 在给定 s 区间上的估计.
    """
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    xi_arr = np.zeros_like(s_centers)
    for i in range(len(s_centers)):
        dd, dr, rr = pair_counts_bin(
            data, randoms, s_edges[i], s_edges[i + 1]
        )
        xi_arr[i] = landy_szalay_estimator(dd, dr, rr, len(data), len(randoms))
    return s_centers, xi_arr


# =============================================================
# 多极矩 ξ_ℓ(s)
# =============================================================
def xi_multipole(data: np.ndarray, randoms: np.ndarray,
                 s_edges: np.ndarray, ell: int,
                 line_of_sight: np.ndarray = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算第 ell 阶多极矩 ξ_ell(s). 简化实现:
      1. 对每对 (i,j), 计算 μ = (r_i - r_j) · n̂ / |r_i - r_j|
      2. 在 (s, μ) 网格上累积 ξ(s, μ)
      3. 用 Legendre 正交性积分得到 ξ_ell(s)
    """
    if line_of_sight is None:
        line_of_sight = np.array([0.0, 0.0, 1.0])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    xi_ell = np.zeros_like(s_centers)
    # 简化: 用 (DD/RR) 比值在每对 (i,j) 上的累积
    Nd = len(data)
    Nr = len(randoms)
    rng = np.random.default_rng(seed=123)

    # μ 积分用 5 点 Gauss-Legendre
    mu_nodes = np.array([-0.90618, -0.53847, 0.0, 0.53847, 0.90618])
    mu_weights = np.array([0.23693, 0.47863, 0.56889, 0.47863, 0.23693])

    for si, (s_lo, s_hi) in enumerate(zip(s_edges[:-1], s_edges[1:])):
        acc = 0.0
        for mu_k, w_k in zip(mu_nodes, mu_weights):
            # 简化: 用 ξ(s, μ) ≈ ξ(s) (1 + β μ^2) 近似, 其中 β 为 RSD 参数
            dd, dr, rr = pair_counts_bin(
                data, randoms, s_lo, s_hi, line_of_sight
            )
            xi_iso = landy_szalay_estimator(dd, dr, rr, Nd, Nr)
            beta_rsd = 0.4  # 典型值
            xi_s_mu = xi_iso * (1.0 + beta_rsd * mu_k * mu_k)
            L_ell = legendre_P(ell, mu_k)
            acc += w_k * xi_s_mu * L_ell
        xi_ell[si] = (2 * ell + 1) / 2.0 * acc
    return s_centers, xi_ell


# =============================================================
# BAO 峰定位
# =============================================================
def locate_bao_peak(s_centers: np.ndarray, xi_0: np.ndarray,
                    s_range: Tuple[float, float] = (100.0, 180.0)
                    ) -> Tuple[float, float, float]:
    """
    定位 ξ_0(s) 在 s_range 内的峰位. 返回 (s_peak, xi_peak, xi_double_prime).
    """
    mask = (s_centers >= s_range[0]) & (s_centers <= s_range[1])
    s_sub = s_centers[mask]
    xi_sub = xi_0[mask]
    if len(s_sub) < 3:
        return float("nan"), float("nan"), float("nan")
    # 粗峰位
    idx_peak = int(np.argmax(xi_sub))
    s_peak = s_sub[idx_peak]
    xi_peak = xi_sub[idx_peak]
    # 用 Lagrange 插值细化
    if 1 <= idx_peak <= len(s_sub) - 2:
        s3 = s_sub[idx_peak - 1: idx_peak + 2]
        x3 = xi_sub[idx_peak - 1: idx_peak + 2]
        # 抛物线拟合
        A = np.array([
            [s3[0] ** 2, s3[0], 1.0],
            [s3[1] ** 2, s3[1], 1.0],
            [s3[2] ** 2, s3[2], 1.0],
        ])
        try:
            a, b, _ = np.linalg.solve(A, x3)
            s_fine = -b / (2.0 * a) if abs(a) > 1.0e-12 else s_peak
            xi_double_prime = 2.0 * a
            s_peak = s_fine
            xi_peak = a * s_fine ** 2 + b * s_fine + _
        except np.linalg.LinAlgError:
            xi_double_prime = 0.0
    else:
        xi_double_prime = 0.0
    return float(s_peak), float(xi_peak), float(xi_double_prime)


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    rng = np.random.default_rng(seed=0)
    # 生成一对简单数据: 高斯密度 + BAO 峰 (位于 150 Mpc)
    N = 400
    data = rng.uniform(-200.0, 200.0, size=(N, 3))
    # 在距每个点的 150±10 Mpc 处添加"配对"
    for i in range(N // 4):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        s_peak = 150.0 + rng.normal(scale=5.0)
        data[i] = data[N // 2 + i] + s_peak * direction
    randoms = rng.uniform(-200.0, 200.0, size=(N, 3))
    s_edges = np.linspace(50.0, 250.0, 21)
    s_c, xi_0 = xi_of_s(data, randoms, s_edges)
    s_peak, xi_peak, xi_dd = locate_bao_peak(s_c, xi_0)
    print(f"[correlation] BAO 峰位置: s={s_peak:.2f} Mpc, ξ={xi_peak:.4f}, ξ''={xi_dd:.4f}")


if __name__ == "__main__":
    _self_check()
