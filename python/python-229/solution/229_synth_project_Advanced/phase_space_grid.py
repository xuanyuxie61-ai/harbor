"""
phase_space_grid.py
===================

来源: 067_ball_grid
-------------------
原项目通过八分象限反射在 3D 球内生成均匀网格, 利用关系
    x^2 + y^2 + z^2 <= r^2
进行球面边界过滤。

物理重构 (高能物理相空间)
-------------------------
在高能碰撞实验中, 每个末态粒子的动量 p = (px, py, pz) 位于一个以
束流轴为 z 轴、横动量 pT 为径向变量的三维动量空间球壳中:

    p^2 = px^2 + py^2 + pz^2 <= p_max^2

对无质量粒子 (E ≈ |p|), 能量 E 与动量模等价。相空间体积元

    d^3p = p^2 dp dΩ = p^2 dp d(cosθ) dφ

在球坐标下展开后, 均匀采样球内等价于对 (p, cosθ, φ) 进行适当权重的网格化。

本模块用途
---------
1. 构造探测器接受度 (acceptance) 采样的相空间网格
2. 为蒙特卡罗积分提供 (px, py, pz) 节点
3. 计算每个相空间单元的有效体积 (Jacobian × Δp Δcosθ Δφ)

数学细节
--------
设 N_r 为径向分段数, N_mu 为 cosθ 分段数, N_phi 为 φ 分段数.
总节点数 = 8 × (octant 内球内网格数) - 重叠修正

对每个节点 (px_i, py_j, pz_k), 物理量:
    E = √(p^2 + m^2)                       (质壳关系)
    η = -ln tan(θ/2) = 0.5 ln((|p|+pz)/(|p|-pz))   (赝快度)
    φ = atan2(py, px)                      (方位角)
    pT = √(px^2 + py^2)                   (横动量)
"""

from __future__ import annotations
from typing import List, Tuple
import math


# ===========================================================================
#            球内均匀网格 (3D ball grid, 八分象限反射)
# ===========================================================================

def ball_grid_3d(radius: float, n_per_axis: int) -> List[Tuple[float, float, float]]:
    """
    生成半径为 `radius` 的三维实心球内均匀网格点.

    算法 (原项目 067_ball_grid 移植):
        - 在每一轴上取 2n+1 个等距点: x_i = -r + i * 2r/(2n+1), i = 0..2n
        - 遍历所有 (x, y, z) 组合, 过滤 x^2 + y^2 + z^2 <= r^2
        - 利用八分象限对称性: 仅遍历非负 (x, y, z), 然后反射

    返回: [(x, y, z), ...] 列表
    """
    if radius <= 0.0:
        return []
    if n_per_axis < 1:
        return []

    spacing = 2.0 * radius / (2 * n_per_axis + 1)
    r2 = radius * radius

    # 仅遍历第一象限
    pts_octant = []
    for i in range(n_per_axis + 1):
        x = i * spacing
        x2 = x * x
        if x2 > r2:
            continue
        for j in range(n_per_axis + 1):
            y = j * spacing
            s2 = x2 + y * y
            if s2 > r2:
                continue
            for k in range(n_per_axis + 1):
                z = k * spacing
                if s2 + z * z <= r2:
                    pts_octant.append((x, y, z))

    # 反射到其他 7 个象限 (去除坐标轴上的重复)
    result = []
    seen = set()

    def _key(px, py, pz):
        # 量化到 1e-12 以避免浮点重复
        return (round(px, 12), round(py, 12), round(pz, 12))

    for (x, y, z) in pts_octant:
        for sx in ([x] if x == 0.0 else [x, -x]):
            for sy in ([y] if y == 0.0 else [y, -y]):
                for sz in ([z] if z == 0.0 else [z, -z]):
                    k = _key(sx, sy, sz)
                    if k not in seen:
                        seen.add(k)
                        result.append((sx, sy, sz))
    return result


def ball_grid_count(radius: float, n_per_axis: int) -> int:
    """
    球内网格点数的解析估计:
        N ≈ (4π/3) (r / h)^3,  h = 2r / (2n+1)
           = (4π/3) ((2n+1)/2)^3
           = (π/6) (2n+1)^3
    实际计数略少 (边界截断).
    """
    n = 2 * n_per_axis + 1
    return max(0, int(round(math.pi / 6.0 * n * n * n)))


# ===========================================================================
#          动量空间 → 实验观测变量 转换 (px, py, pz) → (pT, eta, phi)
# ===========================================================================

def momentum_to_observables(
    pts: List[Tuple[float, float, float]],
    mass: float = 0.0,
) -> List[Tuple[float, float, float, float]]:
    """
    将 (px, py, pz) 转换为 (E, pT, eta, phi).

    质壳关系:
        E = sqrt(p^2 + m^2)

    赝快度:
        η = 0.5 * ln((|p| + pz) / (|p| - pz))
        当 |pz| → |p| 时 η → ±∞ (共线极限), 需安全截断.

    方位角:
        φ = atan2(py, px) ∈ (-π, π]

    横动量:
        pT = sqrt(px^2 + py^2)
    """
    out = []
    m2 = mass * mass
    for (px, py, pz) in pts:
        p2 = px * px + py * py + pz * pz
        p = math.sqrt(p2)
        E = math.sqrt(p2 + m2)
        pT = math.sqrt(px * px + py * py)
        phi = math.atan2(py, px)

        # 赝快度: 安全处理共线极限
        if p < 1e-300:
            eta = 0.0
        else:
            ratio = (p + pz) / (p - pz) if p != pz else 1e300
            if ratio <= 0.0:
                eta = 0.0
            elif ratio > 1e300:
                eta = 0.5 * math.log(1e300)  # ≈ 345
            else:
                eta = 0.5 * math.log(ratio)
            # 截断到物理范围
            eta = max(-20.0, min(20.0, eta))
        out.append((E, pT, eta, phi))
    return out


# ===========================================================================
#          探测器接受度掩模 (基于 pT, eta 窗口)
# ===========================================================================

def acceptance_mask(
    obs: List[Tuple[float, float, float, float]],
    pt_min: float = 0.5,
    pt_max: float = 100.0,
    eta_min: float = -2.5,
    eta_max: float = 2.5,
) -> List[bool]:
    """
    对每个观测点返回是否落入探测器接受度窗口.

    典型 LHC 通用探测器 (ATLAS/CMS):
        pT > 0.5 GeV,  |η| < 2.5  (径迹)
        pT > 20  GeV,  |η| < 2.5  (轻子触发)

    边界处理: 等于边界视为接受.
    """
    mask = []
    for (E, pT, eta, phi) in obs:
        accept = (pt_min <= pT <= pt_max) and (eta_min <= eta <= eta_max)
        mask.append(accept)
    return mask


def phase_space_weights(
    pts: List[Tuple[float, float, float]],
    radius: float,
    n_per_axis: int,
) -> List[float]:
    """
    为球内网格分配相空间权重.

    严格均匀采样时每个网格点权重相同 = 球体积 / 点数.
    球体积: V = (4π/3) r^3.

    对于重要性采样 (如 d^3p ∝ p^2 dp), 可加权为 w_i ∝ |p_i|^2,
    此处返回均匀权重 (供蒙特卡罗积分归一化).
    """
    n = len(pts)
    if n == 0:
        return []
    V = 4.0 / 3.0 * math.pi * radius ** 3
    w = V / n
    return [w] * n


# ===========================================================================
#         球面高斯求积 (角度积分, 用于响应矩阵的角向分量)
# ===========================================================================

def gauss_legendre_cos_theta(n_points: int) -> Tuple[List[float], List[float]]:
    """
    cosθ 方向的 Gauss-Legendre 求积节点与权重,
    区间 [-1, 1] (覆盖全立体角).

    Newton 法求 P_n 根, 用于响应矩阵角向分量的精确积分:
        ∫_{-1}^{1} f(cosθ) d(cosθ) ≈ Σ_i w_i f(μ_i)
    """
    if n_points < 1:
        return [], []
    nodes = []
    weights = []
    m = (n_points + 1) // 2
    for i in range(m):
        # 初值: Tricchenbrodt 近似
        z = math.cos(math.pi * (i + 0.75) / (n_points + 0.5))
        for _ in range(50):
            p0, p1 = 1.0, z
            for k in range(2, n_points + 1):
                p0, p1 = p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
            dp = n_points * (z * p1 - p0) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / dp
            if abs(z - z1) < 1e-15:
                break
        nodes.append(z)
        weights.append(2.0 / ((1.0 - z * z) * dp * dp))
    # 对称补全
    full_nodes = []
    full_weights = []
    for (mu, w) in zip(nodes, weights):
        if abs(mu) > 1e-14:
            full_nodes.append(mu)
            full_weights.append(w)
            full_nodes.append(-mu)
            full_weights.append(w)
        else:
            full_nodes.append(mu)
            full_weights.append(w)
    return full_nodes, full_weights


__all__ = [
    "ball_grid_3d", "ball_grid_count",
    "momentum_to_observables", "acceptance_mask",
    "phase_space_weights", "gauss_legendre_cos_theta",
]
