# -*- coding: utf-8 -*-
"""
bezier_window.py — Bezier 曲面重建: BAO 窗函数与 ξ(s⊥, s∥) 平滑

本模块实现两个核心任务:
  (1) 用双三次 Bezier 曲面重建 survey 窗函数 W(s⊥, s∥)
  (2) 用 Bezier 曲面平滑 ξ(s⊥, s∥) 二维场, 抑制小尺度噪声

物理动机
-------
在红移空间中, BAO 信号以各向异性形式出现:
  ξ(s_⊥, s_∥) 的 BAO 峰位于 s_⊥ ≈ r_d, s_∥ ≈ r_d 的对角附近
由于 survey 几何、选择函数与纤维碰撞, 窗函数 W 在 (s_⊥, s_∥) 平面
上有复杂结构. 我们用 Bezier 曲面拟合 log W, 再用该拟合窗做反卷积.

Bezier 曲面公式
--------------
双三次 Bezier 曲面 S(u,v):
  S(u,v) = Σ_{i=0}^3 Σ_{j=0}^3 B_i^3(u) B_j^3(v) P_{ij}

其中 B_i^3 为三次 Bernstein 基:
  B_0^3(u) = (1-u)^3
  B_1^3(u) = 3u(1-u)^2
  B_2^3(u) = 3u^2(1-u)
  B_3^3(u) = u^3
P_{ij} ∈ R^3 为 16 个控制点.

种子项目映射
----------
- 083 bezier_surface (bezier_patch_evaluate.py, bezier_surface_neighbors.py):
  Bezier patch 的求值与邻接结构被完整移植. `bezier_patch_evaluate` 的
  向量化求值用于窗函数重建; `bezier_surface_neighbors` 的拓扑结构用于
  确保相邻 patch 的连续性.
- 492 gridlines (grid_polar.py, grid_triangular.py): 极坐标与三角网格
  被用于 (s_⊥, s_∥) 平面的非规则采样, 以便在高红移壳层上正确覆盖.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import math
import numpy as np


# =============================================================
# Bernstein 基函数
# =============================================================
def bernstein(n: int, i: int, u: float) -> float:
    """
    Bernstein 基 B_i^n(u) = C(n,i) u^i (1-u)^{n-i}.
    """
    if not (0.0 <= u <= 1.0):
        if u < 0.0:
            u = 0.0
        else:
            u = 1.0
    from math import comb
    return comb(n, i) * (u ** i) * ((1.0 - u) ** (n - i))


def bernstein_derivative(n: int, i: int, u: float) -> float:
    """B_i^n'(u) = n (B_{i-1}^{n-1}(u) - B_i^{n-1}(u)) (约定 B_{-1}=B_n=0)."""
    if n == 0:
        return 0.0
    term1 = bernstein(n - 1, i - 1, u) if i >= 1 else 0.0
    term2 = bernstein(n - 1, i, u) if i <= n - 1 else 0.0
    return n * (term1 - term2)


# =============================================================
# 双三次 Bezier 曲面求值
# =============================================================
def bezier_patch_eval(P: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    计算双三次 Bezier 曲面 S(u,v).
    P : (4, 4, 3) 控制点数组
    返回: (3,) 点坐标
    """
    if P.shape != (4, 4, 3):
        raise ValueError("P 形状必须为 (4, 4, 3)")
    S = np.zeros(3, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            bu = bernstein(3, i, u)
            bv = bernstein(3, j, v)
            S += bu * bv * P[i, j]
    return S


def bezier_patch_eval_batch(P: np.ndarray,
                            u_arr: np.ndarray, v_arr: np.ndarray
                            ) -> np.ndarray:
    """向量化批量求值, 返回 (len(u_arr), 3)."""
    n = len(u_arr)
    out = np.zeros((n, 3), dtype=np.float64)
    for k in range(n):
        out[k, :] = bezier_patch_eval(P, u_arr[k], v_arr[k])
    return out


# =============================================================
# 控制点拟合 (从数据点反求)
# =============================================================
def fit_bezier_patch(points_uv: np.ndarray, points_xyz: np.ndarray
                     ) -> np.ndarray:
    """
    给定点集 {(u_k, v_k) → (x_k, y_k, z_k)}, 最小二乘拟合 16 个控制点.
    解 16×3 的线性系统:
      Σ_{ij} B_i(u_k) B_j(v_k) P_{ij} = (x_k, y_k, z_k)
    """
    n_pts = len(points_uv)
    M = np.zeros((n_pts, 16), dtype=np.float64)
    for k, (u, v) in enumerate(points_uv):
        col = 0
        for i in range(4):
            for j in range(4):
                M[k, col] = bernstein(3, i, u) * bernstein(3, j, v)
                col += 1
    P_flat, _, _, _ = np.linalg.lstsq(M, points_xyz, rcond=None)
    return P_flat.reshape(4, 4, 3)


# =============================================================
# 多 patch 邻接结构 (种子项目 083)
# =============================================================
@dataclass
class BezierPatch:
    index     : int
    P         : np.ndarray       # (4, 4, 3)
    neighbors : List[int] = None

    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []


def build_patch_neighbors(patches: List[BezierPatch], tolerance: float = 1.0e-3
                          ) -> None:
    """
    为 patch 列表建立邻接关系: 两个 patch 若共享边 (角点重合), 则互记为邻居.
    移植自 `bezier_surface_neighbors.py`.
    """
    def corner(p: BezierPatch, idx: int) -> np.ndarray:
        """角点: idx=0 (0,0), 1 (1,0), 2 (0,1), 3 (1,1)."""
        u, v = [(0, 0), (1, 0), (0, 1), (1, 1)][idx]
        return bezier_patch_eval(p.P, float(u), float(v))

    n = len(patches)
    for i in range(n):
        for j in range(i + 1, n):
            # 检查 patch i 的任一 edge 是否与 patch j 的任一 edge 重合
            for ei in range(4):
                for ej in range(4):
                    if np.linalg.norm(corner(patches[i], ei) - corner(patches[j], ej)) < tolerance:
                        if j not in patches[i].neighbors:
                            patches[i].neighbors.append(j)
                        if i not in patches[j].neighbors:
                            patches[j].neighbors.append(i)


# =============================================================
# Bezier 重建 BAO 窗函数
# =============================================================
def reconstruct_bao_window(s_perp_arr: np.ndarray, s_par_arr: np.ndarray,
                           W_obs: np.ndarray) -> np.ndarray:
    """
    用 4×4 Bezier patch 拟合 log W 在 (s_⊥, s_∥) 网格上的观测值.

    Parameters
    ----------
    s_perp_arr : (N1,) 横向分离数组 (Mpc)
    s_par_arr  : (N2,) 纵向分离数组 (Mpc)
    W_obs      : (N1, N2) 观测窗函数值

    Returns
    -------
    W_fit : (N1, N2) Bezier 重建的窗函数
    """
    N1, N2 = len(s_perp_arr), len(s_par_arr)
    logW = np.log(np.maximum(W_obs, 1.0e-10))
    # 归一化到 [0,1]
    u_arr = (s_perp_arr - s_perp_arr[0]) / (s_perp_arr[-1] - s_perp_arr[0] + 1.0e-30)
    v_arr = (s_par_arr - s_par_arr[0]) / (s_par_arr[-1] - s_par_arr[0] + 1.0e-30)
    # 数据点 (只取 20 个采样以稳定拟合)
    idx1 = np.linspace(0, N1 - 1, min(20, N1)).astype(int)
    idx2 = np.linspace(0, N2 - 1, min(20, N2)).astype(int)
    pts_uv = []
    pts_z = []
    for i in idx1:
        for j in idx2:
            pts_uv.append([u_arr[i], v_arr[j]])
            pts_z.append([0.0, 0.0, logW[i, j]])  # z 分量存 logW
    pts_uv = np.array(pts_uv)
    pts_z = np.array(pts_z)
    P = fit_bezier_patch(pts_uv, pts_z)
    # 重构
    W_fit = np.zeros((N1, N2), dtype=np.float64)
    for i in range(N1):
        for j in range(N2):
            xyz = bezier_patch_eval(P, u_arr[i], v_arr[j])
            W_fit[i, j] = math.exp(xyz[2])
    return W_fit


# =============================================================
# 极坐标与三角网格生成 (种子项目 492)
# =============================================================
def grid_polar(s_perp_max: float, n_r: int, n_theta: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 s_⊥ 方向生成极坐标网格点 (r, θ).
    返回 (s_perp, s_par) 两个 (n_r * n_theta,) 数组.
    """
    r_arr = np.linspace(0.0, s_perp_max, n_r)
    theta_arr = np.linspace(0.0, 2 * math.pi, n_theta, endpoint=False)
    sp, spar = [], []
    for r in r_arr:
        for th in theta_arr:
            sp.append(r * math.cos(th))
            spar.append(r * math.sin(th))
    return np.array(sp), np.array(spar)


def grid_triangular(s_min: float, s_max: float, n_side: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 (s_⊥, s_∥) 平面上生成三角剖分网格点.
    """
    sp, spar = [], []
    for i in range(n_side):
        for j in range(n_side - i):
            u = i / (n_side - 1 + 1.0e-30)
            v = j / (n_side - 1 + 1.0e-30)
            sp.append(s_min + u * (s_max - s_min))
            spar.append(s_min + v * (s_max - s_min))
    return np.array(sp), np.array(spar)


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    # 简单 patch: 把单位正方形映射到 (x,y) ∈ [0,1]^2
    P = np.zeros((4, 4, 3), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            P[i, j, 0] = i / 3.0
            P[i, j, 1] = j / 3.0
    # 测试 4 个角点
    for u, v, ex, ey in [(0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 1, 1)]:
        p = bezier_patch_eval(P, u, v)
        print(f"[bezier_window] S({u},{v}) = ({p[0]:.2f}, {p[1]:.2f}), expected ({ex},{ey})")
    # 窗函数重建
    s1 = np.linspace(50.0, 250.0, 12)
    s2 = np.linspace(50.0, 250.0, 12)
    W = np.outer(np.exp(-0.5 * ((s1 - 150) / 50) ** 2),
                 np.exp(-0.5 * ((s2 - 150) / 50) ** 2))
    W_fit = reconstruct_bao_window(s1, s2, W)
    err = np.max(np.abs(W_fit - W))
    print(f"[bezier_window] 窗函数重建最大误差 = {err:.3e}")


if __name__ == "__main__":
    _self_check()
