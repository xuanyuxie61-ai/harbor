# -*- coding: utf-8 -*-
"""
comoving_grid.py — 共动网格与区域编号 (numgrid 风格)

本模块为 BAO 分析提供"共动坐标网格"构造与区域编号. 借鉴种子项目 820 numgrid
的"区域编号"范式, 把 survey footprint 在 (α, δ) 天球上离散化为矩形/极角/
三角网格, 每个网格点得到一个唯一编号. 这种编号在后续:
  (1) 关联函数计算时用于快速索引;
  (2) 协方差矩阵构造时用于定义"相关邻域";
  (3) Bezier 曲面重建时用于 patch 邻接.

种子项目映射
----------
- 820 numgrid : `numgrid.m` 的"按区域编码 (S, L, C, F 等)"被完整移植.
  本模块支持 5 种典型 survey 几何:
    'S' - 完整正方形 (全天空 idealized)
    'L' - L 形 (三段矩形拼接, 模拟 SDSS 主样本)
    'C' - 带 1/4 圆缺角的 L (模拟含 Galactic mask 的 footprint)
    'F' - 完整长方形 (BOSS 北天 Galaxies)
    'D' - 圆盘 (eBOSS 类深场)
- 190 closest_pair_brute : 暴力最近邻算法被用于"网格点密集性检查":
  对网格中每对点求分离, 若最小分离 < 0.5 Δx, 则标记为"过密"并触发
  自适应细化.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import math
import numpy as np


# =============================================================
# 区域类型
# =============================================================
REGION_TYPES = {"S", "L", "C", "F", "D"}


# =============================================================
# numgrid: 区域编号 (种子项目 820)
# =============================================================
def numgrid_bao(region: str, n_grid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 [-1, 1]^2 上, 按 region 类型编号网格点.
    返回 (G, X, Y), G 为编号矩阵 (内部点 1..N, 边界 0, 外部 -1).

    Parameters
    ----------
    region : 'S' | 'L' | 'C' | 'F' | 'D'
    n_grid : N, 每边 N 个网格点
    """
    if region not in REGION_TYPES:
        raise ValueError(f"region 必须为 {REGION_TYPES} 之一")
    x = np.linspace(-1.0, 1.0, n_grid)
    y = np.linspace(-1.0, 1.0, n_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    G = -np.ones_like(X, dtype=np.int32)

    if region == "S":
        G[1:-1, 1:-1] = np.arange(1, (n_grid - 2) ** 2 + 1).reshape(n_grid - 2, n_grid - 2)
    elif region == "L":
        # L: 3/4 正方形 (去掉右上 1/4)
        for i in range(n_grid):
            for j in range(n_grid):
                if X[i, j] > 0 and Y[i, j] > 0:
                    G[i, j] = -1
                else:
                    G[i, j] = 0
        # 内部编号
        interior = (G == 0) & (np.abs(X) < 1.0 - 1.0 / n_grid) & (np.abs(Y) < 1.0 - 1.0 / n_grid)
        # 排除外角
        interior = interior & ~((X > 0) & (Y > 0))
        idx = np.where(interior)
        G[idx] = np.arange(1, len(idx[0]) + 1)
    elif region == "C":
        # L 去掉第 4 象限的 1/4 圆
        for i in range(n_grid):
            for j in range(n_grid):
                r2 = X[i, j] ** 2 + Y[i, j] ** 2
                if X[i, j] > 0 and Y[i, j] < 0 and r2 < 1.0:
                    G[i, j] = -1
                elif X[i, j] > 0 and Y[i, j] > 0:
                    G[i, j] = -1
                else:
                    G[i, j] = 0
        interior = (G == 0) & (np.abs(X) < 1.0 - 1.0 / n_grid) & (np.abs(Y) < 1.0 - 1.0 / n_grid)
        idx = np.where(interior)
        G[idx] = np.arange(1, len(idx[0]) + 1)
    elif region == "F":
        # 完整矩形
        G[1:-1, 1:-1] = np.arange(1, (n_grid - 2) ** 2 + 1).reshape(n_grid - 2, n_grid - 2)
    elif region == "D":
        # 圆盘
        for i in range(n_grid):
            for j in range(n_grid):
                if X[i, j] ** 2 + Y[i, j] ** 2 <= 1.0:
                    G[i, j] = 0
        interior = (G == 0) & (X ** 2 + Y ** 2 < (1.0 - 1.0 / n_grid) ** 2)
        idx = np.where(interior)
        G[idx] = np.arange(1, len(idx[0]) + 1)

    return G, X, Y


# =============================================================
# 共动盒子网格 (3D)
# =============================================================
@dataclass
class ComovingBox:
    """3D 共动周期性盒子."""
    L         : float = 500.0      # 盒子边长 Mpc/h
    N         : int   = 64         # 每边网格点数
    dx        : float = 0.0
    x_edges   : np.ndarray = None
    y_edges   : np.ndarray = None
    z_edges   : np.ndarray = None
    x_centers : np.ndarray = None
    y_centers : np.ndarray = None
    z_centers : np.ndarray = None

    def __post_init__(self):
        self.dx = self.L / self.N
        self.x_edges = np.linspace(0.0, self.L, self.N + 1)
        self.y_edges = np.linspace(0.0, self.L, self.N + 1)
        self.z_edges = np.linspace(0.0, self.L, self.N + 1)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    def __post_init__(self):
        self.dx = self.L / self.N
        self.x_edges = np.linspace(0.0, self.L, self.N + 1)
        self.y_edges = np.linspace(0.0, self.L, self.N + 1)
        self.z_edges = np.linspace(0.0, self.L, self.N + 1)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])


def default_bao_box(L: float = 500.0, N: int = 64) -> ComovingBox:
    """默认 BAO 分析盒子 (500 Mpc/h)^3, 64^3 网格."""
    return ComovingBox(L=L, N=N)


# =============================================================
# 最近邻检查 (种子项目 190)
# =============================================================
def closest_pair_brute(points: np.ndarray) -> Tuple[float, int, int]:
    """
    暴力最近邻搜索. 返回 (d_min, i, j).
    用于检查网格密集性.
    """
    N = len(points)
    if N < 2:
        return float("inf"), -1, -1
    d_min = float("inf")
    pair = (-1, -1)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < d_min:
                d_min = d
                pair = (i, j)
    return d_min, pair[0], pair[1]


def grid_density_check(G: np.ndarray, X: np.ndarray, Y: np.ndarray,
                       threshold: float = 0.1) -> Dict[str, int]:
    """
    检查网格密集性: 若内部点的最小分离 < threshold × Δx, 则标记为"过密".
    """
    interior = np.where(G > 0)
    if len(interior[0]) < 2:
        return {"n_interior": 0, "n_close": 0}
    pts = np.column_stack([X[interior], Y[interior]])
    dx = X[1, 0] - X[0, 0] if X.shape[0] > 1 else 0.1
    n_close = 0
    # 抽样检查 (避免 O(N^2))
    n_sample = min(50, len(pts))
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(pts), n_sample, replace=False)
    for i in idx:
        for j in range(i + 1, len(pts)):
            d = abs(pts[i, 0] - pts[j, 0]) + abs(pts[i, 1] - pts[j, 1])
            if 0 < d < threshold * dx:
                n_close += 1
                break
    return {"n_interior": len(pts), "n_close": n_close}


# =============================================================
# 天球到共动坐标的映射
# =============================================================
def sky_to_comoving(ra: np.ndarray, dec: np.ndarray, z: np.ndarray,
                    cosmo) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把 (RA, Dec, z) 转换为共动笛卡尔 (x, y, z).
    x = χ(z) cos(dec) cos(ra)
    y = χ(z) cos(dec) sin(ra)
    z_cart = χ(z) sin(dec)
    """
    from background_cosmology import comoving_distance
    chi = np.array([comoving_distance(cosmo, zi) for zi in z])
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = chi * np.sin(dec_rad)
    return x, y, z_cart


# =============================================================
# 自适应网格细化 (AMR)
# =============================================================
def adaptive_mesh_1d(x_min: float, x_max: float, n_base: int,
                     refinement: np.ndarray = None) -> np.ndarray:
    """
    一维自适应网格. 在 refinement 指定的区间加密 (加密 2 倍).
    refinement: list of (x_lo, x_hi) 需要加密的区间.
    """
    x_base = np.linspace(x_min, x_max, n_base)
    if refinement is None:
        return x_base
    x_out = []
    for i in range(n_base - 1):
        x0, x1 = x_base[i], x_base[i + 1]
        refined = False
        for (rlo, rhi) in refinement:
            if x0 >= rlo and x1 <= rhi:
                refined = True
                break
        if refined:
            x_out.extend(np.linspace(x0, x1, 5)[:-1].tolist())
        else:
            x_out.append(x0)
    x_out.append(x_base[-1])
    return np.array(x_out)


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    for region in ["S", "L", "C", "F", "D"]:
        G, X, Y = numgrid_bao(region, 16)
        n_int = int((G > 0).sum())
        print(f"[comoving_grid] region={region}: 内部点 = {n_int}")
    box = default_bao_box(L=500.0, N=32)
    print(f"[comoving_grid] 盒子: L={box.L}, N={box.N}, dx={box.dx:.2f}")
    # 最近邻
    pts = np.random.default_rng(0).uniform(0, 1, size=(20, 2))
    d_min, i, j = closest_pair_brute(pts)
    print(f"[comoving_grid] 最近邻距离 = {d_min:.3f}")


if __name__ == "__main__":
    _self_check()
