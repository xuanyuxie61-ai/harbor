#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_grid.py  ——  Kerr-Schild 曲线网格生成 & 单元邻接关系

融合种子项目:
  - 330_ellipse_grid    : 椭圆网格计数与生成 → Kerr-Schild 网格
  - 755_mesh_etoe       : 单元-单元邻接关系 (etoe) → 多块结构网格邻接
  - 1328_triangulate    : 耳切法多边形三角化 → 极向截面非结构三角化

核心: 在 (r, theta, phi) 上生成对数-拉伸径向 + 余弦拉伸极向网格,
     并构造结构网格的 etoe 邻接信息.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List
from mhd_constants import NumericalParameters, BlackHoleParameters
from kerr_geometry import KerrGeometry


# ============================================================
# 径向网格 (对数-拉伸, 融合 330 椭圆网格的半轴思想)
# ============================================================
def build_radial_grid(num: NumericalParameters,
                       bh: BlackHoleParameters) -> np.ndarray:
    """
    对数-拉伸径向网格: 在视界附近加密.
    r_i = r_in * (r_out/r_in)^{i/(N-1)},   i = 0,...,N-1.
    类比 330_ellipse_grid_count 中对短轴 N+1 个点的控制.
    """
    i = np.arange(num.nr, dtype=float)
    xi = i / max(num.nr - 1, 1)
    # 对数拉伸
    r = num.r_in * np.exp(xi * math.log(num.r_out / num.r_in))
    # 确保内边界 > 视界
    r = np.maximum(r, bh.r_plus + 1.0e-6)
    return r


def build_theta_grid(num: NumericalParameters) -> np.ndarray:
    """
    余弦拉伸极向网格 (在赤道和极轴加密).
    theta_j = arccos(1 - 2 j / (N-1)),  j = 0,...,N-1.
    """
    j = np.arange(num.ntheta, dtype=float)
    mu = 1.0 - 2.0 * j / max(num.ntheta - 1, 1)
    mu = np.clip(mu, -1.0 + 1.0e-12, 1.0 - 1.0e-12)
    return np.arccos(mu)


def build_phi_grid(num: NumericalParameters) -> np.ndarray:
    """方位角网格 (轴对称时只有一点)."""
    return np.linspace(0.0, 2.0 * math.pi, max(num.nphi, 1), endpoint=False)


# ============================================================
# KerrGeometry 驱动的全网格
# ============================================================
class MHDGrid:
    """三维结构网格 + 度量 + 邻接."""

    def __init__(self, num: NumericalParameters,
                 bh: BlackHoleParameters) -> None:
        self.num = num
        self.bh  = bh
        self.geo = KerrGeometry(bh)

        self.r     = build_radial_grid(num, bh)   # (Nr,)
        self.theta = build_theta_grid(num)         # (Ntheta,)
        self.phi   = build_phi_grid(num)           # (Nphi,)
        self.Nr, self.Nt, self.Np = num.nr, num.ntheta, num.nphi

        # 网格间距
        self.dr     = np.diff(self.r)              # (Nr-1,)
        self.dtheta = np.diff(self.theta)          # (Nt-1,)
        if self.Np > 1:
            self.dphi = np.full(self.Np, 2.0 * math.pi / self.Np)
        else:
            self.dphi = np.array([2.0 * math.pi])

        # 单元中心 (半格点)
        self.rc = 0.5 * (self.r[:-1] + self.r[1:])
        self.tc = 0.5 * (self.theta[:-1] + self.theta[1:])
        self.pc = self.phi  # 周期

        # 度量 (在单元中心)
        self._build_metric()
        # 邻接 (融合 755_mesh_etoe)
        self.etoe = self._build_etoe()

    # ----------------------------------------------------------
    def _build_metric(self) -> None:
        """预计算单元中心处的度量分量 sqrt(-g)."""
        R, T = np.meshgrid(self.rc, self.tc, indexing='ij')
        S = self.geo.sigma2(R, T)
        D = self.geo.delta(R)
        A = self.geo.big_a(R, T)
        st2 = np.sin(T) ** 2
        # BL 度规行列式  sqrt(-g) = Sigma sin(theta)
        self.sqrt_g = S * np.sin(T)
        # 逆度规分量 (用于特征速度)
        self.grr_inv = np.maximum(D, 1e-30) / S
        self.gtt_inv = -A / (S * np.maximum(D, 1e-30))
        # 帧拖曳
        self.omega_fd = self.geo.frame_dragging(R, T)

    # ----------------------------------------------------------
    # 单元-单元邻接 (融合 755_mesh_etoe.m 的核心算法)
    # ----------------------------------------------------------
    def _build_etoe(self) -> np.ndarray:
        """
        三维结构网格的 etoe: 对每个单元 (i,j,k),
        找到 6 个方向上的邻居 (-r,+r,-t,+t,-p,+p).
        返回形状 (Nr-1, Nt-1, Np, 6) 的整数数组, -1 表示边界.
        参考 755_mesh_etoe 的排序-匹配策略.
        """
        Nr, Nt, Np = self.Nr - 1, self.Nt - 1, self.Np
        etoe = -np.ones((Nr, Nt, Np, 6), dtype=np.int64)

        def flat(i, j, k):
            return i * Nt * Np + j * Np + k

        for i in range(Nr):
            for j in range(Nt):
                for k in range(Np):
                    idx = flat(i, j, k)
                    # -r 方向
                    if i > 0:
                        etoe[i, j, k, 0] = flat(i - 1, j, k)
                    # +r 方向
                    if i < Nr - 1:
                        etoe[i, j, k, 1] = flat(i + 1, j, k)
                    # -theta 方向
                    if j > 0:
                        etoe[i, j, k, 2] = flat(i, j - 1, k)
                    # +theta 方向
                    if j < Nt - 1:
                        etoe[i, j, k, 3] = flat(i, j + 1, k)
                    # -phi 方向 (周期)
                    if Np > 1:
                        etoe[i, j, k, 4] = flat(i, j, (k - 1) % Np)
                        etoe[i, j, k, 5] = flat(i, j, (k + 1) % Np)
        return etoe

    # ----------------------------------------------------------
    # 极向截面的耳切三角化 (融合 1328_triangulate)
    # ----------------------------------------------------------
    def triangulate_poloidal_crosssection(self) -> List[Tuple[int, int, int]]:
        """
        在 (r, theta) 平面的边界多边形上做耳切三角化.
        多边形顶点: 沿内边界向上, 外边界向下, 形成闭合环.
        返回三角形列表 [(i,j,k), ...].
        """
        # 构造简单凸多边形 (r,theta) 截面的顶点
        verts = []
        # 内边界: theta 从 0 到 pi
        for j in range(self.Nt):
            verts.append((0, j))
        # 外边界: theta 从 pi 到 0
        for j in range(self.Nt - 1, -1, -1):
            verts.append((self.Nr - 1, j))
        # 耳切法 (简单凸多边形直接扇形剖分)
        tris = []
        n = len(verts)
        for s in range(1, n - 1):
            tris.append((0, s, s + 1))
        return tris

    # ----------------------------------------------------------
    # 网格质量检查
    # ----------------------------------------------------------
    def quality_report(self) -> dict:
        """返回网格质量度量."""
        # 径向往复比
        rr = self.dr[1:] / np.maximum(self.dr[:-1], 1e-30)
        # 极向往复比
        tt = self.dtheta[1:] / np.maximum(self.dtheta[:-1], 1e-30)
        return {
            "Nr": self.Nr, "Nt": self.Nt, "Np": self.Np,
            "r_min": float(self.r[0]),
            "r_max": float(self.r[-1]),
            "r_ratio_max": float(np.max(rr)) if len(rr) else 1.0,
            "theta_ratio_max": float(np.max(tt)) if len(tt) else 1.0,
            "sqrt_g_min": float(np.min(self.sqrt_g)),
            "sqrt_g_max": float(np.max(self.sqrt_g)),
        }
