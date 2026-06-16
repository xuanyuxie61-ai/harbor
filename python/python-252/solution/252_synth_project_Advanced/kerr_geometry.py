#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kerr_geometry.py  ——  Kerr 度规、Kerr-Schild 坐标、微分几何量

融合种子项目:
  - 330_ellipse_grid   : 椭圆坐标网格生成思想 → Kerr-Schild 曲面坐标
  - 596_interp_trig    : 三角插值 → 角向几何量的光滑插值

核心公式 (Boyer–Lindquist / Kerr-Schild):
  Sigma  = r^2 + a^2 cos^2 theta
  Delta  = r^2 - 2 M r + a^2
  A      = (r^2 + a^2)^2 - Delta a^2 sin^2 theta
  g_tt   = -(1 - 2 M r / Sigma)
  g_tphi = -2 M a r sin^2 theta / Sigma
  g_phiphi = (A sin^2 theta) / Sigma
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple
from mhd_constants import BlackHoleParameters


class KerrGeometry:
    """Kerr 黑洞几何, 全部使用几何单位制 G=c=M=1."""

    def __init__(self, bh: BlackHoleParameters) -> None:
        self.a  = bh.a_geom           # a* in geometric units (M=1)
        self.rp = bh.r_plus
        self.rm = bh.r_minus
        self.Omega_H = bh.omega_h

    # ----------------------------------------------------------
    # 基本几何函数
    # ----------------------------------------------------------
    def sigma2(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Sigma = r^2 + a^2 cos^2 theta."""
        return r * r + self.a * self.a * np.cos(theta) ** 2

    def delta(self, r: np.ndarray) -> np.ndarray:
        """Delta = r^2 - 2 r + a^2  (M=1)."""
        return r * r - 2.0 * r + self.a * self.a

    def big_a(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """A = (r^2+a^2)^2 - Delta a^2 sin^2 theta."""
        r2a2 = r * r + self.a * self.a
        return r2a2 * r2a2 - self.delta(r) * self.a * self.a * np.sin(theta) ** 2

    # ----------------------------------------------------------
    # Boyer-Lindquist 度规分量
    # ----------------------------------------------------------
    def metric_bl(self, r: np.ndarray, theta: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (g_tt, g_tphi, g_phiphi, g_rr) 的 BL 度规分量."""
        S  = self.sigma2(r, theta)
        D  = self.delta(r)
        A  = self.big_a(r, theta)
        st2 = np.sin(theta) ** 2
        g_tt   = -(1.0 - 2.0 * r / S)
        g_tphi = -2.0 * self.a * r * st2 / S
        g_pp   = A * st2 / S
        g_rr   = S / np.maximum(D, 1.0e-30)   # 视界处正则化
        return g_tt, g_tphi, g_pp, g_rr

    # ----------------------------------------------------------
    # 帧拖曳角速度  omega_fd = -g_tphi / g_phiphi
    # ----------------------------------------------------------
    def frame_dragging(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Lense–Thirring 帧拖曳角速度."""
        _, g_tp, g_pp, _ = self.metric_bl(r, theta)
        return -g_tp / np.maximum(g_pp, 1.0e-30)

    # ----------------------------------------------------------
    # 能层位置  r_ergo(theta) = M + sqrt(M^2 - a^2 cos^2 theta)
    # ----------------------------------------------------------
    def ergosphere_radius(self, theta: np.ndarray) -> np.ndarray:
        disc = np.sqrt(np.maximum(1.0 - self.a * self.a * np.cos(theta) ** 2, 0.0))
        return 1.0 + disc

    # ----------------------------------------------------------
    # Kerr-Schild 笛卡尔坐标 (用于网格生成, 融合 330 椭圆网格思想)
    # ----------------------------------------------------------
    def kerr_schild_cartesian(self, r: np.ndarray, theta: np.ndarray,
                               phi: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Kerr-Schild 坐标到笛卡尔 (X, Y, Z).
        X = sqrt(r^2+a^2) sin(theta) cos(phi)
        Y = sqrt(r^2+a^2) sin(theta) sin(phi)
        Z = r cos(theta)
        (类比 330_ellipse_grid 的椭圆坐标生成.)
        """
        R = np.sqrt(r * r + self.a * self.a)
        X = R * np.sin(theta) * np.cos(phi)
        Y = R * np.sin(theta) * np.sin(phi)
        Z = r * np.cos(theta)
        return X, Y, Z

    # ----------------------------------------------------------
    # 三角插值 (融合 596_interp_trig)
    # ----------------------------------------------------------
    @staticmethod
    def trig_interpolate_1d(values: np.ndarray, theta_out: np.ndarray,
                             theta_in: np.ndarray) -> np.ndarray:
        """
        周期三角插值 (用于角向几何量的高精度插值).
        values: 在 theta_in 上的采样 (等距, 不含端点).
        theta_out: 目标角度.
        参考 596_interp_trig.m.
        """
        N = len(theta_in)
        dtheta = theta_in[1] - theta_in[0] if N > 1 else 1.0
        result = np.zeros_like(theta_out, dtype=float)
        for i, th in enumerate(theta_out):
            s = 0.0
            for k in range(N):
                arg = (th - theta_in[k]) / dtheta
                # Dirichlet 核
                if N % 2 == 0:
                    kernel = (np.sin(np.pi * arg)
                              / (N * np.tan(np.pi * arg / N) + 1.0e-30))
                else:
                    kernel = (np.sin(np.pi * arg)
                              / (N * np.tan(np.pi * arg / N) + 1.0e-30))
                s += values[k] * kernel
            result[i] = s / max(N, 1)
        return result

    # ----------------------------------------------------------
    # 表面重力  kappa = (r_+ - r_-) / (2 (r_+^2 + a^2))
    # ----------------------------------------------------------
    def surface_gravity(self) -> float:
        return (self.rp - self.rm) / (2.0 * (self.rp * self.rp + self.a * self.a))

    # ----------------------------------------------------------
    # 黑洞熵 (Bekenstein-Hawking, 几何单位)
    # ----------------------------------------------------------
    def bh_entropy(self) -> float:
        """S_BH = A / 4,  A = 4 pi (r_+^2 + a^2)."""
        area = 4.0 * math.pi * (self.rp * self.rp + self.a * self.a)
        return area / 4.0

    # ----------------------------------------------------------
    # 最内稳定圆轨道  r_isco (近似公式, Bardeen et al. 1972)
    # ----------------------------------------------------------
    def r_isco(self) -> float:
        """对 prograde 轨道的 r_isco 近似 (a>=0)."""
        a = self.a
        Z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
            (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
        Z2 = math.sqrt(3.0 * a * a + Z1 * Z1)
        return 3.0 + Z2 - math.sqrt(max((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2), 0.0))
