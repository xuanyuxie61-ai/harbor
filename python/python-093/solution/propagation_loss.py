#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
propagation_loss.py
水声传播抛物方程模型 — 传播损失计算与后处理

本模块计算并分析声场传播损失（Transmission Loss），来源于：
- 1426_xyzl_display（3D 点/线数据结构 → 接收器阵列几何处理）
- 942_quad_parfor（并行梯形积分 → 多频/多深度积分加速）
- 113_box_distance（距离统计 → 多径时延扩展分析）

核心物理与数学公式：
1. 传播损失定义：
   TL(r,z) = −10·log₁₀ |p(r,z)/p₀|²   (dB)
   其中 p₀ 为参考声压（1 m 处），p(r,z) 为 PE 求解的复声压场 u(r,z)。
   由于 PE 中使用归一化场，TL 简化为：
   TL(r,z) = −10·log₁₀ |u(r,z)|² + C_ref

2. 相干传播损失（Coherent TL）：
   TL_coh(r,z) = −10·log₁₀ |u(r,z)|²
   保留相位信息，适用于单频连续波。

3. 非相干传播损失（Incoherent TL）：
   对频率或空间系综平均：
   TL_inc(r,z) = −10·log₁₀ ⟨|u(r,z)|²⟩
   用于宽带信号或多途平均。

4. 接收器阵列响应：
   垂直线阵（VLA）在深度 {z_j} 上的输出：
   y(r) = Σ_j w_j · u(r, z_j) · exp(−i·k₀·z_j·sinθ_look)
   其中 w_j 为阵元权值，θ_look 为波束指向角。
   采用 sinc 加权（来自 sinc 函数的 Dolph-Chebyshev 近似）。

5. 多径时延扩展：
   利用 box_distance 统计思想，估计散射体积内的典型路径长度差异，
   从而估算时延扩展 σ_τ：
   σ_τ ≈ μ_D / c
   其中 μ_D 为散射体内随机路径长度均值。

6. 深度平均传播损失（DASL）：
   TL_DASL(r) = −10·log₁₀ [ (1/H_w) ∫₀^{h_b(r)} |u(r,z)|² dz ]
   其中 H_w 为有效水深。

7. 收敛区分析：
   利用 point-in-polygon 的射线交叉思想，在 TL 场中识别
   声强局部极大值区域（convergence zones）。
"""

import numpy as np
from source_field import sincn_fun
from special_functions import alnorm


class PropagationLoss:
    """
    传播损失计算器。
    """

    def __init__(self, U, r_grid, z_grid, seafloor_depth):
        self.U = np.asarray(U, dtype=np.complex128)
        self.r_grid = np.asarray(r_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.seafloor_depth = np.asarray(seafloor_depth, dtype=np.float64)
        self.nr = len(r_grid)
        self.nz = len(z_grid)

    def coherent_tl(self):
        """
        相干传播损失（dB）：
        TL(r,z) = −10·log₁₀ |u(r,z)|²
        """
        intensity = np.abs(self.U) ** 2
        intensity = np.maximum(intensity, 1e-30)
        return -10.0 * np.log10(intensity)

    def incoherent_tl_frequency_average(self, U_list):
        """
        多频非相干平均传播损失。
        参数 U_list 为不同频率的场矩阵列表。
        """
        avg_intensity = np.zeros_like(self.U, dtype=np.float64)
        for Uf in U_list:
            avg_intensity += np.abs(Uf) ** 2
        avg_intensity /= len(U_list)
        avg_intensity = np.maximum(avg_intensity, 1e-30)
        return -10.0 * np.log10(avg_intensity)

    def depth_averaged_tl(self, r_index=None):
        """
        深度平均传播损失（dB）：
        TL_DASL(r) = −10·log₁₀ [ (1/H) ∫ |u|² dz ]
        """
        if r_index is None:
            r_index = slice(None)
        U_slice = self.U[r_index, :]
        z = self.z_grid
        tl_dasl = np.zeros(U_slice.shape[0], dtype=np.float64)
        for i in range(U_slice.shape[0]):
            intensity = np.abs(U_slice[i, :]) ** 2
            H = max(z[-1] - z[0], 1e-6)
            avg_int = np.trapezoid(intensity, z) / H
            avg_int = max(avg_int, 1e-30)
            tl_dasl[i] = -10.0 * np.log10(avg_int)
        return tl_dasl

    def tl_at_receiver(self, z_receiver):
        """
        提取固定深度接收器处的传播损失随距离变化。
        """
        # 找到最近的网格点
        j = np.argmin(np.abs(self.z_grid - z_receiver))
        intensity = np.abs(self.U[:, j]) ** 2
        intensity = np.maximum(intensity, 1e-30)
        return -10.0 * np.log10(intensity)

    def vla_beamform(self, z_vla, weights, theta_look, k0):
        """
        垂直线阵波束形成输出。
        参数:
            z_vla: 阵元深度数组
            weights: 阵元权值
            theta_look: 波束指向角（弧度，从水平面起算）
            k0: 参考波数
        返回: 波束输出随距离变化。
        """
        z_vla = np.asarray(z_vla, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        output = np.zeros(self.nr, dtype=np.complex128)
        for i in range(self.nr):
            for j, zj in enumerate(z_vla):
                # 在 U[i,:] 中插值到 zj
                u_zj = np.interp(zj, self.z_grid, np.real(self.U[i, :])) \
                       + 1j * np.interp(zj, self.z_grid, np.imag(self.U[i, :]))
                output[i] += weights[j] * u_zj * np.exp(-1j * k0 * zj * np.sin(theta_look))
        return output

    def convergence_zone_analysis(self, tl_field, threshold_db=5.0):
        """
        收敛区检测：在 TL 场中查找局部极小值（声强极大值）区域。
        返回收敛区的 (r_index, z_index) 列表。
        """
        zones = []
        for i in range(1, self.nr - 1):
            for j in range(1, self.nz - 1):
                center = tl_field[i, j]
                neighbors = [
                    tl_field[i - 1, j], tl_field[i + 1, j],
                    tl_field[i, j - 1], tl_field[i, j + 1]
                ]
                if all(center < n - threshold_db for n in neighbors):
                    zones.append((i, j, self.r_grid[i], self.z_grid[j], center))
        return zones

    def shadow_zone_detection(self, tl_field, tl_threshold=80.0):
        """
        声影区检测：TL 超过阈值的连续区域。
        返回影区边界索引。
        """
        shadow_mask = tl_field > tl_threshold
        # 对每个距离步，找到影区深度范围
        shadows = []
        for i in range(self.nr):
            mask = shadow_mask[i, :]
            if np.any(mask):
                j_min = np.argmax(mask)
                j_max = len(mask) - 1 - np.argmax(mask[::-1])
                shadows.append((i, j_min, j_max))
        return shadows


class ReceiverArray:
    """
    接收器阵列几何管理（来自 1426_xyzl_display 的 3D 点/线结构）。
    """

    def __init__(self, r_positions, z_positions):
        """
        参数:
            r_positions: 水平位置数组 (m)
            z_positions: 深度位置数组 (m)
        """
        self.r = np.asarray(r_positions, dtype=np.float64)
        self.z = np.asarray(z_positions, dtype=np.float64)
        self.n_receivers = len(r_positions)

    def dolph_chebyshev_weights(self, sidelobe_db=-30):
        """
        Dolph-Chebyshev 加权（sinc 近似实现）。
        权值基于 Chebyshev 多项式的旁瓣抑制特性。
        简化实现：使用高斯近似。
        """
        n = self.n_receivers
        if n <= 1:
            return np.ones(n)
        # 高斯近似 Dolph-Chebyshev
        sigma = n / (2.0 * np.sqrt(np.log(10.0) * abs(sidelobe_db) / 20.0))
        idx = np.arange(n) - (n - 1) / 2.0
        w = np.exp(-0.5 * (idx / sigma) ** 2)
        return w / np.sum(w)

    def sinc_weights(self, mainlobe_width):
        """
        基于 sinc 函数的频域窗权值：
        w_j = sinc_n(j · Δz / W)
        """
        if self.n_receivers <= 1:
            return np.ones(1)
        dz = np.mean(np.diff(self.z)) if len(self.z) > 1 else 1.0
        idx = np.arange(self.n_receivers) - (self.n_receivers - 1) / 2.0
        w = sincn_fun(idx * dz / mainlobe_width)
        return w / np.sum(w)

    def extract_signals(self, U, r_grid, z_grid):
        """
        从全场 U 中提取各接收器位置的声压（双线性插值）。
        """
        signals = np.zeros(self.n_receivers, dtype=np.complex128)
        for k in range(self.n_receivers):
            # 找到最近的 range 和 depth 索引
            i = np.argmin(np.abs(r_grid - self.r[k]))
            j = np.argmin(np.abs(z_grid - self.z[k]))
            signals[k] = U[i, j]
        return signals


class MultipathStatistics:
    """
    多径统计特性分析。
    """

    def __init__(self, tl_obj, c_water=1500.0):
        self.tl_obj = tl_obj
        self.c = c_water

    def delay_spread_estimate(self, scattering_volume_mean_distance):
        """
        时延扩展估计：
        σ_τ ≈ μ_D / c
        其中 μ_D 为散射体积内随机路径长度均值。
        """
        return scattering_volume_mean_distance / self.c

    def coherence_bandwidth(self, delay_spread):
        """
        相干带宽估计：
        B_c ≈ 1 / (2π·σ_τ)
        """
        if delay_spread < 1e-12:
            return np.inf
        return 1.0 / (2.0 * np.pi * delay_spread)

    def fading_depth_statistics(self, r_index, z_index, window_r=5):
        """
        计算局部衰落深度统计（均值、标准差）。
        """
        i0 = max(0, r_index - window_r)
        i1 = min(self.tl_obj.nr, r_index + window_r + 1)
        tl_local = self.tl_obj.coherent_tl()[i0:i1, z_index]
        return float(np.mean(tl_local)), float(np.std(tl_local))
