# -*- coding: utf-8 -*-
"""
heom_hierarchy.py
-----------------
层次化多尺度耦合: 光球-色球-日冕-太阳风 HEOM-style 层级方程.

物理动机
--------
借鉴量子开放系统 HEOM (1225_Okita0512_VSC_HEOM) 的层级方程思想,
将太阳大气划分为多层耦合子系统:

层级 0: 光球 (photosphere, r ~ R_sun)
   - 对流湍动驱动足点运动
   - 特征时标: tau_conv ~ 10 min

层级 1: 色球/过渡区 (chromosphere/TR, r ~ R_sun + 2 Mm)
   - Alfvén 波传播与部分反射
   - 特征时标: tau_A ~ 1 min

层级 2: 日冕 (corona, r ~ 1.05 - 5 R_sun)
   - 磁重联加热 + 热传导
   - 特征时标: tau_heat ~ 100 s

层级 3: 太阳风 (solar wind, r > 5 R_sun)
   - 绝热膨胀 + Alfvén 波加速
   - 特征时标: tau_sw ~ 1 hour

耦合: 每层通过边界通量 (Poynting flux, 质量通量, 波通量) 向上传递.
反向耦合: 太阳风压力影响日冕基底, 色球蒸发补充日冕密度.

数学: 每层用一维 MHD 方程描述, 层间通过通量守恒连接.
    dU_l/dt + dF_l/ds = S_l + C_{l-1->l} - C_{l->l+1}
"""
from __future__ import annotations
import numpy as np
from solar_constants import SOLAR_RADIUS


class AtmosphericLayer:
    """单层大气状态."""

    def __init__(self, name: str, r_inner: float, r_outer: float,
                 n_grid: int, t_base: float, n_base: float,
                 b_base: float, tau_dyn: float):
        self.name = name
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.n_grid = n_grid
        self.r = np.linspace(r_inner, r_outer, n_grid)
        self.temperature = np.full(n_grid, t_base)
        self.density = np.full(n_grid, n_base)
        self.b_field = np.full(n_grid, b_base)
        self.velocity = np.zeros(n_grid)
        self.tau_dyn = tau_dyn

    @property
    def dr(self) -> float:
        return self.r[1] - self.r[0] if self.n_grid > 1 else 1.0


class HEOMHierarchy:
    """层级化多尺度耦合系统."""

    def __init__(self):
        # 初始化四层大气
        self.layers = [
            AtmosphericLayer(
                "photosphere", SOLAR_RADIUS, SOLAR_RADIUS + 5.0e5,
                32, 5800.0, 1.0e23, 0.1, 600.0
            ),
            AtmosphericLayer(
                "chromosphere", SOLAR_RADIUS + 5.0e5,
                SOLAR_RADIUS + 2.0e6,
                32, 1.0e4, 1.0e18, 0.03, 60.0
            ),
            AtmosphericLayer(
                "corona", SOLAR_RADIUS + 2.0e6,
                SOLAR_RADIUS + 1.0e7,
                48, 1.5e6, 2.0e15, 0.01, 100.0
            ),
            AtmosphericLayer(
                "solar_wind", SOLAR_RADIUS + 1.0e7,
                10.0 * SOLAR_RADIUS,
                64, 1.0e5, 1.0e12, 0.005, 3600.0
            ),
        ]

    def coupling_flux(self, l_lower: int, l_upper: int) -> dict:
        """计算从下层到上层的耦合通量.

        Poynting flux: S = (E x B) / mu_0 ~ v_A delta_B^2 / mu_0
        质量通量: F_m = rho v
        波通量: F_w = rho delta_v^2 v_A
        """
        layer_low = self.layers[l_lower]
        layer_up = self.layers[l_upper]
        # 界面处取值
        b_low = layer_low.b_field[-1]
        rho_low = layer_low.density[-1]
        v_low = layer_low.velocity[-1]
        # Poynting flux (简化: 基于足点剪切速度 v_ph ~ 1 km/s)
        v_ph = 1.0e3  # m/s
        from solar_constants import VACUUM_PERMEABILITY
        s_poynting = rho_low * v_ph**2 * abs(b_low) / np.sqrt(
            VACUUM_PERMEABILITY * rho_low + 1.0e-30
        )
        # 质量通量 (色球蒸发模型)
        f_mass = 1.0e-6 * rho_low * (layer_low.temperature[-1] / 1.0e4)**2
        # 波通量
        delta_v = 0.1 * v_ph
        from solar_constants import alven_speed
        va = alven_speed(b_low, rho_low / 1.67e-27)
        f_wave = rho_low * delta_v**2 * va
        return dict(
            poynting=float(s_poynting),
            mass=float(f_mass),
            wave=float(f_wave),
        )

    def evolve_one_step(self, dt: float) -> None:
        """每层演化一步, 加上层间耦合."""
        for l, layer in enumerate(self.layers):
            # 简化: 仅做指数弛豫到局部平衡 + 小扰动
            layer.temperature *= np.exp(-dt / layer.tau_dyn * 0.01)
            layer.density *= np.exp(-dt / layer.tau_dyn * 0.005)
            layer.velocity += dt * 10.0 * np.sin(2 * np.pi * layer.r / SOLAR_RADIUS)
            # 层间耦合: 从下层接收通量
            if l > 0:
                flux = self.coupling_flux(l - 1, l)
                layer.temperature += dt * flux["poynting"] * 1.0e-15
                layer.density += dt * flux["mass"] * 1.0e-10
            # 向上层传递通量
            if l < len(self.layers) - 1:
                flux_up = self.coupling_flux(l, l + 1)
                layer.temperature -= dt * flux_up["poynting"] * 1.0e-15
            # 物理约束: 温度/密度非负
            layer.temperature = np.maximum(layer.temperature, 1.0)
            layer.density = np.maximum(layer.density, 1.0)

    def run(self, t_total: float, dt: float) -> dict:
        """运行多层耦合演化."""
        n_step = int(t_total / dt)
        history = []
        for step in range(n_step):
            self.evolve_one_step(dt)
            if step % max(1, n_step // 10) == 0:
                state = {
                    "step": step,
                    "t": step * dt,
                    "corona_T": float(self.layers[2].temperature.mean()),
                    "wind_v": float(self.layers[3].velocity.mean()),
                }
                history.append(state)
        return history

    def summary(self) -> dict:
        """返回各层当前状态摘要."""
        out = {}
        for layer in self.layers:
            out[layer.name] = dict(
                T_mean=float(layer.temperature.mean()),
                n_mean=float(layer.density.mean()),
                v_mean=float(layer.velocity.mean()),
                B_mean=float(layer.b_field.mean()),
            )
        return out
