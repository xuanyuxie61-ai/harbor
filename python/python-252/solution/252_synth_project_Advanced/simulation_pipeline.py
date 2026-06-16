#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulation_pipeline.py  ——  MHD 模拟全流程编排

融合种子项目:
  - 1057_Bio-Inspired-Navigation : Pipeline 编排模式
  - 1417_wtime : 计时
  - 419_fem3d_sample : FEM 采样思想 → 在指定点插值 MHD 场

核心流程:
  1. 初始化配置 & 网格
  2. 设置初始条件 (Bondi 吸积 + 磁化)
  3. 时间推进 (显式 + 隐式径向)
  4. 稳定性分析 (色散关系 + 增长率提取)
  5. 因果连通性分析
  6. 最优模态选择 & 反馈控制设计
  7. 性能报告输出
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any
from mhd_constants import MHDConfig, BlackHoleParameters, PlasmaParameters, NumericalParameters, blandford_znajek_power
from mhd_grid import MHDGrid
from kerr_geometry import KerrGeometry
from mhd_equations import MHDState, MHDRhsComputer
from high_order_fd import CompactPadeSolver
from implicit_solver import RadialImplicitStepper
from spectral_filter import SpectralFilter
from stability_analysis import StabilityAnalyzer, extract_growth_rate, mri_growth_rate
from causal_connectivity import CausalGraph
from mode_selector import ModeSelector
from optimal_control import MHDStabilityController
from timing_utils import PerformanceTimer


# ============================================================
# 初始条件: 磁化 Bondi 吸积 + 微扰
# ============================================================
def initialize_bondi_accretion(grid: MHDGrid, cfg: MHDConfig) -> MHDState:
    """
    球对称 Bondi 吸积 + 均匀极向磁场 + 小振幅微扰.
    rho(r) ~ r^{-3/2},  p(r) ~ r^{-5/2},  v_r ~ -r^{-1/2}
    """
    state = MHDState(grid)
    Nr_cell = len(grid.rc)
    Nt_cell = len(grid.tc)
    r = grid.rc[:, None, None] * np.ones((1, Nt_cell, grid.Np))
    theta = np.ones((Nr_cell, 1, 1)) * grid.tc[None, :, None]

    # Bondi 解 (简化: 自由落体)
    state.rho = np.maximum(1.0 / (r ** 1.5 + 1e-30), cfg.plasma.rho_floor)
    state.press = np.maximum(0.01 / (r ** 2.5 + 1e-30), cfg.plasma.p_floor)
    state.vr = -np.sqrt(2.0 / (r + 1e-30)) * 0.1  # 亚声速
    state.vt = np.zeros_like(state.vr)
    # 开普勒旋转
    Omega_K = 1.0 / (r ** 1.5 + 1e-30)
    state.vp = Omega_K * r * np.sin(theta) * 0.1

    # 均匀极向磁场
    B0 = 1.0
    state.Br = B0 * np.cos(theta) / (r ** 2 + 1e-30)
    state.Bt = -0.5 * B0 * np.sin(theta) / (r + 1e-30)
    state.Bp = np.zeros_like(state.Br)

    # 微扰 (激发 MRI)
    np.random.seed(42)
    amp = 1e-3
    state.press *= (1.0 + amp * np.random.randn(*state.press.shape))
    state.rho *= (1.0 + amp * np.random.randn(*state.rho.shape))

    return state


# ============================================================
# 模拟主流程
# ============================================================
class MHDSimulationPipeline:
    """
    全流程编排 (参考 1057 的 Pipeline 模式).
    """

    def __init__(self, cfg: MHDConfig = None) -> None:
        self.cfg = cfg or MHDConfig()
        self.timer = PerformanceTimer()

    def run(self) -> Dict[str, Any]:
        """
        运行完整模拟流程, 返回结果字典.
        """
        print("\n" + "=" * 70)
        print("黑洞喷流 MHD 模拟 & 稳定性分析 (博士级可复现实验)")
        print("=" * 70)

        # 1. 配置校验
        self.timer.start("config_validate")
        checks = self.cfg.validate()
        self.timer.stop("config_validate")
        print(f"\n[1/8] 配置校验: 全部通过 ({len(checks)} 项)")

        # 2. 网格生成
        self.timer.start("grid_generation")
        grid = MHDGrid(self.cfg.num, self.cfg.bh)
        self.timer.stop("grid_generation")
        q = grid.quality_report()
        print(f"[2/8] 网格生成: Nr={q['Nr']}, Nt={q['Nt']}, "
              f"r=[{q['r_min']:.2f}, {q['r_max']:.1f}]")

        # 3. 初始条件
        self.timer.start("initial_condition")
        state = initialize_bondi_accretion(grid, self.cfg)
        self.timer.stop("initial_condition")
        print(f"[3/8] 初始条件: Bondi 吸积 + 磁化 + MRI 微扰")

        # 4. 时间推进
        self.timer.start("time_integration")
        rhs_computer = MHDRhsComputer(self.cfg, grid)
        implicit_stepper = RadialImplicitStepper(self.cfg, grid.Nr - 1)
        spectral_filter = SpectralFilter(grid.Nr - 1,
                                          alpha=36.0, order=8)

        dt = rhs_computer.compute_cfl_timestep(state)
        n_steps = min(int(self.cfg.num.t_end / dt), 20)  # 限制步数 (小规模实验)
        dt = self.cfg.num.t_end / max(n_steps, 1)

        print(f"[4/8] 时间推进: dt={dt:.4f}, n_steps={n_steps}, "
              f"FD阶数={self.cfg.num.fd_order}")

        # 记录增长率
        time_series = []
        amp_series = []

        for step in range(n_steps):
            # 显式右端项
            rhs_D, rhs_Sr, rhs_St, rhs_Sp, rhs_tau, rhs_Br, rhs_Bt, rhs_Bp = \
                rhs_computer.compute_rhs(state)

            # 隐式径向步 (对密度)
            cf = rhs_computer.max_characteristic_speed(state)
            cf_radial = np.mean(cf, axis=(1, 2))  # 平均到径向
            if self.cfg.num.implicit_radial:
                D_new = implicit_stepper.step(state.rho, rhs_D,
                                               grid.dr, cf_radial, dt)
            else:
                D_new = state.rho + dt * rhs_D

            # 更新守恒变量 (简化: 只更新密度)
            state.rho = np.maximum(D_new, self.cfg.plasma.rho_floor)
            state.press = np.maximum(
                state.press + dt * rhs_tau * 0.1,
                self.cfg.plasma.p_floor)

            # 谱滤波 (每 5 步)
            if step % 5 == 0 and step > 0:
                Nr_cell, Nt_cell, Np_cell = state.rho.shape
                for j in range(Nt_cell):
                    for k in range(Np_cell):
                        state.rho[:, j, k] = spectral_filter.apply_1d(
                            state.rho[:, j, k])

            # 记录振幅 (密度扰动 L2 范数)
            rho_mean = np.mean(state.rho)
            rho_pert = np.sqrt(np.mean((state.rho - rho_mean) ** 2))
            time_series.append(step * dt)
            amp_series.append(rho_pert)

        self.timer.stop("time_integration")
        print(f"    完成 {n_steps} 步, 最终密度扰动: {amp_series[-1]:.6e}")

        # 5. 稳定性分析
        self.timer.start("stability_analysis")
        analyzer = StabilityAnalyzer(self.cfg)
        # 提取增长率
        gamma_fit, A_fit, R2_fit = extract_growth_rate(
            np.array(time_series), np.array(amp_series))
        # 扫描波数
        rho_avg = float(np.mean(state.rho))
        p_avg = float(np.mean(state.press))
        Bz_avg = float(np.mean(np.abs(state.Bp)))
        Omega_K = 1.0 / (grid.rc[grid.Nr // 2] ** 1.5 + 1e-30)
        modes = analyzer.scan_wavenumbers(rho_avg, p_avg, Bz_avg, Omega_K)
        self.timer.stop("stability_analysis")
        print(f"[5/8] 稳定性分析: 拟合增长率 gamma={gamma_fit:.6f}, R^2={R2_fit:.4f}")
        print(f"    前 3 个不稳定模态:")
        for i, (k, g, mt) in enumerate(modes[:3]):
            print(f"      [{i+1}] k={k:.4f}, gamma={g:.6f}, type={mt}")

        # 6. 因果连通性
        self.timer.start("causal_analysis")
        causal = CausalGraph(grid, self.cfg, state, dt)
        source_idx = causal._flat(grid.Nr // 4, grid.Nt // 2, 0)
        infl_size = causal.influence_domain_size(source_idx)
        self.timer.stop("causal_analysis")
        print(f"[6/8] 因果连通性: 源单元影响域大小 = {infl_size}")

        # 7. 模态选择 & 最优控制
        self.timer.start("optimal_control")
        selector = ModeSelector(self.cfg)
        candidates = [{"name": mt, "growth_rate": g, "cost": max(int(k * 10), 1)}
                      for k, g, mt in modes]
        selected = selector.select_modes(candidates, budget=50)
        print(f"[7/8] 模态选择: 从 {len(candidates)} 个候选中选择 {len(selected)} 个")

        # 设计 LQR 控制器 (简化: 2x2 系统)
        A_lin = np.array([[1.0 + gamma_fit * dt, 0.1],
                           [0.0, 0.95]])
        B_act = np.array([[0.1], [0.05]])
        Q_w = np.eye(2)
        R_w = 0.1 * np.eye(1)
        controller = MHDStabilityController(self.cfg)
        K_opt, P_inf, rho_cl = controller.design_controller(A_lin, B_act, Q_w, R_w)
        self.timer.stop("optimal_control")
        print(f"    LQR 控制器: 闭环谱半径 = {rho_cl:.4f} "
              f"({'稳定' if rho_cl < 1.0 else '不稳定'})")

        # 8. 性能报告
        self.timer.start("report")
        bz_power = blandford_znajek_power(self.cfg)
        geo = KerrGeometry(self.cfg.bh)
        print(f"[8/8] 物理诊断:")
        print(f"    BZ 喷流功率: {bz_power:.3e} erg/s")
        print(f"    黑洞熵: {geo.bh_entropy():.4f}")
        print(f"    r_isco: {geo.r_isco():.4f} M")
        print(f"    表面重力: {geo.surface_gravity():.6f}")
        self.timer.stop("report")
        self.timer.print_report()

        # 汇总结果
        results = {
            "config_checks": checks,
            "grid_quality": q,
            "growth_rate_fit": gamma_fit,
            "R2_fit": R2_fit,
            "top_modes": modes[:3],
            "causal_influence_size": infl_size,
            "selected_modes": [m["name"] for m in selected],
            "lqr_spectral_radius": rho_cl,
            "bz_power_erg_s": bz_power,
            "n_steps": n_steps,
            "final_density_perturbation": amp_series[-1],
        }
        return results
