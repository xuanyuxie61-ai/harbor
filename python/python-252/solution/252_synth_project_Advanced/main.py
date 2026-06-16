#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py  ——  统一入口: 黑洞喷流 MHD 模拟 & 稳定性分析

本项目融合 15 个种子项目的核心算法, 围绕计算天体物理前沿问题:
  Kerr 黑洞磁层中 Blandford–Znajek 喷流的高阶有限差分 MHD 模拟
  及其线性稳定性分析 (博士级可复现实验).

零参数运行:  python main.py

种子项目映射:
  [1]  1374_unstable_ode      → MHD 半离散右端项 & 非稳定模式隐式处理
  [2]  419_fem3d_sample       → 3D 场采样插值
  [3]  222_cosine_transform   → DCT 谱滤波 & 去混叠
  [4]  1417_wtime             → 挂钟计时性能报告
  [5]  1057_Bio-Inspired-Nav  → Pipeline 编排 & Config 模式
  [6]  755_mesh_etoe          → 结构网格单元邻接
  [7]  330_ellipse_grid       → Kerr-Schild 曲线网格
  [8]  1134_graphalgo         → BFS/DFS 因果连通性
  [9]  596_interp_trig        → 角向三角插值
  [10] 963_r83_np             → 三对角隐式求解 (Thomas)
  [11] 1008_C15StabilityDataW → 指数增长率拟合 (Arrhenius 思想)
  [12] 1246_finite-infinite   → Riccati 反馈控制 & 策略梯度
  [13] 1328_triangulate       → 极向截面耳切三角化
  [14] 624_knapsack_dynamic   → DP 最优模态选择
  [15] 095_bisection_integer  → 二分搜索临界波数
"""

from __future__ import annotations
import sys
import os

# 确保当前目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mhd_constants import MHDConfig, BlackHoleParameters, PlasmaParameters, NumericalParameters
from simulation_pipeline import MHDSimulationPipeline


def main() -> int:
    """
    主入口: 配置 → 模拟 → 分析 → 报告.
    """
    print("=" * 70)
    print("  PROJECT 252: 黑洞喷流形成与 MHD 模拟")
    print("  高阶有限差分与稳定性分析 (博士级可复现实验)")
    print("=" * 70)

    # 配置 (默认参数, 零参数运行)
    cfg = MHDConfig(
        bh=BlackHoleParameters(M_BH_Msun=1.0e8, a_star=0.9375),
        plasma=PlasmaParameters(
            gamma_ad=5.0 / 3.0,
            rho_floor=1.0e-4,
            p_floor=1.0e-6,
            eta_resist=1.0e-4,
            beta_init=0.1,
        ),
        num=NumericalParameters(
            nr=48,
            ntheta=24,
            nphi=1,
            r_in=2.0,
            r_out=50.0,
            fd_order=6,
            cfl=0.3,
            t_end=10.0,
            integrator="rk3-ssp",
            implicit_radial=True,
            n_modes_kept=12,
            omega_scan_pts=100,
        ),
    )

    # 运行模拟
    pipeline = MHDSimulationPipeline(cfg)
    results = pipeline.run()

    # 输出摘要
    print("\n" + "=" * 70)
    print("模拟结果摘要")
    print("=" * 70)
    print(f"  增长率拟合:     gamma = {results['growth_rate_fit']:.6e}")
    print(f"  拟合优度:       R^2   = {results['R2_fit']:.4f}")
    print(f"  因果影响域:     size  = {results['causal_influence_size']}")
    print(f"  LQR 闭环稳定:   rho   = {results['lqr_spectral_radius']:.4f}")
    print(f"  BZ 功率:        P     = {results['bz_power_erg_s']:.3e} erg/s")
    print(f"  时间步数:       n     = {results['n_steps']}")
    print(f"  最终扰动:       amp   = {results['final_density_perturbation']:.6e}")
    print("=" * 70)
    print("\n合成项目运行成功。所有 15 个种子项目算法已深度融入。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
