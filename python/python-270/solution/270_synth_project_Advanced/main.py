#!/usr/bin/env python3
"""
main.py
=======

统一入口: 自旋玻璃 Monte Carlo 模拟 + 高阶有限差分 + 稳定性分析
(小规模可复现实验)

物理问题
--------
Edwards-Anderson 自旋玻璃模型在三维立方晶格上的数值模拟。
结合:
- Metropolis Monte Carlo + Replica Exchange (Parallel Tempering)
- Langevin 动力学 + 高阶有限差分离散化 (2阶/4阶)
- von Neumann 稳定性分析
- Umbrella Sampling + WHAM (自由能景观)
- Hessian 谱分析 (magnon 态密度)
- 无序平均 (多个淬火耦合实现)
- K-means 聚类 (纯态识别)

核心公式
--------
哈密顿量:
    H = -sum_{<ij>} J_{ij} S_i S_j,   S_i = ±1
    J_{ij} ~ N(0, J_var)  (高斯耦合)

Langevin 方程:
    dS_i/dt = h_i^{eff} + sqrt(2T) eta_i(t)
    h_i^{eff} = sum_{j in nbr(i)} J_{ij} S_j

有限差分格式:
    2阶: (Delta_2 S)_i = sum_{j in nbr(i)} S_j - 6*S_i
    4阶: (Delta_4 S)_i = sum_mu [-(S_{i+2e_mu}+S_{i-2e_mu})
                                 +16*(S_{i+e_mu}+S_{i-e_mu})
                                 -30*S_i] / 12

稳定性条件 (von Neumann):
    2阶: dt <= 1 / (6*D)
    4阶: dt <= 1 / (8*D)

运行方法
--------
    python main.py

无需任何参数。所有配置参数在 simulation_engine.SimulationConfig 中定义。

输出
----
结果输出到 output/ 目录下的文本文件。
"""

import sys
import os

# 确保可以导入同目录的模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np

from simulation_engine import SimulationConfig, run_full_simulation


def main():
    """
    主函数: 配置参数 → 运行模拟 → 输出结果。
    """
    print("=" * 70)
    print("  Edwards-Anderson Spin Glass Simulation")
    print("  High-Order Finite Difference + Stability Analysis")
    print("  (Small-Scale Reproducible Experiment)")
    print("=" * 70)
    print()

    # 设置配置 (所有参数有默认值, 零参数可运行)
    config = SimulationConfig()

    # 打印关键参数
    print("Configuration:")
    print(f"  Lattice: {config.L}x{config.L}x{config.L} "
          f"(N={config.L**3} spins, z=6)")
    print(f"  Boundary: {config.boundary_type}")
    print(f"  Coupling: {config.distribution}, J_var={config.J_var}")
    print(f"  Disorder realizations: {config.n_realizations}")
    print(f"  Temperature list: {config.T_list}")
    print(f"  MC sweeps: {config.n_sweeps} (equil: {config.n_equil})")
    print(f"  Replica Exchange: R={config.RE_n_replicas}, "
          f"T in [{config.RE_T_min}, {config.RE_T_max}]")
    print(f"  Umbrella Sampling: {len(config.US_q0_list)} windows, "
          f"kappa={config.US_kappa}")
    print(f"  Langevin: dt={config.langevin_dt}, "
          f"order={config.langevin_fd_order}, method={config.langevin_method}")
    print(f"  Random seed: {config.seed}")
    print()

    # 运行完整模拟
    try:
        results = run_full_simulation(config)
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 打印最终摘要
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    # 稳定性
    stab = results.get("stability", {})
    af = stab.get("amplification_factors", {})
    print(f"\n[Stability]")
    print(f"  Max amplification (2nd order): {af.get('rho_max_2nd', 'N/A'):.6f}")
    print(f"  Max amplification (4th order): {af.get('rho_max_4th', 'N/A'):.6f}")

    # MC 温度扫描
    print(f"\n[MC Temperature Scan]")
    mc_scan = results.get("mc_temperature_scan", {})
    for T, r in mc_scan.items():
        print(f"  T={T:.2f}: E/N={r['mean_energy']:.4f}, "
              f"|m|={r['mean_mag']:.4f}, "
              f"q={r['mean_q']:.4f}, "
              f"g={r['binder_cumulant']:.4f}")

    # Replica Exchange
    re = results.get("replica_exchange", {})
    print(f"\n[Replica Exchange]")
    print(f"  Exchange ratio: {re.get('overall_exchange_ratio', 0):.4f}")

    # WHAM
    wham = results.get("wham", {})
    print(f"\n[WHAM]")
    print(f"  Converged: {wham.get('converged', False)}, "
          f"iterations: {wham.get('n_iterations', 0)}")

    # Spectral
    spectral = results.get("spectral", {})
    print(f"\n[Hessian Spectrum]")
    print(f"  lambda_min = {spectral.get('lambda_min', 0):.6f}")
    print(f"  n_negative = {spectral.get('n_negative', 0)}")
    print(f"  condition = {spectral.get('condition_number', 0):.2e}")

    # AT criterion
    at = results.get("AT_criterion", {})
    print(f"\n[de Almeida-Thouless]")
    print(f"  AT parameter: {at.get('at_parameter', 0):.4f}")
    print(f"  Stable: {at.get('is_stable', False)}")

    # Disorder average
    da = results.get("disorder_average", {})
    print(f"\n[Disorder Average ({da.get('n_realizations', 0)} realizations)]")
    for row in da.get("summary_table", []):
        print(f"  beta={row['beta']:.3f}: "
              f"E/N={row['mean_energy']:.4f}+/-{row['std_energy']:.4f}, "
              f"chi_SG={row['chi_SG']:.4f}")

    # Cluster
    cluster = results.get("cluster_analysis", {})
    print(f"\n[Cluster Analysis]")
    print(f"  Labels: {cluster.get('labels', [])}")
    print(f"  WSS: {[round(w, 4) for w in cluster.get('wss', [])]}")

    # 时间
    wall_time = results.get("wall_time_seconds", 0)
    print(f"\n[Timing]")
    print(f"  Wall time: {wall_time:.2f} seconds")

    print("\n" + "=" * 70)
    print(f"  All output files saved to: {config.output_dir}/")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
