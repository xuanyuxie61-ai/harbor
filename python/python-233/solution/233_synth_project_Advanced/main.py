#!/usr/bin/env python3
"""
main.py — 顶夸克质量测量统计建模: 高阶有限差分与稳定性分析
============================================================
统一入口文件, 零参数可运行。

本程序实现了一个完整的顶夸克质量测量分析流水线,
融合了 15 个种子项目的核心算法:

  1. fem1d_heat_steady  → 有限元弱形式 → 谱函数 Schrödinger 方程
  2. knapsack_brute     → 二进制暴力枚举 → 喷注配对组合优化
  3. cavity_flow_movie  → Navier-Stokes 压力求解 → 探测器扩散响应核
  4. duffing_ode        → Duffing RK4 求解 → 拟合非线性动力学
  5. hammersley         → Hammersley 准随机序列 → QMC 相空间采样
  6. hypersphere_mc     → 高维球面积分 → 多维 nuisance 积分
  7. cuda_loop          → GPU 线程索引 → 并行核计算
  8. monomial_symmetrize→ 多项式对称化 → 置换不变可观测量
  9. anishchenko_ode    → 开关动力学 → 收敛监测
 10. PierceRyan         → 边界碰撞分岔 → 迭代稳定性分析
 11. hankel_inverse     → 汉克尔变换 → 动量空间重构
 12. polygon_grid       → 多边形网格 → 六边形参数扫描
 13. quad_trapezoid     → 梯形积分 → 似然积分
 14. disk01_integrands  → 圆盘积分 → 角度积分
 15. nereagurru         → 盘演化 ODE → 阈值传播子

科学问题:
  在 LHC pp 碰撞 (√s = 13 TeV) 中, 通过 tt̄ ne变质量谱
  提取顶夸克 pole 质量 m_t, 使用轮廓似然比方法进行统计推断,
  高阶有限差分计算参数灵敏度, 非线性动力学方法分析拟合稳定性。

运行:
  python main.py
"""

import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from top_mass_pipeline import TopMassAnalysisPipeline


def main():
    """主入口函数。"""
    print("=" * 70)
    print("  PROJECT 233: 计算高能物理 — 顶夸克质量测量")
    print("  Top Quark Mass Statistical Modeling")
    print("  High-Order Finite Difference & Stability Analysis")
    print("=" * 70)
    print()
    print("  科学目标: 从 tt̄ ne变质量谱提取顶夸克 pole 质量")
    print("  方法: 轮廓似然比 + NRQCD 阈值截面 + 高阶有限差分")
    print("  实验条件: LHC Run 2, √s = 13 TeV, L = 139 fb⁻¹")
    print()

    # 配置
    config = {
        'm_top_true': 172.5,          # 真值 (GeV)
        'sqrt_s': 13000.0,            # 质心系能量 (GeV)
        'luminosity': 139.0,          # 积分光度 (fb⁻¹)
        'n_events': 10000,
        'm_scan_min': 167.0,          # 扫描下限
        'm_scan_max': 178.0,          # 扫描上限
        'm_scan_npoints': 20,         # 扫描点数
        'mass_bin_edges': None,       # 使用默认值
        'systematic_uncertainties': {
            'JES': 0.01,
            'b_energy': 0.005,
            'luminosity': 0.02,
            'PDF': 0.015,
            'alpha_s': 0.01,
        }
    }

    # 创建并运行流水线
    pipeline = TopMassAnalysisPipeline(config)
    results = pipeline.run(verbose=True)

    # 输出关键数值结果 (供程序化读取)
    print("\n" + "=" * 70)
    print("  数值结果 (机器可读)")
    print("=" * 70)

    fit_m = results.get('fit', {}).get('m_opt', 0)
    like_m = results.get('likelihood', {}).get('m_best_fit', 0)
    gamma_t = results.get('physics', {}).get('gamma_top', 0)
    alpha_s = results.get('physics', {}).get('alpha_s_mt', 0)
    d1 = results.get('finite_diff', {}).get('first_derivative', 0)
    dR = results.get('finite_diff', {}).get('richardson_deriv', 0)
    lyap = results.get('stability', {}).get('lyapunov', 0)

    print(f"  FIT_MTOP_OPT     = {fit_m:.6f}")
    print(f"  FIT_MTOP_PLR     = {like_m:.6f}")
    print(f"  GAMMA_TOP        = {gamma_t:.6f}")
    print(f"  ALPHA_S_MT       = {alpha_s:.6f}")
    print(f"  DSIGMA_DM_4TH    = {d1:.8f}")
    print(f"  DSIGMA_DM_RICH   = {dR:.8f}")
    print(f"  LYAPUNOV_EXP     = {lyap:.8f}")
    print(f"  TOTAL_TIME       = {results.get('total_time', 0):.2f}")

    print("\n  分析完成。")
    return results


if __name__ == '__main__':
    main()
