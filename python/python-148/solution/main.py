"""
main.py — 多尺度神经场脑机接口信号解码系统
=============================================
统一入口，零参数运行。

本项目将 15 个种子项目的核心算法融合为面向
"神经计算：脑机接口信号解码" 的博士级计算系统。

运行后将输出：
  1. 合成神经信号统计摘要
  2. E-I 神经质量模型稳定性分析（Jacobian、对数范数、Cauchy 根界）
  3. 脑连接组拓扑与渗流分析（Fiedler 值、跨越簇检测）
  4. BCI 解码性能评估（训练-测试框架）
  5. 电极阵列几何与空间覆盖度
"""

import sys
import os
import numpy as np

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bci_decoder import BCIPipeline


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print(f"\n--- {title} ---")


def main():
    print("\n")
    print("*" * 70)
    print("  多尺度神经场脑机接口信号解码系统")
    print("  Multiscale Neural Field BCI Signal Decoding System")
    print("*" * 70)
    print("\n初始化系统组件...")
    print("  [1] 多通道 E-I 神经质量振荡器阵列")
    print("  [2] 二维 Amari 神经场 PDE 求解器")
    print("  [3] Chebyshev 谱 / Gauss-Legendre 信号分析器")
    print("  [4] 脑连接组图模型与渗流分析器")
    print("  [5] 稳定性与 Cauchy 根界分析器")
    print("  [6] CVT 最优电极采样与皮层几何")
    print("  [7] BCI 解码器（岭回归最优线性估计）")

    pipeline = BCIPipeline(random_state=148)
    results = pipeline.run_full_analysis()

    # ===== 信号统计 =====
    print_section("1. 合成神经信号统计摘要")
    sig = results['signal_summary']
    print(f"  电极数量      : {sig['n_electrodes']}")
    print(f"  时间采样点数  : {sig['n_timepoints']}")
    print(f"  LFP 峰值幅度  : {sig['lfp_peak_amplitude']:.4f} mV (归一化)")
    print(f"  神经场峰值幅度: {sig['field_peak_amplitude']:.4f} (归一化)")

    # ===== 稳定性分析 =====
    print_section("2. E-I 神经质量模型稳定性分析")
    stab = results['stability_analysis']
    print_subsection("平衡点")
    print(f"  E* = {stab['equilibrium_EI'][0]:.6f}")
    print(f"  I* = {stab['equilibrium_EI'][1]:.6f}")
    print_subsection("Jacobian 特征值")
    for k, (re, im) in enumerate(zip(stab['jacobian_eigenvalues_real'],
                                      stab['jacobian_eigenvalues_imag'])):
        print(f"  λ_{k+1} = {re:+.6f} {im:+.6f}j")
    print_subsection("矩阵对数范数")
    print(f"  μ_1(A)   = {stab['mu_1']:.6f}")
    print(f"  μ_2(A)   = {stab['mu_2']:.6f}")
    print(f"  μ_∞(A)   = {stab['mu_inf']:.6f}")
    print_subsection("稳定性判据")
    print(f"  Cauchy 根界 = {stab['cauchy_root_bound']:.6f}")
    print(f"  开环稳定性  = {'稳定' if stab['is_stable'] else '不稳定'}")
    if stab['is_stable']:
        print("  => 系统指数稳定：||exp(At)||_2 ≤ exp(μ_2(A)·t) → 0")
    else:
        print("  => 系统存在不稳定模式，需要闭环反馈镇定")

    # ===== 连接组分析 =====
    print_section("3. 脑连接组拓扑与渗流分析")
    conn = results['connectome_analysis']
    print(f"  Fiedler 值 (λ_2)     : {conn['fiedler_value']:.6f}")
    print(f"  图连通性             : {'连通' if conn['fiedler_value'] > 1e-10 else '不连通'}")
    print(f"  最大信息传播簇大小   : {conn['max_cluster_size']}")
    print(f"  是否形成跨越簇       : {'是' if conn['is_spanning'] else '否'}")
    print(f"  => Fiedler 值反映图的全局连通强度；")
    print(f"     跨越簇的存在标志信息可从感觉皮层传播至运动皮层")

    # ===== 解码性能 =====
    print_section("4. BCI 运动意图解码性能")
    perf = results['decoding_performance']
    print(f"  训练样本数      : 15")
    print(f"  测试样本数      : 10")
    print(f"  平均解码误差    : {perf['mean_decoding_error']:.4f}")
    print(f"  解码误差标准差  : {perf['std_decoding_error']:.4f}")
    print(f"  各试次误差      : {[round(e, 4) for e in perf['test_errors']]}")
    print(f"  解码器权重维度  : {perf['decoder_weights_shape']}")
    print(f"  => 解码误差量化了从神经信号恢复 2D 运动速度向量的精度")

    # ===== 电极几何 =====
    print_section("5. 电极阵列几何与空间采样")
    geom = results['electrode_geometry']
    print(f"  电极数量        : {geom['n_electrodes']}")
    print(f"  空间覆盖面积    : {geom['spatial_coverage']:.4f} mm²")
    print(f"  Delaunay 三角形 : {geom['n_triangles']} 个")
    print(f"  => CVT 优化布局最小化 Voronoi 能量泛函，")
    print(f"     实现空间采样最优性")

    # ===== 科学公式摘要 =====
    print_section("6. 核心科学公式摘要")
    print("""
  (a) E-I 神经质量模型:
      dE/dt = -E + S_e(a_ee·E - a_ei·I + P_e + k_e·s(t))
      dI/dt = -I + S_i(a_ie·E - a_ii·I + P_i + k_i·s(t))
      s(t) = A_s · (mod(t+π/ω, 2π/ω) - π/ω)

  (b) Amari 神经场方程:
      τ·∂u/∂t = -u + ∫ K(r,r')·S(u(r')) dr' + I_ext(r,t)
      K(r) = A_e·exp(-|r|²/2σ_e²) - A_i·exp(-|r|²/2σ_i²)

  (c) 矩阵对数范数与稳定性:
      ||exp(At)||_p ≤ exp(μ_p(A)·t)
      μ_2(A) = λ_max((A + Aᵀ)/2)

  (d) Cauchy 多项式根界:
      q(x) = |c₁|xⁿ - Σ_{k=2}^{n+1} |c_k| x^{n-k+1} = 0
      唯一正根 r 为所有特征值模的上界

  (e) CVT 最优采样能量:
      F(Z) = Σ_i ∫_{V_i} ρ(r)·|r - z_i|² dr

  (f) 解码目标泛函:
      m* = argmin_m ( ||S - Φ(m)||²_F + λ·||Dm||² )
    """)

    print("\n" + "=" * 70)
    print("  系统运行完毕。所有模块协同工作，无报错。")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    main()
