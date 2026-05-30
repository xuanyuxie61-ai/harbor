#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys


from spike_neuron import (
    HHNeuron, NeuronPopulation, demo_single_neuron, demo_population
)
from axon_propagation import (
    DG1DNeuralCable, MHDNeuralCoupling,
    demo_axon_propagation, demo_mhd_coupling
)
from synaptic_encoding import (
    AlphaSynapse, optimal_synaptic_weights, polynomial_multiply_convolution,
    demo_encoding
)
from spike_pattern import (
    SpikePatternAnalyzer, connected_spike_patterns_2d, pattern_clustering,
    demo_pattern_analysis, demo_polyomino_mapping
)
from signal_reconstruction import (
    HermiteInterpolator, StabilityAnalyzer, SignalReconstructor,
    demo_hermite_reconstruction, demo_stability_analysis
)
from cortical_grid import (
    CorticalGrid, demo_grid_encoding, demo_distance_stats
)
from stochastic_weights import (
    LogNormalSynapse, SynapticWeightSDE, normalize_weights_multiplicative,
    demo_log_normal_weights, demo_weight_sde
)
from brain_field import (
    EthierNavierStokes, KeastTetrahedronRule, NeuralVolumeIntegral,
    demo_navier_stokes, demo_tetrahedron_integral, demo_volume_integral
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_neuron_dynamics():
    print_section("模块 1: Hodgkin-Huxley 脉冲神经元动力学")


    print("\n[单神经元恒定电流注入]")
    V_trace, spikes = demo_single_neuron()
    print(f"  仿真时长: 50 ms, 时间步长: 0.01 ms")
    print(f"  发放脉冲数: {len(spikes)}")
    if len(spikes) > 0:
        print(f"  首次发放时刻: {spikes[0]:.2f} ms")
        isi = np.diff(spikes)
        print(f"  平均 ISI: {np.mean(isi):.2f} ms, 发放率: {1000.0/np.mean(isi):.1f} Hz")


    print("\n[兴奋-抑制神经元群体 (E=10, I=5)]")
    voltage_trace, spike_raster, pop_spikes = demo_population()
    total_spikes = sum(len(s) for s in pop_spikes)
    print(f"  仿真时长: 30 ms, 总脉冲数: {total_spikes}")
    print(f"  群体平均发放率: {total_spikes / (15 * 30.0) * 1000:.1f} Hz/神经元")


def run_axon_propagation():
    print_section("模块 2: 轴突 DG 离散信号传播与 MHD 耦合")

    print("\n[一维神经电缆 DG 仿真]")
    u_final, x_coords = demo_axon_propagation()
    print(f"  空间区间: [0, 10], 单元数: 20, 每单元节点: 4")
    print(f"  最终波形幅度范围: [{np.min(u_final):.3f}, {np.max(u_final):.3f}] mV")
    print(f"  波形质心位置: {np.mean(x_coords * np.abs(u_final)) / (np.mean(np.abs(u_final)) + 1e-12):.2f}")

    print("\n[MHD 电磁耦合对电导率的调制]")
    x, V, correction = demo_mhd_coupling()
    print(f"  离子流修正因子范围: [{np.min(correction):.4f}, {np.max(correction):.4f}]")
    print(f"  平均修正: {np.mean(correction):.4f} (1.0 表示无调制)")


def run_synaptic_encoding():
    print_section("模块 3: 能量约束下的最优突触编码")

    print("\n[突触权重优化 (背包问题)]")
    weights, encoded, mi, spike_times = demo_encoding()
    print(f"  候选脉冲数: {len(spike_times)}")
    print(f"  激活权重数: {np.sum(np.abs(weights) > 0.01)}")
    print(f"  权重 L1 范数: {np.sum(np.abs(weights)):.3f}")
    print(f"  近似互信息: {mi:.3f} bits")

    print("\n[离散卷积验证 (多项式乘法)]")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.0])
    c = polynomial_multiply_convolution(a, b)
    print(f"  a = {a}, b = {b}")
    print(f"  a * b = {c}")

    expected = np.convolve(a, b)
    print(f"  numpy.convolve 验证: {expected}")
    print(f"  误差: {np.max(np.abs(c - expected)):.2e}")


def run_spike_pattern():
    print_section("模块 4: 脉冲模式组合枚举与聚类")

    print("\n[脉冲模式熵与聚类]")
    entropy, labels, capacity = demo_pattern_analysis()
    n_clusters = len(np.unique(labels))
    print(f"  8-bin 脉冲模式, 50 个随机样本")
    print(f"  经验香农熵: {entropy:.3f} bits")
    print(f"  聚类数 (汉明距离<=1): {n_clusters}")
    print(f"  一维连通模式容量: {capacity:.2f} bits")

    print("\n[二维感受野连通模式 (Polyomino 映射)]")
    counts = demo_polyomino_mapping()
    print(f"  3x3 感受野网格, 连通模式计数 (前 5 阶):")
    for m, c in counts[:5]:
        print(f"    阶数 {m}: {c} 种连通模式")


def run_signal_reconstruction():
    print_section("模块 5: Hermite 插值重建与数值稳定性")

    print("\n[Hermite 插值重建]")
    t_fine, v_recon = demo_hermite_reconstruction()
    print(f"  插值节点数: 7, 重建点: 200")
    print(f"  重建信号范围: [{np.min(v_recon):.3f}, {np.max(v_recon):.3f}]")

    print("\n[RK4 稳定性分析]")
    X, Y, Rabs, boundary = demo_stability_analysis()
    print(f"  复平面网格: [{X.min():.1f}, {X.max():.1f}] x [{Y.min():.1f}, {Y.max():.1f}]")
    print(f"  稳定性区域采样点: {len(boundary)}")
    stab = StabilityAnalyzer()
    lambda_max = 50.0
    dt_max = stab.max_stable_timestep(lambda_max)
    print(f"  HH 特征值上界 |lambda| ≈ {lambda_max}")
    print(f"  RK4 最大稳定步长: dt <= {dt_max:.4f} ms")

    print(f"  当前使用 dt=0.01 ms，稳定性裕度: {dt_max/0.01:.1f}x")


def run_cortical_grid():
    print_section("模块 6: 皮层编码网格与距离统计")

    print("\n[皮层连接矩阵与感受野]")
    W, dmu, dvar, rf = demo_grid_encoding()
    print(f"  网格尺寸: 5x5, 总节点: 25")
    print(f"  平均连接权重: {np.mean(W):.4f}")
    print(f"  最大连接权重: {np.max(W):.4f}")
    print(f"  感受野中心权重: {rf[12]:.4f}, 近邻权重: {rf[11]:.4f}")

    print("\n[皮层表面距离统计]")
    dmu2, dvar2 = demo_distance_stats()
    print(f"  10x10 网格, 5000 次随机采样")
    print(f"  平均距离: {dmu2:.4f}")
    print(f"  距离方差: {dvar2:.4f}")
    print(f"  理论参考 (单位正方形连续): E[d]≈0.521, Var[d]≈0.062")


def run_stochastic_weights():
    print_section("模块 7: 对数正态突触权重与随机演化")

    print("\n[对数正态权重分布]")
    result = demo_log_normal_weights()
    print(f"  参数: mu={-0.5}, sigma={0.8}")
    print(f"  理论均值: {result['theoretical_mean']:.4f}")
    print(f"  经验均值 (N=5000): {result['empirical_mean']:.4f}")
    print(f"  理论方差: {result['theoretical_var']:.4f}")
    print(f"  经验方差 (N=5000): {result['empirical_var']:.4f}")

    print("\n[权重 SDE 演化轨迹]")
    traj = demo_weight_sde()
    print(f"  Ornstein-Uhlenbeck 对数权重, T=50 ms")
    print(f"  初始权重: 1.0, 终止权重: {traj[-1]:.4f}")
    print(f"  轨迹均值: {np.mean(traj):.4f}, 标准差: {np.std(traj):.4f}")

    print("\n[突触归一化]")
    raw_weights = np.random.uniform(0.1, 2.0, size=10)
    norm_weights = normalize_weights_multiplicative(raw_weights, target_sum=1.0)
    print(f"  原始权重和: {np.sum(raw_weights):.4f}")
    print(f"  归一化后权重和: {np.sum(norm_weights):.4f}")


def run_brain_field():
    print_section("模块 8: 脑 Navier-Stokes 流场与体积积分")

    print("\n[三维 Navier-Stokes 精确解 (Ethier)]")
    u, v, w, p = demo_navier_stokes()
    print(f"  参数: a=pi/4, d=pi/2")
    print(f"  位置 (0.5,0.5,0.5), t=0.05")
    print(f"  速度: u={u:.4f}, v={v:.4f}, w={w:.4f}")
    print(f"  压力: p={p:.4f}")

    print("\n[四面体 Keast 积分]")
    result = demo_tetrahedron_integral()
    print(f"  积分 f(x,y,z)=x+y+z 在单位四面体上")
    print(f"  数值结果: {result:.6f}")
    print(f"  解析解: 0.125000")
    print(f"  绝对误差: {abs(result - 0.125):.2e}")

    print("\n[神经核团体积积分]")
    total = demo_volume_integral()
    print(f"  两个四面体构成的区域, V_membrane=-50 mV")
    print(f"  总离子电荷密度积分: {total:.4e} C/m^3 * m^3")


def run_integrated_summary():
    print_section("综合评估: 多物理场耦合 SNN 系统性能")


    print("\n[端到端编码-传输-解码验证]")


    t_grid = np.linspace(0, 50, 500)
    signal_true = 3.0 * np.sin(2.0 * np.pi * 0.03 * t_grid)






    raise NotImplementedError("Hole 3: 请补全端到端脉冲编码循环")


    weights, encoded, mi = optimal_synaptic_weights(
        spike_times, signal_true, t_grid, tau_s=2.5, E_budget=5.0
    )


    if len(spike_times) >= 4:
        recon = SignalReconstructor(spike_times)
        metrics = recon.reconstruction_quality(t_grid, signal_true)
    else:
        metrics = {'MSE': float('nan'), 'SNR_dB': float('nan'), 'Correlation': float('nan')}


    grid = CorticalGrid(nx=4, ny=4)
    poly_counts = connected_spike_patterns_2d(4, 4, max_order=5)
    capacity_per_neuron = np.log2(poly_counts[-1][1]) if poly_counts else 0.0

    print(f"  原始信号频率: 0.03 Hz (时间窗 50 ms)")
    print(f"  编码脉冲数: {len(spike_times)}")
    print(f"  编码互信息: {mi:.3f} bits")
    print(f"  重建 MSE: {metrics['MSE']:.4f}")
    print(f"  重建 SNR: {metrics['SNR_dB']:.2f} dB")
    print(f"  重建相关系数: {metrics['Correlation']:.3f}")
    print(f"  4x4 皮层网格单神经元容量上界: {capacity_per_neuron:.1f} bits")

    print("\n" + "=" * 70)
    print("  所有模块运行完毕，无报错。")
    print("=" * 70)


def main():
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("  脉冲神经网络编码与解码 — 多物理场耦合博士级合成项目")
    print("  PROJECT_146: 神经计算方向")
    print("=" * 70)

    run_neuron_dynamics()
    run_axon_propagation()
    run_synaptic_encoding()
    run_spike_pattern()
    run_signal_reconstruction()
    run_cortical_grid()
    run_stochastic_weights()
    run_brain_field()
    run_integrated_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
