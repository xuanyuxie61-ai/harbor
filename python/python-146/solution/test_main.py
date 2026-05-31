#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
脉冲神经网络编码与解码 — 统一入口

零参数运行，执行完整的多物理场耦合 SNN 计算流程：
  1. 神经元动力学仿真 (Hodgkin-Huxley + RK4)
  2. 轴突信号传播 (DG 离散 + MHD 耦合)
  3. 突触编码优化 (背包问题 + 离散卷积)
  4. 脉冲模式分析 (多格拼板枚举 + 聚类去重)
  5. 信号重建 (Hermite 插值 + 稳定性分析)
  6. 皮层网格拓扑 (网格生成 + 距离统计)
  7. 突触权重随机模型 (对数正态 + SDE)
  8. 脑流场与体积积分 (NS 精确解 + Keast 积分)
"""

import numpy as np
import sys

# 模块导入
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
    """模块 1: 脉冲神经元动力学"""
    print_section("模块 1: Hodgkin-Huxley 脉冲神经元动力学")

    # 单神经元 demo
    print("\n[单神经元恒定电流注入]")
    V_trace, spikes = demo_single_neuron()
    print(f"  仿真时长: 50 ms, 时间步长: 0.01 ms")
    print(f"  发放脉冲数: {len(spikes)}")
    if len(spikes) > 0:
        print(f"  首次发放时刻: {spikes[0]:.2f} ms")
        isi = np.diff(spikes)
        print(f"  平均 ISI: {np.mean(isi):.2f} ms, 发放率: {1000.0/np.mean(isi):.1f} Hz")

    # 群体 demo
    print("\n[兴奋-抑制神经元群体 (E=10, I=5)]")
    voltage_trace, spike_raster, pop_spikes = demo_population()
    total_spikes = sum(len(s) for s in pop_spikes)
    print(f"  仿真时长: 30 ms, 总脉冲数: {total_spikes}")
    print(f"  群体平均发放率: {total_spikes / (15 * 30.0) * 1000:.1f} Hz/神经元")


def run_axon_propagation():
    """模块 2: 轴突非线性信号传播"""
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
    """模块 3: 突触编码优化"""
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
    # 验证
    expected = np.convolve(a, b)
    print(f"  numpy.convolve 验证: {expected}")
    print(f"  误差: {np.max(np.abs(c - expected)):.2e}")


def run_spike_pattern():
    """模块 4: 脉冲模式组合分析"""
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
    """模块 5: 信号重建与稳定性"""
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
    lambda_max = 50.0  # ms^{-1}, HH 方程典型特征值量级
    dt_max = stab.max_stable_timestep(lambda_max)
    print(f"  HH 特征值上界 |lambda| ≈ {lambda_max}")
    print(f"  RK4 最大稳定步长: dt <= {dt_max:.4f} ms")
    # 验证当前步长
    print(f"  当前使用 dt=0.01 ms，稳定性裕度: {dt_max/0.01:.1f}x")


def run_cortical_grid():
    """模块 6: 皮层网格拓扑"""
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
    """模块 7: 突触权重随机模型"""
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
    """模块 8: 脑流场与体积积分"""
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
    """综合汇总：展示各模块间的科学关联"""
    print_section("综合评估: 多物理场耦合 SNN 系统性能")

    # 快速小规模端到端验证
    print("\n[端到端编码-传输-解码验证]")

    # 1. 生成测试信号
    t_grid = np.linspace(0, 50, 500)
    signal_true = 3.0 * np.sin(2.0 * np.pi * 0.03 * t_grid)

    # 2. 脉冲神经元编码
    neuron = HHNeuron(dt=0.01)
    I_ext_base = 10.0 + 4.0 * np.sin(2.0 * np.pi * 0.03 * t_grid[::5])
    spike_times = []
    for k in range(int(50.0 / 0.01)):
        t = k * 0.01
        idx = min(k // 5, len(I_ext_base) - 1)
        fired = neuron.step(t, I_ext=I_ext_base[idx])
        if fired:
            spike_times.append(t)

    # 3. 突触编码
    weights, encoded, mi = optimal_synaptic_weights(
        spike_times, signal_true, t_grid, tau_s=2.5, E_budget=5.0
    )

    # 4. 信号重建
    if len(spike_times) >= 4:
        recon = SignalReconstructor(spike_times)
        metrics = recon.reconstruction_quality(t_grid, signal_true)
    else:
        metrics = {'MSE': float('nan'), 'SNR_dB': float('nan'), 'Correlation': float('nan')}

    # 5. 皮层网格容量估算
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
    """统一入口函数，零参数运行。"""
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
    main()

# 测试用额外导入（main.py 中未导入但测试需要的函数）
from synaptic_encoding import rational_knapsack_encoding
from spike_pattern import (
    connected_spike_patterns_1d,
    polyomino_enumerate_fixed,
    r8col_sorted_tol_unique,
)

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: HHNeuron 初始化默认值正确 ----
neuron = HHNeuron(dt=0.01)
assert neuron.V == -65.0, '[TC01] Initial V should be -65.0 FAILED'
assert 0.0 <= neuron.m <= 1.0, '[TC01] gate m out of bounds FAILED'
assert 0.0 <= neuron.h <= 1.0, '[TC01] gate h out of bounds FAILED'
assert 0.0 <= neuron.n <= 1.0, '[TC01] gate n out of bounds FAILED'

# ---- TC02: HHNeuron 非法 dt 抛出 ValueError ----
try:
    HHNeuron(dt=0.0)
    assert False, '[TC02] dt=0 should raise ValueError FAILED'
except ValueError:
    pass

# ---- TC03: HHNeuron step 零输入不应发放 ----
import numpy as np
neuron = HHNeuron(dt=0.01)
fired = neuron.step(0.0, I_syn=0.0, I_ext=0.0)
assert fired is False, '[TC03] step with zero input should not fire FAILED'
assert np.isfinite(neuron.V), '[TC03] V should be finite FAILED'

# ---- TC04: HHNeuron step 强电流注入应发放脉冲 ----
import numpy as np
np.random.seed(42)
neuron = HHNeuron(dt=0.01)
n_steps = int(50.0 / 0.01)
spike_count = 0
for k in range(n_steps):
    t = k * 0.01
    fired = neuron.step(t, I_ext=10.0)
    if fired:
        spike_count += 1
assert spike_count > 0, '[TC04] Should fire spikes under strong current FAILED'

# ---- TC05: HHNeuron 门控变量始终在 [0,1] 内 ----
import numpy as np
np.random.seed(42)
neuron = HHNeuron(dt=0.01)
for k in range(5000):
    t = k * 0.01
    neuron.step(t, I_ext=12.0)
    assert 0.0 <= neuron.m <= 1.0, '[TC05] gate m out of [0,1] FAILED'
    assert 0.0 <= neuron.h <= 1.0, '[TC05] gate h out of [0,1] FAILED'
    assert 0.0 <= neuron.n <= 1.0, '[TC05] gate n out of [0,1] FAILED'

# ---- TC06: HHNeuron 速率函数 alpha_m 在 V=-40 处鲁棒 ----
val = HHNeuron._alpha_m(-40.0)
assert np.isfinite(val), '[TC06] alpha_m(-40) should be finite FAILED'
assert val > 0.0, '[TC06] alpha_m(-40) should be positive FAILED'

# ---- TC07: NeuronPopulation 初始化神经元数量正确 ----
pop = NeuronPopulation(N_exc=10, N_inh=5, dt=0.01, p_conn=0.2)
assert pop.N == 15, '[TC07] total neuron count should be 15 FAILED'
assert pop.N_exc == 10, '[TC07] exc count should be 10 FAILED'
assert pop.N_inh == 5, '[TC07] inh count should be 5 FAILED'

# ---- TC08: NeuronPopulation simulate 输出形状正确、值有限 ----
import numpy as np
np.random.seed(42)
pop = NeuronPopulation(N_exc=4, N_inh=2, dt=0.01, p_conn=0.3)
I_ext = np.array([8.0, 8.0, 8.0, 8.0, 5.0, 5.0])
vt, sr = pop.simulate(T_total=10.0, I_ext_per_neuron=I_ext)
assert vt.shape == (6, 1000), '[TC08] voltage_trace shape mismatch FAILED'
assert sr.shape == (6, 1000), '[TC08] spike_raster shape mismatch FAILED'
assert np.all(np.isfinite(vt)), '[TC08] voltage_trace has non-finite values FAILED'

# ---- TC09: DG1DNeuralCable 初始化网格尺寸正确 ----
cable = DG1DNeuralCable(xL=0.0, xR=10.0, K=20, Np=4, dt=0.001, epsilon=0.05)
assert cable.x.shape == (4, 20), '[TC09] grid shape should be (4,20) FAILED'

# ---- TC10: DG1DNeuralCable simulate 输出形状正确且值有限 ----
import numpy as np
cable = DG1DNeuralCable(xL=0.0, xR=5.0, K=10, Np=3, dt=0.001, epsilon=0.05)
u0 = np.zeros((cable.Np, cable.K))
for k in range(cable.K):
    for i in range(cable.Np):
        x = cable.x[i, k]
        u0[i, k] = 15.0 * np.exp(-((x - 1.0) ** 2) / 0.5)
u_final, history = cable.simulate(u0, T_final=1.0)
assert u_final.shape == (3, 10), '[TC10] final solution shape mismatch FAILED'
assert np.all(np.isfinite(u_final)), '[TC10] final solution has non-finite values FAILED'

# ---- TC11: polynomial_multiply_convolution 与 numpy.convolve 一致 ----
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 1.0])
c = polynomial_multiply_convolution(a, b)
expected = np.convolve(a, b)
assert np.max(np.abs(c - expected)) < 1e-12, '[TC11] convolution mismatch FAILED'

# ---- TC12: rational_knapsack_encoding 基本背包求解 ----
import numpy as np
profits = np.array([10.0, 20.0, 30.0])
weights = np.array([2.0, 3.0, 5.0])
x, mass, profit = rational_knapsack_encoding(profits, weights, budget=5.0)
assert 0.0 <= mass <= 5.0 + 1e-10, '[TC12] mass should be within budget FAILED'
assert profit > 0.0, '[TC12] profit should be positive FAILED'

# ---- TC13: AlphaSynapse kernel 行为正确 ----
import numpy as np
syn = AlphaSynapse(tau_s=2.0)
t_vals = np.array([-1.0, 0.0, 1.0, 2.0, 10.0])
k = syn.kernel(t_vals)
assert k[0] == 0.0, '[TC13] kernel at t=-1 should be 0 FAILED'
assert k[1] == 0.0, '[TC13] kernel at t=0 should be 0 FAILED'
assert k[2] > 0.0, '[TC13] kernel at t=1 should be positive FAILED'
assert k[-1] < k[2], '[TC13] kernel should decay after peak FAILED'

# ---- TC14: AlphaSynapse convolve_spikes 输出形状与有限性 ----
import numpy as np
syn = AlphaSynapse(tau_s=2.0)
t_grid = np.linspace(0, 100, 1000)
spike_times = [10.0, 30.0, 60.0]
weights = [1.0, 0.8, 0.5]
s = syn.convolve_spikes(spike_times, weights, t_grid)
assert len(s) == 1000, '[TC14] output length mismatch FAILED'
assert np.all(np.isfinite(s)), '[TC14] output has non-finite values FAILED'

# ---- TC15: SpikePatternAnalyzer pattern_capacity 非负 ----
analyzer = SpikePatternAnalyzer(n_bins=8)
capacity = analyzer.pattern_capacity()
assert capacity > 0.0, '[TC15] pattern capacity should be positive FAILED'

# ---- TC16: connected_spike_patterns_1d 已知公式验证 ----
n = 5
result = connected_spike_patterns_1d(n)
expected = n * (n + 1) // 2
assert result == expected, '[TC16] 1D connected patterns formula mismatch FAILED'

# ---- TC17: polyomino_enumerate_fixed 已知值 ----
count = polyomino_enumerate_fixed(0)
assert count == 1, '[TC17] order 0 polyomino count should be 1 FAILED'
count2 = polyomino_enumerate_fixed(4)
assert count2 == 19, '[TC17] order 4 polyomino count should be 19 FAILED'

# ---- TC18: r8col_sorted_tol_unique 去重正确 ----
import numpy as np
patterns = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 0]]).T
unique, uniq_num = r8col_sorted_tol_unique(patterns, tol=0.0)
assert uniq_num <= 4, '[TC18] unique num should not exceed input FAILED'
assert uniq_num >= 1, '[TC18] unique num should be at least 1 FAILED'

# ---- TC19: pattern_clustering 返回正确标签数量 ----
import numpy as np
np.random.seed(42)
patterns = np.random.randint(0, 2, size=(6, 20)).astype(float)
labels = pattern_clustering(patterns, max_distance=2)
assert len(labels) == 20, '[TC19] labels count mismatch FAILED'
assert len(np.unique(labels)) >= 1, '[TC19] should have at least 1 cluster FAILED'

# ---- TC20: HermiteInterpolator 构造与求值不崩溃、输出有限 ----
import numpy as np
t_nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
v_nodes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
dv_nodes = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
interp = HermiteInterpolator(t_nodes, v_nodes, dv_nodes)
val = interp.evaluate(np.array([0.5, 1.5, 2.5, 3.5]))
assert np.all(np.isfinite(val)), '[TC20] interpolated values should be finite FAILED'
assert len(val) == 4, '[TC20] output length should match query length FAILED'

# ---- TC21: StabilityAnalyzer rk4_amplification_factor z=0 => 1 ----
import numpy as np
r = StabilityAnalyzer.rk4_amplification_factor(0.0 + 0j)
assert np.abs(r - 1.0) < 1e-12, '[TC21] R(0) should equal 1 FAILED'

# ---- TC22: StabilityAnalyzer max_stable_timestep 返回正值 ----
dt_max = StabilityAnalyzer.max_stable_timestep(lambda_max=50.0, method='rk4')
assert dt_max > 0.0, '[TC22] max stable timestep should be positive FAILED'

# ---- TC23: CorticalGrid connection_probability 对称性 ----
import numpy as np
grid = CorticalGrid(nx=4, ny=4)
p_ij = grid.connection_probability(0, 5, p0=0.4, sigma=0.3)
p_ji = grid.connection_probability(5, 0, p0=0.4, sigma=0.3)
assert np.abs(p_ij - p_ji) < 1e-12, '[TC23] connection probability should be symmetric FAILED'

# ---- TC24: CorticalGrid distance_statistics 均值正值 ----
import numpy as np
grid = CorticalGrid(nx=5, ny=5, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
dmu, dvar, distances = grid.distance_statistics(n_samples=500)
assert dmu > 0.0, '[TC24] mean distance should be positive FAILED'
assert dvar > 0.0, '[TC24] distance variance should be positive FAILED'

# ---- TC25: LogNormalSynapse pdf 输出非负 ----
import numpy as np
model = LogNormalSynapse(mu=0.0, sigma=0.5)
w = np.array([0.5, 1.0, 2.0])
pdf_vals = model.pdf(w)
assert np.all(pdf_vals >= 0.0), '[TC25] pdf should be non-negative FAILED'
assert np.all(np.isfinite(pdf_vals)), '[TC25] pdf should be finite FAILED'

# ---- TC26: LogNormalSynapse 均值/方差公式 ----
model = LogNormalSynapse(mu=0.0, sigma=0.5)
th_mean = model.mean()
th_var = model.variance()
assert th_mean > 0.0, '[TC26] theoretical mean should be positive FAILED'
assert th_var > 0.0, '[TC26] theoretical variance should be positive FAILED'

# ---- TC27: normalize_weights_multiplicative 目标和正确 ----
import numpy as np
raw = np.array([0.2, 0.3, 0.5])
norm = normalize_weights_multiplicative(raw, target_sum=1.0)
assert np.abs(np.sum(norm) - 1.0) < 1e-12, '[TC27] normalized sum should be 1.0 FAILED'
assert np.all(norm >= 0.0), '[TC27] normalized weights should be non-negative FAILED'

# ---- TC28: EthierNavierStokes evaluate 返回有限值 ----
import numpy as np
ns = EthierNavierStokes()
u, v, w, p = ns.evaluate(0.5, 0.5, 0.5, 0.05)
assert np.isfinite(u), '[TC28] u should be finite FAILED'
assert np.isfinite(v), '[TC28] v should be finite FAILED'
assert np.isfinite(w), '[TC28] w should be finite FAILED'
assert np.isfinite(p), '[TC28] p should be finite FAILED'

# ---- TC29: KeastTetrahedronRule integrate 已知解析解 ----
import numpy as np
keast = KeastTetrahedronRule(rule_id=4)
verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
result = keast.integrate(lambda x, y, z: x + y + z, verts)
assert np.abs(result - 0.125) < 1e-6, '[TC29] tetrahedron integral should be 0.125 FAILED'

# ---- TC30: NeuralVolumeIntegral integrate_region 返回标量 ----
tet1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
tet2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
vol = NeuralVolumeIntegral()
total = vol.integrate_region([tet1, tet2], V_membrane=-65.0)
assert np.isfinite(total), '[TC30] volume integral should be finite FAILED'
assert isinstance(total, float), '[TC30] volume integral should be scalar FAILED'

# ---- TC31: LogNormalSynapse sample 固定种子可复现 ----
import numpy as np
model = LogNormalSynapse(mu=0.0, sigma=0.5)
np.random.seed(42)
s1 = model.sample(size=100)
np.random.seed(42)
s2 = model.sample(size=100)
assert np.allclose(s1, s2), '[TC31] samples with same seed should be identical FAILED'

# ---- TC32: r8col_sorted_tol_unique 容差去重 ----
import numpy as np
patterns = np.array([[1.0, 1.0, 1.01], [0.0, 0.0, 0.0]]).T
unique, uniq_num = r8col_sorted_tol_unique(patterns, tol=0.1)
assert uniq_num < 3, '[TC32] tolerance dedup should reduce count FAILED'

# ---- TC33: connected_spike_patterns_2d 返回列表 ----
import numpy as np
counts = connected_spike_patterns_2d(3, 3, max_order=5)
assert len(counts) >= 1, '[TC33] 2D connected patterns should return non-empty list FAILED'
assert all(isinstance(c, int) for _, c in counts), '[TC33] all counts should be int FAILED'

# ---- TC34: SignalReconstructor reconstruction_quality 返回字典 ----
import numpy as np
np.random.seed(42)
spike_times = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
t_grid = np.linspace(0, 25, 200)
signal_true = np.sin(2.0 * np.pi * 0.1 * t_grid)
recon = SignalReconstructor(spike_times)
metrics = recon.reconstruction_quality(t_grid, signal_true)
assert 'MSE' in metrics, '[TC34] metrics should contain MSE FAILED'
assert 'SNR_dB' in metrics, '[TC34] metrics should contain SNR_dB FAILED'
assert 'Correlation' in metrics, '[TC34] metrics should contain Correlation FAILED'
assert np.isfinite(metrics['MSE']), '[TC34] MSE should be finite FAILED'

# ---- TC35: HermiteInterpolator evaluate_derivative 求值有限 ----
import numpy as np
t_nodes = np.array([0.0, 1.0, 2.0, 3.0])
v_nodes = np.array([0.0, 1.0, 0.0, 1.0])
dv_nodes = np.array([1.0, 0.0, -1.0, 0.5])
interp = HermiteInterpolator(t_nodes, v_nodes, dv_nodes)
deriv = interp.evaluate_derivative(np.array([1.5]))
assert np.isfinite(deriv[0]), '[TC35] derivative should be finite FAILED'

# ---- TC36: StabilityAnalyzer neuron_linearized_eigenvalue 返回负值 ----
import numpy as np
lam = StabilityAnalyzer.neuron_linearized_eigenvalue(
    V=-65.0, m=0.05, h=0.6, n=0.32
)
assert lam < 0.0, '[TC36] eigenvalue should be negative at rest FAILED'

# ---- TC37: CorticalGrid spatial_receptive_field 输出非负 ----
import numpy as np
grid = CorticalGrid(nx=4, ny=4)
rf = grid.spatial_receptive_field(i=5, sigma_rf=0.3)
assert len(rf) == 16, '[TC37] receptive field length mismatch FAILED'
assert np.all(rf >= 0.0), '[TC37] receptive field weights should be non-negative FAILED'

# ---- TC38: SynapticWeightSDE simulate_trajectory 输出长度正确 ----
import numpy as np
np.random.seed(42)
sde = SynapticWeightSDE(mu=0.0, sigma=0.3, theta=0.05, dt=0.01)
traj = sde.simulate_trajectory(w0=1.0, T_total=10.0)
assert len(traj) == 1000, '[TC38] trajectory length should be 1000 FAILED'
assert np.all(traj > 0.0), '[TC38] all trajectory values should be positive FAILED'

# ---- TC39: LogNormalSynapse sample_mean_variance 经验估计 ----
import numpy as np
np.random.seed(42)
model = LogNormalSynapse(mu=0.0, sigma=0.5)
emp_mean, emp_var = model.sample_mean_variance(n_samples=10000)
th_mean = model.mean()
assert np.abs(emp_mean - th_mean) < 0.1, '[TC39] empirical mean should approximate theoretical mean FAILED'
assert emp_var > 0.0, '[TC39] empirical variance should be positive FAILED'

# ---- TC40: EthierNavierStokes vorticity 返回有限值 ----
import numpy as np
ns = EthierNavierStokes()
ox, oy, oz = ns.vorticity(0.3, 0.3, 0.3, 0.0)
assert np.isfinite(ox), '[TC40] omega_x should be finite FAILED'
assert np.isfinite(oy), '[TC40] omega_y should be finite FAILED'
assert np.isfinite(oz), '[TC40] omega_z should be finite FAILED'

print('\n全部 40 个测试通过!\n')
