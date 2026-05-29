"""
main.py
海洋内波破碎与混合参数化综合模拟系统

统一入口，零参数可运行。

科学问题:
本项目研究海洋密度分层中内波的生成、传播、破碎及其导致的湍流混合过程。
通过融合15个基础算法的核心思想，构建了一个多尺度、多物理过程耦合的
博士级海洋内波参数化系统。

核心物理过程:
1. 内波非线性动力学 (Duffing型方程 + KdV方程)
2. 谱元数值求解 (间断Galerkin方法)
3. 小波时频分析 (Haar小波)
4. 三维空间索引 (Hilbert曲线)
5. 最优空间离散化 (CVT + Delaunay三角剖分)
6. 蒙特卡洛破碎模拟 (随机相位 + 乘法随机过程 + IFS分形)
7. 最优能量传播路径 (Dijkstra + 置换循环 + 射线追踪)
8. 湍流混合参数化 (Wishart采样 + 不动点迭代 + 波数对称化)
"""

import numpy as np
import sys
import os

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocean_physics import (
    density_profile, buoyancy_frequency, richardson_number,
    internal_wave_dispersion, group_velocity,
    turbulent_dissipation_rate, breaking_criterion,
    thope_internal_wave_spectrum
)
from internal_wave_dynamics import (
    NonlinearInternalWave, kdv_internal_wave
)
from spectral_discretization import DGInternalWaveSolver
from wavelet_analysis import (
    haar_1d_transform, detect_breaking_events,
    multi_scale_spectrum
)
from spatial_indexing import (
    HilbertCurve3D, ocean_volume_indexing
)
from mesh_generation import (
    CVT1D, triangulate_ocean_domain
)
from monte_carlo_breaking import (
    random_phase_superposition,
    monte_carlo_breaking_probability,
    energy_cascade_simulation,
    mixing_patch_ifs
)
from optimal_path import (
    build_energy_propagation_graph,
    dijkstra_shortest_path,
    reconstruct_path,
    permutation_cycle_analysis,
    ray_tracing_cycle
)
from turbulence_parameterization import (
    sample_reynolds_stress_tensor,
    mixing_efficiency_fixed_point,
    cobweb_iteration_analysis,
    symmetrize_wave_spectrum
)


def print_section(title):
    """打印章节分隔符"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_ocean_physics():
    """海洋物理参数计算"""
    print_section("模块 1: 海洋物理参数")
    
    z = np.linspace(-200, 0, 101)
    rho = density_profile(z)
    N = buoyancy_frequency(z)
    
    print(f"  深度范围: [{z.min():.1f}, {z.max():.1f}] m")
    print(f"  密度范围: [{rho.min():.3f}, {rho.max():.3f}] kg/m³")
    print(f"  浮力频率范围: [{N.min():.6f}, {N.max():.6f}] rad/s")
    
    # Richardson数
    dudz = 0.01 * np.sin(np.pi * z / 200)
    dvdz = 0.005 * np.cos(np.pi * z / 200)
    Ri = richardson_number(dudz, dvdz, N)
    print(f"  Richardson数范围: [{Ri.min():.3f}, {Ri.max():.1f}]")
    
    # 色散关系
    kh = np.linspace(0.001, 0.1, 50)
    m = 2.0 * np.pi / 200.0
    omega = internal_wave_dispersion(kh, m, 0.01)
    print(f"  内波频率范围: [{omega.min():.6f}, {omega.max():.6f}] rad/s")
    
    # 破碎判据
    is_breaking, steepness, crit = breaking_criterion(
        amplitude=20.0, wavelength=500.0, N=0.01, depth=200.0
    )
    print(f"  破碎判据: 波陡={steepness:.4f}, 临界波陡={crit:.4f}, 是否破碎={is_breaking}")
    
    return z, rho, N, Ri


def run_nonlinear_dynamics():
    """非线性内波动力学"""
    print_section("模块 2: 非线性内波动力学")
    
    # Duffing型内波方程
    wave = NonlinearInternalWave(
        alpha=1.0, beta=5.0, gamma=8.0, delta=0.02,
        omega=0.5, N=0.01, f=1.0e-4, depth=200.0
    )
    
    t, xi, xi_dot, E = wave.solve(t_span=(0, 50), dt=0.1)
    
    print(f"  求解时间区间: [0, 50] s")
    print(f"  位移范围: [{xi.min():.4f}, {xi.max():.4f}] m")
    print(f"  速度范围: [{xi_dot.min():.4f}, {xi_dot.max():.4f}] m/s")
    print(f"  能量范围: [{E.min():.6f}, {E.max():.6f}] J/kg")
    
    action = wave.compute_wave_action(t, xi, xi_dot)
    print(f"  波作用量范围: [{action.min():.6f}, {action.max():.6f}]")
    
    # KdV孤立波
    x_kdv, t_kdv, eta_kdv = kdv_internal_wave(
        xi0=2.0, c=1.0, alpha_kdv=0.1, beta_kdv=0.01,
        t_span=(0, 20), nx=128
    )
    print(f"  KdV孤立波: 空间域 [{x_kdv.min():.1f}, {x_kdv.max():.1f}] m")
    print(f"  KdV波高范围: [{eta_kdv.min():.4f}, {eta_kdv.max():.4f}] m")
    
    return t, xi, E


def run_spectral_solver():
    """DG谱元求解"""
    print_section("模块 3: DG谱元求解内波传播")
    
    solver = DGInternalWaveSolver(
        N=3, K=10, xmin=0.0, xmax=1000.0,
        wave_speed=1.0, N_buoyancy=0.01
    )
    
    t_hist, u_hist = solver.solve(t_final=10.0, dt=0.2)
    
    print(f"  多项式阶数: {solver.N}")
    print(f"  单元数量: {solver.K}")
    print(f"  时间步数: {len(t_hist)}")
    print(f"  解的范围: [{u_hist.min():.4f}, {u_hist.max():.4f}]")
    
    return t_hist, u_hist


def run_wavelet_analysis():
    """小波分析"""
    print_section("模块 4: Haar小波时频分析")
    
    # 生成测试信号 (内波+噪声)
    t = np.linspace(0, 100, 256)
    signal = 2.0 * np.sin(0.1 * t) + 0.5 * np.sin(0.5 * t) + \
             0.3 * np.random.randn(256)
    
    coeffs, energies = haar_1d_transform(signal)
    print(f"  信号长度: {len(signal)}")
    print(f"  小波分解层数: {len(coeffs) - 1}")
    print(f"  最高频能量: {energies[-1]:.4f}")
    
    breaking_indices, wavelet_energy = detect_breaking_events(signal)
    print(f"  检测到破碎事件数: {len(breaking_indices)}")
    
    scales, spectrum = multi_scale_spectrum(signal)
    print(f"  多尺度谱峰值尺度: {scales[np.argmax(spectrum)]}")
    
    return coeffs, energies


def run_spatial_indexing():
    """空间索引"""
    print_section("模块 5: 3D Hilbert空间索引")
    
    hc = HilbertCurve3D(r=3)
    points = hc.generate_curve()
    
    print(f"  Hilbert分辨率: 2^{hc.r} = {hc.N}")
    print(f"  总点数: {len(points)}")
    print(f"  坐标范围: [0, {hc.N-1}]³")
    
    # 测试双向转换
    test_h = 100
    x, y, z = hc.h_to_xyz(test_h)
    h_back = hc.xyz_to_h(x, y, z)
    print(f"  双向转换测试: h={test_h} -> ({x},{y},{z}) -> h={h_back}")
    
    lpi = hc.locality_preservation_index(n_samples=500)
    print(f"  局部性保持指数: {lpi:.4f}")
    
    return hc


def run_mesh_generation():
    """网格生成"""
    print_section("模块 6: CVT与三角网格")
    
    # CVT垂向节点
    cvt = CVT1D(n_generators=20, z_min=-200.0, z_max=0.0,
                density_type='thermocline')
    generators, energy_history = cvt.lloyd_iteration(
        n_samples=5000, max_iter=30, tol=1.0e-5
    )
    
    print(f"  CVT生成点数量: {len(generators)}")
    print(f"  生成点深度范围: [{generators.min():.2f}, {generators.max():.2f}] m")
    print(f"  CVT能量 (初始/最终): {energy_history[0]:.6f} / {energy_history[-1]:.6f}")
    print(f"  Lloyd迭代次数: {len(energy_history)}")
    
    # 2D三角剖分
    nodes, triangles = triangulate_ocean_domain(
        x_range=(0, 5000), y_range=(0, 5000), n_points=30
    )
    
    print(f"  三角网格节点数: {len(nodes)}")
    print(f"  三角形数量: {len(triangles)}")
    
    return generators, nodes, triangles


def run_monte_carlo():
    """蒙特卡洛模拟"""
    print_section("模块 7: 蒙特卡洛破碎模拟")
    
    # 随机相位叠加
    z = np.linspace(-200, 0, 101)
    u, shear, Ri = random_phase_superposition(
        n_modes=15, z=z, t=0.0, N=0.01
    )
    print(f"  随机相位叠加: 速度范围 [{u.min():.4f}, {u.max():.4f}] m/s")
    print(f"  剪切范围: [{shear.min():.4f}, {shear.max():.4f}] 1/s")
    print(f"  Richardson数 < 0.25 的点数: {np.sum(Ri < 0.25)}")
    
    # 破碎概率
    P_break, P_break_z, z_out = monte_carlo_breaking_probability(
        n_realizations=200, n_modes=15, n_depths=51, N=0.01
    )
    print(f"  破碎概率: {P_break:.4f}")
    print(f"  最大深度破碎概率: {P_break_z.max():.4f}")
    
    # 能量级联
    E_hist, breaking_events = energy_cascade_simulation(
        E0=1.0, n_steps=500, growth_factor=1.03, dissipation_factor=0.98
    )
    print(f"  能量级联: 最终能量={E_hist[-1]:.4f}")
    print(f"  破碎事件数: {len(breaking_events)}")
    
    # IFS分形混合斑块
    x_ifs, y_ifs, intensities = mixing_patch_ifs(n_points=2000)
    print(f"  IFS混合斑块: 点数={len(x_ifs)}")
    print(f"  混合强度范围: [{intensities.min():.4f}, {intensities.max():.4f}]")
    
    return P_break, E_hist


def run_optimal_path():
    """最优路径"""
    print_section("模块 8: 最优能量传播路径")
    
    # 构建能量传播图
    depths = np.linspace(-200, 0, 10)
    horiz = np.linspace(0, 5000, 10)
    N_prof = 0.01 * np.ones_like(depths)
    
    graph, node_coords = build_energy_propagation_graph(
        depths, horiz, N_prof
    )
    
    print(f"  图节点数: {len(node_coords)}")
    print(f"  图边密度: {np.sum(graph < np.inf) / (len(graph)**2):.4f}")
    
    # Dijkstra最短路径
    source = 0
    target = len(node_coords) - 1
    distances, previous = dijkstra_shortest_path(graph, source)
    path = reconstruct_path(previous, target)
    
    print(f"  最短路径长度: {distances[target]:.2f} s")
    print(f"  路径节点数: {len(path)}")
    
    # 置换循环分析
    cycles, cycle_lengths, success_rate = permutation_cycle_analysis(
        n_lockers=50, n_tries=25
    )
    print(f"  模态循环数: {len(cycles)}")
    print(f"  平均循环长度: {np.mean(cycle_lengths):.2f}")
    print(f"  能量传递成功率: {success_rate:.4f}")
    
    # 射线追踪
    z_ray = np.linspace(-200, 0, 101)
    N_ray = 0.01 * (1.0 + 0.5 * np.exp(-(z_ray + 100)**2 / 2000.0))
    x_path, z_path, theta_path = ray_tracing_cycle(
        wave_frequency=0.005, N_profile=N_ray, z=z_ray,
        theta0=np.pi/6, max_steps=200
    )
    print(f"  射线追踪: 最终水平位移={x_path[-1]:.2f} m")
    print(f"  最终深度={z_path[-1]:.2f} m")
    
    return path, x_path, z_path


def run_turbulence():
    """湍流参数化"""
    print_section("模块 9: 湍流混合参数化")
    
    # Wishart采样雷诺应力
    tau = sample_reynolds_stress_tensor(
        shear_magnitude=0.01, buoyancy_flux=1.0e-7, m=3, df=10
    )
    print(f"  雷诺应力张量:\n{tau}")
    print(f"  应力迹: {np.trace(tau):.4f} Pa")
    
    # 不动点混合效率
    Ri_values = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    results = cobweb_iteration_analysis(Ri_values)
    
    for Ri in Ri_values:
        r = results[Ri]
        print(f"  Ri={Ri:.1f}: Γ={r['gamma']:.4f}, "
              f"收敛={r['converged']}, 迭代={r['n_iter']}")
    
    # 波数谱对称化
    kx = np.linspace(-0.1, 0.1, 16)
    kz = np.linspace(-0.1, 0.1, 16)
    KX, KZ = np.meshgrid(kx, kz)
    E_spec = np.exp(-(KX**2 + KZ**2) / 0.001)
    
    E_sym = symmetrize_wave_spectrum(E_spec)
    
    # 对称性检验
    asymmetry = np.max(np.abs(E_sym - E_sym[::-1, :]))
    print(f"  能量谱对称化后非对称性: {asymmetry:.2e}")
    print(f"  能量谱总能量: {np.sum(E_sym):.4f}")
    
    return tau, results


def run_summary():
    """运行完整模拟并输出摘要"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "   海洋内波破碎与混合参数化综合模拟系统".center(60) + "#")
    print("#" + "   PROJECT_57".center(60) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\n科学问题: 海洋密度分层中内波的生成、传播、破碎及湍流混合")
    print("算法融合: 15个种子项目的核心算法")
    print("输出语言: Python 3")
    print("运行模式: 零参数自动运行")
    
    # 运行所有模块
    z, rho, N, Ri = run_ocean_physics()
    t, xi, E = run_nonlinear_dynamics()
    t_hist, u_hist = run_spectral_solver()
    coeffs, energies = run_wavelet_analysis()
    hc = run_spatial_indexing()
    generators, nodes, triangles = run_mesh_generation()
    P_break, E_hist = run_monte_carlo()
    path, x_path, z_path = run_optimal_path()
    tau, results = run_turbulence()
    
    # 综合摘要
    print_section("综合结果摘要")
    
    print("  [物理参数]")
    print(f"    浮力频率均值: {np.mean(N):.6f} rad/s")
    print(f"    Richardson数<0.25占比: {np.mean(Ri < 0.25)*100:.2f}%")
    
    print("  [非线性动力学]")
    print(f"    内波最大位移: {np.max(np.abs(xi)):.2f} m")
    print(f"    能量衰减速率: {(E[-1] - E[0]) / (t[-1] - t[0]):.6f} J/(kg·s)")
    
    print("  [数值求解]")
    print(f"    DG求解器最终能量: {np.sum(u_hist[-1]**2):.4f}")
    
    print("  [小波分析]")
    print(f"    小波分解总能量: {np.sum(energies):.4f}")
    
    print("  [空间索引]")
    print(f"    Hilbert曲线分辨率: {hc.N}³ = {hc.N**3} 点")
    
    print("  [网格生成]")
    print(f"    CVT节点数: {len(generators)}")
    print(f"    三角形数: {len(triangles)}")
    
    print("  [蒙特卡洛]")
    print(f"    破碎概率: {P_break:.4f}")
    print(f"    能量级联破碎事件: {len(E_hist[E_hist > 2.0])}")
    
    print("  [最优路径]")
    print(f"    最短路径节点数: {len(path)}")
    print(f"    射线总位移: {np.sqrt(x_path[-1]**2 + z_path[-1]**2):.2f} m")
    
    print("  [湍流参数化]")
    print(f"    雷诺应力最大值: {np.max(np.abs(tau)):.4f} Pa")
    print(f"    混合效率 (Ri=0.1): {results[0.1]['gamma']:.4f}")
    
    print("\n" + "#" * 70)
    print("  模拟完成。所有模块运行正常，无报错。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    # 设置随机种子保证可复现性
    np.random.seed(57)
    
    try:
        run_summary()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
