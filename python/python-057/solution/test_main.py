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

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: density_profile returns physically reasonable values ----
z_test = np.linspace(-200, 0, 11)
rho_test = density_profile(z_test)
assert np.all(rho_test >= 1020.0) and np.all(rho_test <= 1030.0), '[TC01] density_profile out of range FAILED'

# ---- TC02: buoyancy_frequency is positive and stable ----
N_test = buoyancy_frequency(z_test)
assert np.all(N_test > 0), '[TC02] buoyancy_frequency non-positive FAILED'

# ---- TC03: richardson_number handles zero shear safely ----
Ri_test = richardson_number(np.zeros(5), np.zeros(5), 0.01)
assert np.all(np.isfinite(Ri_test)), '[TC03] richardson_number zero shear non-finite FAILED'
assert np.all(Ri_test >= 1e5), '[TC03] richardson_number zero shear value FAILED'

# ---- TC04: internal_wave_dispersion respects bounds [f, N] ----
kh_test = np.linspace(0.001, 0.1, 10)
omega_test = internal_wave_dispersion(kh_test, 2.0 * np.pi / 200.0, 0.01, 1.0e-4)
assert np.all(omega_test >= 1.0e-4 * 0.99), '[TC04] internal_wave_dispersion below f FAILED'
assert np.all(omega_test <= 0.01 * 1.01), '[TC04] internal_wave_dispersion above N FAILED'

# ---- TC05: group_velocity returns finite values ----
cgx_test, cgz_test = group_velocity(kh_test, 2.0 * np.pi / 200.0, 0.01, 1.0e-4)
assert np.all(np.isfinite(cgx_test)), '[TC05] group_velocity cgx non-finite FAILED'
assert np.all(np.isfinite(cgz_test)), '[TC05] group_velocity cgz non-finite FAILED'

# ---- TC06: breaking_criterion detects large amplitude wave ----
is_breaking, steepness, crit = breaking_criterion(amplitude=50.0, wavelength=100.0, N=0.01, depth=200.0)
assert is_breaking == True, '[TC06] breaking_criterion large amp not breaking FAILED'
assert steepness > crit, '[TC06] breaking_criterion steepness <= crit FAILED'

# ---- TC07: NonlinearInternalWave solve produces non-negative energy ----
np.random.seed(42)
wave = NonlinearInternalWave(alpha=1.0, beta=5.0, gamma=8.0, delta=0.02, omega=0.5, N=0.01, f=1.0e-4, depth=200.0)
t_w, xi_w, xi_dot_w, E_w = wave.solve(t_span=(0, 10), dt=0.1)
assert len(t_w) == len(E_w), '[TC07] NonlinearInternalWave time-energy length mismatch FAILED'
assert np.all(E_w >= 0), '[TC07] NonlinearInternalWave energy negative FAILED'
assert np.all(np.isfinite(xi_w)), '[TC07] NonlinearInternalWave xi non-finite FAILED'

# ---- TC08: NonlinearInternalWave wave action is non-negative ----
action = wave.compute_wave_action(t_w, xi_w, xi_dot_w)
assert np.all(action >= 0), '[TC08] wave_action negative FAILED'
assert np.all(np.isfinite(action)), '[TC08] wave_action non-finite FAILED'

# ---- TC09: kdv_internal_wave output shapes consistent ----
x_kdv, t_kdv, eta_kdv = kdv_internal_wave(xi0=2.0, c=1.0, alpha_kdv=0.1, beta_kdv=0.01, t_span=(0, 5), nx=64)
assert eta_kdv.shape[0] == len(t_kdv), '[TC09] kdv eta time dim FAILED'
assert eta_kdv.shape[1] == len(x_kdv), '[TC09] kdv eta space dim FAILED'
assert np.all(np.isfinite(eta_kdv)), '[TC09] kdv eta non-finite FAILED'

# ---- TC10: DG solver produces bounded finite solution ----
solver = DGInternalWaveSolver(N=2, K=5, xmin=0.0, xmax=100.0, wave_speed=1.0, N_buoyancy=0.01)
t_hist, u_hist = solver.solve(t_final=2.0, dt=0.2)
assert np.all(np.isfinite(u_hist)), '[TC10] DG solver non-finite FAILED'
assert np.all(np.abs(u_hist) <= 10.0), '[TC10] DG solver out of bounds FAILED'

# ---- TC11: haar_1d_transform preserves energy (Parseval) ----
np.random.seed(42)
signal = np.sin(2.0 * np.pi * np.arange(64) / 16.0)
coeffs, energies = haar_1d_transform(signal)
time_energy = np.sum(signal[:64]**2)
wav_energy = np.sum(coeffs[0]**2) + np.sum(energies)
assert np.abs(time_energy - wav_energy) < 1e-10, '[TC11] haar energy conservation FAILED'
assert np.all(np.array(energies) >= 0), '[TC11] haar energies negative FAILED'

# ---- TC12: detect_breaking_events on constant signal returns no events ----
const_signal = np.ones(128)
indices, wenergy = detect_breaking_events(const_signal, threshold_factor=3.0)
assert len(indices) == 0, '[TC12] detect_breaking constant signal found events FAILED'

# ---- TC13: multi_scale_spectrum normalizes to unity ----
scales, spectrum = multi_scale_spectrum(signal)
assert np.abs(np.sum(spectrum) - 1.0) < 1e-12, '[TC13] multi_scale_spectrum normalization FAILED'
assert len(scales) == len(spectrum), '[TC13] multi_scale_spectrum length mismatch FAILED'

# ---- TC14: HilbertCurve3D h_to_xyz returns coordinates in valid range ----
hc = HilbertCurve3D(r=2)
for h_val in [0, 7, 15, 63]:
    x_h, y_h, z_h = hc.h_to_xyz(h_val)
    assert 0 <= x_h <= hc.N - 1, '[TC14] Hilbert x out of range FAILED'
    assert 0 <= y_h <= hc.N - 1, '[TC14] Hilbert y out of range FAILED'
    assert 0 <= z_h <= hc.N - 1, '[TC14] Hilbert z out of range FAILED'

# ---- TC15: HilbertCurve3D generate_curve covers all points in range ----
points = hc.generate_curve()
assert len(points) == 64, '[TC15] Hilbert curve point count FAILED'
assert np.all((points >= 0) & (points <= 3)), '[TC15] Hilbert curve coordinate range FAILED'

# ---- TC16: CVT1D generators stay within bounds ----
np.random.seed(42)
cvt = CVT1D(n_generators=5, z_min=-200.0, z_max=0.0, density_type='uniform')
gens, ehist = cvt.lloyd_iteration(n_samples=1000, max_iter=10, tol=1.0e-5)
assert np.all(gens >= -200.0) and np.all(gens <= 0.0), '[TC16] CVT generators out of bounds FAILED'

# ---- TC17: CVT1D energy history is non-negative ----
assert np.all(np.array(ehist) >= 0), '[TC17] CVT energy negative FAILED'

# ---- TC18: triangulate_ocean_domain produces valid triangles ----
nodes, triangles = triangulate_ocean_domain(x_range=(0, 1000), y_range=(0, 1000), n_points=16)
assert len(triangles) > 0, '[TC18] triangulate no triangles FAILED'
assert np.all(triangles >= 0) and np.all(triangles < len(nodes)), '[TC18] triangulate invalid indices FAILED'

# ---- TC19: random_phase_superposition is reproducible with fixed seed ----
np.random.seed(42)
u1, s1, Ri1 = random_phase_superposition(n_modes=5, z=np.linspace(-50, 0, 11), t=0.0, N=0.01)
np.random.seed(42)
u2, s2, Ri2 = random_phase_superposition(n_modes=5, z=np.linspace(-50, 0, 11), t=0.0, N=0.01)
assert np.allclose(u1, u2), '[TC19] random_phase not reproducible FAILED'
assert np.allclose(s1, s2), '[TC19] random_phase shear not reproducible FAILED'

# ---- TC20: monte_carlo_breaking_probability returns valid probability ----
np.random.seed(42)
P_break, P_break_z, z_out = monte_carlo_breaking_probability(n_realizations=50, n_modes=5, n_depths=21, N=0.01)
assert 0.0 <= P_break <= 1.0, '[TC20] breaking probability out of range FAILED'
assert np.all((P_break_z >= 0.0) & (P_break_z <= 1.0)), '[TC20] depth prob out of range FAILED'

# ---- TC21: energy_cascade_simulation energy stays positive ----
np.random.seed(42)
E_hist, breaking_events = energy_cascade_simulation(E0=1.0, n_steps=100, growth_factor=1.03, dissipation_factor=0.98)
assert np.all(E_hist > 0), '[TC21] energy_cascade non-positive energy FAILED'

# ---- TC22: mixing_patch_ifs output in unit range ----
np.random.seed(42)
x_ifs, y_ifs, intensities = mixing_patch_ifs(n_points=100, n_iterations=5)
assert np.all((x_ifs >= 0.0) & (x_ifs <= 1.0)), '[TC22] IFS x out of range FAILED'
assert np.all((y_ifs >= 0.0) & (y_ifs <= 1.0)), '[TC22] IFS y out of range FAILED'
assert np.all((intensities >= 0.0) & (intensities <= 1.0)), '[TC22] IFS intensity out of range FAILED'

# ---- TC23: dijkstra_shortest_path self distance is zero ----
graph = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
distances, previous = dijkstra_shortest_path(graph, 1)
assert distances[1] == 0.0, '[TC23] dijkstra self distance not zero FAILED'
assert distances[0] == 1.0, '[TC23] dijkstra shortest distance FAILED'

# ---- TC24: permutation_cycle_analysis cycle lengths sum to n ----
np.random.seed(42)
cycles, cycle_lengths, success_rate = permutation_cycle_analysis(n_lockers=20, n_tries=10)
assert np.sum(cycle_lengths) == 20, '[TC24] permutation cycle sum FAILED'
assert 0.0 <= success_rate <= 1.0, '[TC24] permutation success rate FAILED'

# ---- TC25: ray_tracing_cycle output arrays have correct length ----
z_ray = np.linspace(-100, 0, 51)
N_ray = 0.01 * np.ones_like(z_ray)
x_path, z_path, theta_path = ray_tracing_cycle(wave_frequency=0.005, N_profile=N_ray, z=z_ray, theta0=np.pi/6, max_steps=50)
assert len(x_path) == 50, '[TC25] ray tracing x length FAILED'
assert len(z_path) == 50, '[TC25] ray tracing z length FAILED'
assert len(theta_path) == 50, '[TC25] ray tracing theta length FAILED'
assert np.all(np.isfinite(x_path)), '[TC25] ray tracing x non-finite FAILED'

# ---- TC26: mixing_efficiency_fixed_point converges for small Ri ----
gamma, history, converged = mixing_efficiency_fixed_point(Ri=0.1, gamma_max=0.2, alpha=5.0, max_iter=100, tol=1.0e-8)
assert converged, '[TC26] mixing_efficiency not converged FAILED'
assert 0.0 <= gamma <= 0.2, '[TC26] mixing_efficiency gamma out of range FAILED'

# ---- TC27: symmetrize_wave_spectrum produces four-fold symmetry ----
kx = np.linspace(-0.1, 0.1, 8)
kz = np.linspace(-0.1, 0.1, 8)
KX, KZ = np.meshgrid(kx, kz)
E_spec = np.exp(-(KX**2 + KZ**2) / 0.001)
E_sym = symmetrize_wave_spectrum(E_spec)
assert np.allclose(E_sym, E_sym[::-1, :]), '[TC27] spectrum symmetry x FAILED'
assert np.allclose(E_sym, E_sym[:, ::-1]), '[TC27] spectrum symmetry z FAILED'

# ---- TC28: ocean_volume_indexing returns positive scales ----
hc_idx, d_scale, lat_scale, lon_scale = ocean_volume_indexing(depth_levels=10, lat_levels=10, lon_levels=10, r=3)
assert d_scale > 0 and lat_scale > 0 and lon_scale > 0, '[TC28] ocean_volume_indexing scales non-positive FAILED'

# ---- TC29: turbulent_dissipation_rate returns non-negative epsilon and bounded Kz ----
Ri_t = np.array([0.1, 0.3, 0.5])
shear_t = np.array([0.01, 0.005, 0.002])
eps, Kz = turbulent_dissipation_rate(Ri_t, shear_t)
assert np.all(eps >= 0), '[TC29] turbulent_dissipation epsilon negative FAILED'
assert np.all(Kz >= 1.0e-7), '[TC29] Kz below lower bound FAILED'
assert np.all(Kz <= 1.0e-1), '[TC29] Kz above upper bound FAILED'

# ---- TC30: thope_internal_wave_spectrum is non-negative ----
kh_test2 = np.linspace(0.001, 0.1, 10)
spec_test = thope_internal_wave_spectrum(kh_test2, N=0.01, f=1.0e-4, E0=6.3e-5)
assert np.all(spec_test >= 0), '[TC30] spectrum negative FAILED'

print('\n全部 30 个测试通过!\n')
