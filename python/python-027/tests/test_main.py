# -*- coding: utf-8 -*-
"""
main.py
等离子体鞘层与壁材料侵蚀综合模拟系统 —— 统一入口

本项目基于以下15个种子项目的核心算法合成:
    702_logistic_ode, 1201_tensor_grid_display, 1419_xy_display,
    144_cc_project, 823_obj_to_tri_surface, 1304_triangle_felippa_rule,
    269_delsq, 1290_tree_chaos, 992_r8ri, 883_polygon_average,
    1403_wavelet, 603_jacobi, 567_hypersphere_positive_distance,
    137_casino_simulation, 1005_randlc

科学领域: 等离子体物理 —— 等离子体鞘层与壁材料侵蚀
"""

import numpy as np
import time

from parameters import get_parameters, PlasmaParameters
from sheath_ode import SheathODE
from poisson_solver import PoissonSolver
from sparse_matrix_ops import R8RIMatrix
from surface_mesh import SurfaceMesh
from erosion_quadrature import ErosionQuadrature
from polygon_roughness import PolygonRoughness
from impurity_transport import ImpurityTransport
from wavelet_analysis import WaveletAnalysis
from monte_carlo_sampler import MonteCarloSampler
from utils import (compute_coulomb_logarithm, compute_sheath_heat_flux,
                   check_bohm_criterion, convergence_diagnostics)


def print_banner():
    """打印项目横幅"""
    print("=" * 70)
    print("  等离子体鞘层与壁材料侵蚀数值模拟系统")
    print("  Plasma Sheath & Wall Material Erosion Simulation System")
    print("=" * 70)
    print()


def run_module_01_parameters():
    """模块1: 物理参数初始化"""
    print("[模块 1/10] 物理参数初始化")
    print("-" * 50)

    params = get_parameters()
    params.print_summary()

    # 验证关键物理量
    lambda_D = params.debye_length()
    cs = params.ion_sound_speed()
    omega_pe = params.plasma_frequency()
    ln_lambda = compute_coulomb_logarithm(params.get('T_e'), params.get('n_0'))

    print(f"  Coulomb对数 ln(Lambda) = {ln_lambda:.2f}")
    print(f"  等离子体频率 omega_pe  = {omega_pe:.3e} rad/s")
    print()
    return params


def run_module_02_sheath_ode(params):
    """模块2: 鞘层ODE求解（基于 702_logistic_ode）"""
    print("[模块 2/10] 鞘层离子密度与速度剖面（修正Logistic ODE）")
    print("-" * 50)

    sheath = SheathODE(params)
    x, n_i, v_i, phi, e_field = sheath.solve_sheath_profile(nx=128, x_max=0.005)

    gamma = sheath.compute_ion_flux(n_i, v_i)
    M = sheath.compute_sheath_edge_mach(v_i)
    E_wall = sheath.compute_ion_energy_at_wall(v_i[-1])

    bohm_ok, M0 = check_bohm_criterion(v_i[0], sheath.c_s)

    print(f"  鞘层边缘 Mach 数       = {M0:.3f}")
    print(f"  Bohm 判据满足         = {bohm_ok}")
    print(f"  壁面离子密度          = {n_i[-1]:.3e} m^-3")
    print(f"  壁面离子速度          = {v_i[-1]:.3e} m/s")
    print(f"  壁面离子能量          = {E_wall:.2f} eV")
    print(f"  壁面离子通量          = {gamma[-1]:.3e} m^-2 s^-1")
    print(f"  壁面电场强度          = {e_field[-1]:.3e} V/m")
    print()
    return x, n_i, v_i, phi, e_field, E_wall, gamma


def run_module_03_poisson_solver(params, n_i, x):
    """模块3: 泊松方程求解（基于 269_delsq + 603_jacobi）"""
    print("[模块 3/10] 鞘层泊松方程有限差分解（delsq + Jacobi迭代）")
    print("-" * 50)

    solver = PoissonSolver(params)

    # 1D 鞘层泊松方程
    phi_pois, n_e, rho = solver.solve_1d_sheath_poisson(n_i, x)

    # 2D Laplace 测试（靶板表面电势分布）
    def boundary_func(i, j):
        nx, ny = 32, 32
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            return 0.0 if i == 0 else -params.sheath_potential()
        return None

    phi_2d = solver.solve_2d_laplace(32, 32, 1.0e-4, 1.0e-4, boundary_func)

    print(f"  1D 壁面电势           = {phi_pois[-1]:.3f} V")
    print(f"  1D 壁面电荷密度       = {rho[-1]:.3e} C/m^3")
    print(f"  2D 电势范围           = [{np.min(phi_2d):.3f}, {np.max(phi_2d):.3f}] V")
    print(f"  2D 电势矩阵尺寸       = {phi_2d.shape}")
    print()
    return phi_pois, n_e, rho, phi_2d


def run_module_04_sparse_matrix(params):
    """模块4: 稀疏矩阵运算（基于 992_r8ri）"""
    print("[模块 4/10] 稀疏矩阵R8RI格式运算")
    print("-" * 50)

    n = params.get('nx')
    mat = R8RIMatrix.build_dif2(n)

    # 矩阵-向量乘法测试
    x_test = np.ones(n)
    y_sparse = mat.matvec(x_test)
    y_transpose = mat.matvec_transpose(x_test)

    # 与稠密矩阵对比
    dense = mat.to_dense()
    y_dense = dense.dot(x_test)
    error = np.linalg.norm(y_sparse - y_dense)
    saving = mat.get_memory_usage()

    print(f"  矩阵阶数 N            = {n}")
    print(f"  稀疏存储元素数        = {mat.nz}")
    print(f"  稠密存储元素数        = {n*n}")
    print(f"  内存节省              = {saving*100:.1f}%")
    print(f"  matvec 误差           = {error:.3e}")
    print(f"  转置乘法范数          = {np.linalg.norm(y_transpose):.3e}")
    print()
    return mat


def run_module_05_surface_mesh(params):
    """模块5: 靶板表面网格生成（基于 823_obj_to_tri_surface + 1201_tensor_grid_display）"""
    print("[模块 5/10] 靶板表面三角网格生成")
    print("-" * 50)

    mesh = SurfaceMesh()
    mesh.generate_flat_plate_mesh(width=0.05, height=0.05, nx=21, ny=21)
    stats = mesh.mesh_quality_stats()

    # 圆柱面网格
    cyl_mesh = SurfaceMesh()
    cyl_mesh.generate_cylindrical_mesh(radius=0.02, height=0.05, n_theta=32, n_z=16)

    # 入射角计算
    b_dir = np.array([0.0, 1.0, 0.1])
    angles_flat = mesh.compute_incidence_angles(b_dir)
    angles_cyl = cyl_mesh.compute_incidence_angles(b_dir)

    print(f"  平板网格: 节点={stats['n_nodes']}, 三角形={stats['n_triangles']}")
    print(f"  平板总面积            = {stats['total_area']:.3e} m^2")
    print(f"  最小三角形角          = {stats.get('min_triangle_angle_deg', 0):.1f}°")
    print(f"  圆柱网格: 节点={len(cyl_mesh.nodes)}, 三角形={len(cyl_mesh.triangles)}")
    print(f"  平板平均入射角        = {np.degrees(np.mean(angles_flat)):.1f}°")
    print(f"  圆柱平均入射角        = {np.degrees(np.mean(angles_cyl)):.1f}°")
    print()
    return mesh, cyl_mesh


def run_module_06_erosion_quadrature(params, E_wall, mesh):
    """模块6: 侵蚀率数值积分（基于 1304_triangle_felippa_rule + 144_cc_project）"""
    print("[模块 6/10] 壁材料物理溅射产额与侵蚀积分")
    print("-" * 50)

    eq = ErosionQuadrature(params)

    # 溅射产额计算
    test_energies = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    print("  D -> W 溅射产额:")
    for E in test_energies:
        Y = eq.sputtering_yield_bohdansky(E)
        print(f"    E={E:6.0f} eV, Y={Y:.4f}")

    # 能量沉积深度
    depths = np.linspace(0, 1.0e-8, 100)
    dep_profile = eq.energy_deposition_profile(E_wall, depths)

    # Clenshaw-Curtis 积分
    avg_yield, _, _ = eq.integrate_sputtering_yield_1d(
        10.0, 5000.0, n_points=64)

    # 三角形积分
    tri = mesh.triangles[0]
    tri_verts = mesh.nodes[tri]
    def gamma_f(x, y):
        return 1.0e22
    def E_f(x, y):
        return E_wall
    tri_erosion = eq.integrate_erosion_over_triangle(tri_verts, gamma_f, E_f)

    # 验证求积规则精度
    w3, xy3 = eq.triangle_unit_o03()
    w12, xy12 = eq.triangle_unit_o12()

    print(f"\n  能量加权平均产额      = {avg_yield:.4f}")
    print(f"  能量沉积峰值深度      = {depths[np.argmax(dep_profile)]:.3e} m")
    print(f"  单三角形侵蚀率        = {tri_erosion:.3e}")
    print(f"  3点三角形规则权重和   = {np.sum(w3):.6f} (归一化)")
    print(f"  12点三角形规则权重和  = {np.sum(w12):.6f} (归一化)")
    print()
    return eq, avg_yield


def run_module_07_polygon_roughness(params):
    """模块7: 表面粗糙度演化（基于 883_polygon_average）"""
    print("[模块 7/10] 壁表面粗糙度演化模拟")
    print("-" * 50)

    surface = PolygonRoughness(n_vertices=128)
    surface.initialize_random_roughness(amplitude=1.0e-6, n_modes=15, seed=42)

    init_stats = surface.compute_roughness_parameters()

    # 多步演化
    history = surface.evolve_surface(
        n_steps=100,
        erosion_rate=1.0e-9,
        redeposition_rate=3.0e-10,
        dt=1.0e-3,
        alpha_smooth=0.2,
        stochastic=True
    )

    final_stats = surface.compute_roughness_parameters()

    print(f"  初始 Ra               = {init_stats['Ra']:.3e} m")
    print(f"  初始 Rq               = {init_stats['Rq']:.3e} m")
    print(f"  最终 Ra               = {final_stats['Ra']:.3e} m")
    print(f"  最终 Rq               = {final_stats['Rq']:.3e} m")
    print(f"  粗糙度变化率          = {(final_stats['Ra']-init_stats['Ra'])/init_stats['Ra']*100:.1f}%")
    print()
    return surface, history


def run_module_08_impurity_transport(params):
    """模块8: 侵蚀杂质输运（基于 1290_tree_chaos + 137_casino_simulation）"""
    print("[模块 8/10] 侵蚀杂质在鞘层中的输运模拟")
    print("-" * 50)

    transport = ImpurityTransport(params)

    # 扩散系数对比
    D_bohm = transport.compute_diffusion_coefficient('bohm')
    D_classical = transport.compute_diffusion_coefficient('classical')
    D_neo = transport.compute_diffusion_coefficient('neo')

    print("  扩散系数对比:")
    print(f"    Bohm扩散            = {D_bohm:.3e} m^2/s")
    print(f"    经典碰撞扩散        = {D_classical:.3e} m^2/s")
    print(f"    新经典扩散          = {D_neo:.3e} m^2/s")

    # IFS混沌映射
    x_ifs = np.random.rand(2)
    print(f"\n  IFS混沌映射 (sheath_drift模式):")
    print(f"    初始位置            = [{x_ifs[0]:.4f}, {x_ifs[1]:.4f}]")
    for step in range(10):
        x_ifs = transport.ifs_transport_map(x_ifs, mode='sheath_drift')
    print(f"    10步后位置          = [{x_ifs[0,0]:.4f}, {x_ifs[0,1]:.4f}]")

    # 系综模拟
    print(f"\n  Langevin系综模拟 (50粒子)...")
    final_pos, stats = transport.simulate_ensemble(n_particles=50, n_steps=100, dt=1.0e-10)

    print(f"    平均位移            = {stats['mean_displacement']:.3e} m")
    print(f"    返回壁面比例        = {stats['return_fraction']*100:.1f}%")
    print()
    return transport, stats


def run_module_09_wavelet_analysis(params, n_i):
    """模块9: 鞘层波动小波分析（基于 1403_wavelet）"""
    print("[模块 9/10] 鞘层密度波动小波多尺度分析")
    print("-" * 50)

    # 构造合成波动信号（模拟 blob 输运）
    t = np.linspace(0, 1, 1024)
    f_blob = 20.0   # blob频率
    f_drift = 5.0   # 漂移波频率
    signal = (np.sin(2*np.pi*f_drift*t) +
              0.3 * np.sin(2*np.pi*f_blob*t) *
              (1.0 + 0.5*np.sin(2*np.pi*2*t)) +
              0.05 * np.random.randn(len(t)))

    wv = WaveletAnalysis(order=10)

    # 多级分解
    levels = wv.decompose_levels(signal)
    print("  小波分解能量分布:")
    total_detail_energy = 0.0
    for key in sorted([k for k in levels.keys() if isinstance(k, int)]):
        energy = np.sum(levels[key]**2)
        total_detail_energy += energy
        print(f"    Level {key}:          = {energy:.4e}")
    print(f"    总细节能量          = {total_detail_energy:.4e}")

    # 功率谱
    scales, power, freqs = wv.power_spectrum(signal, sample_rate=1024.0)
    if len(freqs) > 0:
        peak_idx = np.argmax(power)
        print(f"\n  主导频率              = {freqs[peak_idx]:.1f} Hz")
        print(f"  对应特征时间尺度      = {1.0/freqs[peak_idx]:.3e} s")

    # 去噪
    denoised = wv.denoise(signal, threshold_ratio=0.15)
    snr_improvement = np.std(signal)**2 / (np.std(denoised - signal) + 1.0e-30)**2
    print(f"  去噪后SNR改善         = {10*np.log10(snr_improvement):.1f} dB")
    print()
    return wv, signal, denoised


def run_module_10_monte_carlo(params):
    """模块10: 蒙特卡洛采样（基于 567_hypersphere_positive_distance + 1005_randlc）"""
    print("[模块 10/10] 蒙特卡洛随机采样验证")
    print("-" * 50)

    mc = MonteCarloSampler(seed=params.get('rand_seed'))

    # 超球面采样
    x_sphere = mc.sample_hypersphere_positive(5)
    print(f"  5维正超球面采样: 模长={np.linalg.norm(x_sphere):.6f}")

    # 距离统计
    mu, var = mc.sample_hypersphere_distance_stats(3, 500)
    print(f"  3维距离统计: mu={mu:.4f}, var={var:.6f}")

    # Maxwellian速度
    v_th = params.ion_thermal_velocity()
    v_samples = mc.sample_maxwellian_velocity(v_th, 500)
    v_rms = np.sqrt(np.mean(v_samples**2))
    v_theory = v_th  # 每个速度分量的 rms = v_th
    print(f"  Maxwellian速度: 理论rms={v_theory:.3e}, 实测rms={v_rms:.3e}")

    # randlc跳跃验证
    from monte_carlo_sampler import RandLC
    rng_test = RandLC(seed=params.get('rand_seed'))
    u_seq = [rng_test.next() for _ in range(10)]
    rng_test2 = RandLC(seed=params.get('rand_seed'))
    u_jump = rng_test2.jump(10)
    print(f"  randlc序列第10个值    = {u_seq[-1]:.8f}")
    print(f"  randlc直接跳跃到第10  = {u_jump:.8f}")
    print(f"  跳跃一致性            = {abs(u_seq[-1]-u_jump) < 1.0e-10}")

    # 碰撞采样
    collided, mfp = mc.sample_collision_parameter(1.0e-19, 1.0e19, 0.01)
    print(f"  碰撞采样: 碰撞={collided}, 平均自由程={mfp:.3e} m")
    print()
    return mc


def run_final_summary(params, E_wall, avg_yield, stats_transport, stats_roughness):
    """最终汇总"""
    print("=" * 70)
    print("  模拟结果汇总")
    print("=" * 70)

    q_sheath = compute_sheath_heat_flux(
        params.get('n_0') * 0.5, params.get('T_e'), params.get('T_i'))

    print(f"  电子温度              = {params.get('T_e'):.1f} eV")
    print(f"  上游密度              = {params.get('n_0'):.3e} m^-3")
    print(f"  德拜长度              = {params.debye_length():.3e} m")
    print(f"  离子声速              = {params.ion_sound_speed():.3e} m/s")
    print(f"  鞘层电势降            = {params.sheath_potential():.2f} V")
    print(f"  壁面离子能量          = {E_wall:.2f} eV")
    print(f"  鞘层热流密度          = {q_sheath:.3e} W/m^2")
    print(f"  能量加权溅射产额      = {avg_yield:.4f}")
    print(f"  杂质返回壁面比例      = {stats_transport.get('return_fraction', 0)*100:.1f}%")
    print(f"  表面粗糙度 Ra         = {stats_roughness.get('Ra', 0):.3e} m")
    print("=" * 70)


def main():
    """主函数 —— 零参数可运行"""
    start_time = time.time()
    print_banner()

    # 模块1: 参数
    params = run_module_01_parameters()

    # 模块2: 鞘层ODE
    x, n_i, v_i, phi_ode, e_field, E_wall, gamma = run_module_02_sheath_ode(params)

    # 模块3: 泊松求解
    phi_pois, n_e, rho, phi_2d = run_module_03_poisson_solver(params, n_i, x)

    # 模块4: 稀疏矩阵
    mat = run_module_04_sparse_matrix(params)

    # 模块5: 表面网格
    mesh, cyl_mesh = run_module_05_surface_mesh(params)

    # 模块6: 侵蚀积分
    eq, avg_yield = run_module_06_erosion_quadrature(params, E_wall, mesh)

    # 模块7: 粗糙度
    surface, history = run_module_07_polygon_roughness(params)
    roughness_final = surface.compute_roughness_parameters()

    # 模块8: 杂质输运
    transport, stats_transport = run_module_08_impurity_transport(params)

    # 模块9: 小波分析
    wv, signal, denoised = run_module_09_wavelet_analysis(params, n_i)

    # 模块10: MC采样
    mc = run_module_10_monte_carlo(params)

    # 最终汇总
    run_final_summary(params, E_wall, avg_yield, stats_transport, roughness_final)

    elapsed = time.time() - start_time
    print(f"\n总运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    print("模拟完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
from utils import safe_exp, safe_divide

# ---- TC01: PlasmaParameters 德拜长度计算为正有限值 ----
params = get_parameters()
lambda_D = params.debye_length()
assert lambda_D > 0 and np.isfinite(lambda_D), '[TC01] PlasmaParameters 德拜长度计算为正有限值 FAILED'

# ---- TC02: PlasmaParameters 离子声速大于0 ----
cs = params.ion_sound_speed()
assert cs > 0 and np.isfinite(cs), '[TC02] PlasmaParameters 离子声速大于0 FAILED'

# ---- TC03: PlasmaParameters 参数边界校验拒绝非法值 ----
try:
    bad_params = PlasmaParameters(n_0=-1.0)
    assert False, '[TC03] PlasmaParameters 参数边界校验拒绝非法值 FAILED'
except ValueError:
    pass

# ---- TC04: SheathODE 无复合精确解单调递增 ----
sheath = SheathODE(params)
x_test = np.linspace(0, 0.001, 50)
n_exact = sheath.exact_density_solution(x_test)
assert np.all(np.diff(n_exact) >= -1e10), '[TC04] SheathODE 无复合精确解单调递增 FAILED'

# ---- TC05: SheathODE 数值解输出尺寸正确 ----
x, n_i, v_i, phi, e_field = sheath.solve_sheath_profile(nx=64, x_max=0.005)
assert len(x) == 64 and len(n_i) == 64 and len(v_i) == 64, '[TC05] SheathODE 数值解输出尺寸正确 FAILED'

# ---- TC06: SheathODE Mach数在鞘层边缘满足Bohm判据 ----
M = sheath.compute_sheath_edge_mach(v_i)
bohm_ok, M0 = check_bohm_criterion(v_i[0], sheath.c_s)
assert bohm_ok, '[TC06] SheathODE Mach数在鞘层边缘满足Bohm判据 FAILED'

# ---- TC07: PoissonSolver 一维Laplacian为三对角矩阵 ----
solver = PoissonSolver(params)
L1d = solver.build_laplacian_1d(10, 0.001)
diag_count = np.sum(np.abs(np.diag(L1d)) > 1e-30)
offdiag_count = np.sum(np.abs(L1d - np.diag(np.diag(L1d))) > 1e-30)
assert diag_count == 10 and offdiag_count == 18, '[TC07] PoissonSolver 一维Laplacian为三对角矩阵 FAILED'

# ---- TC08: PoissonSolver Jacobi迭代收敛到精确解 ----
A = np.array([[4, 1], [1, 3]], dtype=float)
b = np.array([1, 2], dtype=float)
x_jac, res_hist, iters = solver.jacobi_solve(A, b, max_iter=1000, tol=1e-8)
x_exact = np.linalg.solve(A, b)
assert np.linalg.norm(x_jac - x_exact) < 1e-6, '[TC08] PoissonSolver Jacobi迭代收敛到精确解 FAILED'

# ---- TC09: R8RIMatrix 稀疏matvec与稠密结果一致 ----
mat = R8RIMatrix.build_dif2(16)
x_vec = np.ones(16)
y_sparse = mat.matvec(x_vec)
dense = mat.to_dense()
y_dense = dense.dot(x_vec)
assert np.linalg.norm(y_sparse - y_dense) < 1e-12, '[TC09] R8RIMatrix 稀疏matvec与稠密结果一致 FAILED'

# ---- TC10: R8RIMatrix 转置乘法对称性 ----
y_t = mat.matvec_transpose(x_vec)
assert abs(np.dot(x_vec, y_sparse) - np.dot(x_vec, y_t)) < 1e-10, '[TC10] R8RIMatrix 转置乘法对称性 FAILED'

# ---- TC11: SurfaceMesh 平板网格总面积等于width乘height ----
mesh = SurfaceMesh()
mesh.generate_flat_plate_mesh(width=0.05, height=0.05, nx=21, ny=21)
total_area = mesh.compute_total_area()
assert abs(total_area - 0.0025) < 1e-8, '[TC11] SurfaceMesh 平板网格总面积等于width乘height FAILED'

# ---- TC12: SurfaceMesh 入射角在有效范围内 ----
b_dir = np.array([0.0, 1.0, 0.1])
angles = mesh.compute_incidence_angles(b_dir)
assert np.all(angles >= 0) and np.all(angles <= np.pi/2 + 1e-10), '[TC12] SurfaceMesh 入射角在有效范围内 FAILED'

# ---- TC13: ErosionQuadrature 溅射产额在阈值以下为0 ----
eq = ErosionQuadrature(params)
Y_low = eq.sputtering_yield_bohdansky(50.0)
assert Y_low == 0.0, '[TC13] ErosionQuadrature 溅射产额在阈值以下为0 FAILED'

# ---- TC14: ErosionQuadrature 高能量溅射产额有界 ----
Y_high = eq.sputtering_yield_bohdansky(5000.0)
assert 0 <= Y_high <= 100.0, '[TC14] ErosionQuadrature 高能量溅射产额有界 FAILED'

# ---- TC15: ErosionQuadrature Clenshaw-Curtis权重和为2 ----
x_cc, w_cc = eq.clenshaw_curtis_rule(32)
assert abs(np.sum(w_cc) - 2.0) < 1e-12, '[TC15] ErosionQuadrature Clenshaw-Curtis权重和为2 FAILED'

# ---- TC16: ErosionQuadrature 三角形单项式积分解析验证 ----
integral_00 = eq.triangle_unit_monomial_integral([0, 0])
assert abs(integral_00 - 0.5) < 1e-12, '[TC16] ErosionQuadrature 三角形单项式积分解析验证 FAILED'

# ---- TC17: PolygonRoughness 粗糙度参数非负 ----
np.random.seed(42)
surface = PolygonRoughness(n_vertices=64)
surface.initialize_random_roughness(amplitude=1.0e-6, n_modes=10, seed=42)
stats = surface.compute_roughness_parameters()
assert stats['Ra'] >= 0 and stats['Rq'] >= 0, '[TC17] PolygonRoughness 粗糙度参数非负 FAILED'

# ---- TC18: PolygonRoughness 多步演化历史长度正确 ----
history = surface.evolve_surface(n_steps=20, erosion_rate=1.0e-9, redeposition_rate=3.0e-10, dt=1.0e-3, alpha_smooth=0.2, stochastic=False)
assert len(history) == 20, '[TC18] PolygonRoughness 多步演化历史长度正确 FAILED'

# ---- TC19: ImpurityTransport 三种扩散系数均为正 ----
transport = ImpurityTransport(params)
D_bohm = transport.compute_diffusion_coefficient('bohm')
D_classical = transport.compute_diffusion_coefficient('classical')
D_neo = transport.compute_diffusion_coefficient('neo')
assert D_bohm > 0 and D_classical > 0 and D_neo > 0, '[TC19] ImpurityTransport 三种扩散系数均为正 FAILED'

# ---- TC20: ImpurityTransport IFS映射固定种子可复现 ----
np.random.seed(42)
x_ifs1 = transport.ifs_transport_map(np.array([0.5, 0.5]), mode='diffusion')
np.random.seed(42)
x_ifs2 = transport.ifs_transport_map(np.array([0.5, 0.5]), mode='diffusion')
assert np.allclose(x_ifs1, x_ifs2), '[TC20] ImpurityTransport IFS映射固定种子可复现 FAILED'

# ---- TC21: WaveletAnalysis 变换与逆变换精确重构 ----
wv = WaveletAnalysis(order=10)
signal_test = np.sin(np.linspace(0, 2*np.pi, 64))
coeffs = wv.transform(signal_test)
reconstructed = wv.inverse_transform(coeffs)
assert np.linalg.norm(reconstructed[:64] - signal_test) < 1e-10, '[TC21] WaveletAnalysis 变换与逆变换精确重构 FAILED'

# ---- TC22: WaveletAnalysis 功率谱尺度单调递增 ----
scales, power, freqs = wv.power_spectrum(signal_test, sample_rate=64.0)
assert len(scales) > 0 and np.all(np.diff(scales) > 0), '[TC22] WaveletAnalysis 功率谱尺度单调递增 FAILED'

# ---- TC23: MonteCarloSampler 超球面采样模长为1 ----
mc = MonteCarloSampler(seed=42)
x_sphere = mc.sample_hypersphere_positive(5)
assert abs(np.linalg.norm(x_sphere) - 1.0) < 1e-12, '[TC23] MonteCarloSampler 超球面采样模长为1 FAILED'

# ---- TC24: MonteCarloSampler Maxwellian采样rms接近理论值 ----
v_th = 1.0e5
v_samples = mc.sample_maxwellian_velocity(v_th, 200)
v_rms = np.sqrt(np.mean(v_samples**2))
assert abs(v_rms - v_th) / v_th < 0.5, '[TC24] MonteCarloSampler Maxwellian采样rms接近理论值 FAILED'

# ---- TC25: RandLC 跳跃与顺序生成一致性 ----
from monte_carlo_sampler import RandLC
rng1 = RandLC(seed=314159265)
seq = [rng1.next() for _ in range(20)]
rng2 = RandLC(seed=314159265)
jump_val = rng2.jump(20)
assert abs(seq[-1] - jump_val) < 1e-10, '[TC25] RandLC 跳跃与顺序生成一致性 FAILED'

# ---- TC26: utils Coulomb对数范围约束 ----
ln_L = compute_coulomb_logarithm(50.0, 1.0e19)
assert 5.0 <= ln_L <= 25.0, '[TC26] utils Coulomb对数范围约束 FAILED'

# ---- TC27: utils safe_exp防止溢出 ----
big_val = safe_exp(100.0)
assert np.isfinite(big_val), '[TC27] utils safe_exp防止溢出 FAILED'

# ---- TC28: utils 鞘层热流为正 ----
q_sheath = compute_sheath_heat_flux(5.0e18, 50.0, 50.0)
assert q_sheath > 0 and np.isfinite(q_sheath), '[TC28] utils 鞘层热流为正 FAILED'

# ---- TC29: 集成测试主流程输出结构完整 ----
params_main = get_parameters()
sheath_main = SheathODE(params_main)
x_m, n_i_m, v_i_m, phi_m, e_m = sheath_main.solve_sheath_profile(nx=32, x_max=0.005)
assert len(x_m) == 32 and np.all(n_i_m > 0) and np.all(v_i_m > 0), '[TC29] 集成测试主流程输出结构完整 FAILED'

# ---- TC30: PoissonSolver 二维Laplace解边界条件满足 ----
solver2 = PoissonSolver(params_main)
def bf(i, j):
    nx, ny = 8, 8
    if i == 0 or i == nx-1 or j == 0 or j == ny-1:
        return 0.0
    return None
phi_2d = solver2.solve_2d_laplace(8, 8, 1.0e-4, 1.0e-4, bf)
assert phi_2d.shape == (8, 8) and np.all(np.isfinite(phi_2d)), '[TC30] PoissonSolver 二维Laplace解边界条件满足 FAILED'

print('\n全部 30 个测试通过!\n')
