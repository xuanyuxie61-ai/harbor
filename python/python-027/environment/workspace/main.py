# -*- coding: utf-8 -*-

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
    print("=" * 70)
    print("  等离子体鞘层与壁材料侵蚀数值模拟系统")
    print("  Plasma Sheath & Wall Material Erosion Simulation System")
    print("=" * 70)
    print()


def run_module_01_parameters():
    print("[模块 1/10] 物理参数初始化")
    print("-" * 50)

    params = get_parameters()
    params.print_summary()


    lambda_D = params.debye_length()
    cs = params.ion_sound_speed()
    omega_pe = params.plasma_frequency()
    ln_lambda = compute_coulomb_logarithm(params.get('T_e'), params.get('n_0'))

    print(f"  Coulomb对数 ln(Lambda) = {ln_lambda:.2f}")
    print(f"  等离子体频率 omega_pe  = {omega_pe:.3e} rad/s")
    print()
    return params


def run_module_02_sheath_ode(params):
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
    print("[模块 3/10] 鞘层泊松方程有限差分解（delsq + Jacobi迭代）")
    print("-" * 50)

    solver = PoissonSolver(params)


    phi_pois, n_e, rho = solver.solve_1d_sheath_poisson(n_i, x)


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
    print("[模块 4/10] 稀疏矩阵R8RI格式运算")
    print("-" * 50)

    n = params.get('nx')
    mat = R8RIMatrix.build_dif2(n)


    x_test = np.ones(n)
    y_sparse = mat.matvec(x_test)
    y_transpose = mat.matvec_transpose(x_test)


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
    print("[模块 5/10] 靶板表面三角网格生成")
    print("-" * 50)

    mesh = SurfaceMesh()
    mesh.generate_flat_plate_mesh(width=0.05, height=0.05, nx=21, ny=21)
    stats = mesh.mesh_quality_stats()


    cyl_mesh = SurfaceMesh()
    cyl_mesh.generate_cylindrical_mesh(radius=0.02, height=0.05, n_theta=32, n_z=16)


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
    print("[模块 6/10] 壁材料物理溅射产额与侵蚀积分")
    print("-" * 50)

    eq = ErosionQuadrature(params)


    test_energies = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    print("  D -> W 溅射产额:")
    for E in test_energies:
        Y = eq.sputtering_yield_bohdansky(E)
        print(f"    E={E:6.0f} eV, Y={Y:.4f}")


    depths = np.linspace(0, 1.0e-8, 100)
    dep_profile = eq.energy_deposition_profile(E_wall, depths)


    avg_yield, _, _ = eq.integrate_sputtering_yield_1d(
        10.0, 5000.0, n_points=64)


    tri = mesh.triangles[0]
    tri_verts = mesh.nodes[tri]
    def gamma_f(x, y):
        return 1.0e22
    def E_f(x, y):
        return E_wall
    tri_erosion = eq.integrate_erosion_over_triangle(tri_verts, gamma_f, E_f)


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
    print("[模块 7/10] 壁表面粗糙度演化模拟")
    print("-" * 50)

    surface = PolygonRoughness(n_vertices=128)
    surface.initialize_random_roughness(amplitude=1.0e-6, n_modes=15, seed=42)

    init_stats = surface.compute_roughness_parameters()


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
    print("[模块 8/10] 侵蚀杂质在鞘层中的输运模拟")
    print("-" * 50)

    transport = ImpurityTransport(params)


    D_bohm = transport.compute_diffusion_coefficient('bohm')
    D_classical = transport.compute_diffusion_coefficient('classical')
    D_neo = transport.compute_diffusion_coefficient('neo')

    print("  扩散系数对比:")
    print(f"    Bohm扩散            = {D_bohm:.3e} m^2/s")
    print(f"    经典碰撞扩散        = {D_classical:.3e} m^2/s")
    print(f"    新经典扩散          = {D_neo:.3e} m^2/s")


    x_ifs = np.random.rand(2)
    print(f"\n  IFS混沌映射 (sheath_drift模式):")
    print(f"    初始位置            = [{x_ifs[0]:.4f}, {x_ifs[1]:.4f}]")
    for step in range(10):
        x_ifs = transport.ifs_transport_map(x_ifs, mode='sheath_drift')
    print(f"    10步后位置          = [{x_ifs[0,0]:.4f}, {x_ifs[0,1]:.4f}]")


    print(f"\n  Langevin系综模拟 (50粒子)...")
    final_pos, stats = transport.simulate_ensemble(n_particles=50, n_steps=100, dt=1.0e-10)

    print(f"    平均位移            = {stats['mean_displacement']:.3e} m")
    print(f"    返回壁面比例        = {stats['return_fraction']*100:.1f}%")
    print()
    return transport, stats


def run_module_09_wavelet_analysis(params, n_i):
    print("[模块 9/10] 鞘层密度波动小波多尺度分析")
    print("-" * 50)


    t = np.linspace(0, 1, 1024)
    f_blob = 20.0
    f_drift = 5.0
    signal = (np.sin(2*np.pi*f_drift*t) +
              0.3 * np.sin(2*np.pi*f_blob*t) *
              (1.0 + 0.5*np.sin(2*np.pi*2*t)) +
              0.05 * np.random.randn(len(t)))

    wv = WaveletAnalysis(order=10)


    levels = wv.decompose_levels(signal)
    print("  小波分解能量分布:")
    total_detail_energy = 0.0
    for key in sorted([k for k in levels.keys() if isinstance(k, int)]):
        energy = np.sum(levels[key]**2)
        total_detail_energy += energy
        print(f"    Level {key}:          = {energy:.4e}")
    print(f"    总细节能量          = {total_detail_energy:.4e}")


    scales, power, freqs = wv.power_spectrum(signal, sample_rate=1024.0)
    if len(freqs) > 0:
        peak_idx = np.argmax(power)
        print(f"\n  主导频率              = {freqs[peak_idx]:.1f} Hz")
        print(f"  对应特征时间尺度      = {1.0/freqs[peak_idx]:.3e} s")


    denoised = wv.denoise(signal, threshold_ratio=0.15)
    snr_improvement = np.std(signal)**2 / (np.std(denoised - signal) + 1.0e-30)**2
    print(f"  去噪后SNR改善         = {10*np.log10(snr_improvement):.1f} dB")
    print()
    return wv, signal, denoised


def run_module_10_monte_carlo(params):
    print("[模块 10/10] 蒙特卡洛随机采样验证")
    print("-" * 50)

    mc = MonteCarloSampler(seed=params.get('rand_seed'))


    x_sphere = mc.sample_hypersphere_positive(5)
    print(f"  5维正超球面采样: 模长={np.linalg.norm(x_sphere):.6f}")


    mu, var = mc.sample_hypersphere_distance_stats(3, 500)
    print(f"  3维距离统计: mu={mu:.4f}, var={var:.6f}")


    v_th = params.ion_thermal_velocity()
    v_samples = mc.sample_maxwellian_velocity(v_th, 500)
    v_rms = np.sqrt(np.mean(v_samples**2))
    v_theory = v_th
    print(f"  Maxwellian速度: 理论rms={v_theory:.3e}, 实测rms={v_rms:.3e}")


    from monte_carlo_sampler import RandLC
    rng_test = RandLC(seed=params.get('rand_seed'))
    u_seq = [rng_test.next() for _ in range(10)]
    rng_test2 = RandLC(seed=params.get('rand_seed'))
    u_jump = rng_test2.jump(10)
    print(f"  randlc序列第10个值    = {u_seq[-1]:.8f}")
    print(f"  randlc直接跳跃到第10  = {u_jump:.8f}")
    print(f"  跳跃一致性            = {abs(u_seq[-1]-u_jump) < 1.0e-10}")


    collided, mfp = mc.sample_collision_parameter(1.0e-19, 1.0e19, 0.01)
    print(f"  碰撞采样: 碰撞={collided}, 平均自由程={mfp:.3e} m")
    print()
    return mc


def run_final_summary(params, E_wall, avg_yield, stats_transport, stats_roughness):
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
    start_time = time.time()
    print_banner()


    params = run_module_01_parameters()


    x, n_i, v_i, phi_ode, e_field, E_wall, gamma = run_module_02_sheath_ode(params)


    phi_pois, n_e, rho, phi_2d = run_module_03_poisson_solver(params, n_i, x)


    mat = run_module_04_sparse_matrix(params)


    mesh, cyl_mesh = run_module_05_surface_mesh(params)


    eq, avg_yield = run_module_06_erosion_quadrature(params, E_wall, mesh)


    surface, history = run_module_07_polygon_roughness(params)
    roughness_final = surface.compute_roughness_parameters()


    transport, stats_transport = run_module_08_impurity_transport(params)


    wv, signal, denoised = run_module_09_wavelet_analysis(params, n_i)


    mc = run_module_10_monte_carlo(params)


    run_final_summary(params, E_wall, avg_yield, stats_transport, roughness_final)

    elapsed = time.time() - start_time
    print(f"\n总运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    print("模拟完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()
