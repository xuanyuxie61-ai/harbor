#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time


from fault_dynamics import FaultRuptureDynamics, PendulumConservationMonitor
from okada_dislocation import OkadaModel
from tsunami_pde_solver import ShallowWaterSolver
from bathymetry_sampling import BathymetryGenerator
from spherical_geodesics import SphericalGeodesics
from mesh_interpolation import MeshInterpolator
from hilbert_mesh_ordering import HilbertMeshOrderer
from energy_quadrature import EnergyQuadrature
from variomino_adaptive_cover import AdaptiveMeshCover
from matrix_chain_optimizer import MatrixChainOptimizer
from parallel_scheduler import ParallelScheduler


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print(" tsunami_generation_propagation_simulation.py")
    print(" 博士级海啸生成与传播数值模拟系统")
    print("=" * 70)
    
    np.random.seed(42)
    start_time = time.time()
    



    print_section("步骤 1：断层破裂动力学模拟")
    
    fault = FaultRuptureDynamics()
    t_span = (0.0, 30.0)
    y0 = np.array([1.0, 1.0, 0.0, 1.0])
    t_rupt, y_rupt = fault.solve_rupture_ode(t_span, y0, n_steps=500)
    
    print(f"  断层滑动速度演化：t ∈ [{t_rupt[0]:.2f}, {t_rupt[-1]:.2f}] s")
    print(f"  最终滑动位移：{y_rupt[-1, 2]:.4f} m")
    

    pendulum_monitor = PendulumConservationMonitor()
    E_conserved = pendulum_monitor.check_energy_conservation(y_rupt[:, 0], y_rupt[:, 1])
    print(f"  能量守恒偏差（单摆类比）：{E_conserved:.6e}")
    



    print_section("步骤 2：Okada 弹性位错模型")
    
    okada = OkadaModel(
        strike=195.0,
        dip=13.0,
        rake=90.0,
        slip=5.0,
        length=200e3,
        width=100e3,
        depth=25e3,
        nu=0.25
    )
    

    x_grid = np.linspace(-500e3, 500e3, 81)
    y_grid = np.linspace(-500e3, 500e3, 81)
    eta_initial = okada.compute_seafloor_displacement(x_grid, y_grid)
    print(f"  网格尺寸：{len(x_grid)} × {len(y_grid)}")
    print(f"  最大初始海面抬升：{np.max(eta_initial):.4f} m")
    print(f"  最大初始海面沉降：{np.min(eta_initial):.4f} m")
    



    print_section("步骤 3：随机海底地形采样")
    
    bath_gen = BathymetryGenerator(x_grid, y_grid)
    h_bathy = bath_gen.generate_random_bathymetry(
        depth_mean=4000.0,
        depth_std=800.0,
        continental_slope=True
    )
    print(f"  平均水深：{np.mean(h_bathy):.2f} m")
    print(f"  水深范围：[{np.min(h_bathy):.2f}, {np.max(h_bathy):.2f}] m")
    



    print_section("步骤 4：球面测地距离校正")
    
    geo = SphericalGeodesics(earth_radius=6371e3)

    epicenter_lat = 38.0
    epicenter_lon = 142.5
    distances = geo.compute_grid_distances(x_grid, y_grid, epicenter_lat, epicenter_lon)
    print(f"  震中位置：({epicenter_lat}°N, {epicenter_lon}°E)")
    print(f"  最大球面距离：{np.max(distances)/1e3:.2f} km")
    



    print_section("步骤 5：网格插值与边界处理")
    
    mesh_interp = MeshInterpolator(x_grid, y_grid)

    x_fine = np.linspace(-500e3, 500e3, 161)
    y_fine = np.linspace(-500e3, 500e3, 161)
    eta_fine = mesh_interp.bilinear_interpolate(eta_initial, x_fine, y_fine)
    

    eta_periodic = mesh_interp.trigonometric_periodic_boundary(eta_initial, axis=0)
    print(f"  粗网格分辨率：{x_grid[1]-x_grid[0]:.0f} m")
    print(f"  细网格分辨率：{x_fine[1]-x_fine[0]:.0f} m")
    print(f"  周期边界插值连续性误差：{np.max(np.abs(eta_periodic[0,:]-eta_periodic[-1,:])):.6e}")
    



    print_section("步骤 6：Hilbert 曲线空间排序优化")
    
    hilbert = HilbertMeshOrderer(order=6)
    ordered_indices = hilbert.order_2d_grid(len(x_grid), len(y_grid))
    print(f"  Hilbert 曲线阶数：{hilbert.order}")
    print(f"  排序后网格点局部性指数：{hilbert.compute_locality_index(ordered_indices):.4f}")
    



    print_section("步骤 7：多子域并行任务调度")
    
    scheduler = ParallelScheduler(n_tasks=81, n_processors=4)
    task_map = scheduler.divide_tasks()
    print(f"  总任务数：{scheduler.n_tasks}")
    print(f"  处理器数：{scheduler.n_processors}")
    for proc_id, (t_start, t_end) in task_map.items():
        print(f"    处理器 {proc_id}: 任务 {t_start} - {t_end} ({t_end-t_start+1} 个)")
    



    print_section("步骤 8：自适应多分辨率网格覆盖")
    
    adaptive_cover = AdaptiveMeshCover()
    cover_mask = adaptive_cover.generate_adaptive_cover(
        eta_initial, threshold=0.1, max_level=3
    )
    n_fine = np.sum(cover_mask)
    n_total = cover_mask.size
    print(f"  自适应细化区域比例：{n_fine}/{n_total} = {100*n_fine/n_total:.2f}%")
    print(f"  细化阈值：{0.1:.2f} m")
    



    print_section("步骤 9：矩阵链最优计算序列")
    

    chain_opt = MatrixChainOptimizer()
    dims = [81, 81, 81, 81, 81]
    opt_cost, opt_order = chain_opt.find_optimal_chain(dims)
    print(f"  矩阵链维度：{dims}")
    print(f"  最优标量乘法代价：{opt_cost}")
    print(f"  最优乘法顺序：{opt_order}")
    



    print_section("步骤 10：非线性浅水波方程数值求解")
    
    solver = ShallowWaterSolver(
        x=x_grid,
        y=y_grid,
        h_bathy=h_bathy,
        g=9.81,
        Cd=0.0025,
        dt=30.0,
        n_steps=120
    )
    

    solver.set_initial_condition(eta_initial)
    

    print("  开始求解...")
    t_snapshots, eta_snapshots, u_snapshots, v_snapshots = solver.solve()
    
    print(f"  模拟总时长：{solver.dt * solver.n_steps / 60:.1f} 分钟")
    print(f"  时间步长：{solver.dt:.1f} s")
    print(f"  空间分辨率：{x_grid[1]-x_grid[0]:.0f} m")
    print(f"  最终时刻最大波高：{np.max(eta_snapshots[-1]):.4f} m")
    print(f"  最终时刻最小波高：{np.min(eta_snapshots[-1]):.4f} m")
    



    print_section("步骤 11：能量积分与守恒检验")
    
    eq = EnergyQuadrature(x_grid, y_grid)
    

    E0 = eq.compute_total_energy(eta_snapshots[0], u_snapshots[0], v_snapshots[0], h_bathy)

    E_final = eq.compute_total_energy(eta_snapshots[-1], u_snapshots[-1], v_snapshots[-1], h_bathy)
    
    print(f"  初始总能量（势能+动能）：{E0:.6e} J")
    print(f"  最终总能量：{E_final:.6e} J")
    print(f"  相对能量变化：{abs(E_final - E0) / E0:.6e}")
    

    hermite_error = eq.test_hermite_exactness(max_degree=9)
    print(f"  Hermite 求积规则精度（最大 9 阶）：")
    for deg, err in hermite_error.items():
        status = "精确" if err < 1e-12 else "误差"
        print(f"    阶数 {deg}: {err:.6e} [{status}]")
    

    tri_integral = eq.triangle_symmetric_quadrature_test()
    print(f"  三角形对称求积测试值：{tri_integral:.10f}")
    



    print_section("步骤 12：海啸传播特征分析")
    

    wavefront_distances = []
    for k in range(len(t_snapshots)):
        eta_k = eta_snapshots[k]

        threshold = 0.05 * np.max(np.abs(eta_initial))
        mask = np.abs(eta_k) > threshold
        if np.any(mask):

            dists_k = distances[mask]
            wavefront_distances.append(np.max(dists_k))
        else:
            wavefront_distances.append(0.0)
    

    t_arr = np.array(t_snapshots)
    d_arr = np.array(wavefront_distances)
    valid = d_arr > 1e3
    if np.sum(valid) > 2:
        coeffs = np.polyfit(t_arr[valid], d_arr[valid], 1)
        v_wave = coeffs[0]
        print(f"  波前传播速度（拟合）：{v_wave:.2f} m/s")
        print(f"  理论深水波速（√(gh)）：{np.sqrt(9.81 * np.mean(h_bathy)):.2f} m/s")
        print(f"  相对偏差：{abs(v_wave - np.sqrt(9.81 * np.mean(h_bathy))) / np.sqrt(9.81 * np.mean(h_bathy)) * 100:.2f}%")
    

    green_law_error = []
    for k in range(1, len(t_snapshots)):
        eta_k = eta_snapshots[k]
        max_amp_idx = np.unravel_index(np.argmax(np.abs(eta_k)), eta_k.shape)
        h_local = h_bathy[max_amp_idx]

        eta_predicted = np.max(np.abs(eta_initial)) * (np.mean(h_bathy) / h_local) ** (-0.25)
        eta_actual = np.max(np.abs(eta_k))
        if eta_predicted > 0:
            green_law_error.append(abs(eta_actual - eta_predicted) / eta_predicted)
    
    if green_law_error:
        print(f"  格林定律平均相对偏差：{np.mean(green_law_error):.4f}")
    



    print_section("步骤 13：蒙特卡洛不确定性量化")
    
    n_mc = 20
    max_amplitudes = []
    print(f"  运行 {n_mc} 次蒙特卡洛模拟...")
    
    for i_mc in range(n_mc):

        strike_mc = np.random.normal(195.0, 10.0)
        dip_mc = np.clip(np.random.normal(13.0, 3.0), 5.0, 45.0)
        slip_mc = np.clip(np.random.normal(5.0, 1.5), 1.0, 15.0)
        
        okada_mc = OkadaModel(
            strike=strike_mc, dip=dip_mc, rake=90.0,
            slip=slip_mc, length=200e3, width=100e3,
            depth=25e3, nu=0.25
        )
        eta_mc = okada_mc.compute_seafloor_displacement(x_grid, y_grid)
        max_amplitudes.append(np.max(np.abs(eta_mc)))
    
    print(f"  最大波幅均值：{np.mean(max_amplitudes):.4f} m")
    print(f"  最大波幅标准差：{np.std(max_amplitudes):.4f} m")
    print(f"  95% 置信区间：[{np.percentile(max_amplitudes, 2.5):.4f}, {np.percentile(max_amplitudes, 97.5):.4f}] m")
    



    elapsed = time.time() - start_time
    print_section("模拟完成")
    print(f"  总运行时间：{elapsed:.2f} 秒")
    print(f"  所有模块运行正常，无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()
