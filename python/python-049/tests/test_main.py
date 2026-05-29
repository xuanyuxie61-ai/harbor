#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 tsunami_generation_propagation_simulation.py
 
 博士级合成项目：海底地震引发海啸的非线性浅水波方程数值模拟
 
 本项目融合 15 个种子项目的核心算法，围绕地球物理前沿问题——
 海啸生成与传播——构建一个具备不确定性量化、自适应网格覆盖、
 高精度能量守恒监测的数值模拟系统。
 
 核心物理模型：
   1) Okada 弹性半空间位错模型计算初始海面位移
   2) 非线性浅水波方程（Saint-Venant 方程组）
   3) 球面测地距离校正的长距离传播
   4) 海底摩擦与复杂地形耦合
 
 数值方法：
   - 空间离散：交错网格（Arakawa C-grid）有限差分
   - 时间离散：固定点迭代后向 Euler（隐式，无条件稳定）
   - 边界处理：三角插值周期性边界 + 辐射边界条件
   - 积分检验：三角形对称求积 + Hermite 求积精度验证
 
 运行方式：
   python main.py
   （零参数，所有物理参数内置，自动运行完整模拟流程）
"""

import numpy as np
import time

# 导入各模块
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
    
    # ============================================================
    # 步骤 1：断层破裂动力学模拟（tough_ode + pendulum_comparison）
    # ============================================================
    print_section("步骤 1：断层破裂动力学模拟")
    
    fault = FaultRuptureDynamics()
    t_span = (0.0, 30.0)
    y0 = np.array([1.0, 1.0, 0.0, 1.0])
    t_rupt, y_rupt = fault.solve_rupture_ode(t_span, y0, n_steps=500)
    
    print(f"  断层滑动速度演化：t ∈ [{t_rupt[0]:.2f}, {t_rupt[-1]:.2f}] s")
    print(f"  最终滑动位移：{y_rupt[-1, 2]:.4f} m")
    
    # 使用单摆类比监测能量守恒
    pendulum_monitor = PendulumConservationMonitor()
    E_conserved = pendulum_monitor.check_energy_conservation(y_rupt[:, 0], y_rupt[:, 1])
    print(f"  能量守恒偏差（单摆类比）：{E_conserved:.6e}")
    
    # ============================================================
    # 步骤 2：Okada 模型计算初始海面位移
    # ============================================================
    print_section("步骤 2：Okada 弹性位错模型")
    
    okada = OkadaModel(
        strike=195.0,       # 断层走向 (度)
        dip=13.0,           # 断层倾角 (度)
        rake=90.0,          # 滑动角 (度)
        slip=5.0,           # 滑动量 (m)
        length=200e3,       # 断层长度 (m)
        width=100e3,        # 断层宽度 (m)
        depth=25e3,         # 断层顶部深度 (m)
        nu=0.25             # 泊松比
    )
    
    # 计算初始海面位移场
    x_grid = np.linspace(-500e3, 500e3, 81)
    y_grid = np.linspace(-500e3, 500e3, 81)
    eta_initial = okada.compute_seafloor_displacement(x_grid, y_grid)
    print(f"  网格尺寸：{len(x_grid)} × {len(y_grid)}")
    print(f"  最大初始海面抬升：{np.max(eta_initial):.4f} m")
    print(f"  最大初始海面沉降：{np.min(eta_initial):.4f} m")
    
    # ============================================================
    # 步骤 3：随机海底地形生成（rcont + unicycle_random）
    # ============================================================
    print_section("步骤 3：随机海底地形采样")
    
    bath_gen = BathymetryGenerator(x_grid, y_grid)
    h_bathy = bath_gen.generate_random_bathymetry(
        depth_mean=4000.0,
        depth_std=800.0,
        continental_slope=True
    )
    print(f"  平均水深：{np.mean(h_bathy):.2f} m")
    print(f"  水深范围：[{np.min(h_bathy):.2f}, {np.max(h_bathy):.2f}] m")
    
    # ============================================================
    # 步骤 4：球面测地距离校正（sphere_positive_distance）
    # ============================================================
    print_section("步骤 4：球面测地距离校正")
    
    geo = SphericalGeodesics(earth_radius=6371e3)
    # 模拟震中位置与网格点的球面距离
    epicenter_lat = 38.0
    epicenter_lon = 142.5
    distances = geo.compute_grid_distances(x_grid, y_grid, epicenter_lat, epicenter_lon)
    print(f"  震中位置：({epicenter_lat}°N, {epicenter_lon}°E)")
    print(f"  最大球面距离：{np.max(distances)/1e3:.2f} km")
    
    # ============================================================
    # 步骤 5：网格插值系统（quadrilateral_surface_display + trig_interp）
    # ============================================================
    print_section("步骤 5：网格插值与边界处理")
    
    mesh_interp = MeshInterpolator(x_grid, y_grid)
    # 四边形双线性插值：在粗网格上插值到细网格
    x_fine = np.linspace(-500e3, 500e3, 161)
    y_fine = np.linspace(-500e3, 500e3, 161)
    eta_fine = mesh_interp.bilinear_interpolate(eta_initial, x_fine, y_fine)
    
    # 三角插值处理周期性边界
    eta_periodic = mesh_interp.trigonometric_periodic_boundary(eta_initial, axis=0)
    print(f"  粗网格分辨率：{x_grid[1]-x_grid[0]:.0f} m")
    print(f"  细网格分辨率：{x_fine[1]-x_fine[0]:.0f} m")
    print(f"  周期边界插值连续性误差：{np.max(np.abs(eta_periodic[0,:]-eta_periodic[-1,:])):.6e}")
    
    # ============================================================
    # 步骤 6：Hilbert 曲线空间排序（hilbert_curve_3d）
    # ============================================================
    print_section("步骤 6：Hilbert 曲线空间排序优化")
    
    hilbert = HilbertMeshOrderer(order=6)
    ordered_indices = hilbert.order_2d_grid(len(x_grid), len(y_grid))
    print(f"  Hilbert 曲线阶数：{hilbert.order}")
    print(f"  排序后网格点局部性指数：{hilbert.compute_locality_index(ordered_indices):.4f}")
    
    # ============================================================
    # 步骤 7：并行任务调度（task_division）
    # ============================================================
    print_section("步骤 7：多子域并行任务调度")
    
    scheduler = ParallelScheduler(n_tasks=81, n_processors=4)
    task_map = scheduler.divide_tasks()
    print(f"  总任务数：{scheduler.n_tasks}")
    print(f"  处理器数：{scheduler.n_processors}")
    for proc_id, (t_start, t_end) in task_map.items():
        print(f"    处理器 {proc_id}: 任务 {t_start} - {t_end} ({t_end-t_start+1} 个)")
    
    # ============================================================
    # 步骤 8：自适应网格覆盖（variomino_matrix）
    # ============================================================
    print_section("步骤 8：自适应多分辨率网格覆盖")
    
    adaptive_cover = AdaptiveMeshCover()
    cover_mask = adaptive_cover.generate_adaptive_cover(
        eta_initial, threshold=0.1, max_level=3
    )
    n_fine = np.sum(cover_mask)
    n_total = cover_mask.size
    print(f"  自适应细化区域比例：{n_fine}/{n_total} = {100*n_fine/n_total:.2f}%")
    print(f"  细化阈值：{0.1:.2f} m")
    
    # ============================================================
    # 步骤 9：矩阵链最优计算序列（matrix_chain_brute）
    # ============================================================
    print_section("步骤 9：矩阵链最优计算序列")
    
    # 多步状态转移矩阵的最优乘法链
    chain_opt = MatrixChainOptimizer()
    dims = [81, 81, 81, 81, 81]
    opt_cost, opt_order = chain_opt.find_optimal_chain(dims)
    print(f"  矩阵链维度：{dims}")
    print(f"  最优标量乘法代价：{opt_cost}")
    print(f"  最优乘法顺序：{opt_order}")
    
    # ============================================================
    # 步骤 10：非线性浅水波方程数值求解（backward_euler_fixed）
    # ============================================================
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
    
    # 设置初始条件
    solver.set_initial_condition(eta_initial)
    
    # 求解
    print("  开始求解...")
    t_snapshots, eta_snapshots, u_snapshots, v_snapshots = solver.solve()
    
    print(f"  模拟总时长：{solver.dt * solver.n_steps / 60:.1f} 分钟")
    print(f"  时间步长：{solver.dt:.1f} s")
    print(f"  空间分辨率：{x_grid[1]-x_grid[0]:.0f} m")
    print(f"  最终时刻最大波高：{np.max(eta_snapshots[-1]):.4f} m")
    print(f"  最终时刻最小波高：{np.min(eta_snapshots[-1]):.4f} m")
    
    # ============================================================
    # 步骤 11：高精度能量积分与守恒检验
    # ============================================================
    print_section("步骤 11：能量积分与守恒检验")
    
    eq = EnergyQuadrature(x_grid, y_grid)
    
    # 初始总能量
    E0 = eq.compute_total_energy(eta_snapshots[0], u_snapshots[0], v_snapshots[0], h_bathy)
    # 最终总能量
    E_final = eq.compute_total_energy(eta_snapshots[-1], u_snapshots[-1], v_snapshots[-1], h_bathy)
    
    print(f"  初始总能量（势能+动能）：{E0:.6e} J")
    print(f"  最终总能量：{E_final:.6e} J")
    print(f"  相对能量变化：{abs(E_final - E0) / E0:.6e}")
    
    # Hermite 求积精度检验
    hermite_error = eq.test_hermite_exactness(max_degree=9)
    print(f"  Hermite 求积规则精度（最大 9 阶）：")
    for deg, err in hermite_error.items():
        status = "精确" if err < 1e-12 else "误差"
        print(f"    阶数 {deg}: {err:.6e} [{status}]")
    
    # 三角形对称求积
    tri_integral = eq.triangle_symmetric_quadrature_test()
    print(f"  三角形对称求积测试值：{tri_integral:.10f}")
    
    # ============================================================
    # 步骤 12：海啸传播特征分析
    # ============================================================
    print_section("步骤 12：海啸传播特征分析")
    
    # 计算各时刻波前到达距离
    wavefront_distances = []
    for k in range(len(t_snapshots)):
        eta_k = eta_snapshots[k]
        # 找到波高超过阈值的点
        threshold = 0.05 * np.max(np.abs(eta_initial))
        mask = np.abs(eta_k) > threshold
        if np.any(mask):
            # 使用球面距离
            dists_k = distances[mask]
            wavefront_distances.append(np.max(dists_k))
        else:
            wavefront_distances.append(0.0)
    
    # 线性拟合波前传播速度
    t_arr = np.array(t_snapshots)
    d_arr = np.array(wavefront_distances)
    valid = d_arr > 1e3
    if np.sum(valid) > 2:
        coeffs = np.polyfit(t_arr[valid], d_arr[valid], 1)
        v_wave = coeffs[0]
        print(f"  波前传播速度（拟合）：{v_wave:.2f} m/s")
        print(f"  理论深水波速（√(gh)）：{np.sqrt(9.81 * np.mean(h_bathy)):.2f} m/s")
        print(f"  相对偏差：{abs(v_wave - np.sqrt(9.81 * np.mean(h_bathy))) / np.sqrt(9.81 * np.mean(h_bathy)) * 100:.2f}%")
    
    # 海啸波幅衰减分析（格林定律 η ∝ (gh)^{-1/4}）
    green_law_error = []
    for k in range(1, len(t_snapshots)):
        eta_k = eta_snapshots[k]
        max_amp_idx = np.unravel_index(np.argmax(np.abs(eta_k)), eta_k.shape)
        h_local = h_bathy[max_amp_idx]
        # 格林定律预测
        eta_predicted = np.max(np.abs(eta_initial)) * (np.mean(h_bathy) / h_local) ** (-0.25)
        eta_actual = np.max(np.abs(eta_k))
        if eta_predicted > 0:
            green_law_error.append(abs(eta_actual - eta_predicted) / eta_predicted)
    
    if green_law_error:
        print(f"  格林定律平均相对偏差：{np.mean(green_law_error):.4f}")
    
    # ============================================================
    # 步骤 13：不确定性量化（蒙特卡洛）
    # ============================================================
    print_section("步骤 13：蒙特卡洛不确定性量化")
    
    n_mc = 20
    max_amplitudes = []
    print(f"  运行 {n_mc} 次蒙特卡洛模拟...")
    
    for i_mc in range(n_mc):
        # 随机断层参数（unicycle 随机排列思想）
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
    
    # ============================================================
    # 完成
    # ============================================================
    elapsed = time.time() - start_time
    print_section("模拟完成")
    print(f"  总运行时间：{elapsed:.2f} 秒")
    print(f"  所有模块运行正常，无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（32个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: FaultRuptureDynamics friction coefficient returns finite scalar ----
fault = FaultRuptureDynamics()
mu_val = fault.friction_coefficient(1e-6, 1.0)
assert np.isfinite(mu_val), '[TC01] Fault friction coefficient finite FAILED'

# ---- TC02: FaultRuptureDynamics solve_rupture_ode returns monotonic time and correct shape ----
fault = FaultRuptureDynamics()
t_rupt, y_rupt = fault.solve_rupture_ode((0.0, 10.0), np.array([1.0, 1.0, 0.0, 1.0]), n_steps=100)
assert np.all(np.diff(t_rupt) > 0), '[TC02] Rupture ODE time monotonic FAILED'
assert y_rupt.shape == (101, 4), '[TC02] Rupture ODE solution shape FAILED'

# ---- TC03: PendulumConservationMonitor energy deviation is non-negative ----
monitor = PendulumConservationMonitor()
theta = np.linspace(0, np.pi / 4, 50)
omega = np.sin(theta)
dev = monitor.check_energy_conservation(theta, omega)
assert dev >= 0.0, '[TC03] Pendulum energy deviation non-negative FAILED'

# ---- TC04: OkadaModel seafloor displacement has correct shape and finite values ----
okada = OkadaModel(strike=0.0, dip=45.0, rake=90.0, slip=1.0, length=100e3, width=50e3, depth=10e3, nu=0.25)
x_g = np.linspace(-200e3, 200e3, 21)
y_g = np.linspace(-200e3, 200e3, 21)
eta = okada.compute_seafloor_displacement(x_g, y_g)
assert eta.shape == (21, 21), '[TC04] Okada displacement shape FAILED'
assert np.all(np.isfinite(eta)), '[TC04] Okada displacement finite FAILED'

# ---- TC05: OkadaModel init raises ValueError for invalid dip ----
try:
    OkadaModel(strike=0.0, dip=-10.0, rake=0.0, slip=1.0, length=100e3, width=50e3, depth=10e3)
    assert False, '[TC05] Okada invalid dip should raise FAILED'
except ValueError:
    pass

# ---- TC06: SphericalGeodesics haversine distance is symmetric ----
geo = SphericalGeodesics()
d1 = geo.haversine_distance(0.0, 0.0, 10.0, 10.0)
d2 = geo.haversine_distance(10.0, 10.0, 0.0, 0.0)
assert abs(d1 - d2) < 1e-6, '[TC06] Haversine symmetry FAILED'

# ---- TC07: SphericalGeodesics haversine distance at same point is near zero ----
geo = SphericalGeodesics()
d = geo.haversine_distance(35.0, 140.0, 35.0, 140.0)
assert abs(d) < 1e-3, '[TC07] Haversine same point FAILED'

# ---- TC08: SphericalGeodesics compute_grid_distances shape and non-negative ----
geo = SphericalGeodesics()
x_g = np.linspace(-100e3, 100e3, 11)
y_g = np.linspace(-100e3, 100e3, 11)
dist = geo.compute_grid_distances(x_g, y_g, 38.0, 142.0)
assert dist.shape == (11, 11), '[TC08] Grid distances shape FAILED'
assert np.all(dist >= 0), '[TC08] Grid distances non-negative FAILED'

# ---- TC09: SphericalGeodesics compute_travel_time finite and non-negative ----
geo = SphericalGeodesics()
dist = np.array([[1e3, 2e3], [3e3, 4e3]])
depth = np.array([[1000.0, 2000.0], [3000.0, 4000.0]])
tt = geo.compute_travel_time(dist, depth)
assert np.all(np.isfinite(tt)), '[TC09] Travel time finite FAILED'
assert np.all(tt >= 0), '[TC09] Travel time non-negative FAILED'

# ---- TC10: BathymetryGenerator random bathymetry returns positive depths ----
np.random.seed(42)
x_g = np.linspace(-100e3, 100e3, 11)
y_g = np.linspace(-100e3, 100e3, 11)
bath = BathymetryGenerator(x_g, y_g)
h = bath.generate_random_bathymetry(depth_mean=3000.0, depth_std=500.0, continental_slope=False)
assert h.shape == (11, 11), '[TC10] Bathymetry shape FAILED'
assert np.all(h > 0), '[TC10] Bathymetry positive FAILED'

# ---- TC11: BathymetryGenerator rcont constraints produce correct shape and positive values ----
np.random.seed(42)
bath = BathymetryGenerator(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
rows = np.array([10, 10, 10, 10, 10])
cols = np.array([10, 10, 10, 10, 10])
h = bath.generate_bathymetry_with_rcont_constraints(rows, cols)
assert h.shape == (5, 5), '[TC11] Rcont shape FAILED'
assert np.all(h > 0), '[TC11] Rcont positive FAILED'

# ---- TC12: MeshInterpolator bilinear interpolation preserves constant field exactly ----
x_c = np.linspace(0, 1, 5)
y_c = np.linspace(0, 1, 5)
z_c = np.ones((5, 5)) * 3.14
interp = MeshInterpolator(x_c, y_c)
x_f = np.linspace(0, 1, 9)
y_f = np.linspace(0, 1, 9)
z_f = interp.bilinear_interpolate(z_c, x_f, y_f)
assert np.allclose(z_f, 3.14, atol=1e-10), '[TC12] Bilinear constant preservation FAILED'

# ---- TC13: MeshInterpolator trigonometric periodic boundary reduces discontinuity ----
np.random.seed(42)
field = np.random.rand(10, 10)
interp = MeshInterpolator(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
fp = interp.trigonometric_periodic_boundary(field, axis=1)
diff_after = np.max(np.abs(fp[:, 0] - fp[:, -1]))
assert diff_after < 1e-10, '[TC13] Periodic boundary discontinuity FAILED'

# ---- TC14: HilbertMeshOrderer h_to_xy and xy_to_h are inverses ----
hmo = HilbertMeshOrderer(order=4)
h = 42
x, y = hmo.h_to_xy(h)
h_back = hmo.xy_to_h(x, y)
assert h == h_back, '[TC14] Hilbert h_to_xy inverse FAILED'

# ---- TC15: HilbertMeshOrderer locality better than row-major ----
hmo = HilbertMeshOrderer(order=4)
loc_h, loc_r = hmo.compare_orderings(8, 8)
assert loc_h <= loc_r, '[TC15] Hilbert locality not better than row-major FAILED'

# ---- TC16: EnergyQuadrature square monomial integral matches analytical formula ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
assert abs(eq.square_monomial_integral((2, 3)) - 1.0 / 12.0) < 1e-12, '[TC16] Square monomial FAILED'

# ---- TC17: EnergyQuadrature Hermite odd degree exact value is zero ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
assert eq.hermite_integral_exact(3) == 0.0, '[TC17] Hermite odd degree FAILED'

# ---- TC18: EnergyQuadrature triangle symmetric quadrature on unit function equals 0.5 ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
val = eq.triangle_symmetric_quadrature_test()
assert abs(val - 0.5) < 1e-10, '[TC18] Triangle quadrature unit function FAILED'

# ---- TC19: EnergyQuadrature total energy is non-negative with zero velocity ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
eta = np.ones((5, 5))
u = np.zeros((5, 5))
v = np.zeros((5, 5))
h_bathy = np.ones((5, 5)) * 10.0
E = eq.compute_total_energy(eta, u, v, h_bathy)
assert E >= 0.0, '[TC19] Total energy non-negative FAILED'

# ---- TC20: AdaptiveMeshCover generate_adaptive_cover returns correct boolean mask ----
amc = AdaptiveMeshCover()
np.random.seed(42)
field = np.random.rand(16, 16)
mask = amc.generate_adaptive_cover(field, threshold=0.1, max_level=2)
assert mask.shape == (16, 16), '[TC20] Adaptive cover shape FAILED'
assert mask.dtype == bool, '[TC20] Adaptive cover dtype FAILED'

# ---- TC21: AdaptiveMeshCover variomino transformations count does not exceed 4 ----
amc = AdaptiveMeshCover()
tile = np.array([[1, 0], [0, 0]])
variants = amc.variomino_transformations(tile)
assert len(variants) <= 4, '[TC21] Variomino transformations count FAILED'

# ---- TC22: MatrixChainOptimizer Catalan numbers match known values ----
mco = MatrixChainOptimizer()
assert mco.catalan_number(0) == 1, '[TC22] Catalan n=0 FAILED'
assert mco.catalan_number(1) == 1, '[TC22] Catalan n=1 FAILED'
assert mco.catalan_number(2) == 2, '[TC22] Catalan n=2 FAILED'
assert mco.catalan_number(3) == 5, '[TC22] Catalan n=3 FAILED'

# ---- TC23: MatrixChainOptimizer optimal chain cost is not worse than naive ----
mco = MatrixChainOptimizer()
dims = [10, 100, 10, 100]
opt_cost, _ = mco.find_optimal_chain(dims)
naive_cost = mco.pivot_sequence_to_cost(3, [1, 0], dims)
assert opt_cost <= naive_cost, '[TC23] Optimal chain cost FAILED'

# ---- TC24: MatrixChainOptimizer optimal_matrix_power M^1 equals M ----
mco = MatrixChainOptimizer()
M = np.array([[2.0, 1.0], [0.0, 1.0]])
Mp, n_mul = mco.optimal_matrix_power(M, 1)
assert np.allclose(Mp, M), '[TC24] Matrix power M^1 FAILED'
assert n_mul == 0, '[TC24] Matrix power M^1 multiply count FAILED'

# ---- TC25: ParallelScheduler divide_tasks covers all tasks exactly ----
sched = ParallelScheduler(n_tasks=100, n_processors=4)
task_map = sched.divide_tasks()
total = sum(end - start + 1 for start, end in task_map.values())
assert total == 100, '[TC25] Task coverage FAILED'

# ---- TC26: ParallelScheduler load balance imbalance is small for uniform division ----
sched = ParallelScheduler(n_tasks=100, n_processors=4)
task_map = sched.divide_tasks()
imbalance, max_load, min_load = sched.compute_load_balance(task_map)
assert imbalance < 0.1, '[TC26] Load balance imbalance FAILED'

# ---- TC27: ShallowWaterSolver set_initial_condition preserves field shape ----
x_g = np.linspace(0, 1e3, 11)
y_g = np.linspace(0, 1e3, 11)
h_bathy = np.ones((11, 11)) * 100.0
solver = ShallowWaterSolver(x_g, y_g, h_bathy, dt=10.0, n_steps=5)
eta0 = np.zeros((11, 11))
solver.set_initial_condition(eta0)
assert solver.eta.shape == (11, 11), '[TC27] Solver IC shape FAILED'

# ---- TC28: ShallowWaterSolver solve returns consistent snapshot lists ----
x_g = np.linspace(0, 1e3, 11)
y_g = np.linspace(0, 1e3, 11)
h_bathy = np.ones((11, 11)) * 100.0
solver = ShallowWaterSolver(x_g, y_g, h_bathy, dt=10.0, n_steps=5)
eta0 = np.zeros((11, 11))
eta0[5, 5] = 1.0
solver.set_initial_condition(eta0)
t_snap, eta_snap, u_snap, v_snap = solver.solve()
assert len(t_snap) == len(eta_snap) == len(u_snap) == len(v_snap), '[TC28] Solver snapshot lengths FAILED'
assert eta_snap[0].shape == (11, 11), '[TC28] Solver snapshot shape FAILED'

# ---- TC29: SphericalGeodesics geographic-to-cartesian roundtrip ----
geo = SphericalGeodesics()
lat, lon = 35.0, 140.0
xyz = geo.geographic_to_cartesian(lat, lon)
lat_back, lon_back = geo.cartesian_to_geographic(xyz[0], xyz[1], xyz[2])
assert abs(lat_back - lat) < 1e-6, '[TC29] Geographic cartesian roundtrip lat FAILED'
assert abs(lon_back - lon) < 1e-6, '[TC29] Geographic cartesian roundtrip lon FAILED'

# ---- TC30: EnergyQuadrature Gauss-Hermite quadrature exact for even polynomial ----
eq = EnergyQuadrature(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
exact = eq.hermite_integral_exact(4, option=1)
numerical = eq.gauss_hermite_quadrature(lambda x: x ** 4, n_points=10)
assert abs(numerical - exact) / abs(exact) < 1e-10, '[TC30] Gauss-Hermite even polynomial FAILED'

# ---- TC31: MeshInterpolator cardinal basis at node equals 1 ----
interp = MeshInterpolator(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
x_nodes = np.linspace(0, 1, 5)
Cj = interp.cardinal_basis(x_nodes, x_nodes[2], 2)
assert abs(Cj - 1.0) < 1e-10, '[TC31] Cardinal basis at node FAILED'

# ---- TC32: FaultRuptureDynamics init raises ValueError for invalid friction parameter a ----
try:
    FaultRuptureDynamics(a=-0.01)
    assert False, '[TC32] Fault invalid a should raise FAILED'
except ValueError:
    pass

print('\n全部 32 个测试通过!\n')
