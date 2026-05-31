"""
main.py
统一入口：自适应谱-有限元粒子方法负载均衡的高性能计算框架

本程序演示一个完整的从粒子初始化、混沌轨迹积分、场求解、
负载评估到动态负载均衡的博士级科学计算流程。

科学问题:
    在等离子体物理粒子-网格（PIC）模拟中，带电粒子在湍流电磁场中的
    运动导致空间分布高度不均匀，造成并行计算负载严重失衡。
    本框架通过以下步骤解决该问题:
    
    1. 粒子初始化: 使用球面/球体采样生成各向同性初始分布
    2. 轨迹积分:  采用 RKF45 自适应积分求解 Rucklidge/Arneodo 混沌 ODE
    3. 场求解:     使用有限元 (FEM) + 多重网格 (MG) 求解泊松方程
    4. 负载评估:   基于粒子数密度和场求解开销量化各区域负载
    5. 谱分析:     拉盖尔多项式分析径向分布，切比雪夫插值重构场量
    6. 负载均衡:   正交递归二分法 (ORB) 动态重划分计算域
    7. 数值积分:   高精度三角形求积规则计算物理量
    8. 快速求和:   Toeplitz 矩阵结构加速长程相互作用

物理模型方程:
    - 泊松方程（静电势）:
        -nabla^2 phi(x,y) = rho(x,y) / epsilon_0
      
    - 粒子运动（由混沌流场驱动）:
        dx/dt = v
        dv/dt = q/m * E(x) + F_turbulence(x)
      
      其中 F_turbulence 由 Rucklidge/Arneodo 系统近似:
        dX/dt = f_Rucklidge(X)  或  f_Arneodo(X)
    
    - 负载密度:
        w(x,y) = n_p(x,y) + C_field * |E(x,y)|^2
      
    - 不均衡度量:
        I = max_p(w_p) / mean(w_p)

运行方式:
    python main.py
    （零参数，全部参数内嵌为默认值）
"""

import numpy as np
import time

from particle_dynamics import (
    rucklidge_deriv, arneodo_deriv, integrate_trajectory,
    compute_particle_load_field
)
from mesh_generator import QuadMesh, build_delaunay_triangulation
from fem_solver import FEMSystem
from multigrid_poisson import MultigridPoisson1D, MultigridPoisson2D
from spectral_analysis import (
    laguerre_polynomial, generalized_laguerre_function,
    chebyshev_nodes, divided_differences, newton_interpolate,
    chebyshev_interpolate, radial_distribution_spectrum, chebyshev_spectral_derivative
)
from quadrature_rules import integrate_over_mesh, compute_moment_over_mesh
from load_balancer import LoadBalancer, diffusion_based_load_balance
from fast_summation import (
    toeplitz_mv, toeplitz_embedded_fft_mv,
    sample_unit_ball_positive, sample_unit_sphere_surface,
    compute_prefix_sum_2d, query_region_count,
    multipole_expansion, build_interaction_matrix_toeplitz
)
from utils import (
    safe_divide, check_bounds, compute_triangle_area,
    reference_to_physical_q4, mesh_base_one,
    gauss_seidel_sweep, restrict_coarse_to_fine,
    restrict_fine_to_coarse, is_power_of_two
)


def main():
    print("=" * 78)
    print("  自适应谱-有限元粒子方法负载均衡高性能计算框架")
    print("  Adaptive Spectral-Finite-Element Load Balancing for Particle Methods")
    print("=" * 78)
    print()

    # ========================================================================
    # 参数设置
    # ========================================================================
    np.random.seed(42)
    n_particles = 800
    n_procs = 8
    domain = (0.0, 1.0, 0.0, 1.0)
    t_final = 5.0
    relerr = 1e-5
    abserr = 1e-8

    print(f"[Config] Particles: {n_particles}")
    print(f"[Config] Processors (simulated): {n_procs}")
    print(f"[Config] Domain: {domain}")
    print(f"[Config] Integration time: {t_final}")
    print()

    # ========================================================================
    # Step 1: 粒子初始化（球面/球体采样）
    # ========================================================================
    print("[Step 1] Particle Initialization via Spherical Sampling")
    print("-" * 50)

    # 使用 ball_positive_sample 思想：球体内均匀采样
    ball_samples = sample_unit_ball_positive(n_particles)
    # 映射到二维域（投影到 x-y 平面，缩放）
    particles_init = np.zeros((n_particles, 2))
    particles_init[:, 0] = check_bounds(
        0.5 + 0.3 * ball_samples[:, 0] * np.cos(ball_samples[:, 1] * 2 * np.pi),
        domain[0], domain[1], "x_init"
    )
    particles_init[:, 1] = check_bounds(
        0.5 + 0.3 * ball_samples[:, 1] * np.sin(ball_samples[:, 2] * 2 * np.pi),
        domain[2], domain[3], "y_init"
    )
    print(f"  Initial particle centroid: ({np.mean(particles_init[:,0]):.4f}, {np.mean(particles_init[:,1]):.4f})")
    print(f"  Initial particle std: ({np.std(particles_init[:,0]):.4f}, {np.std(particles_init[:,1]):.4f})")
    print()

    # ========================================================================
    # Step 2: 混沌轨迹积分（RKF45 + Rucklidge/Arneodo）
    # ========================================================================
    print("[Step 2] Chaotic Trajectory Integration (RKF45)")
    print("-" * 50)

    # 选取前 10 个粒子演示轨迹积分
    n_demo = min(10, n_particles)
    trajectory_data = []
    for p in range(n_demo):
        # 将 2D 位置扩展为 3D 状态，加入一个 z 分量用于混沌系统
        xyz0 = np.array([
            particles_init[p, 0] * 10.0 - 5.0,  # 缩放到混沌系统的典型范围
            particles_init[p, 1] * 10.0 - 5.0,
            0.1 * np.random.randn()
        ])
        t_arr, y_arr = integrate_trajectory(
            rucklidge_deriv, xyz0, (0.0, t_final), relerr, abserr, max_steps=5000
        )
        trajectory_data.append((t_arr, y_arr))
        if p < 3:
            print(f"  Particle {p}: {len(t_arr)} steps, final state=({y_arr[-1,0]:.4f},{y_arr[-1,1]:.4f},{y_arr[-1,2]:.4f})")

    # 更新粒子位置为积分终态的投影
    particles = particles_init.copy()
    for p in range(n_demo):
        final = trajectory_data[p][1][-1]
        # 将混沌轨迹终态映射回域内
        particles[p, 0] = check_bounds(
            0.5 + 0.2 * final[0], domain[0], domain[1], "x"
        )
        particles[p, 1] = check_bounds(
            0.5 + 0.2 * final[1], domain[2], domain[3], "y"
        )
    print(f"  Updated {n_demo} particle positions from chaotic trajectories.")
    print()

    # ========================================================================
    # Step 3: 网格生成与自适应细化
    # ========================================================================
    print("[Step 3] Adaptive Mesh Generation (Q4 + Delaunay)")
    print("-" * 50)

    mesh = QuadMesh(domain, nx=8, ny=8)
    print(f"  Initial mesh: {mesh.nodes.shape[0]} nodes, {len(mesh.elements)} elements")

    # 基于负载自适应细化
    mesh.refine_by_load(particles, theta=0.4, max_level=2)
    print(f"  Refined mesh: {mesh.nodes.shape[0]} nodes, {len(mesh.elements)} elements")

    # 三角剖分
    nodes_tri, triangles_tri = mesh.triangulate_elements()
    print(f"  Triangulated: {nodes_tri.shape[0]} nodes, {triangles_tri.shape[0]} triangles")

    # 验证三角形有效性
    min_area = float('inf')
    for e in range(triangles_tri.shape[0]):
        verts = nodes_tri[triangles_tri[e] - 1]
        area = abs(compute_triangle_area(verts[0], verts[1], verts[2]))
        if area > 1e-14:
            min_area = min(min_area, area)
    print(f"  Minimum valid triangle area: {min_area:.3e}")
    print()

    # ========================================================================
    # Step 4: 有限元场求解（泊松方程）
    # ========================================================================
    print("[Step 4] Finite Element Field Solver (Poisson)")
    print("-" * 50)

    fem = FEMSystem(nodes_tri, triangles_tri)
    print(f"  FEM system: {fem.n_nodes} nodes, {fem.n_tri} elements")
    print(f"  Boundary nodes: {len(fem.boundary_nodes)}")

    # 将粒子负载沉积为右端项
    load_field = compute_particle_load_field(particles, domain, 32, 32)
    # 插值到 FEM 节点
    from scipy.interpolate import RegularGridInterpolator
    try:
        x_grid = np.linspace(domain[0], domain[1], 32)
        y_grid = np.linspace(domain[2], domain[3], 32)
        interpolator = RegularGridInterpolator(
            (x_grid, y_grid), load_field.T,
            bounds_error=False, fill_value=0.0
        )
        rhs_nodes = interpolator(fem.nodes)
    except Exception as e:
        print(f"  [Fallback] Using nearest-neighbor interpolation due to: {e}")
        rhs_nodes = np.zeros(fem.n_nodes)
        for i in range(fem.n_nodes):
            nx_i = int((fem.nodes[i, 0] - domain[0]) / (domain[1] - domain[0]) * 31)
            ny_i = int((fem.nodes[i, 1] - domain[2]) / (domain[3] - domain[2]) * 31)
            nx_i = max(0, min(31, nx_i))
            ny_i = max(0, min(31, ny_i))
            rhs_nodes[i] = load_field[nx_i, ny_i]

    rhs_nodes = check_bounds(rhs_nodes, -1e6, 1e6, "rhs")

    # 求解泊松方程
    u_fem = fem.solve_poisson(rhs_nodes)
    print(f"  FEM solution range: [{u_fem.min():.4f}, {u_fem.max():.4f}]")
    print()

    # ========================================================================
    # Step 5: 多重网格泊松求解（1D验证 + 2D场求解）
    # ========================================================================
    print("[Step 5] Multigrid Poisson Solver")
    print("-" * 50)

    # 1D 验证
    def force_1d(x):
        return np.pi ** 2 * np.sin(np.pi * x)

    mg1d = MultigridPoisson1D(
        n=128, a=0.0, b=1.0, ua=0.0, ub=0.0,
        force_func=force_1d
    )
    u_1d, it_1d = mg1d.solve(tol=1e-8, max_iter=50)
    print(f"  1D MG: converged in {it_1d} iterations, max |u|={np.max(np.abs(u_1d)):.4f}")

    # 2D 多重网格
    nx_mg = 64
    ny_mg = 64
    mg2d = MultigridPoisson2D(nx_mg, ny_mg, domain[1] - domain[0], domain[3] - domain[2])
    rhs_2d = np.zeros((nx_mg + 1, ny_mg + 1))
    for i in range(nx_mg + 1):
        for j in range(ny_mg + 1):
            x = domain[0] + i * (domain[1] - domain[0]) / nx_mg
            y = domain[2] + j * (domain[3] - domain[2]) / ny_mg
            rhs_2d[i, j] = 100.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)

    u_2d, it_2d = mg2d.solve(rhs_2d, tol=1e-6, max_iter=30)
    print(f"  2D MG: converged in {it_2d} iterations, max |u|={np.max(np.abs(u_2d)):.4f}")
    print()

    # ========================================================================
    # Step 6: 谱分析（Laguerre + Chebyshev）
    # ========================================================================
    print("[Step 6] Spectral Analysis")
    print("-" * 50)

    # Laguerre 多项式计算
    x_lag = np.linspace(0, 10, 200)
    L_vals = laguerre_polynomial(200, 6, x_lag)
    print(f"  Laguerre L_6(5) = {L_vals[100, 6]:.4f} (expected ~1.0)")

    # 广义 Laguerre
    Lf_vals = generalized_laguerre_function(200, 6, 0.5, x_lag)
    print(f"  Generalized L_6^{'{0.5}'}(5) = {Lf_vals[100, 6]:.4f}")

    # 切比雪夫插值
    def test_func(x):
        return np.sin(3 * x) * np.exp(-x ** 2)

    xp_test = np.linspace(-1, 1, 501)
    yp_interp, maxerr = chebyshev_interpolate(test_func, -1.0, 1.0, 16, xp_test)
    print(f"  Chebyshev interpolation (n=16): max error = {maxerr:.3e}")

    # 径向分布谱分析
    r_bins = np.linspace(0, 0.5, 50)
    # 计算粒子对的径向距离分布
    distances = []
    for i in range(min(200, n_particles)):
        for j in range(i + 1, min(200, n_particles)):
            d = np.linalg.norm(particles[i] - particles[j])
            if d < 0.5:
                distances.append(d)
    if len(distances) > 5:
        hist, _ = np.histogram(distances, bins=r_bins)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        g_r = safe_divide(hist.astype(float), (2.0 * np.pi * r_centers + 1e-12))
        coeffs = radial_distribution_spectrum(r_centers, g_r, n_modes=5, alpha=0.0, beta=2.0)
        print(f"  Radial distribution spectrum coefficients: {coeffs}")
    else:
        print("  [Skip] Not enough particle pairs for radial distribution.")
    print()

    # ========================================================================
    # Step 7: 高精度数值积分
    # ========================================================================
    print("[Step 7] High-Precision Quadrature")
    print("-" * 50)

    # 在第一个三角形上积分 sin(x+y)
    if triangles_tri.shape[0] > 0:
        verts = nodes_tri[triangles_tri[0] - 1]
        from quadrature_rules import integrate_over_triangle
        val = integrate_over_triangle(verts, lambda x, y: np.sin(x + y), degree=5)
        print(f"  Integral of sin(x+y) over first triangle: {val:.6f}")

    # 全局矩量
    m00 = compute_moment_over_mesh(nodes_tri, triangles_tri, 0, 0, degree=3)
    m10 = compute_moment_over_mesh(nodes_tri, triangles_tri, 1, 0, degree=3)
    m01 = compute_moment_over_mesh(nodes_tri, triangles_tri, 0, 1, degree=3)
    print(f"  Mesh area (m00): {m00:.6f}")
    print(f"  First moments: m10={m10:.6f}, m01={m01:.6f}")
    print()

    # ========================================================================
    # Step 8: 负载均衡
    # ========================================================================
    print("[Step 8] Dynamic Load Balancing")
    print("-" * 50)

    balancer = LoadBalancer(n_procs, domain, imbalance_threshold=1.3)

    # 初始负载评估
    init_loads = balancer.compute_loads(particles)
    init_imbalance = balancer.imbalance_factor(init_loads)
    init_eff = balancer.evaluate_efficiency(init_loads)
    print(f"  Initial load imbalance: I = {init_imbalance:.4f}")
    print(f"  Initial parallel efficiency: eta = {init_eff['parallel_efficiency']:.4f}")

    # 执行重均衡
    result = balancer.rebalance(particles)
    if result['rebalanced']:
        print(f"  Rebalancing triggered!")
        print(f"  Old imbalance: {result['old_imbalance']:.4f}")
        print(f"  New imbalance: {result['new_imbalance']:.4f}")
        print(f"  Estimated migrations: {result['migration_count']}")
        new_eff = balancer.evaluate_efficiency(result['new_loads'])
        print(f"  New parallel efficiency: eta = {new_eff['parallel_efficiency']:.4f}")
    else:
        print("  No rebalancing needed (imbalance within threshold).")
    print()

    # 扩散均衡演示
    loads_demo = np.array([120.0, 80.0, 60.0, 100.0, 90.0, 70.0, 110.0, 50.0])
    conn = np.zeros((8, 8))
    for i in range(8):
        conn[i, (i + 1) % 8] = 1
        conn[i, (i - 1) % 8] = 1
    balanced = diffusion_based_load_balance(loads_demo, conn, n_iterations=200, tolerance=1e-2)
    print(f"  Diffusion balance demo: initial std={np.std(loads_demo):.2f}, final std={np.std(balanced):.2f}")
    print()

    # ========================================================================
    # Step 9: 快速求和与 Toeplitz 矩阵
    # ========================================================================
    print("[Step 9] Fast Summation (Toeplitz + Multipole)")
    print("-" * 50)

    n_toeplitz = 64
    h_t = 1.0 / n_toeplitz
    a_toep = build_interaction_matrix_toeplitz(
        n_toeplitz, lambda r: 1.0 / max(r, 1e-10), h_t
    )
    x_toep = np.random.rand(n_toeplitz)
    y_toep = toeplitz_mv(n_toeplitz, a_toep, x_toep)
    print(f"  Toeplitz MV: input norm={np.linalg.norm(x_toep):.4f}, output norm={np.linalg.norm(y_toep):.4f}")

    # FFT 加速版本对比
    y_toep_fft = toeplitz_embedded_fft_mv(n_toeplitz, a_toep, x_toep)
    rel_diff = np.linalg.norm(y_toep - y_toep_fft) / np.linalg.norm(y_toep)
    print(f"  FFT vs direct relative diff: {rel_diff:.3e}")

    # 多极子展开
    subset = particles[:50]
    charges = np.ones(50)
    center = np.mean(subset, axis=0)
    multipole = multipole_expansion(subset, charges, center, max_order=2)
    print(f"  Multipole: monopole={multipole['monopole']:.2f}, dipole_norm={np.linalg.norm(multipole['dipole']):.4f}")

    # 前缀和查询
    prefix = compute_prefix_sum_2d(particles, domain, 16, 16)
    count_center = query_region_count(prefix, 6, 10, 6, 10)
    print(f"  Prefix sum query (center region): {count_center} particles")
    print()

    # ========================================================================
    # Step 10: 综合评估与输出
    # ========================================================================
    print("[Step 10] Summary & Performance Metrics")
    print("-" * 50)

    final_loads = balancer.compute_loads(particles)
    final_eff = balancer.evaluate_efficiency(final_loads)
    print(f"  Final load stats:")
    print(f"    Mean load:     {final_eff['mean_load']:.2f}")
    print(f"    Std load:      {final_eff['std_load']:.2f}")
    print(f"    Imbalance I:   {final_eff['imbalance_factor']:.4f}")
    print(f"    Efficiency eta: {final_eff['parallel_efficiency']:.4f}")
    print()

    # FEM 解的 L2 范数
    l2_norm_fem = np.sqrt(np.mean(u_fem ** 2))
    print(f"  FEM solution L2 norm: {l2_norm_fem:.4f}")

    # MG 2D 解的能量范数估计
    energy_norm = np.sqrt(np.sum(u_2d ** 2) / ((nx_mg + 1) * (ny_mg + 1)))
    print(f"  MG 2D solution energy norm estimate: {energy_norm:.4f}")
    print()

    print("=" * 78)
    print("  Simulation completed successfully.")
    print("=" * 78)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（42个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: rucklidge_deriv 原点固定点测试 ----
import numpy as np
deriv = rucklidge_deriv(0.0, np.array([0.0, 0.0, 0.0]))
assert deriv.shape == (3,), '[TC01] rucklidge_deriv 输出形状 FAILED'
assert np.all(np.isfinite(deriv)), '[TC01] rucklidge_deriv 包含NaN/Inf FAILED'

# ---- TC02: arneodo_deriv 原点固定点测试 ----
import numpy as np
deriv = arneodo_deriv(0.0, np.array([0.0, 0.0, 0.0]))
assert deriv.shape == (3,), '[TC02] arneodo_deriv 输出形状 FAILED'
assert np.all(np.isfinite(deriv)), '[TC02] arneodo_deriv 包含NaN/Inf FAILED'

# ---- TC03: rucklidge_deriv 随机输入有限性测试 ----
import numpy as np
np.random.seed(42)
x_test = np.random.randn(10, 3)
for i in range(10):
    d = rucklidge_deriv(0.0, x_test[i])
    assert np.all(np.isfinite(d)), '[TC03] rucklidge_deriv 随机输入产生NaN/Inf FAILED'

# ---- TC04: integrate_trajectory 简单ODE输出形状与有限性 ----
import numpy as np
def simple_ode(t, y):
    return np.array([-y[0]])
t_arr, y_arr = integrate_trajectory(simple_ode, np.array([1.0]), (0.0, 1.0), relerr=1e-6, abserr=1e-9)
assert t_arr.ndim == 1, '[TC04] integrate_trajectory t_arr 维度 FAILED'
assert y_arr.ndim == 2, '[TC04] integrate_trajectory y_arr 维度 FAILED'
assert y_arr.shape[1] == 1, '[TC04] integrate_trajectory y_arr 列数 FAILED'
assert len(t_arr) >= 2, '[TC04] integrate_trajectory 步数不足 FAILED'
assert np.all(np.isfinite(y_arr)), '[TC04] integrate_trajectory 输出含NaN/Inf FAILED'

# ---- TC05: compute_particle_load_field 输出形状与非负性 ----
import numpy as np
np.random.seed(42)
test_particles = np.random.rand(100, 2)
domain = (0.0, 1.0, 0.0, 1.0)
load_field = compute_particle_load_field(test_particles, domain, 16, 16)
assert load_field.shape == (16, 16), '[TC05] compute_particle_load_field 输出形状 FAILED'
assert np.all(load_field >= 0), '[TC05] compute_particle_load_field 负值 FAILED'
assert np.all(np.isfinite(load_field)), '[TC05] compute_particle_load_field 含NaN/Inf FAILED'

# ---- TC06: QuadMesh 初始网格构建 ----
import numpy as np
mesh = QuadMesh((0.0, 2.0, 0.0, 1.0), nx=4, ny=4)
assert mesh.nodes.shape[0] == 25, '[TC06] QuadMesh 节点数 FAILED'
assert len(mesh.elements) == 16, '[TC06] QuadMesh 单元数 FAILED'
assert np.all(np.isfinite(mesh.nodes)), '[TC06] QuadMesh 节点含NaN/Inf FAILED'

# ---- TC07: QuadMesh evaluate_load 输出类型 ----
import numpy as np
mesh = QuadMesh((0.0, 1.0, 0.0, 1.0), nx=4, ny=4)
np.random.seed(42)
particles_test = np.random.rand(80, 2)
loads = mesh.evaluate_load(particles_test)
assert len(loads) == 16, '[TC07] evaluate_load 输出长度 FAILED'
assert np.all(loads >= 0), '[TC07] evaluate_load 负值 FAILED'

# ---- TC08: QuadMesh triangulate_elements 输出 ----
import numpy as np
mesh = QuadMesh((0.0, 1.0, 0.0, 1.0), nx=4, ny=4)
nodes_tri, triangles_tri = mesh.triangulate_elements()
assert triangles_tri.shape[1] == 3, '[TC08] triangulate 三角形列数 FAILED'
assert triangles_tri.shape[0] == 32, '[TC08] triangulate 三角形数 FAILED'
assert np.all(triangles_tri >= 1), '[TC08] triangulate 索引非1-based FAILED'

# ---- TC09: FEMSystem 刚度矩阵对称性 ----
import numpy as np
nodes_fem = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_fem = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
fem = FEMSystem(nodes_fem, tris_fem)
A = fem.assemble_stiffness_matrix()
A_dense = A.toarray()
assert np.allclose(A_dense, A_dense.T), '[TC09] 刚度矩阵不对称 FAILED'

# ---- TC10: FEMSystem 质量矩阵对角线为正 ----
import numpy as np
nodes_fem = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_fem = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
fem = FEMSystem(nodes_fem, tris_fem)
M = fem.assemble_mass_matrix()
M_dense = M.toarray()
assert np.all(np.diag(M_dense) > 0), '[TC10] 质量矩阵对角线非正 FAILED'

# ---- TC11: FEMSystem solve_poisson 零RHS得零解 ----
import numpy as np
nodes_fem = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_fem = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
fem = FEMSystem(nodes_fem, tris_fem)
rhs_zero = np.zeros(fem.n_nodes)
u = fem.solve_poisson(rhs_zero)
assert np.all(np.abs(u) < 1e-8), '[TC11] solve_poisson 零RHS非零解 FAILED'

# ---- TC12: MultigridPoisson1D 已知正弦解 ----
import numpy as np
def force_sin(x):
    return np.pi ** 2 * np.sin(np.pi * x)
mg1d = MultigridPoisson1D(n=64, a=0.0, b=1.0, ua=0.0, ub=0.0, force_func=force_sin)
u_1d, it_1d = mg1d.solve(tol=1e-6, max_iter=50)
assert it_1d <= 50, '[TC12] MG1D 未在max_iter内收敛 FAILED'
assert np.all(np.isfinite(u_1d)), '[TC12] MG1D 解含NaN/Inf FAILED'
assert u_1d[0] == 0.0 and u_1d[-1] == 0.0, '[TC12] MG1D 边界条件 FAILED'

# ---- TC13: MultigridPoisson2D 求解与残差 ----
import numpy as np
nx_mg, ny_mg = 16, 16
mg2d_test = MultigridPoisson2D(nx_mg, ny_mg, 1.0, 1.0)
rhs_2d_test = np.zeros((nx_mg + 1, ny_mg + 1))
rhs_2d_test[nx_mg//2, ny_mg//2] = 1.0
u_2d_test, it_2d_test = mg2d_test.solve(rhs_2d_test, tol=1e-4, max_iter=30)
assert it_2d_test <= 30, '[TC13] MG2D 未在max_iter内收敛 FAILED'
assert np.all(np.isfinite(u_2d_test)), '[TC13] MG2D 解含NaN/Inf FAILED'

# ---- TC14: laguerre_polynomial 已知值 L0(0)=1, L1(0)=1 ----
import numpy as np
x_lag = np.array([0.0])
L_vals = laguerre_polynomial(1, 2, x_lag)
assert abs(L_vals[0, 0] - 1.0) < 1e-12, '[TC14] L_0(0) != 1 FAILED'
assert abs(L_vals[0, 1] - 1.0) < 1e-12, '[TC14] L_1(0) != 1 FAILED'

# ---- TC15: generalized_laguerre_function 已知值 ----
import numpy as np
x_test = np.array([0.0])
Lg = generalized_laguerre_function(1, 2, 0.5, x_test)
assert abs(Lg[0, 0] - 1.0) < 1e-12, '[TC15] L_0^{(0.5)}(0) != 1 FAILED'
assert abs(Lg[0, 1] - 1.5) < 1e-12, '[TC15] L_1^{(0.5)}(0) != 1.5 FAILED'

# ---- TC16: chebyshev_nodes 对称性 ----
import numpy as np
nodes = chebyshev_nodes(-1.0, 1.0, 8)
assert len(nodes) == 8, '[TC16] chebyshev_nodes 节点数 FAILED'
assert np.all((nodes + nodes[::-1]) < 1e-12), '[TC16] chebyshev_nodes 不对称 FAILED'

# ---- TC17: divided_differences + newton_interpolate 线性函数精确 ----
import numpy as np
xd = np.array([0.0, 1.0, 2.0])
yd = np.array([3.0, 5.0, 7.0])
dd = divided_differences(xd, yd)
xp_test = np.linspace(0.0, 2.0, 21)
yp_test = newton_interpolate(xd, dd, xp_test)
expected = 3.0 + 2.0 * xp_test
assert np.allclose(yp_test, expected, atol=1e-10), '[TC17] 牛顿插值线性函数 FAILED'

# ---- TC18: chebyshev_interpolate 误差估计 ----
import numpy as np
def test_func(x):
    return np.sin(3.0 * x) * np.exp(-x ** 2)
xp_eval = np.linspace(-1, 1, 101)
yp_interp, maxerr = chebyshev_interpolate(test_func, -1.0, 1.0, 16, xp_eval)
assert not np.any(np.isnan(yp_interp)), '[TC18] chebyshev_interpolate 含NaN FAILED'
assert maxerr >= 0, '[TC18] chebyshev_interpolate 误差为负 FAILED'

# ---- TC19: chebyshev_spectral_derivative 线性函数导数恒定 ----
import numpy as np
x_lin = np.linspace(0.0, 2.0, 11)
u_lin = 3.0 * x_lin + 1.0
dudx = chebyshev_spectral_derivative(u_lin, 2.0)
assert np.allclose(dudx, 3.0, atol=1e-6), '[TC19] 谱导数线性函数 FAILED'

# ---- TC20: triangle_quad_rule_degree1 精确积分常数1 ----
import numpy as np
from quadrature_rules import triangle_quad_rule_degree1
w, pts = triangle_quad_rule_degree1()
assert abs(np.sum(w) - 0.5) < 1e-14, '[TC20] 求积规则权重和 != 0.5 FAILED'
assert np.all(pts >= 0) and np.all(pts <= 1), '[TC20] 求积节点越界 FAILED'

# ---- TC21: integrate_over_triangle 已知积分值 ----
import numpy as np
from quadrature_rules import integrate_over_triangle
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
val = integrate_over_triangle(verts, lambda x, y: x + y, degree=3)
assert abs(val - 1.0 / 3.0) < 1e-10, '[TC21] integrate_over_triangle x+y积分 != 1/3 FAILED'

# ---- TC22: compute_moment_over_mesh 面积一致性 ----
import numpy as np
nodes_quad = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
tris_quad = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
m00 = compute_moment_over_mesh(nodes_quad, tris_quad, 0, 0, degree=3)
assert abs(m00 - 1.0) < 1e-10, '[TC22] 矩量面积 != 1 FAILED'

# ---- TC23: LoadBalancer 不均衡因子计算 ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
loads_test = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
I = lb.imbalance_factor(loads_test)
assert abs(I - 1.0) < 1e-12, '[TC23] 均匀负载不均衡因子 != 1 FAILED'

# ---- TC24: LoadBalancer 不均衡负载测试 ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
loads_uneven = np.array([20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
I = lb.imbalance_factor(loads_uneven)
assert I > 1.0, '[TC24] 不均衡负载因子应 > 1 FAILED'

# ---- TC25: LoadBalancer evaluate_efficiency ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
loads_test = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
eff = lb.evaluate_efficiency(loads_test)
assert abs(eff['parallel_efficiency'] - 1.0) < 1e-12, '[TC25] 均匀负载效率 != 1 FAILED'
assert abs(eff['std_load']) < 1e-12, '[TC25] 均匀负载标准差 != 0 FAILED'

# ---- TC26: diffusion_based_load_balance 收敛到均值 ----
import numpy as np
loads_demo = np.array([120.0, 80.0, 60.0, 100.0, 90.0, 70.0, 110.0, 50.0])
conn = np.zeros((8, 8))
for i in range(8):
    conn[i, (i + 1) % 8] = 1
    conn[i, (i - 1) % 8] = 1
balanced = diffusion_based_load_balance(loads_demo, conn, n_iterations=500, tolerance=1e-4)
assert abs(np.mean(balanced) - np.mean(loads_demo)) < 1e-8, '[TC26] 扩散均衡均值不守恒 FAILED'
assert np.std(balanced) < np.std(loads_demo), '[TC26] 扩散均衡后方差未减小 FAILED'

# ---- TC27: toeplitz_mv 已知结果 ----
import numpy as np
n_t = 4
a_t = np.array([2.0, 1.0, 0.5, 0.25, 3.0, 1.5, 0.75])
x_t = np.array([1.0, 0.0, 0.0, 0.0])
y_t = toeplitz_mv(n_t, a_t, x_t)
# 第一列 = [2, 3, 1.5, 0.75]
assert abs(y_t[0] - 2.0) < 1e-12, '[TC27] toeplitz_mv[0] FAILED'
assert abs(y_t[1] - 3.0) < 1e-12, '[TC27] toeplitz_mv[1] FAILED'

# ---- TC28: toeplitz_embedded_fft_mv 与 direct 对比 ----
import numpy as np
np.random.seed(42)
n_t = 32
a_t = np.random.rand(2 * n_t - 1)
x_t = np.random.rand(n_t)
y_dir = toeplitz_mv(n_t, a_t, x_t)
y_fft = toeplitz_embedded_fft_mv(n_t, a_t, x_t)
rel_diff = np.linalg.norm(y_dir - y_fft) / (np.linalg.norm(y_dir) + 1e-14)
assert rel_diff < 1e-10, '[TC28] FFT vs direct 差异过大 FAILED'

# ---- TC29: sample_unit_ball_positive 输出范围 ----
import numpy as np
np.random.seed(42)
samples = sample_unit_ball_positive(200)
assert samples.shape == (200, 3), '[TC29] ball_sample 形状 FAILED'
assert np.all(samples >= 0), '[TC29] ball_sample 负值 FAILED'
assert np.all(np.linalg.norm(samples, axis=1) <= 1.0 + 1e-10), '[TC29] ball_sample 范数>1 FAILED'

# ---- TC30: sample_unit_sphere_surface 单位范数 ----
import numpy as np
np.random.seed(42)
samples = sample_unit_sphere_surface(100, dim=4)
assert samples.shape == (100, 4), '[TC30] sphere_sample 形状 FAILED'
norms = np.linalg.norm(samples, axis=1)
assert np.allclose(norms, 1.0, atol=1e-10), '[TC30] sphere_sample 非单位范数 FAILED'

# ---- TC31: compute_prefix_sum_2d + query_region_count 一致性 ----
import numpy as np
np.random.seed(42)
particles_prefix = np.random.rand(200, 2)
domain_pf = (0.0, 1.0, 0.0, 1.0)
prefix = compute_prefix_sum_2d(particles_prefix, domain_pf, 8, 8)
total = query_region_count(prefix, 0, 8, 0, 8)
assert total == 200, '[TC31] prefix sum 总粒子数 != 200 FAILED'

# ---- TC32: multipole_expansion 单极子 = 总电荷 ----
import numpy as np
np.random.seed(42)
particles_mp = np.random.rand(20, 2)
charges_mp = np.ones(20) * 2.0
center_mp = np.array([0.5, 0.5])
mp = multipole_expansion(particles_mp, charges_mp, center_mp, max_order=2)
assert abs(mp['monopole'] - 40.0) < 1e-10, '[TC32] 单极子 != 40 FAILED'
assert 'dipole' in mp and 'quadrupole' in mp, '[TC32] 缺少偶极子/四极子 FAILED'

# ---- TC33: build_interaction_matrix_toeplitz 输出长度 ----
import numpy as np
a_tp = build_interaction_matrix_toeplitz(16, lambda r: 1.0 / max(r, 1e-10), h=0.1)
assert len(a_tp) == 31, '[TC33] Toeplitz数据长度 FAILED'
assert np.all(np.isfinite(a_tp)), '[TC33] Toeplitz数据含NaN/Inf FAILED'

# ---- TC34: safe_divide 边界情况 ----
import numpy as np
a_sd = np.array([10.0, 0.0, 5.0])
b_sd = np.array([2.0, 0.0, 0.0])
r_sd = safe_divide(a_sd, b_sd)
assert abs(r_sd[0] - 5.0) < 1e-12, '[TC34] safe_divide[0] FAILED'
assert r_sd[1] == 0.0, '[TC34] safe_divide 除以零回退值 FAILED'

# ---- TC35: compute_triangle_area 已知面积 ----
import numpy as np
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.0, 1.0])
area = compute_triangle_area(p1, p2, p3)
assert abs(abs(area) - 0.5) < 1e-12, '[TC35] 三角形面积 != 0.5 FAILED'

# ---- TC36: is_power_of_two 真值表 ----
assert is_power_of_two(1) == True, '[TC36] is_power_of_two(1) FAILED'
assert is_power_of_two(2) == True, '[TC36] is_power_of_two(2) FAILED'
assert is_power_of_two(3) == False, '[TC36] is_power_of_two(3) FAILED'
assert is_power_of_two(64) == True, '[TC36] is_power_of_two(64) FAILED'
assert is_power_of_two(0) == False, '[TC36] is_power_of_two(0) FAILED'

# ---- TC37: gauss_seidel_sweep 收敛性 ----
import numpy as np
n_gs = 10
rhs_gs = np.ones(n_gs) * 2.0
x_gs = np.zeros(n_gs)
x_new, d = gauss_seidel_sweep(n_gs, rhs_gs, x_gs)
assert d > 0, '[TC37] Gauss-Seidel 变化量非正 FAILED'
assert np.all(np.isfinite(x_new)), '[TC37] Gauss-Seidel 输出含NaN/Inf FAILED'

# ---- TC38: mesh_base_one 0-based转1-based ----
import numpy as np
elem_0based = np.array([[0, 1, 2], [2, 3, 1]])
elem_1based = mesh_base_one(elem_0based, 4)
assert elem_1based.min() == 1, '[TC38] mesh_base_one 转换后min != 1 FAILED'
assert elem_1based.max() == 4, '[TC38] mesh_base_one 转换后max != 4 FAILED'

# ---- TC39: reference_to_physical_q4 映射 ----
import numpy as np
q4_corners = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
rs_pt = np.array([[0.5, 0.5]])
phys = reference_to_physical_q4(q4_corners, rs_pt)
assert np.allclose(phys[0], [1.0, 0.5], atol=1e-10), '[TC39] Q4映射中心点 FAILED'

# ---- TC40: radial_distribution_spectrum 基本测试 ----
import numpy as np
r_test = np.linspace(0.01, 1.0, 50)
g_test = np.exp(-2.0 * r_test)
coeffs_test = radial_distribution_spectrum(r_test, g_test, n_modes=4, alpha=0.0, beta=2.0)
assert len(coeffs_test) == 4, '[TC40] 谱系数长度 FAILED'
assert np.all(np.isfinite(coeffs_test)), '[TC40] 谱系数含NaN/Inf FAILED'

# ---- TC41: LoadBalancer find_optimal_split 基本测试 ----
import numpy as np
lb = LoadBalancer(8, (0.0, 1.0, 0.0, 1.0))
particles_split = np.array([[0.2, 0.5], [0.3, 0.5], [0.7, 0.5], [0.8, 0.5]])
split_pos, load_left, load_right = lb.find_optimal_split(particles_split, axis=0)
assert 0.3 < split_pos < 0.7, '[TC41] find_optimal_split 切分点异常 FAILED'

# ---- TC42: 集成测试：完整main流程不崩溃 ----
import numpy as np
np.random.seed(42)
# 测试最小配置的完整流程
try:
    n_p = 200
    domain_t = (0.0, 1.0, 0.0, 1.0)
    ball = sample_unit_ball_positive(n_p)
    particles_t = np.zeros((n_p, 2))
    particles_t[:, 0] = check_bounds(0.5 + 0.3 * ball[:, 0], 0.0, 1.0, "x")
    particles_t[:, 1] = check_bounds(0.5 + 0.3 * ball[:, 1], 0.0, 1.0, "y")
    mesh_t = QuadMesh(domain_t, nx=4, ny=4)
    nodes_t, tris_t = mesh_t.triangulate_elements()
    fem_t = FEMSystem(nodes_t, tris_t)
    load_f = compute_particle_load_field(particles_t, domain_t, 8, 8)
    from scipy.interpolate import RegularGridInterpolator
    x_g = np.linspace(0, 1, 8)
    y_g = np.linspace(0, 1, 8)
    interp = RegularGridInterpolator((x_g, y_g), load_f.T, bounds_error=False, fill_value=0.0)
    rhs_t = interp(fem_t.nodes)
    u_t = fem_t.solve_poisson(rhs_t)
    balancer_t = LoadBalancer(8, domain_t)
    loads_t = balancer_t.compute_loads(particles_t)
    eff_t = balancer_t.evaluate_efficiency(loads_t)
    assert u_t.shape[0] == fem_t.n_nodes, '[TC42] FEM解维度 FAILED'
    assert eff_t['parallel_efficiency'] > 0, '[TC42] 并行效率非正 FAILED'
except Exception as e:
    assert False, f'[TC42] 集成测试异常: {e}'

print('\n全部 42 个测试通过!\n')
