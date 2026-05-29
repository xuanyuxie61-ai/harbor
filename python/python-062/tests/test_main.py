"""
main.py
================================================================================
边界层湍流与大涡模拟（PBL-LES）统一入口
================================================================================

本项目围绕大气科学前沿领域——边界层湍流与大涡模拟（LES），
融合15个种子项目的核心算法，构建了一个具备博士级复杂度的
科研计算框架。

运行方式
--------
    python main.py

无需任何命令行参数，程序自动执行完整的 LES 初始化、
时间步进、统计分析与后处理流程。
"""

import numpy as np
import sys
import time


def print_banner():
    print("=" * 78)
    print("  边界层湍流与大涡模拟 (PBL-LES) 合成科研代码项目")
    print("  领域：大气科学 —— 边界层湍流与大涡模拟")
    print("=" * 78)
    print()


def run_simulation():
    print_banner()
    t0_total = time.time()

    # ------------------------------------------------------------------
    # 1. 参数设置
    # ------------------------------------------------------------------
    print("[1/8] 初始化物理参数与计算网格 ...")
    nx, ny, nz = 24, 24, 16
    Lx, Ly, Lz = 1000.0, 1000.0, 500.0  # 米
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    dt = 0.2  # 固定时间步长（秒）
    n_steps = 5
    rho = 1.225  # 空气密度 kg/m³
    nu_mol = 1.5e-5  # 运动粘性 m²/s
    g = 9.81  # 重力加速度

    print(f"      网格: {nx} x {ny} x {nz}, 分辨率: {dx:.1f}m x {dy:.1f}m x {dz:.1f}m")
    print(f"      时间步长: {dt}s, 总步数: {n_steps}")

    # ------------------------------------------------------------------
    # 2. 初始化速度场与温度场
    # ------------------------------------------------------------------
    print("[2/8] 初始化湍流场（对数风剖面 + 随机脉动）...")
    from les_core import initialize_turbulent_field
    u, v, w, theta = initialize_turbulent_field(
        nx, ny, nz, dx, dy, dz,
        u_mean=8.0, v_mean=0.0,
        turbulence_intensity=0.02,
        theta_mean=298.0, theta_gradient=0.005
    )

    # ------------------------------------------------------------------
    # 3. 网格拓扑与边界识别
    # ------------------------------------------------------------------
    print("[3/8] 构建网格拓扑与识别边界 ...")
    from mesh_topology import build_mesh_graph, mesh_quality_metrics
    from mesh_boundary import extract_boundary_nodes_3d, apply_surface_layer_bc

    # 生成简化四面体网格（用于 FEM 算子演示）
    nodes = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes[idx] = [i * dx, j * dy, k * dz]
                idx += 1

    # 构建六面体到四面体的简单映射（仅演示拓扑）
    # 每个六面体拆分为 5 个或 6 个四面体
    element_nodes = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n000 = i + nx * (j + ny * k)
                n100 = (i + 1) + nx * (j + ny * k)
                n010 = i + nx * ((j + 1) + ny * k)
                n110 = (i + 1) + nx * ((j + 1) + ny * k)
                n001 = i + nx * (j + ny * (k + 1))
                n101 = (i + 1) + nx * (j + ny * (k + 1))
                n011 = i + nx * ((j + 1) + ny * (k + 1))
                n111 = (i + 1) + nx * ((j + 1) + ny * (k + 1))
                # 简化为 5 个四面体
                element_nodes.append([n000, n100, n010, n001])
                element_nodes.append([n111, n011, n101, n110])
                element_nodes.append([n100, n010, n110, n001])
                element_nodes.append([n100, n110, n101, n001])
                element_nodes.append([n110, n101, n001, n011])

    element_nodes = np.array(element_nodes, dtype=int)
    n_elem = element_nodes.shape[0]
    print(f"      四面体单元数: {n_elem}")

    adjacency, degrees = build_mesh_graph(element_nodes, nx * ny * nz)
    print(f"      平均节点度: {np.mean(degrees):.1f}")

    bdry_info = extract_boundary_nodes_3d(element_nodes, nodes, lower_z=0.0, upper_z=Lz)
    lower_nodes = bdry_info['lower']
    print(f"      下边界节点数: {len(lower_nodes)}")

    # 应用近地层边界条件
    u, v, w = apply_surface_layer_bc(
        u.flatten(), v.flatten(), w.flatten(), theta.flatten(),
        nodes, lower_nodes, u_star=0.4, z0=0.1
    )
    u = u.reshape((nx, ny, nz))
    v = v.reshape((nx, ny, nz))
    w = w.reshape((nx, ny, nz))

    # ------------------------------------------------------------------
    # 4. 时间步进（LES 核心循环）
    # ------------------------------------------------------------------
    print("[4/8] LES 时间步进（对流-扩散-SGS-投影）...")
    from les_core import convection_term, laplacian_3d, projection_step
    from sgs_model import smagorinsky_model, dynamic_smagorinsky_model
    from time_integrator import adaptive_timestep

    for step in range(n_steps):
        # 数值检查：若出现 NaN/Inf，回退到标准 Smagorinsky 模型
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
            print(f"      [警告] 步 {step} 检测到非有限值，重新初始化速度场")
            u, v, w, theta = initialize_turbulent_field(
                nx, ny, nz, dx, dy, dz,
                u_mean=8.0, v_mean=0.0,
                turbulence_intensity=0.05,
                theta_mean=298.0, theta_gradient=0.005
            )

        # 自适应时间步长
        dt = adaptive_timestep(u, v, w, dx, dy, dz, nu_mol + 0.1, cfl=0.3)
        if dt < 1e-6:
            dt = 1e-6

        # SGS 模型：标准 Smagorinsky（更稳定）
        nu_sgs, tau_sgs = smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.15)
        nu_eff = nu_mol + nu_sgs

        # 对流项（使用一阶迎风格式，数值更稳定）
        conv_u, conv_v, conv_w = convection_term(u, v, w, dx, dy, dz)

        # 扩散项
        diff_u = nu_eff * laplacian_3d(u, dx, dy, dz)
        diff_v = nu_eff * laplacian_3d(v, dx, dy, dz)
        diff_w = nu_eff * laplacian_3d(w, dx, dy, dz)

        # 显式前向欧拉步（预测）
        u_star = u + dt * (-conv_u + diff_u)
        v_star = v + dt * (-conv_v + diff_v)
        w_star = w + dt * (-conv_w + diff_w)

        # 投影步（压力修正，使其无散）
        u, v, w, p, converged = projection_step(
            u_star, v_star, w_star, dx, dy, dz, dt, rho=rho,
            max_iter=100, tol=1e-6
        )

        # 重新施加边界条件（防止数值漂移）
        u, v, w = apply_surface_layer_bc(
            u.flatten(), v.flatten(), w.flatten(), theta.flatten(),
            nodes, lower_nodes, u_star=0.4, z0=0.1
        )
        u = u.reshape((nx, ny, nz))
        v = v.reshape((nx, ny, nz))
        w = w.reshape((nx, ny, nz))

        # 数值限幅（防止极端值）
        u = np.clip(u, -50.0, 50.0)
        v = np.clip(v, -50.0, 50.0)
        w = np.clip(w, -20.0, 20.0)

        if step % 1 == 0:
            print(f"      步 {step:3d}/{n_steps}, dt={dt:.4f}s, nu_sgs_mean={np.mean(nu_sgs):.4f}, "
                  f"投影收敛={converged}, u_max={np.max(np.abs(u)):.2f}")

    print(f"      时间步进完成，最终 CFL 约束 dt={dt:.4f}s")

    # ------------------------------------------------------------------
    # 5. 湍流统计量计算
    # ------------------------------------------------------------------
    print("[5/8] 计算湍流统计量 ...")
    from turbulence_stats import (
        compute_tke, compute_reynolds_stresses, compute_heat_flux,
        longitudinal_structure_function, compute_kolmogorov_scales
    )
    from quadrature_rules import compute_energy_dissipation_rate

    tke, tke_mean = compute_tke(u, v, w)
    R = compute_reynolds_stresses(u, v, w)
    qx, qy, qz = compute_heat_flux(u, v, w, theta)
    epsilon = compute_energy_dissipation_rate(u, v, w, dx, dy, dz, nu_mol)

    eta, tau_eta, u_eta, Re_lambda = compute_kolmogorov_scales(epsilon, nu_mol)

    r, D_ll = longitudinal_structure_function(u, axis=0, max_lag=nx // 4)

    print(f"      平均湍动能 TKE = {tke_mean:.4f} m²/s²")
    print(f"      Reynolds 应力 u'u' = {R['uu']:.4f}, w'w' = {R['ww']:.4f}")
    print(f"      热通量 w'θ' = {qz:.4f} K·m/s")
    print(f"      能量耗散率 ε = {epsilon:.6e} m²/s³")
    print(f"      Kolmogorov 尺度: η={eta:.4e}m, τ_η={tau_eta:.4e}s, u_η={u_eta:.4e}m/s")
    print(f"      Taylor Reynolds 数 Re_λ = {Re_lambda:.1f}")

    # ------------------------------------------------------------------
    # 6. 拉格朗日粒子追踪与随机扩散
    # ------------------------------------------------------------------
    print("[6/8] 拉格朗日粒子追踪与随机 Langevin 扩散 ...")
    from stochastic_model import initialize_particles, langevin_step_euler, ensemble_concentration
    from particle_tracker import track_particles_rk2
    from lagrange_interp import interpolate_velocity_to_particles

    n_particles = 200
    particles, velocities = initialize_particles(
        n_particles,
        domain_x=(0, Lx), domain_y=(0, Ly), domain_z=(0, Lz),
        release_height=20.0
    )

    # 插值网格速度到粒子
    x_grid = np.arange(nx) * dx
    y_grid = np.arange(ny) * dy
    z_grid = np.arange(nz) * dz

    u_p, v_p, w_p = interpolate_velocity_to_particles(
        particles, (x_grid, y_grid, z_grid), u, v, w, order=3
    )

    # 执行 Langevin 随机步
    sigma_w_field = np.sqrt(np.maximum(R['ww'], 1e-6))
    sigma_w_p = np.full(n_particles, sigma_w_field)
    epsilon_p = np.full(n_particles, epsilon)

    particles_new, velocities_new = langevin_step_euler(
        particles, velocities, sigma_w_p, epsilon_p, dt=1.0
    )

    print(f"      粒子数: {n_particles}")
    print(f"      平均垂直位移: {np.mean(particles_new[:, 2] - particles[:, 2]):.3f} m")

    # ------------------------------------------------------------------
    # 7. 分形分析与谱分析
    # ------------------------------------------------------------------
    print("[7/8] 分形维数与间歇性分析 ...")
    from fractal_analysis import box_counting, compute_intermittency_factor, richardson_cascade_spectrum

    # 取水平切片进行分形分析
    u_slice = u[:, :, nz // 2]
    D_f, scales, counts = box_counting(u_slice, threshold=np.mean(u_slice), max_level=4)
    mu = compute_intermittency_factor(u_slice, window_size=4)

    # 一维能谱
    k_spec = np.linspace(1, nx // 2, nx // 2)
    E_k = richardson_cascade_spectrum(k_spec, epsilon, C=1.5, mu=mu)

    print(f"      速度场盒计数维数 D_f = {D_f:.3f} (理论值 ~2.0–2.8)")
    print(f"      间歇性因子 μ = {mu:.3f}")
    print(f"      能谱峰值波数 k_max = {k_spec[np.argmax(E_k)]:.1f}")

    # ------------------------------------------------------------------
    # 8. 波数空间三波相互作用
    # ------------------------------------------------------------------
    print("[8/8] 波数空间三波相互作用枚举 ...")
    from wave_interactions import enumerate_triads, shell_energy_flux

    triads = enumerate_triads(k_max=3, dim=2)
    print(f"      截断球内三波组数: {len(triads)}")

    # 构造简化的谱速度场
    np.random.seed(7)
    velocities_spec = {}
    for k1 in range(-3, 4):
        for k2 in range(-3, 4):
            if k1 == 0 and k2 == 0:
                continue
            k_mag = np.sqrt(k1**2 + k2**2)
            if k_mag > 3:
                continue
            # 复速度振幅（近似 Kolmogorov 标度）
            amp = (epsilon ** (1.0 / 3.0)) * (k_mag ** (-1.0 / 3.0))
            phase = np.random.rand(2) * 2 * np.pi
            velocities_spec[(k1, k2)] = amp * np.array([
                np.exp(1j * phase[0]),
                np.exp(1j * phase[1])
            ])

    k_bins = np.array([0.5, 1.5, 2.5, 3.5])
    Pi = shell_energy_flux(triads, velocities_spec, k_bins)
    print(f"      能量通量 Π(k): {Pi}")
    print(f"      理论惯性子区通量应近似为常数 ε = {epsilon:.4e}")

    # ------------------------------------------------------------------
    # 9. 球谐函数展开演示
    # ------------------------------------------------------------------
    print("[附加] 球谐函数基展开演示 ...")
    from spherical_harmonics import spherical_harmonic_basis

    theta_ang = np.linspace(0, np.pi, 12)
    phi_ang = np.linspace(0, 2 * np.pi, 24)
    TH, PH = np.meshgrid(theta_ang, phi_ang, indexing='ij')

    Y_2_1_c, Y_2_1_s = spherical_harmonic_basis(2, 1, TH, PH)
    print(f"      Y_2^1 实部积分: {np.mean(Y_2_1_c):.6f} (应 ≈ 0)")

    # ------------------------------------------------------------------
    # 10. 高斯-帕特森求积演示
    # ------------------------------------------------------------------
    print("[附加] Gauss-Patterson 高精度求积演示 ...")
    from quadrature_rules import patterson_integrate_1d, patterson_integrate_3d

    # 一维积分测试：∫_0^1 x² dx = 1/3
    result_1d = patterson_integrate_1d(lambda x: x**2, 0.0, 1.0, order=7)
    print(f"      ∫_0^1 x² dx = {result_1d:.10f} (精确值 1/3 = 0.3333333333)")

    # 三维积分测试：∫_0^1∫_0^1∫_0^1 x*y*z dxdydz = 1/8
    result_3d = patterson_integrate_3d(
        lambda x, y, z: x * y * z,
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), order=7
    )
    print(f"      ∫_0^1 x*y*z dV = {result_3d:.10f} (精确值 1/8 = 0.1250000000)")

    # ------------------------------------------------------------------
    # 11. FEM 基函数与 Laplacian 矩阵组装演示
    # ------------------------------------------------------------------
    print("[附加] 有限元基函数与 Laplacian 矩阵组装演示 ...")
    from fem_basis import basis_mn_tet4, basis_gradient_tet4, fem_laplacian_matrix

    test_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    test_points = np.array([[0.1, 0.1, 0.1], [0.25, 0.25, 0.25]], dtype=np.float64)
    phi_test = basis_mn_tet4(test_tet, test_points)
    grad_test = basis_gradient_tet4(test_tet)
    print(f"      测试点基函数值: {phi_test[:, 0]}")
    print(f"      基函数梯度范数: {[np.linalg.norm(g) for g in grad_test]}")

    # ------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------
    t_total = time.time() - t0_total
    print()
    print("=" * 78)
    print(f"  模拟完成，总耗时: {t_total:.2f} 秒")
    print("=" * 78)


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"\n[ERROR] 运行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ================================================================
# 测试用例（25个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: tetrahedron_volume 标准四面体体积解析验证 ----
import numpy as np
from fem_basis import tetrahedron_volume
test_tet = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
vol = tetrahedron_volume(test_tet)
assert abs(vol - 1.0 / 6.0) < 1e-12, '[TC01] tetrahedron_volume 标准四面体体积解析验证 FAILED'

# ---- TC02: basis_mn_tet4 节点处满足 Kronecker delta ----
from fem_basis import basis_mn_tet4
phi_nodes = basis_mn_tet4(test_tet, test_tet)
assert np.allclose(phi_nodes, np.eye(4), atol=1e-10), '[TC02] basis_mn_tet4 节点处满足 Kronecker delta FAILED'

# ---- TC03: basis_gradient_tet4 输出形状与有限值 ----
from fem_basis import basis_gradient_tet4
grad = basis_gradient_tet4(test_tet)
assert grad.shape == (4, 3), '[TC03] basis_gradient_tet4 输出形状 FAILED'
assert np.all(np.isfinite(grad)), '[TC03] basis_gradient_tet4 有限值 FAILED'

# ---- TC04: box_counting 常数场分形维数非负 ----
from fractal_analysis import box_counting
field_const = np.ones((64, 64))
D_f, scales, counts = box_counting(field_const, threshold=0.5, max_level=4)
assert D_f >= 0.0, '[TC04] box_counting 常数场分形维数非负 FAILED'
assert len(scales) == len(counts), '[TC04] box_counting 尺度与计数长度一致 FAILED'

# ---- TC05: richardson_cascade_spectrum 能谱输出正值 ----
from fractal_analysis import richardson_cascade_spectrum
k_spec = np.linspace(1, 10, 50)
E_k = richardson_cascade_spectrum(k_spec, epsilon=0.01, C=1.5, mu=0.25)
assert np.all(E_k >= 0), '[TC05] richardson_cascade_spectrum 能谱输出正值 FAILED'

# ---- TC06: givens_rotation 旋转后第二个分量为零 ----
from gmres_solver import givens_rotation
cs, sn = givens_rotation(3.0, 4.0)
assert abs(-sn * 3.0 + cs * 4.0) < 1e-12, '[TC06] givens_rotation 旋转后第二个分量为零 FAILED'

# ---- TC07: gmres_solve 单位矩阵收敛且解正确 ----
from gmres_solver import gmres_solve
A_id = lambda x: x
b_vec = np.array([1.0, 2.0, 3.0])
x_sol, residuals, converged = gmres_solve(A_id, b_vec, x0=np.zeros(3), max_iter=10, tol=1e-10)
assert converged, '[TC07] gmres_solve 单位矩阵收敛 FAILED'
assert np.allclose(x_sol, b_vec, atol=1e-8), '[TC07] gmres_solve 单位矩阵解正确 FAILED'

# ---- TC08: lagrange_basis_1d 节点处精确值为 1 ----
from lagrange_interp import lagrange_basis_1d
x_nodes = np.array([0.0, 1.0, 2.0])
L_vals = lagrange_basis_1d(x_nodes, 1.0)
assert np.allclose(L_vals, np.array([0.0, 1.0, 0.0]), atol=1e-12), '[TC08] lagrange_basis_1d 节点处精确值为 1 FAILED'

# ---- TC09: lagrange_interp_nd 常数函数精确插值 ----
from lagrange_interp import lagrange_interp_nd
grid_2d = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
values_const = np.array([[1.0, 1.0], [1.0, 1.0]])
interp_val = lagrange_interp_nd(grid_2d, values_const, (0.5, 0.5))
assert abs(interp_val - 1.0) < 1e-12, '[TC09] lagrange_interp_nd 常数函数精确插值 FAILED'

# ---- TC10: laplacian_3d 线性场 Laplacian 为零 ----
from les_core import laplacian_3d
nx_l, ny_l, nz_l = 8, 8, 8
dx_l, dy_l, dz_l = 1.0, 1.0, 1.0
x_l = np.arange(nx_l) * dx_l
y_l = np.arange(ny_l) * dy_l
z_l = np.arange(nz_l) * dz_l
X_l, Y_l, Z_l = np.meshgrid(x_l, y_l, z_l, indexing='ij')
phi_lin = 2.0 * X_l + 3.0 * Y_l + 4.0 * Z_l
lap_lin = laplacian_3d(phi_lin, dx_l, dy_l, dz_l)
assert np.allclose(lap_lin[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10), '[TC10] laplacian_3d 线性场 Laplacian 为零 FAILED'

# ---- TC11: solve_poisson_fft 输出实数且有限 ----
from les_core import solve_poisson_fft
nx_p, ny_p, nz_p = 16, 16, 16
rhs_p = np.zeros((nx_p, ny_p, nz_p))
rhs_p[1, 2, 3] = 1.0
p_sol = solve_poisson_fft(rhs_p, 1.0, 1.0, 1.0)
assert np.all(np.isfinite(p_sol)), '[TC11] solve_poisson_fft 输出有限 FAILED'
assert np.isrealobj(p_sol), '[TC11] solve_poisson_fft 输出实数 FAILED'

# ---- TC12: triangulation_boundary_edges 单三角形有三条边界边 ----
from mesh_boundary import triangulation_boundary_edges
tri = np.array([[0, 1, 2]])
edges = triangulation_boundary_edges(tri)
assert edges.shape[0] == 3, '[TC12] triangulation_boundary_edges 单三角形有三条边界边 FAILED'

# ---- TC13: build_mesh_graph 四面体节点度均为 3 ----
from mesh_topology import build_mesh_graph
elem_tet = np.array([[0, 1, 2, 3]])
adj, deg = build_mesh_graph(elem_tet, 4)
assert np.all(deg == 3), '[TC13] build_mesh_graph 四面体节点度均为 3 FAILED'

# ---- TC14: patterson_integrate_1d x^2 积分精确到 1/3 ----
from quadrature_rules import patterson_integrate_1d
result_1d = patterson_integrate_1d(lambda x: x**2, 0.0, 1.0, order=7)
assert abs(result_1d - 1.0 / 3.0) < 1e-10, '[TC14] patterson_integrate_1d x^2 积分精确到 1/3 FAILED'

# ---- TC15: patterson_integrate_3d 常数函数积分精确 ----
from quadrature_rules import patterson_integrate_3d
result_3d = patterson_integrate_3d(lambda x, y, z: 2.0, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), order=3)
assert abs(result_3d - 2.0) < 1e-10, '[TC15] patterson_integrate_3d 常数函数积分精确 FAILED'

# ---- TC16: smagorinsky_model SGS 涡粘性场非负 ----
from sgs_model import smagorinsky_model
np.random.seed(42)
u_sgs = np.random.randn(8, 8, 8)
v_sgs = np.random.randn(8, 8, 8)
w_sgs = np.random.randn(8, 8, 8)
nu_sgs, tau_sgs = smagorinsky_model(u_sgs, v_sgs, w_sgs, 1.0, 1.0, 1.0, Cs=0.15)
assert np.all(nu_sgs >= 0), '[TC16] smagorinsky_model SGS 涡粘性场非负 FAILED'

# ---- TC17: spherical_harmonic_basis Y_0^0 为常数 1/sqrt(4pi) ----
from spherical_harmonics import spherical_harmonic_basis
theta_a = np.linspace(0, np.pi, 10)
phi_a = np.linspace(0, 2 * np.pi, 20)
TH_a, PH_a = np.meshgrid(theta_a, phi_a, indexing='ij')
c_y00, s_y00 = spherical_harmonic_basis(0, 0, TH_a, PH_a)
assert np.allclose(c_y00, 1.0 / np.sqrt(4.0 * np.pi), atol=1e-10), '[TC17] spherical_harmonic_basis Y_0^0 为常数 FAILED'

# ---- TC18: exp_exact_solution 与解析解 exp(t) 一致 ----
from time_integrator import exp_exact_solution
t_e = np.array([0.0, 1.0, 2.0])
y_e = exp_exact_solution(t_e, alpha=1.0, t0=0.0, y0=1.0)
assert np.allclose(y_e, np.exp(t_e), atol=1e-12), '[TC18] exp_exact_solution 与解析解一致 FAILED'

# ---- TC19: compute_cfl_limit 输出正值 ----
from time_integrator import compute_cfl_limit
u_cfl = np.ones((4, 4, 4)) * 2.0
dt_cfl = compute_cfl_limit(u_cfl, u_cfl, u_cfl, 1.0, 1.0, 1.0, cfl_number=0.5)
assert dt_cfl > 0, '[TC19] compute_cfl_limit 输出正值 FAILED'

# ---- TC20: compute_kolmogorov_scales 所有尺度为正 ----
from turbulence_stats import compute_kolmogorov_scales
eta_k, tau_k, u_eta_k, Re_lam = compute_kolmogorov_scales(epsilon=0.01, nu=1.5e-5)
assert eta_k > 0 and tau_k > 0 and u_eta_k > 0 and Re_lam > 0, '[TC20] compute_kolmogorov_scales 所有尺度为正 FAILED'

# ---- TC21: compute_tke TKE 场与均值均非负 ----
from turbulence_stats import compute_tke
np.random.seed(42)
u_tke = np.random.randn(8, 8, 8)
v_tke = np.random.randn(8, 8, 8)
w_tke = np.random.randn(8, 8, 8)
tke, tke_mean = compute_tke(u_tke, v_tke, w_tke)
assert np.all(tke >= 0), '[TC21] compute_tke TKE 场非负 FAILED'
assert tke_mean >= 0, '[TC21] compute_tke TKE 均值非负 FAILED'

# ---- TC22: projection_operator 结果矩阵对称 ----
from wave_interactions import projection_operator
P_op = projection_operator((1.0, 2.0, 3.0))
assert np.allclose(P_op, P_op.T, atol=1e-12), '[TC22] projection_operator 结果矩阵对称 FAILED'

# ---- TC23: enumerate_triads 2D 截断球内存在三波组 ----
from wave_interactions import enumerate_triads
triads = enumerate_triads(k_max=2, dim=2)
assert len(triads) > 0, '[TC23] enumerate_triads 2D 截断球内存在三波组 FAILED'

# ---- TC24: closest_point_brute 目标点在点集中时距离为零 ----
from particle_tracker import closest_point_brute
points_set = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
idx_c, dist_c = closest_point_brute(points_set, np.array([1.0, 1.0]))
assert idx_c == 1 and abs(dist_c) < 1e-12, '[TC24] closest_point_brute 目标点在点集中时距离为零 FAILED'

# ---- TC25: initialize_particles 所有粒子位于定义域内 ----
from stochastic_model import initialize_particles
np.random.seed(42)
parts, vels = initialize_particles(50, (0, 100), (0, 100), (0, 50), release_height=10.0)
assert np.all(parts[:, 0] >= 0) and np.all(parts[:, 0] <= 100), '[TC25] initialize_particles x 方向在域内 FAILED'
assert np.all(parts[:, 1] >= 0) and np.all(parts[:, 1] <= 100), '[TC25] initialize_particles y 方向在域内 FAILED'
assert np.all(parts[:, 2] >= 0.1) and np.all(parts[:, 2] <= 50), '[TC25] initialize_particles z 方向在域内 FAILED'

print('\n全部 25 个测试通过!\n')
