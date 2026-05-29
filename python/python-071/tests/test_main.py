# -*- coding: utf-8 -*-
"""
main.py
湍流大涡模拟与亚格子模型综合计算平台

统一入口文件，零参数可运行。

项目主题: 计算流体力学 —— 湍流大涡模拟 (LES) 与亚格子尺度 (SGS) 建模

科学问题:
  本程序模拟三维不可压缩湍流的大涡模拟过程，结合以下前沿技术:
  1. 有限元/谱元混合离散方法
  2. Smagorinsky 与动态 Smagorinsky 亚格子模型
  3. 拉格朗日粒子追踪与湍流扩散分析
  4. 基于渗流理论的湍流相干结构拓扑分析
  5. 压力 Poisson 方程的稀疏求解
  6. 球面谐波初始化湍流场

数学控制方程:
  不可压缩 Navier-Stokes 方程（滤波形式）:
    du_i/dt + d/dx_j (u_i * u_j) = -1/rho * dp/dx_i + nu * d2u_i/dx_j2 - d tau_{ij}/dx_j
    du_i/dx_i = 0

  其中 tau_{ij} 为亚格子应力张量，通过 SGS 模型闭合。
"""

import numpy as np
import time

from utils import log_message, print_numeric_matrix
from fem_core import basis_mn_t3, local_stiffness_matrix_t3, local_mass_matrix_t3, wedge_boundary_layer_integral
from kronrod_integrator import kronrod_rule, adaptive_kronrod_integrate
from spectral_lobatto import gll_nodes_weights, spectral_laplacian_1d, spectral_derivative_2d
from sphere_geometry import sphere_distance1, ll_to_xyz, generate_turbulent_initial_field
from sparse_solver import conjugate_gradient, bicgstab, st_to_ge
from time_marching import fractional_step_ns_3d, rk4_step, pendulum_exact_solution
from sgs_closure import smagorinsky_model, dynamic_smagorinsky, ifs_turbulence_generator, structure_function_model
from lagrangian_tracker import lagrangian_particle_tracker, turbulent_diffusion_coefficient, pair_separation_statistics
from topology_percolation import q_criterion, percolation_analysis_3d, energy_cascade_topology


def demo_fem_basis():
    """演示有限元基函数计算"""
    log_message("=== 有限元基函数验证 ===")
    vertices = np.array([[0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=float)
    p = np.array([[0.25, 0.25]], dtype=float)
    phi, dphidx, dphidy = basis_mn_t3(vertices, 1, p)
    log_message(f"T3 基函数值: {phi[:, 0]}")
    log_message(f"T3 基函数导数 dphi/dx: {dphidx[:, 0]}")

    K_local = local_stiffness_matrix_t3(vertices, nu=0.01)
    M_local = local_mass_matrix_t3(vertices)
    log_message(f"局部刚度矩阵 K (3x3):")
    print_numeric_matrix(K_local)
    log_message(f"局部质量矩阵 M (3x3):")
    print_numeric_matrix(M_local)


def demo_kronrod_integration():
    """演示 Kronrod 积分"""
    log_message("=== Gauss-Kronrod 数值积分验证 ===")
    x, w1, w2 = kronrod_rule(n=7)
    log_message(f"Kronrod 节点数: {len(x)}")

    # 测试积分: integral_{-1}^{1} x^6 dx = 2/7
    f = lambda x: x ** 6
    result = adaptive_kronrod_integrate(f, -1.0, 1.0, n=7, tol=1e-12)
    exact = 2.0 / 7.0
    log_message(f"Integral x^6 from [-1,1]: {result:.10f} (exact: {exact:.10f}, error: {abs(result-exact):.2e})")

    # 测试积分: integral_{-1}^{1} exp(x) dx = e - 1/e
    f2 = lambda x: np.exp(x)
    result2 = adaptive_kronrod_integrate(f2, -1.0, 1.0, n=7, tol=1e-12)
    exact2 = np.exp(1.0) - np.exp(-1.0)
    log_message(f"Integral exp(x) from [-1,1]: {result2:.10f} (exact: {exact2:.10f}, error: {abs(result2-exact2):.2e})")


def demo_spectral_lobatto():
    """演示 Lobatto 谱元方法"""
    log_message("=== Lobatto 谱元方法验证 ===")
    n = 8
    nodes, weights = gll_nodes_weights(n)
    log_message(f"GLL 节点 (n={n}): {nodes}")
    log_message(f"GLL 权重和: {np.sum(weights):.10f} (应为 2.0)")

    L, nodes_chk, weights_chk = spectral_laplacian_1d(n)
    log_message(f"谱 Laplacian 矩阵特征值范围: [{np.min(np.linalg.eigvalsh(L)):.4f}, {np.max(np.linalg.eigvalsh(L)):.4f}]")


def demo_sphere_geometry():
    """演示球面几何计算"""
    log_message("=== 球面几何验证 ===")
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = np.pi / 2.0, 0.0
    r = 6371.0  # 地球半径 km
    d = sphere_distance1(lat1, lon1, lat2, lon2, r)
    log_message(f"赤道到北极的大圆距离: {d:.2f} km (理论值: {np.pi*r/2:.2f} km)")

    xyz = ll_to_xyz(r, 1, np.array([0.0]), np.array([0.0]))
    log_message(f"(lat=0, lon=0) -> XYZ: {xyz[0]}")


def demo_sparse_solver():
    """演示稀疏矩阵求解"""
    log_message("=== 稀疏线性系统求解验证 ===")
    n = 20
    A = np.diag(2.0 * np.ones(n)) + np.diag(-1.0 * np.ones(n - 1), k=1) + np.diag(-1.0 * np.ones(n - 1), k=-1)
    b = np.ones(n, dtype=float)
    x, info = conjugate_gradient(A, b, tol=1e-10, max_iter=100)
    log_message(f"CG 求解 {n}x{n} 三对角系统: 迭代次数={info['iter']}, 残差={info['residual']:.2e}")

    # 验证 ST 到 GE 转换
    ist = np.array([1, 2, 2, 3])
    jst = np.array([1, 2, 1, 3])
    Ast = np.array([1.0, 2.0, 0.5, 3.0])
    Age = st_to_ge(len(ist), ist, jst, Ast)
    log_message(f"ST 到 GE 转换结果 (3x3):")
    print_numeric_matrix(Age)


def demo_time_marching():
    """演示时间积分与精确解验证"""
    log_message("=== 时间推进验证 ===")
    t_test = np.linspace(0, 2.0, 100)
    theta_exact, omega_exact = pendulum_exact_solution(t_test, theta0=0.5, omega0=0.0)
    log_message(f"非线性摆精确解: theta(2)={theta_exact[-1]:.6f}, omega(2)={omega_exact[-1]:.6f}")


def demo_sgs_models():
    """演示亚格子模型"""
    log_message("=== 亚格子模型验证 ===")
    nx, ny, nz = 16, 16, 16
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    z = np.linspace(0, 2 * np.pi, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 构造一个简单剪切流
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(X)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    nu_sgs_smag, tau_smag = smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.18)
    log_message(f"Smagorinsky SGS 粘度范围: [{np.min(nu_sgs_smag):.6f}, {np.max(nu_sgs_smag):.6f}]")

    nu_sgs_dyn, C_dyn = dynamic_smagorinsky(u, v, w, dx, dy, dz)
    log_message(f"动态 Smagorinsky 系数 C 均值: {np.mean(C_dyn):.6f}")

    nu_sgs_sf = structure_function_model(u, v, w, dx, dy, dz)
    log_message(f"结构函数模型 SGS 粘度范围: [{np.min(nu_sgs_sf):.6f}, {np.max(nu_sgs_sf):.6f}]")


def demo_ifs_turbulence():
    """演示 IFS 湍流分形生成"""
    log_message("=== IFS 湍流分形结构生成 ===")
    points, energies = ifs_turbulence_generator(n_points=1000, n_iter=5000, seed=42)
    log_message(f"生成 {len(points)} 个分形点")
    log_message(f"能量分布范围: [{np.min(energies):.4f}, {np.max(energies):.4f}]")


def demo_lagrangian_tracking():
    """演示拉格朗日粒子追踪"""
    log_message("=== 拉格朗日粒子追踪 ===")
    nx, ny, nz = 16, 16, 8
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 0.5, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 构造一个简单的三维速度场（Taylor-Green 涡的变体）
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(np.pi * Z)
    v = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.cos(np.pi * Z)
    w = 0.1 * np.sin(np.pi * Z)

    trajectories, msd = lagrangian_particle_tracker(
        u, v, w, x, y, z, n_particles=20, n_steps=50, dt=0.001, D_diff=0.001, seed=42)

    D_turb = turbulent_diffusion_coefficient(msd, dt=0.001)
    log_message(f"拉格朗日粒子追踪: {len(trajectories)} 个粒子, {len(msd)} 步")
    log_message(f"最终均方位移 MSD: {msd[-1]:.6f}")
    log_message(f"最终湍流扩散系数 D: {D_turb[-1]:.6f}")

    # 粒子对分离统计
    r2 = pair_separation_statistics(trajectories, dt=0.001)
    log_message(f"最终均方分离距离: {r2[-1]:.6f}")


def demo_topology_analysis():
    """演示湍流拓扑分析"""
    log_message("=== 湍流相干结构拓扑分析 ===")
    nx, ny, nz = 16, 16, 8
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 0.5, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(np.pi * Z)
    v = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * np.cos(np.pi * Z)
    w = 0.1 * np.sin(np.pi * Z)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    Q = q_criterion(u, v, w, dx, dy, dz)
    log_message(f"Q-准则范围: [{np.min(Q):.4f}, {np.max(Q):.4f}]")

    perc_results = percolation_analysis_3d(Q, thresholds=[np.percentile(Q, 50), np.percentile(Q, 75)])
    for r in perc_results:
        log_message(f"阈值={r['threshold']:.4f}: 占据率={r['p_occupied']:.3f}, "
                    f"团簇数={r['n_components']}, 最大团簇={r['max_size']}, "
                    f"分形维数={r['fractal_dim']:.3f}")

    scales, metrics = energy_cascade_topology(u, v, w, dx, dy, dz)
    for m in metrics:
        log_message(f"尺度 {m['scale']}x: 团簇数={m['n_components']}, 分形维数={m['fractal_dim']:.3f}")


def demo_les_simulation():
    """
    演示完整的 LES 模拟流程。
    """
    log_message("=== 完整 LES 模拟流程 ===")
    log_message("初始化三维湍流场...")

    nx, ny, nz = 16, 16, 8
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    z = np.linspace(0, np.pi, nz)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    # 使用球面谐波生成初始湍流场
    u, v, w = generate_turbulent_initial_field(nx, ny, nz, max_l=6, seed=42)
    p = np.zeros((nx, ny, nz), dtype=float)

    # 物理参数
    nu = 1.0e-3
    dt = 0.001
    n_steps = 10

    log_message(f"网格: {nx}x{ny}x{nz}, 粘性: {nu}, 时间步长: {dt}, 总步数: {n_steps}")

    # 零体积力
    fu = np.zeros_like(u)
    fv = np.zeros_like(v)
    fw = np.zeros_like(w)

    for step in range(n_steps):
        # 添加 SGS 力作为有效体积力
        nu_sgs, _ = smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.18)
        # SGS 效应通过增加有效粘度体现
        nu_eff = nu + np.mean(nu_sgs)

        # 分数步时间推进
        u, v, w, p = fractional_step_ns_3d(
            u, v, w, p, dt, dx, dy, dz, nu_eff, fu, fv, fw)

        # 动能监控
        ke = 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)
        if step % 2 == 0:
            log_message(f"  Step {step+1}/{n_steps}: 动能={ke:.6f}")

    # 最终统计
    ke_final = 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)
    enstrophy = 0.5 * np.mean(
        ((np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)) ** 2
        + ((np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)) ** 2
        + ((np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2 * dz)) ** 2
    )

    log_message(f"LES 模拟完成: 最终动能={ke_final:.6f}, 涡量拟能={enstrophy:.6f}")

    # 边界层动量厚度计算
    theta_bl = wedge_boundary_layer_integral(nu, delta=0.1, order=4)
    log_message(f"边界层动量厚度估计: {theta_bl:.6f}")


def main():
    """
    统一入口函数。
    执行所有模块的验证与完整 LES 模拟流程。
    """
    log_message("=" * 60)
    log_message("湍流大涡模拟与亚格子模型综合计算平台")
    log_message("Computational Fluid Dynamics: LES & SGS Modeling")
    log_message("=" * 60)

    t_start = time.time()

    # 各模块独立验证
    demo_fem_basis()
    demo_kronrod_integration()
    demo_spectral_lobatto()
    demo_sphere_geometry()
    demo_sparse_solver()
    demo_time_marching()
    demo_sgs_models()
    demo_ifs_turbulence()
    demo_lagrangian_tracking()
    demo_topology_analysis()

    # 完整 LES 模拟
    demo_les_simulation()

    t_elapsed = time.time() - t_start
    log_message("=" * 60)
    log_message(f"所有计算完成，总耗时: {t_elapsed:.3f} 秒")
    log_message("=" * 60)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（29个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: T3基函数在节点处为单位向量 ----
vertices = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
phi, dphidx, dphidy = basis_mn_t3(vertices, 1, vertices[:, 0:1])
assert np.allclose(phi[:, 0], [1.0, 0.0, 0.0], atol=1e-10), '[TC01] T3基函数在节点处为单位向量 FAILED'

# ---- TC02: T3基函数在单元内和为1 ----
p_test = np.array([[0.25, 0.25]], dtype=float)
phi, _, _ = basis_mn_t3(vertices, 1, p_test)
assert np.isclose(np.sum(phi), 1.0, atol=1e-10), '[TC02] T3基函数在单元内和为1 FAILED'

# ---- TC03: 局部刚度矩阵对称正定 ----
K = local_stiffness_matrix_t3(vertices, nu=0.01)
assert np.allclose(K, K.T, atol=1e-10), '[TC03] 局部刚度矩阵对称 FAILED'
assert np.all(np.linalg.eigvalsh(K) >= -1e-12), '[TC03] 局部刚度矩阵正定 FAILED'

# ---- TC04: 局部质量矩阵为正对角阵 ----
M = local_mass_matrix_t3(vertices)
assert np.allclose(M, np.diag(np.diag(M)), atol=1e-14), '[TC04] 局部质量矩阵为对角阵 FAILED'
assert np.all(np.diag(M) > 0), '[TC04] 局部质量矩阵对角元为正 FAILED'

# ---- TC05: 边界层动量厚度为正有限值 ----
theta_bl = wedge_boundary_layer_integral(0.001, delta=0.1, order=4)
assert np.isfinite(theta_bl) and theta_bl > 0, '[TC05] 边界层动量厚度为正有限值 FAILED'

# ---- TC06: Kronrod规则返回有效节点和权重 ----
x, wk, wg = kronrod_rule(n=7)
assert len(x) == len(wk) == len(wg) and len(x) > 0 and np.all(wk >= 0), '[TC06] Kronrod规则返回有效节点和权重 FAILED'

# ---- TC07: 自适应Kronrod积分多项式精确 ----
f = lambda x: x ** 6
result = adaptive_kronrod_integrate(f, -1.0, 1.0, n=7, tol=1e-12)
exact = 2.0 / 7.0
assert np.isclose(result, exact, atol=1e-8), '[TC07] 自适应Kronrod积分多项式精确 FAILED'

# ---- TC08: GLL权重和接近2 ----
nodes, weights = gll_nodes_weights(n=8)
assert np.isclose(np.sum(weights), 2.0, atol=1e-4), '[TC08] GLL权重和接近2 FAILED'

# ---- TC09: 谱Laplacian矩阵对称 ----
L, _, _ = spectral_laplacian_1d(n=6)
assert np.allclose(L, L.T, atol=1e-10), '[TC09] 谱Laplacian矩阵对称 FAILED'

# ---- TC10: 球面距离赤道到北极为四分之一周长 ----
d = sphere_distance1(0.0, 0.0, np.pi / 2.0, 0.0, 6371.0)
assert np.isclose(d, np.pi * 6371.0 / 2.0, atol=1e-3), '[TC10] 球面距离赤道到北极 FAILED'

# ---- TC11: ll_to_xyz坐标转换正确 ----
lat, lon = np.array([0.0]), np.array([0.0])
xyz = ll_to_xyz(1.0, 1, lat, lon)
expected = np.array([1.0, 0.0, 0.0])
assert np.allclose(xyz[0], expected, atol=1e-10), '[TC11] ll_to_xyz坐标转换正确 FAILED'

# ---- TC12: CG求解三对角系统残差足够小 ----
n = 10
A = np.diag(2.0 * np.ones(n)) + np.diag(-1.0 * np.ones(n - 1), k=1) + np.diag(-1.0 * np.ones(n - 1), k=-1)
b = np.ones(n, dtype=float)
x, info = conjugate_gradient(A, b, tol=1e-10, max_iter=100)
residual = np.linalg.norm(A @ x - b)
assert residual < 1e-8, '[TC12] CG求解三对角系统残差 FAILED'

# ---- TC13: BiCGSTAB求解单位矩阵精确 ----
A = np.eye(5)
b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x, info = bicgstab(A, b, tol=1e-10)
assert np.allclose(x, b, atol=1e-8), '[TC13] BiCGSTAB求解单位矩阵 FAILED'

# ---- TC14: ST到GE转换正确 ----
ist = np.array([1, 2, 2, 3])
jst = np.array([1, 2, 1, 3])
Ast = np.array([1.0, 2.0, 0.5, 3.0])
Age = st_to_ge(len(ist), ist, jst, Ast)
expected = np.array([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]])
assert np.allclose(Age, expected, atol=1e-10), '[TC14] ST到GE转换正确 FAILED'

# ---- TC15: 非线性摆小角度近似初值正确 ----
t = np.array([0.0, 0.1, 0.2])
theta, omega = pendulum_exact_solution(t, theta0=1e-7, omega0=0.0)
assert np.isclose(theta[0], 1e-7, atol=1e-10), '[TC15] 非线性摆小角度近似初值 FAILED'

# ---- TC16: RK4对线性ODE达到四阶精度 ----
f_lin = lambda t, y: y
y0 = np.array([1.0])
dt = 0.1
y1 = rk4_step(f_lin, 0.0, y0, dt)
assert np.isclose(y1[0], np.exp(dt), atol=1e-6), '[TC16] RK4对线性ODE精度 FAILED'

# ---- TC17: 分数步NS三维零输入输出保持输入尺寸 ----
u3 = np.zeros((4, 4, 4))
v3 = np.zeros((4, 4, 4))
w3 = np.zeros((4, 4, 4))
p3 = np.zeros((4, 4, 4))
u_new, v_new, w_new, p_new = fractional_step_ns_3d(u3, v3, w3, p3, 0.01, 0.1, 0.1, 0.1, 0.001, u3, v3, w3)
assert u_new.shape == u3.shape, '[TC17] 分数步NS三维输出尺寸 FAILED'

# ---- TC18: Smagorinsky SGS粘度非负 ----
nx, ny, nz = 8, 8, 4
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
z = np.linspace(0, 2 * np.pi, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
u = np.sin(X) * np.cos(Y) * np.cos(Z)
v = -np.cos(X) * np.sin(Y) * np.cos(Z)
w = np.zeros_like(X)
dx = x[1] - x[0]
nu_sgs, _ = smagorinsky_model(u, v, w, dx, dx, dx, Cs=0.18)
assert np.all(nu_sgs >= 0), '[TC18] Smagorinsky SGS粘度非负 FAILED'

# ---- TC19: 动态Smagorinsky系数在裁剪范围内 ----
nu_sgs_dyn, C_dyn = dynamic_smagorinsky(u, v, w, dx, dx, dx)
assert np.all(C_dyn >= -0.5) and np.all(C_dyn <= 0.5), '[TC19] 动态Smagorinsky系数范围 FAILED'

# ---- TC20: IFS湍流生成器相同种子结果可复现 ----
points1, energies1 = ifs_turbulence_generator(n_points=100, n_iter=500, seed=42)
points2, energies2 = ifs_turbulence_generator(n_points=100, n_iter=500, seed=42)
assert np.allclose(points1, points2, atol=1e-10), '[TC20] IFS湍流生成器可复现性 FAILED'

# ---- TC21: 结构函数模型SGS粘度非负 ----
nu_sgs_sf = structure_function_model(u, v, w, dx, dx, dx)
assert np.all(nu_sgs_sf >= 0), '[TC21] 结构函数模型SGS粘度非负 FAILED'

# ---- TC22: 拉格朗日粒子追踪输出尺寸正确 ----
nx, ny, nz = 8, 8, 4
x_g = np.linspace(0, 1, nx)
y_g = np.linspace(0, 1, ny)
z_g = np.linspace(0, 0.5, nz)
Xg, Yg, Zg = np.meshgrid(x_g, y_g, z_g, indexing='ij')
u_f = np.sin(2 * np.pi * Xg) * np.cos(2 * np.pi * Yg)
v_f = -np.cos(2 * np.pi * Xg) * np.sin(2 * np.pi * Yg)
w_f = np.zeros_like(Xg)
np.random.seed(42)
traj, msd = lagrangian_particle_tracker(u_f, v_f, w_f, x_g, y_g, z_g, n_particles=10, n_steps=20, dt=0.001, D_diff=0.001, seed=42)
assert traj.shape == (10, 21, 3) and len(msd) == 21, '[TC22] 拉格朗日粒子追踪输出尺寸 FAILED'

# ---- TC23: 湍流扩散系数首项为零 ----
msd_test = np.array([0.0, 0.1, 0.2])
D = turbulent_diffusion_coefficient(msd_test, dt=0.1)
assert D[0] == 0.0, '[TC23] 湍流扩散系数首项为零 FAILED'

# ---- TC24: 粒子对分离统计长度匹配 ----
r2 = pair_separation_statistics(traj, dt=0.001)
assert len(r2) == 21, '[TC24] 粒子对分离统计长度 FAILED'

# ---- TC25: Q准则输出尺寸与输入速度场一致 ----
Q = q_criterion(u, v, w, dx, dx, dx)
assert Q.shape == u.shape, '[TC25] Q准则输出尺寸匹配 FAILED'

# ---- TC26: 渗流分析返回值结构正确 ----
field = np.random.RandomState(42).rand(4, 4, 4)
results = percolation_analysis_3d(field, thresholds=[0.2, 0.5, 0.8])
assert len(results) == 3 and 'n_components' in results[0], '[TC26] 渗流分析返回值结构 FAILED'

# ---- TC27: 能量级串拓扑输出为列表结构 ----
scales, metrics = energy_cascade_topology(u, v, w, dx, dx, dx)
assert isinstance(scales, list) and isinstance(metrics, list) and len(metrics) <= len(scales), '[TC27] 能量级串拓扑输出结构 FAILED'

# ---- TC28: 湍流初始场经过去均值处理 ----
u_init, v_init, w_init = generate_turbulent_initial_field(8, 8, 4, max_l=3, seed=42)
assert np.isclose(np.mean(u_init), 0.0, atol=1e-10), '[TC28] 湍流初始场零均值 FAILED'

# ---- TC29: 谱微分二维常数场导数近似为零 ----
u_const = np.ones((5, 5))
dudx, dudy = spectral_derivative_2d(u_const, nx=4, ny=4)
assert np.allclose(dudx, 0.0, atol=1e-6) and np.allclose(dudy, 0.0, atol=1e-6), '[TC29] 谱微分二维常数场导数 FAILED'

print('\n全部 29 个测试通过!\n')
