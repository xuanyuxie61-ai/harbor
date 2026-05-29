#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
三维肿瘤细胞趋化迁移与微环境交互的多尺度计算框架

统一入口，零参数可运行。

执行流程：
  1. 生成三维 ECM 区域与四面体网格
  2. 初始化趋化因子浓度场并求解 RDA 方程
  3. 生成细胞群体并执行迁移动力学
  4. 对细胞进行 Monte Carlo 采样与体积估计
  5. 使用高阶棱柱求积计算细胞-ECM 接触力学
  6. 执行 CVT 自适应采样与三角插值
  7. 构建浓度场快照的 SVD 降阶模型
  8. 执行细胞周期动力学与敏感性分析
  9. 输出完整的数值结果摘要
"""

import numpy as np
import time

# ---------------------------------------------------------------------------
# 模块导入
# ---------------------------------------------------------------------------
from mesh_engine import TetrahedralMesh, generate_uniform_box_mesh, gmsh_format_string
from chemotaxis_solver import ChemotaxisSolver
from cell_dynamics import CellPopulation, ellipsoid_surface_area_rudolf, ellipsoid_volume
from monte_carlo_sampler import CellMonteCarloSampler, ellipsoid_volume_mc
from quadrature_rules import average_concentration_in_cell, cell_ecm_contact_integral
from rom_analysis import ChemotaxisROM
from adaptive_grid import AdaptiveChemotaxisSampler
from cell_cycle import (advance_cell_cycle, population_weighted_chemotaxis_sensitivity,
                        caesar_cycle_shift, CellCyclePhase)
from special_math import jacobi_elliptic, cordic_sin_cos, tridiag_solve


def print_header(title):
    print("\n" + "=" * 70)
    print(" " + title)
    print("=" * 70)


def ecm_density_func(position):
    """ECM 密度场（高斯型非均匀分布）。"""
    x, y, z = position[0], position[1], position[2]
    return 0.5 + 0.5 * np.exp(-2.0 * (x ** 2 + y ** 2))


def concentration_func_3d(position):
    """三维空间中的化学浓度分布（高斯型趋化因子源）。"""
    x, y, z = position[0], position[1], position[2]
    return 1.0 * np.exp(-(x ** 2 + y ** 2 + 0.5 * z ** 2))


def main():
    start_time = time.time()
    np.random.seed(42)

    # =====================================================================
    # 1. 网格生成 (融合 378_fem_to_gmsh + 1350_triangulation_refine + 1168_stla_to_tri_surface_fast)
    # =====================================================================
    print_header("STEP 1: 三维 ECM 区域四面体网格生成")
    mesh = generate_uniform_box_mesh(
        xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.5, 0.5),
        nx=6, ny=6, nz=4
    )
    print("  初始网格: 节点数 = %d, 单元数 = %d" % (mesh.n_nodes, mesh.n_elements))
    print("  网格总体积 = %.6f" % mesh.total_volume())

    # 执行一次一致细化
    mesh_refined = mesh.refine_uniform()
    print("  细化后网格: 节点数 = %d, 单元数 = %d" % (mesh_refined.n_nodes, mesh_refined.n_elements))
    print("  细化后总体积 = %.6f" % mesh_refined.total_volume())

    # Gmsh 格式导出验证（字符串长度）
    gmsh_str = gmsh_format_string(mesh_refined)
    print("  Gmsh 格式字符串长度 = %d 字节" % len(gmsh_str))

    # 提取边界面
    bfaces = mesh_refined.compute_boundary_faces()
    print("  边界面片数量 = %d" % bfaces.shape[0])

    # =====================================================================
    # 2. 趋化因子浓度场求解 (融合 357_fd1d_burgers_leap + 058_atkinson/heat2)
    # =====================================================================
    print_header("STEP 2: 趋化因子反应-扩散-对流方程求解")
    solver = ChemotaxisSolver(
        nx=24, ny=24, nz=12,
        xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.5, 0.5),
        D=0.02, lambda_deg=0.05, Vmax=0.8, Km=0.3
    )
    solver.set_initial_condition(lambda x, y, z: np.exp(-(x ** 2 + y ** 2)))
    print("  初始浓度场总质量 = %.6f" % solver.total_mass())

    n_steps = 15
    dt_fixed = 0.02
    snapshots = []
    for step in range(n_steps):
        # 对流速度场：由细胞分泌和组织液流动共同驱动
        vx = 0.05 * np.sin(0.5 * solver.x)
        vy = 0.03 * np.cos(0.5 * solver.y)
        vz = np.zeros(solver.nz)
        dt = solver.step(vx, vy, vz, dt=dt_fixed)
        if step % 3 == 0:
            snapshots.append(solver.c.copy())
            print("    Step %2d, dt=%.4f, 总质量=%.6f" % (step, dt, solver.total_mass()))

    print("  最终浓度场总质量 = %.6f" % solver.total_mass())
    gx, gy, gz = solver.gradient()
    print("  浓度梯度 max|∇c| = %.6f" % np.max(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)))

    # =====================================================================
    # 3. 细胞群体迁移 (融合 332_ellipsoid + 1064_sensitive_ode)
    # =====================================================================
    print_header("STEP 3: 椭球形细胞群体迁移动力学")
    population = CellPopulation(n_cells=30, domain=((-1, 1), (-1, 1), (-0.5, 0.5)))
    print("  初始细胞数 = %d" % population.n_cells)
    print("  初始群体平均位置 = [% .4f, % .4f, % .4f]" % tuple(population.compute_mean_position()))
    print("  初始群体扩散度 = %.4f" % population.compute_spread())

    # 插值梯度函数供细胞查询
    def grad_c_at_position(pos):
        x, y, z = pos[0], pos[1], pos[2]
        ix = int(np.clip((x - solver.xlim[0]) / (solver.xlim[1] - solver.xlim[0]) * (solver.nx - 1),
                         0, solver.nx - 1))
        iy = int(np.clip((y - solver.ylim[0]) / (solver.ylim[1] - solver.ylim[0]) * (solver.ny - 1),
                         0, solver.ny - 1))
        iz = int(np.clip((z - solver.zlim[0]) / (solver.zlim[1] - solver.zlim[0]) * (solver.nz - 1),
                         0, solver.nz - 1))
        return np.array([gx[ix, iy, iz], gy[ix, iy, iz], gz[ix, iy, iz]])

    n_migration_steps = 20
    for step in range(n_migration_steps):
        population.step_all(grad_c_at_position, dt=0.05,
                            ecm_density_func=ecm_density_func,
                            mu=0.8, gamma=0.4, beta=0.3, sigma=0.03)

    print("  迁移后群体平均位置 = [% .4f, % .4f, % .4f]" % tuple(population.compute_mean_position()))
    print("  迁移后群体扩散度 = %.4f" % population.compute_spread())
    print("  细胞总表面积 = %.4f μm²" % population.total_surface_area())
    print("  细胞总体积 = %.4f μm³" % population.total_volume())

    # 敏感性分析
    sens = population.sensitivity_analysis(grad_c_at_position, dt=0.05, n_steps=10, eps=1e-3)
    print("  初始条件敏感性 (L2 偏差末态) = %.6f" % sens[-1])

    # =====================================================================
    # 4. Monte Carlo 采样 (融合 334_ellipsoid_monte_carlo)
    # =====================================================================
    print_header("STEP 4: Monte Carlo 细胞体积与受体结合估计")
    sampler = CellMonteCarloSampler(population.cells[0], n_samples=800)
    pts = sampler.sample_cell_body()
    print("  采样点数 = %d" % pts.shape[1])
    vol_est = sampler.estimate_local_volume()
    print("  Monte Carlo 估计细胞体积 = %.4f μm³" % vol_est)

    # 解析体积对比
    a, b, c = population.cells[0].shape
    vol_exact = ellipsoid_volume(a, b, c)
    print("  解析椭球体积 = %.4f μm³" % vol_exact)
    print("  体积相对误差 = %.4f%%" % (100.0 * abs(vol_est - vol_exact) / vol_exact))

    binding = sampler.estimate_receptor_binding(concentration_func_3d)
    print("  受体结合概率估计 = %.4f" % binding)

    # =====================================================================
    # 5. 高阶棱柱求积 (融合 916_prism_jaskowiec_rule)
    # =====================================================================
    print_header("STEP 5: 高阶棱柱求积与细胞-ECM 接触力学")
    cell0 = population.cells[0]
    force_p3 = cell_ecm_contact_integral(cell0.position, cell0.shape,
                                          ecm_density_func,
                                          contact_stiffness=1.0, p=3)
    force_p5 = cell_ecm_contact_integral(cell0.position, cell0.shape,
                                          ecm_density_func,
                                          contact_stiffness=1.0, p=5)
    print("  接触力积分 (p=3) = %.6f" % force_p3)
    print("  接触力积分 (p=5) = %.6f" % force_p5)
    print("  不同阶数差异 = %.6f" % abs(force_p3 - force_p5))

    avg_conc = average_concentration_in_cell(cell0.position, cell0.shape,
                                              concentration_func_3d,
                                              n_prisms=8, p=4)
    print("  细胞内平均浓度 (棱柱求积) = %.6f" % avg_conc)

    # =====================================================================
    # 6. CVT 自适应采样与三角插值 (融合 253_cvt_circle_nonuniform + 596_interp_trig)
    # =====================================================================
    print_header("STEP 6: CVT 自适应采样与周期信号三角插值")
    adaptive = AdaptiveChemotaxisSampler(
        lambda pt: concentration_func_3d(np.array([pt[0], pt[1], 0.0])),
        domain=((-1, 1), (-1, 1))
    )
    adaptive_points = adaptive.sample_adaptive(n_points=20, n_iter=12)
    print("  CVT 自适应采样点数 = %d" % adaptive_points.shape[0])
    print("  采样点质心 = [% .4f, % .4f]" % tuple(adaptive_points.mean(axis=0)))

    # 三角插值：模拟周期化学振荡信号
    t_nodes = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    signal = np.sin(t_nodes) + 0.3 * np.cos(2.0 * t_nodes)
    t_query = np.linspace(0.0, 2.0 * np.pi, 50)
    signal_interp = adaptive.interpolate_periodic_signal(t_nodes, signal, t_query)
    interp_error = np.max(np.abs(signal_interp - (np.sin(t_query) + 0.3 * np.cos(2.0 * t_query))))
    print("  三角插值最大误差 = %.6f" % interp_error)

    # =====================================================================
    # 7. SVD 降阶模型 (融合 1184_svd_basis)
    # =====================================================================
    print_header("STEP 7: 浓度场 POD / SVD 降阶模型")
    rom = ChemotaxisROM()
    rom.build_basis(snapshots, energy_threshold=0.99)
    print(rom.summary())

    # 验证重构精度
    err_list = [rom.relative_error(s) for s in snapshots]
    print("  快照重构相对误差范围: [%.4e, %.4e]" % (min(err_list), max(err_list)))

    # =====================================================================
    # 8. 细胞周期动力学 (融合 132_caesar → 循环置换)
    # =====================================================================
    print_header("STEP 8: 细胞周期相位动力学")
    phase_dist = np.array([0.5, 0.2, 0.2, 0.1], dtype=float)  # G1, S, G2, M
    print("  初始相位分布 G1/S/G2/M = [%s]" % ", ".join("%.2f" % v for v in phase_dist))
    for step in range(5):
        phase_dist = advance_cell_cycle(phase_dist, dt=0.5)
    print("  5 步后相位分布 G1/S/G2/M = [%s]" % ", ".join("%.2f" % v for v in phase_dist))

    w_avg = population_weighted_chemotaxis_sensitivity(phase_dist, w_max=1.0, w_min=0.1)
    print("  群体加权 chemotaxis 敏感性 = %.4f" % w_avg)

    # Caesar 循环置换验证
    shifted = caesar_cycle_shift(phase_dist, k=1)
    print("  循环移位 k=1 后 = [%s]" % ", ".join("%.2f" % v for v in shifted))

    # =====================================================================
    # 9. 特殊函数验证 (融合 1096_sncndn + 219_cordic + 058_atkinson)
    # =====================================================================
    print_header("STEP 9: 特殊函数与数值代数验证")
    sn, cn, dn = jacobi_elliptic(0.8, 0.5)
    print("  Jacobi elliptic (u=0.8, m=0.5): sn=%.8f, cn=%.8f, dn=%.8f" % (sn, cn, dn))
    print("  验证 sn²+cn² = %.2e" % (sn ** 2 + cn ** 2 - 1.0))

    c_val, s_val = cordic_sin_cos(np.pi / 6.0, n=30)
    print("  CORDIC sin(π/6) = %.8f (误差 %.2e)" % (s_val, abs(s_val - 0.5)))
    print("  CORDIC cos(π/6) = %.8f (误差 %.2e)" % (c_val, abs(c_val - np.sqrt(3.0) / 2.0)))

    # 三对角系统验证
    n_test = 50
    a_tri = np.full(n_test, -1.0, dtype=float)
    b_tri = np.full(n_test, 2.0, dtype=float)
    c_tri = np.full(n_test, -1.0, dtype=float)
    f_tri = np.ones(n_test, dtype=float)
    x_tri = tridiag_solve(a_tri, b_tri, c_tri, f_tri)
    # 验证残差
    residual = np.zeros(n_test, dtype=float)
    residual[0] = b_tri[0] * x_tri[0] + c_tri[0] * x_tri[1] - f_tri[0]
    for i in range(1, n_test - 1):
        residual[i] = a_tri[i] * x_tri[i - 1] + b_tri[i] * x_tri[i] + c_tri[i] * x_tri[i + 1] - f_tri[i]
    residual[-1] = a_tri[-1] * x_tri[-2] + b_tri[-1] * x_tri[-1] - f_tri[-1]
    print("  三对角求解最大残差 = %.2e" % np.max(np.abs(residual)))

    # =====================================================================
    # 10. 结果汇总
    # =====================================================================
    print_header("计算结果汇总")
    elapsed = time.time() - start_time
    print("  总计算时间 = %.3f 秒" % elapsed)
    print("  细胞数量 = %d" % population.n_cells)
    print("  网格单元数 = %d" % mesh_refined.n_elements)
    print("  浓度场时间步数 = %d" % n_steps)
    print("  ROM 保留模态数 = %d" % rom.L)
    print("  ROM 压缩比 = %.1f" % (rom.U.shape[0] / max(1, rom.L)))
    print("  最终群体扩散度 = %.4f" % population.compute_spread())
    print("  敏感性末态偏差 = %.6f" % sens[-1])
    print("  CVT 采样点数 = %d" % adaptive_points.shape[0])
    print("  三角插值误差 = %.6f" % interp_error)
    print("  所有模块执行完毕，无错误。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例 (42个，assert模式，涉及随机值均使用固定种子)
# ================================================================

# ---- TC01: mesh_engine 四面体体积公式正确性 ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
elements = np.array([[0, 1, 2, 3]])
mesh = TetrahedralMesh(nodes, elements)
vol = mesh.element_volume(0)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC01] 四面体体积公式正确性 FAILED'

# ---- TC02: mesh_engine 均匀盒子网格体积 ----
mesh2 = generate_uniform_box_mesh(xlim=(0.0, 2.0), ylim=(0.0, 3.0), zlim=(0.0, 4.0), nx=3, ny=3, nz=3)
assert abs(mesh2.total_volume() - 24.0) < 0.1, '[TC02] 均匀盒子网格体积 FAILED'

# ---- TC03: mesh_engine 网格细化后体积守恒 ----
mesh3 = generate_uniform_box_mesh(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-1.0, 1.0), nx=3, ny=3, nz=3)
vol_before = mesh3.total_volume()
mesh_ref = mesh3.refine_uniform()
vol_after = mesh_ref.total_volume()
assert abs(vol_after - vol_before) < 1e-9, '[TC03] 网格细化后体积守恒 FAILED'

# ---- TC04: mesh_engine 边界面提取 ----
bfaces = mesh_ref.compute_boundary_faces()
assert bfaces.shape[0] > 0, '[TC04] 边界面提取 FAILED'
assert bfaces.shape[1] == 3, '[TC04] 边界面提取 FAILED'

# ---- TC05: mesh_engine Gmsh 格式导出字符串非空 ----
gmsh_str = gmsh_format_string(mesh_ref)
assert len(gmsh_str) > 0, '[TC05] Gmsh 格式导出字符串非空 FAILED'
assert '$Nodes' in gmsh_str, '[TC05] Gmsh 格式导出字符串非空 FAILED'
assert '$Elements' in gmsh_str, '[TC05] Gmsh 格式导出字符串非空 FAILED'

# ---- TC06: chemotaxis_solver 初始条件与总质量 ----
solver = ChemotaxisSolver(nx=8, ny=8, nz=4, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0), D=0.01)
solver.set_initial_condition(lambda x, y, z: 2.0)
mass = solver.total_mass()
expected_mass = 2.0 * solver.nx * solver.ny * solver.nz * solver.dx * solver.dy * solver.dz
assert abs(mass - expected_mass) < 1e-12, '[TC06] 初始条件与总质量 FAILED'

# ---- TC07: chemotaxis_solver 零速度场步进正性 ----
solver2 = ChemotaxisSolver(nx=8, ny=8, nz=4, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.5, 0.5), D=0.1)
solver2.set_initial_condition(lambda x, y, z: np.exp(-x**2 - y**2))
for _ in range(3):
    solver2.step(0.0, 0.0, 0.0, dt=0.01)
assert np.all(solver2.c >= -1e-12), '[TC07] 零速度场步进正性 FAILED'

# ---- TC08: chemotaxis_solver 常数场梯度为零 ----
solver3 = ChemotaxisSolver(nx=8, ny=8, nz=4, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0), D=0.01)
solver3.set_initial_condition(lambda x, y, z: 5.0)
gx, gy, gz = solver3.gradient()
assert np.allclose(gx, 0.0), '[TC08] 常数场梯度为零 FAILED'
assert np.allclose(gy, 0.0), '[TC08] 常数场梯度为零 FAILED'
assert np.allclose(gz, 0.0), '[TC08] 常数场梯度为零 FAILED'

# ---- TC09: cell_dynamics 椭球体积解析公式 ----
v = ellipsoid_volume(2.0, 3.0, 4.0)
assert abs(v - (4.0/3.0)*np.pi*2.0*3.0*4.0) < 1e-12, '[TC09] 椭球体积解析公式 FAILED'

# ---- TC10: cell_dynamics 球体表面积 Rudolf 近似 ----
s = ellipsoid_surface_area_rudolf(1.0, 1.0, 1.0)
assert abs(s - 4.0*np.pi) < 0.01, '[TC10] 球体表面积 Rudolf 近似 FAILED'

# ---- TC11: cell_dynamics 椭球表面积椭圆积分公式 ----
from cell_dynamics import ellipsoid_surface_area_elliptic
s_ellip = ellipsoid_surface_area_elliptic(1.0, 1.0, 1.0)
assert abs(s_ellip - 4.0*np.pi) < 0.02, '[TC11] 椭球表面积椭圆积分公式 FAILED'

# ---- TC12: cell_dynamics CellAgent chemotaxis_velocity 零梯度 ----
from cell_dynamics import CellAgent
cell = CellAgent(np.array([0.0, 0.0, 0.0]))
v_cell = cell.chemotaxis_velocity(np.array([0.0, 0.0, 0.0]))
assert np.allclose(v_cell, 0.0), '[TC12] CellAgent chemotaxis_velocity 零梯度 FAILED'

# ---- TC13: cell_dynamics CellAgent 饱和迁移速度上界 ----
cell2 = CellAgent(np.array([0.0, 0.0, 0.0]))
grad = np.array([100.0, 0.0, 0.0])
v_sat = cell2.chemotaxis_velocity(grad, mu=1.0, gamma=0.5)
assert np.linalg.norm(v_sat) <= 2.0, '[TC13] CellAgent 饱和迁移速度上界 FAILED'

# ---- TC14: cell_dynamics CellPopulation 均值在域内 ----
np.random.seed(42)
pop = CellPopulation(n_cells=10, domain=((-1, 1), (-1, 1), (-0.5, 0.5)))
mean_pos = pop.compute_mean_position()
assert np.all(mean_pos >= np.array([-1.0, -1.0, -0.5])), '[TC14] CellPopulation 均值在域内 FAILED'
assert np.all(mean_pos <= np.array([1.0, 1.0, 0.5])), '[TC14] CellPopulation 均值在域内 FAILED'

# ---- TC15: cell_dynamics CellPopulation spread 非负 ----
spread = pop.compute_spread()
assert spread >= 0.0, '[TC15] CellPopulation spread 非负 FAILED'

# ---- TC16: monte_carlo_sampler cholesky_upper 单位矩阵 ----
from monte_carlo_sampler import cholesky_upper
U = cholesky_upper(np.eye(3))
assert np.allclose(U, np.eye(3)), '[TC16] cholesky_upper 单位矩阵 FAILED'

# ---- TC17: monte_carlo_sampler ellipsoid_volume_mc 球体体积 ----
A = np.eye(3)
vol_mc = ellipsoid_volume_mc(A, 1.0, 3)
assert abs(vol_mc - 4.0*np.pi/3.0) < 1e-9, '[TC17] ellipsoid_volume_mc 球体体积 FAILED'

# ---- TC18: monte_carlo_sampler solve_upper_triangular ----
from monte_carlo_sampler import solve_upper_triangular
U2 = np.array([[2.0, 1.0], [0.0, 3.0]])
b = np.array([4.0, 6.0])
x_sol = solve_upper_triangular(U2, b)
expected = np.array([1.0, 2.0])
assert np.allclose(x_sol, expected), '[TC18] solve_upper_triangular FAILED'

# ---- TC19: quadrature_rules prism_rule_order 权重和等于棱柱体积 (p=0,1,2,4) ----
from quadrature_rules import prism_rule_order
for p in [0, 1, 2, 4]:
    x, y, z, w = prism_rule_order(p)
    assert abs(np.sum(w) - 0.5) < 1e-12, '[TC19] prism_rule_order 权重和 (p=%d) FAILED' % p

# ---- TC20: quadrature_rules integrate_over_prism 常数函数 ----
from quadrature_rules import integrate_over_prism
val = integrate_over_prism(lambda x, y, z: 3.0, p=4)
assert abs(val - 1.5) < 1e-12, '[TC20] integrate_over_prism 常数函数 FAILED'

# ---- TC21: rom_analysis compute_pod_basis 简单矩阵能量 ----
from rom_analysis import compute_pod_basis
A2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
U3, sigma, Vt, L, energy = compute_pod_basis(A2, energy_threshold=0.99)
assert L >= 1, '[TC21] compute_pod_basis 简单矩阵能量 FAILED'
assert energy[-1] >= 0.99, '[TC21] compute_pod_basis 简单矩阵能量 FAILED'

# ---- TC22: rom_analysis ChemotaxisROM 构造与重构一致性 ----
snapshots = [np.ones((4, 4, 4)) * float(i) for i in range(1, 4)]
rom = ChemotaxisROM()
rom.build_basis(snapshots, energy_threshold=0.99)
coeff = rom.project(snapshots[0])
recon = rom.reconstruct(coeff)
assert recon.shape == (4, 4, 4), '[TC22] ChemotaxisROM 重构形状 FAILED'

# ---- TC23: rom_analysis ChemotaxisROM relative_error 自洽 ----
snap_r = np.random.RandomState(42).rand(4, 4, 4)
rom2 = ChemotaxisROM()
rom2.build_basis([snap_r, snap_r*2, snap_r*3], energy_threshold=0.99)
err_r = rom2.relative_error(snap_r)
assert err_r >= 0.0, '[TC23] ChemotaxisROM relative_error 非负 FAILED'
assert err_r < 1.0, '[TC23] ChemotaxisROM relative_error 自洽 FAILED'

# ---- TC24: adaptive_grid trig_interpolant 数据节点精确恢复 ----
from adaptive_grid import trig_interpolant
xd = np.linspace(0.0, 2.0*np.pi, 8, endpoint=False)
yd = np.sin(xd)
xi = xd[2]
yi = trig_interpolant(xd, yd, xi)
assert abs(yi - yd[2]) < 1e-12, '[TC24] trig_interpolant 数据节点精确恢复 FAILED'

# ---- TC25: adaptive_grid trig_interpolant 正弦函数重构 ----
xd2 = np.linspace(0.0, 2.0*np.pi, 16, endpoint=False)
yd2 = np.sin(xd2)
xi2 = np.linspace(0.0, 2.0*np.pi, 100)
yi2 = trig_interpolant(xd2, yd2, xi2)
err_t = np.max(np.abs(yi2 - np.sin(xi2)))
assert err_t < 0.1, '[TC25] trig_interpolant 正弦函数重构 FAILED'

# ---- TC26: adaptive_grid AdaptiveChemotaxisSampler 采样点在域内 ----
np.random.seed(42)
adaptive = AdaptiveChemotaxisSampler(lambda pt: np.exp(-pt[0]**2 - pt[1]**2), domain=((-1, 1), (-1, 1)))
pts = adaptive.sample_adaptive(n_points=8, n_iter=5)
assert pts.shape == (8, 2), '[TC26] AdaptiveChemotaxisSampler 采样点形状 FAILED'
assert np.all(pts[:, 0] >= -1.0) and np.all(pts[:, 0] <= 1.0), '[TC26] AdaptiveChemotaxisSampler 采样点 x 范围 FAILED'
assert np.all(pts[:, 1] >= -1.0) and np.all(pts[:, 1] <= 1.0), '[TC26] AdaptiveChemotaxisSampler 采样点 y 范围 FAILED'

# ---- TC27: cell_cycle advance_cell_cycle 归一化保持 ----
phase = np.array([0.5, 0.2, 0.2, 0.1])
new_phase = advance_cell_cycle(phase, dt=0.1)
assert abs(new_phase.sum() - 1.0) < 1e-12, '[TC27] advance_cell_cycle 归一化保持 FAILED'

# ---- TC28: cell_cycle caesar_cycle_shift 循环 4 次回到原值 ----
phase2 = np.array([0.1, 0.2, 0.3, 0.4])
phase_temp = phase2.copy()
for _ in range(4):
    phase_temp = caesar_cycle_shift(phase_temp, k=1)
assert np.allclose(phase_temp, np.array([0.1, 0.2, 0.3, 0.4])), '[TC28] caesar_cycle_shift 循环 4 次 FAILED'

# ---- TC29: cell_cycle population_weighted_chemotaxis_sensitivity 范围 ----
phase3 = np.array([0.25, 0.25, 0.25, 0.25])
w = population_weighted_chemotaxis_sensitivity(phase3, w_max=1.0, w_min=0.1)
assert 0.1 <= w <= 1.0, '[TC29] population_weighted_chemotaxis_sensitivity 范围 FAILED'

# ---- TC30: cell_cycle chemotaxis_sensitivity_by_phase 值域范围 ----
from cell_cycle import chemotaxis_sensitivity_by_phase
w_g2 = chemotaxis_sensitivity_by_phase(2, w_max=1.0, w_min=0.1)
w_g1 = chemotaxis_sensitivity_by_phase(0, w_max=1.0, w_min=0.1)
assert abs(w_g2 - 1.0) < 1e-10, '[TC30] chemotaxis_sensitivity_by_phase G2=w_max FAILED'
assert abs(w_g1 - 0.1) < 1e-10, '[TC30] chemotaxis_sensitivity_by_phase G1=w_min FAILED'

# ---- TC31: special_math jacobi_elliptic 恒等式 sn²+cn²=1 ----
sn, cn, dn = jacobi_elliptic(0.8, 0.5)
assert abs(sn**2 + cn**2 - 1.0) < 1e-10, '[TC31] jacobi_elliptic 恒等式 FAILED'

# ---- TC32: special_math jacobi_elliptic m=0 退化到三角函数 ----
sn_m0, cn_m0, dn_m0 = jacobi_elliptic(0.5, 0.0)
assert abs(sn_m0 - np.sin(0.5)) < 1e-10, '[TC32] jacobi_elliptic m=0 退化到三角函数 FAILED'
assert abs(cn_m0 - np.cos(0.5)) < 1e-10, '[TC32] jacobi_elliptic m=0 退化到三角函数 FAILED'
assert abs(dn_m0 - 1.0) < 1e-10, '[TC32] jacobi_elliptic m=0 退化到三角函数 FAILED'

# ---- TC33: special_math cordic_sin_cos π/2 ----
c, s = cordic_sin_cos(np.pi/2.0, n=40)
assert abs(s - 1.0) < 1e-10, '[TC33] cordic_sin_cos π/2 FAILED'
assert abs(c - 0.0) < 1e-10, '[TC33] cordic_sin_cos π/2 FAILED'

# ---- TC34: special_math cordic_sin_cos π/6 ----
c2, s2 = cordic_sin_cos(np.pi/6.0, n=30)
assert abs(s2 - 0.5) < 1e-4, '[TC34] cordic_sin_cos π/6 FAILED'
assert abs(c2 - np.sqrt(3.0)/2.0) < 1e-4, '[TC34] cordic_sin_cos π/6 FAILED'

# ---- TC35: special_math tridiag_solve 单位三对角系统 ----
n = 5
a_t = np.zeros(n)
b_t = np.ones(n)
c_t = np.zeros(n)
f_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_t = tridiag_solve(a_t, b_t, c_t, f_t)
assert np.allclose(x_t, f_t), '[TC35] tridiag_solve 单位三对角系统 FAILED'

# ---- TC36: special_math tridiag_solve 已知解析解 ----
n2 = 20
a_t2 = np.full(n2, -1.0, dtype=float)
b_t2 = np.full(n2, 2.0, dtype=float)
c_t2 = np.full(n2, -1.0, dtype=float)
f_t2 = np.ones(n2, dtype=float)
x_t2 = tridiag_solve(a_t2, b_t2, c_t2, f_t2)
res = np.zeros(n2)
res[0] = b_t2[0]*x_t2[0] + c_t2[0]*x_t2[1] - f_t2[0]
for i in range(1, n2-1):
    res[i] = a_t2[i]*x_t2[i-1] + b_t2[i]*x_t2[i] + c_t2[i]*x_t2[i+1] - f_t2[i]
res[-1] = a_t2[-1]*x_t2[-2] + b_t2[-1]*x_t2[-1] - f_t2[-1]
assert np.max(np.abs(res)) < 1e-10, '[TC36] tridiag_solve 已知解析解 FAILED'

# ---- TC37: special_math tridiag_solve_multi 多右端项 ----
from special_math import tridiag_solve_multi
n3 = 6
a_t3 = np.zeros(n3)
b_t3 = np.ones(n3)
c_t3 = np.zeros(n3)
F_t3 = np.column_stack([np.arange(1, n3+1, dtype=float), np.arange(2, n3+2, dtype=float)])
X_t3 = tridiag_solve_multi(a_t3, b_t3, c_t3, F_t3)
assert np.allclose(X_t3, F_t3), '[TC37] tridiag_solve_multi 多右端项 FAILED'

# ---- TC38: mesh_engine 1-based 索引自动检测 ----
nodes2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
elements2 = np.array([[1, 2, 3, 4]])
mesh1b = TetrahedralMesh(nodes2, elements2)
assert mesh1b.elements[0, 0] == 0, '[TC38] 1-based 索引自动检测 FAILED'

# ---- TC39: mesh_engine 质心计算 ----
nodes_c = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
elements_c = np.array([[0, 1, 2, 3]])
mesh_c = TetrahedralMesh(nodes_c, elements_c)
centroids = mesh_c.compute_centroids()
assert np.allclose(centroids[0], np.array([0.5, 0.5, 0.5])), '[TC39] 质心计算 FAILED'

# ---- TC40: cell_dynamics CellAgent step 确定性更新 ----
np.random.seed(42)
cell3 = CellAgent(np.array([1.0, 2.0, 3.0]))
pos_before = cell3.position.copy()
cell3.step(np.array([0.1, 0.0, 0.0]), dt=0.1, ecm_density_func=None, mu=1.0, gamma=0.0, sigma=0.0)
assert not np.allclose(cell3.position, pos_before), '[TC40] CellAgent step 确定性更新 FAILED'

# ---- TC41: cell_dynamics 敏感性分析返回非负偏差 ----
np.random.seed(42)
pop_s = CellPopulation(n_cells=5, domain=((-1, 1), (-1, 1), (-0.5, 0.5)))
sens = pop_s.sensitivity_analysis(lambda p: np.zeros(3), dt=0.1, n_steps=5, eps=0.01)
assert np.all(sens >= 0.0), '[TC41] 敏感性分析返回非负偏差 FAILED'

# ---- TC42: cell_cycle cycle_transition_matrix 性质 ----
from cell_cycle import cycle_transition_matrix
P1 = cycle_transition_matrix(k=1)
assert P1.shape == (4, 4), '[TC42] cycle_transition_matrix 形状 FAILED'
assert np.allclose(P1.sum(axis=0), 1.0), '[TC42] cycle_transition_matrix 列和 FAILED'
assert np.allclose(P1.sum(axis=1), 1.0), '[TC42] cycle_transition_matrix 行和 FAILED'

print('\n全部 42 个测试通过!\n')
