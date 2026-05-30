#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time




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
    x, y, z = position[0], position[1], position[2]
    return 0.5 + 0.5 * np.exp(-2.0 * (x ** 2 + y ** 2))


def concentration_func_3d(position):
    x, y, z = position[0], position[1], position[2]
    return 1.0 * np.exp(-(x ** 2 + y ** 2 + 0.5 * z ** 2))


def main():
    start_time = time.time()
    np.random.seed(42)




    print_header("STEP 1: 三维 ECM 区域四面体网格生成")
    mesh = generate_uniform_box_mesh(
        xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.5, 0.5),
        nx=6, ny=6, nz=4
    )
    print("  初始网格: 节点数 = %d, 单元数 = %d" % (mesh.n_nodes, mesh.n_elements))
    print("  网格总体积 = %.6f" % mesh.total_volume())


    mesh_refined = mesh.refine_uniform()
    print("  细化后网格: 节点数 = %d, 单元数 = %d" % (mesh_refined.n_nodes, mesh_refined.n_elements))
    print("  细化后总体积 = %.6f" % mesh_refined.total_volume())


    gmsh_str = gmsh_format_string(mesh_refined)
    print("  Gmsh 格式字符串长度 = %d 字节" % len(gmsh_str))


    bfaces = mesh_refined.compute_boundary_faces()
    print("  边界面片数量 = %d" % bfaces.shape[0])




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




    print_header("STEP 3: 椭球形细胞群体迁移动力学")
    population = CellPopulation(n_cells=30, domain=((-1, 1), (-1, 1), (-0.5, 0.5)))
    print("  初始细胞数 = %d" % population.n_cells)
    print("  初始群体平均位置 = [% .4f, % .4f, % .4f]" % tuple(population.compute_mean_position()))
    print("  初始群体扩散度 = %.4f" % population.compute_spread())


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


    sens = population.sensitivity_analysis(grad_c_at_position, dt=0.05, n_steps=10, eps=1e-3)
    print("  初始条件敏感性 (L2 偏差末态) = %.6f" % sens[-1])




    print_header("STEP 4: Monte Carlo 细胞体积与受体结合估计")
    sampler = CellMonteCarloSampler(population.cells[0], n_samples=800)
    pts = sampler.sample_cell_body()
    print("  采样点数 = %d" % pts.shape[1])
    vol_est = sampler.estimate_local_volume()
    print("  Monte Carlo 估计细胞体积 = %.4f μm³" % vol_est)


    a, b, c = population.cells[0].shape
    vol_exact = ellipsoid_volume(a, b, c)
    print("  解析椭球体积 = %.4f μm³" % vol_exact)
    print("  体积相对误差 = %.4f%%" % (100.0 * abs(vol_est - vol_exact) / vol_exact))

    binding = sampler.estimate_receptor_binding(concentration_func_3d)
    print("  受体结合概率估计 = %.4f" % binding)




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




    print_header("STEP 6: CVT 自适应采样与周期信号三角插值")
    adaptive = AdaptiveChemotaxisSampler(
        lambda pt: concentration_func_3d(np.array([pt[0], pt[1], 0.0])),
        domain=((-1, 1), (-1, 1))
    )
    adaptive_points = adaptive.sample_adaptive(n_points=20, n_iter=12)
    print("  CVT 自适应采样点数 = %d" % adaptive_points.shape[0])
    print("  采样点质心 = [% .4f, % .4f]" % tuple(adaptive_points.mean(axis=0)))


    t_nodes = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    signal = np.sin(t_nodes) + 0.3 * np.cos(2.0 * t_nodes)
    t_query = np.linspace(0.0, 2.0 * np.pi, 50)
    signal_interp = adaptive.interpolate_periodic_signal(t_nodes, signal, t_query)
    interp_error = np.max(np.abs(signal_interp - (np.sin(t_query) + 0.3 * np.cos(2.0 * t_query))))
    print("  三角插值最大误差 = %.6f" % interp_error)




    print_header("STEP 7: 浓度场 POD / SVD 降阶模型")
    rom = ChemotaxisROM()
    rom.build_basis(snapshots, energy_threshold=0.99)
    print(rom.summary())


    err_list = [rom.relative_error(s) for s in snapshots]
    print("  快照重构相对误差范围: [%.4e, %.4e]" % (min(err_list), max(err_list)))




    print_header("STEP 8: 细胞周期相位动力学")
    phase_dist = np.array([0.5, 0.2, 0.2, 0.1], dtype=float)
    print("  初始相位分布 G1/S/G2/M = [%s]" % ", ".join("%.2f" % v for v in phase_dist))
    for step in range(5):
        phase_dist = advance_cell_cycle(phase_dist, dt=0.5)
    print("  5 步后相位分布 G1/S/G2/M = [%s]" % ", ".join("%.2f" % v for v in phase_dist))

    w_avg = population_weighted_chemotaxis_sensitivity(phase_dist, w_max=1.0, w_min=0.1)
    print("  群体加权 chemotaxis 敏感性 = %.4f" % w_avg)


    shifted = caesar_cycle_shift(phase_dist, k=1)
    print("  循环移位 k=1 后 = [%s]" % ", ".join("%.2f" % v for v in shifted))




    print_header("STEP 9: 特殊函数与数值代数验证")
    sn, cn, dn = jacobi_elliptic(0.8, 0.5)
    print("  Jacobi elliptic (u=0.8, m=0.5): sn=%.8f, cn=%.8f, dn=%.8f" % (sn, cn, dn))
    print("  验证 sn²+cn² = %.2e" % (sn ** 2 + cn ** 2 - 1.0))

    c_val, s_val = cordic_sin_cos(np.pi / 6.0, n=30)
    print("  CORDIC sin(π/6) = %.8f (误差 %.2e)" % (s_val, abs(s_val - 0.5)))
    print("  CORDIC cos(π/6) = %.8f (误差 %.2e)" % (c_val, abs(c_val - np.sqrt(3.0) / 2.0)))


    n_test = 50
    a_tri = np.full(n_test, -1.0, dtype=float)
    b_tri = np.full(n_test, 2.0, dtype=float)
    c_tri = np.full(n_test, -1.0, dtype=float)
    f_tri = np.ones(n_test, dtype=float)
    x_tri = tridiag_solve(a_tri, b_tri, c_tri, f_tri)

    residual = np.zeros(n_test, dtype=float)
    residual[0] = b_tri[0] * x_tri[0] + c_tri[0] * x_tri[1] - f_tri[0]
    for i in range(1, n_test - 1):
        residual[i] = a_tri[i] * x_tri[i - 1] + b_tri[i] * x_tri[i] + c_tri[i] * x_tri[i + 1] - f_tri[i]
    residual[-1] = a_tri[-1] * x_tri[-2] + b_tri[-1] * x_tri[-1] - f_tri[-1]
    print("  三对角求解最大残差 = %.2e" % np.max(np.abs(residual)))




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
