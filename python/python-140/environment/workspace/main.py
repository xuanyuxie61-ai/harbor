
import numpy as np
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import timestamp, check_bounds, print_matrix, cond_number_estimate
from geometry_utils import (
    cylinder_distance_function, rectangle_distance_function,
    torus_distance_function, compute_reactor_boundary_word
)
from reactor_mesh import distmesh_2d, ball_grid, triangulation_refine_local, compute_mesh_quality
from pyrolysis_kinetics import BiomassPyrolysisKinetics, solve_doughnut_flow_rk4
from thermal_model import ThermalReactorModel, compute_reaction_heat_source
from quadrature_integrator import (
    integrate_daem_activation_energy, reactor_cross_section_average,
    quadrature_error_analysis, square01_monte_carlo_integrate
)
from property_interpolator import default_biomass_properties
from particle_dynamics import simulate_particle_transport, compute_local_temperature_from_kinetic
from fem_assembler import (
    assemble_fem_data, write_tecplot_ascii,
    compute_fem_mass_matrix, compute_fem_stiffness_matrix
)


def main():
    print("=" * 72)
    print("生物质热解反应器多物理场耦合模拟系统")
    print("Multi-Physics Coupled Simulation of Biomass Pyrolysis Reactor")
    print("=" * 72)
    timestamp()
    print()




    print("[1] 反应器几何与网格生成")
    print("-" * 48)

    reactor_radius = 0.5
    reactor_bbox = np.array([[-0.7, -0.7], [0.7, 0.7]], dtype=np.float64)

    def reactor_fd(p):
        return cylinder_distance_function(p, radius=reactor_radius)

    p, t = distmesh_2d(reactor_fd, lambda p: np.ones(p.shape[0]),
                        h0=0.12, bbox=reactor_bbox, max_iter=80)
    print(f"    初始网格: {len(p)} 个节点, {len(t)} 个三角形单元")

    if len(t) > 0:

        qualities = compute_mesh_quality(p, t)
        worst_elem = np.argmin(qualities)
        p_refined, t_refined = triangulation_refine_local(p, t, worst_elem)
        print(f"    局部细化最差单元 (质量={qualities[worst_elem]:.4f}) 后: "
              f"{len(p_refined)} 个节点, {len(t_refined)} 个单元")
    else:
        p_refined, t_refined = p, t
        print("    警告: 初始网格为空，跳过局部细化")


    ball_points = ball_grid(n=3, r=0.05, c=np.array([0.0, 0.0, 0.0]))
    print(f"    催化剂颗粒内部网格点: {len(ball_points)} 个")
    print()




    print("[2] 热解反应动力学求解")
    print("-" * 48)

    kinetics = BiomassPyrolysisKinetics()
    tspan = [0.0, 60.0]
    n_steps = 200


    T0 = 300.0
    beta = 10.0

    def T_profile(t):
        return T0 + beta * t


    t_rk4, y_rk4 = kinetics.solve_rk4(tspan, kinetics.y0, n_steps, T_profile)
    print(f"    RK4 求解完成: {n_steps} 步, 终态质量分数:")
    print(f"      Biomass={y_rk4[-1, 0]:.6f}, Hemicellulose={y_rk4[-1, 1]:.6f}, "
          f"Lignin={y_rk4[-1, 2]:.6f}")
    print(f"      Active={y_rk4[-1, 3]:.6f}, Volatiles={y_rk4[-1, 4]:.6f}, "
          f"Char={y_rk4[-1, 5]:.6f}, Tar+Gas={y_rk4[-1, 6]:.6f}")


    t_mp, y_mp = kinetics.solve_midpoint(tspan, kinetics.y0, n_steps // 2, T_profile)
    print(f"    隐式中点法求解完成: {n_steps // 2} 步")


    t_torus, y_torus = solve_doughnut_flow_rk4([0.0, 10.0], [1.0, 0.0, 0.0], 100)
    print(f"    环面反应器流动模拟完成: 100 步, 终态=[{y_torus[-1, 0]:.4f}, "
          f"{y_torus[-1, 1]:.4f}, {y_torus[-1, 2]:.4f}]")
    print()




    print("[3] 反应器一维传热模型")
    print("-" * 48)

    thermal = ThermalReactorModel(L=1.0, nx=40, rho=200.0, Cp=1500.0,
                                   k_eff=0.15, u=0.05)
    T_init = np.full(thermal.nx, 300.0, dtype=np.float64)
    dt = 0.5
    n_thermal_steps = 40

    def Q_source_func(t, x):


        pass

    t_thermal, T_history = thermal.simulate(T_init, dt, n_thermal_steps,
                                            Q_source_func, T_inlet=350.0)
    print(f"    传热模拟完成: {n_thermal_steps} 步, Δt={dt}s")
    print(f"    入口温度=350K, 出口温度={T_history[-1, -1]:.2f}K")
    print(f"    温度场范围: [{np.min(T_history):.2f}, {np.max(T_history):.2f}] K")
    print()




    print("[4] 高精度数值积分与误差分析")
    print("-" * 48)


    E_mean = 200e3
    sigma_E = 25e3
    T_test = 600.0
    k_eff_daem = integrate_daem_activation_energy(E_mean, sigma_E, T_test, n_quad=16)
    print(f"    DAEM 有效反应因子 (T={T_test}K): {k_eff_daem:.6e}")


    def heat_release_profile(x, y):
        r = np.sqrt(x * x + y * y)
        return np.exp(-r * r / (0.2 * 0.2))

    avg_val, err_val = reactor_cross_section_average(heat_release_profile,
                                                       radius=reactor_radius,
                                                       n_samples=5000)
    print(f"    反应器截面热释放平均值 (MC, N=5000): {avg_val:.6e} ± {err_val:.6e}")


    errors = quadrature_error_analysis(lambda x: x ** 4, None, max_degree=8, alpha=0.0)
    print(f"    Gauss-Hermite 求积误差分析 (degree 0-8):")
    for deg, exact, quad, err in errors[:5]:
        print(f"      degree={deg}: exact={exact:.6e}, quad={quad:.6e}, err={err:.2e}")
    print()




    print("[5] 温度相关材料物性插值")
    print("-" * 48)

    props = default_biomass_properties()
    test_T = np.linspace(300.0, 900.0, 7)
    kappa_vals = props['kappa_interp'](test_T)
    Cp_vals = props['Cp_interp'](test_T)
    rho_vals = props['rho_interp'](test_T)
    print(f"    温度 [K]:    {'  '.join(f'{T:7.1f}' for T in test_T)}")
    print(f"    κ [W/m·K]:   {'  '.join(f'{k:7.4f}' for k in kappa_vals)}")
    print(f"    Cp [J/kg·K]: {'  '.join(f'{c:7.1f}' for c in Cp_vals)}")
    print(f"    ρ [kg/m³]:   {'  '.join(f'{r:7.1f}' for r in rho_vals)}")
    print()




    print("[6] 颗粒运动学与能量守恒")
    print("-" * 48)

    np_particles = 20
    ndim = 2
    box = np.array([1.0, 1.0], dtype=np.float64)
    traj, energy = simulate_particle_transport(np_particles, ndim, box,
                                                dt=0.01, n_steps=50,
                                                mass=1.0, temperature=500.0,
                                                interaction_type='sinsq')
    e0 = energy[0, 2]
    max_rel_err = np.max(np.abs((energy[:, 2] - e0) / e0)) if abs(e0) > 1e-10 else 0.0
    print(f"    颗粒数={np_particles}, 步数=50, dt=0.01s")
    print(f"    总能量相对误差: {max_rel_err:.6e}")
    print()




    print("[7] FEM 数据组装与结果输出")
    print("-" * 48)

    if len(p_refined) > 0 and len(t_refined) > 0:

        node_temp = props['kappa_interp'](np.ones(len(p_refined)) * 600.0)


        prefix = "reactor_result"
        node_file, elem_file, val_file = assemble_fem_data(
            prefix, p_refined, t_refined, node_temp
        )
        print(f"    节点文件: {node_file}")
        print(f"    单元文件: {elem_file}")
        print(f"    数值文件: {val_file}")


        tec_file = prefix + ".dat"
        write_tecplot_ascii(tec_file, p_refined, t_refined, node_temp,
                            var_names=["X", "Y", "Temperature"])
        print(f"    TECPLOT 文件: {tec_file}")


        M_lumped = compute_fem_mass_matrix(p_refined, t_refined)
        kappa_uniform = 0.15
        K_stiff = compute_fem_stiffness_matrix(p_refined, t_refined, kappa_uniform)
        print(f"    FEM 质量矩阵迹 (总质量): {np.sum(M_lumped):.6f}")
        print(f"    FEM 刚度矩阵条件数估计: {cond_number_estimate(K_stiff):.4e}")
    else:
        print("    警告: 网格为空，跳过 FEM 数据组装")
    print()




    print("[8] 综合结果汇总")
    print("-" * 48)
    print(f"    反应器半径: {reactor_radius} m")
    print(f"    网格节点数: {len(p_refined)}")
    print(f"    热解转化率 (RK4): {1.0 - y_rk4[-1, 0] - y_rk4[-1, 1] - y_rk4[-1, 2]:.4f}")
    print(f"    出口温度: {T_history[-1, -1]:.2f} K")
    print(f"    MD 能量守恒误差: {max_rel_err:.2e}")
    print(f"    DAEM 积分结果: {k_eff_daem:.4e}")
    print()
    print("=" * 72)
    print("模拟完成。")
    timestamp()
    print("=" * 72)


if __name__ == "__main__":
    main()
