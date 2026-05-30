
import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from machine_constants import d1mach, get_safe_tol
from special_functions import digamma, tidal_digamma_modulation, polygamma2
from tidal_calendar import (
    ymdf_to_jed_gregorian,
    compute_tidal_phases,
    generate_tidal_elevation,
    generate_tidal_velocity,
)
from serial_quadrature import quad_serial, quad_simpson, integrate_power_density
from cobweb_iterator import fsi_fixed_point_solver, contraction_factor_estimate
from toeplitz_solver import r8sto_sl, build_toeplitz_first_row, solve_periodic_boundary_system
from sine_transform_solver import solve_poisson_1d, solve_helmholtz_1d, compute_wake_potential
from pdf_sampler import sample_velocity_2d, estimate_power_statistics
from boundary_layer_solver import (
    shooting_method,
    finite_difference_bvp,
    boundary_layer_thickness,
    compute_blade_boundary_layer,
)
from fem1d_structure import solve_beam_static, compute_mooring_tension, legendre_com
from wandzura_quadrature import integrate_triangle, compute_hydrofoil_lift
from praxis_optimizer import praxis, optimize_turbine_array
from array_layout_optimizer import optimize_maintenance_route, compute_graph_metrics


def print_section(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_machine_precision_diagnostics():
    print_section("[1/10] 机器精度常数与数值稳定性诊断 (来源于 706_machine)")
    eps = d1mach(4)
    safe_tol = get_safe_tol(scale=1.0)
    print(f"  双精度机器 epsilon     : {eps:.3e}")
    print(f"  安全迭代容差           : {safe_tol:.3e}")
    print(f"  最小正数               : {d1mach(1):.3e}")
    print(f"  最大可表示数           : {d1mach(2):.3e}")


def run_tidal_potential_analysis():
    print_section("[2/10] 潮汐势频谱分析与 Digamma 调制 (来源于 036_asa103 + 135_calpak)")

    jed_base = ymdf_to_jed_gregorian(2024, 1, 1, 0.0)
    print(f"  基准儒略日 (JED)       : {jed_base:.6f}")


    phases = compute_tidal_phases(jed_base)
    print(f"  M2 分潮相位 (rad)      : {phases['M2']:.4f}")
    print(f"  S2 分潮相位 (rad)      : {phases['S2']:.4f}")
    print(f"  K1 分潮相位 (rad)      : {phases['K1']:.4f}")


    mod = tidal_digamma_modulation(freq_ratio=1.08, n_harmonics=8)
    print(f"  Digamma 调制因子 M(ν)  : {mod:.6f}")


    t_hours = np.linspace(0, 48, 97)
    eta = generate_tidal_elevation(t_hours, jed_base)
    u = generate_tidal_velocity(t_hours, jed_base, max_velocity=2.5)
    print(f"  48h 潮高范围           : [{np.min(eta):.3f}, {np.max(eta):.3f}] m")
    print(f"  48h 流速范围           : [{np.min(u):.3f}, {np.max(u):.3f}] m/s")


    power_density = integrate_power_density(u)
    print(f"  平均水动能功率密度     : {power_density:.2f} W/m²")
    return t_hours, eta, u


def run_pde_solvers():
    print_section("[3/10] 正弦变换快速PDE求解与尾流场计算 (来源于 1085_sine_transform)")
    N = 128
    L = 100.0
    x = np.linspace(0, L, N)

    thrust = 5.0e4 * np.exp(-((x - 30.0) ** 2) / 50.0)


    phi = solve_poisson_1d(thrust, L=L)
    print(f"  泊松方程求解网格       : {N} 点")
    print(f"  速度势范围             : [{np.min(phi):.3f}, {np.max(phi):.3f}] m²/s")


    wake = solve_helmholtz_1d(thrust, kappa=0.1, L=L)
    print(f"  亥姆霍兹波数 κ         : 0.1")
    print(f"  尾流修正势范围         : [{np.min(wake):.3f}, {np.max(wake):.3f}] m²/s")


    potential = compute_wake_potential(thrust, domain_length=L)
    print(f"  尾流速度势范围         : [{np.min(potential):.3f}, {np.max(potential):.3f}] m²/s")
    return x, phi, wake


def run_toeplitz_solver():
    print_section("[4/10] 对称Toeplitz系统快速求解 (来源于 999_r8sto)")
    n = 256
    a = build_toeplitz_first_row(n, correlation_length=8.0)

    b = np.exp(-np.linspace(0, 5, n))
    x_sol = r8sto_sl(n, a, b)
    residual = np.linalg.norm(np.convolve(a[:n], x_sol, mode='full')[:n] - b)
    print(f"  矩阵阶数               : {n}")
    print(f"  求解残差 ||Ax-b||      : {residual:.3e}")
    print(f"  解向量范数             : {np.linalg.norm(x_sol):.6f}")


    rhs = np.sin(np.linspace(0, 2 * np.pi, n))
    x_per = solve_periodic_boundary_system(rhs, correlation_length=5.0)
    print(f"  周期边界解范数         : {np.linalg.norm(x_per):.6f}")


def run_boundary_layer_analysis():
    print_section("[5/10] 病态边界层边值问题求解 (来源于 572_ill_bvp)")
    epsilon = 1e-2
    Re = 1.0 / epsilon
    print(f"  摄动参数 ε             : {epsilon:.0e}")
    print(f"  等效雷诺数 Re          : {Re:.0e}")


    x_shoot, y_shoot, conv = shooting_method(
        epsilon, ya=2.0, yb=1.0, a=-1.0, b=1.0, n_shoot=2000
    )
    print(f"  打靶法收敛             : {conv}")
    mid_idx = len(y_shoot) // 2
    y_mid = y_shoot[mid_idx] if np.isfinite(y_shoot[mid_idx]) else float('nan')
    print(f"  解在 x=0 处值          : {y_mid:.6f}")


    x_fd, y_fd = finite_difference_bvp(epsilon, ya=2.0, yb=1.0, n=500)
    fd_mid = y_fd[len(y_fd)//2] if np.isfinite(y_fd[len(y_fd)//2]) else float('nan')
    print(f"  有限差分解在 x=0 处值  : {fd_mid:.6f}")


    delta = boundary_layer_thickness(epsilon, U_ref=2.5, L_ref=2.0)
    print(f"  估计边界层厚度 δ       : {delta:.6f} m")


    x_blade, u_blade = compute_blade_boundary_layer(Re, chord_length=2.0, n_points=200)
    u_min = np.min(u_blade) if np.all(np.isfinite(u_blade)) else float('nan')
    u_max = np.max(u_blade) if np.all(np.isfinite(u_blade)) else float('nan')
    print(f"  叶片表面速度范围       : [{u_min:.4f}, {u_max:.4f}]")


def run_structure_analysis():
    print_section("[6/10] 支撑结构一维有限元分析 (来源于 395_fem1d_pack)")
    x_beam, w_beam = solve_beam_static(
        n_elements=30,
        length=30.0,
        E=2.1e11,
        I=0.5,
        rho=7850.0,
        A=2.0,
        drag_force=8.0e4,
    )
    print(f"  塔架高度               : 30.0 m")
    print(f"  顶部挠度               : {w_beam[-1]:.6f} m")
    print(f"  最大应力估算 (EI·w'')  : {2.1e11 * 0.5 * abs(w_beam[-1] - 2*w_beam[-2] + w_beam[-3]):.3e} Pa")


    T_max = compute_mooring_tension(
        anchor_distance=200.0,
        water_depth=40.0,
        line_density=50.0,
        horizontal_force=1.5e6,
    )
    print(f"  最大系泊张力           : {T_max:.3e} N")


    lift = compute_hydrofoil_lift(
        chord=2.0, span=10.0, angle_of_attack=8.0, velocity=2.5
    )
    print(f"  单叶片升力 (8°攻角)    : {lift:.3e} N")


    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    area_integral = integrate_triangle(lambda x, y: 1.0, vertices)
    print(f"  Wandzura 求积验证 (面积): {area_integral:.6f} (理论 0.5)")


def run_fsi_coupling():
    print_section("[7/10] 流固耦合不动点迭代 (来源于 194_cobweb_plot)")
    n_nodes = 50

    def fluid_solver(delta: np.ndarray) -> np.ndarray:
        rho = 1025.0
        U = 2.5
        Cd = 1.2
        A = 10.0
        return 0.5 * rho * U * U * Cd * A * (1.0 + 0.1 * delta ** 2)

    def structure_solver(load: np.ndarray) -> np.ndarray:
        k = 1.0e7
        return load / k

    delta0 = np.zeros(n_nodes)
    delta, load, converged, iters = fsi_fixed_point_solver(
        fluid_solver,
        structure_solver,
        delta0,
        max_iter=50,
        tol=1e-8,
        relaxation=0.6,
    )
    print(f"  耦合迭代收敛           : {converged}")
    print(f"  迭代次数               : {iters}")
    print(f"  最大结构变形           : {np.max(np.abs(delta)):.6f} m")
    print(f"  最大流体载荷           : {np.max(load):.3e} N")


def run_optimization():
    print_section("[8/10] 涡轮阵列布局与维护路径优化 (来源于 907_praxis + 1365_tsp_greedy + 484_graph_representation)")


    total_power, positions = optimize_turbine_array(
        n_turbines=5, domain_size=500.0, min_spacing=50.0
    )
    print(f"  涡轮数量               : 5")
    print(f"  优化后总功率           : {total_power:.3e} W")
    print(f"  平均单机功率           : {total_power/5:.3e} W")
    for i, pos in enumerate(positions):
        print(f"    涡轮 {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}) m")


    route, route_dist, metrics = optimize_maintenance_route(positions)
    print(f"  最优维护路径长度       : {route_dist:.2f} m")
    print(f"  网络平均度             : {metrics['average_degree']:.2f}")
    print(f"  聚类系数               : {metrics['clustering_coefficient']:.4f}")
    print(f"  图直径                 : {metrics['diameter']:.1f}")


def run_uncertainty_quantification():
    print_section("[9/10] 流速不确定性量化与功率统计 (来源于 542_histogram_pdf_2d_sample)")

    nx, ny = 40, 40
    x_grid = np.linspace(-1.0, 3.5, nx)
    y_grid = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    mu_u, sigma_u = 2.0, 0.5
    mu_v, sigma_v = 0.0, 0.2
    pdf = np.exp(-0.5 * ((X - mu_u) / sigma_u) ** 2 - 0.5 * ((Y - mu_v) / sigma_v) ** 2)
    pdf /= np.sum(pdf)

    stats = estimate_power_statistics(
        pdf, u_range=(-1.0, 3.5), v_range=(-1.0, 1.0),
        turbine_area=20.0, rho=1025.0, n_samples=5000
    )
    print(f"  平均功率输出           : {stats['mean_power']:.3e} W")
    print(f"  功率标准差             : {stats['std_power']:.3e} W")
    print(f"  最大功率               : {stats['max_power']:.3e} W")
    print(f"  额定功率               : {stats['rated_power']:.3e} W")
    print(f"  容量因子               : {stats['capacity_factor']:.4f}")


def run_integration_verification():
    print_section("[10/10] 数值积分与求积规则验证 (来源于 944_quad_serial + 1324_triangle_wandzura_rule)")

    f = lambda x: np.sin(x)
    q_trap = quad_serial(f, 0.0, np.pi, 101)
    print(f"  ∫₀^π sin(x) dx (梯形)  : {q_trap:.8f} (理论 2.0)")


    q_simp = quad_simpson(f, 0.0, np.pi, 101)
    print(f"  ∫₀^π sin(x) dx (Simpson): {q_simp:.8f}")


    xtab, wgt = legendre_com(8)
    q_gl = np.sum(wgt * np.sin((xtab + 1) * np.pi / 2)) * np.pi / 2
    print(f"  ∫₀^π sin(x) dx (GL-8)  : {q_gl:.8f}")


    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    q_tri = integrate_triangle(lambda x, y: x + y, verts, rule=1)
    print(f"  ∫∫_T (x+y) dA (Wandzura): {q_tri:.8f} (理论 1/3)")


def main():
    print("=" * 78)
    print("  潮汐能提取与流固耦合 — 博士级综合计算平台")
    print("  Tidal Energy Extraction & Fluid-Structure Interaction")
    print("=" * 78)

    np.random.seed(42)

    try:
        run_machine_precision_diagnostics()
        run_tidal_potential_analysis()
        run_pde_solvers()
        run_toeplitz_solver()
        run_boundary_layer_analysis()
        run_structure_analysis()
        run_fsi_coupling()
        run_optimization()
        run_uncertainty_quantification()
        run_integration_verification()
    except Exception as e:
        print(f"\n[ERROR] 计算过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 78)
    print("  全部计算流程已完成，无报错。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
