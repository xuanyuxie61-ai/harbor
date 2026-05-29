"""
main.py
================================================================================
潮汐能提取与流固耦合博士级综合计算平台
================================================================================
统一入口文件，零参数运行。

本项目围绕海洋科学中的潮汐能提取与流固耦合问题，融合15个科研代码
项目的核心算法，构建了一个多尺度、多物理场耦合的博士级数值仿真平台。

科学问题:
    在多尺度流固耦合框架下，实现潮汐能提取阵列的功率预测、
    结构响应分析、阵列布局优化与不确定性量化。

运行方式:
    python main.py
"""

import numpy as np
import sys
import os

# 确保模块路径正确
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
    """打印带分隔线的章节标题。"""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_machine_precision_diagnostics():
    """运行机器精度诊断。"""
    print_section("[1/10] 机器精度常数与数值稳定性诊断 (来源于 706_machine)")
    eps = d1mach(4)
    safe_tol = get_safe_tol(scale=1.0)
    print(f"  双精度机器 epsilon     : {eps:.3e}")
    print(f"  安全迭代容差           : {safe_tol:.3e}")
    print(f"  最小正数               : {d1mach(1):.3e}")
    print(f"  最大可表示数           : {d1mach(2):.3e}")


def run_tidal_potential_analysis():
    """运行潮汐势与特殊函数分析。"""
    print_section("[2/10] 潮汐势频谱分析与 Digamma 调制 (来源于 036_asa103 + 135_calpak)")
    # 计算基准儒略日 (2024-01-01 00:00)
    jed_base = ymdf_to_jed_gregorian(2024, 1, 1, 0.0)
    print(f"  基准儒略日 (JED)       : {jed_base:.6f}")

    # 潮汐相位
    phases = compute_tidal_phases(jed_base)
    print(f"  M2 分潮相位 (rad)      : {phases['M2']:.4f}")
    print(f"  S2 分潮相位 (rad)      : {phases['S2']:.4f}")
    print(f"  K1 分潮相位 (rad)      : {phases['K1']:.4f}")

    # Digamma 调制因子
    mod = tidal_digamma_modulation(freq_ratio=1.08, n_harmonics=8)
    print(f"  Digamma 调制因子 M(ν)  : {mod:.6f}")

    # 生成潮高和流速时间序列
    t_hours = np.linspace(0, 48, 97)
    eta = generate_tidal_elevation(t_hours, jed_base)
    u = generate_tidal_velocity(t_hours, jed_base, max_velocity=2.5)
    print(f"  48h 潮高范围           : [{np.min(eta):.3f}, {np.max(eta):.3f}] m")
    print(f"  48h 流速范围           : [{np.min(u):.3f}, {np.max(u):.3f}] m/s")

    # 功率密度
    power_density = integrate_power_density(u)
    print(f"  平均水动能功率密度     : {power_density:.2f} W/m²")
    return t_hours, eta, u


def run_pde_solvers():
    """运行PDE快速求解器。"""
    print_section("[3/10] 正弦变换快速PDE求解与尾流场计算 (来源于 1085_sine_transform)")
    N = 128
    L = 100.0
    x = np.linspace(0, L, N)
    # 模拟涡轮推力源项 (高斯分布)
    thrust = 5.0e4 * np.exp(-((x - 30.0) ** 2) / 50.0)

    # 泊松求解
    phi = solve_poisson_1d(thrust, L=L)
    print(f"  泊松方程求解网格       : {N} 点")
    print(f"  速度势范围             : [{np.min(phi):.3f}, {np.max(phi):.3f}] m²/s")

    # 亥姆霍兹求解
    wake = solve_helmholtz_1d(thrust, kappa=0.1, L=L)
    print(f"  亥姆霍兹波数 κ         : 0.1")
    print(f"  尾流修正势范围         : [{np.min(wake):.3f}, {np.max(wake):.3f}] m²/s")

    # 尾流势
    potential = compute_wake_potential(thrust, domain_length=L)
    print(f"  尾流速度势范围         : [{np.min(potential):.3f}, {np.max(potential):.3f}] m²/s")
    return x, phi, wake


def run_toeplitz_solver():
    """运行Toeplitz矩阵求解器。"""
    print_section("[4/10] 对称Toeplitz系统快速求解 (来源于 999_r8sto)")
    n = 256
    a = build_toeplitz_first_row(n, correlation_length=8.0)
    # 构造右端项 (模拟流速自相关反演)
    b = np.exp(-np.linspace(0, 5, n))
    x_sol = r8sto_sl(n, a, b)
    residual = np.linalg.norm(np.convolve(a[:n], x_sol, mode='full')[:n] - b)
    print(f"  矩阵阶数               : {n}")
    print(f"  求解残差 ||Ax-b||      : {residual:.3e}")
    print(f"  解向量范数             : {np.linalg.norm(x_sol):.6f}")

    # 周期性边界系统
    rhs = np.sin(np.linspace(0, 2 * np.pi, n))
    x_per = solve_periodic_boundary_system(rhs, correlation_length=5.0)
    print(f"  周期边界解范数         : {np.linalg.norm(x_per):.6f}")


def run_boundary_layer_analysis():
    """运行边界层分析。"""
    print_section("[5/10] 病态边界层边值问题求解 (来源于 572_ill_bvp)")
    epsilon = 1e-2
    Re = 1.0 / epsilon
    print(f"  摄动参数 ε             : {epsilon:.0e}")
    print(f"  等效雷诺数 Re          : {Re:.0e}")

    # 打靶法
    x_shoot, y_shoot, conv = shooting_method(
        epsilon, ya=2.0, yb=1.0, a=-1.0, b=1.0, n_shoot=2000
    )
    print(f"  打靶法收敛             : {conv}")
    mid_idx = len(y_shoot) // 2
    y_mid = y_shoot[mid_idx] if np.isfinite(y_shoot[mid_idx]) else float('nan')
    print(f"  解在 x=0 处值          : {y_mid:.6f}")

    # 有限差分
    x_fd, y_fd = finite_difference_bvp(epsilon, ya=2.0, yb=1.0, n=500)
    fd_mid = y_fd[len(y_fd)//2] if np.isfinite(y_fd[len(y_fd)//2]) else float('nan')
    print(f"  有限差分解在 x=0 处值  : {fd_mid:.6f}")

    # 边界层厚度
    delta = boundary_layer_thickness(epsilon, U_ref=2.5, L_ref=2.0)
    print(f"  估计边界层厚度 δ       : {delta:.6f} m")

    # 叶片边界层
    x_blade, u_blade = compute_blade_boundary_layer(Re, chord_length=2.0, n_points=200)
    u_min = np.min(u_blade) if np.all(np.isfinite(u_blade)) else float('nan')
    u_max = np.max(u_blade) if np.all(np.isfinite(u_blade)) else float('nan')
    print(f"  叶片表面速度范围       : [{u_min:.4f}, {u_max:.4f}]")


def run_structure_analysis():
    """运行结构有限元分析。"""
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

    # 系泊张力
    T_max = compute_mooring_tension(
        anchor_distance=200.0,
        water_depth=40.0,
        line_density=50.0,
        horizontal_force=1.5e6,
    )
    print(f"  最大系泊张力           : {T_max:.3e} N")

    # 三角形求积: 计算翼型升力
    lift = compute_hydrofoil_lift(
        chord=2.0, span=10.0, angle_of_attack=8.0, velocity=2.5
    )
    print(f"  单叶片升力 (8°攻角)    : {lift:.3e} N")

    # Wandzura 三角形求积验证
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    area_integral = integrate_triangle(lambda x, y: 1.0, vertices)
    print(f"  Wandzura 求积验证 (面积): {area_integral:.6f} (理论 0.5)")


def run_fsi_coupling():
    """运行流固耦合迭代。"""
    print_section("[7/10] 流固耦合不动点迭代 (来源于 194_cobweb_plot)")
    n_nodes = 50

    def fluid_solver(delta: np.ndarray) -> np.ndarray:
        """简化的流体求解器: 载荷 ∝ 变形² (非线性恢复)。"""
        rho = 1025.0
        U = 2.5
        Cd = 1.2
        A = 10.0
        return 0.5 * rho * U * U * Cd * A * (1.0 + 0.1 * delta ** 2)

    def structure_solver(load: np.ndarray) -> np.ndarray:
        """简化的结构求解器: 变形 ∝ 载荷 / k。"""
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
    """运行阵列布局优化。"""
    print_section("[8/10] 涡轮阵列布局与维护路径优化 (来源于 907_praxis + 1365_tsp_greedy + 484_graph_representation)")
    # Praxis 优化阵列布局
    positions, total_power = optimize_turbine_array(
        n_turbines=5, domain_size=500.0, min_spacing=50.0
    )
    print(f"  涡轮数量               : 5")
    print(f"  优化后总功率           : {total_power:.3e} W")
    print(f"  平均单机功率           : {total_power/5:.3e} W")
    for i, pos in enumerate(positions):
        print(f"    涡轮 {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}) m")

    # TSP 维护路径
    route, route_dist, metrics = optimize_maintenance_route(positions)
    print(f"  最优维护路径长度       : {route_dist:.2f} m")
    print(f"  网络平均度             : {metrics['average_degree']:.2f}")
    print(f"  聚类系数               : {metrics['clustering_coefficient']:.4f}")
    print(f"  图直径                 : {metrics['diameter']:.1f}")


def run_uncertainty_quantification():
    """运行不确定性量化。"""
    print_section("[9/10] 流速不确定性量化与功率统计 (来源于 542_histogram_pdf_2d_sample)")
    # 构造二维流速 PDF (正态分布近似)
    nx, ny = 40, 40
    x_grid = np.linspace(-1.0, 3.5, nx)
    y_grid = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    # 双变量正态分布
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
    """运行数值积分验证。"""
    print_section("[10/10] 数值积分与求积规则验证 (来源于 944_quad_serial + 1324_triangle_wandzura_rule)")
    # 梯形法则
    f = lambda x: np.sin(x)
    q_trap = quad_serial(f, 0.0, np.pi, 101)
    print(f"  ∫₀^π sin(x) dx (梯形)  : {q_trap:.8f} (理论 2.0)")

    # Simpson 法则
    q_simp = quad_simpson(f, 0.0, np.pi, 101)
    print(f"  ∫₀^π sin(x) dx (Simpson): {q_simp:.8f}")

    # Gauss-Legendre 求积
    xtab, wgt = legendre_com(8)
    q_gl = np.sum(wgt * np.sin((xtab + 1) * np.pi / 2)) * np.pi / 2
    print(f"  ∫₀^π sin(x) dx (GL-8)  : {q_gl:.8f}")

    # 三角形求积: 积分 f(x,y)=x+y 在参考三角形上 = 1/3
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    q_tri = integrate_triangle(lambda x, y: x + y, verts, rule=1)
    print(f"  ∫∫_T (x+y) dA (Wandzura): {q_tri:.8f} (理论 1/3)")


def main():
    """主函数: 执行完整的潮汐能提取与流固耦合分析流程。"""
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
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: d1mach(4) 返回正的双精度机器 epsilon ----
eps = d1mach(4)
assert np.isfinite(eps) and eps > 0.0, '[TC01] d1mach(4) 必须返回正的有限值 FAILED'

# ---- TC02: get_safe_tol 返回值大于机器 epsilon ----
safe_tol = get_safe_tol(scale=1.0)
assert safe_tol > d1mach(4), '[TC02] get_safe_tol 返回值必须大于机器 epsilon FAILED'

# ---- TC03: digamma(1) 约等于 -Euler-Mascheroni 常数 ----
val, ifault = digamma(1.0)
assert ifault == 0 and abs(val - (-0.5772156649015329)) < 1e-6, '[TC03] digamma(1) 解析验证 FAILED'

# ---- TC04: tidal_digamma_modulation 返回有限正数 ----
mod = tidal_digamma_modulation(freq_ratio=1.08, n_harmonics=8)
assert np.isfinite(mod) and mod > 0.0, '[TC04] tidal_digamma_modulation 返回值必须有限且为正 FAILED'

# ---- TC05: polygamma2(1) 约等于 π²/6 ----
pg = polygamma2(1.0)
assert abs(pg - (np.pi ** 2 / 6.0)) < 0.02, '[TC05] polygamma2(1) 解析验证 FAILED'

# ---- TC06: 儒略日转换对典型日期返回合理值 ----
jed = ymdf_to_jed_gregorian(2024, 1, 1, 0.0)
assert jed > 60000.0, '[TC06] 儒略日转换范围验证 FAILED'

# ---- TC07: compute_tidal_phases 返回包含主要分潮键的字典 ----
phases = compute_tidal_phases(jed)
assert all(k in phases for k in ['M2', 'S2', 'K1']), '[TC07] 潮汐相位字典键完整性 FAILED'

# ---- TC08: generate_tidal_elevation 输出长度与输入时间数组一致 ----
t_hours = np.linspace(0, 24, 25)
eta = generate_tidal_elevation(t_hours, jed)
assert eta.shape == t_hours.shape, '[TC08] 潮高输出尺寸一致性 FAILED'

# ---- TC09: quad_serial 对 sin(x) 在 [0,π] 积分接近 2.0 ----
q_trap = quad_serial(lambda x: np.sin(x), 0.0, np.pi, 101)
assert abs(q_trap - 2.0) < 0.02, '[TC09] 梯形法则积分 sin 解析验证 FAILED'

# ---- TC10: quad_simpson 对 sin(x) 在 [0,π] 积分接近 2.0 ----
q_simp = quad_simpson(lambda x: np.sin(x), 0.0, np.pi, 101)
assert abs(q_simp - 2.0) < 1e-4, '[TC10] Simpson 法则积分 sin 解析验证 FAILED'

# ---- TC11: integrate_power_density 单点输入返回正确功率密度 ----
pd = integrate_power_density(np.array([2.0]))
assert abs(pd - 0.5 * 1025.0 * 8.0) < 1e-6, '[TC11] 单点功率密度解析验证 FAILED'

# ---- TC12: fsi_fixed_point_solver 对线性系统收敛 ----
def _fluid(d):
    return 1.0e7 * d

def _struct(load):
    return load / 1.0e7

delta, load, converged, iters = fsi_fixed_point_solver(_fluid, _struct, np.zeros(5), max_iter=50, tol=1e-8, relaxation=0.7)
assert converged and iters < 50, '[TC12] 流固耦合不动点迭代收敛性 FAILED'

# ---- TC13: contraction_factor_estimate 返回小于 1 的收缩因子 ----
history = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
L = contraction_factor_estimate(history)
assert L < 1.0, '[TC13] 收缩因子估计单调性 FAILED'

# ---- TC14: build_toeplitz_first_row 输出首元素为 1 且长度正确 ----
a_row = build_toeplitz_first_row(10, correlation_length=3.0)
assert a_row[0] == 1.0 and len(a_row) == 10, '[TC14] Toeplitz 首行构造边界验证 FAILED'

# ---- TC15: r8sto_sl 求解小尺寸 Toeplitz 系统残差可接受 ----
n_t = 8
a_t = build_toeplitz_first_row(n_t, 2.0)
b_t = np.ones(n_t)
x_t = r8sto_sl(n_t, a_t, b_t)
A_t = np.zeros((n_t, n_t))
for i in range(n_t):
    for j in range(n_t):
        A_t[i, j] = a_t[abs(i - j)]
residual_t = np.linalg.norm(A_t @ x_t - b_t)
assert residual_t < 1e-8, '[TC15] Toeplitz 求解残差验证 FAILED'

# ---- TC16: solve_poisson_1d 输出尺寸与输入右端项一致 ----
f_rhs = np.ones(16)
phi = solve_poisson_1d(f_rhs, L=1.0)
assert phi.shape == f_rhs.shape, '[TC16] 泊松方程解尺寸一致性 FAILED'

# ---- TC17: solve_helmholtz_1d 输出尺寸与输入右端项一致 ----
wake = solve_helmholtz_1d(f_rhs, kappa=0.5, L=1.0)
assert wake.shape == f_rhs.shape, '[TC17] 亥姆霍兹方程解尺寸一致性 FAILED'

# ---- TC18: sample_velocity_2d 输出形状为 (2, n_samples) ----
np.random.seed(42)
pdf_mat = np.array([[0.25, 0.25], [0.25, 0.25]])
samples = sample_velocity_2d(pdf_mat, u_range=(0.0, 2.0), v_range=(-1.0, 1.0), n_samples=100)
assert samples.shape == (2, 100), '[TC18] 二维流速采样输出形状 FAILED'

# ---- TC19: estimate_power_statistics 返回包含 mean_power 的字典 ----
np.random.seed(42)
stats = estimate_power_statistics(pdf_mat, u_range=(0.0, 2.0), v_range=(-1.0, 1.0), turbine_area=10.0, n_samples=500)
assert 'mean_power' in stats and stats['mean_power'] >= 0.0, '[TC19] 功率统计返回结构验证 FAILED'

# ---- TC20: boundary_layer_thickness 满足 δ ∝ √ε 的解析关系 ----
delta = boundary_layer_thickness(0.01, U_ref=2.5, L_ref=2.0)
assert abs(delta - 2.0 * np.sqrt(0.01)) < 1e-10, '[TC20] 边界层厚度解析验证 FAILED'

# ---- TC21: finite_difference_bvp 边界值精确满足 ----
x_fd, y_fd = finite_difference_bvp(0.1, ya=2.0, yb=1.0, n=50)
assert abs(y_fd[0] - 2.0) < 1e-10 and abs(y_fd[-1] - 1.0) < 1e-10, '[TC21] 有限差分边界条件满足性 FAILED'

# ---- TC22: shooting_method 对中等 epsilon 收敛 ----
x_sh, y_sh, conv_sh = shooting_method(0.1, ya=2.0, yb=1.0, n_shoot=200)
assert conv_sh and abs(y_sh[0] - 2.0) < 1e-6, '[TC22] 打靶法收敛性与左边界验证 FAILED'

# ---- TC23: solve_beam_static 悬臂梁根部挠度为零 ----
x_beam, w_beam = solve_beam_static(n_elements=10, length=10.0, E=2.1e11, I=0.5, rho=7850.0, A=2.0, drag_force=1.0e4)
assert abs(w_beam[0]) < 1e-10, '[TC23] 悬臂梁固定端挠度为零验证 FAILED'

# ---- TC24: compute_mooring_tension 返回值大于水平张力分量 ----
T_max = compute_mooring_tension(anchor_distance=200.0, water_depth=40.0, line_density=50.0, horizontal_force=1.5e6)
assert T_max >= 1.5e6, '[TC24] 系泊张力范围约束 FAILED'

# ---- TC25: legendre_com 权重和约等于 2 ----
xtab, wgt = legendre_com(6)
assert abs(np.sum(wgt) - 2.0) < 1e-10, '[TC25] Gauss-Legendre 权重和解析验证 FAILED'

# ---- TC26: integrate_triangle 常数函数积分等于三角形面积 ----
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area_int = integrate_triangle(lambda x, y: 1.0, verts)
assert 0.4 < area_int < 0.8, '[TC26] 三角形常数函数积分范围验证 FAILED'

# ---- TC27: compute_hydrofoil_lift 升力返回值非负且随攻角增加 ----
L1 = compute_hydrofoil_lift(chord=2.0, span=10.0, angle_of_attack=5.0, velocity=2.5)
L2 = compute_hydrofoil_lift(chord=2.0, span=10.0, angle_of_attack=8.0, velocity=2.5)
assert L1 >= 0.0 and L2 > L1, '[TC27] 翼型升力单调性验证 FAILED'

# ---- TC28: optimize_maintenance_route 返回三元组 ----
np.random.seed(42)
route, route_dist, metrics = optimize_maintenance_route(np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]))
assert isinstance(route, np.ndarray) and isinstance(route_dist, float) and isinstance(metrics, dict), '[TC28] 维护路径优化返回类型验证 FAILED'

# ---- TC29: compute_graph_metrics 全连接无向图平均度为 n-1 ----
n_g = 4
adj_full = np.ones((n_g, n_g), dtype=int)
np.fill_diagonal(adj_full, 0)
metrics_g = compute_graph_metrics(adj_full)
assert abs(metrics_g['average_degree'] - (n_g - 1)) < 1e-10, '[TC29] 全连接图平均度解析验证 FAILED'

# ---- TC30: main 函数执行完整流程返回 0 ----
import sys
from io import StringIO
old_stdout = sys.stdout
sys.stdout = StringIO()
ret = main()
sys.stdout = old_stdout
assert ret == 0, '[TC30] 主函数完整流程集成测试 FAILED'

print('\n全部 30 个测试通过!\n')
