"""
main.py
生物质热解反应器多物理场耦合模拟系统 — 统一入口

科学问题:
    化学工程：生物质热解反应器模拟
    
    本项目构建了一个多尺度、多物理场耦合的生物质热解反应器数值模拟平台，
    集成了以下核心物理过程：
    1. 复杂多组分热解反应动力学（Arrhenius 定律 + 分布式活化能模型）
    2. 反应器内非稳态传热（热传导-对流-反应热源耦合）
    3. 催化剂颗粒内部三维传热传质
    4. 颗粒尺度运动学与碰撞（Velocity Verlet 分子动力学）
    5. 高精度数值积分（Gauss-Hermite 求积 + Monte Carlo）
    6. 材料物性随温度变化的高阶插值
    7. 自适应非结构化网格生成与局部细化
    8. FEM 数据组装与结果输出

运行方式:
    python main.py
    （零参数运行，所有参数在代码内部设定）
"""

import numpy as np
import os
import sys

# 将当前目录加入路径
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

    # ============================================================
    # 1. 反应器网格生成 (distmesh + triangulation_refine_local)
    # ============================================================
    print("[1] 反应器几何与网格生成")
    print("-" * 48)

    reactor_radius = 0.5  # m
    reactor_bbox = np.array([[-0.7, -0.7], [0.7, 0.7]], dtype=np.float64)

    def reactor_fd(p):
        return cylinder_distance_function(p, radius=reactor_radius)

    p, t = distmesh_2d(reactor_fd, lambda p: np.ones(p.shape[0]),
                        h0=0.12, bbox=reactor_bbox, max_iter=80)
    print(f"    初始网格: {len(p)} 个节点, {len(t)} 个三角形单元")

    if len(t) > 0:
        # 局部细化质量最差的单元
        qualities = compute_mesh_quality(p, t)
        worst_elem = np.argmin(qualities)
        p_refined, t_refined = triangulation_refine_local(p, t, worst_elem)
        print(f"    局部细化最差单元 (质量={qualities[worst_elem]:.4f}) 后: "
              f"{len(p_refined)} 个节点, {len(t_refined)} 个单元")
    else:
        p_refined, t_refined = p, t
        print("    警告: 初始网格为空，跳过局部细化")

    # 催化剂颗粒球体网格
    ball_points = ball_grid(n=3, r=0.05, c=np.array([0.0, 0.0, 0.0]))
    print(f"    催化剂颗粒内部网格点: {len(ball_points)} 个")
    print()

    # ============================================================
    # 2. 热解反应动力学 (RK4 + Midpoint + Doughnut flow)
    # ============================================================
    print("[2] 热解反应动力学求解")
    print("-" * 48)

    kinetics = BiomassPyrolysisKinetics()
    tspan = [0.0, 60.0]  # 60 秒
    n_steps = 200

    # 线性升温: T(t) = T0 + beta * t  (beta 单位 K/s)
    T0 = 300.0
    beta = 10.0  # K/s

    def T_profile(t):
        return T0 + beta * t

    # RK4 求解
    t_rk4, y_rk4 = kinetics.solve_rk4(tspan, kinetics.y0, n_steps, T_profile)
    print(f"    RK4 求解完成: {n_steps} 步, 终态质量分数:")
    print(f"      Biomass={y_rk4[-1, 0]:.6f}, Hemicellulose={y_rk4[-1, 1]:.6f}, "
          f"Lignin={y_rk4[-1, 2]:.6f}")
    print(f"      Active={y_rk4[-1, 3]:.6f}, Volatiles={y_rk4[-1, 4]:.6f}, "
          f"Char={y_rk4[-1, 5]:.6f}, Tar+Gas={y_rk4[-1, 6]:.6f}")

    # 隐式中点法求解
    t_mp, y_mp = kinetics.solve_midpoint(tspan, kinetics.y0, n_steps // 2, T_profile)
    print(f"    隐式中点法求解完成: {n_steps // 2} 步")

    # 环面反应器流动（非线性动力学）
    t_torus, y_torus = solve_doughnut_flow_rk4([0.0, 10.0], [1.0, 0.0, 0.0], 100)
    print(f"    环面反应器流动模拟完成: 100 步, 终态=[{y_torus[-1, 0]:.4f}, "
          f"{y_torus[-1, 1]:.4f}, {y_torus[-1, 2]:.4f}]")
    print()

    # ============================================================
    # 3. 传热模型求解 (Banded Matrix Solver)
    # ============================================================
    print("[3] 反应器一维传热模型")
    print("-" * 48)

    thermal = ThermalReactorModel(L=1.0, nx=40, rho=200.0, Cp=1500.0,
                                   k_eff=0.15, u=0.05)
    T_init = np.full(thermal.nx, 300.0, dtype=np.float64)
    dt = 0.5
    n_thermal_steps = 40

    def Q_source_func(t, x):
        T_local = min(T0 + beta * t, 900.0)
        y_current = kinetics.y0  # 简化: 使用初始组成
        # 热解为吸热反应，热源项为负（吸收热量）
        return compute_reaction_heat_source(x, T_local, kinetics, y_current,
                                            reaction_enthalpy=+200e3)

    t_thermal, T_history = thermal.simulate(T_init, dt, n_thermal_steps,
                                            Q_source_func, T_inlet=350.0)
    print(f"    传热模拟完成: {n_thermal_steps} 步, Δt={dt}s")
    print(f"    入口温度=350K, 出口温度={T_history[-1, -1]:.2f}K")
    print(f"    温度场范围: [{np.min(T_history):.2f}, {np.max(T_history):.2f}] K")
    print()

    # ============================================================
    # 4. 数值积分与误差分析 (Gauss-Hermite + Monte Carlo)
    # ============================================================
    print("[4] 高精度数值积分与误差分析")
    print("-" * 48)

    # DAEM 活化能分布积分
    E_mean = 200e3  # J/mol
    sigma_E = 25e3
    T_test = 600.0
    k_eff_daem = integrate_daem_activation_energy(E_mean, sigma_E, T_test, n_quad=16)
    print(f"    DAEM 有效反应因子 (T={T_test}K): {k_eff_daem:.6e}")

    # Monte Carlo 截面平均
    def heat_release_profile(x, y):
        r = np.sqrt(x * x + y * y)
        return np.exp(-r * r / (0.2 * 0.2))

    avg_val, err_val = reactor_cross_section_average(heat_release_profile,
                                                       radius=reactor_radius,
                                                       n_samples=5000)
    print(f"    反应器截面热释放平均值 (MC, N=5000): {avg_val:.6e} ± {err_val:.6e}")

    # 求积规则精确度分析
    errors = quadrature_error_analysis(lambda x: x ** 4, None, max_degree=8, alpha=0.0)
    print(f"    Gauss-Hermite 求积误差分析 (degree 0-8):")
    for deg, exact, quad, err in errors[:5]:
        print(f"      degree={deg}: exact={exact:.6e}, quad={quad:.6e}, err={err:.2e}")
    print()

    # ============================================================
    # 5. 材料物性插值 (Barycentric Interpolation)
    # ============================================================
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

    # ============================================================
    # 6. 颗粒运动学模拟 (MD Velocity Verlet)
    # ============================================================
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

    # ============================================================
    # 7. FEM 数据组装与输出
    # ============================================================
    print("[7] FEM 数据组装与结果输出")
    print("-" * 48)

    if len(p_refined) > 0 and len(t_refined) > 0:
        # 在网格节点上插值温度
        node_temp = props['kappa_interp'](np.ones(len(p_refined)) * 600.0)

        # 组装 FEM 数据
        prefix = "reactor_result"
        node_file, elem_file, val_file = assemble_fem_data(
            prefix, p_refined, t_refined, node_temp
        )
        print(f"    节点文件: {node_file}")
        print(f"    单元文件: {elem_file}")
        print(f"    数值文件: {val_file}")

        # TECPLOT 输出
        tec_file = prefix + ".dat"
        write_tecplot_ascii(tec_file, p_refined, t_refined, node_temp,
                            var_names=["X", "Y", "Temperature"])
        print(f"    TECPLOT 文件: {tec_file}")

        # FEM 质量矩阵与刚度矩阵
        M_lumped = compute_fem_mass_matrix(p_refined, t_refined)
        kappa_uniform = 0.15
        K_stiff = compute_fem_stiffness_matrix(p_refined, t_refined, kappa_uniform)
        print(f"    FEM 质量矩阵迹 (总质量): {np.sum(M_lumped):.6f}")
        print(f"    FEM 刚度矩阵条件数估计: {cond_number_estimate(K_stiff):.4e}")
    else:
        print("    警告: 网格为空，跳过 FEM 数据组装")
    print()

    # ============================================================
    # 8. 综合结果汇总
    # ============================================================
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

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: safe_exp 基本功能测试 ----
from utils import safe_exp
result = safe_exp(0.0)
assert isinstance(result, np.ndarray), '[TC01] safe_exp(0.0) 应返回 numpy 数组 FAILED'
assert abs(float(result) - 1.0) < 1e-10, '[TC01] safe_exp(0.0) 应等于 1.0 FAILED'

# ---- TC02: safe_exp 数值稳定性 (极大负数不应产生 NaN/Inf) ----
from utils import safe_exp
result = safe_exp(-1000.0)
assert np.all(np.isfinite(result)), '[TC02] safe_exp(-1000.0) 应返回有限值 FAILED'
assert float(result) >= 0.0, '[TC02] safe_exp(-1000.0) 应非负 FAILED'

# ---- TC03: safe_divide 基本除法 ----
from utils import safe_divide
result = safe_divide(10.0, 2.0)
assert abs(float(result) - 5.0) < 1e-10, '[TC03] safe_divide(10, 2) 应等于 5.0 FAILED'

# ---- TC04: safe_divide 除以零保护 ----
from utils import safe_divide
result = safe_divide(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
assert abs(float(result[0]) - 0.0) < 1e-10, '[TC04] 除以零应返回 fill_value=0.0 FAILED'
assert abs(float(result[1]) - 1.0) < 1e-10, '[TC04] safe_divide(2, 2) 应等于 1.0 FAILED'

# ---- TC05: check_bounds 边界内不裁剪 ----
from utils import check_bounds
result = check_bounds(np.array([0.5]), 0.0, 1.0, name="test")
assert abs(float(result[0]) - 0.5) < 1e-10, '[TC05] 边界内值不应被修改 FAILED'

# ---- TC06: check_bounds 越界裁剪 ----
from utils import check_bounds
result = check_bounds(np.array([-0.5, 1.5]), 0.0, 1.0, name="test")
assert abs(float(result[0]) - 0.0) < 1e-10, '[TC06] 下界裁剪应为 0.0 FAILED'
assert abs(float(result[1]) - 1.0) < 1e-10, '[TC06] 上界裁剪应为 1.0 FAILED'

# ---- TC07: cond_number_estimate 单位矩阵条件数 ----
from utils import cond_number_estimate
A = np.eye(5, dtype=np.float64)
cond = cond_number_estimate(A)
assert abs(cond - 1.0) < 0.01, '[TC07] 单位矩阵条件数应≈1.0 FAILED'

# ---- TC08: rotation_matrix_2d 正交性检验 ----
from geometry_utils import rotation_matrix_2d
R = rotation_matrix_2d(0.7)
RtR = R.T @ R
assert np.allclose(RtR, np.eye(2), atol=1e-10), '[TC08] 旋转矩阵应正交 FAILED'

# ---- TC09: cylinder_distance_function 边界点距离应为 0 ----
from geometry_utils import cylinder_distance_function
d = cylinder_distance_function(np.array([1.0, 0.0]), radius=1.0)
assert abs(float(d)) < 1e-10, '[TC09] 圆柱边界上点距离应为 0 FAILED'

# ---- TC10: rectangle_distance_function 内部点负距离 ----
from geometry_utils import rectangle_distance_function
d = rectangle_distance_function(np.array([0.0, 0.0]), -1.0, 1.0, -1.0, 1.0)
assert float(d) < 0.0, '[TC10] 矩形内部点距离应为负 FAILED'

# ---- TC11: torus_distance_function 对称性 ----
from geometry_utils import torus_distance_function
d1 = torus_distance_function(np.array([1.0, 0.0, 0.0]), R_major=2.0, R_minor=0.8)
d2 = torus_distance_function(np.array([-1.0, 0.0, 0.0]), R_major=2.0, R_minor=0.8)
assert abs(float(d1) - float(d2)) < 1e-10, '[TC11] 环面距离函数 x 方向应对称 FAILED'

# ---- TC12: union_distance 和 intersect_distance 基本运算 ----
from geometry_utils import union_distance, intersect_distance, cylinder_distance_function
d_cyl1 = cylinder_distance_function(np.array([0.0, 0.0]), radius=1.0)
d_cyl2 = cylinder_distance_function(np.array([0.5, 0.0]), radius=1.0)
d_union = union_distance(d_cyl1, d_cyl2)
d_inter = intersect_distance(d_cyl1, d_cyl2)
assert float(d_union) < float(d_inter), '[TC12] 并集距离应≤交集距离 FAILED'

# ---- TC13: circumcenter 等边三角形外心应为重心 ----
from geometry_utils import circumcenter
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([0.5, np.sqrt(3.0) / 2.0])
cc = circumcenter(p1, p2, p3)
centroid = (p1 + p2 + p3) / 3.0
assert np.allclose(cc, centroid, atol=1e-10), '[TC13] 等边三角形外心应等于重心 FAILED'

# ---- TC14: huniform 返回值类型检查 ----
from reactor_mesh import huniform
v = huniform(np.array([0.5, 0.5]))
assert isinstance(v, np.ndarray) or isinstance(v, float), '[TC14] huniform 应返回标量或数组 FAILED'

# ---- TC15: BiomassPyrolysisKinetics.reaction_rates 温度单调性 ----
from pyrolysis_kinetics import BiomassPyrolysisKinetics
kin = BiomassPyrolysisKinetics()
k_low = kin.reaction_rates(400.0)
k_high = kin.reaction_rates(800.0)
k_low_sum = float(np.sum(k_low))
k_high_sum = float(np.sum(k_high))
assert k_high_sum > k_low_sum, '[TC15] 高温反应速率应大于低温 FAILED'

# ---- TC16: RK4 求解质量守恒 ----
from pyrolysis_kinetics import BiomassPyrolysisKinetics
import numpy as np
kin = BiomassPyrolysisKinetics()
t, y = kin.solve_rk4([0.0, 10.0], kin.y0, 50, lambda t: 500.0)
total_mass = np.sum(y[-1, :])
assert abs(total_mass - 1.0) < 1e-6, '[TC16] RK4 求解后总质量应≈1.0 FAILED'

# ---- TC17: generalized_hermite_integral 奇数指数应为 0 ----
from quadrature_integrator import generalized_hermite_integral
val = generalized_hermite_integral(3, 0.0)
assert abs(val) < 1e-15, '[TC17] 奇数指数 Hermite 积分应为 0 FAILED'

# ---- TC18: generalized_hermite_integral 偶数指数精确值 ----
from quadrature_integrator import generalized_hermite_integral
from scipy.special import gamma as scipy_gamma
val = generalized_hermite_integral(2, 0.0)
expected = scipy_gamma(1.5)
assert abs(val - expected) / max(abs(expected), 1e-15) < 1e-10, '[TC18] n=2, α=0 时 Hermite 积分应等于 Γ(1.5) FAILED'

# ---- TC19: integrate_daem_activation_energy 有限输出 ----
from quadrature_integrator import integrate_daem_activation_energy
val = integrate_daem_activation_energy(200e3, 25e3, 600.0)
assert np.isfinite(val), '[TC19] DAEM 积分应返回有限值 FAILED'
assert val > 0.0, '[TC19] DAEM 积分应为正 FAILED'

# ---- TC20: square01_monte_carlo_integrate 可复现性 ----
from quadrature_integrator import square01_monte_carlo_integrate
import numpy as np
np.random.seed(42)
v1, e1 = square01_monte_carlo_integrate(lambda x, y: x + y, 2000)
np.random.seed(42)
v2, e2 = square01_monte_carlo_integrate(lambda x, y: x + y, 2000)
assert abs(v1 - v2) < 1e-15, '[TC20] 相同种子的 MC 积分应可复现 FAILED'

# ---- TC21: sinsq_potential 值域检查 ----
from particle_dynamics import sinsq_potential
import numpy as np
vals = sinsq_potential(np.linspace(0.0, 5.0, 100))
assert np.min(vals) >= 0.0, '[TC21] sin² 势能应非负 FAILED'
assert np.max(vals) <= 1.0, '[TC21] sin² 势能应≤1.0 FAILED'

# ---- TC22: sinsq_force 力在零点应为 0 ----
from particle_dynamics import sinsq_force
f0 = sinsq_force(0.0)
assert abs(float(f0)) < 1e-10, '[TC22] r=0 处 sin² 力应为 0 FAILED'

# ---- TC23: compute_fem_mass_matrix 质量矩阵非负 ----
from fem_assembler import compute_fem_mass_matrix
import numpy as np
p_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
t_test = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
M = compute_fem_mass_matrix(p_test, t_test)
assert np.all(M >= 0.0), '[TC23] 质量矩阵元素应非负 FAILED'
assert np.sum(M) > 0.0, '[TC23] 质量矩阵总质量应为正 FAILED'

# ---- TC24: barycentric_interp_1d 节点处精确恢复 ----
from property_interpolator import barycentric_interp_1d, chebyshev1_nodes
import numpy as np
xd = chebyshev1_nodes(5, 0.0, 1.0)
yd = np.sin(xd)
yi = barycentric_interp_1d(xd, yd, xd)
assert np.allclose(yi, yd, atol=1e-10), '[TC24] 重心插值在节点处应精确恢复原值 FAILED'

# ---- TC25: ball_grid 所有点应在球内 ----
from reactor_mesh import ball_grid
import numpy as np
c = np.array([0.0, 0.0, 0.0])
r = 1.0
pts = ball_grid(4, r, c)
dists = np.sqrt(np.sum((pts - c) ** 2, axis=1))
assert np.all(dists <= r + 1e-10), '[TC25] ball_grid 所有点应在球内 (容差内) FAILED'

# ---- TC26: compute_local_temperature_from_kinetic 确定性输出 ----
from particle_dynamics import compute_local_temperature_from_kinetic
import numpy as np
vel = np.ones((2, 10), dtype=np.float64) * 100.0
T = compute_local_temperature_from_kinetic(vel, 1.0)
expected = 1.0 * (100.0**2 + 100.0**2) / (2.0 * 1.380649e-23)
assert abs(T - expected) / expected < 1e-10, '[TC26] 局部温度计算应正确 FAILED'

# ---- TC27: quadrature_error_analysis 返回结构正确 ----
from quadrature_integrator import quadrature_error_analysis
import numpy as np
errors = quadrature_error_analysis(lambda x: x**4, None, max_degree=4, alpha=0.0)
assert len(errors) == 5, '[TC27] 误差分析应返回 5 个 degree 条目 FAILED'
for deg, exact, quad, err in errors:
    assert isinstance(deg, int), f'[TC27] degree 应为 int FAILED'

# ---- TC28: compute_reaction_heat_source 有限输出 ----
from thermal_model import compute_reaction_heat_source
from pyrolysis_kinetics import BiomassPyrolysisKinetics
import numpy as np
kin = BiomassPyrolysisKinetics()
x = np.array([0.0, 0.5, 1.0])
Q = compute_reaction_heat_source(x, 600.0, kin, kin.y0, reaction_enthalpy=-500e3)
assert np.all(np.isfinite(Q)), '[TC28] 反应热源应返回有限值 FAILED'

# ---- TC29: solve_doughnut_flow_rk4 输出形状正确 ----
from pyrolysis_kinetics import solve_doughnut_flow_rk4
import numpy as np
t, y = solve_doughnut_flow_rk4([0.0, 5.0], [1.0, 0.0, 0.0], 50)
assert t.shape == (51,), '[TC29] 时间数组形状应为 (51,) FAILED'
assert y.shape == (51, 3), '[TC29] 解数组形状应为 (51, 3) FAILED'

# ---- TC30: ThermalReactorModel 模拟输出温度范围合理 ----
from thermal_model import ThermalReactorModel
import numpy as np
def q_zero(t, x):
    return np.zeros_like(x, dtype=np.float64)
thermal = ThermalReactorModel(L=1.0, nx=20, rho=200.0, Cp=1500.0, k_eff=0.15, u=0.05)
T_init = np.full(20, 300.0)
t_hist, T_hist = thermal.simulate(T_init, 0.5, 10, q_zero, T_inlet=350.0)
assert np.all(T_hist >= 250.0), '[TC30] 温度不应低于 250K FAILED'
assert np.all(T_hist <= 1500.0), '[TC30] 温度不应高于 1500K FAILED'

# ---- TC31: 反应器截面 MC 积分 seed 可复现性 ----
from quadrature_integrator import reactor_cross_section_average
import numpy as np
np.random.seed(123)
def f(x, y):
    return x * x + y * y
v1, e1 = reactor_cross_section_average(f, radius=0.5, n_samples=3000)
np.random.seed(123)
v2, e2 = reactor_cross_section_average(f, radius=0.5, n_samples=3000)
assert abs(v1 - v2) < 1e-15, '[TC31] 反应器截面 MC 积分应可复现 FAILED'

# ---- TC32: default_biomass_properties 插值器可用性 ----
from property_interpolator import default_biomass_properties
import numpy as np
props = default_biomass_properties()
T_test = np.array([450.0])
k = props['kappa_interp'](T_test)
cp = props['Cp_interp'](T_test)
r = props['rho_interp'](T_test)
assert np.all(np.isfinite(k)), '[TC32] κ 插值应返回有限值 FAILED'
assert np.all(np.isfinite(cp)), '[TC32] Cp 插值应返回有限值 FAILED'
assert np.all(np.isfinite(r)), '[TC32] ρ 插值应返回有限值 FAILED'

# ---- TC33: distmesh_2d 网格生成基本检查 ----
from reactor_mesh import distmesh_2d
from geometry_utils import cylinder_distance_function
import numpy as np
def fd_small(p):
    return cylinder_distance_function(p, radius=0.3)
p, t = distmesh_2d(fd_small, lambda p: np.ones(p.shape[0]), h0=0.15,
                   bbox=np.array([[-0.5, -0.5], [0.5, 0.5]], dtype=np.float64), max_iter=40)
assert len(p) > 0, '[TC33] 网格应生成至少 1 个节点 FAILED'
if len(t) > 0:
    assert t.shape[1] == 3, '[TC33] 三角形单元应含 3 个节点 FAILED'

# ---- TC34: compute_mesh_quality 质量范围 ----
from reactor_mesh import compute_mesh_quality
import numpy as np
p_test2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]], dtype=np.float64)
t_test2 = np.array([[0, 1, 2]], dtype=np.int64)
q = compute_mesh_quality(p_test2, t_test2)
assert len(q) > 0, '[TC34] 质量数组不应为空 FAILED'
assert 0.0 <= float(q[0]) <= 1.0, '[TC34] 网格质量应在 [0, 1] 范围 FAILED'

# ---- TC35: triangulation_refine_local 局部细化后节点增加 ----
from reactor_mesh import triangulation_refine_local
import numpy as np
node_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [0.2, 0.3]], dtype=np.float64)
elem = np.array([[0, 1, 2]], dtype=np.int64)
new_nodes, new_elem = triangulation_refine_local(node_xy, elem, 0)
assert new_nodes.shape[0] == node_xy.shape[0] + 3, '[TC35] 局部细化应新增 3 个节点 FAILED'
assert new_elem.shape[0] == elem.shape[0] + 3, '[TC35] 局部细化应新增 3 个单元 FAILED'

print('\n全部 35 个测试通过!\n')
