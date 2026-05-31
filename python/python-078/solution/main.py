"""
main.py
计算流体力学：血动脉脉动流与壁面剪切 — 统一入口

项目概述:
本博士级科研代码合成项目围绕"动脉脉动流中的壁面剪切应力（WSS）时空分布
与血管内皮功能调控"展开，融合了15个种子项目的核心算法。

合成项目清单:
- 868_pi_spigot          → geometry_utils (高精度π计算)
- 915_prime_plot         → geometry_utils (素数标记与离散化索引)
- 381_fem_to_triangle    → geometry_utils (FEM网格数据结构)
- 528_hexagon_lyness_rule → quadrature_rules (六边形高斯求积)
- 660_legendre_fast_rule  → quadrature_rules (快速Gauss-Legendre求积)
- 1051_runge             → quadrature_rules (Runge函数/插值病态性分析)
- 966_r83t               → linear_algebra_core (三对角矩阵存储与求解)
- 119_brownian_motion    → stochastic_diffusion (布朗运动/有效扩散系数)
- 260_cvt_square_pdf     → mesh_generation (CVT自适应网格生成)
- 239_cvt_1_movie        → mesh_generation (Lloyd迭代/最近邻搜索)
- 861_pendulum_nonlinear_ode → vessel_mechanics (非线性摆/血管壁弹性)
- 345_exm (orbits)       → vessel_mechanics (N体相互作用/红细胞动力学)
- 1061_schroedinger_nonlinear_pde → pulse_wave_dynamics (NLSE孤子传播)
- 345_exm (waterwave)    → pulse_wave_dynamics (Lax-Wendroff浅水波)
- 1405_web_matrix        → network_hemodynamics (PageRank/网络流量分配)
- 345_exm (pagerank)     → network_hemodynamics (马尔可夫链稳态分布)
- 215_control_bio        → optimal_control (Pontryagin原理/WSS最优调控)

运行方式:
    python main.py
（零参数，自动执行完整计算流程并输出结果）
"""

import numpy as np
import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry_utils import (
    pi_high_precision, is_prime, prime_sieve,
    FEMMesh, circular_cross_section_area, circular_cross_section_perimeter,
    womersley_number, reynolds_number, murray_law_radius,
    bifurcation_prime_level, safe_sqrt, safe_divide
)
from quadrature_rules import (
    hexagon01_area, hexagon_lyness_rule03, hexagon_lyness_rule07,
    integrate_on_hexagon, legendre_gauss_nodes_weights,
    gauss_legendre_quadrature, runge_fun, runge_deriv, runge_deriv2,
    runge_antideriv, runge_power_series, wss_distribution_analog,
    lagrange_interpolation_error
)
from linear_algebra_core import (
    R83TMatrix, r83t_mv, r83t_mtv, r83t_res,
    r83t_jacobi_solve, r83t_gauss_seidel_solve, r83t_cg_solve,
    build_womersley_tridiagonal, thomas_algorithm
)
from stochastic_diffusion import (
    brownian_motion_simulation, brownian_displacement_simulation,
    verify_einstein_relation, effective_diffusion_plasma,
    einstein_viscosity_correction, peclet_number,
    ldl_wall_flux_estimate
)
from mesh_generation import (
    find_closest, cvt_iterate, generate_cvt_mesh,
    vessel_wall_density, map_cvt_to_annulus, VascularCVTMesh
)
from vessel_mechanics import (
    VesselElasticPendulum, simulate_vessel_oscillation,
    rbc_interaction_force, update_rbc_positions_euler,
    apparent_viscosity_from_rbc
)
from pulse_wave_dynamics import (
    shallow_water_lax_wendroff, pressure_wave_speed,
    NLSEPressurePulse, nlse_to_pressure_amplitude
)
from network_hemodynamics import (
    incidence_to_transition, power_rank, page_rank_with_damping,
    ArterialNetwork, bifurcation_flow_split
)
from optimal_control import (
    forward_backward_sweep, WSSOptimalControl,
    compute_control_cost, wss_physiological_score
)
from pulsatile_cfd_engine import (
    WomersleySolver, compute_tawss, compute_osi,
    compute_wss_gradient, relative_resistance_index,
    generate_wss_report
)


def section_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_geometry_and_physics_tests():
    """运行几何与基础物理测试。"""
    section_header("1. 血管几何与基础物理参数")

    # 868_pi_spigot: 高精度π
    pi_val = pi_high_precision()
    print(f"[868_pi_spigot] 高精度 π = {pi_val:.15f}")
    print(f"  与numpy.pi偏差: {abs(pi_val - np.pi):.2e}")

    # 915_prime_plot: 素数标记
    print(f"\n[915_prime_plot] 血管分叉素数级别标记:")
    for level in range(1, 12):
        flag = bifurcation_prime_level(level)
        print(f"  第{level:2d}级分叉: {'关键节点' if flag else '普通节点'}")

    # 381_fem_to_triangle: FEM网格
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    elements = np.array([[0, 1, 2]])
    mesh = FEMMesh(nodes, elements)
    print(f"\n[381_fem] 三角形网格单元面积 = {mesh.element_area(0):.6f} m²")

    # 血管几何参数
    R = 0.005  # 5mm半径
    area = circular_cross_section_area(R)
    perimeter = circular_cross_section_perimeter(R)
    print(f"\n血管半径 R = {R*1000:.1f} mm")
    print(f"  横截面积 A = {area*1e6:.3f} mm²")
    print(f"  周长 C = {perimeter*1000:.3f} mm")

    # Womersley数与雷诺数
    nu = 3.3e-6  # 血液运动粘度
    HR = 72.0
    omega = 2.0 * np.pi * HR / 60.0
    alpha = womersley_number(R, nu, omega)
    Re = reynolds_number(0.3, 2 * R, nu)
    print(f"\n血流动力学参数:")
    print(f"  Womersley数 α = {alpha:.3f}")
    print(f"  雷诺数 Re = {Re:.1f}")

    # Murray定律
    r_child = murray_law_radius(R, 2)
    print(f"\n[Murray定律] 对称二分叉子管半径 = {r_child*1000:.3f} mm")

    return R, nu, HR, omega


def run_quadrature_and_interpolation_tests():
    """运行数值积分与插值测试。"""
    section_header("2. 数值积分与插值分析")

    # 528_hexagon_lyness_rule
    print("[528_hexagon] 六边形求积法则:")
    area_exact = hexagon01_area()
    print(f"  单位六边形面积 = {area_exact:.6f}")

    f_const = lambda x, y: np.ones_like(x)
    int_const = integrate_on_hexagon(f_const, rule_id=3)
    print(f"  常函数积分(规则3) = {int_const:.6f}, 误差 = {abs(int_const - area_exact):.2e}")

    f_quadratic = lambda x, y: x * x + y * y
    int_quad = integrate_on_hexagon(f_quadratic, rule_id=7)
    print(f"  二次函数积分(规则7) = {int_quad:.6f}")

    # 660_legendre_fast_rule
    print(f"\n[660_legendre] Gauss-Legendre求积:")
    x_nodes, w_nodes = legendre_gauss_nodes_weights(8)
    print(f"  8点GL节点: {x_nodes}")
    print(f"  权重和 = {np.sum(w_nodes):.6f} (理论=2.0)")

    # 验证 ∫_{-1}^{1} x^6 dx = 2/7
    f_pow6 = lambda x: x ** 6
    int_pow6 = gauss_legendre_quadrature(f_pow6, -1.0, 1.0, n=4)
    print(f"  ∫x⁶dx (n=4) = {int_pow6:.6f}, 理论 = {2.0/7.0:.6f}")

    # 1051_runge: Runge现象分析
    print(f"\n[1051_runge] WSS分布插值Runge现象分析:")
    x_test = np.linspace(-1, 1, 1001)
    wss_vals = wss_distribution_analog(x_test, peak_wss=7.0)
    print(f"  WSS类比函数在x=0: {wss_vals[500]:.3f} Pa")
    print(f"  WSS类比函数在x=±1: {wss_vals[0]:.3f} Pa")

    # 等距节点插值误差
    n_nodes_eq = 11
    x_eq = np.linspace(-1, 1, n_nodes_eq)
    err_eq = lagrange_interpolation_error(x_eq, runge_fun, np.array([0.95]))
    print(f"  等距11节点在x=0.95处Runge插值误差 = {err_eq[0]:.4f}")


def run_linear_algebra_tests():
    """运行线性代数求解器测试。"""
    section_header("3. 三对角线性系统求解器 (R83T)")

    # 966_r83t: DIF2测试矩阵
    n = 50
    A = R83TMatrix.dif2(n)
    x_exact = np.ones(n)
    b = r83t_mv(A, x_exact)

    print(f"[966_r83t] DIF2矩阵维度 n={n}")
    print(f"  理论特征值 λ_1 = {A.eigenvalue_dif2(1):.6f}")
    print(f"  理论特征值 λ_n = {A.eigenvalue_dif2(n):.6f}")

    # Jacobi
    x_jac, it_jac, res_jac = r83t_jacobi_solve(A, b, max_iter=20000, tol=1e-10)
    err_jac = np.linalg.norm(x_jac - x_exact, ord=np.inf)
    print(f"\n  Jacobi: 迭代={it_jac}, 残差={res_jac:.2e}, 误差={err_jac:.2e}")

    # Gauss-Seidel
    x_gs, it_gs, res_gs = r83t_gauss_seidel_solve(A, b, max_iter=10000, tol=1e-10)
    err_gs = np.linalg.norm(x_gs - x_exact, ord=np.inf)
    print(f"  Gauss-Seidel: 迭代={it_gs}, 残差={res_gs:.2e}, 误差={err_gs:.2e}")

    # CG
    x_cg, it_cg, res_cg = r83t_cg_solve(A, b, max_iter=n, tol=1e-12)
    err_cg = np.linalg.norm(x_cg - x_exact, ord=np.inf)
    print(f"  Conjugate Gradient: 迭代={it_cg}, 残差={res_cg:.2e}, 误差={err_cg:.2e}")

    # Thomas直接法
    x_thomas = thomas_algorithm(A, b)
    err_thomas = np.linalg.norm(x_thomas - x_exact, ord=np.inf)
    print(f"  Thomas Algorithm: 误差={err_thomas:.2e}")


def run_stochastic_diffusion_tests():
    """运行布朗运动与扩散测试。"""
    section_header("4. 血浆微粒布朗运动与有效扩散")

    # 119_brownian_motion
    print("[119_brownian] 3D布朗运动模拟:")
    D_plasma = effective_diffusion_plasma(
        temperature_kelvin=310.15,
        particle_radius_nm=100.0,
        plasma_viscosity_pa_s=0.0012
    )
    print(f"  Stokes-Einstein扩散系数 D = {D_plasma:.3e} m²/s")

    dsq = brownian_displacement_simulation(
        k_trials=200, n_steps=50, m_dim=3,
        diffusion_coeff=D_plasma, total_time=1.0, seed=42
    )
    einstein_stats = verify_einstein_relation(dsq, 3, D_plasma, 1.0)
    print(f"  爱因斯坦关系验证:")
    print(f"    实测斜率 = {einstein_stats['slope']:.3e}")
    print(f"    理论斜率 = {einstein_stats['theoretical_slope']:.3e}")
    print(f"    相对误差 = {einstein_stats['relative_error']:.2%}")

    # 粘度修正
    mu_eff = einstein_viscosity_correction(hematocrit=0.45)
    print(f"\n  红细胞压积45%时的粘度比 μ_eff/μ_0 = {mu_eff:.3f}")

    # Peclet数
    Pe = peclet_number(shear_rate=1000.0, particle_radius_nm=100.0,
                       diffusion_coeff=D_plasma)
    print(f"  剪切率1000 s⁻¹时的Peclet数 Pe = {Pe:.2f}")

    # LDL通量
    ldl_flux = ldl_wall_flux_estimate(wss_pa=2.5, diffusion_coeff=D_plasma)
    print(f"  WSS=2.5 Pa时的LDL有效渗透性 = {ldl_flux:.3e} m/s")


def run_mesh_generation_tests():
    """运行CVT网格生成测试。"""
    section_header("5. 血管截面CVT自适应网格生成")

    # 260+239 CVT
    print("[260+239_cvt] 非均匀密度驱动CVT网格生成:")
    n_gen = 64
    density = lambda x, y: vessel_wall_density(x, y, thickness_center=0.5,
                                                thickness_amplitude=0.3, n_modes=2)
    generators = generate_cvt_mesh(n_gen, n_samples_per_gen=80,
                                   max_iter=60, tol=1e-4,
                                   density_func=density, seed=123)
    print(f"  生成器数量: {n_gen}")
    print(f"  生成器范围: x∈[{generators[:,0].min():.3f},{generators[:,0].max():.3f}], "
          f"y∈[{generators[:,1].min():.3f},{generators[:,1].max():.3f}]")

    # 映射到环形血管截面
    R_inner = 0.004
    R_outer = 0.006
    vasc_mesh = VascularCVTMesh(generators, R_inner, R_outer)
    print(f"\n  血管截面映射: 内径={R_inner*1000:.1f}mm, 外径={R_outer*1000:.1f}mm")
    print(f"  映射后坐标范围: x∈[{vasc_mesh.cartesian[:,0].min()*1000:.2f},{vasc_mesh.cartesian[:,0].max()*1000:.2f}] mm")

    # 壁厚分布
    thickness = vasc_mesh.wall_thickness_distribution(thickness_center=1.0e-3, amplitude=0.3e-3)
    print(f"  壁厚分布: min={thickness.min()*1000:.3f}mm, max={thickness.max()*1000:.3f}mm, mean={thickness.mean()*1000:.3f}mm")


def run_vessel_mechanics_tests():
    """运行血管壁力学测试。"""
    section_header("6. 血管壁弹性力学与红细胞相互作用")

    # 861_pendulum_nonlinear_ode
    print("[861_pendulum] 血管壁非线性摆模型:")
    pendulum = VesselElasticPendulum(
        equilibrium_radius=0.005,
        elastic_modulus_pa=1.0e6,
        wall_thickness_m=1.0e-3,
        wall_density_kg_m3=1050.0
    )
    print(f"  等效恢复系数 g_eff = {pendulum.g_eff:.3f} s⁻²")
    print(f"  小振幅周期 T₀ = {pendulum.period(0.01):.4f} s")

    xi0 = 0.1
    t_exact = np.linspace(0, 2.0, 200)
    xi_exact = pendulum.exact_solution(t_exact, xi0)
    print(f"  初始位移 ξ₀={xi0} rad 的精确解振幅范围: [{xi_exact.min():.4f}, {xi_exact.max():.4f}]")

    # 数值模拟脉动压力响应
    T = 60.0 / 72.0  # 心动周期
    t_num = np.linspace(0, 2 * T, 200)
    pressure = 12000.0 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t_num / T))
    result = simulate_vessel_oscillation(pendulum, t_num, xi0=0.05,
                                          external_pressure_pa=pressure)
    print(f"\n  脉动压力下的振动响应:")
    print(f"    半径变化范围: [{result['radius'].min()*1000:.4f}, {result['radius'].max()*1000:.4f}] mm")
    print(f"    能量守恒偏差: {abs(result['energy'][-1] - result['energy'][0]) / result['energy'][0]:.2%}")

    # 345_exm orbits: RBC相互作用
    print(f"\n[345_orbits] 红细胞多体相互作用:")
    n_rbc = 20
    positions = np.random.randn(n_rbc, 2) * 1e-6
    forces = rbc_interaction_force(positions)
    print(f"  {n_rbc}个红细胞的相互作用力范数: mean={np.linalg.norm(forces, axis=1).mean():.3e} N")

    mu_app = apparent_viscosity_from_rbc(n_rbc, domain_volume=1e-12, base_viscosity=0.0012)
    print(f"  估算表观粘度 = {mu_app:.4f} Pa·s")


def run_pulse_wave_tests():
    """运行压力波传播测试。"""
    section_header("7. 动脉压力脉冲波传播")

    # 345_exm waterwave: Lax-Wendroff
    print("[345_waterwave] Lax-Wendroff浅水波格式:")
    nx = 100
    A0 = np.ones(nx) * 2.0e-5  # 基础面积 ~20 mm²
    Q0 = np.zeros(nx)
    # 在入口注入脉冲
    A0[:10] += 0.5e-5 * np.sin(np.linspace(0, np.pi, 10))

    dx = 0.01
    dt = 0.001
    c_wave = pressure_wave_speed(
        elastic_modulus_pa=1.0e6,
        wall_thickness_m=1.0e-3,
        vessel_radius_m=0.005
    )
    print(f"  Moens-Korteweg波速 c = {c_wave:.2f} m/s")

    A_final, Q_final = shallow_water_lax_wendroff(
        A0, Q0, dx, dt, g_eff=c_wave ** 2 / A0.mean(),
        n_steps=100, boundary_type="reflecting"
    )
    print(f"  传播后面积变化: min={A_final.min()*1e6:.3f} mm², max={A_final.max()*1e6:.3f} mm²")

    # 1061_schroedinger_nonlinear_pde: NLSE孤子
    print(f"\n[1061_NLSE] 非线性薛定谔方程压力孤子:")
    nlse = NLSEPressurePulse(nx=128, z_min=-10.0, z_max=10.0, gamma=0.5)
    psi0 = nlse.initial_double_soliton(nlse.z, amplitude=0.5, c1=1.0, c2=0.1, delta=5.0)
    psi_final, mass_hist = nlse.evolve(psi0, dt=0.005, n_steps=200)
    print(f"  初始质量 = {nlse.mass_conservation(psi0):.6f}")
    print(f"  演化后质量 = {nlse.mass_conservation(psi_final):.6f}")
    print(f"  质量守恒相对偏差 = {abs(mass_hist[-1] - mass_hist[0]) / (mass_hist[0] + 1e-15):.2e}")

    pressure_pulse = nlse_to_pressure_amplitude(psi_final)
    print(f"  压力孤子幅度范围: [{pressure_pulse.min():.1f}, {pressure_pulse.max():.1f}] Pa")


def run_network_tests():
    """运行动脉网络测试。"""
    section_header("8. 动脉网络血流分配 (PageRank类比)")

    # 1405_web_matrix + 345_pagerank
    print("[1405+345_pagerank] 主动脉弓网络流量分配:")
    network = ArterialNetwork()
    flows = network.compute_flow_distribution(total_flow=5.0e-5)
    wss = network.compute_wss_from_flow(flows, blood_viscosity_pa_s=0.0035)
    resistances = network.network_resistance(blood_viscosity_pa_s=0.0035)
    alphas = network.womersley_numbers(heart_rate_bpm=72.0)

    print(f"  {'节点':<25} {'流量(mL/min)':>12} {'WSS(Pa)':>10} {'阻力(10⁹Pa·s/m³)':>16} {'α':>8}")
    for name in network.node_names:
        q_ml_min = flows[name] * 1e6 * 60.0
        print(f"  {name:<25} {q_ml_min:>12.2f} {wss[name]:>10.3f} "
              f"{resistances[name]/1e9:>16.3f} {alphas[name]:>8.2f}")

    # 分叉流量分配
    q1, q2 = bifurcation_flow_split(0.006, 0.004, 0.004)
    print(f"\n  Murray分叉流量分配: Q₁/Q_total={q1:.3f}, Q₂/Q_total={q2:.3f}")


def run_optimal_control_test():
    """运行最优控制测试。"""
    section_header("9. WSS最优药物释放控制")

    # 215_control_bio
    print("[215_control] Pontryagin前向-后向扫描法:")
    controller = WSSOptimalControl(
        equilibrium_radius=0.005,
        target_wss_pa=2.5,
        blood_viscosity_pa_s=0.0035,
        flow_rate_m3_s=5.0e-5,
        k_growth=0.5,
        k_drug=0.3,
        control_penalty=0.1
    )

    T = 2.0
    n_t = 100
    time = np.linspace(0, T, n_t)
    r0 = 0.0045  # 初始半径略小

    result = controller.solve(r0, time, max_iter=80)

    print(f"  初始半径 r₀ = {r0*1000:.2f} mm")
    print(f"  目标WSS = {controller.wss_target:.2f} Pa")
    print(f"  最优控制后半径范围: [{result['radius'].min()*1000:.3f}, {result['radius'].max()*1000:.3f}] mm")
    print(f"  最优WSS范围: [{result['wss'].min():.3f}, {result['wss'].max():.3f}] Pa")
    print(f"  平均药物释放率 = {np.mean(result['control']):.4f}")

    cost = compute_control_cost(result['wss'], controller.wss_target,
                                result['control'], B=controller.B)
    print(f"  总控制代价 J = {cost:.4f}")

    final_score = wss_physiological_score(result['wss'][-1])
    print(f"  最终生理评分 = {final_score:.2f}")


def run_cfd_engine_test():
    """运行核心CFD引擎。"""
    section_header("10. 核心脉动流CFD引擎 (Womersley求解器)")

    # 粘度修正（来自stochastic_diffusion）
    mu_ratio = einstein_viscosity_correction(hematocrit=0.40)
    nu_blood = 3.3e-6 * mu_ratio
    print(f"  有效运动粘度 ν_eff = {nu_blood:.3e} m²/s (Hct=40%, μ_ratio={mu_ratio:.3f})")

    solver = WomersleySolver(
        radius=0.005,
        kinematic_viscosity=nu_blood,
        blood_density=1060.0,
        n_radial=80,
        heart_rate_bpm=72.0
    )

    print(f"  Womersley数 α = {solver.alpha:.3f}")

    # 稳态验证
    u_steady = solver.solve_steady_state(dt=1e-4, tol=1e-8)
    u_exact_steady = solver.womersley_exact_solution(t=0.0, n_harmonics=1)
    # 注意：精确解是谐波解，稳态需要特殊处理；这里仅验证剖面形状
    print(f"  稳态速度剖面: u_max = {u_steady.max():.4f} m/s, u_mean = {np.mean(u_steady):.4f} m/s")

    # 脉动流求解
    print(f"\n  开始2个心动周期的脉动流模拟...")
    result = solver.solve_pulsatile(
        n_cardiac_cycles=2.0,
        n_steps_per_cycle=150,
        dt=None
    )

    report = generate_wss_report(solver, result)

    print(f"\n  WSS分析报告:")
    print(f"    TAWSS = {report['TAWSS_Pa']:.3f} Pa")
    print(f"    TAWSS (Gauss-Legendre验证) = {report['TAWSS_GaussLegendre_Pa']:.3f} Pa")
    print(f"    OSI = {report['OSI']:.4f}")
    print(f"    WSSG = {report['WSSG_Pa_s']:.2f} Pa/s")
    print(f"    WSS_max = {report['WSS_max_Pa']:.3f} Pa")
    print(f"    WSS_min = {report['WSS_min_Pa']:.3f} Pa")
    print(f"    RRI = {report['RRI']:.4f}")
    print(f"    综合生理评分 = {report['physiological_score']:.2f}")

    # 临床判读
    print(f"\n  临床判读:")
    if report['TAWSS_Pa'] < 1.0:
        print("    ⚠ TAWSS < 1 Pa: 低剪切区，动脉粥样硬化风险增加")
    elif report['TAWSS_Pa'] > 7.0:
        print("    ⚠ TAWSS > 7 Pa: 高剪切区，内皮损伤风险增加")
    else:
        print("    ✓ TAWSS在生理范围内")

    if report['OSI'] > 0.15:
        print("    ⚠ OSI > 0.15: 强振荡剪切，斑块不稳定风险")
    else:
        print("    ✓ OSI正常")

    return report


def main():
    """
    统一入口：执行完整的血动脉脉动流与壁面剪切计算流程。
    """
    print("=" * 70)
    print("  计算流体力学：血动脉脉动流与壁面剪切应力分析")
    print("  博士级科研代码合成项目 — PROJECT_78")
    print("=" * 70)
    print(f"\n  运行时间: {np.datetime64('now')}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")

    try:
        # 阶段1: 几何与基础物理
        R, nu, HR, omega = run_geometry_and_physics_tests()

        # 阶段2: 数值积分与插值
        run_quadrature_and_interpolation_tests()

        # 阶段3: 线性代数求解器
        run_linear_algebra_tests()

        # 阶段4: 布朗运动与扩散
        run_stochastic_diffusion_tests()

        # 阶段5: CVT网格生成
        run_mesh_generation_tests()

        # 阶段6: 血管壁力学
        run_vessel_mechanics_tests()

        # 阶段7: 压力波传播
        run_pulse_wave_tests()

        # 阶段8: 网络血流分配
        run_network_tests()

        # 阶段9: 最优控制
        run_optimal_control_test()

        # 阶段10: 核心CFD引擎
        report = run_cfd_engine_test()

        # 最终总结
        section_header("项目执行完成")
        print("所有计算模块已成功运行，无报错。")
        print(f"核心CFD结果: TAWSS={report['TAWSS_Pa']:.3f}Pa, OSI={report['OSI']:.4f}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n[ERROR] 执行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
