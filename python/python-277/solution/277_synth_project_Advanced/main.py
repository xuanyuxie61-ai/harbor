# -*- coding: utf-8 -*-
"""
main.py
=======
位错运动与塑性变形模拟 — 高阶有限差分与稳定性分析
统一入口文件 (零参数可运行)

项目名称: DislocationDynamics-FD-HighOrder
科学领域: 计算材料 — 位错运动与塑性变形模拟
数值方法: 高阶有限差分 + von Neumann稳定性分析 + Monte Carlo热激活

作者: 合成项目自动生成
日期: 2026-06-08

================================================================================
核心科学问题
================================================================================

本项目解决以下前沿计算材料科学问题:

1. Peierls-Nabarro位错核心结构:
   - 使用椭圆积分精确计算位错核心能量
   - CORDIC算法加速广义层错能面计算
   - 高阶有限差分离散化PN方程

2. 有限差分格式的稳定性分析:
   - von Neumann方法分析放大因子
   - 边界轨迹法参数化稳定性域
   - CFL条件自动推导

3. 热激活位错脱钉扎:
   - 球面Monte Carlo积分计算平均激活能
   - 列联表分析脱钉扎统计
   - Kramers跃迁率理论

4. 双稳态位错动力学:
   - 分岔分析识别临界应力
   - Kramers逃逸率计算
   - Gillespie随机模拟

5. 多晶塑性:
   - Voronoi晶粒结构生成
   - 位错塞积与Hall-Petch关系
   - LCSS时间序列模式分析

================================================================================
运行方式
================================================================================

python main.py

无需任何参数。程序将依次执行:
1. 材料参数初始化与物理常数验证
2. 晶体滑移系分析 (CRT索引)
3. Peierls-Nabarro位错核心求解
4. 高阶有限差分精度验证
5. von Neumann稳定性分析
6. 热激活Monte Carlo模拟
7. 双稳态分岔分析
8. 多晶Voronoi结构生成
9. 位错塞积与Hall-Petch分析
10. 综合验证报告
"""

import math
import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_constants import (DEFAULT_MATERIAL, AluminumParameters,
                                 CopperParameters, TungstenParameters,
                                 compute_elisington_tensor, PI)
from crystal_lattice import (SlipSystemDatabase, GrainBoundaryLocus,
                              slip_system_index, decode_slip_system)
from peierls_nabarro import (PeierlsNabarroModel, CORDICTrigo,
                              elliptic_k_series, elliptic_e_series)
from high_order_fd import (StructuredGrid1D, HighOrderFDOperators,
                            DelayIntegrator, BoundaryCondition)
from stability_analysis import (VonNeumannStability, FloydCycleDetector,
                                 StabilityLocusAnalysis)
from thermal_activation import (ThermalActivationModel, BallMonteCarloSampler,
                                 ContingencyTableAnalysis)
from dislocation_dynamics import (BistableDislocationDynamics,
                                   StochasticDislocationTransition)
from grain_structure import (VoronoiGrainStructure, DislocationPileup,
                              LCSSTimeSeriesAnalysis)
from validation import (ReproducibilityValidator, DislocationDensityKernel,
                         run_comprehensive_validation)


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def phase1_material_characterization():
    """
    第一阶段: 材料参数表征

    验证物理常数，计算基本材料属性
    """
    print_section("阶段 1: 材料参数表征")

    # 三种材料对比
    materials = {
        'Al (FCC)': AluminumParameters(),
        'Cu (FCC)': CopperParameters(),
        'W (BCC)': TungstenParameters(),
    }

    print(f"\n{'材料':<12} {'μ (GPa)':<10} {'ν':<8} {'b (nm)':<10} "
          f"{'σ_P (MPa)':<12} {'E_line (eV/nm)':<16}")
    print("-" * 70)

    for name, mat in materials.items():
        mu_GPa = mat.mu / 1e9
        b_nm = mat.b_magnitude * 1e9
        sigma_P = mat.peierls_stress() / 1e6  # MPa

        # 位错线能量 (eV/nm)
        eV = 1.602e-19  # J/eV
        E_line = mat.E_line_screw / eV * 1e-9  # eV/nm

        print(f"{name:<12} {mu_GPa:<10.1f} {mat.nu:<8.3f} {b_nm:<10.4f} "
              f"{sigma_P:<12.4f} {E_line:<16.3f}")

    # 弹性刚度张量
    al = DEFAULT_MATERIAL
    C = compute_elisington_tensor(al.mu, al.nu)
    print(f"\nAl弹性刚度矩阵 C (Voigt, GPa):")
    for i in range(3):
        row = [f"{C[i][j]/1e9:.1f}" for j in range(3)]
        print(f"  [{'  '.join(row)}]")

    return materials


def phase2_slip_system_analysis():
    """
    第二阶段: 滑移系分析 (CRT索引)

    融合: 170_chinese_remainder_theorem
    """
    print_section("阶段 2: 滑移系分析 (中国剩余定理索引)")

    db = SlipSystemDatabase(crystal_type='FCC')
    print(f"\nFCC {db.crystal_type} 滑移系总数: {db.n_systems}")
    print(f"\n{'索引':<6} {'滑移面':<20} {'Burgers矢':<20} {'|b| (nm)':<10}")
    print("-" * 60)

    for i in range(db.n_systems):
        n = db.plane_normals[i]
        b = db.burgers_dirs[i]
        b_mag = db.burgers_magnitudes[i] * 1e9
        print(f"{i:<6} ({n[0]:5.3f},{n[1]:5.3f},{n[2]:5.3f}) "
              f"({b[0]:5.3f},{b[1]:5.3f},{b[2]:5.3f})  {b_mag:.4f}")

    # Schmid因子分析
    print(f"\nSchmid因子 (沿[001]拉伸):")
    max_m, active = db.max_schmid_factor((0, 0, 1))
    print(f"  最大Schmid因子: {max_m:.4f} (滑移系 #{active})")

    # 应力分解
    sigma_applied = 50e6  # 50 MPa单轴
    stress_tensor = [[0.0]*3 for _ in range(3)]
    stress_tensor[0][0] = sigma_applied  # σ_xx

    print(f"\n施加单轴应力 σ_xx = {sigma_applied/1e6:.1f} MPa:")
    print(f"  {'系':<4} {'RSS (MPa)':<12}")
    for i in range(min(6, db.n_systems)):
        tau = db.resolved_shear_stress(stress_tensor, i)
        print(f"  {i:<4} {tau/1e6:<12.4f}")

    # 晶界轨迹
    gb = GrainBoundaryLocus(n_theta=5, n_axis=3)
    print(f"\nRead-Shockley晶界能量 (小角晶界):")
    for theta_deg in [1, 5, 10, 15]:
        theta_rad = theta_deg * PI / 180.0
        e = gb.read_shockley_energy(theta_rad)
        fb = gb.frank_bilby_dislocation_content(theta_rad, (0, 0, 1), (1, 0, 0))
        print(f"  θ={theta_deg:2d}°: γ={e:.4f} J/m², D={fb['dislocation_spacing']:.3e} m")

    return db


def phase3_peierls_nabarro():
    """
    第三阶段: Peierls-Nabarro位错核心

    融合: 335_elliptic_integral, 219_cordic
    """
    print_section("阶段 3: Peierls-Nabarro位错核心模型")

    pn = PeierlsNabarroModel(n_grid=64)

    # Peierls应力
    sigma_gsf, sigma_pn = pn.peierls_stress_analytical()
    print(f"\nPeierls应力对比:")
    print(f"  γ面导数法: {sigma_gsf:.4e} Pa ({sigma_gsf/1e6:.4f} MPa)")
    print(f"  PN指数法:  {sigma_pn:.4e} Pa ({sigma_pn/1e6:.6f} MPa)")

    # 椭圆积分
    print(f"\n椭圆积分 (核心能量):")
    for m in [0.5, 0.9, 0.99, 0.999]:
        K = elliptic_k_series(m)
        E = elliptic_e_series(m)
        print(f"  m={m:.3f}: K(m)={K:.6f}, E(m)={E:.6f}, K-E={K-E:.6f}")

    # CORDIC验证
    cordic = CORDICTrigo()
    print(f"\nCORDIC三角函数精度:")
    max_err = 0.0
    for deg in range(0, 361, 15):
        theta = deg * PI / 180.0
        c_cos, c_sin = cordic.cossin(theta)
        err = max(abs(c_cos - math.cos(theta)), abs(c_sin - math.sin(theta)))
        max_err = max(max_err, err)
    print(f"  最大误差: {max_err:.2e}")

    # 位错核心轮廓
    x, u, dudx = pn.dislocation_profile_analytical()
    print(f"\n位错核心轮廓:")
    print(f"  计算域: [{x[0]:.3e}, {x[-1]:.3e}] m")
    print(f"  网格点: {len(x)}")
    print(f"  u范围: [{min(u):.4e}, {max(u):.4e}] m")
    print(f"  du/dx最大: {max(dudx):.4e}")

    # Kink对能量
    kink = pn.kink_pair_energy(kink_separation=50e-10)
    print(f"\nKink对形核:")
    print(f"  单个kink能量: {kink['E_kink_single']:.4e} J/m")
    print(f"  kink对总能:   {kink['E_pair_total']:.4e} J/m")
    print(f"  临界应力:     {kink['critical_stress']:.4e} Pa")

    # 数值求解PN方程
    print(f"\nPN方程数值求解 (σ = 0.5σ_P):")
    sigma_test = 0.5 * sigma_pn
    result = pn.solve_pn_equation(sigma_applied=sigma_test, max_iter=100)
    print(f"  收敛: {result['converged']}, 迭代: {result['iterations']}")
    print(f"  最终残差: {result['final_residual']:.4e}")

    return pn


def phase4_finite_difference():
    """
    第四阶段: 高阶有限差分精度验证

    融合: 491_grid_display, 1107_Gelens-Lab (延迟积分)
    """
    print_section("阶段 4: 高阶有限差分算子")

    # 网格质量
    print(f"\n网格生成与质量:")
    for grid_type in ['uniform', 'clustered']:
        grid = StructuredGrid1D(-1e-8, 1e-8, 41, grid_type, cluster_center=0.0)
        q = grid.get_quality_metrics()
        print(f"  {grid_type:<10}: dx_min={q['min_dx']:.3e}, "
              f"uniformity={q['uniformity']:.4f}")

    # 精度测试: f(x) = sin(kx)
    print(f"\n有限差分精度 (f = sin(kx), k=1e8):")
    n_test = 201
    L = 1e-8
    dx = 2*L / (n_test - 1)
    x = [-L + i*dx for i in range(n_test)]
    k_wave = 1e8

    f = [math.sin(k_wave * xi) for xi in x]
    df_exact = [k_wave * math.cos(k_wave * xi) for xi in x]

    for order in [2, 4, 6, 8]:
        fd = HighOrderFDOperators(order=order)
        df_fd = fd.apply_deriv1(f, dx)
        r = fd.radius
        max_err = max(abs(df_fd[i] - df_exact[i])
                      for i in range(r + 1, n_test - r - 1))
        print(f"  {order}阶精度: max_error = {max_err:.4e}")

    # 2阶导数测试
    print(f"\n2阶导数精度 (f = sin(kx)):")
    d2f_exact = [-k_wave**2 * math.sin(k_wave * xi) for xi in x]
    for order in [2, 4, 6]:
        fd = HighOrderFDOperators(order=order)
        d2f_fd = fd.apply_deriv2(f, dx)
        r = 2  # 2阶导数模板半径
        max_err = max(abs(d2f_fd[i] - d2f_exact[i])
                      for i in range(r + 1, n_test - r - 1))
        print(f"  {order}阶精度: max_error = {max_err:.4e}")

    # 修正波数 (数值色散)
    print(f"\n数值色散分析 (kh = π/4):")
    fd4 = HighOrderFDOperators(order=4)
    info = fd4.modified_wavenumber(PI / (4 * dx), dx)
    print(f"  4阶: k'h/(kh) = {1.0 + info['dispersion_error']:.8f}")
    print(f"  每波长节点数: {info['resolution_elements_per_wavelength']:.1f}")

    # 延迟积分器
    print(f"\n延迟微分方程测试 (du/dt = -u + 0.3u(t-τ)):")
    delay = 5e-14  # 50 fs 延迟
    dde = DelayIntegrator(delay_time=delay, n_history=100)
    u_val = 1.0
    t_val = 0.0
    dt = 1e-15

    dde.update_history(0.0, 1.0)

    for step in range(50):
        u_val = dde.rk4_step(
            lambda u, u_del, t: -u + 0.3 * u_del,
            u_val, t_val, dt, 1
        )
        t_val += dt
        dde.update_history(t_val, u_val)

    print(f"  50步后: u = {u_val:.6f} (t = {t_val:.3e} s)")

    # 边界条件
    print(f"\n边界条件处理:")
    bc = BoundaryCondition(bc_type='mirror')
    u_test = [0.0, 1.0, 2.0, 3.0, 4.0]
    u_bc = bc.apply(u_test, dx=0.1, mu=26e9, nu=0.345, b=2.86e-10, L=1e-8)
    print(f"  镜像边界: {u_test} → {u_bc}")

    return True


def phase5_stability():
    """
    第五阶段: 稳定性分析

    融合: 267_cycle_floyd, 104_boundary_locus
    """
    print_section("阶段 5: von Neumann稳定性分析")

    al = DEFAULT_MATERIAL
    vn = VonNeumannStability()

    mu_eff = al.mu * al.b_magnitude**2 / (4.0 * PI * (1.0 - al.nu))
    alpha = mu_eff * al.zeta_screw**2 / 10.0
    # Peierls势垒曲率 κ = d²γ/du²|_{u=0} = 2π² γ_usf / b²
    kappa = 2.0 * PI**2 * al.gamma_usf / al.b_magnitude**2

    print(f"\n物理参数:")
    print(f"  μ_eff = {mu_eff:.4e} N")
    print(f"  α = {alpha:.4e} N·m²")
    print(f"  κ = {kappa:.4e} N/m²")

    # CFL条件
    print(f"\nCFL条件:")
    for dx_factor in [1, 5, 10, 20]:
        dx = al.a_lattice * dx_factor
        cfl = vn.cfl_condition(dx, mu_eff, alpha)
        print(f"  dx={dx_factor}a₀: Δt_max={cfl['dt_max']:.4e} s, "
              f"CFL={cfl['cfl_diffusion']:.4f}")

    # Floyd周期检测
    print(f"\nFloyd周期检测 (位错振荡):")
    detector = FloydCycleDetector(tolerance=1e-6, max_iterations=5000)

    # 构造一个具有已知周期的映射
    omega_dt = 0.5
    A = 0.3
    def dislocation_map(theta):
        return (theta + omega_dt - A * math.sin(theta)) % (2 * PI)

    result = detector.detect_cycle(dislocation_map, 1.0)
    print(f"  检测到周期: {result['has_cycle']}")
    if result['has_cycle']:
        print(f"  周期长度: {result['period']}")
        print(f"  收敛迭代: {result['convergence_iterations']}")

    # 非周期映射
    def non_periodic(x):
        return x * 1.1 + 0.01
    result2 = detector.detect_cycle(non_periodic, 1.0)
    print(f"  非周期序列: has_cycle={result2['has_cycle']}")

    # 稳定性边界轨迹
    print(f"\n稳定性边界轨迹:")
    locus = StabilityLocusAnalysis()
    k_values = [PI * i / (10 * al.a_lattice * 10) for i in range(1, 30)]
    dx_test = al.a_lattice * 10
    pts = locus.compute_locus_points(k_values, dx_test, mu_eff, alpha, kappa)
    print(f"  边界点数: {len(pts)}")
    if pts:
        dt_min = min(p['dt_critical'] for p in pts)
        dt_max = max(p['dt_critical'] for p in pts)
        print(f"  Δt临界范围: [{dt_min:.4e}, {dt_max:.4e}] s")

    return True


def phase6_thermal_activation():
    """
    第六阶段: 热激活Monte Carlo模拟

    融合: 069_ball_monte_carlo, 042_asa144
    """
    print_section("阶段 6: 热激活位错脱钉扎")

    model = ThermalActivationModel(temperature=300.0, seed=42)
    sigma_p = model.sigma_p

    print(f"\nPeierls应力: {sigma_p:.4e} Pa ({sigma_p/1e6:.4f} MPa)")

    # Gibbs能垒与激活率
    print(f"\n热激活参数 (T=300K):")
    print(f"  {'τ/τ_P':<8} {'ΔG (eV)':<12} {'V* (b³)':<12} {'Γ (1/s)':<12} {'T_c (K)':<10}")
    print("-" * 60)

    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tau = frac * sigma_p
        dG = model.gibbs_barrier(tau)
        V_star = model.activation_volume(tau)
        rate = model.activation_rate(tau)
        T_c = model.critical_temperature(tau)

        dG_eV = dG / 1.602e-19
        V_b3 = V_star / model.mat.b_magnitude**3

        print(f"  {frac:<8.1f} {dG_eV:<12.4f} {V_b3:<12.2f} "
              f"{rate:<12.4e} {T_c:<10.1f}")

    # Monte Carlo积分
    print(f"\n球面Monte Carlo积分 (平均激活能):")
    mc_result = model.monte_carlo_activation_energy(n_samples=3000)
    print(f"  采样数: {mc_result['n_samples']}")
    print(f"  平均激活能: {mc_result['mean']:.4e} J ({mc_result['mean']/1.602e-19:.4f} eV)")
    print(f"  相对误差: {mc_result['relative_error']:.4f}")

    # 列联表分析
    print(f"\n脱钉扎事件统计分析:")
    stats = model.depinning_statistics(n_obstacles=30, seed=42)
    print(f"  列联表 (障碍×模式):")
    for i, row in enumerate(stats['table']):
        print(f"    类型{i}: {row}")
    print(f"  χ² = {stats['chi_squared']:.4f}, df = {stats['degrees_of_freedom']}")
    print(f"  p值 = {stats['p_value']:.4f}")

    # 速度-应力曲线
    print(f"\n位错速度-应力关系:")
    v_curve = model.velocity_stress_curve(n_points=8)
    print(f"  {'τ (MPa)':<12} {'v (m/s)':<14} {'主导机制':<12}")
    print("-" * 40)
    for pt in v_curve[::2]:  # 每隔一个显示
        regime = 'thermal' if pt['v_thermal'] > pt['v_drag'] else 'drag'
        print(f"  {pt['tau']/1e6:<12.2f} {pt['velocity']:<14.4e} {regime:<12}")

    return model


def phase7_bistability():
    """
    第七阶段: 双稳态位错动力学

    融合: 1107_Gelens-Lab_cellcyclemodules, 1215_riskychoice
    """
    print_section("阶段 7: 双稳态位错动力学")

    dynamics = BistableDislocationDynamics(kappa_back=1e-3)
    sigma_p = dynamics.mat.peierls_stress()

    # 分岔分析
    print(f"\n分岔分析 (扫描施加应力):")
    tau_min = 0.1 * sigma_p
    tau_max = 2.0 * sigma_p
    bif = dynamics.bifurcation_analysis((tau_min, tau_max), n_tau=20)

    print(f"  分岔点数量: {len(bif['bifurcation_points'])}")
    for bp in bif['bifurcation_points'][:5]:
        print(f"    τ = {bp['tau']/1e6:.4f} MPa: "
              f"{bp['n_stable_before']}→{bp['n_stable_after']} 稳定态")

    # 固定点分析
    print(f"\n固定点分析 (τ = 0.5σ_P):")
    tau_test = 0.5 * sigma_p
    fps = dynamics.find_fixed_points(tau_test)
    for fp in fps:
        print(f"  u = {fp['u']:.4e} m: {fp['type']}, "
              f"dF/du = {fp['F_prime']:.4e}")

    # 能垒
    barrier = dynamics.energy_barrier(tau_test)
    print(f"\n能垒信息:")
    print(f"  能垒高度: {barrier['barrier_height']:.4e} J/m")
    print(f"  稳定态数: {barrier['n_stable']}")

    # 势能面
    print(f"\n势能面 (τ = 0.3σ_P):")
    tau_v = 0.3 * sigma_p
    b = dynamics.mat.b_magnitude
    u_values = [-b + i * 2*b/20 for i in range(21)]
    V_values = [dynamics.potential_energy(u, tau_v) for u in u_values]
    V_min = min(V_values)
    V_max = max(V_values)
    print(f"  V范围: [{V_min:.4e}, {V_max:.4e}] J/m")
    print(f"  势阱数: {V_values.count(V_min) if V_min == max(V_values) else 'multiple'}")

    # Kramers跃迁率
    print(f"\nKramers跃迁率:")
    stochastic = StochasticDislocationTransition(dynamics, temperature=300.0, seed=42)

    if len(fps) >= 2:
        stable_fps = [fp for fp in fps if fp['stable']]
        unstable_fps = [fp for fp in fps if not fp['stable']]
        if stable_fps and unstable_fps:
            k_rate = stochastic.kramers_rate(
                stable_fps[0]['u'], unstable_fps[0]['u'], tau_test
            )
            print(f"  跃迁率: {k_rate:.4e} /s")

    # Gillespie轨迹 (短模拟)
    print(f"\n随机跃迁模拟 (短轨迹):")
    traj = stochastic.simulate_trajectory(
        tau=tau_test, t_total=1e-9, dt=1e-11
    )
    print(f"  势阱数: {traj['n_wells']}")
    print(f"  跃迁次数: {traj['n_transitions']}")
    print(f"  轨迹长度: {len(traj['time'])}")

    return dynamics


def phase8_polycrystal():
    """
    第八阶段: 多晶塑性分析

    融合: 1398_voronoi_plot, 1229_maidens_LCSS
    """
    print_section("阶段 8: 多晶晶粒与位错塞积")

    # Voronoi晶粒
    print(f"\nVoronoi多晶晶粒生成:")
    voronoi = VoronoiGrainStructure(
        domain_size=(2e-6, 2e-6), n_grains=16, seed=42
    )
    voronoi.build_grain_map(resolution=30)

    d_avg = voronoi.get_average_grain_size()
    dist = voronoi.get_grain_size_distribution()
    print(f"  晶粒数: {voronoi.n_grains}")
    print(f"  平均尺寸: {d_avg*1e9:.1f} nm")
    print(f"  尺寸标准差: {dist['std']*1e9:.1f} nm")
    print(f"  晶界总数: {len(voronoi.grain_boundaries)}")

    # 邻居分析
    print(f"\n晶粒邻居统计:")
    n_neighbors = [len(voronoi.get_neighbors(i)) for i in range(voronoi.n_grains)]
    avg_neighbors = sum(n_neighbors) / len(n_neighbors)
    print(f"  平均邻居数: {avg_neighbors:.1f}")
    print(f"  邻居数范围: [{min(n_neighbors)}, {max(n_neighbors)}]")

    # 位错塞积
    print(f"\n位错塞积与Hall-Petch:")
    pileup = DislocationPileup()

    print(f"\n  {'d (nm)':<10} {'n_disl':<10} {'σ_y (MPa)':<12} {'σ_tip/σ (n)':<14}")
    print("-" * 50)

    sigma_app = 100e6  # 100 MPa
    for d_nm in [100, 500, 1000, 5000]:
        d_m = d_nm * 1e-9
        n = pileup.number_of_dislocations(d_m, sigma_app)
        sigma_y = pileup.hall_petch_strength(d_m)
        sigma_tip_ratio = n

        print(f"  {d_nm:<10} {n:<10} {sigma_y/1e6:<12.2f} {sigma_tip_ratio:<14}")

    # 反Hall-Petch
    print(f"\n反Hall-Petch效应 (纳米晶):")
    for d_nm in [5, 10, 15, 20, 30, 50]:
        d_m = d_nm * 1e-9
        sigma_y = pileup.inverse_hall_petch(d_m)
        print(f"  d = {d_nm:2d} nm: σ_y = {sigma_y/1e6:.2f} MPa")

    # LCSS分析
    print(f"\nLCSS位错速度模式分析:")
    lcss = LCSSTimeSeriesAnalysis(epsilon=0.2, delta=3)

    # 模拟不同晶粒中的位错速度信号
    signals = []
    for grain_idx in range(5):
        # 不同晶粒的位错速度有不同特征
        freq = 1.0 + 0.2 * grain_idx
        phase = 0.3 * grain_idx
        signal = [math.sin(2*PI*freq*i/50 + phase) + 0.1*grain_idx
                  for i in range(50)]
        signals.append(signal)

    analysis = lcss.analyze_velocity_patterns(signals)
    print(f"  信号数: {analysis['n_signals']}")
    print(f"  最相似: 信号{analysis['most_similar_pair']}, "
          f"距离={analysis['most_similar_distance']:.4f}")
    print(f"  最不相似: 信号{analysis['most_dissimilar_pair']}, "
          f"距离={analysis['most_dissimilar_distance']:.4f}")

    return voronoi


def phase9_validation():
    """
    第九阶段: 综合验证
    """
    print_section("阶段 9: 综合验证与可复现性")

    # 核方法密度重建
    print(f"\n位错密度核回归:")
    kernel = DislocationDensityKernel(kernel_type='gaussian', bandwidth=5e-10)

    # 模拟位错位置
    disl_pos = [i * 2e-9 for i in range(10)]
    grid_pts = [i * 1e-9 for i in range(30)]

    density = kernel.density_field_reconstruction(disl_pos, grid_pts)
    max_density = max(density)
    print(f"  位错数: {len(disl_pos)}")
    print(f"  网格点: {len(grid_pts)}")
    print(f"  最大密度: {max_density:.4f}")
    print(f"  密度总和: {sum(density):.4f}")

    # 不同核函数对比
    for ktype in ['gaussian', 'wendland', 'matern']:
        k = DislocationDensityKernel(kernel_type=ktype, bandwidth=5e-10)
        d = k.density_field_reconstruction(disl_pos, grid_pts[:5])
        print(f"  {ktype:<10}: ρ[0:5] = {[f'{v:.4f}' for v in d]}")

    # Richardson外推
    print(f"\nRichardson外推:")
    validator = ReproducibilityValidator()

    # 模拟不同网格精度的结果
    values = [1.001, 1.0001, 1.00001]
    rich = validator.richardson_extrapolation(values, [10.0, 10.0])
    print(f"  观测收敛阶: {rich['observed_order']:.2f}")
    print(f"  外推值: {rich['extrapolated_value']:.8f}")
    print(f"  误差估计: {rich['error_estimate']:.4e}")

    # 能量守恒
    print(f"\n能量守恒测试:")
    E_history = [1.0 + 1e-6 * math.sin(0.1*i) for i in range(100)]
    ec = validator.energy_conservation_check(E_history, tolerance=1e-4)
    print(f"  守恒: {ec['conserved']}")
    print(f"  最大偏差: {ec['max_deviation']:.4e}")

    # 完整验证报告
    print(f"\n运行完整验证套件...")
    report = run_comprehensive_validation()
    print(f"\n验证总结: {report['passed']}/{report['total_tests']} 通过 "
          f"({report['pass_rate']*100:.0f}%)")

    return report


def main():
    """
    主程序入口

    依次执行9个分析阶段，覆盖位错运动与塑性变形模拟的
    核心计算方法。
    """
    print("*" * 78)
    print("*" + " " * 76 + "*")
    print("*  位错运动与塑性变形模拟 — 高阶有限差分与稳定性分析".center(76) + "*")
    print("*  DislocationDynamics-FD-HighOrder".center(76) + "*")
    print("*" + " " * 76 + "*")
    print("*" * 78)
    print(f"\n科学领域: 计算材料 — 位错运动与塑性变形模拟")
    print(f"数值方法: 高阶有限差分 + von Neumann稳定性 + Monte Carlo")
    print(f"材料系统: Al (FCC), Cu (FCC), W (BCC)")

    results = {}

    try:
        results['materials'] = phase1_material_characterization()
        results['slip_systems'] = phase2_slip_system_analysis()
        results['peierls_nabarro'] = phase3_peierls_nabarro()
        results['finite_diff'] = phase4_finite_difference()
        results['stability'] = phase5_stability()
        results['thermal'] = phase6_thermal_activation()
        results['bistability'] = phase7_bistability()
        results['polycrystal'] = phase8_polycrystal()
        results['validation'] = phase9_validation()

        # 最终总结
        print_section("计算完成")
        print(f"\n所有9个分析阶段已成功执行。")
        print(f"计算涵盖:")
        print(f"  • 3种材料 (Al/Cu/W) 的物理参数表征")
        print(f"  • 12个FCC滑移系的CRT索引与Schmid分析")
        print(f"  • Peierls-Nabarro位错核心 (椭圆积分 + CORDIC)")
        print(f"  • 2-8阶有限差分精度验证")
        print(f"  • von Neumann稳定性分析与CFL条件")
        print(f"  • Monte Carlo热激活模拟 (3000采样)")
        print(f"  • 双稳态分岔与Kramers跃迁")
        print(f"  • 16晶粒Voronoi多晶结构")
        print(f"  • Hall-Petch/反Hall-Petch关系")
        print(f"  • 综合验证报告")

        print(f"\n{'='*78}")
        print(f"  计算完成 ✓")
        print(f"{'='*78}\n")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
