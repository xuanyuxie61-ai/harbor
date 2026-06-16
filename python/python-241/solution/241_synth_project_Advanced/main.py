"""
main.py
===================================================================
核反应光学模型截面预测与高阶有限差分稳定性分析
— 统一入口 (零参数可运行)

科学问题:
    求解核子-重核散射的径向薛定谔方程, 使用复数光学模型势 (Woods-Saxon
    + 表面吸收 + 自旋-轨道 + 库仑), 通过 Numerov 四阶有限差分计算散射
    相移和反应截面, 并系统分析高阶差分格式的数值稳定性与收敛性.

物理系统:
    中子 + 208Pb @ E_lab = 14.1 MeV (DD 聚变中子能量)

计算方法:
    1. Numerov 四阶方法 — 主算法
    2. 二阶中心差分 — 基准对比
    3. FTCS 时间推进 — 不稳定性演示
    4. 离散层传递矩阵 — 波函数传播

输出内容:
    1. 光学势参数与分波相移
    2. 弹性/反应/总截面
    3. 稳定性分析报告
    4. 收敛性研究结果
    5. 模型比较检验
===================================================================
"""

import numpy as np
import sys
import os

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optical_potential import OpticalPotential, OpticalPotentialParams
from radial_schrodinger import RadialSchrodingerSolver
from stability_analysis import StabilityAnalyzer
from phase_shift import PhaseShiftCalculator
from cross_section import CrossSectionCalculator
from channel_coupling import (ReactionChannel, CouplingGraph,
                               AngularMomentumCoupler)
from wave_propagation import DiscreteLayerPropagator
from model_selection import SequentialModelTester, OpticalPotentialModelFactory
from convergence_analysis import ConvergenceStudy
from norms_utils import (l2_norm_radial, rms_norm_radial,
                          probability_current_density, continuity_residual,
                          wavefunction_norm_check)


# ---------- 全局物理参数 ----------
HBAR_C = 197.3269804


def print_header(title: str):
    """打印章节标题"""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str):
    """打印子标题"""
    print(f"\n--- {title} ---")


# ===================================================================
# 阶段 1: 光学势构造
# ===================================================================
def stage1_optical_potential():
    """构造中子+208Pb光学势"""
    print_header("阶段 1: 光学模型势构造")

    # 中子 + 208Pb @ 14.1 MeV
    params = OpticalPotentialParams(
        A_target=208, Z_target=82,
        A_projectile=1, Z_projectile=0,
        energy=14.1,
        V_0=48.0, r_v=1.25, a_v=0.65,
        W_0=12.0, r_w=1.25, a_w=0.55,
        W_d=6.0,
        V_so=6.5, r_so=1.20, a_so=0.60,
        fractal_dim=2.15, fractal_amp=0.06, fractal_levels=3,
        logistic_k=1.0, logistic_r=0.5,
    )

    pot = OpticalPotential(params)

    # 生成径向网格
    r = np.linspace(0.1, 30.0, 2000)
    V_total = pot.total_potential(r, l_quantum=0)

    print(f"\n  靶核: {params.A_target}Pb{params.Z_target}")
    print(f"  入射粒子: 中子")
    print(f"  实验室能量: {params.energy} MeV")
    print(f"  核半径 R_v: {pot.R_v:.3f} fm")
    print(f"  库仑半径 R_C: {pot.R_C:.3f} fm")
    print(f"  波数 k: {params.k_wavevector:.4f} fm^-1")
    print(f"  约化质量: {params.mass_reduced:.2f} MeV/c^2")
    print(f"\n  势深度参数:")
    for k, v in params.depths.items():
        print(f"    {k}: {v:.2f} MeV")
    print(f"\n  几何参数:")
    for k, v in params.geometry.items():
        print(f"    {k}: {v:.2f} fm")
    print(f"\n  分形表面参数:")
    print(f"    分形维数: {params.fractal['dim']}")
    print(f"    微扰振幅: {params.fractal['amplitude']}")
    print(f"    迭代层数: {params.fractal['levels']}")

    # 势的范数
    V_real_norm = l2_norm_radial(V_total.real, r)
    print(f"\n  实部势 L2 范数: {V_real_norm:.4f} MeV·fm^(1/2)")

    # 分形微扰核表面
    theta = np.linspace(0, 2 * np.pi, 100)
    from optical_potential import fractal_surface_perturbation
    R_fractal = fractal_surface_perturbation(
        theta, pot.R_v, amplitude=0.06, n_levels=3, seed=42)
    print(f"  分形表面 R 范围: [{np.min(R_fractal):.3f}, {np.max(R_fractal):.3f}] fm")
    print(f"  分形表面 R 均值: {np.mean(R_fractal):.3f} fm")

    return params, pot


# ===================================================================
# 阶段 2: 径向薛定谔方程求解
# ===================================================================
def stage2_radial_equation(pot: OpticalPotential, params: OpticalPotentialParams):
    """求解径向薛定谔方程"""
    print_header("阶段 2: 径向薛定谔方程数值求解")

    r_max = 30.0
    n_points = 2000
    solver = RadialSchrodingerSolver(pot, r_max=r_max, n_points=n_points)

    l_max = 8  # 计算 l=0 到 l=8 的分波
    phase_shifts = {}
    s_matrix_elements = []

    ps_calc = PhaseShiftCalculator(
        k=params.k_wavevector, r_match=r_max * 0.85)

    print(f"\n  径向网格: [0.01, {r_max}] fm, {n_points} 点, dr={solver.dr:.4f} fm")
    print(f"\n  {'l':>4} {'delta_real':>12} {'delta_imag':>12} {'|S_l|':>10} {'eta_l':>10}")
    print(f"  {'-'*50}")

    for l in range(l_max + 1):
        V = pot.total_potential(solver.r, l_quantum=l)
        _, u = solver.solve_numerov(l, V)

        delta = ps_calc.extract_phase_shift(u, solver.r, l)
        S_l = ps_calc.compute_s_matrix(delta)
        eta_l = ps_calc.compute_eta(delta)

        phase_shifts[l] = delta
        s_matrix_elements.append(S_l)

        print(f"  {l:4d} {np.real(delta):12.6f} {np.imag(delta):12.6f} "
              f"{abs(S_l):10.6f} {eta_l:10.6f}")

    s_matrix = np.array(s_matrix_elements)

    # 波函数范数检查 (对 l=0)
    V0 = pot.total_potential(solver.r, l_quantum=0)
    _, u0 = solver.solve_numerov(0, V0)
    norm_info = wavefunction_norm_check(u0.real, solver.r)
    print(f"\n  l=0 波函数诊断:")
    print(f"    L2 范数: {norm_info['l2_norm']:.4f}")
    print(f"    RMS 范数: {norm_info['rms_norm']:.6f}")
    print(f"    均方根半径: {norm_info['rms_radius']:.3f} fm")

    return phase_shifts, s_matrix, solver


# ===================================================================
# 阶段 3: 截面计算
# ===================================================================
def stage3_cross_sections(s_matrix: np.ndarray, params: OpticalPotentialParams):
    """计算各反应截面"""
    print_header("阶段 3: 反应截面计算")

    k = params.k_wavevector
    l_max = len(s_matrix) - 1

    calc = CrossSectionCalculator(k, l_max)
    results = calc.compute_total_cross_sections(s_matrix)

    print(f"\n  入射波数 k = {k:.4f} fm^-1")
    print(f"  分波数 l_max = {l_max}")
    print(f"\n  === 总截面结果 ===")
    print(f"  弹性截面 sigma_el:    {results['sigma_elastic']:10.4f} fm^2 = "
          f"{results['sigma_elastic']/100.0:.4f} barn")
    print(f"  反应截面 sigma_re:    {results['sigma_reaction']:10.4f} fm^2 = "
          f"{results['sigma_reaction']/100.0:.4f} barn")
    print(f"  总截面 sigma_tot:     {results['sigma_total']:10.4f} fm^2 = "
          f"{results['sigma_total']/100.0:.4f} barn")
    print(f"\n  === 光学定理验证 ===")
    print(f"  光学定理 sigma_tot:   {results['sigma_total_optical_theorem']:10.4f} fm^2")
    print(f"  相对偏差:             {results['optical_theorem_check']:.2e}")
    print(f"  分波收敛:             {'是' if results['converged'] else '否'}")

    # 微分截面
    theta = np.linspace(0.01, np.pi, 50)
    dsigma = calc.differential_cross_section(s_matrix, theta)
    print(f"\n  === 微分截面 (部分角度) ===")
    for i in range(0, len(theta), 10):
        print(f"    theta = {np.degrees(theta[i]):6.1f} deg: "
              f"d(sigma)/dOmega = {dsigma[i]:.4f} fm^2/sr")

    # 分波截面分布
    partial = calc.compute_partial_cross_sections(s_matrix)
    print(f"\n  === 分波弹性截面分布 ===")
    for l in range(min(l_max + 1, 6)):
        print(f"    l={l}: sigma_el = {partial['sigma_el_l'][l]:.4f} fm^2")

    # 强度函数
    strength = calc.strength_function(s_matrix)
    print(f"\n  === 强度函数 S_l (前5个分波) ===")
    for l in range(min(l_max + 1, 5)):
        print(f"    l={l}: S_l = {strength['strength_function'][l]:.6f}")

    return results


# ===================================================================
# 阶段 4: 稳定性分析
# ===================================================================
def stage4_stability(pot: OpticalPotential, params: OpticalPotentialParams):
    """有限差分稳定性分析"""
    print_header("阶段 4: 高阶有限差分稳定性分析")

    analyzer = StabilityAnalyzer(
        n_spatial=200, dr=0.15, mass_reduced=params.mass_reduced)

    r = np.linspace(0.1, 30.0, 200)
    V = pot.total_potential(r, l_quantum=0)

    # 4.1 FTCS 稳定性
    print_subheader("4.1 FTCS 格式 von Neumann 分析")
    dt_ftcs = 0.005
    ftcs_result = analyzer.von_neumann_ftcs(V, dt_ftcs, l_quantum=0)
    print(f"  时间步长 dt = {dt_ftcs} fm/c")
    print(f"  格式: {ftcs_result['scheme']}")
    print(f"  稳定性: {'稳定' if ftcs_result['stable'] else '不稳定 (FTCS 对薛定谔方程无条件不稳定)'}")
    print(f"  最大增长因子 |G|: {ftcs_result['G_max']:.6f}")
    print(f"  增长率 (1/|G|-1)/dt: {ftcs_result['max_growth_rate']:.4e}")
    print(f"  扩散 CFL 数: {ftcs_result['cfl_diffusion']:.4f}")
    print(f"  e-fold 时间: {ftcs_result['instability_e_folding_time']:.4f} fm/c")

    # 4.2 Crank-Nicolson 稳定性
    print_subheader("4.2 Crank-Nicolson 格式 von Neumann 分析")
    cn_result = analyzer.von_neumann_crank_nicolson(V, dt_ftcs, l_quantum=0)
    print(f"  格式: {cn_result['scheme']}")
    print(f"  稳定性: {'稳定' if cn_result['stable'] else '不稳定'}")
    print(f"  酉性质: {'是' if cn_result['is_unitary'] else '否'}")
    print(f"  |G| 最大偏差: {cn_result['max_G_deviation']:.2e}")

    # 4.3 矩阵谱半径
    print_subheader("4.3 传播矩阵谱半径分析")
    for scheme in ['forward_euler', 'backward_euler', 'crank_nicolson']:
        spec = analyzer.matrix_spectral_radius(V, dt_ftcs, scheme=scheme)
        print(f"  {scheme:20s}: rho(A)={spec['spectral_radius']:.6f}, "
              f"stable={'Y' if spec['stable'] else 'N'}, "
              f"cond={spec['condition_number']:.2e}")

    # 4.4 Numerov 稳定性
    print_subheader("4.4 Numerov 方法稳定性检查")
    solver = RadialSchrodingerSolver(pot, r_max=30.0, n_points=2000)
    V_dense = pot.total_potential(solver.r, l_quantum=0)
    K2 = solver._local_wavevector_squared(V_dense.real, l=0)
    num_stab = analyzer.numerov_stability_check(K2, solver.dr)
    print(f"  稳定性参数 h^2*K^2_max/12: {num_stab['stability_parameter']:.6f}")
    print(f"  稳定: {'是' if num_stab['stable'] else '否'}")
    print(f"  临界步长: {num_stab['critical_step_size']:.4f} fm")
    print(f"  实际步长: {num_stab['actual_step_size']:.4f} fm")
    print(f"  安全因子: {num_stab['safety_factor']:.2f}")
    print(f"  振荡区点数: {num_stab['n_oscillatory_points']}")
    print(f"  指数区点数: {num_stab['n_exponential_points']}")

    # 4.5 不稳定 ODE 诊断
    print_subheader("4.5 不稳定 ODE 系统诊断")
    ode_diag = analyzer.unstable_ode_diagnostic(mu=5.0)
    print(f"  测试系统 y' = [[mu,1/mu],[-1/mu,mu]]*y, mu={ode_diag['mu']}")
    print(f"  特征值: {ode_diag['eigenvalues']}")
    print(f"  物理不稳定: {'是' if ode_diag['physical_instability'] else '否'}")
    print(f"  精确增长因子: {ode_diag['growth_exact']:.2e}")
    print(f"  数值增长因子: {ode_diag['growth_numerical']:.2e}")

    return ftcs_result, cn_result


# ===================================================================
# 阶段 5: 多道耦合分析
# ===================================================================
def stage5_channel_coupling(params: OpticalPotentialParams):
    """多道耦合通道分析"""
    print_header("阶段 5: 多道耦合与角动量分解")

    # 5.1 角动量耦合通道枚举
    print_subheader("5.1 角动量耦合通道 (J=1/2+, 208Pb 基态)")
    coupler = AngularMomentumCoupler(
        J_total=0.5, parity=1, J_target=0.0,
        s_projectile=0.5, pi_target=1)

    channels = coupler.enumerate_channels()
    print(f"  总角动量 J = {coupler.J_total}")
    print(f"  宇称 pi = {'+' if coupler.parity > 0 else '-'}")
    print(f"  靶核自旋 J_t = {coupler.J_target}")
    print(f"  允许通道数: {len(channels)}")

    for ch in channels[:8]:
        print(f"    {ch['label']:12s}: CG权重={ch['cg_weight']:.3f}")

    topo = coupler.coupling_topology()
    print(f"\n  耦合拓扑:")
    print(f"    通道数: {topo['n_channels']}")
    print(f"    耦合对数: {topo['n_couplings']}")
    print(f"    耦合密度: {topo['coupling_density']:.3f}")

    # 5.2 耦合图构建
    print_subheader("5.2 反应通道耦合图")
    graph = CouplingGraph()

    # 添加弹性道和几个非弹性道
    excitation_energies = [0.0, 2.615, 3.475, 4.089]  # 208Pb 低激发态 [MeV]
    state_labels = ['gs(0+)', '3-(2.615)', '4+(3.475)', '5-(4.089)']

    for i, (Ex, label) in enumerate(zip(excitation_energies, state_labels)):
        ch = ReactionChannel(
            alpha=label,
            A_target=208, J_target=0.0 if i == 0 else float(i),
            excitation=Ex,
            l_quantum=i, J_total=0.5,
            mass_reduced=params.mass_reduced,
            energy_cm=params.energy * 207.0 / 209.0,
        )
        graph.add_channel(ch)
        print(f"  通道 {i}: {ch}")

    # 添加耦合 (四极耦合为主)
    for i in range(len(excitation_energies)):
        for j in range(len(excitation_energies)):
            if i != j and abs(i - j) <= 2:
                V_ij = 2.0 / (1.0 + abs(i - j))  # 简化耦合强度
                graph.add_coupling(i, j, V_ij)

    # 图论分析
    in_deg, out_deg = graph.compute_degrees()
    print(f"\n  通道入度: {in_deg}")
    print(f"  通道出度: {out_deg}")

    euler = graph.is_eulerian()
    euler_desc = {0: '非欧拉', 1: '开欧拉迹', 2: '闭欧拉回路'}
    print(f"  欧拉性质: {euler_desc[euler]}")

    components = graph.coupling_connectivity()
    print(f"  连通分量数: {len(components)}")
    for i, comp in enumerate(components):
        print(f"    分量 {i}: {comp}")

    return graph, channels


# ===================================================================
# 阶段 6: 离散层传播
# ===================================================================
def stage6_wave_propagation(pot: OpticalPotential, params: OpticalPotentialParams):
    """离散层传递矩阵波函数传播"""
    print_header("阶段 6: 离散层传递矩阵波函数传播")

    propagator = DiscreteLayerPropagator(
        pot, n_layers=200, r_max=30.0, r_start=0.1)

    r_grid = np.linspace(0.1, 30.0, 2000)

    for l in range(4):
        V = pot.total_potential(r_grid, l_quantum=l)
        S_l, M_total = propagator.propagate(V, r_grid, l)
        T_l = 1.0 - abs(S_l) ** 2

        print(f"\n  l={l}: |S_l|={abs(S_l):.6f}, "
              f"T_l={T_l:.6f}, "
              f"delta={np.angle(S_l)/2:.4f} rad")

    # 波函数剖面 (l=0)
    V0 = pot.total_potential(r_grid, l_quantum=0)
    r_bnd, u_prof = propagator.compute_wavefunction_profile(V0, r_grid, l=0)
    print(f"\n  l=0 波函数传播:")
    print(f"    层数: {propagator.n_layers}")
    print(f"    层宽: {propagator.layer_width:.4f} fm")
    print(f"    |u(0)|: {abs(u_prof[0]):.2e}")
    print(f"    |u(R_max)|: {abs(u_prof[-1]):.4f}")
    print(f"    max|u|: {np.max(np.abs(u_prof)):.4f}")


# ===================================================================
# 阶段 7: 模型比较
# ===================================================================
def stage7_model_comparison(pot: OpticalPotential, params: OpticalPotentialParams):
    """光学势模型序贯比较检验"""
    print_header("阶段 7: 光学势模型序贯比较检验")

    factory = OpticalPotentialModelFactory()
    tester = SequentialModelTester(alpha=0.05, lambda_max=5.0, bet_strategy='ons')

    r = np.linspace(0.5, 15.0, 100)
    R = pot.R_v

    # 生成参考数据 (Woods-Saxon)
    V_ws = factory.woods_saxon(r, 48.0, R, 0.65)
    # 加噪声
    rng = np.random.RandomState(42)
    data = V_ws + rng.randn(len(r)) * 1.0

    # 模型 A: Woods-Saxon (正确)
    model_a = factory.woods_saxon(r, 48.0, R, 0.65)
    # 模型 B: 方势阱 (错误)
    model_b = factory.square_well(r, 48.0, R, 0.5)

    print(f"\n  比较: Woods-Saxon (模型A) vs 方势阱 (模型B)")
    print(f"  数据点: {len(r)}")

    result = tester.test_models(data, model_a, model_b, sigma_noise=1.0)
    print(f"\n  === 序贯检验结果 ===")
    print(f"  拒绝 H0: {'是' if result['reject_null'] else '否'}")
    print(f"  获胜模型: {result['winner']}")
    print(f"  停时: {result['stopping_time']} / {result['n_data']}")
    print(f"  最终财富: {result['final_wealth']:.4e}")
    print(f"  阈值 1/alpha: {result['threshold']:.1f}")

    # 功效分析
    print_subheader("7.1 蒙特卡洛功效分析")
    power_result = tester.power_analysis(effect_size=0.5, n_samples=100,
                                          n_trials=30, seed=42)
    print(f"  效应量: {power_result['effect_size']}")
    print(f"  检验功效: {power_result['power']:.2f}")
    print(f"  试验次数: {power_result['n_trials']}")


# ===================================================================
# 阶段 8: 收敛性研究
# ===================================================================
def stage8_convergence(pot: OpticalPotential, params: OpticalPotentialParams):
    """有限差分收敛性系统研究"""
    print_header("阶段 8: 高阶有限差分收敛性研究")

    study = ConvergenceStudy(pot, r_max=30.0)

    # 8.1 Numerov 网格收敛
    print_subheader("8.1 Numerov 四阶格式网格收敛")
    conv_num = study.grid_convergence_study(l=0, scheme='numerov')
    print(f"  {'N':>8} {'dr':>10} {'delta':>12} {'error':>12}")
    for i, n in enumerate(conv_num['n_points']):
        err = (conv_num['errors_vs_reference'][i] if i < len(conv_num['errors_vs_reference'])
               else 0.0)
        print(f"  {n:8d} {conv_num['spacings'][i]:10.6f} "
              f"{conv_num['phase_shifts'][i]:12.8f} {err:12.2e}")
    print(f"  理论阶数: {conv_num['theoretical_order']}")
    if conv_num['convergence_rates']:
        print(f"  观测收敛阶: {[f'{p:.2f}' for p in conv_num['convergence_rates']]}")

    # 8.2 二阶 CD 网格收敛
    print_subheader("8.2 二阶中心差分格式网格收敛")
    conv_cd = study.grid_convergence_study(l=0, scheme='cd2',
                                            n_points_list=[200, 400, 800, 1600])
    print(f"  {'N':>8} {'dr':>10} {'delta':>12} {'error':>12}")
    for i, n in enumerate(conv_cd['n_points']):
        err = (conv_cd['errors_vs_reference'][i] if i < len(conv_cd['errors_vs_reference'])
               else 0.0)
        print(f"  {n:8d} {conv_cd['spacings'][i]:10.6f} "
              f"{conv_cd['phase_shifts'][i]:12.8f} {err:12.2e}")
    print(f"  理论阶数: {conv_cd['theoretical_order']}")
    if conv_cd['convergence_rates']:
        print(f"  观测收敛阶: {[f'{p:.2f}' for p in conv_cd['convergence_rates']]}")

    # 8.3 Richardson 外推
    print_subheader("8.3 Richardson 外推")
    values = np.array(conv_num['phase_shifts'][-3:])
    spacings = np.array(conv_num['spacings'][-3:])
    extrap, err_est = study.richardson_extrapolation(values, spacings, order=4)
    print(f"  Numerov 外推相移: {extrap:.8f}")
    print(f"  误差估计: {err_est:.2e}")

    # 8.4 多 l 值收敛
    print_subheader("8.4 多分波收敛性")
    multi_l = study.multi_l_convergence(l_values=[0, 1, 2, 3, 4], n_points=2000)
    for key, val in multi_l.items():
        print(f"  {key}: delta={val['phase_shift_real']:.6f}, "
              f"|Im(delta)|={abs(val['phase_shift_imag']):.6f}, "
              f"norm={val['wavefunction_norm']:.2f}")

    # 8.5 概率流连续性验证
    print_subheader("8.5 概率流连续性方程验证")
    solver = RadialSchrodingerSolver(pot, r_max=30.0, n_points=2000)
    V0 = pot.total_potential(solver.r, l_quantum=0)
    _, u0 = solver.solve_numerov(0, V0)
    cont_res = continuity_residual(u0, solver.r, V0, params.energy, params.mass_reduced)
    print(f"  连续性方程残差: {cont_res:.4e}")
    print(f"  物理含义: 残差反映数值解的概率流守恒程度")
    print(f"  理想值: 0 (弹性散射时)")


# ===================================================================
# 主程序
# ===================================================================
def main():
    """
    核反应光学模型计算 — 完整分析流程

    物理系统: n + 208Pb @ E_lab = 14.1 MeV
    方法: 高阶有限差分 + 稳定性分析 + 截面预测
    """
    print("=" * 70)
    print("  核反应光学模型: 高阶有限差分与稳定性分析")
    print("  Nuclear Optical Model: High-Order Finite Difference")
    print("  and Stability Analysis")
    print("=" * 70)
    print(f"\n  物理系统: 中子 + 208Pb @ E_lab = 14.1 MeV")
    print(f"  (DD 聚变中子能量)")
    print(f"  计算方法: Numerov 四阶 + 传递矩阵 + 序贯检验")
    print(f"  日期: 2026-06-07 (可复现实验)")

    # 执行所有阶段
    params, pot = stage1_optical_potential()
    phase_shifts, s_matrix, solver = stage2_radial_equation(pot, params)
    cross_sections = stage3_cross_sections(s_matrix, params)
    ftcs_result, cn_result = stage4_stability(pot, params)
    graph, channels = stage5_channel_coupling(params)
    stage6_wave_propagation(pot, params)
    stage7_model_comparison(pot, params)
    stage8_convergence(pot, params)

    # 总结
    print_header("计算完成 — 总结")
    print(f"\n  已完成分析:")
    print(f"  [1] 光学势构造 (Woods-Saxon + 分形表面 + Logistic 吸收)")
    print(f"  [2] 径向薛定谔方程求解 (Numerov 四阶, {len(phase_shifts)} 个分波)")
    print(f"  [3] 反应截面计算 (sigma_tot = {cross_sections['sigma_total']:.2f} fm^2)")
    print(f"  [4] 稳定性分析 (FTCS 不稳定, CN 酉稳定, Numerov 条件稳定)")
    print(f"  [5] 多道耦合图论分析 ({len(channels)} 个耦合通道)")
    print(f"  [6] 离散层波函数传播 (200 层传递矩阵)")
    print(f"  [7] 光学势模型序贯检验 (赌徒财富过程)")
    print(f"  [8] 收敛性研究 (Richardson 外推, 多格式对比)")

    print(f"\n  种子项目映射:")
    print(f"  [P01] 1172_sabrin1997 SEMAFOVAE     → 层次化光学势参数化")
    print(f"  [P02] 1141_sshekhar17 betting        → 序贯模型选择检验")
    print(f"  [P03] 792_nearest_interp_1d          → 势层间最近邻插值")
    print(f"  [P04] 353_fd1d_advection_ftcs        → FTCS 不稳定格式演示")
    print(f"  [P05] 286_digraph_arc                → 反应通道耦合有向图")
    print(f"  [P06] 446_fractal_coastline          → 分形核表面微扰")
    print(f"  [P07] 211_continuity_exact           → 概率流连续性验证")
    print(f"  [P08] 1373_uniform                   → 蒙特卡洛均匀采样")
    print(f"  [P09] 1428_zero_chandrupatla         → S 矩阵极点共振搜索")
    print(f"  [P10] 905_pram                       → 角动量通道组合分解")
    print(f"  [P11] 1069_discrete_flow_models      → 离散层波函数传播")
    print(f"  [P12] 1374_unstable_ode              → 不稳定 ODE 诊断")
    print(f"  [P13] 702_logistic_ode               → Logistic 表面吸收势")
    print(f"  [P14] 815_norm_rms                   → RMS 波函数范数")
    print(f"  [P15] 813_norm_l2                    → L2 散射波函数范数")

    print("\n" + "=" * 70)
    print("  计算全部完成, 无报错.")
    print("=" * 70)


if __name__ == '__main__':
    main()
