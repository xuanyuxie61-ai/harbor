"""
main.py - 火箭发动机燃烧不稳定性的多物理场耦合分析系统
========================================================
统一入口文件，零参数可运行。

项目概述:
=========
本系统围绕"液体火箭发动机燃烧室热声不稳定性"这一前沿燃烧科学问题，
集成了15个种子项目的核心算法，构建了从燃烧室几何建模、喷注器布局优化、
喷雾动力学、两相流动、燃烧波传播、声场模态分析、火焰响应函数到
热声耦合振荡器预测的完整分析链条。

执行流程:
    1. 燃烧室几何建模 (Joukowsky变换 + CORDIC + ICE网格)
    2. 喷注器面板优化布局 (背包问题 + 拼板密铺)
    3. 喷雾液滴分布优化 (CVT + 蒸发模拟)
    4. 气液两相流动分析 (Stokes流)
    5. 一维燃烧波求解 (Jacobi迭代 + Newton插值)
    6. 声场模态分析 (FEM + Bessel函数)
    7. 火焰传递函数 (Chebyshev插值 + Lebesgue稳定性)
    8. 热声耦合振荡器预测 (非线性ODE)
    9. 综合稳定性评估与报告输出
"""

import numpy as np
import sys
import time

# ============================================================
# 导入各模块
# ============================================================
from utils import (
    cordic_cos_sin, cordic_arctan2,
    circle_monomial_integral, gamma_function_half_integer,
    specific_impulse_ideal, combustion_temperature,
    safe_divide, robust_sqrt, check_finite_array
)
from geometry_model import CombustionChamberGeometry
from injector_layout import InjectorLayoutOptimizer
from spray_dynamics import SprayDistributionCVT
from two_phase_flow import StokesDropletFlow, TwoPhaseFlowSolver
from combustion_wave import NewtonInterpolation, CombustionRateModel, ReactionDiffusionSolver
from acoustic_modes import FEMBasis2DTriangle, AcousticModeAnalyzer, FEMHelmholtzSolver
from flame_response import ChebyshevNDInterpolation, LebesgueStabilityAnalyzer, FlameTransferFunction
from thermoacoustic_oscillator import ThermoacousticOscillator, MultiModeThermoacousticSystem


def print_section(title: str):
    """打印分隔的章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_geometry_modeling():
    """步骤1: 燃烧室几何建模。"""
    print_section("STEP 1: 燃烧室几何建模")
    
    geo = CombustionChamberGeometry(
        chamber_length=0.60,
        chamber_diameter=0.30,
        throat_radius=0.075,
        exit_radius=0.30,
        nozzle_half_angle_deg=15.0
    )
    
    print(f"  燃烧室体积: {geo.volume:.6f} m³")
    print(f"  面积扩张比: {geo.epsilon:.2f}")
    print(f"  声学等效长度: {geo.acoustic_length():.4f} m")
    
    # 纵向模态频率
    freqs = geo.longitudinal_mode_frequencies(n_modes=5, sound_speed=1200.0)
    print(f"  纵向声学模态频率: {np.round(freqs, 1)} Hz")
    
    # 生成网格
    grid = geo.generate_axisymmetric_grid(n_z=60, n_r=20)
    print(f"  计算网格: {grid['n_vertices']} 顶点, {grid['n_elements']} 单元")
    
    # Joukowsky喷管型线
    contour = geo.apply_joukowsky_nozzle_contour(center_offset=-0.05, circle_radius=0.12)
    print(f"  Joukowsky喷管型线: {contour.shape[0]} 点")
    
    # CORDIC验证
    c, s = cordic_cos_sin(np.deg2rad(15.0), n_iter=50)
    print(f"  CORDIC验证: cos(15°)={c:.12f}, sin(15°)={s:.12f}")
    
    return geo


def run_injector_optimization():
    """步骤2: 喷注器面板优化布局。"""
    print_section("STEP 2: 喷注器面板优化布局")
    
    opt = InjectorLayoutOptimizer(
        panel_radius=0.12,
        element_outer_diameter=8.0e-3,
        element_mass=0.015,
        target_total_flow=120.0
    )
    
    # 生成候选位置
    n_cand = opt.generate_candidate_positions_triangular(n_layers=6)
    print(f"  候选位置数: {n_cand}")
    
    # 贪心启发式求解
    result = opt.solve_greedy_heuristic()
    print(f"  选中单元数: {result['n_selected']}")
    print(f"  总流量: {result['total_weight']:.2f} kg/s")
    print(f"  布局均匀性指数: {result['uniformity_index']:.4f} (越小越均匀)")
    
    # 氧燃比分布
    mr = opt.compute_mixture_ratio_distribution(result['selected_indices'])
    print(f"  氧燃比分布: mean={mr['mean']:.3f}, std={mr['std']:.4f}")
    
    # 背包问题求解 (如果候选数可控)
    if n_cand <= 25:
        result_bf = opt.solve_brute_force_knapsack(time_limit_seconds=5.0)
        print(f"  暴力求解结果: {result_bf['n_selected']} 单元, "
              f"uniformity={result_bf['uniformity_index']:.4f}")
    
    return opt, result


def run_spray_dynamics():
    """步骤3: 喷雾液滴分布优化与蒸发。"""
    print_section("STEP 3: 喷雾液滴分布优化 (CVT)")
    
    spray = SprayDistributionCVT(
        n_droplets=300,
        droplet_diameter_mean=80e-6,
        gas_temperature=3000.0,
        gas_pressure=7.0e6
    )
    
    # CVT优化
    result = spray.optimize_distribution(n_iterations=40, n_samples=5000)
    print(f"  CVT收敛迭代: {result['iterations']}")
    print(f"  最终能量泛函: {result['final_energy']:.6e}")
    print(f"  液滴Sauter平均直径: {result['mean_diameter']*1e6:.2f} μm")
    
    # 蒸发寿命模拟
    lifetime = spray.simulate_droplet_lifetime(dt=2e-5, n_steps=800)
    print(f"  蒸发完成比例: {lifetime['evaporation_fraction']:.3f}")
    
    # 喷雾统计
    stats = spray.compute_spray_statistics()
    print(f"  平均轴向位置: {stats['mean_axial_position']:.4f} m")
    print(f"  平均径向位置: {stats['mean_radial_position']:.4f} m")
    
    return spray


def run_two_phase_flow(geo):
    """步骤4: 气液两相流动分析。"""
    print_section("STEP 4: 气液两相流动分析 (Stokes流)")
    
    # 单液滴Stokes分析
    stokes = StokesDropletFlow(
        droplet_radius=40e-6,
        free_stream_velocity=30.0,
        gas_viscosity=8.5e-5,
        liquid_viscosity=1.0e-3
    )
    
    print(f"  液滴雷诺数: {stokes.Re:.4f}")
    print(f"  Stokes阻力系数: {stokes.stokes_drag_coefficient():.4f}")
    print(f"  阻力: {stokes.stokes_drag_force():.6e} N")
    print(f"  Nusselt数: {stokes.compute_nusselt_number():.3f}")
    print(f"  Sherwood数: {stokes.compute_sherwood_number():.3f}")
    
    # 1D两相流求解
    flow = TwoPhaseFlowSolver(geo, n_z=150)
    result = flow.solve_steady_1d()
    print(f"  蒸发特征长度: {result['evaporation_length']:.4f} m")
    print(f"  最大气相速度: {np.max(result['gas_velocity']):.2f} m/s")
    print(f"  出口液滴直径: {result['droplet_diameter'][-1]*1e6:.2f} μm")
    
    return stokes, flow


def run_combustion_wave():
    """步骤5: 一维燃烧波传播。"""
    print_section("STEP 5: 一维燃烧波 (反应-扩散方程)")
    
    # Newton插值测试
    x_data = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]) * 1e6
    rate_model = CombustionRateModel(a_coeff=1.5e-5, n_coeff=0.5)
    y_data = np.array([rate_model.regression_rate(x) for x in x_data])
    interp = NewtonInterpolation(x_data, y_data)
    
    test_pressures = np.array([2.0, 6.0, 12.0]) * 1e6
    print("  Newton插值验证 (燃烧速率):")
    for p in test_pressures:
        exact = rate_model.regression_rate(p)
        approx = interp.evaluate(p)
        print(f"    P={p/1e6:.1f} MPa: 精确={exact*1e3:.4f} mm/s, "
              f"插值={approx*1e3:.4f} mm/s, 误差={abs(exact-approx)/exact*100:.4f}%")
    
    # 反应扩散求解
    rd = ReactionDiffusionSolver(
        domain_length=0.015,
        n_points=151,
        activation_energy=1.26e5,
        temperature_burned=3600.0
    )
    
    print(f"  Zeldovich数: {rd.beta:.2f}")
    print(f"  理论层流火焰速度: {rd.S_L_theoretical:.4f} m/s")
    
    result = rd.solve_steady_jacobi(max_iterations=50000, tolerance=1e-8)
    print(f"  Jacobi迭代收敛: {result['iterations']} 次")
    print(f"  火焰位置: {result['flame_position']*1e3:.2f} mm")
    print(f"  火焰厚度: {result['flame_thickness']*1e6:.2f} μm")
    print(f"  数值火焰速度: {result['S_L_numerical']:.4f} m/s")
    
    return rd, result


def run_acoustic_modes(geo):
    """步骤6: 声场模态分析。"""
    print_section("STEP 6: 声场模态分析 (FEM)")
    
    analyzer = AcousticModeAnalyzer(
        chamber_length=geo.L_c,
        chamber_radius=geo.R_c,
        sound_speed=1200.0
    )
    
    # 纵向模态
    L_modes = analyzer.longitudinal_modes()
    print("  纵向声学模态:")
    for m in L_modes['modes']:
        print(f"    L{m['n']}: f = {m['frequency']:.1f} Hz")
    
    # 径向/切向模态
    R_modes = analyzer.radial_modes()
    print("  径向/切向模态 (前6个):")
    for m in R_modes['modes'][:6]:
        print(f"    {m['type']}{m['m']}{m['n']}: f = {m['frequency']:.1f} Hz")
    
    # 正交性验证
    ortho = analyzer.compute_orthogonality_integrals("L")
    diag = np.diag(ortho)
    offdiag_max = np.max(np.abs(ortho - np.diag(diag)))
    print(f"  模态正交性验证: 最大非对角元 = {offdiag_max:.6e}")
    
    # 圆积分验证
    I22 = circle_monomial_integral(2, 2)
    print(f"  圆积分验证 x²y²: {I22:.10f} (理论=π/4≈0.785398)")
    
    # FEM本征值验证
    fem = FEMHelmholtzSolver(length=geo.L_c, n_elements=100)
    fem_result = fem.solve_eigenvalue(n_modes=5)
    print(f"  FEM本征频率: {np.round(fem_result['frequencies'], 1)} Hz")
    
    # Rayleigh准则测试
    z = np.linspace(0, geo.L_c, 100)
    p_mode = np.cos(np.pi * z / (2 * geo.L_c))
    q_osc = np.exp(-10 * (z - geo.L_c * 0.3) ** 2)
    rayleigh = analyzer.rayleigh_criterion(q_osc, p_mode)
    print(f"  Rayleigh准则: {rayleigh:.4e} (正值->不稳定)")
    
    return analyzer


def run_flame_response():
    """步骤7: 火焰传递函数与插值稳定性。"""
    print_section("STEP 7: 火焰传递函数 (FTF)")
    
    # Chebyshev插值
    coeffs_2d = np.array([
        [1.0, 0.3, -0.1],
        [0.2, -0.05, 0.02],
        [-0.05, 0.01, 0.005]
    ])
    cheb = ChebyshevNDInterpolation(coeffs_2d, domains=[(0, 1), (0, 1)])
    cheb_val = cheb.evaluate(np.array([0.5, 0.5]))
    print(f"  2D Chebyshev插值 (0.5,0.5): {cheb_val:.6f}")
    
    # Lebesgue常数比较
    n_test = 12
    eq_nodes = np.linspace(-1, 1, n_test)
    leb_eq = LebesgueStabilityAnalyzer(eq_nodes)
    lambda_eq = leb_eq.lebesgue_constant()
    
    cheb_nodes = leb_eq.chebyshev_nodes(n_test, -1, 1)
    leb_cheb = LebesgueStabilityAnalyzer(cheb_nodes)
    lambda_cheb = leb_cheb.lebesgue_constant()
    
    print(f"  Lebesgue常数 (等距节点, n={n_test}): {lambda_eq:.2f}")
    print(f"  Lebesgue常数 (Chebyshev节点, n={n_test}): {lambda_cheb:.2f}")
    print(f"  插值稳定性提升: {lambda_eq/lambda_cheb:.1f}x")
    
    # 火焰传递函数
    ftf = FlameTransferFunction(
        interaction_index=1.2,
        time_delay_ms=2.0,
        cutoff_frequency_hz=1000.0
    )
    
    data = ftf.generate_discrete_data(n_points=30)
    print(f"  FTF离散数据点: {len(data['frequencies'])}")
    
    f_test = 500.0
    ftf_analytical = ftf.analytical_ftf(f_test)
    ftf_interp = ftf.interpolate_ftf(f_test)
    print(f"  FTF @ {f_test} Hz:")
    print(f"    解析: |F|={np.abs(ftf_analytical):.4f}, "
          f"arg={np.degrees(np.angle(ftf_analytical)):.1f}°")
    print(f"    插值: |F|={np.abs(ftf_interp):.4f}, "
          f"arg={np.degrees(np.angle(ftf_interp)):.1f}°")
    
    stability = ftf.compute_nyquist_stability_margin()
    print(f"  增益裕度: {stability['gain_margin_db']:.2f} dB")
    print(f"  相位裕度: {stability['phase_margin_deg']:.2f}°")
    
    return ftf


def run_thermoacoustic_oscillator(analyzer):
    """步骤8: 热声耦合振荡器预测。"""
    print_section("STEP 8: 热声耦合振荡器")
    
    # TODO(Hole_3): 正确建立热声耦合振荡器参数桥梁
    # 需要从 analyzer (AcousticModeAnalyzer) 中提取声学模态信息,
    # 并选择合适的阻尼系数、火焰增益系数和非线性饱和系数,
    # 使得振荡器能够正确反映燃烧室的热声不稳定性特征。
    # 关键约束:
    #   1. 单模态振荡器的 natural_frequency_hz 必须来自 acoustic_modes 分析结果
    #   2. acoustic_damping 和 flame_gain_coefficient 的选取必须满足物理一致性
    #      (即 alpha_eff = alpha_acoustic - gamma*n 的符号决定稳定性)
    #   3. 多模态系统的频率和阻尼数组必须与单模态参数物理兼容
    # 提示: 参考声学模态分析器的 longitudinal_modes() 返回结构
    
    # 单模态振荡器 (当前参数不完整, 需要修复)
    osc = ThermoacousticOscillator(
        natural_frequency_hz=0.0,  # TODO: 从 analyzer 获取基频
        acoustic_damping=0.0,      # TODO: 选择物理合理的阻尼系数
        flame_gain_coefficient=0.0,# TODO: 选择物理合理的火焰增益
        nonlinear_saturation=0.0,  # TODO: 选择合理的非线性饱和系数
        initial_pressure_disturbance_pa=200.0
    )
    
    print(f"  基频: {osc.omega/(2*np.pi):.1f} Hz")
    print(f"  有效阻尼: {osc.alpha_eff:.2f} 1/s")
    print(f"  线性稳定性: {'UNSTABLE (自激振荡)' if osc.is_unstable else 'STABLE'}")
    print(f"  极限环振幅预测: {osc.limit_cycle_amplitude():.2f} Pa")
    
    result = osc.rk4_integrate((0, 0.04), n_steps=20000)
    metrics = osc.compute_oscillation_metrics(result["t"], result["pressure"])
    
    print(f"  振荡特性:")
    print(f"    峰峰值压力脉动: {metrics['peak_to_peak_pa']:.2f} Pa")
    print(f"    RMS压力脉动: {metrics['rms_pa']:.2f} Pa")
    print(f"    估计频率: {metrics['estimated_frequency_hz']:.1f} Hz")
    print(f"    线性增长率: {metrics['growth_rate_1_per_s']:.2f} 1/s")
    
    # 多模态系统 (当前参数不完整, 需要修复)
    freqs = np.array([0.0, 0.0, 0.0])  # TODO: 从 analyzer 获取前3阶频率
    damping = np.array([0.0, 0.0, 0.0])  # TODO: 设置与各阶模态兼容的阻尼
    multi = MultiModeThermoacousticSystem(freqs, damping)
    multi_result = multi.integrate(t_span=(0, 0.03), n_steps=15000)
    
    for i in range(min(3, len(freqs))):
        amp = np.max(np.abs(multi_result['mode_pressures'][i][-500:]))
        print(f"  多模态 Mode {i+1} 稳态振幅: {amp:.2f} Pa")
    
    return osc, multi


def run_comprehensive_assessment(geo, osc, analyzer, ftf):
    """步骤9: 综合稳定性评估。"""
    print_section("STEP 9: 综合稳定性评估")
    
    # 比冲估算
    I_sp = specific_impulse_ideal(7e6, geo.epsilon)
    print(f"  理想比冲估算: {I_sp:.2f} s")
    
    # 燃烧温度
    T_c = combustion_temperature(7e6, 2.56)
    print(f"  绝热燃烧温度: {T_c:.1f} K")
    
    # 稳定性判据汇总
    print("  稳定性判据汇总:")
    print(f"    1. 纵向基频: {analyzer.longitudinal_modes()['modes'][0]['frequency']:.1f} Hz")
    print(f"    2. 有效阻尼: {osc.alpha_eff:.2f} 1/s")
    print(f"    3. 线性稳定性: {'UNSTABLE' if osc.is_unstable else 'STABLE'}")
    
    if osc.is_unstable:
        A_lim = osc.limit_cycle_amplitude()
        p_mean = 7e6
        print(f"    4. 极限环压力脉动比: {A_lim/p_mean*100:.4f}%")
        if A_lim / p_mean > 0.05:
            print(f"    5. 风险评估: HIGH (压力脉动 > 5%)")
        elif A_lim / p_mean > 0.02:
            print(f"    5. 风险评估: MEDIUM (压力脉动 2-5%)")
        else:
            print(f"    5. 风险评估: LOW (压力脉动 < 2%)")
    else:
        print(f"    4. 系统稳定，无自激振荡风险")
    
    # Nyquist稳定性
    stability = ftf.compute_nyquist_stability_margin()
    gm_db = stability['gain_margin_db']
    if gm_db != np.inf and gm_db < 6.0:
        print(f"    6. Nyquist增益裕度不足: {gm_db:.2f} dB < 6 dB")
    
    print("\n" + "=" * 70)
    print("  分析完成。")
    print("=" * 70)


def main():
    """主程序入口。"""
    print("\n" + "#" * 70)
    print("#  火箭发动机燃烧不稳定性的多物理场耦合分析系统")
    print("#  Rocket Engine Combustion Instability Analysis System")
    print("#" * 70)
    print(f"\n  运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python版本: {sys.version.split()[0]}")
    
    start_time = time.time()
    
    try:
        # 执行各分析步骤
        geo = run_geometry_modeling()
        opt, opt_result = run_injector_optimization()
        spray = run_spray_dynamics()
        stokes, flow = run_two_phase_flow(geo)
        rd, rd_result = run_combustion_wave()
        analyzer = run_acoustic_modes(geo)
        ftf = run_flame_response()
        osc, multi = run_thermoacoustic_oscillator(analyzer)
        run_comprehensive_assessment(geo, osc, analyzer, ftf)
        
        elapsed = time.time() - start_time
        print(f"\n  总运行时间: {elapsed:.3f} 秒")
        print("\n  [SUCCESS] 所有分析步骤成功完成，无报错。\n")
        
    except Exception as e:
        print(f"\n  [ERROR] 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
