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
    
    # 单模态振荡器
    osc = ThermoacousticOscillator(
        natural_frequency_hz=analyzer.longitudinal_modes()['modes'][0]['frequency'],
        acoustic_damping=80.0,
        flame_gain_coefficient=120.0,
        nonlinear_saturation=5e7,
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
    
    # 多模态系统
    freqs = L_modes = analyzer.longitudinal_modes()['frequencies'][:3]
    damping = np.array([80.0, 200.0, 350.0])
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


# ================================================================
# 测试用例（60个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: cordic_cos_sin 计算 cos(0)=1, sin(0)=0 ----
c0, s0 = cordic_cos_sin(0.0)
assert abs(c0 - 1.0) < 1e-12, '[TC01] cos(0) != 1 FAILED'
assert abs(s0 - 0.0) < 1e-12, '[TC01] sin(0) != 0 FAILED'

# ---- TC02: cordic_cos_sin 计算 cos(π/3)=0.5, sin(π/3)=√3/2 ----
import numpy as np
c60, s60 = cordic_cos_sin(np.pi / 3.0)
assert abs(c60 - 0.5) < 1e-6, '[TC02] cos(π/3) != 0.5 FAILED'
assert abs(s60 - np.sqrt(3.0) / 2.0) < 1e-6, '[TC02] sin(π/3) != √3/2 FAILED'

# ---- TC03: cordic_cos_sin 处理非有限值输入返回 NaN ----
import numpy as np
cn, sn = cordic_cos_sin(np.nan)
assert np.isnan(cn) and np.isnan(sn), '[TC03] CORDIC non-finite input should return NaN FAILED'

# ---- TC04: cordic_cos_sin 可复现性——同一输入两次结果一致 ----
c1, s1 = cordic_cos_sin(np.pi / 4.0)
c2, s2 = cordic_cos_sin(np.pi / 4.0)
assert abs(c1 - c2) < 1e-15, '[TC04] CORDIC cos not reproducible FAILED'
assert abs(s1 - s2) < 1e-15, '[TC04] CORDIC sin not reproducible FAILED'

# ---- TC05: cordic_arctan2 计算 arctan2(1,0) = π/2 ----
import numpy as np
a90 = cordic_arctan2(1.0, 0.0)
assert abs(a90 - np.pi / 2.0) < 1e-6, '[TC05] arctan2(1,0) != π/2 FAILED'

# ---- TC06: cordic_arctan2 计算 arctan2(1,1) = π/4 ----
import numpy as np
a45 = cordic_arctan2(1.0, 1.0)
assert abs(a45 - np.pi / 4.0) < 1e-6, '[TC06] arctan2(1,1) != π/4 FAILED'

# ---- TC07: cordic_arctan2 原点 (0,0) 返回 0 ----
a00 = cordic_arctan2(0.0, 0.0)
assert abs(a00) < 1e-15, '[TC07] arctan2(0,0) != 0 FAILED'

# ---- TC08: gamma_function_half_integer Γ(2) = 1 ----
g2 = gamma_function_half_integer(2)
assert abs(g2 - 1.0) < 1e-12, '[TC08] Γ(2) != 1 FAILED'

# ---- TC09: gamma_function_half_integer Γ(4/2)=Γ(2) = 1 ----
g4 = gamma_function_half_integer(4)
assert abs(g4 - 1.0) < 1e-12, '[TC09] Γ(2) != 1 FAILED'

# ---- TC10: gamma_function_half_integer Γ(1) = √π ----
import numpy as np
g1 = gamma_function_half_integer(1)
assert abs(g1 - np.sqrt(np.pi)) < 1e-10, '[TC10] Γ(1) != √π FAILED'

# ---- TC11: gamma_function_half_integer 非正值抛出 ValueError ----
try:
    gamma_function_half_integer(0)
    assert False, '[TC11] Γ(0) should raise ValueError FAILED'
except ValueError:
    pass

# ---- TC12: circle_monomial_integral 奇指数返回 0 ----
I_odd1 = circle_monomial_integral(1, 0)
assert abs(I_odd1) < 1e-15, '[TC12] odd exponent (1,0) should return 0 FAILED'
I_odd2 = circle_monomial_integral(0, 3)
assert abs(I_odd2) < 1e-15, '[TC12] odd exponent (0,3) should return 0 FAILED'

# ---- TC13: circle_monomial_integral x²y² 理论值 = π/4 ----
import numpy as np
I22 = circle_monomial_integral(2, 2)
assert abs(I22 - np.pi / 4.0) < 1e-6, '[TC13] ∮x²y² != π/4 FAILED'

# ---- TC14: circle_monomial_integral 对称性 x^2y^0 == x^0y^2 ----
I20 = circle_monomial_integral(2, 0)
I02 = circle_monomial_integral(0, 2)
assert abs(I20 - I02) < 1e-12, '[TC14] symmetry x² vs y² broken FAILED'

# ---- TC15: safe_divide 正常除法 6/2 = 3 ----
sd = safe_divide(6.0, 2.0)
assert abs(sd - 3.0) < 1e-15, '[TC15] safe_divide(6,2) != 3 FAILED'

# ---- TC16: safe_divide 除零返回默认值 ----
sd0 = safe_divide(6.0, 0.0, default=99.0)
assert abs(sd0 - 99.0) < 1e-15, '[TC16] safe_divide by zero should return default FAILED'

# ---- TC17: robust_sqrt 正数 sqrt(16) = 4 ----
rs = robust_sqrt(16.0)
assert abs(rs - 4.0) < 1e-12, '[TC17] robust_sqrt(16) != 4 FAILED'

# ---- TC18: robust_sqrt 负数返回 NaN ----
import numpy as np
rs_neg = robust_sqrt(-100.0)
assert np.isnan(rs_neg), '[TC18] robust_sqrt(-100) should be NaN FAILED'

# ---- TC19: robust_sqrt 微小负数返回 0 ----
rs_tiny = robust_sqrt(-1e-15)
assert abs(rs_tiny) < 1e-14, '[TC19] robust_sqrt(-1e-15) should be 0 FAILED'

# ---- TC20: check_finite_array 有限数组不抛异常 ----
import numpy as np
arr_fine = np.array([1.0, 2.0, 3.0])
try:
    check_finite_array(arr_fine, "test")
except ValueError:
    assert False, '[TC20] check_finite_array failed on valid array FAILED'

# ---- TC21: combustion_temperature 返回值在合理范围且有限 ----
import numpy as np
Tc = combustion_temperature(7e6, 2.56)
assert Tc > 3000.0 and Tc < 4000.0, '[TC21] combustion temperature out of range FAILED'
assert np.isfinite(Tc), '[TC21] combustion temperature not finite FAILED'

# ---- TC22: specific_impulse_ideal 返回正值且有限 ----
import numpy as np
Isp = specific_impulse_ideal(7e6, 20.0)
assert Isp > 100.0, '[TC22] specific impulse too low FAILED'
assert np.isfinite(Isp), '[TC22] specific impulse not finite FAILED'

# ---- TC23: NewtonInterpolation 节点处精确恢复 ----
import numpy as np
x_ni = np.array([1.0, 3.0, 5.0])
y_ni = np.array([2.0, 8.0, 26.0])
interp_ni = NewtonInterpolation(x_ni, y_ni)
assert abs(interp_ni.evaluate(3.0) - 8.0) < 1e-12, '[TC23] Newton interp at node FAILED'
assert abs(interp_ni.evaluate(1.0) - 2.0) < 1e-12, '[TC23] Newton interp at first node FAILED'

# ---- TC24: CombustionRateModel regression_rate 正值 ----
import numpy as np
rm = CombustionRateModel(a_coeff=1.5e-5, n_coeff=0.5)
r7 = rm.regression_rate(7e6)
assert r7 > 0.0, '[TC24] regression rate not positive FAILED'
assert np.isfinite(r7), '[TC24] regression rate not finite FAILED'

# ---- TC25: StokesDropletFlow stokes_drag_coefficient 在 (0,1] 内 ----
stokes = StokesDropletFlow(droplet_radius=40e-6, free_stream_velocity=30.0)
Cd = stokes.stokes_drag_coefficient()
assert Cd > 0.0 and Cd <= 1.0, '[TC25] Stokes drag coefficient out of (0,1] FAILED'

# ---- TC26: StokesDropletFlow stokes_drag_force 正值 ----
Fd = stokes.stokes_drag_force()
assert Fd > 0.0, '[TC26] Stokes drag force not positive FAILED'

# ---- TC27: StokesDropletFlow Nusselt >= 2 ----
Nu = stokes.compute_nusselt_number()
assert Nu >= 2.0, '[TC27] Nusselt number < 2 FAILED'

# ---- TC28: StokesDropletFlow Sherwood >= 2 ----
Sh = stokes.compute_sherwood_number()
assert Sh >= 2.0, '[TC28] Sherwood number < 2 FAILED'

# ---- TC29: StokesDropletFlow Basset历史力对小历史返回0 ----
import numpy as np
v_hist = np.array([30.0])
fb = stokes.basset_history_force(v_hist, 1e-5)
assert abs(fb) < 1e-10, '[TC29] Basset force for single-point history should be 0 FAILED'

# ---- TC30: CombustionChamberGeometry 体积正值且有限 ----
import numpy as np
geo = CombustionChamberGeometry(chamber_length=0.60, chamber_diameter=0.30)
assert geo.volume > 0.0, '[TC30] chamber volume not positive FAILED'
assert np.isfinite(geo.volume), '[TC30] chamber volume not finite FAILED'

# ---- TC31: CombustionChamberGeometry 面积扩张比 > 1 ----
assert geo.epsilon > 1.0, '[TC31] expansion ratio <= 1 FAILED'

# ---- TC32: CombustionChamberGeometry 声学等效长度正值 ----
L_eff = geo.acoustic_length()
assert L_eff > 0.0, '[TC32] acoustic length not positive FAILED'

# ---- TC33: CombustionChamberGeometry 纵向模态频率单调递增 ----
freqs = geo.longitudinal_mode_frequencies(5, 1200.0)
assert len(freqs) == 5, '[TC33] wrong number of modes FAILED'
for i in range(4):
    assert freqs[i] < freqs[i+1], f'[TC33] frequencies not monotonic at {i} FAILED'

# ---- TC34: CombustionChamberGeometry 网格生成正确 ----
grid = geo.generate_axisymmetric_grid(n_z=30, n_r=10)
assert grid['n_vertices'] > 0, '[TC34] grid vertices = 0 FAILED'
assert grid['n_elements'] > 0, '[TC34] grid elements = 0 FAILED'
assert grid['vertices'].shape[1] == 2, '[TC34] vertices shape wrong FAILED'

# ---- TC35: CombustionChamberGeometry Joukowsky型线输出形状正确 ----
contour = geo.apply_joukowsky_nozzle_contour()
assert contour.shape[0] == 200, '[TC35] contour points count wrong FAILED'
assert contour.shape[1] == 2, '[TC35] contour dimension wrong FAILED'

# ---- TC36: InjectorLayoutOptimizer 候选位置生成 ----
import numpy as np
np.random.seed(42)
opt = InjectorLayoutOptimizer(panel_radius=0.12, element_outer_diameter=8e-3,
                               element_mass=0.015, target_total_flow=120.0)
n_cand = opt.generate_candidate_positions_triangular(n_layers=4)
assert n_cand > 0, '[TC36] no candidates generated FAILED'

# ---- TC37: InjectorLayoutOptimizer 贪心求解选到单元 ----
result_g = opt.solve_greedy_heuristic()
assert result_g['n_selected'] > 0, '[TC37] greedy solver selected nothing FAILED'

# ---- TC38: InjectorLayoutOptimizer 氧燃比分布统计 ----
mr_dist = opt.compute_mixture_ratio_distribution(result_g['selected_indices'])
assert mr_dist['mean'] > 0, '[TC38] mixture ratio mean <= 0 FAILED'
assert mr_dist['std'] >= 0, '[TC38] mixture ratio std < 0 FAILED'

# ---- TC39: SprayDistributionCVT 初始分布形状正确 ----
import numpy as np
np.random.seed(42)
spray = SprayDistributionCVT(n_droplets=100)
pos = spray.generate_initial_distribution()
assert pos.shape == (100, 3), '[TC39] initial distribution shape wrong FAILED'
assert np.all(np.isfinite(pos)), '[TC39] initial positions not finite FAILED'

# ---- TC40: SprayDistributionCVT 优化收敛 ----
import numpy as np
np.random.seed(42)
result_cvt = spray.optimize_distribution(n_iterations=20, n_samples=3000, tolerance=1e-4)
assert result_cvt['iterations'] > 0, '[TC40] CVT did zero iterations FAILED'
assert np.isfinite(result_cvt['final_energy']), '[TC40] CVT energy not finite FAILED'

# ---- TC41: SprayDistributionCVT 喷雾统计 ----
stats = spray.compute_spray_statistics()
assert stats['n_droplets'] == 100, '[TC41] droplet count wrong FAILED'
assert stats['sauter_mean_diameter'] > 0, '[TC41] SMD not positive FAILED'

# ---- TC42: AcousticModeAnalyzer 纵向模态频率输出 ----
analyzer = AcousticModeAnalyzer(chamber_length=0.60, chamber_radius=0.15, sound_speed=1200.0)
L_modes = analyzer.longitudinal_modes()
assert len(L_modes['modes']) == 5, '[TC42] wrong number of longitudinal modes FAILED'
assert L_modes['modes'][0]['frequency'] > 0, '[TC42] frequency not positive FAILED'

# ---- TC43: AcousticModeAnalyzer 径向模态频率输出 ----
R_modes = analyzer.radial_modes()
assert len(R_modes['modes']) > 0, '[TC43] no radial modes FAILED'
assert R_modes['modes'][0]['frequency'] > 0, '[TC43] radial frequency not positive FAILED'

# ---- TC44: AcousticModeAnalyzer Rayleigh准则返回标量 ----
import numpy as np
z = np.linspace(0, 0.60, 100)
p_mode = np.cos(np.pi * z / (2 * 0.60))
q_osc = np.exp(-10 * (z - 0.60 * 0.3) ** 2)
rayleigh = analyzer.rayleigh_criterion(q_osc, p_mode)
assert np.isfinite(rayleigh), '[TC44] Rayleigh criterion not finite FAILED'

# ---- TC45: AcousticModeAnalyzer 模态正交性对角占优 ----
ortho = analyzer.compute_orthogonality_integrals("L")
diag = np.diag(ortho)
offdiag_max = np.max(np.abs(ortho - np.diag(diag)))
assert diag[0] > 0, '[TC45] diagonal not positive FAILED'
assert offdiag_max < 1e-6, '[TC45] orthogonality broken FAILED'

# ---- TC46: FEMBasis2DTriangle 基函数在自身节点为 1 ----
fem2d = FEMBasis2DTriangle(degree=2)
L0 = fem2d.evaluate_basis(0, 0.0, 0.0)
assert abs(L0 - 1.0) < 1e-10, '[TC46] basis function not 1 at own node FAILED'

# ---- TC47: FEMBasis2DTriangle 基函数在另一节点为 0 ----
L_other = fem2d.evaluate_basis(0, 1.0, 0.0)
assert abs(L_other) < 1e-10, '[TC47] basis function not 0 at other node FAILED'

# ---- TC48: ChebyshevNDInterpolation 2D 求值有限 ----
import numpy as np
coeffs2d = np.array([[1.0, 0.3, -0.1], [0.2, -0.05, 0.02], [-0.05, 0.01, 0.005]])
cheb = ChebyshevNDInterpolation(coeffs2d, domains=[(0, 1), (0, 1)])
val = cheb.evaluate(np.array([0.5, 0.5]))
assert np.isfinite(val), '[TC48] Chebyshev result not finite FAILED'

# ---- TC49: LebesgueStabilityAnalyzer Chebyshev节点优于等距节点 ----
leb_eq = LebesgueStabilityAnalyzer(np.linspace(-1, 1, 10))
lambda_eq = leb_eq.lebesgue_constant()
cheb_nodes = leb_eq.chebyshev_nodes(10, -1, 1)
leb_cheb = LebesgueStabilityAnalyzer(cheb_nodes)
lambda_cheb = leb_cheb.lebesgue_constant()
assert lambda_cheb < lambda_eq, '[TC49] Chebyshev not better than equidistant FAILED'

# ---- TC50: FlameTransferFunction 解析FTF @ 0Hz = interaction_index ----
ftf = FlameTransferFunction(interaction_index=1.2, time_delay_ms=2.0, cutoff_frequency_hz=1000.0)
F0 = ftf.analytical_ftf(0.0)
assert abs(abs(F0) - 1.2) < 1e-10, '[TC50] FTF magnitude at 0 Hz != 1.2 FAILED'

# ---- TC51: FlameTransferFunction 离散数据生成 ----
data = ftf.generate_discrete_data(n_points=30)
assert len(data['frequencies']) == 30, '[TC51] wrong number of discrete points FAILED'
assert np.all(np.isfinite(data['ftf_magnitude'])), '[TC51] FTF magnitude not finite FAILED'

# ---- TC52: FlameTransferFunction Nyquist稳定性裕度有限 ----
stability = ftf.compute_nyquist_stability_margin()
assert 'gain_margin_db' in stability, '[TC52] gain_margin_db missing FAILED'
assert 'phase_margin_deg' in stability, '[TC52] phase_margin_deg missing FAILED'

# ---- TC53: ThermoacousticOscillator limit_cycle_amplitude 非负有限 ----
osc = ThermoacousticOscillator(natural_frequency_hz=500.0, acoustic_damping=80.0,
                               flame_gain_coefficient=120.0, nonlinear_saturation=5e7)
A_lim = osc.limit_cycle_amplitude()
assert A_lim >= 0.0, '[TC53] limit cycle amplitude negative FAILED'
assert np.isfinite(A_lim), '[TC53] limit cycle amplitude not finite FAILED'

# ---- TC54: ThermoacousticOscillator 线性稳定性判断 ----
assert isinstance(osc.is_unstable, bool), '[TC54] is_unstable not bool FAILED'

# ---- TC55: ThermoacousticOscillator RK4 积分输出正确 ----
import numpy as np
np.random.seed(42)
result_rk4 = osc.rk4_integrate((0, 0.02), n_steps=5000)
assert len(result_rk4['pressure']) == 5001, '[TC55] RK4 pressure length wrong FAILED'
assert np.all(np.isfinite(result_rk4['pressure'])), '[TC55] RK4 pressure not finite FAILED'
assert result_rk4['t'][0] == 0.0, '[TC55] RK4 start time wrong FAILED'

# ---- TC56: ThermoacousticOscillator compute_oscillation_metrics ----
import numpy as np
np.random.seed(42)
t_test = np.linspace(0, 0.05, 1000)
p_test = 100.0 * np.sin(2 * np.pi * 500.0 * t_test)
metrics = osc.compute_oscillation_metrics(t_test, p_test)
assert metrics['peak_to_peak_pa'] > 0, '[TC56] peak-to-peak not positive FAILED'
assert metrics['rms_pa'] > 0, '[TC56] rms not positive FAILED'
assert metrics['estimated_frequency_hz'] > 0, '[TC56] frequency not positive FAILED'

# ---- TC57: MultiModeThermoacousticSystem 积分输出正确 ----
import numpy as np
freqs = np.array([500.0, 1500.0, 2500.0])
damping = np.array([80.0, 200.0, 350.0])
multi = MultiModeThermoacousticSystem(freqs, damping)
multi_result = multi.integrate(t_span=(0, 0.02), n_steps=5000)
assert len(multi_result['mode_pressures']) == 3, '[TC57] wrong number of mode pressures FAILED'
assert np.all(np.isfinite(multi_result['mode_pressures'][0])), '[TC57] mode 0 not finite FAILED'

# ---- TC58: ReactionDiffusionSolver Jacobi收敛且结果有限 ----
import numpy as np
rd = ReactionDiffusionSolver(domain_length=0.015, n_points=101,
                              activation_energy=1.26e5, temperature_burned=3600.0)
result_rd = rd.solve_steady_jacobi(max_iterations=10000, tolerance=1e-6)
assert result_rd['iterations'] < 10000, '[TC58] Jacobi did not converge FAILED'
assert np.isfinite(result_rd['flame_position']), '[TC58] flame position not finite FAILED'
assert np.isfinite(result_rd['flame_thickness']), '[TC58] flame thickness not finite FAILED'

# ---- TC59: ReactionDiffusionSolver Zeldovich数正值 ----
assert rd.beta > 0, '[TC59] Zeldovich number not positive FAILED'

# ---- TC60: FEMHelmholtzSolver 本征频率单调递增 ----
import numpy as np
fem = FEMHelmholtzSolver(length=0.60, n_elements=50)
fem_result = fem.solve_eigenvalue(n_modes=3)
assert len(fem_result['frequencies']) == 3, '[TC60] FEM eigenvalue count wrong FAILED'
assert fem_result['frequencies'][0] > 0, '[TC60] FEM frequency 0 not positive FAILED'
for i in range(2):
    assert fem_result['frequencies'][i] < fem_result['frequencies'][i+1], \
        f'[TC60] FEM frequencies not monotonic at {i} FAILED'

print('\n全部 60 个测试通过!\n')

# main.py 原有函数/类定义，原样保留
