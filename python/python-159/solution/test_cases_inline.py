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
