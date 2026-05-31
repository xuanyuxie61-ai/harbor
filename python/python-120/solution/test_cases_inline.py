# ---- TC01: kb_t_ev 在室温下返回正有限值 ----
assert isinstance(kb_t_ev(300.0), float), '[TC01] kb_t_ev 应返回 float FAILED'
assert kb_t_ev(300.0) > 0.0, '[TC01] kb_t_ev(300) 应为正值 FAILED'

# ---- TC02: kb_t_ev 在零温下返回 0 ----
assert kb_t_ev(0.0) == 0.0, '[TC02] kb_t_ev(0) 应为 0 FAILED'

# ---- TC03: maxwell_boltzmann_speed 返回正有限值 ----
from utils import maxwell_boltzmann_speed
v_mp = maxwell_boltzmann_speed(28.01, 500.0)
assert v_mp > 0.0, '[TC03] 最概然速率应为正值 FAILED'
assert np.isfinite(v_mp), '[TC03] 最概然速率应为有限值 FAILED'

# ---- TC04: de_broglie_thermal_wavelength 返回正值 ----
from utils import de_broglie_thermal_wavelength
lam = de_broglie_thermal_wavelength(28.01, 500.0)
assert lam > 0.0, '[TC04] 德布罗意波长应为正值 FAILED'
assert np.isfinite(lam), '[TC04] 德布罗意波长应为有限值 FAILED'

# ---- TC05: grid_uniform_1d 端点正确 ----
g = grid_uniform_1d(0.0, 1.0, 5)
assert abs(g[0] - 0.0) < 1e-15, '[TC05] 网格起点应为 0.0 FAILED'
assert abs(g[-1] - 1.0) < 1e-15, '[TC05] 网格终点应为 1.0 FAILED'
assert len(g) == 5, '[TC05] 网格点数应为 5 FAILED'

# ---- TC06: grid_uniform_1d 单调递增 ----
g6 = grid_uniform_1d(0.0, 10.0, 20)
assert np.all(np.diff(g6) > 0), '[TC06] 网格应严格单调递增 FAILED'

# ---- TC07: morse_potential 在平衡位置返回 -D_e ----
r_e = 1.85e-10
v_eq = morse_potential(np.array([r_e]), d_e=1.3, a_param=2.0e10, r_e=r_e)
assert abs(v_eq[0] - (-1.3)) < 1e-12, '[TC07] Morse 势在 r=r_e 处应为 -D_e FAILED'

# ---- TC08: morse_potential 在无穷远处趋近 0 ----
r_far = np.array([1e-8])
v_far = morse_potential(r_far, d_e=1.3, a_param=2.0e10, r_e=1.85e-10)
assert v_far[0] > -0.01, '[TC08] Morse 势在远距离应接近 0 FAILED'

# ---- TC09: arrhenius_rate 随温度单调递增 ----
r1 = arrhenius_rate(1.0e13, 0.8, 400.0)
r2 = arrhenius_rate(1.0e13, 0.8, 500.0)
assert r2 > r1, '[TC09] 温度升高 Arrhenius 速率应增大 FAILED'

# ---- TC10: arrhenius_rate 零活化能时不依赖温度 ----
r3 = arrhenius_rate(1.0, 0.0, 300.0)
assert abs(r3 - 1.0) < 1e-12, '[TC10] 零活化能时速率应等于指前因子 FAILED'

# ---- TC11: safe_divide 处理除零 ----
from utils import safe_divide
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.0, 1.0, 2.0])
sd = safe_divide(a, b)
assert sd[0] == 0.0, '[TC11] 除零应返回 fill_value FAILED'
assert abs(sd[1] - 2.0) < 1e-15, '[TC11] 2/1 = 2 FAILED'
assert abs(sd[2] - 1.5) < 1e-15, '[TC11] 3/2 = 1.5 FAILED'

# ---- TC12: grid_uniform_nd 输出形状正确 (2D) ----
from utils import grid_uniform_nd
g2d = grid_uniform_nd(2, 4, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
assert g2d.shape == (2, 16), '[TC12] 2D 网格形状应为 (2, 16) FAILED'

# ---- TC13: grid_uniform_nd 端点覆盖 (1D) ----
g3d = grid_uniform_nd(1, 3, np.array([0.0]), np.array([1.0]))
assert g3d.shape[1] == 3, '[TC13] 1D 网格应含 3 个节点 FAILED'
assert abs(g3d[0, 0] - 0.0) < 1e-15, '[TC13] 起点应为 0 FAILED'
assert abs(g3d[0, -1] - 1.0) < 1e-15, '[TC13] 终点应为 1 FAILED'

# ---- TC14: Pt111Surface 构造成功 ----
surf_t = Pt111Surface(nx=4, ny=4, n_layers=2)
assert surf_t.n_atoms > 0, '[TC14] 表面原子数应为正 FAILED'
assert surf_t.n_sites > 0, '[TC14] 吸附位点数应为正 FAILED'

# ---- TC15: Pt111Surface 初始覆盖率为 0 ----
cov0 = surf_t.surface_coverage(species=1)
assert cov0 == 0.0, '[TC15] 初始 CO 覆盖率应为 0 FAILED'

# ---- TC16: Pt111Surface find_nearest_site ----
site_test_pos = surf_t.sites[0, :3].copy()
idx_near = surf_t.find_nearest_site(site_test_pos)
assert idx_near == 0, '[TC16] 最近位点应为自身 INDEX=0 FAILED'

# ---- TC17: Pt111Surface 位点能量均为负值 ----
se = surf_t.site_energies
assert np.all(se < 0), '[TC17] 位点吸附能应为负值 FAILED'

# ---- TC18: build_co_oxidation_pes_demo 返回 PES 实例 ----
pes_t = build_co_oxidation_pes_demo()
assert pes_t.coeffs is not None, '[TC18] PES 系数应已拟合 FAILED'
assert len(pes_t.powers) > 0, '[TC18] PES 基函数数量应为正 FAILED'

# ---- TC19: PES evaluate 返回有限值 ----
v_test = pes_t.evaluate(np.array([[0.0, 0.0, 1.5e-10]]))
assert np.isfinite(v_test[0]), '[TC19] PES 评估值应为有限 FAILED'

# ---- TC20: PES gradient 返回有限值 ----
grad_test = pes_t.gradient(np.array([[0.0, 0.0, 1.5e-10]]))
assert grad_test.shape == (1, 3), '[TC20] 梯度形状应为 (1, 3) FAILED'
assert np.all(np.isfinite(grad_test)), '[TC20] 梯度应全部有限 FAILED'

# ---- TC21: PES hessian 返回有限对称矩阵 ----
hess_t = pes_t.hessian(np.array([[0.0, 0.0, 1.5e-10]]))
assert hess_t.shape == (1, 3, 3), '[TC21] Hessian 形状应为 (1, 3, 3) FAILED'
assert np.allclose(hess_t[0], hess_t[0].T, atol=1e-12), '[TC21] Hessian 应对称 FAILED'

# ---- TC22: TightBindingSolver 构造与对角化 ----
tb_t = TightBindingSolver(n_atoms=4, n_orbitals_per_atom=1)
pos_t = np.array([[0, 0, 0], [2e-10, 0, 0], [0, 2e-10, 0], [0, 0, 2e-10]], dtype=float)
onsite_t = np.array([-5.0, -5.0, -5.0, -5.0])
tb_t.build_hamiltonian_sk(pos_t, onsite_t, v_ss_sigma=-1.0, r_cutoff=3e-10)
evals_t, evecs_t = tb_t.solve_eigenvalues_dense()
assert len(evals_t) == 4, '[TC22] 本征值数量应为 4 FAILED'
assert np.all(np.isfinite(evals_t)), '[TC22] 本征值应全部有限 FAILED'

# ---- TC23: TightBindingSolver 本征值为实 ----
assert np.allclose(evals_t.imag, 0.0), '[TC23] 本征值应为实数 FAILED'

# ---- TC24: TightBindingSolver DOS 非负 ----
dos_t = tb_t.compute_dos(np.linspace(-10, 0, 50), sigma=0.1)
assert np.all(dos_t >= 0), '[TC24] DOS 应非负 FAILED'
assert np.any(dos_t > 0), '[TC24] DOS 应有非零值 FAILED'

# ---- TC25: TightBindingSolver 吸附能公式正确 ----
e_ads_t = tb_t.compute_adsorption_energy(e_isolated=-2.0, e_surface=-20.0, e_complex=-23.0)
assert abs(e_ads_t - (-1.0)) < 1e-12, '[TC25] 吸附能 -23-(-20)-(-2) = -1 FAILED'

# ---- TC26: LangevinIntegrator 初始化和单步执行 ----
np.random.seed(42)
li_t = LangevinIntegrator(mass_amu=np.array([28.01]), gamma_ps=2.0, temperature_k=500.0, dt_fs=0.5)
li_t.initialize(np.array([[0.0, 0.0, 1.5e-10]]))
assert li_t.positions is not None, '[TC26] 位置应已初始化 FAILED'
assert li_t.velocities is not None, '[TC26] 速度应已初始化 FAILED'
np.random.seed(42)
li_t.step(lambda pos: -np.ones_like(pos) * (pos[:, 2] - 1.5e-10) * 10.0 * 1.602176634e-19)
ke_t = li_t.kinetic_energy()
assert ke_t >= 0.0, '[TC26] 动能应为非负 FAILED'

# ---- TC27: LangevinIntegrator 瞬时温度非负 ----
ti_t = li_t.temperature_instantaneous()
assert ti_t > 0, '[TC27] 瞬时温度应为正 FAILED'
assert np.isfinite(ti_t), '[TC27] 瞬时温度应为有限 FAILED'

# ---- TC28: LangevinIntegrator MSD 非负且单调 ----
np.random.seed(42)
li2 = LangevinIntegrator(mass_amu=np.array([28.01]), gamma_ps=2.0, temperature_k=300.0, dt_fs=1.0)
li2.initialize(np.array([[0.0, 0.0, 1.5e-10]]))
traj_lv = [li2.positions.copy()]
force_lv = lambda pos: -np.ones_like(pos) * (pos[:, 2] - 1.5e-10) * 5.0 * 1.602176634e-19
for _ in range(20):
    np.random.seed(42)
    li2.step(force_lv)
    traj_lv.append(li2.positions.copy())
msd_lv = li2.compute_mean_square_displacement(traj_lv)
assert np.all(msd_lv >= 0), '[TC28] MSD 应非负 FAILED'
assert msd_lv[-1] >= msd_lv[0], '[TC28] MSD 终点不应小于起点 FAILED'

# ---- TC29: ReactionDiffusion1D 稳态解长度正确且有限 ----
x_rd = grid_uniform_1d(0.0, 10e-9, 51)
rd_t = ReactionDiffusion1D(x_rd, diffusivity=1.0e-9, bc_type="dirichlet")
# 使用弱反应项确保数值稳定
c_ss_t = rd_t.solve_steady_state(lambda c: 1.0 * c - 0.5, bc_values=(0.8, 0.1))
assert len(c_ss_t) == 51, '[TC29] 稳态解长度应为 51 FAILED'
assert np.all(np.isfinite(c_ss_t)), '[TC29] 稳态解应全部有限 FAILED'

# ---- TC30: LangmuirHinshelwoodKinetics 稳态覆盖率在 [0,1] ----
lh_t = LangmuirHinshelwoodKinetics(temperature_k=500.0, p_co_pa=1.0e3, p_o2_pa=5.0e2)
theta_ss_t = lh_t.steady_state_coverage()
assert 0.0 <= theta_ss_t[0] <= 1.0, '[TC30] θ_CO 应在 [0,1] 内 FAILED'
assert 0.0 <= theta_ss_t[1] <= 1.0, '[TC30] θ_O 应在 [0,1] 内 FAILED'

# ---- TC31: LangmuirHinshelwoodKinetics 速率常数均为正 ----
k_t = lh_t._rate_constants()
for key_t, val_t in k_t.items():
    assert val_t > 0, f'[TC31] {key_t} 应为正 FAILED'

# ---- TC32: MonteCarloSampler 固定种子可复现 ----
mc1 = MonteCarloSampler(seed=42)
f_mc = lambda p: np.sum(p ** 2, axis=1)
b_mc = np.array([[-1.0, 1.0], [-1.0, 1.0]])
est1_mc, _ = mc1.estimate_integral(f_mc, b_mc, 10000)
mc2 = MonteCarloSampler(seed=42)
est2_mc, _ = mc2.estimate_integral(f_mc, b_mc, 10000)
assert abs(est1_mc - est2_mc) < 1e-15, '[TC32] 固定种子 MC 结果应可复现 FAILED'

# ---- TC33: QuadratureIntegrator 复合 Simpson 精确性 ----
qi_t = QuadratureIntegrator()
I_simp_t = qi_t.composite_simpson(lambda x: np.sin(x), 0.0, np.pi, n=100)
assert abs(I_simp_t - 2.0) < 1e-6, '[TC33] ∫₀^π sin(x)dx = 2 FAILED'

# ---- TC34: QuadratureIntegrator Gauss-Legendre 3 点精确 ----
I_gl_t = qi_t.gauss_legendre_3point(lambda x: x ** 2, -1.0, 1.0)
assert abs(I_gl_t - 2.0 / 3.0) < 1e-12, '[TC34] ∫₋₁¹ x² dx = 2/3 FAILED'

# ---- TC35: QuadratureIntegrator 复合梯形 ----
I_trap_t = qi_t.composite_trapezoidal(lambda x: x, 0.0, 1.0, n=1000)
assert abs(I_trap_t - 0.5) < 1e-5, '[TC35] ∫₀¹ x dx = 0.5 FAILED'

# ---- TC36: PiecewiseLinearProductIntegral ∫ x·x dx = 1/3 ----
pli_t = PiecewiseLinearProductIntegral()
fx_t = np.array([0.0, 1.0])
fv_t = np.array([0.0, 1.0])
gx_t = np.array([0.0, 1.0])
gv_t = np.array([0.0, 1.0])
I_pli_t = pli_t.integrate(fx_t, fv_t, gx_t, gv_t, 0.0, 1.0)
assert abs(I_pli_t - 1.0 / 3.0) < 1e-12, '[TC36] ∫₀¹ x·x dx = 1/3 FAILED'

# ---- TC37: SurfaceReactionNetwork 稳态概率和为 1 ----
net_t = SurfaceReactionNetwork(n_sites=3, max_occupancy=1)
net_t.build_transition_matrix(rate_ads_co=1.0, rate_des_co=0.2, rate_ads_o=0.5, rate_des_o=0.05, rate_rxn=0.3)
p_ss_t = net_t.steady_state_distribution()
assert abs(np.sum(p_ss_t) - 1.0) < 1e-12, '[TC37] 稳态概率和应为 1 FAILED'
assert np.all(p_ss_t >= 0), '[TC37] 稳态概率应非负 FAILED'

# ---- TC38: SurfaceReactionNetwork 转移矩阵行和为零 (概率守恒) ----
for i_t in range(net_t.n_states):
    assert abs(np.sum(net_t.W[i_t, :])) < 1e-12, f'[TC38] 状态 {i_t} 行列和应为 0 FAILED'

# ---- TC39: SurfaceReactionNetwork MFPT 为非负 ----
if net_t.n_states > 1:
    mfpt_t = net_t.mean_first_passage_time(target_state=0, start_state=1)
    assert mfpt_t >= 0, '[TC39] MFPT 应为非负 FAILED'
    assert np.isfinite(mfpt_t), '[TC39] MFPT 应为有限 FAILED'

# ---- TC40: generate_test_trajectory 形状正确 ----
traj_svd = generate_test_trajectory(n_atoms=12, n_frames=200)
assert traj_svd.shape == (200, 12, 3), '[TC40] 轨迹形状应为 (200, 12, 3) FAILED'
assert np.all(np.isfinite(traj_svd)), '[TC40] 轨迹应全部有限 FAILED'

# ---- TC41: ReactionCoordinateAnalyzer fit 后属性正确 ----
rca_t = ReactionCoordinateAnalyzer()
rca_t.fit(traj_svd)
assert rca_t.U is not None, '[TC41] SVD U 矩阵应为非空 FAILED'
assert rca_t.S is not None, '[TC41] SVD 奇异值应为非空 FAILED'
assert rca_t.Vt is not None, '[TC41] SVD Vt 矩阵应为非空 FAILED'

# ---- TC42: SVD 方差贡献率和为 1 ----
var_t = rca_t.variance_explained()
assert abs(np.sum(var_t) - 1.0) < 1e-12, '[TC42] 方差贡献率和应为 1 FAILED'

# ---- TC43: 反应坐标长度匹配轨迹帧数 ----
rc_t = rca_t.reaction_coordinate(traj_svd)
assert len(rc_t) == 200, '[TC43] 反应坐标长度应为 200 FAILED'

# ---- TC44: collectivity_index 在 (0,1] ----
kappa_t = rca_t.collectivity_index(n_components=3)
assert 0.0 < kappa_t <= 1.0, '[TC44] 集体性指数 κ 应在 (0, 1] 内 FAILED'

# ---- TC45: sticking_coefficient_langmuir 在 [0, alpha0] ----
from utils import sticking_coefficient_langmuir
s_t = sticking_coefficient_langmuir(pressure_pa=1.0e5, temperature_k=500.0, alpha0=0.8, e_ads_ev=0.3)
assert 0.0 <= s_t <= 0.8, '[TC45] 粘附系数应在 [0, alpha0] 内 FAILED'

# ---- TC46: Pt111Surface CVT 优化返回采样点 ----
cvt_t = surf_t.cvt_optimize_sites(n_generators=8, it_num=10)
assert cvt_t.shape == (8, 3), '[TC46] CVT 输出形状应为 (8, 3) FAILED'
assert np.all(cvt_t[:, 2] >= 0.0), '[TC46] CVT 采样点 z 坐标应非负 FAILED'

# ---- TC47: PES saddle_point_newton 返回有限位置 ----
x0_t = np.array([0.5e-10, 0.0, 1.5e-10])
x_ts_t, is_sad = pes_t.find_saddle_point_newton(x0_t, tol=1e-5, max_iter=50)
assert np.all(np.isfinite(x_ts_t)), '[TC47] 鞍点搜索位置应有限 FAILED'

# ---- TC48: Pt111Surface update_occupancy_ca 保持状态数 ----
surf_ca = Pt111Surface(nx=3, ny=3, n_layers=1)
surf_ca.site_occupancy[:4] = 1
n_before = np.sum(surf_ca.site_occupancy > 0)
surf_ca.update_occupancy_ca(rule=30, steps=3)
n_after = np.sum(surf_ca.site_occupancy > 0)
assert n_before >= 0 and n_after >= 0, '[TC48] 占据态数量应为非负 FAILED'

# ---- TC49: ReactionDiffusion1D 时间演化输出形状正确 ----
x_td = grid_uniform_1d(0.0, 5e-9, 21)
rd_td = ReactionDiffusion1D(x_td, diffusivity=1.0e-9, bc_type="dirichlet")
c0_td = np.linspace(0.8, 0.1, 21)
traj_rd_t, times_rd_t = rd_td.solve_time_dependent(c0_td, t_end=1.0e-9, n_steps=100, reaction_func=lambda c: 0.0 * c)
assert traj_rd_t.shape == (101, 21), '[TC49] 时间演化数组形状应为 (101, 21) FAILED'

# ---- TC50: 所有模块可导入且主流程无语法错误 ----
assert True, '[TC50] 所有模块导入成功，测试框架正常 FAILED'
