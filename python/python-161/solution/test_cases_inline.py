# ---- TC01: photon_energy_ev 解析验证 (hc/λ) ----
E1 = photon_energy_ev(np.array([620.0]))
assert abs(E1[0] - 1239.8 / 620.0) < 0.01, '[TC01] photon_energy_ev 解析 FAILED'

# ---- TC02: wedge01_volume 已知尺寸验证 ----
V = wedge01_volume(1e-4, 5e-5)
assert abs(V - 1e-4**2 * 5e-5 / 2.0) < 1e-16, '[TC02] wedge01_volume FAILED'

# ---- TC03: Gauss-Laguerre 权重和 ≈ 1 ----
_, wg = laguerre_quadrature_rule(8)
assert abs(wg.sum() - 1.0) < 1e-6, '[TC03] Laguerre 权重和 FAILED'

# ---- TC04: 光谱采样固定种子可复现 ----
np.random.seed(42)
lams1, thetas1 = sample_photons(n_photons=500)
np.random.seed(42)
lams2, thetas2 = sample_photons(n_photons=500)
assert np.allclose(lams1, lams2), '[TC04] 光谱采样可复现性 FAILED'

# ---- TC05: 材料参数 T=300K, x=0 带隙验证 ----
mat = PerovskiteMaterial()
p = mat.get_params(300.0, 0.0)
assert abs(p['bandgap_eV'] - 1.41) < 0.05, '[TC05] 带隙参数 FAILED'

# ---- TC06: 稀疏矩阵基本操作 ----
from sparse_matrix_io import SparseMatrix
sm = SparseMatrix(3, 3)
sm.add(0, 0, 1.0); sm.add(1, 1, 2.0); sm.add(2, 2, 3.0)
assert sm.nnz() == 3, '[TC06] 稀疏矩阵非零元数 FAILED'
dense = sm.to_dense()
assert abs(dense[0, 0] - 1.0) < 1e-12, '[TC06] 稀疏矩阵稠密值 FAILED'

# ---- TC07: Humps ODE 求解器验证返回有限值 ----
err = verify_solver()
assert np.isfinite(err), '[TC07] verify_solver 返回值 FAILED'
assert err >= 0, '[TC07] verify_solver 误差非负 FAILED'

# ---- TC08: 网格生成与面积验证 ----
np.random.seed(42)
mesh = generate_grain_mesh(4, 4)
areas = mesh.compute_areas()
assert mesh.vertices.shape[0] > 0, '[TC08] 网格顶点数 FAILED'
assert mesh.faces.shape[0] > 0, '[TC08] 网格面数 FAILED'
assert np.all(areas > 0), '[TC08] 三角形面积非正 FAILED'
assert np.all(np.isfinite(areas)), '[TC08] 面积非有限 FAILED'

# ---- TC09: 缺陷采样固定种子可复现 ----
np.random.seed(42)
v1 = mesh.vertices[mesh.faces[:, 0]]
v2 = mesh.vertices[mesh.faces[:, 1]]
v3 = mesh.vertices[mesh.faces[:, 2]]
defects1 = sample_defect_positions(200, (v1, v2, v3))
np.random.seed(42)
defects2 = sample_defect_positions(200, (v1, v2, v3))
assert np.allclose(defects1, defects2), '[TC09] 缺陷采样可复现性 FAILED'

# ---- TC10: 复合率非负且有限 ----
import numpy as np
np.random.seed(42)
rates = total_recombination_rate(n=1e15, p=1e15, n_i=1e10, T=300.0, tau_n=1e-8, tau_p=1e-8)
for k in ['SRH', 'radiative', 'auger', 'tail', 'total']:
    assert np.isfinite(rates[k]), f'[TC10] {k} 复合率非有限 FAILED'
    assert rates[k] >= 0, f'[TC10] {k} 复合率为负 FAILED'

# ---- TC11: 屈曲分析输出键完整性 ----
np.random.seed(42)
result = compute_buckling_impact_on_efficiency(delta_T=60.0)
expected_keys = ['thermal_stress_MPa', 'critical_stress_MPa', 'buckled', 'max_deflection_nm',
                  'bandgap_shift_meV', 'estimated_efficiency_loss_percent']
for k in expected_keys:
    assert k in result, f'[TC11] 屈曲结果缺键 {k} FAILED'
assert np.isfinite(result['thermal_stress_MPa']), '[TC11] 热应力非有限 FAILED'

# ---- TC12: PCE 不确定性量化输出 ----
np.random.seed(42)
uq = pce_efficiency_uq(efficiency_mean=0.21, efficiency_std=0.025, np_deg=5)
assert np.isfinite(uq['pce_mean_efficiency']), '[TC12] PCE 均值非有限 FAILED'
assert uq['pce_std_efficiency'] >= 0, '[TC12] PCE 标准差为负 FAILED'
assert 'sensitivity_indices' in uq, '[TC12] 缺敏感性指标 FAILED'

# ---- TC13: 离子动力学输出尺寸 ----
np.random.seed(42)
t, V_I, n_e = predprey_style_ion_dynamics(tspan=(0.0, 20.0), n_steps=500)
assert len(t) == 501, '[TC13] 离子动力学时间步数 FAILED'
assert V_I.shape == t.shape, '[TC13] V_I 形状 FAILED'
assert n_e.shape == t.shape, '[TC13] n_e 形状 FAILED'

# ---- TC14: SVD 模型降阶输出 ----
np.random.seed(42)
mor = apply_mor_to_drift_diffusion(n_spatial=30, n_time_snapshots=10, n_pod_modes=3)
assert mor['n_pod_modes'] == 3, '[TC14] POD 模态数 FAILED'
assert np.isfinite(mor['relative_reconstruction_error']), '[TC14] 重建误差非有限 FAILED'
assert mor['relative_reconstruction_error'] >= 0, '[TC14] 重建误差为负 FAILED'

# ---- TC15: compute_final_efficiency 输出类型与键 ----
buckling_test = {'estimated_efficiency_loss_percent': 5.0}
uq_test = {'pce_std_efficiency': 0.02}
eff = compute_final_efficiency(1e18, 1e21, 1e20, buckling_test, uq_test)
assert isinstance(eff, dict), '[TC15] 效率输出非字典 FAILED'
required_keys = ['short_circuit_current_mA_cm2', 'open_circuit_voltage_V',
                 'fill_factor', 'efficiency_no_stress', 'efficiency_with_stress']
for k in required_keys:
    assert k in eff, f'[TC15] 效率缺键 {k} FAILED'

# ---- TC16: 效率范围约束 0 ≤ η ≤ 0.35 ----
eff = compute_final_efficiency(1e18, 1e21, 1e20, buckling_test, uq_test)
assert 0 <= eff['efficiency_no_stress'] <= 0.35, '[TC16] 效率超出范围 FAILED'
assert 0 <= eff['efficiency_with_stress'] <= 0.35, '[TC16] 应力效率超出范围 FAILED'

# ---- TC17: 极端输入鲁棒性（零产生率） ----
eff_zero = compute_final_efficiency(1.0, 1.0, 1e30, {'estimated_efficiency_loss_percent': 0.0}, {'pce_std_efficiency': 0.0})
assert np.isfinite(eff_zero['efficiency_no_stress']), '[TC17] 零输入效率非有限 FAILED'
assert eff_zero['efficiency_no_stress'] >= 0, '[TC17] 零输入效率为负 FAILED'

# ---- TC18: buckling_lambda_mu 形状验证 ----
L_arr = np.linspace(0.3, 1.7, 5)
lam, mu = buckling_lambda_mu(L_arr, np.pi / 6)
assert lam.shape == L_arr.shape, '[TC18] lambda 形状 FAILED'
assert mu.shape == L_arr.shape, '[TC18] mu 形状 FAILED'

# ---- TC19: 低秩近似压缩比与能量占比 ----
np.random.seed(42)
A_test = np.random.randn(30, 20)
_, comp, energy = low_rank_approximation(A_test, 3)
assert comp > 0, '[TC19] 压缩比非正 FAILED'
assert 0 <= energy <= 1.0, '[TC19] 能量占比范围 FAILED'

# ---- TC20: SRH 复合率解析验证 ----
r_srh = srh_recombination_rate(n=1e15, p=1e15, n_i=1e10, tau_n=1e-8, tau_p=1e-8)
assert np.isfinite(r_srh), '[TC20] SRH 复合率非有限 FAILED'
assert r_srh >= 0, '[TC20] SRH 复合率为负 FAILED'

# ---- TC21: PCE 时间积分器输出尺寸 ----
np.random.seed(42)
t_arr, u_coeff = pce_time_integrator(0.0, 1.0, 20, 0.2, 3, 0.1, 0.05)
assert len(t_arr) == 21, '[TC21] PCE 积分器时间步数 FAILED'
assert u_coeff.ndim == 2, '[TC21] PCE 系数维度 FAILED'

# ---- TC22: I-V 迟滞循环输出尺寸 ----
np.random.seed(42)
V_sweep = np.linspace(0.0, 0.5, 10)
V, J, n_ion, E_ion = solve_hysteresis_cycle(V_sweep, time_per_step=1e-3)
assert len(V) == len(V_sweep), '[TC22] 迟滞电压长度 FAILED'
assert len(J) == len(V_sweep), '[TC22] 迟滞电流长度 FAILED'
assert np.all(np.isfinite(J)), '[TC22] 电流密度非有限 FAILED'

# ---- TC23: run_spectrum_and_absorption 返回有限值 ----
tg, ag = run_spectrum_and_absorption()
assert np.isfinite(tg), '[TC23] 总产生率非有限 FAILED'
assert np.isfinite(ag), '[TC23] 平均产生密度非有限 FAILED'
assert tg > 0, '[TC23] 总产生率非正 FAILED'

# ---- TC24: run_material_properties 返回正确类型 ----
mat2 = run_material_properties()
assert isinstance(mat2, PerovskiteMaterial), '[TC24] 材料对象类型 FAILED'
p2 = mat2.get_params(300.0, 0.0)
assert 'bandgap_eV' in p2, '[TC24] 材料缺带隙 FAILED'

# ---- TC25: run_mechanical_stress 返回字典含预期键 ----
buck = run_mechanical_stress()
assert isinstance(buck, dict), '[TC25] 屈曲结果非字典 FAILED'
assert 'estimated_efficiency_loss_percent' in buck, '[TC25] 缺效率损失 FAILED'

# ---- TC26: run_uncertainty_quantification 输出完整性 ----
uq2 = run_uncertainty_quantification()
assert uq2['pce_mean_efficiency'] > 0, '[TC26] PCE 均值非正 FAILED'
assert uq2['pce_std_efficiency'] >= 0, '[TC26] PCE 标准差为负 FAILED'

# ---- TC27: run_recombination 返回非负值 ----
R = run_recombination(1e-8, 1e-8)
assert np.isfinite(R), '[TC27] 总复合率非有限 FAILED'
assert R >= 0, '[TC27] 总复合率为负 FAILED'
