# ---- TC01: setup_physical_parameters 返回字典且关键参数为正 ----
params = setup_physical_parameters()
assert isinstance(params, dict), '[TC01] params type FAILED'
assert params['T'] > 0, '[TC01] temperature FAILED'
assert params['F'] > 0, '[TC01] Faraday constant FAILED'

# ---- TC02: balance_orr_stoichiometry 返回正确键与整数系数 ----
stoich = balance_orr_stoichiometry()
assert stoich['o2'] == 1, '[TC02] O2 coefficient FAILED'
assert stoich['water'] == 2, '[TC02] water coefficient FAILED'
assert stoich['h_plus'] == 4, '[TC02] H+ coefficient FAILED'
assert stoich['electrons'] == 4, '[TC02] electrons coefficient FAILED'

# ---- TC03: verify_stoichiometry_solution 对正确解返回全零残差 ----
res = verify_stoichiometry_solution(stoich)
assert res['r_o'] == 0, '[TC03] O residual FAILED'
assert res['r_h'] == 0, '[TC03] H residual FAILED'
assert res['r_e'] == 0, '[TC03] e residual FAILED'

# ---- TC04: butler_volmer_kinetics 零过电位电流近似为零 ----
import numpy as np
eta_z = np.array([0.0])
j_z = butler_volmer_kinetics(eta_z, params)
assert abs(j_z[0]) < 1e-12, '[TC04] zero eta current FAILED'

# ---- TC05: butler_volmer_kinetics 输出形状与输入一致 ----
import numpy as np
eta_arr = np.linspace(-0.3, 0.3, 100)
j_arr = butler_volmer_kinetics(eta_arr, params)
assert j_arr.shape == eta_arr.shape, '[TC05] shape mismatch FAILED'

# ---- TC06: butler_volmer_kinetics 输出单调递增 ----
import numpy as np
j_diff = np.diff(j_arr)
assert np.all(j_diff >= -1e-12), '[TC06] not monotonic FAILED'

# ---- TC07: compute_exchange_current_density 返回正有限值 ----
j0 = compute_exchange_current_density(params)
assert j0 > 0, '[TC07] j0 positive FAILED'
assert np.isfinite(j0), '[TC07] j0 finite FAILED'

# ---- TC08: generate_pemfc_mesh 返回 8 个三维节点和四面体单元 ----
nodes, elements = generate_pemfc_mesh()
assert nodes.shape == (8, 3), '[TC08] nodes shape FAILED'
assert elements.shape[1] == 4, '[TC08] not tetrahedra FAILED'

# ---- TC09: refine_mesh 增加节点和单元数量 ----
nodes_r, elements_r = refine_mesh(nodes, elements)
assert nodes_r.shape[0] > nodes.shape[0], '[TC09] refined nodes not increased FAILED'
assert elements_r.shape[0] > elements.shape[0], '[TC09] refined elements not increased FAILED'

# ---- TC10: compute_mesh_quality 返回正体积 ----
quality = compute_mesh_quality(nodes_r, elements_r)
assert quality['min_volume'] > 0, '[TC10] min volume FAILED'
assert quality['mean_volume'] > 0, '[TC10] mean volume FAILED'

# ---- TC11: solve_proton_potential 输出二维数组且形状匹配 ----
phi_m, x_grid, y_grid = solve_proton_potential(params)
assert phi_m.ndim == 2, '[TC11] phi not 2D FAILED'
assert phi_m.shape == (len(x_grid), len(y_grid)), '[TC11] phi shape mismatch FAILED'

# ---- TC12: interpolate_proton_potential_hermite 返回有限值 ----
phi_q = interpolate_proton_potential_hermite(phi_m, x_grid, y_grid, 0.5, 0.5)
assert np.isfinite(phi_q), '[TC12] hermite interp FAILED'

# ---- TC13: water_content_wave_exact 满足波动方程 ----
import numpy as np
z_test = np.linspace(0.0, params['t_membrane'], 11)
u, ut, utt, uz, uzz = water_content_wave_exact(z_test, 1.0, params)
c_w = 1.0e-3
wave_res = np.max(np.abs(utt - c_w**2 * uzz))
assert wave_res < 1e-10, '[TC13] wave equation residual FAILED'

# ---- TC14: solve_membrane_water_transport 输出在物理范围内 ----
lam, t_grid = solve_membrane_water_transport(params)
assert lam.ndim == 1, '[TC14] lambda not 1D FAILED'
assert np.all(np.isfinite(lam)), '[TC14] lambda not finite FAILED'
assert np.all(lam >= 0.0), '[TC14] lambda negative FAILED'
assert np.all(lam <= 22.0), '[TC14] lambda exceeds max FAILED'

# ---- TC15: solve_gdl_saturation 饱和度在 [0,1] 范围内 ----
s_gdl, x_gdl = solve_gdl_saturation(params)
assert np.all(s_gdl >= 0.0), '[TC15] saturation negative FAILED'
assert np.all(s_gdl <= 1.0), '[TC15] saturation exceeds 1 FAILED'
assert s_gdl.ndim == 1, '[TC15] saturation not 1D FAILED'

# ---- TC16: 边界条件: GDL 流道侧低饱和度、催化层侧较高 ----
import numpy as np
assert abs(s_gdl[0] - 0.05) < 1e-12, '[TC16] GDL channel side BC FAILED'
assert abs(s_gdl[-1] - 0.6) < 1e-12, '[TC16] GDL catalyst side BC FAILED'

# ---- TC17: porous_medium_exact 在 t=0 返回全零 ----
import numpy as np
x_pm = np.array([-0.1, 0.0, 0.1])
u_pm, ut_pm, ux_pm, uxx_pm = porous_medium_exact(x_pm, 0.0, params['m_porous'], params)
assert np.all(u_pm == 0.0), '[TC17] t=0 not zero FAILED'

# ---- TC18: porous_medium_exact 在 t>0 返回非负有限值 ----
import numpy as np
u_pm2, ut_pm2, ux2, uxx2 = porous_medium_exact(x_pm, 0.5, params['m_porous'], params)
assert np.all(u_pm2 >= 0.0), '[TC18] solution negative FAILED'
assert np.all(np.isfinite(u_pm2)), '[TC18] solution not finite FAILED'

# ---- TC19: estimate_effective_diffusivity_monte_carlo 返回正值 ----
D_eff = estimate_effective_diffusivity_monte_carlo(nodes_r, elements_r, params, n_samples_per_tet=30)
assert D_eff > 0, '[TC19] D_eff not positive FAILED'
assert np.isfinite(D_eff), '[TC19] D_eff not finite FAILED'

# ---- TC20: optimize_sensor_placement 返回正确形状和范围 ----
np.random.seed(42)
sensors = optimize_sensor_placement(params, n_iter=20)
assert sensors.shape[0] == params['N_sensors'], '[TC20] sensor count FAILED'
assert sensors.shape[1] == 2, '[TC20] sensor dim FAILED'
assert np.all(sensors >= 0.0), '[TC20] sensor negative FAILED'

# ---- TC21: solve_banded_linear_system 残差足够小且耗时为正 ----
resid, t_solve = solve_banded_linear_system(params)
assert resid < 1e-6, '[TC21] residual too large FAILED'
assert t_solve > 0, '[TC21] solve time non-positive FAILED'

# ---- TC22: hankel_covariance_factor 输出方阵 ----
R_cov = hankel_covariance_factor(lam)
assert R_cov.shape[0] > 0, '[TC22] cov factor empty FAILED'
assert R_cov.shape[0] == R_cov.shape[1], '[TC22] cov factor not square FAILED'

# ---- TC23: 协方差矩阵为正定 ----
import numpy as np
cov = R_cov @ R_cov.T
eigvals = np.linalg.eigvalsh(cov)
assert np.all(eigvals > 0), '[TC23] covariance not SPD FAILED'

# ---- TC24: generate_polarization_curve 电压单调递减 ----
V_cell, I_cell = generate_polarization_curve(params)
assert V_cell.shape == I_cell.shape, '[TC24] polarization shape FAILED'
assert V_cell[0] > V_cell[-1], '[TC24] voltage not decreasing FAILED'
assert np.all(np.isfinite(V_cell)), '[TC24] voltage not finite FAILED'

# ---- TC25: generate_impedance_spectrum 输出维度正确且实部为正 ----
freq, Z_real, Z_imag = generate_impedance_spectrum(params)
assert freq.shape == Z_real.shape, '[TC25] freq-real shape FAILED'
assert freq.shape == Z_imag.shape, '[TC25] freq-imag shape FAILED'
assert np.all(Z_real > 0), '[TC25] real impedance non-positive FAILED'

# ---- TC26: compute_residuals 返回正确键和有限值 ----
residuals = compute_residuals(phi_m, lam, s_gdl, params)
assert set(residuals.keys()) == {'proton', 'water', 'porous'}, '[TC26] residual keys FAILED'
assert np.isfinite(residuals['proton']), '[TC26] proton residual FAILED'
assert np.isfinite(residuals['water']), '[TC26] water residual FAILED'

# ---- TC27: compute_mass_balance_error 返回非负误差 ----
import numpy as np
j_profile = np.linspace(5000.0, 10000.0, len(lam))
mb_err, lambda_total = compute_mass_balance_error(lam, j_profile, params)
assert mb_err >= 0, '[TC27] mass balance negative FAILED'
assert lambda_total > 0, '[TC27] total lambda non-positive FAILED'

# ---- TC28: 可复现性: 固定种子两次调用 optimize_sensor_placement 结果一致 ----
import numpy as np
np.random.seed(42)
s1 = optimize_sensor_placement(params, n_iter=10)
np.random.seed(42)
s2 = optimize_sensor_placement(params, n_iter=10)
assert np.allclose(s1, s2), '[TC28] reproducibility FAILED'

# ---- TC29: capillary_diffusivity 正值 ----
from porous_gdl_transport import capillary_diffusivity
import numpy as np
s_test = np.linspace(0.1, 0.9, 20)
D_cap = capillary_diffusivity(s_test, {'epsilon_gdl': params['epsilon_gdl']})
assert np.all(D_cap > 0), '[TC29] capillary diffusivity non-positive FAILED'
assert np.all(np.isfinite(D_cap)), '[TC29] capillary diffusivity not finite FAILED'

# ---- TC30: 集成测试: 所有模块协调运行且输出核心量有限 ----
import numpy as np
p_int = setup_physical_parameters()
p_int['Nx'] = 11
p_int['Nt'] = 10
p_int['t_final'] = 0.1
stoich_int = balance_orr_stoichiometry()
nodes_int, elements_int = generate_pemfc_mesh()
nodes_r_int, elements_r_int = refine_mesh(nodes_int, elements_int)
eta_int = np.linspace(-0.3, 0.3, 10)
j_bv_int = butler_volmer_kinetics(eta_int, p_int)
phi_int, xg_int, yg_int = solve_proton_potential(p_int)
lam_int, tg_int = solve_membrane_water_transport(p_int)
s_int, zg_int = solve_gdl_saturation(p_int)
D_int = estimate_effective_diffusivity_monte_carlo(nodes_r_int, elements_r_int, p_int, n_samples_per_tet=5)
sensors_int = optimize_sensor_placement(p_int, n_iter=5)
resid_int, _ = solve_banded_linear_system(p_int)
V_int, I_int = generate_polarization_curve(p_int)
assert phi_int.ndim == 2, '[TC30] integration phi FAILED'
assert lam_int.ndim == 1, '[TC30] integration lambda FAILED'
assert s_int.ndim == 1, '[TC30] integration saturation FAILED'
assert np.isfinite(resid_int), '[TC30] integration residual FAILED'
assert np.isfinite(D_int), '[TC30] integration D_eff FAILED'
