# ---- TC01: get_membrane_parameters 返回包含所有必要键的字典 ----
p = get_membrane_parameters()
required_keys = ["membrane_thickness", "fiber_inner_radius", "fiber_outer_radius",
                 "temperature", "Nx", "Nt", "pore_count", "stages"]
for k in required_keys:
    assert k in p, f'[TC01] Missing key: {k} FAILED'
assert p["membrane_thickness"] > 0, '[TC01] membrane_thickness must be positive FAILED'

# ---- TC02: get_feed_composition 摩尔分数和为 1 ----
feed_test = get_feed_composition()
total = feed_test["CO2"] + feed_test["CH4"] + feed_test["N2"]
assert abs(total - 1.0) < 1e-12, f'[TC02] Feed fractions sum = {total} FAILED'

# ---- TC03: compute_permeability 返回正值 ----
perm = compute_permeability(p)
assert perm["CO2"] > 0, '[TC03] CO2 permeability must be positive FAILED'
assert perm["CH4"] > 0, '[TC03] CH4 permeability must be positive FAILED'

# ---- TC04: compute_sorption_heat 返回负值（吸附放热） ----
sorp = compute_sorption_heat()
assert sorp["CO2"] < 0, '[TC04] CO2 sorption heat must be negative FAILED'
assert sorp["CH4"] < 0, '[TC04] CH4 sorption heat must be negative FAILED'

# ---- TC05: validate_parameters 对合法参数不抛出异常 ----
validate_parameters(p)

# ---- TC06: compute_dimensionless_numbers 返回有限正值 ----
dim = compute_dimensionless_numbers(p, species="CO2")
assert np.isfinite(dim["Damkohler"]), '[TC06] Damkohler must be finite FAILED'
assert dim["Damkohler"] > 0, '[TC06] Damkohler must be positive FAILED'
assert dim["Thiele_modulus"] > 0, '[TC06] Thiele_modulus must be positive FAILED'

# ---- TC07: safe_sqrt 正确处理正常输入 ----
import numpy as np
from utils import safe_sqrt
assert abs(safe_sqrt(4.0) - 2.0) < 1e-12, '[TC07] sqrt(4) = 2 FAILED'
assert abs(safe_sqrt(0.0) - 0.0) < 1e-12, '[TC07] sqrt(0) = 0 FAILED'

# ---- TC08: safe_divide 除零返回默认值 ----
from utils import safe_divide
assert safe_divide(1.0, 0.0) == 0.0, '[TC08] divide by zero returns default FAILED'
assert abs(safe_divide(6.0, 2.0) - 3.0) < 1e-12, '[TC08] 6/2 = 3 FAILED'

# ---- TC09: modular_wrap 正确回绕 ----
from utils import modular_wrap
assert modular_wrap(32, 1, 31) == 1, '[TC09] wrap(32,1,31) = 1 FAILED'
assert modular_wrap(15, 1, 31) == 15, '[TC09] wrap(15,1,31) = 15 FAILED'
assert modular_wrap(0, 1, 31) == 31, '[TC09] wrap(0,1,31) = 31 FAILED'

# ---- TC10: van_t_hoff_correction 高温下增大溶解度 ----
S_T = van_t_hoff_correction(1.0, -24000.0, 308.15)
assert S_T > 0, '[TC10] van_t_Hoff output must be positive FAILED'

# ---- TC11: linear_interpolate 节点处精确插值 ----
from utils import linear_interpolate, bracket_interval
x_nodes = np.array([0.0, 1.0, 2.0, 3.0])
y_vals = np.array([0.0, 2.0, 4.0, 6.0])
y_interp = linear_interpolate(x_nodes, y_vals, np.array([1.0]))
assert abs(y_interp[0] - 2.0) < 1e-12, '[TC11] linear interp at node FAILED'

# ---- TC12: ge_to_ccs 保持矩阵-向量乘积 ----
A_test = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 2.0], [1.0, 0.0, 5.0]])
ccs = ge_to_ccs(A_test)
A_dense = ccs.to_dense()
x_test = np.array([1.0, 2.0, 3.0])
assert np.allclose(A_dense.dot(x_test), A_test.dot(x_test), rtol=1e-12), '[TC12] CCS preserves mat-vec FAILED'

# ---- TC13: circulant_matrix_vector 结果与稠密矩阵等价 ----
n = 5
first_row = np.array([2.0, -1.0, 0.0, 0.0, -1.0])
x_vec = np.ones(n)
b_circ = circulant_matrix_vector(n, first_row, x_vec)
# 手动构建循环矩阵
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        C[i, j] = first_row[(j - i) % n]
b_dense = C.dot(x_vec)
assert np.allclose(b_circ, b_dense, rtol=1e-12), '[TC13] circulant = dense mat-vec FAILED'

# ---- TC14: solve_sparse_system 正确求解对角系统 ----
A_diag = np.diag(np.array([2.0, 3.0, 4.0]))
b_diag = np.array([1.0, 1.0, 1.0])
sol = solve_sparse_system(A_diag, b_diag)
expected = np.array([0.5, 1.0/3.0, 0.25])
assert np.allclose(sol, expected, rtol=1e-12), '[TC14] solve diag system FAILED'

# ---- TC15: monomial_to_legendre_matrix 可逆且乘积为单位阵 ----
A_m2l = monomial_to_legendre_matrix(5)
A_l2m = legendre_to_monomial_matrix(5)
product = A_m2l.dot(A_l2m)
assert np.allclose(product, np.eye(6), atol=1e-10), '[TC15] Legendre matrices not inverse FAILED'

# ---- TC16: spectral_interpolate_legendre 正确求值 P0 ----
from polynomial_spectral import spectral_interpolate_legendre
coeffs = np.zeros(4)
coeffs[0] = 3.0  # 3 * P_0(x) = 3
x_q = np.linspace(-1.0, 1.0, 5)
val = spectral_interpolate_legendre(coeffs, x_q)
assert np.allclose(val, 3.0, rtol=1e-12), '[TC16] Legendre interp of constant FAILED'

# ---- TC17: map_domain_01_to_m1p1 正确映射端点 ----
assert abs(map_domain_01_to_m1p1(0.0) - (-1.0)) < 1e-12, '[TC17] map 0 -> -1 FAILED'
assert abs(map_domain_01_to_m1p1(1.0) - 1.0) < 1e-12, '[TC17] map 1 -> 1 FAILED'
assert abs(map_domain_01_to_m1p1(0.5) - 0.0) < 1e-12, '[TC17] map 0.5 -> 0 FAILED'

# ---- TC18: generate_hollow_fiber_cross_section 生成正确点数 ----
x_poly, y_poly = generate_hollow_fiber_cross_section(32, 1e-4, 2e-4)
assert len(x_poly) == 32, '[TC18] wrong number of vertices FAILED'
assert len(y_poly) == 32, '[TC18] wrong number of y FAILED'

# ---- TC19: polygon_triangulate 产生 n-2 个三角形 ----
n_tri = 16
theta = np.linspace(0.0, 2.0 * np.pi, n_tri, endpoint=False)
x_tri = np.cos(theta)
y_tri = np.sin(theta)
triangles = polygon_triangulate(n_tri, x_tri, y_tri)
assert triangles.shape == (n_tri - 2, 3), f'[TC19] expected ({n_tri-2},3) got {triangles.shape} FAILED'

# ---- TC20: integrate_flux_over_triangles 返回非负值 ----
flux_field = np.ones(n_tri)
total_flux = integrate_flux_over_triangles(triangles, x_tri, y_tri, flux_field)
assert total_flux > 0, '[TC20] flux must be positive FAILED'

# ---- TC21: build_fem_mesh 返回正确尺寸 ----
x_mesh, dx_mesh = build_fem_mesh(1.5e-7, 128)
assert len(x_mesh) == 128, '[TC21] mesh size wrong FAILED'
assert x_mesh[0] == 0.0, '[TC21] first node must be 0 FAILED'
assert abs(x_mesh[-1] - 1.5e-7) < 1e-20, '[TC21] last node must be L FAILED'

# ---- TC22: solve_steady_state_diffusion_reaction 浓度有界 ----
c_feed = 150.0
c_perm = 1.0
x_ss, c_ss = solve_steady_state_diffusion_reaction(1.5e-7, 64, 2.5e-10, 4.2e-3, c_feed, c_perm)
assert np.all(c_ss >= c_perm - 1e-12), '[TC22] concentration below perm FAILED'
assert np.all(c_ss <= c_feed + 1e-12), '[TC22] concentration above feed FAILED'
assert len(c_ss) == 64, '[TC22] wrong profile size FAILED'

# ---- TC23: compute_molar_flux 返回有限值 ----
J_flux = compute_molar_flux(x_ss, c_ss, 2.5e-10)
assert np.all(np.isfinite(J_flux)), '[TC23] flux must be finite FAILED'

# ---- TC24: compute_separation_factor 返回非负值兼容 ----
alpha_test = compute_separation_factor(100.0, 10.0, 800.0, 80.0)
assert alpha_test >= 0, '[TC24] separation factor must be >= 0 FAILED'

# ---- TC25: reaction_deriv 返回 3 分量数组 ----
rp = reaction_parameters()
dy = reaction_deriv(0.0, rp["y0"], rp["k"], rp["K_co2"], rp["K_ch4"], rp["P_total"])
assert len(dy) == 3, '[TC25] reaction_deriv must return 3 components FAILED'

# ---- TC26: kepler_like_trajectory_deriv 返回 4 分量数组 ----
kp = kepler_parameters()
dy_k = kepler_like_trajectory_deriv(0.0, kp["y0"], mu=kp["mu"])
assert len(dy_k) == 4, '[TC26] kepler deriv must return 4 components FAILED'

# ---- TC27: quasiperiodic_forcing_deriv 返回 4 分量数组 ----
qp = quasiperiodic_parameters()
dy_q = quasiperiodic_forcing_deriv(0.0, qp["y0"], qp["omega1"])
assert len(dy_q) == 4, '[TC27] quasiperiodic deriv must return 4 components FAILED'

# ---- TC28: runge_function(0) = 1 ----
assert abs(runge_function(0.0) - 1.0) < 1e-12, '[TC28] Runge(0) = 1 FAILED'

# ---- TC29: runge_derivative(0) = 0 ----
assert abs(runge_derivative(0.0) - 0.0) < 1e-12, '[TC29] Runge\'(0) = 0 FAILED'

# ---- TC30: runge_second_derivative(0) = -50 ----
assert abs(runge_second_derivative(0.0) - (-50.0)) < 1e-10, '[TC30] Runge\'\'(0) = -50 FAILED'

# ---- TC31: runge_kutta4 保持输出形状 ----
t_rk, y_rk = runge_kutta4(lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), 100)
assert y_rk.shape[1] == 1, '[TC31] RK4 output dim incorrect FAILED'

# ---- TC32: adaptive_rk45 对简单系统正确积分（指数衰减） ----
import numpy as np
np.random.seed(42)
t_ad, y_ad = adaptive_rk45(
    lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), atol=1e-10, rtol=1e-8
)
assert y_ad[-1, 0] > 0, '[TC32] adaptive RK45 final value must be positive FAILED'

# ---- TC33: forward_euler 保持输出形状 ----
from time_integrator import forward_euler
t_fe, y_fe = forward_euler(lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), 100)
assert y_fe.shape == (101, 1), '[TC33] forward Euler shape incorrect FAILED'

# ---- TC34: broyden_solve 求解简单标量方程 -------
def f_lin(x):
    return np.array([x[0] - 2.0])
sol_b, ierr_b, _ = broyden_solve(f_lin, np.array([0.0]))
assert ierr_b == 0, '[TC34] Broyden did not converge FAILED'
assert abs(sol_b[0] - 2.0) < 1e-6, '[TC34] Broyden x[0] incorrect FAILED'

# ---- TC35: knudsen_diffusivity 返回正值 ----
D_k = knudsen_diffusivity(5e-9, 308.15, 44.01e-3)
assert D_k > 0, '[TC35] Knudsen diffusivity must be positive FAILED'

# ---- TC36: effective_diffusivity_support 有效扩散系数 ----
D_eff = effective_diffusivity_support(D_k, 0.35, 2.8)
assert D_eff > 0, '[TC36] effective diffusivity must be positive FAILED'
assert D_eff < D_k, '[TC36] effective D < Knudsen D FAILED'

# ---- TC37: build_cascade_adjacency 返回正确形状 ----
A_cas = build_cascade_adjacency(4, recycle_ratio=0.15)
assert A_cas.shape == (4, 4), '[TC37] adjacency shape incorrect FAILED'

# ---- TC38: adjacency_to_google_matrix 列和为 1 ----
G_mat = adjacency_to_google_matrix(A_cas, damping=0.15)
col_sums = np.sum(G_mat, axis=0)
assert np.allclose(col_sums, 1.0, rtol=1e-12), '[TC38] Google matrix columns must sum to 1 FAILED'

# ---- TC39: power_method_rank 返回概率分布 ----
rank_vec = power_method_rank(G_mat, max_iter=200, tol=1e-12)
assert abs(np.sum(rank_vec) - 1.0) < 1e-12, '[TC39] rank must sum to 1 FAILED'
assert np.all(rank_vec >= 0), '[TC39] rank entries must be non-negative FAILED'

# ---- TC40: compute_stage_cuts_from_rank 输出在 [0.05, 0.95] ----
cuts = compute_stage_cuts_from_rank(rank_vec, 0.3)
assert np.all(cuts >= 0.05), '[TC40] stage cuts >= 0.05 FAILED'
assert np.all(cuts <= 0.95), '[TC40] stage cuts <= 0.95 FAILED'

# ---- TC41: subset_sum_optimal_loading 找到可行子集 ----
capacities = np.array([50, 80, 120, 200, 350, 500], dtype=int)
selected = subset_sum_optimal_loading(capacities, 200)
assert len(selected) > 0, '[TC41] subset_sum must find non-empty subset FAILED'

# ---- TC42: cascade_mass_balance 返回正确数组长度 ----
feed_comp = get_feed_composition()
perm_f, ret_f, perm_c = cascade_mass_balance(1000.0, cuts, rank_vec, feed_comp)
assert len(perm_f) == 4, '[TC42] permeate flow array length incorrect FAILED'
assert len(ret_f) == 4, '[TC42] retentate flow array length incorrect FAILED'

# ---- TC43: solve_permeation_nonlinear 收敛并返回 3 分量 ----
p_test = get_membrane_parameters()
perm_test = compute_permeability(p_test)
pf_co2 = 0.15 * p_test["pressure_feed"]
pf_ch4 = 0.80 * p_test["pressure_feed"]
sol_perm, ierr_perm = solve_permeation_nonlinear(
    pf_co2, pf_ch4, perm_test["CO2"], perm_test["CH4"],
    p_test["pressure_permeate"], p_test["membrane_thickness"],
    pf_co2 * 0.1, pf_ch4 * 0.05, T=p_test["temperature"]
)
assert len(sol_perm) == 3, '[TC43] permeation solution must have 3 components FAILED'
assert sol_perm[2] >= 0.0 and sol_perm[2] <= 1.0, '[TC43] stage cut in [0,1] FAILED'

# ---- TC44: runge_mesh_adaptation_nodes 节点数正确 ----
x_adapt = runge_mesh_adaptation_nodes(64, 1.5e-7)
assert len(x_adapt) == 64, '[TC44] adaptation nodes count incorrect FAILED'
assert x_adapt[0] == 0.0, '[TC44] first node must be 0 FAILED'
assert abs(x_adapt[-1] - 1.5e-7) < 1e-20, '[TC44] last node must be L FAILED'

# ---- TC45: coupled_membrane_reaction_ode 返回 9 分量 ----
coupled_params = {
    "k_reaction": 4.2e-3, "K_ads_co2": 1.2e-3, "K_ads_ch4": 2.5e-4,
    "omega1": np.pi, "h_mt": 1e-4,
}
y0_coupled = np.array([150.0, 800.0, 0.0, 150.0, 800.0, 0.01, 0.0, -0.01*np.pi**2, 0.0])
dy_coupled = coupled_membrane_reaction_ode(0.0, y0_coupled, coupled_params)
assert len(dy_coupled) == 9, '[TC45] coupled ODE must return 9 components FAILED'

# ---- TC46: 确定性重复调用 adaptive_rk45 结果一致 ----
import numpy as np
np.random.seed(42)
t1, y1 = adaptive_rk45(
    lambda t, y: np.array([-y[0]]), (0.0, 1.0), np.array([1.0]), atol=1e-10, rtol=1e-8
)
assert abs(y1[-1, 0] - np.exp(-1.0)) < 0.05, '[TC46] adaptive RK45 near exp(-1) FAILED'

# ---- TC47: power_series_solution_ode 在 t=0 处等于常数项 ----
from time_integrator import power_series_solution_ode
coeffs_ps = [3.0, 2.0, 1.0]
val_ps = power_series_solution_ode(0.0, coeffs_ps)
assert abs(val_ps - 3.0) < 1e-12, '[TC47] power series at 0 = c0 FAILED'

# ---- TC48: 确定性 runge_kutta4 结果可复现 ----
import numpy as np
np.random.seed(42)
t_rk1, y_rk1 = runge_kutta4(lambda t, y: np.array([-y[0]]), (0.0, 2.0), np.array([1.0]), 200)
np.random.seed(42)
t_rk2, y_rk2 = runge_kutta4(lambda t, y: np.array([-y[0]]), (0.0, 2.0), np.array([1.0]), 200)
assert np.allclose(y_rk1, y_rk2, rtol=1e-12), '[TC48] RK4 reproducibility FAILED'

# ---- TC49: polygon_area 对单位正方形返回 1.0 ----
from membrane_geometry import polygon_area
x_sq = np.array([0.0, 1.0, 1.0, 0.0])
y_sq = np.array([0.0, 0.0, 1.0, 1.0])
area_sq = polygon_area(4, x_sq, y_sq)
assert abs(area_sq - 1.0) < 1e-12, '[TC49] unit square area = 1 FAILED'

# ---- TC50: safe_sqrt 对极小负数返回 0 不崩溃 ----
from utils import safe_sqrt
assert safe_sqrt(-1e-15) == 0.0, '[TC50] safe_sqrt small negative returns 0 FAILED'
