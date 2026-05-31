# ---- TC01: complex_log_stable 实数输入返回正确对数值 ----
ln1 = complex_log_stable(1.0 + 0j)
assert abs(ln1.real) < 1e-14 and abs(ln1.imag) < 1e-14, '[TC01] ln(1+0j)=0 FAILED'

# ---- TC02: complex_log_stable 纯虚数返回正确的幅角 ----
lni = complex_log_stable(1j)
assert abs(lni.real) < 1e-14, '[TC02] ln(i) real part ~ 0 FAILED'
assert abs(lni.imag - np.pi / 2.0) < 1e-14, '[TC02] ln(i) imag ~ pi/2 FAILED'

# ---- TC03: gegenbauer_integral 零次单项式返回正值 ----
val0 = gegenbauer_integral(0, 0.5)
assert val0 > 0, '[TC03] gegenbauer_integral(0, 0.5) > 0 FAILED'

# ---- TC04: gegenbauer_integral 奇数次单项式返回 0 ----
val_odd = gegenbauer_integral(1, 0.5)
assert abs(val_odd) < 1e-14, '[TC04] gegenbauer_integral(1, 0.5) = 0 FAILED'

# ---- TC05: thiele_modulus_efficiency_factor phi=0 返回 1 ----
eta0 = thiele_modulus_efficiency_factor(0.0, shape_factor=3)
assert abs(eta0 - 1.0) < 1e-14, '[TC05] eta(0) = 1 FAILED'

# ---- TC06: thiele_modulus_efficiency_factor 强扩散限制下 eta < 0.3 ----
eta_big = thiele_modulus_efficiency_factor(10.0, shape_factor=3)
assert 0.0 < eta_big < 0.3, '[TC06] eta(10) in (0, 0.3) FAILED'

# ---- TC07: thiele_modulus_efficiency_factor 大 phi 单调递减 ----
eta_small = thiele_modulus_efficiency_factor(1.0, shape_factor=3)
eta_medium = thiele_modulus_efficiency_factor(5.0, shape_factor=3)
assert eta_small > eta_medium, '[TC07] eta monotonically decreasing FAILED'

# ---- TC08: knudsen_diffusivity 返回正值 ----
D_kn_test = knudsen_diffusivity(15e-9, 573.0, 28.01e-3)
assert D_kn_test > 0, '[TC08] D_Kn > 0 FAILED'

# ---- TC09: effective_diffusivity 结果小于体扩散系数 ----
D_e_test = effective_diffusivity(15e-9, 573.0, 28.01e-3, 2.0e-5, 3.5, 0.42)
assert 0 < D_e_test < 2.0e-5, '[TC09] 0 < D_e < D_bulk FAILED'

# ---- TC10: arrhenius_rate 在基础温度下返回正数 ----
k_rate = arrhenius_rate(1.2e8, 75000.0, 573.0)
assert k_rate > 0, '[TC10] k > 0 FAILED'

# ---- TC11: solve_tridiagonal 简单已知系统的解 ----
n_tri = 5
a_tri = 2.0 * np.ones(n_tri)
b_tri = -1.0 * np.ones(n_tri - 1)
c_tri = -1.0 * np.ones(n_tri - 1)
rhs_tri = np.ones(n_tri)
x_tri = solve_tridiagonal(a_tri, b_tri, c_tri, rhs_tri)
assert x_tri.size == n_tri, '[TC11] x_tri size correct FAILED'
assert x_tri[0] > 0, '[TC11] x_tri[0] > 0 FAILED'

# ---- TC12: conjugate_gradient_rc 单位矩阵系统返回精确解 ----
n_cg = 10
b_cg = np.ones(n_cg)
A_mat_cg = np.eye(n_cg)
precon_cg = jacobi_preconditioner(A_mat_cg)
x_cg, info_cg = conjugate_gradient_rc(
    n_cg, b_cg,
    matvec=lambda p: A_mat_cg @ p,
    precon_solve=precon_cg,
    tol=1e-12
)
assert np.allclose(x_cg, b_cg), '[TC12] CG solves Ix=b FAILED'

# ---- TC13: gauss_legendre_rule 权重之和等于区间长度 ----
x_gl, w_gl = gauss_legendre_rule(7, a=-2.0, b=3.0)
assert abs(np.sum(w_gl) - 5.0) < 1e-14, '[TC13] GL weights sum to (b-a) FAILED'

# ---- TC14: gauss_genlaguerre_rule 节点均在 [a, +∞) ----
x_lag, w_lag = gauss_genlaguerre_rule(n=10, alpha=0.5, a=0.0, b=1.0)
assert np.all(x_lag >= 0.0), '[TC14] Laguerre nodes >= a FAILED'

# ---- TC15: radial_quadrature_sphere 节点均在 [0, R] 内 ----
r_q, w_q_sphere = radial_quadrature_sphere(8, R=1.0)
assert np.all((r_q >= 0.0) & (r_q <= 1.0)), '[TC15] sphere nodes in [0, R] FAILED'

# ---- TC16: gegenbauer_quadrature_exactness 零次多项式精确 ----
errs_geg = gegenbauer_quadrature_exactness(alpha=0.5, n_points=6, degree_max=0)
assert errs_geg.get(0, 1.0) < 1e-13, '[TC16] gegenbauer exactness degree=0 FAILED'

# ---- TC17: integrate_reaction_rate_radial 常数反应速率的体积分 ----
R_test_sphere = 1.0
const_rate = 0.5
total_rate_test = integrate_reaction_rate_radial(
    lambda r: const_rate, R=R_test_sphere, n_quad=24
)
expected_vol = const_rate * (4.0 / 3.0) * np.pi * (R_test_sphere ** 3)
assert abs(total_rate_test - expected_vol) < 1e-12, '[TC17] constant rate integral FAILED'

# ---- TC18: pore_size_moment_quadrature 零阶矩为 1 ----
d_test = np.array([-0.5, 0.0, 0.5])
w_test = np.array([1.0, 1.0, 1.0])
mom0 = pore_size_moment_quadrature(d_test, w_test, moment_order=0)
assert abs(mom0) > 0, '[TC18] zero-order moment finite FAILED'

# ---- TC19: cvt_1d_lloyd 生成器均在域内 ----
import numpy as np
np.random.seed(42)
gen_1d, _ = cvt_1d_lloyd(
    n_generators=12, n_iterations=8, n_samples=15000,
    density_func=None, domain=(0.0, 5.0)
)
assert np.all((gen_1d >= 0.0) & (gen_1d <= 5.0)), '[TC19] 1D CVT generators in domain FAILED'

# ---- TC20: adaptive_radial_mesh 包含边界 0 和 R ----
nodes_test = adaptive_radial_mesh(R=2.0, n_nodes=21, reaction_steepness=3.0)
assert nodes_test[0] == 0.0, '[TC20] first node = 0 FAILED'
assert nodes_test[-1] == 2.0, '[TC20] last node = R FAILED'

# ---- TC21: pwl_interp_2d 网格点插值精确 ----
from interpolation import pwl_interp_2d_scalar
xd_test = np.linspace(0, 1, 10)
yd_test = np.linspace(0, 1, 10)
Xg, Yg = np.meshgrid(xd_test, yd_test, indexing='ij')
zd_test = Xg + Yg
zi_exact = pwl_interp_2d_scalar(xd_test, yd_test, zd_test, 0.5, 0.5)
assert abs(zi_exact - 1.0) < 1e-13, '[TC21] PWL interp exact at midpoint FAILED'

# ---- TC22: random_triangle_area_in_disk 可复现性 ----
import numpy as np
rng_test = np.random.default_rng(42)
mean1, _ = random_triangle_area_in_disk(5000, rng=rng_test)
rng_test2 = np.random.default_rng(42)
mean2, _ = random_triangle_area_in_disk(5000, rng=rng_test2)
assert abs(mean1 - mean2) < 1e-15, '[TC22] MC reproducibility FAILED'

# ---- TC23: pore_tortuosity_from_mc 结果在 [1, 10] 之间 ----
rng_tau = np.random.default_rng(42)
tau_mc_test = pore_tortuosity_from_mc(n_trials=5000, rng=rng_tau)
assert 1.0 <= tau_mc_test <= 10.0, '[TC23] tortuosity in [1, 10] FAILED'

# ---- TC24: validate_2d_quadrature_rule 低次多项式精确 ----
max_err_v, _ = validate_2d_quadrature_rule(n_points=5, degree_max=2)
assert max_err_v < 1e-13, '[TC24] 2D quadrature exact for low-degree FAILED'

# ---- TC25: black_scholes_diffusion_analogy T=0 返回内在价值 ----
call_t0, _, _ = black_scholes_diffusion_analogy(S=100.0, K=80.0, T=0.0, r=0.05, sigma=0.2)
assert call_t0 == 20.0, '[TC25] BS T=0 intrinsic value FAILED'

# ---- TC26: black_scholes_diffusion_analogy 返回有限值 ----
call_p, d1, d2 = black_scholes_diffusion_analogy(S=100.0, K=95.0, T=1.0, r=0.05, sigma=0.2)
assert np.isfinite(call_p), '[TC26] BS call price finite FAILED'
assert d1 > d2, '[TC26] d1 > d2 FAILED'

# ---- TC27: LangmuirHinshelwoodKinetics 零浓度返回零速率 ----
kin_test = LangmuirHinshelwoodKinetics(
    k0=1.0, Ea=50000.0, KA0=1e-5, dH_ads_A=-40000.0,
    KB0=1e-4, dH_ads_B=-30000.0,
)
rate_zero = kin_test.rate(0.0, 0.0, 500.0)
assert rate_zero == 0.0, '[TC27] LH zero concentration -> zero rate FAILED'

# ---- TC28: PowerLawKinetics 一阶反应线性关系 ----
pl_kin = PowerLawKinetics(k0=1.0, Ea=0.0, nA=1.0, nB=0.0)
rate_a = pl_kin.rate(2.0, 0.0, 500.0)
rate_b = pl_kin.rate(4.0, 0.0, 500.0)
assert abs(rate_b / rate_a - 2.0) < 1e-13, '[TC28] first-order linearity FAILED'

# ---- TC29: solve_diffusion_reaction_fd 返回正确形状数组 ----
import numpy as np
np.random.seed(42)
r_test_nodes = np.linspace(0.0, 0.001, 31)
D_test = 1e-6
C_surf_test = 10.0
def reac_test(C, r):
    return 0.1 * C
C_fd_test, _ = solve_diffusion_reaction_fd(r_test_nodes, D_test, reac_test, C_surf_test)
assert C_fd_test.size == r_test_nodes.size, '[TC29] FDM output size correct FAILED'
assert C_fd_test[-1] == C_surf_test, '[TC29] FDM surface BC FAILED'

# ---- TC30: diffusion_flux_at_surface 返回有限值 ----
r_flux_nodes = np.linspace(0.0, 0.001, 11)
C_flux = np.linspace(8.0, 10.0, 11)
J_test = diffusion_flux_at_surface(C_flux, r_flux_nodes, 1e-6)
assert np.isfinite(J_test), '[TC30] surface flux finite FAILED'
assert J_test < 0, '[TC30] surface flux negative (inward diffusion) FAILED'

# ---- TC31: effectiveness_factor_from_profile 返回 [0, inf) 值 ----
r_eff = np.linspace(0.0, 0.001, 21)
C_eff = np.linspace(5.0, 10.0, 21)
eta_test = effectiveness_factor_from_profile(C_eff, r_eff, 0.001, lambda C, r: 0.1 * C)
assert eta_test >= 0, '[TC31] effectiveness factor non-negative FAILED'

# ---- TC32: validate_reaction_diffusion_conservation 误差有限 ----
r_cons = np.linspace(0.0, 0.001, 51)
C_cons = 10.0 * np.ones(51)
rates_cons = 0.5 * np.ones(51)
_, _, rel_err = validate_reaction_diffusion_conservation(C_cons, 0.001, r_cons, rates_cons)
assert np.isfinite(rel_err), '[TC32] conservation error finite FAILED'

# ---- TC33: diffusion_green_function_integral 返回有限积分值 ----
r_g = np.linspace(0, 1e-5, 200)
int_val, exact, _ = diffusion_green_function_integral(r_g, t=1e-4, D=1e-6, R=1e-5)
assert int_val > 0, '[TC33] Green integral > 0 FAILED'
assert int_val < exact, '[TC33] truncated integral < full-space FAILED'

# ---- TC34: pwl_interp_2d_scalar 两点插值一致性 ----
from interpolation import pwl_interp_2d_scalar
xd_batch = np.array([0.0, 0.5, 1.0])
yd_batch = np.array([0.0, 0.5, 1.0])
Zb = np.ones((3, 3))
for i in range(3):
    for j in range(3):
        Zb[i, j] = float(i + j)
z34_a = pwl_interp_2d_scalar(xd_batch, yd_batch, Zb, 0.25, 0.25)
z34_b = pwl_interp_2d_scalar(xd_batch, yd_batch, Zb, 0.75, 0.75)
assert np.isfinite(z34_a), '[TC34] PWL interp point A finite FAILED'
assert np.isfinite(z34_b), '[TC34] PWL interp point B finite FAILED'
assert z34_a != z34_b, '[TC34] PWL interp points differ FAILED'

# ---- TC35: jacobi_preconditioner 对角线矩阵返回逆对角线 ----
A_precon2 = np.diag(np.array([2.0, 3.0, 4.0]))
precon2 = jacobi_preconditioner(A_precon2)
r_test_p = np.array([4.0, 9.0, 16.0])
z_test_p = precon2(r_test_p)
assert np.allclose(z_test_p, np.array([2.0, 3.0, 4.0])), '[TC35] Jacobi preconditioner FAILED'

# ---- TC36: write_xy_profile 与 read_xy_profile 往返测试 ----
import os
import tempfile
tmp_filename = os.path.join(tempfile.gettempdir(), '_test_profile_136.txt')
x_write = np.array([0.0, 0.5, 1.0])
y_write = np.array([10.0, 20.0, 30.0])
write_xy_profile(tmp_filename, x_write, y_write, header="Test profile")
x_read, y_read = read_xy_profile(tmp_filename)
assert np.allclose(x_read, x_write), '[TC36] xy roundtrip x FAILED'
assert np.allclose(y_read, y_write), '[TC36] xy roundtrip y FAILED'
os.remove(tmp_filename)
