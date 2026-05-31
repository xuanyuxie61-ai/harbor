# Additional imports for functions not brought into main.py namespace
from utils import fresnel_cos, fresnel_sin, llsq_fit_through_origin, gauss_legendre_1d

# ---- TC01: plasma_frequency returns positive finite value for typical density ----
omega_p = pdm.plasma_frequency(1.0e18)
assert np.isfinite(omega_p) and omega_p > 0.0, '[TC01] plasma_frequency positive finite FAILED'

# ---- TC02: drude_permittivity for overdense plasma has eps_r < 1 ----
eps = pdm.drude_permittivity(1.0e18, 1.0e10, 2.0 * math.pi * 10.0e9)
assert eps.real < 1.0, '[TC02] drude_permittivity eps_r < 1 FAILED'
assert eps.imag >= 0.0, '[TC02] drude_permittivity eps_i non-negative FAILED'

# ---- TC03: drude_permittivity approaches vacuum at very high frequency ----
eps_high = pdm.drude_permittivity(1.0e18, 1.0e10, 2.0 * math.pi * 1000.0e9)
assert abs(eps_high.real - 1.0) < 1e-3, '[TC03] high-frequency eps_r ~ 1 FAILED'
assert abs(eps_high.imag) < 1e-3, '[TC03] high-frequency eps_i ~ 0 FAILED'

# ---- TC04: wave_number has positive imaginary part indicating attenuation ----
k = pdm.wave_number(1.0e18, 1.0e10, 2.0 * math.pi * 10.0e9)
assert k.imag > 0.0, '[TC04] wave_number imag positive FAILED'

# ---- TC05: collision_frequency increases with electron temperature ----
nu_300 = pdm.collision_frequency_from_temperature(300.0, 100.0)
nu_5000 = pdm.collision_frequency_from_temperature(5000.0, 100.0)
assert nu_5000 > nu_300, '[TC05] collision_frequency monotonic with T_e FAILED'

# ---- TC06: skin_depth decreases with increasing electron density ----
delta_low = pdm.skin_depth(1.0e16, 1.0e10, 2.0 * math.pi * 10.0e9)
delta_high = pdm.skin_depth(1.0e19, 1.0e10, 2.0 * math.pi * 10.0e9)
assert delta_high < delta_low, '[TC06] skin_depth decreases with n_e FAILED'

# ---- TC07: effective_permittivity_profile output shape matches input z ----
z_test = np.linspace(0.0, 1.0e-3, 20)
n_e_test = np.full(20, 1.0e17)
nu_test = np.full(20, 1.0e10)
eps_prof = pdm.effective_permittivity_profile(z_test, n_e_test, nu_test, 2.0 * math.pi * 10.0e9)
assert eps_prof.shape == z_test.shape, '[TC07] permittivity_profile shape FAILED'

# ---- TC08: Fresnel reflection at normal incidence between identical media is zero ----
r = fc.fresnel_reflection_coefficient(1.0 + 0j, 1.0 + 0j, 0.0, "TE")
assert abs(r) < 1e-12, '[TC08] Fresnel r=0 for identical media FAILED'

# ---- TC09: reflection_power_ratio of zero amplitude reflection is zero ----
R = fc.reflection_power_ratio(0.0 + 0j)
assert abs(R) < 1e-15, '[TC09] reflection_power_ratio zero FAILED'

# ---- TC10: multilayer stack with single vacuum layer has zero reflection ----
n_layers = np.array([1.0 + 0j])
d_layers = np.array([1.0e-3])
r_total = fc.multilayer_reflection_stack(n_layers, d_layers, 2.0 * math.pi * 1.0e9, theta0=0.0, polarization="TE")
assert abs(r_total) < 1e-10, '[TC10] vacuum layer reflection zero FAILED'

# ---- TC11: field_transfer_matrix output length equals z input length ----
z_fd = np.linspace(0.0, 5.0e-3, 15)
eps_fd = np.ones(15, dtype=complex)
E_fd = lfs.field_transfer_matrix(z_fd, eps_fd, 2.0 * math.pi * 10.0e9, E0=1.0, theta=0.0)
assert E_fd.shape == z_fd.shape, '[TC11] field_transfer_matrix shape FAILED'

# ---- TC12: build_fd_matrix returns square matrix of correct size ----
z_mat = np.linspace(0.0, 1.0e-3, 10)
eps_mat = np.ones(10, dtype=complex)
A_mat, b_mat = lfs.build_fd_matrix(z_mat, eps_mat, 2.0 * math.pi * 10.0e9)
assert A_mat.shape == (10, 10), '[TC12] FD matrix shape FAILED'
assert b_mat.shape == (10,), '[TC12] FD rhs shape FAILED'

# ---- TC13: compute_power_density is non-negative for all points ----
E_test = np.ones(10, dtype=complex)
eps_test = np.ones(10, dtype=complex) * (1.0 - 0.5j)
P_test = lfs.compute_power_density(E_test, eps_test, 2.0 * math.pi * 10.0e9)
assert np.all(P_test >= 0.0), '[TC13] power_density non-negative FAILED'

# ---- TC14: RCS reduction is negative when coating reflects less than metal ----
red_db = ea.rcs_reduction_db(0.1, 1.0)
assert red_db < 0.0, '[TC14] RCS reduction negative FAILED'

# ---- TC15: absorption_efficiency lies in [0, 1] ----
eta = ea.absorption_efficiency(5.0, 10.0, 1.0)
assert 0.0 <= eta <= 1.0, '[TC15] absorption_efficiency range FAILED'

# ---- TC16: 3D Gauss-Legendre integrates constant function exactly ----
def const_func(x, y, z):
    return 2.0
integral_val = ea.integrate_3d_cube_gauss(const_func, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), order_x=1, order_y=1, order_z=1)
assert abs(integral_val - 2.0) < 1e-12, '[TC16] constant 3D integral FAILED'

# ---- TC17: Haar transform followed by inverse reproduces original signal ----
signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
v_haar = wvd.haar_1d_transform(signal)
signal_recon = wvd.haar_1d_inverse(v_haar)
recon_err = float(np.linalg.norm(signal - signal_recon))
assert recon_err < 1e-12, '[TC17] Haar invertibility FAILED'

# ---- TC18: multiscale_energy_distribution preserves total energy (Parseval) ----
sig = np.array([1.5, -0.5, 2.0, 1.0])
energies = wvd.multiscale_energy_distribution(sig)
total_energy = float(np.sum(energies))
original_energy = float(np.sum(sig ** 2))
assert abs(total_energy - original_energy) < 1e-12, '[TC18] Parseval energy conservation FAILED'

# ---- TC19: Sobol sequence is deterministic for same parameters ----
pts1 = qmc.sobol_sequence(3, 16, skip=0)
pts2 = qmc.sobol_sequence(3, 16, skip=0)
assert np.allclose(pts1, pts2), '[TC19] Sobol sequence determinism FAILED'

# ---- TC20: lattice_rule_integrate is exact for constant function ----
const_val = qmc.lattice_rule_integrate(lambda x: 4.0, dim_num=2, m=64)
assert abs(const_val - 4.0) < 1e-12, '[TC20] lattice_rule constant integral FAILED'

# ---- TC21: Gaussian density profile peak equals specified peak_density ----
z_prof = np.linspace(0.0, 1.0e-3, 51)
n_e_prof = dprof.generate_density_profile(z_prof, 2.0e18, 2.0e-4, profile_type="gaussian")
assert abs(np.max(n_e_prof) - 2.0e18) < 1.0e12, '[TC21] gaussian peak density FAILED'

# ---- TC22: pwl_interp_1d interpolates control points exactly ----
xc_pwl = np.array([0.0, 1.0, 2.0, 3.0])
yc_pwl = np.array([0.0, 1.0, 4.0, 9.0])
yi_pwl = dprof.pwl_interp_1d(xc_pwl, yc_pwl, xc_pwl)
assert np.allclose(yi_pwl, yc_pwl), '[TC22] pwl_interp exact at control points FAILED'

# ---- TC23: Chinese Remainder Theorem satisfies all original congruences ----
remainders = np.array([2, 3, 1])
moduli = np.array([3, 5, 7])
x_sol = wcrt.chinese_remainder_theorem(remainders, moduli)
for i in range(remainders.size):
    assert x_sol % int(moduli[i]) == int(remainders[i]), '[TC23] CRT congruence FAILED'

# ---- TC24: encode/decode frequency bands roundtrip consistency ----
freqs = np.array([1.0e9, 2.5e9, 5.0e9])
rem, mods, comp = wcrt.encode_frequency_bands(freqs, bin_width_hz=1.0e8)
decoded = wcrt.decode_frequency_bands(comp, mods, bin_width_hz=1.0e8)
assert np.allclose(decoded, freqs, rtol=1e-10), '[TC24] CRT encode/decode roundtrip FAILED'

# ---- TC25: Fresnel integrals at origin are zero ----
assert abs(fresnel_cos(0.0)) < 1e-15, '[TC25] fresnel_cos(0) FAILED'
assert abs(fresnel_sin(0.0)) < 1e-15, '[TC25] fresnel_sin(0) FAILED'

# ---- TC26: Newton solve converges for simple quadratic equation ----
def f_q(x):
    return x * x - 4.0
def fp_q(x):
    return 2.0 * x
root, f_root, iters, conv = newton_solve(f_q, fp_q, 3.0)
assert conv and abs(root - 2.0) < 1e-10, '[TC26] Newton solve quadratic FAILED'

# ---- TC27: Jacobi solver converges for diagonally dominant system ----
A_jac = np.array([[4.0, 1.0], [1.0, 4.0]], dtype=float)
b_jac = np.array([5.0, 5.0], dtype=float)
x_sol_jac, res_jac, it_jac, conv_jac = jacobi_solve(A_jac, b_jac, max_iter=5000, tol=1e-10)
assert conv_jac and res_jac < 1e-9, '[TC27] Jacobi solver convergence FAILED'

# ---- TC28: llsq_fit_through_origin recovers exact slope for y=3x ----
x_llsq = np.array([1.0, 2.0, 3.0, 4.0])
y_llsq = 3.0 * x_llsq
slope_llsq, resid_llsq = llsq_fit_through_origin(x_llsq, y_llsq)
assert abs(slope_llsq - 3.0) < 1e-12, '[TC28] llsq_fit_through_origin slope FAILED'
assert resid_llsq < 1e-12, '[TC28] llsq_fit_through_origin residual FAILED'
