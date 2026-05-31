# ---- TC01: keplerian_angular_velocity returns positive values ----
r_test = np.array([1e8, 1e9, 1e10])
omega_k = keplerian_angular_velocity(r_test, 1.98847e30)
assert np.all(omega_k > 0), '[TC01] keplerian_angular_velocity returns positive values FAILED'
assert len(omega_k) == len(r_test), '[TC01] Output length mismatch FAILED'

# ---- TC02: sound_speed returns positive for positive temperature ----
T_test = np.array([100.0, 1000.0, 10000.0])
cs = sound_speed(T_test)
assert np.all(cs > 0), '[TC02] sound_speed returns positive for positive temperature FAILED'
assert len(cs) == len(T_test), '[TC02] Output length mismatch FAILED'

# ---- TC03: scale_height returns positive values ----
r_test = np.array([1e8, 1e9, 1e10])
H = scale_height(r_test, 1.98847e30, np.array([1000.0, 1000.0, 1000.0]))
assert np.all(H > 0), '[TC03] scale_height returns positive values FAILED'

# ---- TC04: shakura_sunyaev_sigma output length matches input ----
r_test = np.linspace(1e8, 1e10, 10)
Sigma, T, H = shakura_sunyaev_sigma(r_test, 1e14, 1.98847e30, 0.1)
assert len(Sigma) == len(r_test), '[TC04] shakura_sunyaev_sigma Sigma length mismatch FAILED'
assert len(T) == len(r_test), '[TC04] shakura_sunyaev_sigma T length mismatch FAILED'
assert len(H) == len(r_test), '[TC04] shakura_sunyaev_sigma H length mismatch FAILED'

# ---- TC05: shakura_sunyaev_sigma surface density non-negative ----
assert np.all(Sigma >= 0), '[TC05] shakura_sunyaev_sigma surface density non-negative FAILED'

# ---- TC06: schwarzschild_potential matches analytic Newton potential ----
r_test = np.array([1e8, 1e9, 1e10])
M_test = 1.98847e30
phi = schwarzschild_potential(r_test, M_test)
phi_expected = -G_GRAV * M_test / r_test
assert np.allclose(phi, phi_expected, rtol=1e-10), '[TC06] schwarzschild_potential matches analytic Newton potential FAILED'

# ---- TC07: paczynski_wiita_potential more negative than Newton ----
r_test = np.array([1e8, 1e9, 1e10])
M_test = 1.98847e30
phi_pw = paczynski_wiita_potential(r_test, M_test)
phi_newton = schwarzschild_potential(r_test, M_test)
assert np.all(phi_pw <= phi_newton), '[TC07] paczynski_wiita_potential more negative than Newton FAILED'

# ---- TC08: jet_launching_criterion escape velocity positive ----
r_test = np.array([1e8, 1e9, 1e10])
launched, v_A, v_esc = jet_launching_criterion(r_test, 1e5, 1.0, 1.98847e30)
assert np.all(v_esc > 0), '[TC08] jet_launching_criterion escape velocity positive FAILED'
assert np.all(v_A > 0), '[TC08] jet_launching_criterion Alfven velocity positive FAILED'

# ---- TC09: magnetic_braking_torque sign consistency ----
r_test = np.array([1e8, 1e9, 1e10])
T_mag = magnetic_braking_torque(r_test, 1e2, 1e1, np.ones(3), 1.98847e30)
assert np.all(T_mag >= 0), '[TC09] magnetic_braking_torque sign consistency FAILED'

# ---- TC10: polynomial_multiply analytic verification ----
p = np.array([1.0, 1.0])
q = np.array([1.0, 1.0])
r = polynomial_multiply(p, q)
expected = np.array([1.0, 2.0, 1.0])
assert np.allclose(r, expected), '[TC10] polynomial_multiply analytic verification FAILED'

# ---- TC11: chebyshev_polynomial T2 matches 2x^2 - 1 ----
T2 = chebyshev_polynomial(2)
expected_T2 = np.array([-1.0, 0.0, 2.0])
assert np.allclose(T2, expected_T2), '[TC11] chebyshev_polynomial T2 matches 2x^2 - 1 FAILED'

# ---- TC12: legendre_polynomial P2 matches 0.5*(3x^2-1) ----
P2 = legendre_polynomial(2)
expected_P2 = np.array([-0.5, 0.0, 1.5])
assert np.allclose(P2, expected_P2), '[TC12] legendre_polynomial P2 matches 0.5*(3x^2-1) FAILED'

# ---- TC13: legendre_set weights sum to 2 ----
for n in [1, 2, 3, 4, 5]:
    x, w = legendre_set(n)
    assert abs(np.sum(w) - 2.0) < 1e-14, f'[TC13] legendre_set weights sum to 2 FAILED for n={n}'

# ---- TC14: monomial_integral_3d constant function equals volume ----
box = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
val = monomial_integral_3d([0, 0, 0], box)
assert abs(val - 1.0) < 1e-14, '[TC14] monomial_integral_3d constant function equals volume FAILED'

# ---- TC15: triangle_grid_count analytic formula ----
assert triangle_grid_count(5) == 21, '[TC15] triangle_grid_count analytic formula FAILED'
assert triangle_grid_count(0) == 1, '[TC15] triangle_grid_count N=0 FAILED'

# ---- TC16: triangle_grid points inside triangle ----
vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
pts = triangle_grid(5, vertices)
assert np.all(pts[:, 0] >= -1e-14), '[TC16] triangle_grid x>=0 FAILED'
assert np.all(pts[:, 1] >= -1e-14), '[TC16] triangle_grid y>=0 FAILED'
assert np.all(pts[:, 0] + pts[:, 1] <= 1.0 + 1e-14), '[TC16] triangle_grid x+y<=1 FAILED'

# ---- TC17: lagrange_basis_1d Kronecker delta property ----
x_nodes = np.array([0.0, 0.5, 1.0])
phi = lagrange_basis_1d(x_nodes, x_nodes)
expected = np.eye(3)
assert np.allclose(phi, expected, atol=1e-14), '[TC17] lagrange_basis_1d Kronecker delta property FAILED'

# ---- TC18: rbf_interpolate passes through data points exactly ----
data_pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
data_vals = np.array([1.0, 2.0, 3.0, 4.0])
weights = rbf_weights(data_pts, data_vals, r0=1.0, basis='multiquadric')
interp_vals = rbf_interpolate(data_pts, data_pts, weights, r0=1.0, basis='multiquadric')
assert np.allclose(interp_vals, data_vals, atol=1e-10), '[TC18] rbf_interpolate passes through data points exactly FAILED'

# ---- TC19: r8sd_cg solves identity system ----
n = 3
offset = np.array([0], dtype=np.int64)
A = R8SDMatrix(n, 1, offset)
for i in range(n):
    A.a[i, 0] = 1.0
b = np.array([1.0, 2.0, 3.0])
x, info = r8sd_cg(A, b)
assert info['converged'], '[TC19] r8sd_cg solves identity system convergence FAILED'
assert np.allclose(x, b, atol=1e-10), '[TC19] r8sd_cg solves identity system accuracy FAILED'

# ---- TC20: R8PBLMatrix Cholesky solves simple tridiagonal system ----
n = 3
ml = 1
A_band = R8PBLMatrix(n, ml)
A_band.set_diagonal([2.0, 2.0, 2.0])
A_band.set_subdiagonal(1, [1.0, 1.0])
b = np.array([3.0, 4.0, 3.0])
x = A_band.cholesky_band_solve(b)
assert np.allclose(x, np.ones(3), atol=1e-10), '[TC20] R8PBLMatrix Cholesky solves simple tridiagonal system FAILED'

# ---- TC21: rk3_integrate exponential decay accuracy ----
def exp_rhs(t, y):
    return -y
t_span = [0.0, 1.0]
y0 = np.array([1.0])
t_hist, y_hist = rk3_integrate(exp_rhs, t_span, y0, 100)
y_final = y_hist[-1, 0]
y_exact = np.exp(-1.0)
assert abs(y_final - y_exact) < 1e-6, '[TC21] rk3_integrate exponential decay accuracy FAILED'

# ---- TC22: gradient_1d of linear function is constant ----
f = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
dx = 1.0
grad = gradient_1d(f, dx)
assert np.allclose(grad[1:-1], np.ones(3), atol=1e-14), '[TC22] gradient_1d of linear function is constant FAILED'

# ---- TC23: laplacian_1d of quadratic function is constant ----
f = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
dx = 1.0
lap = laplacian_1d(f, dx)
assert np.allclose(lap[1:-1], 2.0 * np.ones(3), atol=1e-14), '[TC23] laplacian_1d of quadratic function is constant FAILED'

# ---- TC24: velocity_verlet_step free particle trajectory ----
pos = np.array([[0.0, 0.0]])
vel = np.array([[1.0, 0.0]])
force = np.array([[0.0, 0.0]])
new_pos, new_vel, new_force = velocity_verlet_step(pos, vel, force, 0.01)
assert np.allclose(new_pos, np.array([[0.01, 0.0]]), atol=1e-14), '[TC24] velocity_verlet_step position FAILED'
assert np.allclose(new_vel, np.array([[1.0, 0.0]]), atol=1e-14), '[TC24] velocity_verlet_step velocity FAILED'

# ---- TC25: compute_kinetic_energy value correctness ----
vel = np.array([[1.0, 2.0], [3.0, 4.0]])
ke = compute_kinetic_energy(vel)
assert ke >= 0, '[TC25] compute_kinetic_energy non-negative FAILED'
assert abs(ke - 0.5 * (1.0 + 4.0 + 9.0 + 16.0)) < 1e-14, '[TC25] compute_kinetic_energy value FAILED'

# ---- TC26: wedge01_monomial_integral constant function equals volume ----
val = wedge01_monomial_integral([0, 0, 0])
assert abs(val - 1.0) < 1e-14, '[TC26] wedge01_monomial_integral constant function equals volume FAILED'

# ---- TC27: ball_distance_pdf integrates to 1 ----
d_pts = np.linspace(0, 2, 10001)
pdf_vals = ball_distance_pdf(d_pts)
integral = np.trapz(pdf_vals, d_pts)
assert abs(integral - 1.0) < 1e-4, '[TC27] ball_distance_pdf integrates to 1 FAILED'

# ---- TC28: ball_unit_sample points inside unit ball ----
np.random.seed(42)
pts = ball_unit_sample(100, dim=3)
norms = np.linalg.norm(pts, axis=1)
assert np.all(norms <= 1.0 + 1e-14), '[TC28] ball_unit_sample points inside unit ball FAILED'

# ---- TC29: sample_jet_particles speed magnitude correct ----
np.random.seed(42)
pos, vel = sample_jet_particles(50, 1e8, np.pi / 6.0, 1e7)
speeds = np.linalg.norm(vel, axis=1)
assert np.allclose(speeds, np.full(50, 1e7), rtol=1e-10), '[TC29] sample_jet_particles speed magnitude correct FAILED'

# ---- TC30: safe_divide handles division by zero ----
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.0, 1.0, 0.0])
res = safe_divide(a, b, fill_value=999.0)
expected = np.array([999.0, 2.0, 999.0])
assert np.allclose(res, expected), '[TC30] safe_divide handles division by zero FAILED'

# ---- TC31: magic4_matrix row sums equal ----
M = magic4_matrix(4)
row_sums = np.sum(M, axis=1)
assert np.allclose(row_sums, np.full(4, row_sums[0])), '[TC31] magic4_matrix row sums equal FAILED'

# ---- TC32: run_simulation returns expected keys ----
result = run_simulation()
expected_keys = ['r_grid', 'Sigma', 'T_disk', 'phi_grav', 'Sigma_final', 'jet_positions', 'jet_velocities', 'spectrum_nu', 'spectrum_Lnu', 'cg_info', 'md_energy_error']
for key in expected_keys:
    assert key in result, f'[TC32] run_simulation missing key {key} FAILED'
