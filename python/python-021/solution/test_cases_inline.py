# ---- TC01: equilibrium params contains required keys ----
from parameters import get_equilibrium_params
eqp = get_equilibrium_params()
assert "R0" in eqp and "a_minor" in eqp and "q0" in eqp and "nr" in eqp, '[TC01] equilibrium params keys FAILED'

# ---- TC02: Coulomb logarithm clipped to valid range ----
from collision_transport import coulomb_logarithm
lnL = coulomb_logarithm(1e20, 20000)
assert 5.0 <= lnL <= 25.0, '[TC02] Coulomb log range FAILED'

# ---- TC03: electron-ion collision frequency is finite non-negative ----
from collision_transport import electron_ion_collision_frequency
nu = electron_ion_collision_frequency(1e20, 20000, 1.8)
assert nu >= 0.0 and np.isfinite(nu), '[TC03] collision frequency FAILED'

# ---- TC04: thermal velocity increases with temperature ----
from collision_transport import thermal_velocity_electron
v1 = thermal_velocity_electron(1000)
v2 = thermal_velocity_electron(2000)
assert v2 > v1 > 0.0, '[TC04] thermal velocity monotonicity FAILED'

# ---- TC05: mean free path is finite positive ----
from collision_transport import mean_free_path
mfp = mean_free_path(1e20, 20000, 1.8)
assert mfp > 0.0 and np.isfinite(mfp), '[TC05] mean free path FAILED'

# ---- TC06: hypersphere angle approaches 90 deg in higher dimensions ----
np.random.seed(42)
from collision_transport import hypersphere_velocity_sampling
s3 = hypersphere_velocity_sampling(3, 2000, 20000)
s5 = hypersphere_velocity_sampling(5, 2000, 20000)
assert s5["theta_mean_deg"] > s3["theta_mean_deg"], '[TC06] hypersphere angle trend FAILED'

# ---- TC07: Miller boundary returns correct array length ----
from equilibrium_solver import miller_boundary
theta = np.linspace(0, 2 * np.pi, 10)
R_b, Z_b = miller_boundary(theta)
assert len(R_b) == 10 and len(Z_b) == 10, '[TC07] Miller boundary shape FAILED'

# ---- TC08: pressure profile vanishes at boundary ----
from equilibrium_solver import pressure_profile
psi = np.array([0.0, 0.5, 1.0])
p = pressure_profile(psi)
assert abs(p[-1]) < 1e-10, '[TC08] pressure at boundary FAILED'

# ---- TC09: F profile on axis equals R0 * B0 ----
from equilibrium_solver import f_profile
psi = np.array([0.0, 0.5, 1.0])
F = f_profile(psi)
assert abs(F[0] - R0 * B0) < 1e-6, '[TC09] F profile on axis FAILED'

# ---- TC10: simplified DT reactivity lies in expected range ----
from fusion_kinetics import simplified_reactivity
sv = simplified_reactivity(64.0)
assert sv > 1e-24 and sv < 1e-16, '[TC10] reactivity range FAILED'

# ---- TC11: bremsstrahlung power is non-negative and finite ----
from fusion_kinetics import compute_bremsstrahlung
Pb = compute_bremsstrahlung(1e20, 20000, 1.8)
assert Pb >= 0.0 and np.isfinite(Pb), '[TC11] bremsstrahlung FAILED'

# ---- TC12: Lawson criterion is finite positive ----
from fusion_kinetics import lawson_criterion
ntau = lawson_criterion(15.0)
assert ntau > 0.0 and np.isfinite(ntau), '[TC12] Lawson criterion FAILED'

# ---- TC13: magnetic axis point is inside flux surface ----
from geometry_utils import point_in_flux_surface
inside, Rp, Zp = point_in_flux_surface(R0, 0.0)
assert inside is True, '[TC13] point inside center FAILED'

# ---- TC14: far-away point is outside flux surface ----
from geometry_utils import point_in_flux_surface
inside, Rp, Zp = point_in_flux_surface(R0 + 3 * a_minor, 0.0)
assert inside is False, '[TC14] point outside FAILED'

# ---- TC15: poloidal cross-section area is positive ----
from geometry_utils import compute_poloidal_area
area, Rp, Zp = compute_poloidal_area(n_theta=64)
assert area > 0.0, '[TC15] poloidal area FAILED'

# ---- TC16: toroidal volume and analytic approximation are positive ----
from geometry_utils import compute_toroidal_volume
vol, vol_approx = compute_toroidal_volume(n_theta=32, n_radial=16)
assert vol > 0.0 and vol_approx > 0.0, '[TC16] toroidal volume FAILED'

# ---- TC17: UTT determinant equals analytic a0^n ----
from matrix_algebra import r8utt_det
n = 5
a = np.array([3.0, -1.0, 0.5, 0.0, 0.0])
det = r8utt_det(n, a)
assert abs(det - 3.0 ** 5) < 1e-10, '[TC17] UTT determinant FAILED'

# ---- TC18: UTT solve residual is near zero ----
from matrix_algebra import r8utt_solve
n = 5
a = np.array([3.0, -1.0, 0.5, 0.0, 0.0])
b = np.ones(n)
x = r8utt_solve(n, a, b)
A = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        A[i, j] = a[j - i]
res = np.linalg.norm(A @ x - b)
assert res < 1e-10, '[TC18] UTT solve FAILED'

# ---- TC19: Matrix Market write-read roundtrip is consistent ----
from matrix_algebra import write_matrix_market, read_matrix_market
A = np.array([[1.0, 2.0], [3.0, 4.0]])
write_matrix_market("/tmp/test_mm.mtx", A)
A_read, info = read_matrix_market("/tmp/test_mm.mtx")
assert np.allclose(A, A_read), '[TC19] Matrix Market consistency FAILED'

# ---- TC20: triangle stiffness matrix is symmetric ----
from quadrature_engine import assemble_stiffness_triangle
v1 = np.array([0.0, 0.0])
v2 = np.array([1.0, 0.0])
v3 = np.array([0.0, 1.0])
K = assemble_stiffness_triangle(v1, v2, v3)
assert np.allclose(K, K.T), '[TC20] stiffness symmetry FAILED'

# ---- TC21: Gauss quadrature integrates constant function exactly ----
from quadrature_engine import gauss_quadrature
result = gauss_quadrature(lambda x: 1.0, 0.0, 1.0, n=8)
assert abs(result - 1.0) < 1e-12, '[TC21] Gauss quadrature constant FAILED'

# ---- TC22: Alfven dispersion yields positive frequency and velocity ----
from spectral_analysis import alfvén_dispersion
omega, vA = alfvén_dispersion(1.0, 2.0, 5.3, 1e-19)
assert omega > 0.0 and vA > 0.0, '[TC22] Alfven dispersion FAILED'

# ---- TC23: growth rate from exponential spectrum equals expected value ----
from spectral_analysis import compute_growth_rate_from_spectrum
t = np.arange(100) * 0.01
P = np.exp(2.0 * t)
gamma, r2 = compute_growth_rate_from_spectrum(P, dt=0.01)
assert abs(gamma - 1.0) < 0.1 and r2 > 0.99, '[TC23] growth rate FAILED'

# ---- TC24: MHD transition matrix columns sum to unity ----
from mhd_stability import build_mhd_transition_matrix
P_mhd, labels = build_mhd_transition_matrix()
col_sums = P_mhd.sum(axis=0)
assert np.allclose(col_sums, 1.0), '[TC24] MHD transition matrix FAILED'

# ---- TC25: Mercier criterion returns array of correct length ----
from mhd_stability import compute_mercier_criterion
r = np.linspace(0, a_minor, 32)
q = q0 + (q_edge - q0) * (r / a_minor) ** 2
p = np.ones_like(r) * 1e5
Bphi = np.ones_like(r) * B0
Bth = np.ones_like(r) * 0.5
D_M, regions = compute_mercier_criterion(q, r, p, Bphi, Bth)
assert len(D_M) == 32, '[TC25] Mercier criterion FAILED'

# ---- TC26: critical beta is finite positive ----
from mhd_stability import compute_critical_beta
beta_c = compute_critical_beta(q, r, Bphi)
assert beta_c > 0.0 and np.isfinite(beta_c), '[TC26] critical beta FAILED'

# ---- TC27: history interpolation extrapolates left boundary ----
from transport_dde import interpolate_history
t_hist = np.array([0.0, 1.0, 2.0])
y_hist = np.array([1.0, 2.0, 3.0])
y = interpolate_history(-1.0, t_hist, y_hist)
assert y == 1.0, '[TC27] interpolation boundary FAILED'

# ---- TC28: ITER89-P confinement time is finite positive ----
from transport_dde import compute_confinement_time_scaling
tau_E = compute_confinement_time_scaling(15.0, 5.3, 1.0, 50.0, R0, a_minor, KAPPA)
assert tau_E > 0.0 and np.isfinite(tau_E), '[TC28] confinement time FAILED'

# ---- TC29: neoclassical diffusivity is finite positive ----
from transport_dde import compute_particle_diffusivity
D_neo = compute_particle_diffusivity(2.0, R0, a_minor, 1e3, 3e-3)
assert D_neo > 0.0 and np.isfinite(D_neo), '[TC29] particle diffusivity FAILED'

# ---- TC30: drift derivative returns 3-D vector ----
from particle_drift import drift_derivative
y = np.array([1.0, 0.0, 0.0])
dydt = drift_derivative(0.0, y, 1.6, 1.0, 2.0 / 3.0)
assert dydt.shape == (3,), '[TC30] drift derivative shape FAILED'

# ---- TC31: RK2 integrator returns correct array shape ----
from particle_drift import rk2_integrate
def f(t, y):
    return np.array([-y[0]])
t_arr, y_arr = rk2_integrate(f, (0.0, 1.0), np.array([1.0]), n_steps=10)
assert y_arr.shape == (11, 1), '[TC31] RK2 shape FAILED'

# ---- TC32: triangular mesh yields non-empty vertex and triangle arrays ----
from geometry_utils import generate_triangular_mesh
verts, tris = generate_triangular_mesh(n_r=3, n_theta=6)
assert len(verts) > 0 and len(tris) > 0, '[TC32] triangular mesh FAILED'

# ---- TC33: global stiffness matrix shape matches vertex count ----
from matrix_algebra import assemble_global_stiffness
from geometry_utils import generate_triangular_mesh
verts, tris = generate_triangular_mesh(n_r=2, n_theta=4)
K = assemble_global_stiffness(verts, tris)
assert K.shape == (len(verts), len(verts)), '[TC33] global stiffness shape FAILED'

# ---- TC34: CG solver achieves small residual on diagonal system ----
from matrix_algebra import solve_stiffness_system
n = 5
K_test = np.diag(np.arange(1, n + 1, dtype=float))
b_test = np.ones(n)
x_test, info = solve_stiffness_system(K_test, b_test)
assert info["residual"] < 1e-10, '[TC34] CG solver FAILED'

# ---- TC35: FFT spectrum returns matching positive-frequency arrays ----
from spectral_analysis import compute_fft_spectrum
signal = np.sin(2 * np.pi * 5 * np.arange(128) * 0.001)
freqs, power = compute_fft_spectrum(signal, dt=0.001)
assert len(freqs) == len(power) and len(power) > 0, '[TC35] FFT spectrum FAILED'

# ---- TC36: turbulent signal length matches requested samples ----
from spectral_analysis import generate_turbulent_signal
np.random.seed(42)
sig, params = generate_turbulent_signal(n_t=256, dt=1e-4)
assert len(sig) == 256 and "modes" in params, '[TC36] turbulent signal FAILED'

# ---- TC37: DDE simulation produces finite non-negative energy ----
from transport_dde import simulate_energy_transport
t_arr, W_arr, P_loss, info = simulate_energy_transport(n_steps=100)
assert np.all(np.isfinite(W_arr)) and np.all(W_arr >= 0.0), '[TC37] DDE simulation FAILED'

# ---- TC38: fusion burn produces finite non-negative densities ----
from fusion_kinetics import simulate_fusion_burn
t_arr, y_arr, P_fus, Q = simulate_fusion_burn(Ti_keV=15.0, n_steps=100)
assert np.all(np.isfinite(y_arr)) and np.all(y_arr >= 0.0), '[TC38] fusion burn FAILED'

# ---- TC39: guiding center simulation returns correct trajectory shape ----
from particle_drift import simulate_guiding_center
t_arr, y_arr, energy = simulate_guiding_center(n_steps=100)
assert y_arr.shape[1] == 3 and len(energy) == len(t_arr), '[TC39] guiding center FAILED'

# ---- TC40: curvature is finite and non-negative everywhere ----
from geometry_utils import compute_curvature_and_torsion
theta = np.linspace(0, 2 * np.pi, 100)
kappa, rho_c = compute_curvature_and_torsion(theta)
assert np.all(np.isfinite(kappa)) and np.all(kappa >= 0.0), '[TC40] curvature FAILED'
