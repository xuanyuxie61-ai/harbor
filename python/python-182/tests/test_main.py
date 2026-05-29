"""
main.py

Unified zero-parameter entry point for the Bayesian Hierarchical Calibration
of a Spatially-Coupled Reaction-Diffusion System on an Annular Domain.

Running this script executes the complete pipeline:
  1. Synthetic data generation on an annular grid
  2. FEM basis spatial discretization and GMRF prior construction
  3. Periodic tridiagonal covariance computation via r83p
  4. Polynomial surrogate construction for FHN forward model
  5. Latin-hypercube-initialized adaptive MCMC with unicycle rotation proposals
  6. Bayesian quadrature for model evidence (line, square, triangle domains)
  7. Posterior summary output
"""
from inference_engine import run_bayesian_inference


def main():
    print("=" * 70)
    print("Bayesian Inference for Spatially-Coupled Reaction-Diffusion on Annulus")
    print("Domain: Bayesian Inference & MCMC Sampling")
    print("=" * 70)

    results = run_bayesian_inference()

    print("\n" + "=" * 70)
    print("FINAL POSTERIOR SUMMARIES")
    print("=" * 70)
    names = ['a', 'b', 'gamma', 'd0', 'c0', 'c1', 'c2', 'c3', 'log_sigma']
    for i, name in enumerate(names):
        print(f"  {name:12s}: mean = {results['posterior_mean'][i]:.5f},  "
              f"std = {results['posterior_std'][i]:.5f}")

    true = results['data']['true_params']
    print(f"\n  True a       = {true['a']}")
    print(f"  True b       = {true['b']}")
    print(f"  True gamma   = {true['gamma']}")
    print(f"  True d0      = {true['d0']}")
    print(f"  True sigma   = {true['sigma']}")

    print(f"\n  Acceptance rate        = {results['accept_rate']:.3f}")
    print(f"  Evidence slice (gamma) = {results['evidence_gamma']:.4e}")
    print(f"  Evidence slice (a,b)   = {results['evidence_ab']:.4e}")
    print(f"  Evidence slice (simplex)= {results['evidence_triangle']:.4e}")
    print("=" * 70)
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（52个，assert模式，涉及随机值均使用固定种子）
# ================================================================
import numpy as np

# ---- TC01: WichmannHill uniform returns float in (0,1) ----
from prng_asa183 import WichmannHill
wh = WichmannHill(12345, 30306, 13579)
u = wh.uniform()
assert isinstance(u, float), '[TC01] uniform is not float FAILED'
assert 0.0 < u < 1.0, f'[TC01] uniform out of (0,1): {u} FAILED'

# ---- TC02: WichmannHill with fixed seed is reproducible ----
wh1 = WichmannHill(1, 1, 1)
wh2 = WichmannHill(1, 1, 1)
v1 = [wh1.uniform() for _ in range(10)]
v2 = [wh2.uniform() for _ in range(10)]
assert all(abs(a - b) < 1e-15 for a, b in zip(v1, v2)), '[TC02] reproducibility FAILED'

# ---- TC03: WichmannHill.normals returns correct length and finite values ----
wh3 = WichmannHill(12345, 30306, 13579)
narr = wh3.normals(10, 0.0, 1.0)
assert len(narr) == 10, f'[TC03] len={len(narr)}, expected 10 FAILED'
assert np.all(np.isfinite(narr)), '[TC03] normals contain non-finite values FAILED'

# ---- TC04: LEcuyer uniform returns float in (0,1) ----
from prng_asa183 import LEcuyer
le = LEcuyer(12345, 67890)
u_le = le.uniform()
assert isinstance(u_le, float), '[TC04] LEcuyer uniform is not float FAILED'
assert 0.0 < u_le < 1.0, '[TC04] LEcuyer uniform out of (0,1) FAILED'

# ---- TC05: annulus_grid_fibonacci returns correct array length ----
from spatial_domain import annulus_grid_fibonacci
x, y = annulus_grid_fibonacci(0.5, 1.0, 16)
assert len(x) == 16, f'[TC05] len(x)={len(x)}, expected 16 FAILED'
assert len(y) == 16, f'[TC05] len(y)={len(y)}, expected 16 FAILED'

# ---- TC06: fem_basis_2d degree=0 returns 1.0 ----
from spatial_domain import fem_basis_2d
b0 = fem_basis_2d(0, 0, 0, 0.5, 0.5)
assert abs(b0 - 1.0) < 1e-12, f'[TC06] basis(0,0,0)={b0}, expected 1.0 FAILED'

# ---- TC07: fem_basis_2d at node (1,0) for (1,0,0) equals 1 ----
b100 = fem_basis_2d(1, 0, 0, 1.0, 0.0)
assert abs(b100 - 1.0) < 1e-12, f'[TC07] basis(1,0,0)@(1,0)={b100}, expected 1.0 FAILED'

# ---- TC08: fem_basis_2d at node (0,1) for (0,1,0) equals 1 ----
b010 = fem_basis_2d(0, 1, 0, 0.0, 1.0)
assert abs(b010 - 1.0) < 1e-12, f'[TC08] basis(0,1,0)@(0,1)={b010}, expected 1.0 FAILED'

# ---- TC09: fem_basis_2d at node (0,0) for (0,0,1) equals 1 ----
b001 = fem_basis_2d(0, 0, 1, 0.0, 0.0)
assert abs(b001 - 1.0) < 1e-12, f'[TC09] basis(0,0,1)@(0,0)={b001}, expected 1.0 FAILED'

# ---- TC10: circle_distance_pdf at d=0 returns 1/pi ----
from spatial_domain import circle_distance_pdf
import math
pdf0 = circle_distance_pdf(0.0)
assert abs(pdf0 - 1.0 / math.pi) < 1e-12, f'[TC10] pdf(0)={pdf0}, expected 1/pi FAILED'

# ---- TC11: circle_distance_exact_mean equals 4/pi ----
from spatial_domain import circle_distance_exact_mean
mu_exact = circle_distance_exact_mean()
assert abs(mu_exact - 4.0 / math.pi) < 1e-14, f'[TC11] mu={mu_exact}, expected 4/pi FAILED'

# ---- TC12: circle_distance_exact_variance equals 2-16/pi^2 ----
from spatial_domain import circle_distance_exact_variance
var_exact = circle_distance_exact_variance()
assert abs(var_exact - (2.0 - 16.0 / (math.pi * math.pi))) < 1e-14, f'[TC12] var={var_exact}, expected 2-16/pi^2 FAILED'

# ---- TC13: circle_distance_pdf at d=2.0 returns 0 (edge of domain) ----
pdf2 = circle_distance_pdf(2.0)
assert abs(pdf2 - 0.0) < 1e-15, f'[TC13] pdf(2)={pdf2}, expected 0 FAILED'

# ---- TC14: circle_distance_chord for antipodal points equals 2 ----
from utils import circle_distance_chord
d_anti = circle_distance_chord(0.0, math.pi)
assert abs(d_anti - 2.0) < 1e-14, f'[TC14] chord={d_anti}, expected 2 FAILED'

# ---- TC15: circle_distance_chord for coincident points equals 0 ----
d_same = circle_distance_chord(0.3, 0.3)
assert abs(d_same - 0.0) < 1e-14, f'[TC15] chord={d_same}, expected 0 FAILED'

# ---- TC16: latin_edge output shape is (dim, points) ----
from utils import latin_edge
wh_lhs = WichmannHill(42, 97, 13)
lhs = latin_edge(4, 5, wh_lhs)
assert lhs.shape == (4, 5), f'[TC16] shape={lhs.shape}, expected (4,5) FAILED'

# ---- TC17: latin_edge values are all in [0,1] ----
lhs2 = latin_edge(3, 6, wh_lhs)
assert np.all(lhs2 >= 0.0) and np.all(lhs2 <= 1.0), '[TC17] latin_edge values out of [0,1] FAILED'

# ---- TC18: unicycle_next first call returns [1,2,3,4] ----
from utils import unicycle_next
u1, r1 = unicycle_next(4, np.zeros(4, dtype=int), -1)
assert np.array_equal(u1, np.array([1, 2, 3, 4])), f'[TC18] first unicycle={u1}, expected [1,2,3,4] FAILED'

# ---- TC19: perm_lex_next first call returns [1,2,3] with rank 0 ----
from utils import perm_lex_next
p0, rk0 = perm_lex_next(3, np.zeros(3, dtype=int), -1)
assert np.array_equal(p0, np.array([1, 2, 3])), f'[TC19] first perm={p0}, expected [1,2,3] FAILED'
assert rk0 == 0, f'[TC19] rank={rk0}, expected 0 FAILED'

# ---- TC20: jed_to_nyt with date before epoch returns negative ----
from utils import jed_to_nyt
v0, i0 = jed_to_nyt(2300000.0)
assert v0 == -1 and i0 == -1, f'[TC20] jed_to_nyt(2300000.0)=({v0},{i0}), expected (-1,-1) FAILED'

# ---- TC21: nyt_to_jed roundtrip with known date ----
from utils import nyt_to_jed
v_known, i_known = jed_to_nyt(2460000.5)
assert v_known > 0 and i_known > 0, f'[TC21] unexpected negative for date 2460000.5 FAILED'
v_r, i_r = jed_to_nyt(nyt_to_jed(v_known, i_known))
assert v_r == v_known and i_r == i_known, f'[TC21] roundtrip: ({v_known},{i_known}) -> ({v_r},{i_r}) FAILED'

# ---- TC22: r83_np_fa nonzero diagonal succeeds ----
from periodic_solver import r83_np_fa
import numpy as np
a_np = np.zeros((3, 3), dtype=float)
a_np[0, 1] = 1.0; a_np[0, 2] = 1.0   # superdiag
a_np[1, 0] = 3.0; a_np[1, 1] = 2.0; a_np[1, 2] = 1.0  # diag
a_np[2, 0] = 1.0; a_np[2, 1] = 1.0   # subdiag
a_lu_np, info_np = r83_np_fa(3, a_np)
assert info_np == 0, f'[TC22] r83_np_fa info={info_np}, expected 0 FAILED'

# ---- TC23: r83_np_sl solves diagonal system correctly ----
from periodic_solver import r83_np_sl
a_simple = np.zeros((3, 3), dtype=float)
a_simple[1, 0] = 2.0; a_simple[1, 1] = 3.0; a_simple[1, 2] = 4.0
a_lu_s, _ = r83_np_fa(3, a_simple)
b_s = np.array([2.0, 6.0, 12.0])
x_s = r83_np_sl(3, a_lu_s, b_s, 0)
assert abs(x_s[0] - 1.0) < 1e-12, f'[TC23] x[0]={x_s[0]}, expected 1.0 FAILED'
assert abs(x_s[1] - 2.0) < 1e-12, f'[TC23] x[1]={x_s[1]}, expected 2.0 FAILED'
assert abs(x_s[2] - 3.0) < 1e-12, f'[TC23] x[2]={x_s[2]}, expected 3.0 FAILED'

# ---- TC24: r83p_fa + r83p_sl solve periodic tridiagonal ----
from periodic_solver import r83p_fa, r83p_sl
a_per = np.zeros((3, 3), dtype=float)
a_per[0, 0] = 1.0   # lower-left wrap A(2,0)
a_per[0, 1] = 1.0; a_per[0, 2] = 1.0  # superdiag
a_per[1, 0] = 4.0; a_per[1, 1] = 4.0; a_per[1, 2] = 4.0  # diag
a_per[2, 0] = 1.0; a_per[2, 1] = 1.0  # subdiag
a_per[2, 2] = 1.0  # upper-right wrap A(0,2)
alu_p, w2, w3, w4, info_p = r83p_fa(3, a_per)
assert info_p == 0, f'[TC24] r83p_fa info={info_p}, expected 0 FAILED'
x_p = r83p_sl(3, alu_p, np.array([6.0, 6.0, 6.0]), 0, w2, w3, w4)
assert abs(x_p[0] - 1.0) < 1e-10, f'[TC24] x[0]={x_p[0]}, expected 1.0 FAILED'
assert abs(x_p[1] - 1.0) < 1e-10, f'[TC24] x[1]={x_p[1]}, expected 1.0 FAILED'
assert abs(x_p[2] - 1.0) < 1e-10, f'[TC24] x[2]={x_p[2]}, expected 1.0 FAILED'

# ---- TC25: besselzero returns k increasing zeros ----
from forward_models import besselzero
zs = besselzero(0, 3, 1)
assert len(zs) == 3, f'[TC25] len={len(zs)}, expected 3 FAILED'
assert zs[0] > 0, f'[TC25] first zero={zs[0]}, expected >0 FAILED'
assert zs[0] < zs[1] < zs[2], f'[TC25] zeros not increasing: {zs} FAILED'

# ---- TC26: besselzero J_0 first zero close to 2.4048 ----
z1 = besselzero(0, 1, 1)
assert abs(z1[0] - 2.404825557695773) < 1e-8, f'[TC26] j0,1={z1[0]}, expected ~2.4048 FAILED'

# ---- TC27: fitzhugh_nagumo_deriv returns length-2 array ----
from forward_models import fitzhugh_nagumo_deriv
dydt = fitzhugh_nagumo_deriv(0.0, np.array([0.0, 0.0]), 0.7, 0.8, 3.0, 0.0)
assert len(dydt) == 2, f'[TC27] len={len(dydt)}, expected 2 FAILED'
assert np.all(np.isfinite(dydt)), '[TC27] dydt contains non-finite values FAILED'

# ---- TC28: euler_integrate returns correct output shape ----
from forward_models import euler_integrate
t_eu, y_eu = euler_integrate(fitzhugh_nagumo_deriv, (0.0, 1.0), np.array([0.0, 0.0]), 10, a=0.7, b=0.8, c=3.0, d=0.0)
assert len(t_eu) == 11, f'[TC28] len(t)={len(t_eu)}, expected 11 FAILED'
assert y_eu.shape == (11, 2), f'[TC28] y shape={y_eu.shape}, expected (11,2) FAILED'

# ---- TC29: fhn_stationary_voltage returns finite float ----
from forward_models import fhn_stationary_voltage
v_end = fhn_stationary_voltage(0.7, 0.8, 3.0, 0.0)
assert isinstance(v_end, float), f'[TC29] type={type(v_end)}, expected float FAILED'
assert np.isfinite(v_end), f'[TC29] v_end={v_end}, expected finite FAILED'

# ---- TC30: helmholtz_exact basic evaluation does not crash ----
from forward_models import helmholtz_exact
z_h = helmholtz_exact(1.0, 1, 1, 1.0, 0.0, 1.0, np.array([0.5]), np.array([0.0]))
assert np.isfinite(z_h[0]), f'[TC30] Z={z_h[0]}, expected finite FAILED'

# ---- TC31: least_squares_approximant_coef exact line fit ----
from surrogate import least_squares_approximant_coef
xd_lin = np.linspace(0.0, 1.0, 5)
yd_lin = 2.0 + 3.0 * xd_lin
c_lin = least_squares_approximant_coef(5, xd_lin, yd_lin, 2)
assert abs(c_lin[0] - 2.0) < 1e-12, f'[TC31] c[0]={c_lin[0]}, expected 2.0 FAILED'
assert abs(c_lin[1] - 3.0) < 1e-12, f'[TC31] c[1]={c_lin[1]}, expected 3.0 FAILED'

# ---- TC32: poly_value evaluates linear polynomial correctly ----
from surrogate import poly_value
c_32 = np.array([1.0, 2.0])
pv = poly_value(c_32, np.array([3.0]))
assert abs(pv[0] - 7.0) < 1e-12, f'[TC32] poly_value={pv[0]}, expected 7.0 FAILED'

# ---- TC33: build_fhn_surrogate returns dict with expected keys ----
from surrogate import build_fhn_surrogate
sur = build_fhn_surrogate(a_fixed=0.7, b_fixed=0.8, c_fixed=3.0, degree=4, n_train=8, d_min=-0.3, d_max=0.3)
for k in ['c', 'd_min', 'd_max', 'a_fixed', 'b_fixed']:
    assert k in sur, f'[TC33] surrogate missing key "{k}" FAILED'

# ---- TC34: surrogate_predict returns finite float within clamped range ----
from surrogate import surrogate_predict
v_sur_in = surrogate_predict(sur, 0.0)
v_sur_out = surrogate_predict(sur, 10.0)  # will be clamped
assert np.isfinite(v_sur_in), f'[TC34] in-bounds pred non-finite: {v_sur_in} FAILED'
assert np.isfinite(v_sur_out), f'[TC34] out-bounds pred non-finite: {v_sur_out} FAILED'

# ---- TC35: line01_monomial_integral formula correctness ----
from bayesian_quadrature import line01_monomial_integral
assert abs(line01_monomial_integral(0) - 1.0) < 1e-14, '[TC35] int_0^1 x^0 = 1 FAILED'
assert abs(line01_monomial_integral(1) - 0.5) < 1e-14, '[TC35] int_0^1 x^1 = 0.5 FAILED'
assert abs(line01_monomial_integral(2) - 1.0/3.0) < 1e-14, '[TC35] int_0^1 x^2 = 1/3 FAILED'

# ---- TC36: square01_monomial_integral formula correctness ----
from bayesian_quadrature import square01_monomial_integral
assert abs(square01_monomial_integral(0, 0) - 1.0) < 1e-14, '[TC36] int x^0 y^0 = 1 FAILED'
assert abs(square01_monomial_integral(1, 0) - 0.5) < 1e-14, '[TC36] int x^1 y^0 = 0.5 FAILED'
assert abs(square01_monomial_integral(1, 1) - 0.25) < 1e-14, '[TC36] int x^1 y^1 = 0.25 FAILED'

# ---- TC37: triangle01_monomial_integral formula correctness ----
from bayesian_quadrature import triangle01_monomial_integral
assert abs(triangle01_monomial_integral(0, 0) - 0.5) < 1e-14, '[TC37] int_T 1 = 0.5 FAILED'
assert abs(triangle01_monomial_integral(1, 0) - 1.0/6.0) < 1e-14, f'[TC37] int_T x = {triangle01_monomial_integral(1,0)}, expected 1/6 FAILED'

# ---- TC38: line01_sample_ergodic returns values in [0,1] ----
from bayesian_quadrature import line01_sample_ergodic
s_erg = line01_sample_ergodic(50, 0.3)
assert np.all(s_erg >= 0.0) and np.all(s_erg <= 1.0), '[TC38] ergodic sample out of [0,1] FAILED'

# ---- TC39: _build_gmrf_precision is symmetric and positive diagonal ----
from inference_engine import _build_gmrf_precision
Q39 = _build_gmrf_precision(4, 100.0)
assert Q39.shape == (4, 4), f'[TC39] shape={Q39.shape}, expected (4,4) FAILED'
assert np.allclose(Q39, Q39.T), '[TC39] Q is not symmetric FAILED'
assert np.all(np.diag(Q39) > 0), '[TC39] diagonal not positive FAILED'

# ---- TC40: _compute_gmrf_covariance_via_r83p trace matches numpy inverse ----
from inference_engine import _compute_gmrf_covariance_via_r83p
Sigma40 = _compute_gmrf_covariance_via_r83p(4, 100.0)
Q40 = _build_gmrf_precision(4, 100.0)
Sigma_np = np.linalg.inv(Q40)
assert abs(np.trace(Sigma40) - np.trace(Sigma_np)) < 1e-10 * np.trace(Sigma_np), f'[TC40] trace mismatch: r83p={np.trace(Sigma40)}, np={np.trace(Sigma_np)} FAILED'

# ---- TC41: log_prior returns -inf for a out of bounds ----
from inference_engine import log_prior
Q41 = _build_gmrf_precision(4, 100.0)
lp_oob = log_prior(np.array([5.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, math.log(0.1)]), Q41)
assert not np.isfinite(lp_oob), f'[TC41] log_prior for OOB a={lp_oob}, expected -inf FAILED'

# ---- TC42: log_prior returns finite for valid centered parameters ----
params42 = np.array([0.7, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, math.log(0.1)])
lp42 = log_prior(params42, Q41)
assert np.isfinite(lp42), f'[TC42] log_prior for valid params={lp42}, expected finite FAILED'

# ---- TC43: generate_synthetic_data returns dict with expected keys ----
from inference_engine import generate_synthetic_data
wh43 = WichmannHill(12345, 30306, 13579)
data43 = generate_synthetic_data(wh43)
for k in ['x', 'y', 'obs', 'n_sectors', 'true_params', 'batch_volume', 'batch_issue']:
    assert k in data43, f'[TC43] data missing key "{k}" FAILED'

# ---- TC44: run_adaptive_mcmc chain has correct shape ----
from inference_engine import log_posterior
from mcmc_sampler import run_adaptive_mcmc
import numpy as np, math
wh44 = WichmannHill(12345, 30306, 13579)
data44 = generate_synthetic_data(wh44)
Q44 = _build_gmrf_precision(4, 100.0)
sur44 = build_fhn_surrogate(a_fixed=0.7, b_fixed=0.8, c_fixed=3.0, degree=4, n_train=8, d_min=-0.3, d_max=0.3)
init44 = np.array([0.7, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, math.log(0.1)])
lp44 = lambda p: log_posterior(p, data44, Q44, sur44)
chain44, lpt44, ar44 = run_adaptive_mcmc(lp44, init44, wh44, n_iter=30, proposal_scale=0.05, rotation_period=10)
assert chain44.shape == (31, 9), f'[TC44] chain shape={chain44.shape}, expected (31,9) FAILED'
assert len(lpt44) == 31, f'[TC44] logpost len={len(lpt44)}, expected 31 FAILED'
assert 0.0 <= ar44 <= 1.0, f'[TC44] accept rate={ar44}, expected in [0,1] FAILED'

# ---- TC45: WichmannHill seed validation rejects invalid values ----
try:
    WichmannHill(0, 1, 1)
    assert False, '[TC45] should have raised ValueError for invalid s1 FAILED'
except ValueError:
    pass

# ---- TC46: LEcuyer seed validation rejects invalid values ----
try:
    LEcuyer(0, 5)
    assert False, '[TC46] should have raised ValueError for invalid s1 FAILED'
except ValueError:
    pass

# ---- TC47: annulus_grid raises ValueError for invalid parameters ----
from spatial_domain import annulus_grid
try:
    annulus_grid(1.0, 0.5, 3, 4)
    assert False, '[TC47] annulus_grid should raise for r1>r2 FAILED'
except ValueError:
    pass

# ---- TC48: euler_integrate raises ValueError for n<1 ----
try:
    euler_integrate(fitzhugh_nagumo_deriv, (0.0, 1.0), np.array([0.0, 0.0]), 0, a=0.7, b=0.8, c=3.0, d=0.0)
    assert False, '[TC48] euler_integrate should raise for n=0 FAILED'
except ValueError:
    pass

# ---- TC49: fhn_stationary_voltage deterministic with fixed seed PRNG not used (no randomness in FHN) ----
# The FHN Euler integration is deterministic; verify by calling twice
v49a = fhn_stationary_voltage(0.7, 0.8, 3.0, 0.0)
v49b = fhn_stationary_voltage(0.7, 0.8, 3.0, 0.0)
assert abs(v49a - v49b) < 1e-15, '[TC49] FHN stationary voltage not deterministic FAILED'

# ---- TC50: integrate_1d of x^2 via ergodic approx 1/3 ----
from bayesian_quadrature import integrate_1d
def f_sq(x):
    return x * x
mu50, se50 = integrate_1d(f_sq, n=1000, method='ergodic', shift=0.1)
assert abs(mu50 - 1.0/3.0) < 0.05, f'[TC50] int x^2={mu50}, expected ~1/3 (tol 0.05) FAILED'

# ---- TC51: integrate_triangle of constant 1 returns ~0.5 ----
from bayesian_quadrature import integrate_triangle
wh51 = WichmannHill(12345, 30306, 13579)
def f_one(pts):
    return np.ones(pts.shape[1])
mu51, se51 = integrate_triangle(f_one, n=200, method='random', rng=wh51)
assert abs(mu51 - 0.5) < 0.1, f'[TC51] int_T 1={mu51}, expected ~0.5 FAILED'

# ---- TC52: WichmannHill.uniforms returns correct length ----
wh52 = WichmannHill(1, 1, 1)
u52 = wh52.uniforms(15)
assert len(u52) == 15, f'[TC52] len={len(u52)}, expected 15 FAILED'
assert np.all(u52 >= 0.0) and np.all(u52 < 1.0), '[TC52] values out of [0,1) FAILED'

print('\n全部 52 个测试通过!\n')
