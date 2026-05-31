# ---- TC01: i4_gcd - Euclidean algorithm GCD(48,18)=6 ----
from integer_utils import i4_gcd
assert i4_gcd(48, 18) == 6, '[TC01] GCD(48,18)=6 FAILED'

# ---- TC02: i4_lcm - LCM(12,18)=36 ----
from integer_utils import i4_lcm
assert i4_lcm(12, 18) == 36, '[TC02] LCM(12,18)=36 FAILED'

# ---- TC03: i4_choose - binomial C(10,3)=120 ----
from integer_utils import i4_choose
assert i4_choose(10, 3) == 120, '[TC03] C(10,3)=120 FAILED'

# ---- TC04: i4_factorial - 5!=120 ----
from integer_utils import i4_factorial
assert i4_factorial(5) == 120, '[TC04] 5!=120 FAILED'

# ---- TC05: i4_factorial2 - 5!!=15 ----
from integer_utils import i4_factorial2
v = i4_factorial2(5)
assert v == 15, '[TC05] 5!!=15 FAILED'

# ---- TC06: prime_list - first 5 primes are correct ----
from integer_utils import prime_list
p = prime_list(5)
assert len(p) == 5, '[TC06] prime_list length=5 FAILED'
assert p[0] == 2 and p[4] == 11, '[TC06] first/last prime FAILED'

# ---- TC07: halton_sequence - deterministic output match ----
import numpy as np
from integer_utils import halton_sequence
vals = np.array([halton_sequence(i, 2) for i in range(1, 6)])
assert vals.shape == (5, 2), '[TC07] halton_sequence shape FAILED'
assert np.all(np.isfinite(vals)), '[TC07] halton_sequence not finite FAILED'
assert np.all((vals >= 0.0) & (vals < 1.0)), '[TC07] halton_sequence range FAILED'

# ---- TC08: dof_count - P1 in 3D = C(1+3,3) = 4 ----
from integer_utils import dof_count
assert dof_count(1, 3) == 4, '[TC08] dof_count(1,3)=4 FAILED'
assert dof_count(2, 3) == 10, '[TC08] dof_count(2,3)=10 FAILED'

# ---- TC09: primitive_to_conservative roundtrip consistency ----
import numpy as np
from euler_equations import primitive_to_conservative, conservative_to_primitive
U = primitive_to_conservative(1.0, 0.2, 0.3, -0.1, 1.0)
rho2, u2, v2, w2, p2 = conservative_to_primitive(U)
assert abs(rho2 - 1.0) < 1e-12, '[TC09] rho roundtrip FAILED'
assert abs(u2 - 0.2) < 1e-12, '[TC09] u roundtrip FAILED'
assert abs(p2 - 1.0) < 1e-12, '[TC09] p roundtrip FAILED'

# ---- TC10: pressure from conservative state ----
import numpy as np
from euler_equations import primitive_to_conservative, pressure
U = primitive_to_conservative(1.0, 0.0, 0.0, 0.0, 2.0)
p = pressure(U)
assert abs(p - 2.0) < 1e-12, '[TC10] pressure=2.0 FAILED'

# ---- TC11: speed_of_sound - c = sqrt(gamma*p/rho) ----
import numpy as np
from euler_equations import primitive_to_conservative, speed_of_sound, GAMMA
U = primitive_to_conservative(1.0, 0.0, 0.0, 0.0, 1.0)
c = speed_of_sound(U)
c_expected = np.sqrt(GAMMA * 1.0 / 1.0)
assert abs(c - c_expected) < 1e-12, '[TC11] speed_of_sound FAILED'

# ---- TC12: rusanov_flux - zero jump gives average ----
import numpy as np
from euler_equations import primitive_to_conservative, rusanov_flux, flux_dot_n
U0 = primitive_to_conservative(1.0, 0.2, 0.0, 0.0, 1.0)
flux_avg = 0.5 * (flux_dot_n(U0, 1.0, 0.0, 0.0) + flux_dot_n(U0, 1.0, 0.0, 0.0))
flux_rus = rusanov_flux(U0, U0, 1.0, 0.0, 0.0)
assert np.all(np.abs(flux_rus - flux_avg) < 1e-12), '[TC12] rusanov_flux zero-jump FAILED'

# ---- TC13: manufactured_solution_3d positivity ----
import numpy as np
from euler_equations import manufactured_solution_3d, conservative_to_primitive
U = manufactured_solution_3d(0.5, 0.5, 0.5, 0.0)
rho, u, v, w, p = conservative_to_primitive(U)
assert rho > 0.9, '[TC13] rho positivity FAILED'
assert p > 0.9, '[TC13] p positivity FAILED'

# ---- TC14: tetrahedron_volume - reference tet volume = 1/6 ----
import numpy as np
from tetrahedron_geometry import tetrahedron_volume, REFERENCE_TET4_VERTICES
vol = tetrahedron_volume(REFERENCE_TET4_VERTICES)
assert abs(vol - 1.0 / 6.0) < 1e-14, '[TC14] reference tet volume=1/6 FAILED'

# ---- TC15: shape_function_linear_tet4 - partition of unity ----
import numpy as np
from tetrahedron_geometry import shape_function_linear_tet4
phi = shape_function_linear_tet4(0.25, 0.25, 0.25)
assert abs(np.sum(phi) - 1.0) < 1e-14, '[TC15] partition of unity FAILED'

# ---- TC16: r8mat_det_3d - 3x3 determinant ----
import numpy as np
from tetrahedron_geometry import r8mat_det_3d
A = np.array([[1,0,0],[0,2,0],[0,0,3]], dtype=np.float64)
detA = r8mat_det_3d(A)
assert abs(detA - 6.0) < 1e-14, '[TC16] r8mat_det_3d=6 FAILED'

# ---- TC17: rk4_step on dy/dt = y, y0=1, dt=0.1 ----
import numpy as np
from time_integrator import rk4_step
y0 = np.array([1.0])
def rhs(y, t):
    return y
y1 = rk4_step(y0, 0.0, 0.1, rhs)
assert abs(y1[0] - np.exp(0.1)) < 1e-7, '[TC17] rk4_step exp FAILED'

# ---- TC18: ssp_rk3_step on dy/dt = y, y0=1, dt=0.1 ----
import numpy as np
from time_integrator import ssp_rk3_step
y0 = np.array([1.0])
def rhs(y, t):
    return y
y1 = ssp_rk3_step(y0, 0.0, 0.1, rhs)
assert abs(y1[0] - np.exp(0.1)) < 1e-3, '[TC18] ssp_rk3_step exp FAILED'

# ---- TC19: exact_monomial_integral - constant = 1/6 ----
import numpy as np
from quadrature_rules import exact_monomial_integral_tetrahedron
val = exact_monomial_integral_tetrahedron((0, 0, 0))
assert abs(val - 1.0 / 6.0) < 1e-14, '[TC19] exact_monomial(0,0,0)=1/6 FAILED'

# ---- TC20: estimate_convergence_rate - synthetic p=2 ----
import numpy as np
from error_estimator import estimate_convergence_rate
h = np.array([0.5, 0.25, 0.125, 0.0625])
e = 0.1 * h ** 2.0
p, C = estimate_convergence_rate(e, h)
assert abs(p - 2.0) < 1e-6, '[TC20] convergence rate p=2 FAILED'

# ---- TC21: dual_weighted_residual_estimate ----
import numpy as np
from error_estimator import dual_weighted_residual_estimate
res = np.array([0.1, 0.2, 0.05])
dw = np.array([1.0, 2.0, 0.5])
vol = np.array([0.1, 0.1, 0.1])
eta = dual_weighted_residual_estimate(res, dw, vol)
expected = 0.1*1.0*0.1 + 0.2*2.0*0.1 + 0.05*0.5*0.1
assert abs(eta - expected) < 1e-14, '[TC21] DWR estimate FAILED'

# ---- TC22: build_sparse_laplacian_1d - CRS matrix-vector ----
import numpy as np
from mesh_io import build_sparse_laplacian_1d
n = 10
lap = build_sparse_laplacian_1d(n)
x = np.ones(n, dtype=np.float64)
y = lap.mv(x)
assert y.shape == (n,), '[TC22] Laplacian MV shape FAILED'
assert abs(y[0] - 1.0) < 1e-12, '[TC22] Laplacian[0]=1 FAILED'
assert abs(y[n-1] - 1.0) < 1e-12, '[TC22] Laplacian[-1]=1 FAILED'

# ---- TC23: minmod - basic cases ----
import numpy as np
from limiter import minmod
assert minmod(1.0, 2.0) == 1.0, '[TC23] minmod(1,2)=1 FAILED'
assert minmod(-1.0, 2.0) == 0.0, '[TC23] minmod(-1,2)=0 FAILED'
assert minmod(-3.0, -1.0) == -1.0, '[TC23] minmod(-3,-1)=-1 FAILED'
assert minmod(0.0, 5.0) == 0.0, '[TC23] minmod(0,5)=0 FAILED'

# ---- TC24: rbf_multiquadric - positive values ----
import numpy as np
from rbf_reconstruction import rbf_multiquadric
r = np.array([0.0, 1.0, 2.0])
v = rbf_multiquadric(r, r0=1.0)
assert np.all(v > 0.0), '[TC24] rbf_multiquadric positivity FAILED'
assert v[0] == 1.0, '[TC24] rbf_multiquadric r=0 FAILED'

# ---- TC25: generate_unit_cube_tetrahedra - mesh integrity ----
import numpy as np
from mesh_io import generate_unit_cube_tetrahedra
mesh = generate_unit_cube_tetrahedra()
assert mesh.n_elem == 6, '[TC25] unit cube n_elem=6 FAILED'
assert mesh.n_nodes == 8, '[TC25] unit cube n_nodes=8 FAILED'
vol_total = sum(mesh.element_volume(e) for e in range(mesh.n_elem))
assert abs(vol_total - 1.0) < 1e-12, '[TC25] total volume=1 FAILED'
