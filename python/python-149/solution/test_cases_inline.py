from sde_integrator import generate_brownian_path
from numeric_utils import himmelblau_function, rosenbrock_gradient
from state_space_tools import deserialize_state_trajectory

# ---- TC01: sigmoid_activation extreme values return clean 0 or 1 ----
assert sigmoid_activation(-1000.0, 1.0, 0.0) == 0.0, '[TC01] sigmoid large negative FAILED'
assert sigmoid_activation(1000.0, 1.0, 0.0) == 1.0, '[TC01] sigmoid large positive FAILED'

# ---- TC02: get_neural_parameters returns tuple of correct length ----
params = get_neural_parameters()
assert len(params) == 17, '[TC02] param count FAILED'
assert all(isinstance(v, (float, int, np.ndarray)) for v in params), '[TC02] param types FAILED'

# ---- TC03: neural_mass_deriv returns finite array of shape (2,) ----
import numpy as np
y_test = np.array([0.2, 0.1])
dydt = neural_mass_deriv(0.0, y_test)
assert dydt.shape == (2,), '[TC03] deriv shape FAILED'
assert np.all(np.isfinite(dydt)), '[TC03] deriv not finite FAILED'

# ---- TC04: neural_mass_jacobian returns 2x2 matrix ----
import numpy as np
J = neural_mass_jacobian(np.array([0.3, 0.15]))
assert J.shape == (2, 2), '[TC04] jacobian shape FAILED'
assert np.all(np.isfinite(J)), '[TC04] jacobian not finite FAILED'

# ---- TC05: neural_oscillation_period returns positive finite value ----
T = neural_oscillation_period(linearized=True)
assert np.isfinite(T), '[TC05] period not finite FAILED'
assert T > 0, '[TC05] period non-positive FAILED'

# ---- TC06: compute_running_cost is non-negative ----
import numpy as np
y = np.array([0.5, 0.5])
cost = compute_running_cost(y, 1.0, np.array([0.6, 0.3]))
assert cost >= 0.0, '[TC06] running cost negative FAILED'
assert np.isfinite(cost), '[TC06] running cost not finite FAILED'

# ---- TC07: compute_terminal_cost is zero at target ----
import numpy as np
y_target = np.array([0.6, 0.3])
cost0 = compute_terminal_cost(y_target, y_target)
assert abs(cost0) < 1e-12, '[TC07] terminal cost at target not zero FAILED'

# ---- TC08: euler_maruyama output shape for deterministic case ----
import numpy as np
np.random.seed(42)
def drift_zero(t, y):
    return np.zeros(2)
def diff_zero(t, y):
    return np.zeros(2)
t_em, y_em = euler_maruyama(drift_zero, diff_zero, (0.0, 10.0), np.array([0.5, 0.5]), 100)
assert t_em.shape == (101,), '[TC08] time shape FAILED'
assert y_em.shape == (101, 2), '[TC08] state shape FAILED'
assert np.all(np.isfinite(y_em)), '[TC08] state not finite FAILED'

# ---- TC09: milstein_method output shape for deterministic case ----
import numpy as np
np.random.seed(42)
def drift_const(t, y):
    return np.array([0.1, 0.2])
def diff_const(t, y):
    return np.array([0.0, 0.0])
def diff_deriv_const(t, y):
    return np.array([0.0, 0.0])
t_mil, y_mil = milstein_method(drift_const, diff_const, diff_deriv_const, (0.0, 5.0), np.array([0.0, 0.0]), 200)
assert t_mil.shape == (201,), '[TC09] time shape FAILED'
assert y_mil.shape == (201, 2), '[TC09] state shape FAILED'
assert np.all(np.isfinite(y_mil)), '[TC09] state not finite FAILED'

# ---- TC10: stochastic_explicit_midpoint output shape for deterministic case ----
import numpy as np
np.random.seed(42)
def drift_lin(t, y):
    return -0.1 * y
def diff_lin(t, y):
    return np.array([0.0, 0.0])
t_sem, y_sem = stochastic_explicit_midpoint(drift_lin, diff_lin, (0.0, 10.0), np.array([1.0, 0.5]), 100)
assert t_sem.shape == (101,), '[TC10] time shape FAILED'
assert y_sem.shape == (101, 2), '[TC10] state shape FAILED'
assert np.all(np.isfinite(y_sem)), '[TC10] state not finite FAILED'

# ---- TC11: mean_square_stability_check reports stable for lambda=-10, mu=0, dt=0.01 ----
stable = mean_square_stability_check(-10.0, 0.0, 0.01)
assert stable, '[TC11] should be stable FAILED'

# ---- TC12: mean_square_stability_check reports unstable for lambda=10, mu=5, dt=0.5 ----
unstable = mean_square_stability_check(10.0, 5.0, 0.5)
assert not unstable, '[TC12] should be unstable FAILED'

# ---- TC13: tetrahedron_volume of regular tetrahedron edge=2^(1/3)*6^(1/6) ~ computed via formula ----
import numpy as np
verts_regular = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
vol = tetrahedron_volume(verts_regular)
assert vol > 0, '[TC13] volume non-positive FAILED'
assert abs(vol - 1.0/6.0) < 1e-10, '[TC13] regular tet volume FAILED'

# ---- TC14: bicg_solver solves simple diagonal system ----
import numpy as np
n_test = 10
A_diag = 2.0 * np.eye(n_test)
b_test = np.ones(n_test)
x, err, iters, flag = bicg_solver(A_diag, b_test, max_iter=100, tol=1e-10)
assert flag == 0, '[TC14] BiCG did not converge FAILED'
assert np.allclose(x, 0.5 * np.ones(n_test), atol=1e-8), '[TC14] BiCG solution incorrect FAILED'

# ---- TC15: rosenbrock_function at global minimum f(1,1)=0 ----
import numpy as np
f_min = rosenbrock_function(np.array([1.0, 1.0]))
assert abs(f_min) < 1e-12, '[TC15] rosenbrock at minimum not zero FAILED'

# ---- TC16: rosenbrock_function at non-minimum is positive ----
import numpy as np
f_other = rosenbrock_function(np.array([0.0, 0.0]))
assert f_other > 0, '[TC16] rosenbrock at (0,0) should be positive FAILED'

# ---- TC17: bisection_find_root on f(x)=x-0.5 in [0,1] ----
import numpy as np
def f_linear(x):
    return x - 0.5
a, b, it = bisection_find_root(f_linear, 0.0, 1.0, tol=1e-10)
root = (a + b) / 2.0
assert abs(root - 0.5) < 1e-9, '[TC17] bisection root incorrect FAILED'

# ---- TC18: safe_divide normal and division by zero ----
val_normal = safe_divide(10.0, 2.0)
assert val_normal == 5.0, '[TC18] safe_divide normal FAILED'
val_zero = safe_divide(10.0, 0.0, default=99.0)
assert val_zero == 99.0, '[TC18] safe_divide zero default FAILED'

# ---- TC19: assert_numeric_stability on finite vs NaN array ----
import numpy as np
assert assert_numeric_stability(np.array([1.0, 2.0, 3.0])), '[TC19] finite array should be stable FAILED'
assert not assert_numeric_stability(np.array([1.0, np.nan])), '[TC19] NaN array should not be stable FAILED'

# ---- TC20: nelder_mead_optimize on convex quadratic f(x)=(x-3)^2 ----
import numpy as np
np.random.seed(42)
def quad(x):
    return (x[0] - 3.0) ** 2
x_opt, n_feval = nelder_mead_optimize(quad, np.array([0.0]), tolerance=1e-8, max_feval=500)
assert abs(x_opt[0] - 3.0) < 0.02, '[TC20] nelder-mead optimum incorrect FAILED'
assert n_feval > 0, '[TC20] nelder-mead zero fevals FAILED'

# ---- TC21: linear_feedback_policy returns value in [-5, 5] ----
import numpy as np
u = linear_feedback_policy(np.array([2.0, 1.0]), np.array([0.3, 0.2]), x_eq=np.array([0.1, 0.1]))
assert -5.0 - 1e-10 <= u <= 5.0 + 1e-10, '[TC21] control out of bounds FAILED'
assert np.isfinite(u), '[TC21] control not finite FAILED'

# ---- TC22: pyramid_jaskowiec_rule weight sum equals 4/3 ----
import numpy as np
for p in [0, 2, 4, 6]:
    n, x, y, z, w = pyramid_jaskowiec_rule(p)
    w_sum = np.sum(w)
    assert abs(w_sum - 4.0/3.0) < 1e-10, f'[TC22] p={p} weight sum FAILED'

# ---- TC23: integrate_over_pyramid of unit function equals 4/3 ----
import numpy as np
def f_one(x_arr, y_arr, z_arr):
    return np.ones_like(x_arr)
val = integrate_over_pyramid(f_one, p=4)
assert abs(val - 4.0/3.0) < 1e-8, '[TC23] pyramid unit integral FAILED'

# ---- TC24: gauss_hermite_quad_1d E[X^2] with sigma=1 equals 1.0 ----
import numpy as np
def f_xsq(x):
    return x ** 2
val_gh = gauss_hermite_quad_1d(5, f_xsq, sigma=1.0)
assert abs(val_gh - 1.0) < 1e-10, '[TC24] gauss-hermite E[X^2] FAILED'

# ---- TC25: StateEncoder encode/decode consistency ----
import numpy as np
np.random.seed(42)
encoder = StateEncoder(20)
for idx in range(20):
    coded = encoder.encode(idx)
    decoded = encoder.decode(coded)
    assert idx == decoded, f'[TC25] encode/decode mismatch at {idx} FAILED'

# ---- TC26: metric_tensor euclidean returns identity matrix ----
import numpy as np
A_euc = metric_tensor(np.array([0.5, 0.5]), metric_type="euclidean")
assert np.allclose(A_euc, np.eye(2)), '[TC26] euclidean metric not identity FAILED'

# ---- TC27: check_environment returns dict with required keys ----
env = check_environment()
assert isinstance(env, dict), '[TC27] check_environment not dict FAILED'
assert 'numpy_version' in env, '[TC27] numpy_version missing FAILED'
assert 'float_info' in env, '[TC27] float_info missing FAILED'

# ---- TC28: estimate_convergence_rate on perfect power law h^0.5 ----
import numpy as np
h_test = np.array([0.01, 0.02, 0.04, 0.08])
err_test = 2.0 * h_test ** 0.5
p_est, logC, resid = estimate_convergence_rate(h_test, err_test)
assert abs(p_est - 0.5) < 0.01, '[TC28] convergence rate estimate FAILED'
assert resid < 1e-10, '[TC28] convergence fit residual too large FAILED'

# ---- TC29: lyapunov_exponential_decay_rate on exponential decay ----
import numpy as np
t_lyap = np.linspace(0, 10, 200)
y_decay = np.column_stack([np.exp(-0.5 * t_lyap), np.exp(-0.5 * t_lyap)])
rate = lyapunov_exponential_decay_rate(t_lyap, y_decay)
assert abs(rate - 0.5) < 0.05, '[TC29] lyapunov rate incorrect FAILED'
assert rate > 0, '[TC29] lyapunov rate non-positive FAILED'

# ---- TC30: regular_tetrahedral_mesh returns correct node count ----
import numpy as np
nodes, tets = regular_tetrahedral_mesh((np.zeros(3), np.ones(3)), n_per_dim=4)
assert nodes.shape == (64, 3), '[TC30] node count FAILED'
assert tets.shape[0] > 0, '[TC30] zero tets FAILED'
assert tets.shape[1] == 4, '[TC30] tet indexing FAILED'

# ---- TC31: compute_tet_quality returns value in [0, 1] ----
import numpy as np
verts_good = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
q = compute_tet_quality(verts_good)
assert 0.0 <= q <= 1.0, '[TC31] tet quality out of bounds FAILED'

# ---- TC32: monte_carlo_expectation on standard normal E[X1^2+X2^2]≈2 ----
import numpy as np
np.random.seed(42)
def sampler_mc(n, rng):
    return rng.standard_normal((n, 2))
def f_mc(x):
    return x[0]**2 + x[1]**2
mean_est, std_err = monte_carlo_expectation(f_mc, sampler_mc, n_samples=3000)
assert abs(mean_est - 2.0) < 0.2, '[TC32] MC expectation too far from 2.0 FAILED'
assert std_err > 0, '[TC32] MC std_err non-positive FAILED'

# ---- TC33: ActorCriticAgent initialization produces valid action ----
import numpy as np
np.random.seed(42)
agent = ActorCriticAgent(state_dim=2, n_rbf=8, rng=np.random.default_rng(seed=42))
a = agent.select_action(np.array([0.3, 0.2]))
assert -5.0 - 1e-10 <= a <= 5.0 + 1e-10, '[TC33] action out of bounds FAILED'
assert np.isfinite(a), '[TC33] action not finite FAILED'

# ---- TC34: serialize_state_trajectory / deserialize roundtrip ----
import numpy as np
t_orig = np.array([0.0, 0.5, 1.0])
y_orig = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
u_orig = np.array([0.0, 0.1, 0.2])
data = serialize_state_trajectory(t_orig, y_orig, u_orig)
t_rt, y_rt, u_rt = deserialize_state_trajectory(data)
assert np.allclose(t_rt, t_orig), '[TC34] time roundtrip FAILED'
assert np.allclose(y_rt, y_orig), '[TC34] state roundtrip FAILED'
assert np.allclose(u_rt, u_orig), '[TC34] control roundtrip FAILED'

# ---- TC35: himmelblau_function at known minimum (3,2) -> 0 ----
import numpy as np
f_himm = himmelblau_function(np.array([3.0, 2.0]))
assert abs(f_himm) < 1e-10, '[TC35] himmelblau at (3,2) not zero FAILED'

# ---- TC36: generate_brownian_path shape and length ----
import numpy as np
np.random.seed(42)
t_w, W = generate_brownian_path((0.0, 1.0), 100, dim=2)
assert t_w.shape == (101,), '[TC36] W time shape FAILED'
assert W.shape == (101, 2), '[TC36] W shape FAILED'
assert abs(W[0, 0]) < 1e-12, '[TC36] W(0) not zero FAILED'

# ---- TC37: compute_strong_error returns decreasing dt ----
import numpy as np
np.random.seed(42)
def f_sde(t, y):
    return -0.5 * y
def g_sde(t, y):
    return 0.3 * y
dt_vals, errors = compute_strong_error(f_sde, g_sde, 1.0, (0.0, 1.0), 512, [64, 128, 256], n_paths=200)
assert len(dt_vals) >= 2, '[TC37] too few dt values FAILED'
assert np.all(np.diff(dt_vals) < 0), '[TC37] dt not decreasing FAILED'

# ---- TC38: analyze_ms_stability_region output shapes ----
import numpy as np
L, MU, mask = analyze_ms_stability_region((-5.0, 1.0), (0.0, 3.0), dt=0.1, n_lambda=20, n_mu=20)
assert L.shape == (20, 20), '[TC38] L shape FAILED'
assert MU.shape == (20, 20), '[TC38] MU shape FAILED'
assert mask.dtype == bool, '[TC38] mask not boolean FAILED'

# ---- TC39: perform_stability_sweep returns correct shape matrix ----
import numpy as np
np.random.seed(42)
mat = perform_stability_sweep(None, [-3.0, -1.0], [0.5, 1.5], dt=0.1, n_paths=200, tmax=2.0)
assert mat.shape == (2, 2), '[TC39] stability matrix shape FAILED'

# ---- TC40: rosenbrock_gradient at minimum is zero ----
import numpy as np
grad_min = rosenbrock_gradient(np.array([1.0, 1.0]))
assert np.allclose(grad_min, np.zeros(2), atol=1e-10), '[TC40] gradient at min not zero FAILED'
