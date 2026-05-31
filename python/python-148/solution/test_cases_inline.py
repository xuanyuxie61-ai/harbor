# ---- TC01: clenshaw_chebyshev_eval — T_0(x) = 1, single coefficient ----
from utils import clenshaw_chebyshev_eval
val = clenshaw_chebyshev_eval(0.5, [3.0])
assert abs(val - 3.0) < 1e-12, '[TC01] clenshaw_chebyshev_eval T_0(0.5)=3 FAILED'

# ---- TC02: clenshaw_chebyshev_eval — T_1(x) = x ----
val = clenshaw_chebyshev_eval(0.5, [0.0, 1.0])
assert abs(val - 0.5) < 1e-12, '[TC02] clenshaw_chebyshev_eval T_1(0.5)=0.5 FAILED'

# ---- TC03: clenshaw_chebyshev_eval — T_2(x) = 2x^2 - 1 ----
val = clenshaw_chebyshev_eval(0.5, [0.0, 0.0, 1.0])
expected = 2.0 * 0.5 ** 2 - 1.0  # = -0.5
assert abs(val - expected) < 1e-10, '[TC03] clenshaw_chebyshev_eval T_2(0.5)=-0.5 FAILED'

# ---- TC04: clenshaw_chebyshev_eval — sum of 3 T_0 + 2 T_1 + 1 T_2 at x=0 ----
val = clenshaw_chebyshev_eval(0.0, [3.0, 2.0, 1.0])
expected = 3.0 * 1.0 + 2.0 * 0.0 + 1.0 * (-1.0)  # = 2.0
assert abs(val - expected) < 1e-10, '[TC04] clenshaw_chebyshev_eval combo at x=0 FAILED'

# ---- TC05: sawtooth_wave — at t=0, value = -pi/omega * amplitude ----
from utils import sawtooth_wave
import numpy as np
omega_test = 2.0
amp_test = 1.5
expected_saw = amp_test * (np.mod(0 + np.pi / omega_test, 2 * np.pi / omega_test) - np.pi / omega_test)
val = sawtooth_wave(0.0, omega_test, amp_test)
assert abs(val - expected_saw) < 1e-12, '[TC05] sawtooth_wave at t=0 FAILED'

# ---- TC06: sawtooth_wave — periodicity: s(t + period) = s(t) ----
omega_test = 2.0
period = 2.0 * np.pi / omega_test
v1 = sawtooth_wave(0.3, omega_test, 1.0)
v2 = sawtooth_wave(0.3 + period, omega_test, 1.0)
assert abs(v1 - v2) < 1e-12, '[TC06] sawtooth_wave periodicity FAILED'

# ---- TC07: sigmoid_activation — S(0) = 0.5 at theta=0, sigma=1 ----
from utils import sigmoid_activation
val = sigmoid_activation(0.0, theta=0.0, sigma=1.0)
assert abs(val - 0.5) < 1e-12, '[TC07] sigmoid S(0)=0.5 FAILED'

# ---- TC08: sigmoid_activation — large positive -> 1 ----
val = sigmoid_activation(20.0, theta=0.0, sigma=1.0)
assert val > 0.9999, '[TC08] sigmoid S(20)~1 FAILED'

# ---- TC09: sigmoid_activation — large negative -> 0 ----
val = sigmoid_activation(-20.0, theta=0.0, sigma=1.0)
assert val < 1e-6, '[TC09] sigmoid S(-20)~0 FAILED'

# ---- TC10: softplus — softplus(0) = log(2) ----
from utils import softplus
val = softplus(np.array([0.0]))[0]
assert abs(val - np.log(2.0)) < 1e-10, '[TC10] softplus(0)=log(2) FAILED'

# ---- TC11: rk4_step — y' = y, y(0) = 1, h = 0.1, known approx ----
from utils import rk4_step
def exp_ode(t, y):
    return y
val = rk4_step(exp_ode, 0.0, np.array(1.0), 0.1)
expected = 1.0 + 0.1 + 0.1 ** 2 / 2 + 0.1 ** 3 / 6 + 0.1 ** 4 / 24  # Taylor for exp(0.1) ~= 1.105170833...
assert abs(float(val) - np.exp(0.1)) < 1e-6, '[TC11] rk4 exp FAILED'

# ---- TC12: gauss_legendre_nodes_weights — sum of weights = 2 ----
from utils import gauss_legendre_nodes_weights
x, w = gauss_legendre_nodes_weights(10)
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC12] GL weights sum=2 FAILED'

# ---- TC13: gauss_legendre_nodes_weights — nodes in [-1, 1] ----
x, w = gauss_legendre_nodes_weights(15)
assert np.all(x >= -1.0) and np.all(x <= 1.0), '[TC13] GL nodes in [-1,1] FAILED'

# ---- TC14: sparse_adjacency_to_laplacian — simple 2x2 ----
from utils import sparse_adjacency_to_laplacian
A = np.array([[0.0, 1.0], [1.0, 0.0]])
L = sparse_adjacency_to_laplacian(A)
expected_L = np.array([[1.0, -1.0], [-1.0, 1.0]])
assert np.allclose(L, expected_L, atol=1e-12), '[TC14] laplacian 2x2 FAILED'

# ---- TC15: softmax_stable — sums to 1 ----
from utils import softmax_stable
v = softmax_stable(np.array([1.0, 2.0, 3.0]))
assert abs(np.sum(v) - 1.0) < 1e-12, '[TC15] softmax sum=1 FAILED'

# ---- TC16: safe_log1p_exp — positive large x ~ x ----
from utils import safe_log1p_exp
val = safe_log1p_exp(np.array([50.0]))[0]
assert abs(val - 50.0) < 1e-8, '[TC16] safe_log1p_exp(50)~50 FAILED'

# ---- TC17: point_in_triangle_2d — point inside ----
from electrode_sampling import point_in_triangle_2d
A = np.array([0.0, 0.0])
B = np.array([2.0, 0.0])
C = np.array([0.0, 2.0])
P = np.array([0.5, 0.5])
assert point_in_triangle_2d(A, B, C, P) == True, '[TC17] point inside triangle FAILED'

# ---- TC18: point_in_triangle_2d — point outside ----
P_out = np.array([2.0, 2.0])
assert point_in_triangle_2d(A, B, C, P_out) == False, '[TC18] point outside triangle FAILED'

# ---- TC19: hexagonal_grid_points — center point present ----
from electrode_sampling import hexagonal_grid_points
pts = hexagonal_grid_points(center=[0.0, 0.0], radius=1.0, n_layers=3)
assert np.any(np.all(np.abs(pts - np.array([0.0, 0.0])) < 1e-12, axis=1)), '[TC19] hex grid has center FAILED'

# ---- TC20: mexican_hat_kernel_2d — near excitation (positive) ----
from neural_field_solver import mexican_hat_kernel_2d
val_near = mexican_hat_kernel_2d(0.0, 0.0, sigma_e=1.0, sigma_i=2.0, A_e=1.0, A_i=0.5)
assert val_near > 0, '[TC20] mexican hat near positive FAILED'

# ---- TC21: mexican_hat_kernel_2d — far inhibition (negative) ----
val_far = mexican_hat_kernel_2d(5.0, 0.0, sigma_e=0.1, sigma_i=0.5, A_e=1.0, A_i=0.8)
assert val_far < 0, '[TC21] mexican hat far negative FAILED'

# ---- TC22: logarithmic_norm — mu_2 of zero matrix = 0 ----
from stability_and_roots import logarithmic_norm
Z = np.zeros((3, 3), dtype=float)
mu = logarithmic_norm(Z, p=2)
assert abs(mu - 0.0) < 1e-12, '[TC22] mu_2(zero)=0 FAILED'

# ---- TC23: logarithmic_norm — mu_2 of negative identity = -1 ----
mu_negI = logarithmic_norm(-np.eye(4), p=2)
assert abs(mu_negI + 1.0) < 1e-12, '[TC23] mu_2(-I)=-1 FAILED'

# ---- TC24: cauchy_polynomial_root_bound — z^2 - 5z + 6 => roots 2, 3 => bound ~5 ----
from stability_and_roots import cauchy_polynomial_root_bound
bound = cauchy_polynomial_root_bound(np.array([1.0, -5.0, 6.0]))
assert bound >= 3.0, '[TC24] Cauchy bound >= 3 FAILED'

# ---- TC25: polynomial_from_matrix — identity matrix -> (z-1)^n ----
from stability_and_roots import polynomial_from_matrix
coeffs_id = polynomial_from_matrix(np.eye(3))
# Coefficients should be for (z-1)^3 = z^3 -3z^2 +3z -1
expected_coeffs = np.array([1.0, -3.0, 3.0, -1.0])
assert np.allclose(coeffs_id, expected_coeffs, atol=1e-10), '[TC25] char poly of I FAILED'

# ---- TC26: generate_square_grid — correct shape ----
from neural_field_solver import generate_square_grid
X, Y = generate_square_grid(xlim=(-1, 1), ylim=(-1, 1), nx=32, ny=32, centering='cell')
assert X.shape == (32, 32) and Y.shape == (32, 32), '[TC26] square grid shape FAILED'

# ---- TC27: generate_padua_points — correct count for n=3 ----
from spectral_signal_analysis import generate_padua_points
pts = generate_padua_points(3)
expected_count = (3 + 1) * (3 + 2) // 2  # = 10
assert len(pts) >= expected_count, '[TC27] Padua point count FAILED'

# ---- TC28: EIOscillator — simulate returns correct shape ----
from neural_mass_ode import EIOscillator
import numpy as np
np.random.seed(42)
osc = EIOscillator(omega=2.0 * np.pi * 6.0)
t, state = osc.simulate(E0=0.1, I0=0.05, t_span=(0.0, 0.1), dt=0.001)
assert t.shape[0] == state.shape[0], '[TC28] EIOscillator t/state mismatch FAILED'
assert state.shape[1] == 2, '[TC28] EIOscillator state cols FAILED'

# ---- TC29: EIOscillator — compute_lfp returns correct length ----
np.random.seed(42)
osc2 = EIOscillator(omega=2.0 * np.pi * 6.0)
t2, state2 = osc2.simulate(E0=0.1, I0=0.05, t_span=(0.0, 0.1), dt=0.001)
lfp = osc2.compute_lfp(state2, k_E=1.0, k_I=1.5, noise_std=0.02, dt=0.001)
assert len(lfp) == len(t2), '[TC29] LFP length mismatch FAILED'

# ---- TC30: BrainConnectomeGraph — fiedler value > 0 for connected graph ----
from connectome_topology import BrainConnectomeGraph
import numpy as np
np.random.seed(42)
graph = BrainConnectomeGraph(n_regions=30, connection_prob=0.15, weight_dist='lognormal', random_state=42)
fv = graph.compute_fiedler_value()
assert fv > 1e-10, '[TC30] Fiedler value > 0 FAILED'

# ---- TC31: ChebyshevSpectrumAnalyzer — analyze on constant signal ----
from spectral_signal_analysis import ChebyshevSpectrumAnalyzer
import numpy as np
np.random.seed(42)
analyzer = ChebyshevSpectrumAnalyzer(n_modes=32)
t_sig = np.linspace(0, 1, 200)
const_sig = np.ones(200) * 3.0
result = analyzer.analyze(const_sig, t_min=0.0, t_max=1.0)
assert result['dc_component'] > 0, '[TC31] Chebyshev DC component > 0 FAILED'
assert np.isfinite(result['dc_component']), '[TC31] Chebyshev DC component NaN/Inf FAILED'

# ---- TC32: GaussLegendreSignalIntegrator — signal_moments on constant ----
from spectral_signal_analysis import GaussLegendreSignalIntegrator
import numpy as np
np.random.seed(42)
integrator = GaussLegendreSignalIntegrator(n_points=32)
t_sig = np.linspace(0, 1, 100)
const_sig = np.ones(100) * 5.0
moments = integrator.signal_moments(const_sig, t_sig)
assert abs(moments['mean'] - 5.0) < 1e-8, '[TC32] GL moments mean=5 FAILED'
assert abs(moments['std']) < 1e-8, '[TC32] GL moments std~0 FAILED'
assert abs(moments['skewness']) < 1e-8, '[TC32] GL moments skew~0 FAILED'

# ---- TC33: CorticalSurfaceGeometry — surface_height at origin ----
from electrode_sampling import CorticalSurfaceGeometry
geom = CorticalSurfaceGeometry(curvature_radius=80.0, patch_radius=5.0)
h = geom.surface_height(0.0, 0.0)
assert abs(h) < 1e-12, '[TC33] surface_height at origin = 0 FAILED'

# ---- TC34: CorticalSurfaceGeometry — surface_height at patch edge ----
h_edge = geom.surface_height(5.0, 0.0)
assert h_edge > 0, '[TC34] surface_height at edge > 0 FAILED'

# ---- TC35: main() — integration test runs without error, returns dict with all required keys ----
import numpy as np
np.random.seed(148)
results = main()
assert isinstance(results, dict), '[TC35] main returns dict FAILED'
required_keys = ['signal_summary', 'stability_analysis', 'connectome_analysis',
                 'decoding_performance', 'electrode_geometry']
for key in required_keys:
    assert key in results, f'[TC35] results missing key {key} FAILED'
sig = results['signal_summary']
assert sig['n_electrodes'] > 0, '[TC35] n_electrodes > 0 FAILED'
assert sig['n_timepoints'] > 0, '[TC35] n_timepoints > 0 FAILED'
stab = results['stability_analysis']
assert isinstance(stab['is_stable'], (bool, np.bool_)), '[TC35] is_stable is bool FAILED'
assert 'mu_1' in stab, '[TC35] mu_1 present FAILED'
assert 'mu_2' in stab, '[TC35] mu_2 present FAILED'
assert 'mu_inf' in stab, '[TC35] mu_inf present FAILED'
perf = results['decoding_performance']
assert len(perf['test_errors']) == 10, '[TC35] 10 test errors FAILED'
assert isinstance(perf['mean_decoding_error'], float), '[TC35] mean decoding error float FAILED'

# ---- TC36: repr return type and finite check — main() result values are finite ----
for key in required_keys:
    if key == 'signal_summary':
        assert np.isfinite(sig['lfp_peak_amplitude']), '[TC36] lfp_peak finite FAILED'
        assert np.isfinite(sig['field_peak_amplitude']), '[TC36] field_peak finite FAILED'
    if key == 'stability_analysis':
        assert np.isfinite(stab['cauchy_root_bound']), '[TC36] root_bound finite FAILED'
    if key == 'electrode_geometry':
        assert np.isfinite(results['electrode_geometry']['spatial_coverage']), '[TC36] spatial_coverage finite FAILED'

# ---- TC37: point_in_polygon_2d — point inside square ----
from electrode_sampling import point_in_polygon_2d
square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
assert point_in_polygon_2d(square, np.array([1.0, 1.0])) == True, '[TC37] point inside square FAILED'

# ---- TC38: point_in_polygon_2d — point outside square ----
assert point_in_polygon_2d(square, np.array([3.0, 3.0])) == False, '[TC38] point outside square FAILED'

# ---- TC39: barycentric_lagrange_interpolate — exact recovery at nodes ----
from utils import barycentric_lagrange_interpolate
import numpy as np
x_nodes = np.array([-1.0, 0.0, 1.0])
y_nodes = np.array([2.0, -1.0, 3.0])
x_eval = np.array([-1.0, 0.0, 1.0])
vals = barycentric_lagrange_interpolate(x_nodes, y_nodes, x_eval)
assert np.allclose(vals, y_nodes, atol=1e-10), '[TC39] barycentric exact FAILED'
