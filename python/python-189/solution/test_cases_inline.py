# ---- TC01: sine_integral(0.0) 边界值为 0 ----
assert abs(sine_integral(0.0)) < 1.0e-12, '[TC01] sine_integral(0) should be 0 FAILED'

# ---- TC02: sine_integral 在常用点返回有限值 ----
import numpy as np
for xv in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, -3.0]:
    si = sine_integral(xv)
    assert np.isfinite(si), f'[TC02] sine_integral({xv}) should be finite FAILED'

# ---- TC03: sine_integral 渐近收敛 (大 x 接近 pi/2) ----
si100 = sine_integral(100.0)
assert abs(si100 - np.pi / 2.0) < 0.2, '[TC03] sine_integral asymptotic convergence FAILED'

# ---- TC04: incomplete_beta 返回合法概率 [0,1] ----
prob, ier = incomplete_beta(0.5, 2.0, 3.0)
assert ier == 0, '[TC04] incomplete_beta should succeed (ier==0) FAILED'
assert 0.0 <= prob <= 1.0, '[TC04] incomplete_beta probability out of [0,1] FAILED'

# ---- TC05: incomplete_beta 边界: x=0 返回 0 ----
prob0, ier0 = incomplete_beta(0.0, 2.0, 3.0)
assert ier0 == 0, '[TC05] incomplete_beta at x=0 ier should be 0 FAILED'
assert abs(prob0) < 1.0e-12, '[TC05] incomplete_beta(0,2,3) should be 0 FAILED'

# ---- TC06: incomplete_beta 边界: x=1 返回 1 ----
prob1, ier1 = incomplete_beta(1.0, 2.0, 3.0)
assert ier1 == 0, '[TC06] incomplete_beta at x=1 ier should be 0 FAILED'
assert abs(prob1 - 1.0) < 1.0e-12, '[TC06] incomplete_beta(1,2,3) should be 1 FAILED'

# ---- TC07: beta_cdf 边界值 ----
assert abs(beta_cdf(0.0, 2.0, 3.0)) < 1.0e-12, '[TC07] beta_cdf(0,2,3) should be 0 FAILED'
assert abs(beta_cdf(1.0, 2.0, 3.0) - 1.0) < 1.0e-12, '[TC07] beta_cdf(1,2,3) should be 1 FAILED'

# ---- TC08: rref_solve 精确求解满秩线性系统 ----
import numpy as np
A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 8.0])
x_rref = rref_solve(A, b)
x_true = np.linalg.solve(A, b)
assert np.allclose(x_rref.flatten(), x_true, atol=1.0e-6), '[TC08] rref_solve solution mismatch FAILED'

# ---- TC09: rref_rank 正确计算矩阵秩 ----
import numpy as np
A_full = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 0.0]])
rk = rref_rank(A_full)
assert rk == 2, f'[TC09] rref_rank should be 2, got {rk} FAILED'

# ---- TC10: toeplitz_cholesky_lower 重构验证 ----
import numpy as np
n = 5
first_col = np.array([2.0, 0.5, 0.3, 0.2, 0.1])
T = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        T[i, j] = first_col[abs(i - j)]
L = toeplitz_cholesky_lower(n, T)
recon = L @ L.T
assert np.max(np.abs(recon - T)) < 1.0e-6, '[TC10] Toeplitz Cholesky reconstruction FAILED'

# ---- TC11: sawtooth_wave 输出在 [-1, 1] 范围内 ----
import numpy as np
for tv in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 10.0]:
    sw = sawtooth_wave(tv)
    assert -1.0 - 1.0e-12 <= sw <= 1.0 + 1.0e-12, f'[TC11] sawtooth_wave({tv})={sw} out of [-1,1] FAILED'

# ---- TC12: sawtooth_wave 周期为 1 (omega=2pi 时 t 周期为 1) ----
import numpy as np
sw0 = sawtooth_wave(0.0)
sw1 = sawtooth_wave(1.0)
assert abs(sw0 - sw1) < 1.0e-12, '[TC12] sawtooth_wave periodicity at t=0 and t=1 FAILED'

# ---- TC13: grazing_coupling 返回有限值 ----
import numpy as np
for x1v, x3v in [(0.0, 0.0), (1.0, 1.0), (-1.0, 2.0), (10.0, -10.0), (1000.0, 1000.0)]:
    gc = grazing_coupling(x1v, x3v)
    assert np.isfinite(gc), f'[TC13] grazing_coupling({x1v},{x3v}) should be finite FAILED'

# ---- TC14: brownian_motion 输出形状正确且起点为原点 ----
import numpy as np
np.random.seed(42)
n_steps, dim = 100, 3
traj = brownian_motion(n_steps, dim, sigma=1.0)
assert traj.shape == (n_steps, dim), f'[TC14] brownian_motion shape {traj.shape} != ({n_steps},{dim}) FAILED'
assert np.allclose(traj[0, :], 0.0, atol=1.0e-12), '[TC14] brownian_motion should start at origin FAILED'

# ---- TC15: ornstein_uhlenbeck_process 输出尺寸正确 ----
import numpy as np
np.random.seed(42)
n_ou, d_ou = 200, 2
ou = ornstein_uhlenbeck_process(n_ou, d_ou, theta=0.15, sigma=0.2, dt=0.01)
assert ou.shape == (n_ou, d_ou), f'[TC15] OU process shape {ou.shape} != ({n_ou},{d_ou}) FAILED'

# ---- TC16: gaussian_kernel_matrix 对称且对角线为 1 ----
import numpy as np
np.random.seed(42)
pts = np.random.randn(10, 3)
K = gaussian_kernel_matrix(pts, sigma=1.0)
assert np.allclose(K, K.T, atol=1.0e-12), '[TC16] gaussian_kernel_matrix should be symmetric FAILED'
assert np.allclose(np.diag(K), 1.0, atol=1.0e-12), '[TC16] gaussian_kernel_matrix diagonal should be 1 FAILED'

# ---- TC17: pca_vectors 输出形状正确 ----
import numpy as np
np.random.seed(42)
data = np.random.randn(10, 50)
V, vals, Psi = pca_vectors(data, 5)
assert V.shape == (10, 5), f'[TC17] pca_vectors V shape {V.shape} != (10,5) FAILED'
assert len(vals) == 50, f'[TC17] pca_vectors vals len {len(vals)} != 50 FAILED'
assert len(Psi) == 10, f'[TC17] pca_vectors Psi len {len(Psi)} != 10 FAILED'

# ---- TC18: pca_transform 输出维度正确 ----
import numpy as np
np.random.seed(42)
data = np.random.randn(10, 50)
V, vals, Psi = pca_vectors(data, 5)
sample = np.random.randn(10)
proj = pca_transform(sample, V, Psi)
assert proj.shape == (5,), f'[TC18] pca_transform output shape {proj.shape} != (5,) FAILED'

# ---- TC19: build_legendre_basis 输出形状正确 ----
import numpy as np
np.random.seed(42)
X = np.random.uniform(-1, 1, (2, 10))
B = build_legendre_basis(2, 2, X)
from math import comb
expected = comb(2 + 2, 2)
assert B.shape[0] == expected, f'[TC19] legendre basis shape[0] {B.shape[0]} != {expected} FAILED'
assert B.shape[1] == 10, f'[TC19] legendre basis shape[1] {B.shape[1]} != 10 FAILED'

# ---- TC20: compute_discounted_returns gamma=0 时仅第一步非零 ----
import numpy as np
rewards = [1.0, 2.0, 3.0]
returns_g0 = compute_discounted_returns(rewards, 0.0)
assert abs(returns_g0[0] - 1.0) < 1.0e-12, '[TC20] discounted return with gamma=0 FAILED'
assert abs(returns_g0[1] - 2.0) < 1.0e-12, '[TC20] discounted return with gamma=0 FAILED'

# ---- TC21: compute_discounted_returns gamma=1 时全为总和 ----
rewards2 = [1.0, 2.0, 3.0]
returns_g1 = compute_discounted_returns(rewards2, 1.0)
assert abs(returns_g1[0] - 6.0) < 1.0e-12, '[TC21] discounted return with gamma=1 FAILED'

# ---- TC22: trust_region_probability 返回 [0,1] 范围值 ----
import numpy as np
prob_tr = trust_region_probability(0.01, 10, 100)
assert 0.0 <= prob_tr <= 1.0, f'[TC22] trust_region_probability {prob_tr} out of [0,1] FAILED'

# ---- TC23: CosineAnnealingScheduler 输出在 [alpha_min, alpha_max] 中 ----
import numpy as np
sched = CosineAnnealingScheduler(alpha_max=0.01, alpha_min=1.0e-5, T_period=100)
for _ in range(200):
    a = sched.step()
    assert 1.0e-5 - 1.0e-12 <= a <= 0.01 + 1.0e-12, f'[TC23] scheduler alpha={a} out of bounds FAILED'

# ---- TC24: conjugate_gradient_solve 求解简单正定系统 ----
import numpy as np
A_mat = np.array([[4.0, 1.0], [1.0, 3.0]])
b_vec = np.array([1.0, 2.0])
x_cg = conjugate_gradient_solve(lambda v: A_mat @ v, b_vec, max_iter=50, damping=0.0)
x_direct = np.linalg.solve(A_mat, b_vec)
assert np.allclose(x_cg, x_direct, atol=1.0e-6), '[TC24] conjugate_gradient_solve mismatch FAILED'

# ---- TC25: ControlledNonlinearOscillator reset 返回 4 维观测 ----
import numpy as np
np.random.seed(42)
env = ControlledNonlinearOscillator(dt=0.01)
obs = env.reset()
assert obs.shape == (4,), f'[TC25] env observation shape {obs.shape} != (4,) FAILED'
assert np.all(np.isfinite(obs)), '[TC25] env observation should be finite FAILED'

# ---- TC26: ControlledNonlinearOscillator step 返回四元组 ----
import numpy as np
np.random.seed(42)
env = ControlledNonlinearOscillator(dt=0.01)
env.reset()
action = np.zeros(4)
obs, reward, done, info = env.step(action)
assert obs.shape == (4,), f'[TC26] step observation shape {obs.shape} != (4,) FAILED'
assert np.isfinite(reward), '[TC26] reward should be finite FAILED'
assert isinstance(done, bool), '[TC26] done should be bool FAILED'
assert 't' in info, '[TC26] info should contain t FAILED'

# ---- TC27: SpectralPolicyNetwork sample 输出在动作边界内 ----
import numpy as np
np.random.seed(42)
pn = SpectralPolicyNetwork(state_dim=4, action_dim=4, max_degree=2, action_bounds=(-2.0, 2.0))
state = np.zeros(4)
act = pn.sample(state)
assert act.shape == (4,), f'[TC27] policy sample shape {act.shape} != (4,) FAILED'
assert np.all(act >= -2.0) and np.all(act <= 2.0), '[TC27] policy sample out of bounds FAILED'

# ---- TC28: SpectralPolicyNetwork log_prob 数值有限 ----
import numpy as np
np.random.seed(42)
pn = SpectralPolicyNetwork(state_dim=4, action_dim=4, max_degree=2)
state = np.random.randn(4)
act = np.random.randn(4) * 0.5
lp = pn.log_prob(state, act)
assert np.isfinite(lp), f'[TC28] log_prob should be finite, got {lp} FAILED'

# ---- TC29: SpectralValueFunction predict 未拟合时返回 0 ----
import numpy as np
svf = SpectralValueFunction(state_dim=4, max_degree=2)
val = svf.predict(np.array([1.0, 0.0, 0.5, -0.5]))
assert abs(val) < 1.0e-12, f'[TC29] unfitted value function should return 0, got {val} FAILED'

# ---- TC30: lp_action_projection 无约束时等同于 clip ----
import numpy as np
raw = np.array([1.5, -3.0, 0.0, 2.5])
proj = lp_action_projection(raw, bounds=(-2.0, 2.0))
expected = np.clip(raw, -2.0, 2.0)
assert np.allclose(proj, expected), '[TC30] lp_action_projection simple clip FAILED'

# ---- TC31: multivariate_normal_distance_stats 返回正均值 ----
import numpy as np
np.random.seed(42)
mu, var = multivariate_normal_distance_stats(m=3, n_samples=2000)
assert mu > 0, f'[TC31] distance mean should be positive, got {mu} FAILED'
assert var > 0, f'[TC31] distance variance should be positive, got {var} FAILED'

# ---- TC32: check_trust_region 在 kl <= max_kl 时返回 True ----
assert check_trust_region(0.005, 0.01, 10, 100) == True, '[TC32] check_trust_region should be True FAILED'

# ---- TC33: rref_compute 返回的矩阵是阶梯形 (pivot=1) ----
import numpy as np
A_rref = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 2.0]])
ARREF, pivots = rref_compute(A_rref)
for col_idx in pivots:
    col = ARREF[:, col_idx]
    assert abs(max(abs(col)) - 1.0) < 1.0e-10, '[TC33] RREF pivot column should have max 1 FAILED'

# ---- TC34: bessel_spectral_filter 输出在 [0,1] 中 ----
import numpy as np
freqs = np.linspace(0, 10, 100)
resp = bessel_spectral_filter(freqs, n=0.0, k=2, kind=1, bandwidth=1.0)
assert np.all(resp >= 0.0) and np.all(resp <= 1.0 + 1.0e-12), '[TC34] bessel filter response out of [0,1] FAILED'
