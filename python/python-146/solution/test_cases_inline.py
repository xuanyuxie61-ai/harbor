# ---- TC01: HHNeuron 初始化默认值正确 ----
neuron = HHNeuron(dt=0.01)
assert neuron.V == -65.0, '[TC01] Initial V should be -65.0 FAILED'
assert 0.0 <= neuron.m <= 1.0, '[TC01] gate m out of bounds FAILED'
assert 0.0 <= neuron.h <= 1.0, '[TC01] gate h out of bounds FAILED'
assert 0.0 <= neuron.n <= 1.0, '[TC01] gate n out of bounds FAILED'

# ---- TC02: HHNeuron 非法 dt 抛出 ValueError ----
try:
    HHNeuron(dt=0.0)
    assert False, '[TC02] dt=0 should raise ValueError FAILED'
except ValueError:
    pass

# ---- TC03: HHNeuron step 零输入不应发放 ----
import numpy as np
neuron = HHNeuron(dt=0.01)
fired = neuron.step(0.0, I_syn=0.0, I_ext=0.0)
assert fired is False, '[TC03] step with zero input should not fire FAILED'
assert np.isfinite(neuron.V), '[TC03] V should be finite FAILED'

# ---- TC04: HHNeuron step 强电流注入应发放脉冲 ----
import numpy as np
np.random.seed(42)
neuron = HHNeuron(dt=0.01)
n_steps = int(50.0 / 0.01)
spike_count = 0
for k in range(n_steps):
    t = k * 0.01
    fired = neuron.step(t, I_ext=10.0)
    if fired:
        spike_count += 1
assert spike_count > 0, '[TC04] Should fire spikes under strong current FAILED'

# ---- TC05: HHNeuron 门控变量始终在 [0,1] 内 ----
import numpy as np
np.random.seed(42)
neuron = HHNeuron(dt=0.01)
for k in range(5000):
    t = k * 0.01
    neuron.step(t, I_ext=12.0)
    assert 0.0 <= neuron.m <= 1.0, '[TC05] gate m out of [0,1] FAILED'
    assert 0.0 <= neuron.h <= 1.0, '[TC05] gate h out of [0,1] FAILED'
    assert 0.0 <= neuron.n <= 1.0, '[TC05] gate n out of [0,1] FAILED'

# ---- TC06: HHNeuron 速率函数 alpha_m 在 V=-40 处鲁棒 ----
val = HHNeuron._alpha_m(-40.0)
assert np.isfinite(val), '[TC06] alpha_m(-40) should be finite FAILED'
assert val > 0.0, '[TC06] alpha_m(-40) should be positive FAILED'

# ---- TC07: NeuronPopulation 初始化神经元数量正确 ----
pop = NeuronPopulation(N_exc=10, N_inh=5, dt=0.01, p_conn=0.2)
assert pop.N == 15, '[TC07] total neuron count should be 15 FAILED'
assert pop.N_exc == 10, '[TC07] exc count should be 10 FAILED'
assert pop.N_inh == 5, '[TC07] inh count should be 5 FAILED'

# ---- TC08: NeuronPopulation simulate 输出形状正确、值有限 ----
import numpy as np
np.random.seed(42)
pop = NeuronPopulation(N_exc=4, N_inh=2, dt=0.01, p_conn=0.3)
I_ext = np.array([8.0, 8.0, 8.0, 8.0, 5.0, 5.0])
vt, sr = pop.simulate(T_total=10.0, I_ext_per_neuron=I_ext)
assert vt.shape == (6, 1000), '[TC08] voltage_trace shape mismatch FAILED'
assert sr.shape == (6, 1000), '[TC08] spike_raster shape mismatch FAILED'
assert np.all(np.isfinite(vt)), '[TC08] voltage_trace has non-finite values FAILED'

# ---- TC09: DG1DNeuralCable 初始化网格尺寸正确 ----
cable = DG1DNeuralCable(xL=0.0, xR=10.0, K=20, Np=4, dt=0.001, epsilon=0.05)
assert cable.x.shape == (4, 20), '[TC09] grid shape should be (4,20) FAILED'

# ---- TC10: DG1DNeuralCable simulate 输出形状正确且值有限 ----
import numpy as np
cable = DG1DNeuralCable(xL=0.0, xR=5.0, K=10, Np=3, dt=0.001, epsilon=0.05)
u0 = np.zeros((cable.Np, cable.K))
for k in range(cable.K):
    for i in range(cable.Np):
        x = cable.x[i, k]
        u0[i, k] = 15.0 * np.exp(-((x - 1.0) ** 2) / 0.5)
u_final, history = cable.simulate(u0, T_final=1.0)
assert u_final.shape == (3, 10), '[TC10] final solution shape mismatch FAILED'
assert np.all(np.isfinite(u_final)), '[TC10] final solution has non-finite values FAILED'

# ---- TC11: polynomial_multiply_convolution 与 numpy.convolve 一致 ----
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.5, 1.0])
c = polynomial_multiply_convolution(a, b)
expected = np.convolve(a, b)
assert np.max(np.abs(c - expected)) < 1e-12, '[TC11] convolution mismatch FAILED'

# ---- TC12: rational_knapsack_encoding 基本背包求解 ----
import numpy as np
profits = np.array([10.0, 20.0, 30.0])
weights = np.array([2.0, 3.0, 5.0])
x, mass, profit = rational_knapsack_encoding(profits, weights, budget=5.0)
assert 0.0 <= mass <= 5.0 + 1e-10, '[TC12] mass should be within budget FAILED'
assert profit > 0.0, '[TC12] profit should be positive FAILED'

# ---- TC13: AlphaSynapse kernel 行为正确 ----
import numpy as np
syn = AlphaSynapse(tau_s=2.0)
t_vals = np.array([-1.0, 0.0, 1.0, 2.0, 10.0])
k = syn.kernel(t_vals)
assert k[0] == 0.0, '[TC13] kernel at t=-1 should be 0 FAILED'
assert k[1] == 0.0, '[TC13] kernel at t=0 should be 0 FAILED'
assert k[2] > 0.0, '[TC13] kernel at t=1 should be positive FAILED'
assert k[-1] < k[2], '[TC13] kernel should decay after peak FAILED'

# ---- TC14: AlphaSynapse convolve_spikes 输出形状与有限性 ----
import numpy as np
syn = AlphaSynapse(tau_s=2.0)
t_grid = np.linspace(0, 100, 1000)
spike_times = [10.0, 30.0, 60.0]
weights = [1.0, 0.8, 0.5]
s = syn.convolve_spikes(spike_times, weights, t_grid)
assert len(s) == 1000, '[TC14] output length mismatch FAILED'
assert np.all(np.isfinite(s)), '[TC14] output has non-finite values FAILED'

# ---- TC15: SpikePatternAnalyzer pattern_capacity 非负 ----
analyzer = SpikePatternAnalyzer(n_bins=8)
capacity = analyzer.pattern_capacity()
assert capacity > 0.0, '[TC15] pattern capacity should be positive FAILED'

# ---- TC16: connected_spike_patterns_1d 已知公式验证 ----
n = 5
result = connected_spike_patterns_1d(n)
expected = n * (n + 1) // 2
assert result == expected, '[TC16] 1D connected patterns formula mismatch FAILED'

# ---- TC17: polyomino_enumerate_fixed 已知值 ----
count = polyomino_enumerate_fixed(0)
assert count == 1, '[TC17] order 0 polyomino count should be 1 FAILED'
count2 = polyomino_enumerate_fixed(4)
assert count2 == 19, '[TC17] order 4 polyomino count should be 19 FAILED'

# ---- TC18: r8col_sorted_tol_unique 去重正确 ----
import numpy as np
patterns = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 0]]).T
unique, uniq_num = r8col_sorted_tol_unique(patterns, tol=0.0)
assert uniq_num <= 4, '[TC18] unique num should not exceed input FAILED'
assert uniq_num >= 1, '[TC18] unique num should be at least 1 FAILED'

# ---- TC19: pattern_clustering 返回正确标签数量 ----
import numpy as np
np.random.seed(42)
patterns = np.random.randint(0, 2, size=(6, 20)).astype(float)
labels = pattern_clustering(patterns, max_distance=2)
assert len(labels) == 20, '[TC19] labels count mismatch FAILED'
assert len(np.unique(labels)) >= 1, '[TC19] should have at least 1 cluster FAILED'

# ---- TC20: HermiteInterpolator 构造与求值不崩溃、输出有限 ----
import numpy as np
t_nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
v_nodes = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
dv_nodes = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
interp = HermiteInterpolator(t_nodes, v_nodes, dv_nodes)
val = interp.evaluate(np.array([0.5, 1.5, 2.5, 3.5]))
assert np.all(np.isfinite(val)), '[TC20] interpolated values should be finite FAILED'
assert len(val) == 4, '[TC20] output length should match query length FAILED'

# ---- TC21: StabilityAnalyzer rk4_amplification_factor z=0 => 1 ----
import numpy as np
r = StabilityAnalyzer.rk4_amplification_factor(0.0 + 0j)
assert np.abs(r - 1.0) < 1e-12, '[TC21] R(0) should equal 1 FAILED'

# ---- TC22: StabilityAnalyzer max_stable_timestep 返回正值 ----
dt_max = StabilityAnalyzer.max_stable_timestep(lambda_max=50.0, method='rk4')
assert dt_max > 0.0, '[TC22] max stable timestep should be positive FAILED'

# ---- TC23: CorticalGrid connection_probability 对称性 ----
import numpy as np
grid = CorticalGrid(nx=4, ny=4)
p_ij = grid.connection_probability(0, 5, p0=0.4, sigma=0.3)
p_ji = grid.connection_probability(5, 0, p0=0.4, sigma=0.3)
assert np.abs(p_ij - p_ji) < 1e-12, '[TC23] connection probability should be symmetric FAILED'

# ---- TC24: CorticalGrid distance_statistics 均值正值 ----
import numpy as np
grid = CorticalGrid(nx=5, ny=5, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
dmu, dvar, distances = grid.distance_statistics(n_samples=500)
assert dmu > 0.0, '[TC24] mean distance should be positive FAILED'
assert dvar > 0.0, '[TC24] distance variance should be positive FAILED'

# ---- TC25: LogNormalSynapse pdf 输出非负 ----
import numpy as np
model = LogNormalSynapse(mu=0.0, sigma=0.5)
w = np.array([0.5, 1.0, 2.0])
pdf_vals = model.pdf(w)
assert np.all(pdf_vals >= 0.0), '[TC25] pdf should be non-negative FAILED'
assert np.all(np.isfinite(pdf_vals)), '[TC25] pdf should be finite FAILED'

# ---- TC26: LogNormalSynapse 均值/方差公式 ----
model = LogNormalSynapse(mu=0.0, sigma=0.5)
th_mean = model.mean()
th_var = model.variance()
assert th_mean > 0.0, '[TC26] theoretical mean should be positive FAILED'
assert th_var > 0.0, '[TC26] theoretical variance should be positive FAILED'

# ---- TC27: normalize_weights_multiplicative 目标和正确 ----
import numpy as np
raw = np.array([0.2, 0.3, 0.5])
norm = normalize_weights_multiplicative(raw, target_sum=1.0)
assert np.abs(np.sum(norm) - 1.0) < 1e-12, '[TC27] normalized sum should be 1.0 FAILED'
assert np.all(norm >= 0.0), '[TC27] normalized weights should be non-negative FAILED'

# ---- TC28: EthierNavierStokes evaluate 返回有限值 ----
import numpy as np
ns = EthierNavierStokes()
u, v, w, p = ns.evaluate(0.5, 0.5, 0.5, 0.05)
assert np.isfinite(u), '[TC28] u should be finite FAILED'
assert np.isfinite(v), '[TC28] v should be finite FAILED'
assert np.isfinite(w), '[TC28] w should be finite FAILED'
assert np.isfinite(p), '[TC28] p should be finite FAILED'

# ---- TC29: KeastTetrahedronRule integrate 已知解析解 ----
import numpy as np
keast = KeastTetrahedronRule(rule_id=4)
verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
result = keast.integrate(lambda x, y, z: x + y + z, verts)
assert np.abs(result - 0.125) < 1e-6, '[TC29] tetrahedron integral should be 0.125 FAILED'

# ---- TC30: NeuralVolumeIntegral integrate_region 返回标量 ----
tet1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
tet2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
vol = NeuralVolumeIntegral()
total = vol.integrate_region([tet1, tet2], V_membrane=-65.0)
assert np.isfinite(total), '[TC30] volume integral should be finite FAILED'
assert isinstance(total, float), '[TC30] volume integral should be scalar FAILED'

# ---- TC31: LogNormalSynapse sample 固定种子可复现 ----
import numpy as np
model = LogNormalSynapse(mu=0.0, sigma=0.5)
np.random.seed(42)
s1 = model.sample(size=100)
np.random.seed(42)
s2 = model.sample(size=100)
assert np.allclose(s1, s2), '[TC31] samples with same seed should be identical FAILED'

# ---- TC32: r8col_sorted_tol_unique 容差去重 ----
import numpy as np
patterns = np.array([[1.0, 1.0, 1.01], [0.0, 0.0, 0.0]]).T
unique, uniq_num = r8col_sorted_tol_unique(patterns, tol=0.1)
assert uniq_num < 3, '[TC32] tolerance dedup should reduce count FAILED'

# ---- TC33: connected_spike_patterns_2d 返回列表 ----
import numpy as np
counts = connected_spike_patterns_2d(3, 3, max_order=5)
assert len(counts) >= 1, '[TC33] 2D connected patterns should return non-empty list FAILED'
assert all(isinstance(c, int) for _, c in counts), '[TC33] all counts should be int FAILED'

# ---- TC34: SignalReconstructor reconstruction_quality 返回字典 ----
import numpy as np
np.random.seed(42)
spike_times = np.array([2.0, 5.0, 10.0, 15.0, 20.0])
t_grid = np.linspace(0, 25, 200)
signal_true = np.sin(2.0 * np.pi * 0.1 * t_grid)
recon = SignalReconstructor(spike_times)
metrics = recon.reconstruction_quality(t_grid, signal_true)
assert 'MSE' in metrics, '[TC34] metrics should contain MSE FAILED'
assert 'SNR_dB' in metrics, '[TC34] metrics should contain SNR_dB FAILED'
assert 'Correlation' in metrics, '[TC34] metrics should contain Correlation FAILED'
assert np.isfinite(metrics['MSE']), '[TC34] MSE should be finite FAILED'

# ---- TC35: HermiteInterpolator evaluate_derivative 求值有限 ----
import numpy as np
t_nodes = np.array([0.0, 1.0, 2.0, 3.0])
v_nodes = np.array([0.0, 1.0, 0.0, 1.0])
dv_nodes = np.array([1.0, 0.0, -1.0, 0.5])
interp = HermiteInterpolator(t_nodes, v_nodes, dv_nodes)
deriv = interp.evaluate_derivative(np.array([1.5]))
assert np.isfinite(deriv[0]), '[TC35] derivative should be finite FAILED'

# ---- TC36: StabilityAnalyzer neuron_linearized_eigenvalue 返回负值 ----
import numpy as np
lam = StabilityAnalyzer.neuron_linearized_eigenvalue(
    V=-65.0, m=0.05, h=0.6, n=0.32
)
assert lam < 0.0, '[TC36] eigenvalue should be negative at rest FAILED'

# ---- TC37: CorticalGrid spatial_receptive_field 输出非负 ----
import numpy as np
grid = CorticalGrid(nx=4, ny=4)
rf = grid.spatial_receptive_field(i=5, sigma_rf=0.3)
assert len(rf) == 16, '[TC37] receptive field length mismatch FAILED'
assert np.all(rf >= 0.0), '[TC37] receptive field weights should be non-negative FAILED'

# ---- TC38: SynapticWeightSDE simulate_trajectory 输出长度正确 ----
import numpy as np
np.random.seed(42)
sde = SynapticWeightSDE(mu=0.0, sigma=0.3, theta=0.05, dt=0.01)
traj = sde.simulate_trajectory(w0=1.0, T_total=10.0)
assert len(traj) == 1000, '[TC38] trajectory length should be 1000 FAILED'
assert np.all(traj > 0.0), '[TC38] all trajectory values should be positive FAILED'

# ---- TC39: LogNormalSynapse sample_mean_variance 经验估计 ----
import numpy as np
np.random.seed(42)
model = LogNormalSynapse(mu=0.0, sigma=0.5)
emp_mean, emp_var = model.sample_mean_variance(n_samples=10000)
th_mean = model.mean()
assert np.abs(emp_mean - th_mean) < 0.1, '[TC39] empirical mean should approximate theoretical mean FAILED'
assert emp_var > 0.0, '[TC39] empirical variance should be positive FAILED'

# ---- TC40: EthierNavierStokes vorticity 返回有限值 ----
import numpy as np
ns = EthierNavierStokes()
ox, oy, oz = ns.vorticity(0.3, 0.3, 0.3, 0.0)
assert np.isfinite(ox), '[TC40] omega_x should be finite FAILED'
assert np.isfinite(oy), '[TC40] omega_y should be finite FAILED'
assert np.isfinite(oz), '[TC40] omega_z should be finite FAILED'
