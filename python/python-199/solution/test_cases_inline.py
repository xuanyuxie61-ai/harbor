# ---- TC01: verify_sorted 对有序列表返回True ----
sorted_list = [(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)]
assert verify_sorted(sorted_list, 0) == True, '[TC01] verify_sorted on sorted list FAILED'

# ---- TC02: verify_sorted 对无序列表返回False ----
unsorted_list = [(5.0,), (1.0,), (3.0,), (2.0,), (4.0,)]
assert verify_sorted(unsorted_list, 0) == False, '[TC02] verify_sorted on unsorted list FAILED'

# ---- TC03: verify_sorted 空列表返回True ----
assert verify_sorted([], 0) == True, '[TC03] verify_sorted on empty list FAILED'

# ---- TC04: ReplacementSelection 生成归并段数量合理 ----
import random
random.seed(42)
test_records = [(random.random(), i) for i in range(200)]
rs = ReplacementSelection(memory_capacity=30)
runs = rs.generate_runs(test_records, 0)
assert len(runs) > 0, '[TC04] ReplacementSelection generated no runs FAILED'
# 每个段内部应有序
for run in runs:
    for j in range(1, len(run)):
        assert run[j][0] >= run[j-1][0], '[TC04] Run is not sorted internally FAILED'

# ---- TC05: KWayMerge 输出全局有序 ----
import random
random.seed(123)
a = sorted([(random.random(),) for _ in range(10)], key=lambda r: r[0])
b = sorted([(random.random(),) for _ in range(10)], key=lambda r: r[0])
c = sorted([(random.random(),) for _ in range(10)], key=lambda r: r[0])
merger = KWayMerge([a, b, c], k=3, key_index=0)
merged = merger.merge()
assert len(merged) == 30, f'[TC05] KWayMerge produced wrong count: {len(merged)} FAILED'
assert verify_sorted(merged, 0) == True, '[TC05] KWayMerge output not sorted FAILED'

# ---- TC06: ExternalSortPipeline 空数据返回空列表 ----
pipeline = ExternalSortPipeline(memory_capacity=50, k_way=4, key_index=0)
result = pipeline.sort([])
assert result == [], '[TC06] ExternalSortPipeline empty input FAILED'

# ---- TC07: ExternalSortPipeline 小数据直接排序 ----
import random
random.seed(99)
small_data = [(random.random(),) for _ in range(30)]
pipeline = ExternalSortPipeline(memory_capacity=50, k_way=4, key_index=0)
result = pipeline.sort(small_data)
assert len(result) == 30, '[TC07] ExternalSortPipeline small data length mismatch FAILED'
assert verify_sorted(result, 0) == True, '[TC07] ExternalSortPipeline small data not sorted FAILED'

# ---- TC08: ExternalSortPipeline 超内存数据外排序 ----
import random
random.seed(42)
large_data = [(random.random(),) for _ in range(500)]
pipeline = ExternalSortPipeline(memory_capacity=50, k_way=4, key_index=0)
result = pipeline.sort(large_data)
assert len(result) == 500, '[TC08] ExternalSortPipeline large data length mismatch FAILED'
assert verify_sorted(result, 0) == True, '[TC08] ExternalSortPipeline large data not sorted FAILED'

# ---- TC09: robust_division 正常除法 ----
import math
result = robust_division(10.0, 2.0)
assert abs(result - 5.0) < 1e-12, f'[TC09] robust_division normal FAILED: {result}'

# ---- TC10: robust_division 除零返回fallback ----
result = robust_division(10.0, 0.0, fallback=999.0)
assert result == 999.0, '[TC10] robust_division zero division FAILED'

# ---- TC11: robust_division 产生inf返回fallback ----
result = robust_division(1e300, 1e-300, fallback=-1.0)
assert result == -1.0, '[TC11] robust_division inf fallback FAILED'

# ---- TC12: generate_primes 生成正确数量和正整数值 ----
primes = generate_primes(10, 5)
assert len(primes) == 5, f'[TC12] generate_primes count: {len(primes)} FAILED'
for p in primes:
    assert isinstance(p, int) and p > 1, f'[TC12] {p} not valid prime-like int FAILED'

# ---- TC13: generate_primes 可复现性 ----
import random
random.seed(42)
p1 = generate_primes(100, 5)
random.seed(42)
p2 = generate_primes(100, 5)
assert p1 == p2, '[TC13] generate_primes reproducibility FAILED'

# ---- TC14: entropy_of_distribution 均匀分布熵最大 ----
probs = [0.25, 0.25, 0.25, 0.25]
h = entropy_of_distribution(probs)
assert abs(h - 2.0) < 1e-10, f'[TC14] entropy uniform: {h} FAILED'

# ---- TC15: kldivergence 相同分布为零 ----
p = [0.5, 0.5]
q = [0.5, 0.5]
d = kldivergence(p, q)
assert abs(d) < 1e-12, f'[TC15] KL divergence of identical distributions: {d} FAILED'

# ---- TC16: R8LTTSolver matvec 正确性 ----
solver = R8LTTSolver([1.0, 2.0, 3.0])
x = [1.0, 0.0, 0.0]
y = solver.matvec(x)
assert abs(y[0] - 1.0) < 1e-12, f'[TC16] matvec[0]: {y[0]} FAILED'
assert abs(y[1] - 2.0) < 1e-12, f'[TC16] matvec[1]: {y[1]} FAILED'
assert abs(y[2] - 3.0) < 1e-12, f'[TC16] matvec[2]: {y[2]} FAILED'

# ---- TC17: R8LTTSolver solve 正确性 ----
solver = R8LTTSolver([1.0, 0.5, 0.25])
b = [1.0, 2.5, 4.25]
x = solver.solve(b)
# 验证 T*x = b
recovered = solver.matvec(x)
for i, (bi, ri) in enumerate(zip(b, recovered)):
    assert abs(bi - ri) < 1e-10, f'[TC17] solve mismatch at {i}: {bi} vs {ri} FAILED'

# ---- TC18: R8LTTSolver determinant 正确性 ----
solver = R8LTTSolver([2.0, 1.0, 0.5])
det = solver.determinant()
assert abs(det - 8.0) < 1e-12, f'[TC18] determinant: {det} FAILED'

# ---- TC19: R8LTTSolver inverse 验证 T * T^{-1} ≈ I ----
solver = R8LTTSolver([1.0, 0.5, 0.25])
inv = solver.inverse()
e1 = [1.0, 0.0, 0.0]
inv_e1 = inv.matvec(e1)
result = solver.matvec(inv_e1)
assert abs(result[0] - 1.0) < 1e-10, f'[TC19] inverse result[0]: {result[0]} FAILED'
assert abs(result[1]) < 1e-10, f'[TC19] inverse result[1]: {result[1]} FAILED'
assert abs(result[2]) < 1e-10, f'[TC19] inverse result[2]: {result[2]} FAILED'

# ---- TC20: HistogramPDFSampler 分位数边界 ----
data = [float(i) for i in range(100)]
hist = HistogramPDFSampler(data, num_bins=20)
q0 = hist.quantile(0.0)
q50 = hist.quantile(0.5)
q100 = hist.quantile(1.0)
assert q0 <= hist.x_min + 1.0, f'[TC20] q0: {q0} FAILED'
assert q100 >= hist.x_max - 1.0, f'[TC20] q100: {q100} FAILED'

# ---- TC21: HistogramPDFSampler 分区边界单调性 ----
data = [float(i) + 0.5 * math.sin(i) for i in range(200)]
hist = HistogramPDFSampler(data, num_bins=32)
bounds = hist.partition_boundaries(8)
for i in range(1, len(bounds)):
    assert bounds[i] >= bounds[i-1], f'[TC21] bounds not monotonic at {i} FAILED'

# ---- TC22: EulerErrorAnalysis 截断误差为正 ----
def f_test(t, y): return -y + math.sin(t)
def df_dt_test(t, y): return math.cos(t)
def df_dy_test(t, y): return -1.0

euler = EulerErrorAnalysis(f_test, df_dt_test, df_dy_test, y0=1.0, t0=0.0, t_end=1.0)
lte = euler.local_truncation_error(h=0.01, t=0.5, y=0.8)
assert lte >= 0, f'[TC22] LTE should be non-negative: {lte} FAILED'
assert not math.isnan(lte), '[TC22] LTE is NaN FAILED'
assert not math.isinf(lte), '[TC22] LTE is Inf FAILED'

# ---- TC23: EulerErrorAnalysis 全局误差上界 ----
geb = euler.global_error_bound(h=0.01, L=1.0, M2=2.0)
assert geb > 0, f'[TC23] GEB <= 0: {geb} FAILED'
assert not math.isnan(geb), '[TC23] GEB is NaN FAILED'

# ---- TC24: EulerErrorAnalysis 推荐步长为正 ----
rec_h = euler.recommend_step(target_error=1e-4, L=1.0, M2=2.0)
assert rec_h > 0, f'[TC24] recommended step <= 0: {rec_h} FAILED'

# ---- TC25: LogisticBifurcationAnalyzer Lyapunov指数 ----
la = LogisticBifurcationAnalyzer(r=3.9, x0=0.5)
lyap = la.lyapunov_exponent(n_iter=2000, n_transient=500)
assert lyap > 0, f'[TC25] Lyapunov exponent should be > 0 for chaotic r=3.9: {lyap} FAILED'

# ---- TC26: LogisticBifurcationAnalyzer 混沌判定 ----
la = LogisticBifurcationAnalyzer(r=3.9, x0=0.5)
assert la.is_chaotic() == True, '[TC26] r=3.9 should be chaotic FAILED'

# ---- TC27: LogisticBifurcationAnalyzer 不动点 ----
la = LogisticBifurcationAnalyzer(r=2.0, x0=0.3)
fps = la.fixed_points()
assert len(fps) >= 1, f'[TC27] fixed points count: {len(fps)} FAILED'

# ---- TC28: SIRStabilityAnalyzer 基本再生数 ----
sir_sa = SIRStabilityAnalyzer(alpha=0.3, beta=0.1, gamma=0.05, N=1000.0)
r0 = sir_sa.reproduction_number()
assert abs(r0 - 3.0) < 1e-10, f'[TC28] R0: {r0} FAILED'

# ---- TC29: SIRStabilityAnalyzer 地方病平衡点 ----
eeq = sir_sa.endemic_equilibrium()
S_star, I_star, R_star = eeq
assert abs(S_star - 1000.0 / 3.0) < 1e-6, f'[TC29] S*: {S_star} FAILED'
assert I_star >= 0, f'[TC29] I* negative: {I_star} FAILED'
assert R_star >= 0, f'[TC29] R* negative: {R_star} FAILED'

# ---- TC30: SIRStabilityAnalyzer R0 <= 1 返回无病平衡点 ----
sir_sa2 = SIRStabilityAnalyzer(alpha=0.05, beta=0.1, gamma=0.05, N=1000.0)
eeq2 = sir_sa2.endemic_equilibrium()
assert abs(eeq2[0] - 1000.0) < 1e-10, f'[TC30] S* for R0<1: {eeq2[0]} FAILED'

# ---- TC31: exact_circle_integral 奇数指数返回0 ----
val = exact_circle_integral(1, 2)
assert abs(val) < 1e-12, f'[TC31] odd exponent integral: {val} FAILED'

# ---- TC32: exact_circle_integral 偶数指数返回正值 ----
val = exact_circle_integral(2, 2)
assert val > 0, f'[TC32] even exponent integral <= 0: {val} FAILED'

# ---- TC33: logistic_chaotic_sequence 长度正确且值域在(0,1)内 ----
seq = logistic_chaotic_sequence(500, r=3.9, x0=0.314159)
assert len(seq) == 500, f'[TC33] seq length: {len(seq)} FAILED'
for v in seq:
    assert 0 < v < 1, f'[TC33] value out of (0,1): {v} FAILED'

# ---- TC34: ArteryFlowModel 模拟输出长度正确 ----
artery = ArteryFlowModel(alpha=2.0, beta=0.8, gamma=1.0, a=1.0, b=0.5, omega=1.5, x=1.0, dp_dx=1.0)
t_vals, u_vals, v_vals = artery.simulate_euler(u0=0.1, v0=0.0, t_end=10.0, n_steps=100)
assert len(t_vals) == 101, f'[TC34] time steps: {len(t_vals)} FAILED'
assert len(u_vals) == 101, f'[TC34] u steps: {len(u_vals)} FAILED'

# ---- TC35: ArteryFlowModel 解析振幅非负 ----
amp = artery.analytical_amplitude()
assert amp >= 0, f'[TC35] amplitude negative: {amp} FAILED'

# ---- TC36: LaxWendroffBuffer 守恒性质 ----
import math
nx = 20
lw = LaxWendroffBuffer(nx=nx, c=0.5, dx=0.1, dt=0.05)
rho0 = [0.5 + 0.3 * math.sin(2.0 * math.pi * i / nx) for i in range(nx)]
initial_total = sum(rho0)
history = lw.simulate(rho0, n_steps=50)
final_total = sum(history[-1])
assert abs(final_total - initial_total) / abs(initial_total) < 0.02, f'[TC36] conservation violated: {initial_total} vs {final_total} FAILED'

# ---- TC37: PiecewiseConstantPartition 分配正确 ----
pwc = PiecewiseConstantPartition([0.0, 0.25, 0.5, 0.75, 1.0])
assert pwc.assign_partition(0.1) == 0, f'[TC37] partition 0.1: {pwc.assign_partition(0.1)} FAILED'
assert pwc.assign_partition(0.4) == 1, f'[TC37] partition 0.4: {pwc.assign_partition(0.4)} FAILED'
assert pwc.assign_partition(0.6) == 2, f'[TC37] partition 0.6: {pwc.assign_partition(0.6)} FAILED'
assert pwc.assign_partition(0.9) == 3, f'[TC37] partition 0.9: {pwc.assign_partition(0.9)} FAILED'

# ---- TC38: CoverConstraintSolver 贪心分配覆盖所有块 ----
solver_cc = CoverConstraintSolver(num_partitions=4, capacity_per_partition=200)
block_sizes = [30, 50, 80, 40, 60]
assignment = solver_cc.greedy_assign(block_sizes)
assert len(assignment) == len(block_sizes), f'[TC38] assignment length: {len(assignment)} FAILED'
for a in assignment:
    assert a >= 0, f'[TC38] negative assignment: {a} FAILED'

# ---- TC39: CoverConstraintSolver 均衡评分为正 ----
score = solver_cc.balance_score(block_sizes, assignment)
assert score > 0, f'[TC39] balance score: {score} FAILED'

# ---- TC40: SortStabilityMonitor 完全均衡时倾斜度为0 ----
monitor = SortStabilityMonitor(target_partitions=4, memory_size=100)
skew = monitor.partition_skew([50, 50, 50, 50])
assert abs(skew) < 1e-10, f'[TC40] skew for balanced: {skew} FAILED'

# ---- TC41: SortStabilityMonitor 不均时倾斜度为正 ----
skew = monitor.partition_skew([80, 40, 40, 40])
assert skew > 0, f'[TC41] skew for unbalanced: {skew} FAILED'

# ---- TC42: toeplitz_transform_keys 输出长度正确 ----
keys = [1.0, 2.0, 3.0, 4.0, 5.0]
transformed = toeplitz_transform_keys(keys, decay=0.95)
assert len(transformed) == len(keys), f'[TC42] transform length: {len(transformed)} FAILED'

# ---- TC43: generate_heterogeneous_dataset 输出结构正确 ----
import random
random.seed(42)
dataset = generate_heterogeneous_dataset(total_records=100, memory_limit=30, seed=42)
assert len(dataset) == 100, f'[TC43] dataset length: {len(dataset)} FAILED'
for rec in dataset:
    assert len(rec) == 7, f'[TC43] record length: {len(rec)} FAILED'
    assert isinstance(rec[0], float), f'[TC43] key not float FAILED'

# ---- TC44: generate_heterogeneous_dataset 可复现性 ----
import random
random.seed(42)
d1 = generate_heterogeneous_dataset(total_records=50, memory_limit=20, seed=42)
random.seed(42)
d2 = generate_heterogeneous_dataset(total_records=50, memory_limit=20, seed=42)
for i, (r1, r2) in enumerate(zip(d1, d2)):
    assert abs(r1[0] - r2[0]) < 1e-12, f'[TC44] reproducibility failed at {i} FAILED'

# ---- TC45: predict_optimal_buffer_size 返回正值 ----
import math
buf = predict_optimal_buffer_size(alpha=2.0, beta=0.8, gamma=1.0, omega=1.5, safety_factor=1.5)
assert buf > 0, f'[TC45] buffer size: {buf} FAILED'

# ---- TC46: PolygonPartitionSampler 采样点数正确 ----
import random
random.seed(42)
poly = PolygonPartitionSampler([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
samples = poly.sample(20, seed=42)
assert len(samples) == 20, f'[TC46] sample count: {len(samples)} FAILED'
for sx, sy in samples:
    assert 0.0 <= sx <= 1.0, f'[TC46] x out of range: {sx} FAILED'
    assert 0.0 <= sy <= 1.0, f'[TC46] y out of range: {sy} FAILED'

# ---- TC47: FEM1DProjector 投影输出长度正确 ----
nodes = [float(i) for i in range(11)]
fem = FEM1DProjector(nodes)
def cdf_test(x): return (x - nodes[0]) / (nodes[-1] - nodes[0])
proj = fem.project(cdf_test)
assert len(proj) == len(nodes), f'[TC47] projection length: {len(proj)} FAILED'

# ---- TC48: RBFInterpolator 插值点处精确再现 ----
x_data = [0.0, 1.0, 2.0, 3.0]
f_data = [1.0, 2.0, 0.5, 1.5]
rbf = RBFInterpolator(x_data, f_data, r0=1.0, kernel_type=1)
for xi, fi in zip(x_data, f_data):
    val = rbf.evaluate(xi)
    assert abs(val - fi) < 1e-8, f'[TC48] RBF at {xi}: {val} vs {fi} FAILED'

# ---- TC49: main() 函数返回结果结构正确 ----
import random
random.seed(199)
result = main()
assert isinstance(result, dict), f'[TC49] result is not dict FAILED'
assert 'sorted' in result, '[TC49] missing "sorted" FAILED'
assert 'records' in result, '[TC49] missing "records" FAILED'
assert 'time' in result, '[TC49] missing "time" FAILED'

# ---- TC50: main() 排序结果正确 ----
assert result['sorted'] == True, '[TC50] main sort result FAILED'
