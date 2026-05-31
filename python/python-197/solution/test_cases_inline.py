# ---- TC01: TetrahedralMesh generate_uniform_box 返回正确节点数 ----
from mesh_geometry import TetrahedralMesh
mesh = TetrahedralMesh.generate_uniform_box(nx=4, ny=4, nz=4, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
assert mesh.n_nodes == 64, '[TC01] 节点数应为 4*4*4=64 FAILED'

# ---- TC02: TetrahedralMesh 单元数为 6*(nx-1)*(ny-1)*(nz-1) ----
assert mesh.n_elements == 6 * 3 * 3 * 3, '[TC02] 单元数应为 6*3*3*3=162 FAILED'

# ---- TC03: TetrahedralMesh compute_volumes 所有体积为正 ----
vols = mesh.compute_volumes()
import numpy as np
assert np.all(vols > 0.0), '[TC03] 所有单元体积必须为正 FAILED'
assert np.all(np.isfinite(vols)), '[TC03] 所有单元体积必须有限 FAILED'

# ---- TC04: TetrahedralMesh barycenters 形状正确 ----
bc = mesh.barycenters()
assert bc.shape == (mesh.n_elements, 3), '[TC04] 重心形状应为 (n_elements, 3) FAILED'
assert np.all(np.isfinite(bc)), '[TC04] 重心坐标必须有限 FAILED'

# ---- TC05: TetrahedralMesh bounding_box 返回正确范围 ----
bbox = mesh.bounding_box()
assert bbox == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), '[TC05] bounding box 应与输入一致 FAILED'

# ---- TC06: TetrahedralMesh element_diameter 为正有限 ----
h_max = mesh.element_diameter()
assert h_max > 0.0, '[TC06] 最大单元直径必须为正 FAILED'
assert np.isfinite(h_max), '[TC06] 最大单元直径必须有限 FAILED'

# ---- TC07: AdvectionDiffusionSolver initial_condition gaussian 非负 ----
from mesh_geometry import TetrahedralMesh
from pde_solver import AdvectionDiffusionSolver
import numpy as np
mesh = TetrahedralMesh.generate_uniform_box(nx=3, ny=3, nz=3, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
solver = AdvectionDiffusionSolver(mesh, D=0.01, velocity=np.array([0.0, 0.0, 0.0]), reaction_rate=0.0)
u0 = solver.initial_condition(mode="gaussian")
assert np.all(u0 >= 0.0), '[TC07] Gaussian 初始条件必须非负 FAILED'
assert np.all(np.isfinite(u0)), '[TC07] Gaussian 初始条件必须有限 FAILED'

# ---- TC08: AdvectionDiffusionSolver 边界节点为零 ----
assert np.all(np.abs(u0[solver.boundary_nodes]) < 1.0e-14), '[TC08] 边界节点值必须为零 FAILED'

# ---- TC09: AdvectionDiffusionSolver initial_condition 随机模式形状正确 ----
u_rand = solver.initial_condition(mode="random")
assert len(u_rand) == mesh.n_nodes, '[TC09] 随机初始条件长度应等于节点数 FAILED'
assert np.all(u_rand >= 0.0), '[TC09] 随机初始条件必须非负 FAILED'

# ---- TC10: AdvectionDiffusionSolver step_explicit 保持形状 ----
import numpy as np
u = solver.initial_condition(mode="gaussian")
u_new = solver.step_explicit(u, dt=0.0001)
assert u_new.shape == u.shape, '[TC10] step_explicit 必须保持数组形状 FAILED'
assert np.all(np.isfinite(u_new)), '[TC10] step_explicit 输出必须有限 FAILED'

# ---- TC11: AdvectionDiffusionSolver compute_energy 非负 ----
energy = solver.compute_energy(u)
assert energy >= 0.0, '[TC11] 离散能量必须非负 FAILED'
assert np.isfinite(energy), '[TC11] 离散能量必须有限 FAILED'

# ---- TC12: GammaFaultModel mean = alpha/beta ----
from fault_model import GammaFaultModel
model = GammaFaultModel(alpha=2.5, beta=0.0025)
assert abs(model.mean() - 1000.0) < 1.0e-6, '[TC12] mean=alpha/beta=1000 FAILED'

# ---- TC13: GammaFaultModel variance = alpha/beta^2 ----
assert abs(model.variance() - 2.5/0.0025**2) < 1.0e-6, '[TC13] variance=alpha/beta^2 FAILED'

# ---- TC14: GammaFaultModel cdf(0)=0, survival(0)=1 ----
assert model.cdf(0.0) == 0.0, '[TC14] Gamma CDF(0) 必须为 0 FAILED'
assert model.survival(0.0) == 1.0, '[TC14] Gamma survival(0) 必须为 1 FAILED'

# ---- TC15: GammaFaultModel pdf 在 t>0 处为正 ----
import numpy as np
p = model.pdf(500.0)
assert p > 0.0, '[TC15] Gamma PDF 在 t=500 处必须为正 FAILED'
assert np.isfinite(p), '[TC15] Gamma PDF 必须有限 FAILED'

# ---- TC16: GammaFaultModel entropy 有限 ----
import numpy as np
h = model.entropy()
assert np.isfinite(h), '[TC16] Gamma 分布熵必须有限 FAILED'

# ---- TC17: GammaFaultModel hazard 函数行为 ----
import numpy as np
h0 = model.hazard(0.0)
h_large = model.hazard(1.0e6)
assert h0 == 0.0, '[TC17] hazard(0) 必须为 0 FAILED'
assert np.isfinite(h_large), '[TC17] hazard 大 t 必须有限 FAILED'

# ---- TC18: FaultPredictor observe 与 test_increase 初态 ----
from fault_model import FaultPredictor
predictor = FaultPredictor(significance=0.05)
is_inc, pv = predictor.test_increase()
assert is_inc is False, '[TC18] 无历史时 test_increase 应为 False FAILED'
assert pv == 1.0, '[TC18] 无历史时 p_value 应为 1.0 FAILED'

# ---- TC19: FaultPredictor recommended_checkpoint_interval 默认值 ----
rec = predictor.recommended_checkpoint_interval()
assert rec == 100.0, '[TC19] 无历史时推荐间隔应为 100.0 FAILED'

# ---- TC20: FaultPredictor test_increase 有限概率样本 ----
from fault_model import FaultPredictor
import numpy as np
predictor2 = FaultPredictor(significance=0.05)
predictor2.observe(100.0)
predictor2.observe(110.0)
predictor2.observe(90.0)
predictor2.observe(105.0)
is_inc2, pv2 = predictor2.test_increase()
assert is_inc2 in (True, False), '[TC20] test_increase 返回值应为 bool 类型 FAILED'
assert 0.0 <= pv2 <= 1.0, '[TC20] p_value 必须在 [0,1] FAILED'

# ---- TC21: CheckpointManager create 与 restore 压缩周期 ----
from checkpoint_manager import CheckpointManager
from fault_model import FaultPredictor, GammaFaultModel
import numpy as np
manager = CheckpointManager(compression_method="svd", target_compression_ratio=0.12)
state = np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 128))
manager.create_checkpoint(step=10, state=state, level=0)
restored = manager.restore_checkpoint(step=10, level=0)
assert restored.shape == state.shape, '[TC21] 恢复状态形状需与原始一致 FAILED'
assert np.all(np.isfinite(restored)), '[TC21] 恢复状态必须有限 FAILED'

# ---- TC22: CheckpointManager compression_error 非负 ----
err = manager.compression_error(step=10, true_state=state)
assert err >= 0.0, '[TC22] 压缩误差必须非负 FAILED'

# ---- TC23: CheckpointManager find_latest_checkpoint 行为 ----
ck = manager.find_latest_checkpoint(current_step=15)
assert ck == 10, '[TC23] 最新检查点步应为 10 FAILED'

# ---- TC24: CheckpointNode write_time 与 read_time 正比 ----
from checkpoint_tree import CheckpointNode
node = CheckpointNode(level=0, name="DRAM", write_bw=10.0, read_bw=20.0, capacity=64.0, cost_per_gb=100.0)
wt = node.write_time(2.0)
rt = node.read_time(2.0)
assert abs(wt - 2.0/10.0) < 1.0e-10, '[TC24] write_time=2/10=0.2 FAILED'
assert abs(rt - 2.0/20.0) < 1.0e-10, '[TC24] read_time=2/20=0.1 FAILED'

# ---- TC25: CheckpointTree expected_recovery_time 单调性 ----
from checkpoint_tree import build_default_tree
tree = build_default_tree()
ert_low = tree.expected_recovery_time(1.0, {0: 1.0, 1: 0.0, 2: 0.0})
ert_high = tree.expected_recovery_time(1.0, {0: 0.0, 1: 0.0, 2: 1.0})
assert ert_high > ert_low, '[TC25] 远程恢复时间应大于内存恢复时间 FAILED'

# ---- TC26: CheckpointTree tree_distance 自身为零 ----
root = tree.root
d = tree.tree_distance(root, root)
assert d == 0, '[TC26] 节点到自身距离必须为 0 FAILED'

# ---- TC27: CheckpointMDP value_iteration 收敛 ----
from recovery_mdp import CheckpointMDP
import numpy as np
mdp = CheckpointMDP(
    p_fault=0.02, p_fault_during_ckpt=0.01,
    recover_probs=np.array([0.99, 0.95, 0.90]),
    step_costs=np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [0.5, 0.5, 0.5],
        [5.0, 3.0, 1.5],
        [0.0, 0.0, 0.0],
    ])
)
V, policy = mdp.value_iteration(gamma=0.95, tol=1.0e-8)
assert len(V) == 5, '[TC27] 值函数维度应为 5 FAILED'
assert len(policy) == 5, '[TC27] 策略维度应为 5 FAILED'
assert np.all(np.isfinite(V)), '[TC27] 值函数必须有限 FAILED'

# ---- TC28: CheckpointMDP stationary_distribution 和为 1 ----
pi = mdp.stationary_distribution(action=0)
assert len(pi) == 4, '[TC28] 稳态分布维度应为 4 FAILED'
assert abs(np.sum(pi) - 1.0) < 1.0e-8, '[TC28] 稳态分布之和必须为 1 FAILED'

# ---- TC29: CheckpointMDP expected_time_to_done 确定性 ----
import numpy as np
np.random.seed(42)
t1 = mdp.expected_time_to_done(action=0, max_steps=500)
np.random.seed(42)
t2 = mdp.expected_time_to_done(action=0, max_steps=500)
assert abs(t1 - t2) < 1.0e-10, '[TC29] 相同种子下期望时间必须相等 FAILED'

# ---- TC30: CheckpointStrategyOptimizer objective 最优解析解 ----
from sampling_optimizer import CheckpointStrategyOptimizer
import numpy as np
opt = CheckpointStrategyOptimizer(n_samples=50)
# 解析最优: T* = sqrt(2 * state_gb * compression_ratio / (bw_ratio * fault_rate))
fr, bw, sg, cr = 1.0e-4, 0.1, 5.0, 0.2
expected_opt = np.sqrt(2.0 * sg * cr / (bw * fr))
t_candidates = np.array([expected_opt])
loss = opt.objective(fr, bw, sg, cr, expected_opt)
assert np.isfinite(loss), '[TC30] 目标函数必须有限 FAILED'
assert loss >= 0.0, '[TC30] 目标函数必须非负 FAILED'

# ---- TC31: CheckpointStrategyOptimizer optimize_interval 返回有限结果 ----
best_int, best_loss = opt.optimize_interval(fr, bw, sg, cr)
assert best_int > 0.0, '[TC31] 最优间隔必须为正 FAILED'
assert np.isfinite(best_loss), '[TC31] 最优损失必须有限 FAILED'

# ---- TC32: CheckpointStrategyOptimizer robust_optimize 可复现 ----
opt1 = CheckpointStrategyOptimizer(n_samples=50)
r1 = opt1.robust_optimize(seed=42)
opt2 = CheckpointStrategyOptimizer(n_samples=50)
r2 = opt2.robust_optimize(seed=42)
assert abs(r1["mean_interval"] - r2["mean_interval"]) < 1.0e-10, '[TC32] 相同种子 robust_optimize 必须可复现 FAILED'

# ---- TC33: legendre_rule 积分 sin(pi*x) 精度 ----
from quadrature_engine import legendre_rule
import numpy as np
x_l, w_l = legendre_rule(12, a=0.0, b=1.0)
integral_l = np.sum(w_l * np.sin(np.pi * x_l))
exact_l = 2.0 / np.pi
assert abs(integral_l - exact_l) < 1.0e-6, '[TC33] Gauss-Legendre 12点积分误差应 < 1e-6 FAILED'

# ---- TC34: laguerre_rule 积分 x^2*exp(-x) 精度 ----
from quadrature_engine import laguerre_rule
x_la, w_la = laguerre_rule(10, alpha=0.0)
integral_la = np.sum(w_la * (x_la ** 2))
exact_la = 2.0
assert abs(integral_la - exact_la) < 1.0e-6, '[TC34] Gauss-Laguerre 10点积分误差应 < 1e-6 FAILED'

# ---- TC35: integrate_tetrahedron 常数函数等于体积 ----
from quadrature_engine import integrate_tetrahedron
def f_one(xyz):
    return 1.0
vol_int = integrate_tetrahedron(f_one, order=4)
assert abs(vol_int - 1.0/6.0) < 1.0e-10, '[TC35] 四面体积分常数 1 应等于 1/6 FAILED'

# ---- TC36: integrate_tetrahedron 二次函数精确积分 ----
def f_quad(xyz):
    return xyz[0]**2 + xyz[1]**2 + xyz[2]**2
quad_int = integrate_tetrahedron(f_quad, order=4)
assert abs(quad_int - 1.0/20.0) < 1.0e-6, '[TC36] 四面体积分 x^2+y^2+z^2 应等于 1/20 FAILED'

# ---- TC37: R83SMatrix mv 与 to_dense 一致 ----
from sparse_linear_algebra import R83SMatrix
import numpy as np
a_vec = np.array([-1.0, 2.0, -1.0])
A = R83SMatrix(6, 6, a_vec)
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y_mv = A.mv(x)
y_dense = A.to_dense() @ x
assert np.linalg.norm(y_mv - y_dense) < 1.0e-12, '[TC37] mv 与 to_dense@x 必须一致 FAILED'

# ---- TC38: R83SMatrix residual 计算正确 ----
b = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
r = A.residual(x, b)
assert np.linalg.norm(r - (b - A.mv(x))) < 1.0e-12, '[TC38] residual 计算错误 FAILED'

# ---- TC39: r83s_cg 求解精度 ----
from sparse_linear_algebra import r83s_cg
b_test = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0])
x_cg = r83s_cg(6, a_vec, b_test, tol=1.0e-12)
r_norm = np.linalg.norm(A.residual(x_cg, b_test))
assert r_norm < 1.0e-6, '[TC39] CG 残差应 < 1e-6 FAILED'

# ---- TC40: r83s_jacobi 求解精度 ----
from sparse_linear_algebra import r83s_jacobi
x_jac = r83s_jacobi(6, a_vec, b_test, tol=1.0e-10)
r_norm_jac = np.linalg.norm(A.residual(x_jac, b_test))
assert r_norm_jac < 1.0e-6, '[TC40] Jacobi 残差应 < 1e-6 FAILED'

# ---- TC41: r83s_gauss_seidel 求解精度 ----
from sparse_linear_algebra import r83s_gauss_seidel
x_gs = r83s_gauss_seidel(6, a_vec, b_test, tol=1.0e-10)
r_norm_gs = np.linalg.norm(A.residual(x_gs, b_test))
assert r_norm_gs < 1.0e-6, '[TC41] Gauss-Seidel 残差应 < 1e-6 FAILED'

# ---- TC42: cholesky_decompose 重构精度 ----
from sparse_linear_algebra import cholesky_decompose
cov = np.array([[4.0, 1.2, 0.5], [1.2, 3.0, 0.8], [0.5, 0.8, 2.5]])
L, nullty, ifault = cholesky_decompose(cov, eta=1.0e-12)
recon = L @ L.T
assert np.linalg.norm(cov - recon) < 1.0e-10, '[TC42] Cholesky 重构误差应 < 1e-10 FAILED'
assert nullty >= 0, '[TC42] nullty 必须非负 FAILED'

# ---- TC43: alnorm 对称性 ----
from special_functions import alnorm
v1 = alnorm(1.0, upper=False)
v2 = alnorm(-1.0, upper=True)
assert abs(v1 - v2) < 1.0e-14, '[TC43] alnorm 对称性 alnorm(1,F)=alnorm(-1,T) FAILED'

# ---- TC44: alnorm 输出在 [0,1] 范围内 ----
for xv in [0.0, 1.0, 2.0, 5.0]:
    val = alnorm(xv, upper=False)
    assert 0.0 <= val <= 1.0, f'[TC44] alnorm({xv}) 必须在 [0,1] 内 FAILED'

# ---- TC45: fresnel 奇函数性 ----
from special_functions import fresnel
c1, s1 = fresnel(2.0)
c2, s2 = fresnel(-2.0)
assert abs(c1 + c2) < 1.0e-14, '[TC45] fresnel C(x) 为奇函数 FAILED'
assert abs(s1 + s2) < 1.0e-14, '[TC45] fresnel S(x) 为奇函数 FAILED'

# ---- TC46: fresnel(0) = (0,0) ----
c0, s0 = fresnel(0.0)
assert abs(c0) < 1.0e-14 and abs(s0) < 1.0e-14, '[TC46] fresnel(0) 必须为 (0,0) FAILED'

# ---- TC47: digamma 已知值 ----
from special_functions import digamma
psi1, _ = digamma(1.0)
euler_gamma = -0.5772156649015329
assert abs(psi1 - euler_gamma) < 1.0e-6, '[TC47] digamma(1) 应等于 Euler 常数 FAILED'

# ---- TC48: svd_compress 压缩比正确 ----
from state_compression import svd_compress, svd_reconstruct
import numpy as np
state = np.random.default_rng(42).random((16, 8))
U_r, s_r, Vt_r, comp = svd_compress(state, rank=4)
assert U_r.shape == (16, 4), '[TC48] U_r 形状应为 (16, 4) FAILED'
assert Vt_r.shape == (4, 8), '[TC48] Vt_r 形状应为 (4, 8) FAILED'
assert comp.shape == (16, 8), '[TC48] 压缩状态形状应为 (16, 8) FAILED'

# ---- TC49: svd_reconstruct 自洽性 ----
reconstructed = svd_reconstruct(U_r, s_r, Vt_r)
diff = np.linalg.norm(comp - reconstructed) / max(np.linalg.norm(comp), 1.0e-14)
assert diff < 1.0e-12, '[TC49] SVD 重构必须自洽 FAILED'

# ---- TC50: compress_state_trig 与 reconstruct_state_trig 形状正确 ----
from state_compression import compress_state_trig, reconstruct_state_trig
state_1d = np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 64))
xd_c, yd_c, N = compress_state_trig(state_1d, n_coarse=8)
rec = reconstruct_state_trig(xd_c, yd_c, N)
assert len(rec) == N, '[TC50] 三角插值恢复长度应等于 N=64 FAILED'
assert np.all(np.isfinite(rec)), '[TC50] 三角插值恢复必须有限 FAILED'

# ---- TC51: legendre_rule 权重和等于区间长度 ----
x_l2, w_l2 = legendre_rule(8, a=-1.0, b=2.0)
assert abs(np.sum(w_l2) - 3.0) < 1.0e-12, '[TC51] Gauss-Legendre 权重和应等于 (b-a)=3 FAILED'

# ---- TC52: GammaFaultModel sample 形状正确 ----
import numpy as np
samples = model.sample(size=100, seed=42)
assert len(samples) == 100, '[TC52] 采样数量应为 100 FAILED'
assert np.all(samples > 0.0), '[TC52] Gamma 采样必须全为正 FAILED'
assert np.all(np.isfinite(samples)), '[TC52] Gamma 采样必须全有限 FAILED'

# ---- TC53: TetrahedralMesh 体积和等于网格总体积 ----
vols = mesh.compute_volumes()
assert abs(np.sum(vols) - 1.0) < 1.0e-8, '[TC53] 单元体积和应等于总体积 1.0 FAILED'

# ---- TC54: AdvectionDiffusionSolver step_explicit 确定性 ----
import numpy as np
np.random.seed(123)
mesh2 = TetrahedralMesh.generate_uniform_box(nx=4, ny=4, nz=4, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
solver2 = AdvectionDiffusionSolver(mesh2, D=0.02, velocity=np.array([0.1, 0.0, 0.0]), reaction_rate=0.01)
u2 = solver2.initial_condition(mode="gaussian")
u3 = solver2.step_explicit(u2, 0.0001)
assert np.all(np.isfinite(u3)), '[TC54] 对流 step_explicit 输出必须有限 FAILED'

# ---- TC55: 集成测试: 主流程基本结构验证 ----
# 确认 main() 执行后关键变量存在且合理
try:
    test_mesh = TetrahedralMesh.generate_uniform_box(nx=3, ny=3, nz=3)
    test_solver = AdvectionDiffusionSolver(test_mesh, D=0.02)
    test_u = test_solver.initial_condition(mode="gaussian")
    test_fault = GammaFaultModel(alpha=2.0, beta=0.001)
    test_pred = FaultPredictor(significance=0.05)
    test_mgr = CheckpointManager(compression_method="svd", target_compression_ratio=0.1)
    test_mgr.create_checkpoint(0, test_u, level=0)
    restored_u = test_mgr.restore_checkpoint(0, level=0)
    assert restored_u.shape == test_u.shape, '[TC55] 集成流程：恢复状态形状需一致 FAILED'
except Exception as e:
    assert False, f'[TC55] 集成流程异常: {e} FAILED'
