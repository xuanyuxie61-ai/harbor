# ---- TC01: P_2 Legendre 多项式已知值 (cosθ=1→1, cosθ=0→-0.5, cosθ=-1→1) ----
sys_tc = LipidBilayerSystem(nx=4, ny=4)
assert abs(sys_tc._p2_legendre(1.0) - 1.0) < 1e-12, '[TC01] P_2(1)=1 FAILED'
assert abs(sys_tc._p2_legendre(0.0) - (-0.5)) < 1e-12, '[TC01] P_2(0)=-0.5 FAILED'
assert abs(sys_tc._p2_legendre(-1.0) - 1.0) < 1e-12, '[TC01] P_2(-1)=1 FAILED'
cos_13 = np.sqrt(1.0 / 3.0)
assert abs(sys_tc._p2_legendre(cos_13)) < 1e-12, '[TC01] P_2(√(1/3))=0 FAILED'

# ---- TC02: spherical_harmonic_y20_approx 在 cosθ=1 处的值 ----
from order_parameters import spherical_harmonic_y20_approx
val_y20 = spherical_harmonic_y20_approx(1.0)
expected_y20 = np.sqrt(5.0 / (16.0 * np.pi)) * 2.0
assert abs(val_y20 - expected_y20) < 1e-12, '[TC02] Y_2^0(θ=0) FAILED'

# ---- TC03: debye_waller_factor 基本计算 ----
B_dw = debye_waller_factor(order_param=0.8, temperature=300.0, moment_inertia=1.0)
assert B_dw > 0, '[TC03] Debye-Waller B 非正 FAILED'
assert np.isfinite(B_dw), '[TC03] Debye-Waller B 非有限 FAILED'

# ---- TC04: Jacobi 多项式 P_0 恒为 1 ----
import numpy as np
from order_parameters import jacobi_polynomial
x_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
P_j = jacobi_polynomial(4, 0.0, 2.0, x_test)
assert np.allclose(P_j[:, 0], 1.0), '[TC04] Jacobi P_0 ≠ 1 FAILED'

# ---- TC05: Jacobi 多项式 P_1 解析验证 ----
# P_1^{(α,β)}(x) = ((α+β+2)x + (α-β))/2
P1_expected = ((0.0 + 2.0 + 2.0) * x_test + (0.0 - 2.0)) / 2.0
assert np.allclose(P_j[:, 1], P1_expected), '[TC05] Jacobi P_1 解析不匹配 FAILED'

# ---- TC06: Jacobi 归一化常数已知值 ----
from order_parameters import jacobi_norm_constant
h0 = jacobi_norm_constant(0, 0.0, 0.0)
assert abs(h0 - 2.0) < 1e-12, '[TC06] Jacobi h_0^{(0,0)} != 2 FAILED'
h1 = jacobi_norm_constant(1, 0.0, 0.0)
assert abs(h1 - 2.0 / 3.0) < 1e-12, '[TC06] Jacobi h_1^{(0,0)} != 2/3 FAILED'

# ---- TC07: Bernstein 基函数单位分解性 (∑ B_{n,k}(u) = 1) ----
u_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
B10 = bernstein_basis(10, u_vals)
assert np.allclose(np.sum(B10, axis=-1), 1.0), '[TC07] Bernstein 单位分解 FAILED'

# ---- TC08: Bernstein 基函数边界值 B_{n,0}(0)=1, B_{n,n}(1)=1 ----
assert abs(B10[0, 0] - 1.0) < 1e-12, '[TC08] B_{10,0}(0) != 1 FAILED'
assert abs(B10[-1, -1] - 1.0) < 1e-12, '[TC08] B_{10,10}(1) != 1 FAILED'

# ---- TC09: Bernstein 基函数输出维度 ----
from density_profile import bernstein_basis as bb
B5 = bernstein_basis(5, 0.3)
assert B5.shape == (1, 6), '[TC09] Bernstein 标量输入维度 FAILED'
B5_arr = bb(5, np.linspace(0, 1, 20))
assert B5_arr.shape == (20, 6), '[TC09] Bernstein 数组输入维度 FAILED'

# ---- TC10: GridGenerator 矩形网格形状与间距 ----
Xr, Yr, dxr, dyr = GridGenerator.rectangular_grid(12, 12)
assert Xr.shape == (12, 12), '[TC10] 矩形网格 X 形状 FAILED'
assert Yr.shape == (12, 12), '[TC10] 矩形网格 Y 形状 FAILED'
assert dxr > 0 and dyr > 0, '[TC10] 网格间距 FAILED'

# ---- TC11: GridGenerator 极坐标网格形状 ----
Rp, Tp, Xp, Yp = GridGenerator.polar_grid(0.5, 3.0, 8, 16)
assert Rp.shape == (8, 16), '[TC11] 极坐标 R 形状 FAILED'
assert Xp.shape == (8, 16), '[TC11] 极坐标 X 形状 FAILED'

# ---- TC12: GridGenerator 三角网格节点数与三角形数 ----
nodes_tri, triangs = GridGenerator.triangular_grid(8, 8)
assert len(nodes_tri) == 64, '[TC12] 三角网格节点数 FAILED'
assert len(triangs) == 2 * (8 - 1) * (8 - 1), '[TC12] 三角网格三角形数 FAILED'

# ---- TC13: BoundaryTracer 等边三角形周长 > 0 ----
bt = BoundaryTracer(grid_type='hex')
eq_word = bt.equilateral_triangle_boundary(side_length=4)
peri_tri, area_tri = bt.compute_perimeter_and_area(eq_word)
assert peri_tri > 0, '[TC13] 等边三角形周长非正 FAILED'
assert area_tri > 0, '[TC13] 等边三角形面积非正 FAILED'

# ---- TC14: membrane_surface_metric 张量正定性 ----
from grid_topology import membrane_surface_metric
E, F, G = membrane_surface_metric(nodes_tri, triangs)
assert E > 0, '[TC14] E 非正 FAILED'
assert G > 0, '[TC14] G 非正 FAILED'
assert E * G - F * F > 0, '[TC14] 度量张量非正定 FAILED'

# ---- TC15: CombinatorialEnumerators.gray_code 前 4 个值 ----
from combinatorial_sampler import CombinatorialEnumerators
gc = CombinatorialEnumerators.gray_code(3)
assert gc[0] == 0, '[TC15] Gray 码第一个值 != 0 FAILED'
assert gc[1] == 1, '[TC15] Gray 码第二个值 != 1 FAILED'
assert gc[2] == 3, '[TC15] Gray 码第三个值 != 3 FAILED'
assert gc[3] == 2, '[TC15] Gray 码第四个值 != 2 FAILED'

# ---- TC16: Stirling 第二类数已知值 ----
S_nn = CombinatorialEnumerators.stirling_second(5, 2)
assert S_nn == 15, '[TC16] S(5,2) != 15 FAILED'
S_0 = CombinatorialEnumerators.stirling_second(0, 0)
assert S_0 == 1, '[TC16] S(0,0) != 1 FAILED'
S_ni = CombinatorialEnumerators.stirling_second(5, 6)
assert S_ni == 0, '[TC16] S(5,6) != 0 FAILED'

# ---- TC17: 整数划分计数 ----
parts = CombinatorialEnumerators.integer_partitions(5)
assert len(parts) == 7, '[TC17] p(5) != 7 FAILED'

# ---- TC18: k_subset_lex 计数匹配 C(n,k) ----
subsets_5_2 = CombinatorialEnumerators.k_subset_lex(5, 2)
expected_c52 = 10
assert len(subsets_5_2) == expected_c52, '[TC18] C(5,2) 子集数 FAILED'

# ---- TC19: BacktrackSampler 空约束搜索 ----
from combinatorial_sampler import BacktrackSampler
bs = BacktrackSampler(n_vars=3, n_states=2, constraint_func=None)
sols = bs.search(max_solutions=10)
assert len(sols) == 8, '[TC19] 3 vars × 2 states 应产生 8 个解 FAILED'

# ---- TC20: ConfigurationSampler 固定种子可复现 ----
import numpy as np
np.random.seed(42)
cs = ConfigurationSampler(nx=4, ny=4, n_orient_states=6)
config1 = cs.random_configuration(seed=123)
np.random.seed(42)
config2 = cs.random_configuration(seed=123)
assert np.array_equal(config1, config2), '[TC20] 固定种子不复现 FAILED'

# ---- TC21: build_lipid_adjacency 节点数 ----
adj = build_lipid_adjacency(4, 4, interaction_range=1)
assert len(adj) == 16, '[TC21] 邻接图节点数 != 16 FAILED'

# ---- TC22: SparseMatrixOps.bandwidth 简单测试 ----
from sparse_matrix_ops import SparseMatrixOps
adj_small = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
perm = np.array([0, 1, 2])
bw = SparseMatrixOps.bandwidth(adj_small, perm)
assert bw >= 0, '[TC22] 带宽应为非负 FAILED'

# ---- TC23: RCM 重排序（4×4 格点） ----
import numpy as np
adj_rcm = build_lipid_adjacency(4, 4, interaction_range=1)
n_nodes = 4 * 4
adj_row, adj_col = SparseMatrixOps.adjacency_to_csr(adj_rcm, n_nodes)
perm_rcm = SparseMatrixOps.rcm_reorder(8, adj_row, adj_col, n_nodes)
assert len(perm_rcm) == n_nodes, '[TC23] RCM 排列长度 FAILED'
assert len(set(perm_rcm)) == n_nodes, '[TC23] RCM 排列不唯一 FAILED'

# ---- TC24: RCM 降低带宽 ----
bw_before = SparseMatrixOps.bandwidth(adj_rcm, np.arange(n_nodes))
bw_after = SparseMatrixOps.bandwidth(adj_rcm, perm_rcm)
assert bw_after <= bw_before, '[TC24] RCM 未能降低带宽 FAILED'

# ---- TC25: MarkovStateModel 稳态分布和为 1 ----
import numpy as np
np.random.seed(42)
P_rand = np.random.rand(6, 6)
P_rand = P_rand / P_rand.sum(axis=1, keepdims=True)
msm = MarkovStateModel(P_rand)
pi = msm.power_method_steady_state(max_iter=500)
assert abs(np.sum(pi) - 1.0) < 1e-8, '[TC25] 稳态分布和 != 1 FAILED'
assert np.all(pi >= -1e-12), '[TC25] 稳态分布含负值 FAILED'

# ---- TC26: MarkovStateModel PageRank 和为 1 ----
pr = msm.pagerank_style_rank(damping=0.85)
assert abs(np.sum(pr) - 1.0) < 1e-8, '[TC26] PageRank 和 != 1 FAILED'

# ---- TC27: build_diffusion_matrix_from_adjacency 行和为零 ----
L = build_diffusion_matrix_from_adjacency(adj_rcm, n_nodes, D=1.0, dt=0.001)
row_sums = np.sum(L, axis=1)
assert np.allclose(row_sums, 0.0, atol=1e-10), '[TC27] 扩散矩阵行和不零 FAILED'

# ---- TC28: SparseGridIntegration 常数函数积分 = 2^D ----
from free_energy import SparseGridIntegration
sgi = SparseGridIntegration(dim_num=2, level_max=3)
integral_const = sgi.integrate(lambda x: 1.0)
assert abs(abs(integral_const) - 4.0) < 0.01, '[TC28] 常数在 [-1,1]² 上积分绝对值 ≠ 4 FAILED'

# ---- TC29: FreeEnergyCalculator.transition_temperature_estimate 返回正值 ----
Tc_est = FreeEnergyCalculator.transition_temperature_estimate(J=2.5)
assert Tc_est > 0, '[TC29] T_c 估计值应为正 FAILED'

# ---- TC30: IntegratorStability Euler 放大因子在 z=0 处为 1 ----
stab_euler = IntegratorStability('euler')
R0 = stab_euler.amplification_factor(0.0 + 0.0j)
assert abs(R0 - 1.0) < 1e-12, '[TC30] Euler R(0) != 1 FAILED'

# ---- TC31: IntegratorStability Verlet 稳定性检验 ----
stab_verlet = IntegratorStability('verlet')
stable, z_pts = stab_verlet.check_system_stability(omega_max=2.0, gamma=0.5, dt=0.002)
assert stable, '[TC31] Verlet 应稳定 (ω·dt=0.004 < 2) FAILED'
assert len(z_pts) == 2, '[TC31] z_points 长度 != 2 FAILED'

# ---- TC32: IntegratorStability Verlet 不稳定检测 ----
stable_unst, _ = stab_verlet.check_system_stability(omega_max=2000.0, gamma=0.0, dt=0.1)
assert not stable_unst, '[TC32] Verlet 应检测到不稳定 FAILED'

# ---- TC33: NewtonMaehly 求解 z² - 1 = 0 ----
import numpy as np
coeffs = np.array([-1.0, 0.0, 1.0], dtype=complex)
nms = NewtonMaehlySolver(coeffs, max_iter=100, tol=1e-12)
roots = nms.solve()
roots_sorted = sorted(roots, key=lambda r: abs(r - 1.0))
assert abs(roots_sorted[0] - 1.0) < 1e-8, '[TC33] 根 +1 未找到 FAILED'
assert abs(roots_sorted[1] - (-1.0)) < 1e-8, '[TC33] 根 -1 未找到 FAILED'

# ---- TC34: SelfConsistentTransition.critical_temperature 公式验证 ----
from phase_diagram import SelfConsistentTransition
sc = SelfConsistentTransition(J_coupling=2.5, kb=0.008314)
Tc_val = sc.critical_temperature()
Tc_expected = 2.5 / (2.0 * 0.008314)
assert abs(Tc_val - Tc_expected) < 1e-6, '[TC34] T_c 公式 FAILED'

# ---- TC35: DuffingMembraneDynamics integrate_rk4 输出形状 ----
import numpy as np
np.random.seed(42)
duff_tc = DuffingMembraneDynamics(delta=0.3, alpha=-0.5, beta=1.0,
                                   gamma=0.4, omega=1.2, noise_amp=0.05, seed=99)
t_d, y_d = duff_tc.integrate_rk4(y0=[0.5, 0.0], t_span=(0.0, 10.0), n_steps=100)
assert t_d.shape == (101,), '[TC35] t 形状 FAILED'
assert y_d.shape == (101, 2), '[TC35] y 形状 FAILED'

# ---- TC36: Duffing Lyapunov 指数有限 ----
np.random.seed(42)
lam = duff_tc.lyapunov_exponent_estimate(y0=[0.5, 0.0], t_span=(0.0, 10.0), n_steps=100)
assert np.isfinite(lam), '[TC36] Lyapunov 指数非有限 FAILED'

# ---- TC37: FreeEnergyFieldGenerator.asymmetric_double_well 有限 ----
z_test = np.linspace(-3.0, 3.0, 50)
fe_z = FreeEnergyFieldGenerator.asymmetric_double_well(z_test, z0=1.2, V0=12.0, sigma=0.4, asym=1.5)
assert np.all(np.isfinite(fe_z)), '[TC37] 双势阱含非有限值 FAILED'
assert np.max(fe_z) > 0, '[TC37] 双势阱无正值 FAILED'

# ---- TC38: FreeEnergyFieldGenerator.generate_3d_field 形状 ----
fe_3d = FreeEnergyFieldGenerator.generate_3d_field(
    nx=8, ny=8, nz=16,
    xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), zlim=(-3.0, 3.0),
    model='double_well', z0=1.2, V0=12.0, sigma=0.4, asym=1.5
)
assert fe_3d.shape == (8, 8, 16), '[TC38] 3D 自由能场形状 FAILED'

# ---- TC39: DijkstraPermeation node_index/inverse_index 互逆 ----
dijk = DijkstraPermeation(nx=4, ny=4, nz=10)
for i, j, k in [(0, 0, 0), (3, 3, 9), (1, 2, 5)]:
    idx = dijk.node_index(i, j, k)
    ii, jj, kk = dijk.inverse_index(idx)
    assert (ii, jj, kk) == (i, j, k), f'[TC39] 索引互逆失败 ({i},{j},{k}) FAILED'

# ---- TC40: Dijkstra 小规模图最短路径 ----
import numpy as np
fe_small = np.zeros((4, 4, 10))
fe_small[:, :, 5:7] = 10.0  # 中央能垒
dijk_small = DijkstraPermeation(nx=4, ny=4, nz=10)
path, cost = dijk_small.find_mfep(fe_small, source_z_layer=0, target_z_layer=9, beta=0.1)
assert len(path) > 0, '[TC40] MFEP 路径为空 FAILED'
assert cost < np.inf, '[TC40] MFEP 代价无穷 FAILED'

# ---- TC41: chain_letter_style_symmetrization 对称性 ----
D_raw = np.random.rand(20, 20)
np.random.seed(42)
D_sym = chain_letter_style_symmetrization(D_raw)
assert np.allclose(D_sym, D_sym.T), '[TC41] 对称化后矩阵非对称 FAILED'

# ---- TC42: order_parameter_to_feature_vector 输出形状 ----
S2_fake = np.random.rand(16)
area_fake = np.random.rand(16)
head_fake = np.random.rand(16)
feats = order_parameter_to_feature_vector(S2_fake, area_fake, head_fake)
assert feats.shape == (16, 3), '[TC42] 特征向量形状 FAILED'

# ---- TC43: LipidBilayerSystem 全局序参数范围 [-0.5, 1] ----
sys_tc2 = LipidBilayerSystem(nx=6, ny=6)
s2_global = sys_tc2.global_order_parameter()
assert -0.5 - 1e-10 <= s2_global <= 1.0 + 1e-10, '[TC43] S_2 全局序参数超出范围 FAILED'

# ---- TC44: LipidBilayerSystem 局域序参数范围 ----
s2_local = sys_tc2.compute_local_order_parameter()
assert np.all(s2_local >= -0.5 - 1e-10), '[TC44] S_2 局域下限 FAILED'
assert np.all(s2_local <= 1.0 + 1e-10), '[TC44] S_2 局域上限 FAILED'

# ---- TC45: LipidBilayerSystem.get_positions 返回结构 ----
X_pos, Y_pos = sys_tc2.get_positions()
assert X_pos.shape == (6, 6), '[TC45] get_positions X 形状 FAILED'
assert Y_pos.shape == (6, 6), '[TC45] get_positions Y 形状 FAILED'

# ---- TC46: LipidBilayerSystem 总能量为有限值 ----
E_total = sys_tc2.compute_total_energy()
assert np.isfinite(E_total), '[TC46] 系统总能量非有限 FAILED'

# ---- TC47: MembraneDensityProfile 拟合与评估一致性 ----
dens = MembraneDensityProfile(z_min=-3.0, z_max=3.0, n_bernstein=8)
z_samples = np.linspace(-3.0, 3.0, 50)
rho_s = 0.5 * np.exp(-(z_samples - 1.5) ** 2 / 0.3) + 0.5 * np.exp(-(z_samples + 1.5) ** 2 / 0.3) + 0.2
dens.fit_density(z_samples, rho_s)
rho_eval = dens.evaluate(z_samples)
assert np.all(rho_eval >= -1e-12), '[TC47] 密度评估含负值 FAILED'

# ---- TC48: MembraneDensityProfile.headgroup_distance 返回非负 ----
d_hh = dens.headgroup_distance(threshold=0.5)
assert d_hh >= 0, '[TC48] 头基距离负值 FAILED'

# ---- TC49: PhaseDiagramBuilder 相图输出维度一致 ----
import numpy as np
pdb = PhaseDiagramBuilder(J=2.5)
T_vals, S_vals, P_vals = pdb.build_diagram(T_range=(250, 400), n_T=20)
assert len(T_vals) == 20, '[TC49] T 维度 FAILED'
assert len(S_vals) == 20, '[TC49] S 维度 FAILED'
assert len(P_vals) == 20, '[TC49] P 维度 FAILED'

# ---- TC50: PhaseDiagramBuilder.latent_heat 非负 ----
lh = pdb.latent_heat(Tc=300.0, S_gel=0.8, S_fluid=0.1)
assert lh > 0, '[TC50] 相变潜热非正 FAILED'

# ---- TC51: OrientationalOrderAnalysis 展开系数维度 ----
oa = OrientationalOrderAnalysis(n_max=6, alpha=0.0, beta=2.0)
cos_samples = np.cos(np.linspace(0.0, np.pi, 30))
coeffs = oa.expand_odf(cos_samples)
assert len(coeffs) == 7, '[TC51] 展开系数长度 FAILED'

# ---- TC52: OrientationalOrderAnalysis 重构 ODF 非负 ----
x_grid = np.linspace(-0.99, 0.99, 50)
f_recon = oa.reconstruct_odf(x_grid, coeffs)
assert np.all(np.isfinite(f_recon)), '[TC52] 重构 ODF 含非有限值 FAILED'
assert np.max(np.abs(f_recon)) > 0, '[TC52] 重构 ODF 全零 FAILED'

# ---- TC53: MDIntegrator 初始化与平衡化运行 ----
import numpy as np
np.random.seed(42)
sys_md = LipidBilayerSystem(nx=6, ny=6, dt_md=0.002)
md_int = MDIntegrator(sys_md, friction_gamma=0.5, seed=42)
e_trace, s2_trace = md_int.run_equilibration(n_steps=50)
assert len(e_trace) == 5, '[TC53] 能量轨迹长度 FAILED (50步每10步记录)'
assert np.all(np.isfinite(e_trace)), '[TC53] 能量含非有限值 FAILED'
assert np.all(np.isfinite(s2_trace)), '[TC53] S_2 含非有限值 FAILED'

# ---- TC54: LipidDistanceMatrix 距离矩阵对称 ----
from clustering_analysis import LipidDistanceMatrix
np.random.seed(42)
ldm = LipidDistanceMatrix(nx=4, ny=4, spatial_weight=2.0, sigma=1.5)
feat_test = np.random.rand(16, 3)
dist_mat = ldm.compute_distance_matrix(feat_test)
assert dist_mat.shape == (16, 16), '[TC54] 距离矩阵形状 FAILED'
assert np.allclose(dist_mat, dist_mat.T), '[TC54] 距离矩阵不对称 FAILED'
assert np.all(np.diag(dist_mat) >= 0), '[TC54] 对角线含负值 FAILED'

# ---- TC55: HierarchicalClustering 聚类与切割 ----
hc = HierarchicalClustering(dist_mat)
linkage = hc.cluster()
assert len(linkage) > 0, '[TC55] 聚类 linkage 为空 FAILED'
labels = hc.cut_tree(linkage, n_clusters=3)
assert len(np.unique(labels)) == 3, '[TC55] 聚类标签数 != 3 FAILED'

# ---- TC56: HierarchicalClustering.domain_size_distribution 总数 ----
sizes = hc.domain_size_distribution(linkage, n_clusters=3)
assert np.sum(sizes) == 16, '[TC56] 畴大小之和不等于总数 FAILED'

# ---- TC57: HierarchicalClustering.interface_energy_estimate 非负 ----
gamma_if = hc.interface_energy_estimate(linkage, temperature=300.0)
assert gamma_if >= 0, '[TC57] 界面能负值 FAILED'

# ---- TC58: CombinatorialEnumerators.stirling_second 边界 S(n,n)=1 ----
for n_val in [1, 2, 3, 5, 8]:
    assert CombinatorialEnumerators.stirling_second(n_val, n_val) == 1, \
        f'[TC58] S({n_val},{n_val}) != 1 FAILED'

# ---- TC59: SparseGridIntegration integrate 多项式 ----
sgi2 = SparseGridIntegration(dim_num=2, level_max=4)
def linear_func(x):
    return x[0]
integral_linear = sgi2.integrate(linear_func)
# ∫_{-1}^{1}∫_{-1}^{1} x dx dy = 0
assert abs(integral_linear) < 0.5, '[TC59] 奇函数积分应接近 0 FAILED'

# ---- TC60: 零阶 Bernstein 基 ----
B0 = bernstein_basis(0, np.array([0.5]))
assert B0.shape == (1,), '[TC60] n=0 Bernstein 维度 FAILED'
assert abs(B0[0] - 1.0) < 1e-12, '[TC60] B_{0,0}(0.5) != 1 FAILED'
