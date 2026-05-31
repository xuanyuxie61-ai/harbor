# ---- TC01: r8_hyper_2f1 返回有限标量 ----
val = r8_hyper_2f1(0.5, 1.0, 1.5, 0.25)
assert np.isscalar(val), '[TC01] r8_hyper_2f1 返回非标量 FAILED'
assert np.isfinite(val), '[TC01] r8_hyper_2f1 返回非有限值 FAILED'

# ---- TC02: r8_psi 在 x=2.0 处返回已知解析值 ----
val = r8_psi(2.0)
assert np.isscalar(val), '[TC02] r8_psi 返回非标量 FAILED'
assert np.isfinite(val), '[TC02] r8_psi 返回非有限值 FAILED'
assert abs(val - (1.0 - 0.5772156649015329)) < 1e-6, '[TC02] r8_psi(2.0) 值不正确 FAILED'

# ---- TC03: gegenbauer_exactness_monomial 返回非负误差 ----
alpha_t = 0.5
order_t = 6
x_t, w_t = clenshaw_curtis_compute(order_t)
err = gegenbauer_exactness_monomial(0, alpha_t, order_t, w_t, x_t)
assert err >= 0.0, '[TC03] gegenbauer_exactness_monomial 返回负数误差 FAILED'

# ---- TC04: gegenbauer_integral 奇次幂积分为零（对称性） ----
gi_val = gegenbauer_integral(3, alpha=0.5)
assert abs(gi_val) < 1e-12, '[TC04] gegenbauer_integral(3, alpha=0.5) 应为0 FAILED'

# ---- TC05: membrane_vibration_bessel 输出形状与有限性 ----
r_arr = np.linspace(0.0, 1.0, 50)
mu_test = np.array([3.37561065, 4.27534072, 5.13562230])
u = membrane_vibration_bessel(r_arr, t=1.0, mu_n=mu_test)
assert u.shape == r_arr.shape, '[TC05] membrane_vibration_bessel 输出形状不匹配 FAILED'
assert np.all(np.isfinite(u)), '[TC05] membrane_vibration_bessel 输出含非有限值 FAILED'

# ---- TC06: screened_coulomb_green 返回正有限标量 ----
g = screened_coulomb_green(5.0, kappa=0.1, epsilon=80.0)
assert np.isscalar(g), '[TC06] screened_coulomb_green 返回非标量 FAILED'
assert np.isfinite(g), '[TC06] screened_coulomb_green 返回非有限值 FAILED'
assert g > 0.0, '[TC06] screened_coulomb_green 值应为正 FAILED'

# ---- TC07: clenshaw_curtis_compute 输出形状与 x 在 [-1,1] ----
x, w = clenshaw_curtis_compute(8)
assert x.shape == (8,), '[TC07] clenshaw_curtis_compute x 形状错误 FAILED'
assert w.shape == (8,), '[TC07] clenshaw_curtis_compute w 形状错误 FAILED'
assert np.all(np.isfinite(x)), '[TC07] x 含非有限值 FAILED'
assert np.all(np.isfinite(w)), '[TC07] w 含非有限值 FAILED'
assert np.all(np.abs(x) <= 1.0 + 1e-12), '[TC07] x 不在 [-1,1] FAILED'

# ---- TC08: jacobi_compute 输出形状与有限性 ----
x, w = jacobi_compute(7, alpha=0.5, beta=-0.5)
assert x.shape == (7,), '[TC08] jacobi_compute x 形状错误 FAILED'
assert w.shape == (7,), '[TC08] jacobi_compute w 形状错误 FAILED'
assert np.all(np.isfinite(x)), '[TC08] x 含非有限值 FAILED'
assert np.all(np.isfinite(w)), '[TC08] w 含非有限值 FAILED'

# ---- TC09: gen_hermite_compute 输出形状与有限性 ----
x, w = gen_hermite_compute(6, alpha=0.5)
assert x.shape == (6,), '[TC09] gen_hermite_compute x 形状错误 FAILED'
assert w.shape == (6,), '[TC09] gen_hermite_compute w 形状错误 FAILED'
assert np.all(np.isfinite(x)), '[TC09] x 含非有限值 FAILED'
assert np.all(np.isfinite(w)), '[TC09] w 含非有限值 FAILED'

# ---- TC10: laguerre_quadrature_rule 输出形状与有限性 ----
x, w = laguerre_quadrature_rule(8)
assert x.shape == (8,), '[TC10] laguerre x 形状错误 FAILED'
assert w.shape == (8,), '[TC10] laguerre w 形状错误 FAILED'
assert np.all(np.isfinite(x)), '[TC10] x 含非有限值 FAILED'
assert np.all(np.isfinite(w)), '[TC10] w 含非有限值 FAILED'
assert np.all(x >= 0), '[TC10] Laguerre 节点应为非负 FAILED'

# ---- TC11: integrate_triangle 单位三角形面积 = 0.5 ----
def f_one(p):
    return np.ones(p.shape[0])
tri_test = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
val_tri = integrate_triangle(f_one, tri_test, rule_index=1)
assert np.isfinite(val_tri), '[TC11] 三角形积分为非有限值 FAILED'
assert abs(val_tri - 0.5) < 1e-6, '[TC11] 单位三角形面积应为0.5 FAILED'

# ---- TC12: sparse_grid_total_poly_size 返回正整数 ----
sg_size = sparse_grid_total_poly_size(3, 2)
assert isinstance(sg_size, (int, np.integer)), '[TC12] sparse_grid_total_poly_size 返回非整数 FAILED'
assert sg_size > 0, '[TC12] sparse_grid_total_poly_size 返回非正数 FAILED'

# ---- TC13: sparse_grid_integrate 常数积分返回正值 ----
def f_const(pts):
    return np.ones(pts.shape[1])
sg_val = sparse_grid_integrate(2, 2, f_const)
assert np.isfinite(sg_val), '[TC13] 稀疏网格积分返回非有限值 FAILED'
assert sg_val > 0, '[TC13] 常数函数积分应为正 FAILED'

# ---- TC14: SparseMatrix 构造/to_dense/spmv 正确性 ----
import numpy as np
np.random.seed(42)
A_dense = np.random.randn(5, 5) * 0.5
sp = SparseMatrix(5, 5).from_dense(A_dense, drop_tol=1e-12)
A_recon = sp.to_dense()
assert A_recon.shape == (5, 5), '[TC14] to_dense 形状错误 FAILED'
v = np.ones(5)
Av = sp.spmv(v)
assert Av.shape == (5,), '[TC14] spmv 输出形状错误 FAILED'
assert np.all(np.isfinite(Av)), '[TC14] spmv 输出含非有限值 FAILED'

# ---- TC15: spdiags 构造与基本乘法 ----
d_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
sp_diag = spdiags(d_data, [0], 5, 5)
v5 = np.ones(5)
Av5 = sp_diag.spmv(v5)
assert Av5.shape == (5,), '[TC15] spdiags spmv 形状错误 FAILED'
assert np.allclose(Av5, np.array([1.0, 2.0, 3.0, 4.0, 5.0])), '[TC15] spdiags 对角线乘法错误 FAILED'

# ---- TC16: assemble_mass_stiffness_1d 形状、对称性与有限性 ----
M, K = assemble_mass_stiffness_1d(16, L=30.0)
n_nodes = 17
assert M.shape == (n_nodes, n_nodes), '[TC16] 质量矩阵形状错误 FAILED'
assert K.shape == (n_nodes, n_nodes), '[TC16] 刚度矩阵形状错误 FAILED'
assert np.allclose(M, M.T), '[TC16] M 不对称 FAILED'
assert np.allclose(K, K.T), '[TC16] K 不对称 FAILED'
assert np.all(np.isfinite(M)), '[TC16] M 含非有限值 FAILED'
assert np.all(np.isfinite(K)), '[TC16] K 含非有限值 FAILED'

# ---- TC17: solve_poisson_boltzmann_membrane 输出形状与有限性 ----
z, phi, eps, kappa_pb = solve_poisson_boltzmann_membrane(n=33)
assert z.shape == (33,), '[TC17] z 形状错误 FAILED'
assert phi.shape == (33,), '[TC17] phi 形状错误 FAILED'
assert eps.shape == (33,), '[TC17] eps 形状错误 FAILED'
assert kappa_pb.shape == (33,), '[TC17] kappa_pb 形状错误 FAILED'
assert np.all(np.isfinite(phi)), '[TC17] phi 含非有限值 FAILED'
assert np.all(np.isfinite(eps)), '[TC17] eps 含非有限值 FAILED'

# ---- TC18: cvt_triangle_uniform 输出形状与有限性 ----
import numpy as np
np.random.seed(42)
tri = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]], dtype=float)
g_cvt, tri_cvt = cvt_triangle_uniform(tri, n=5, sample_num=200, it_num=10)
assert g_cvt.shape == (5, 2), '[TC18] cvt_triangle_uniform 生成点形状错误 FAILED'
assert tri_cvt.ndim == 2, '[TC18] cvt_triangle_uniform 三角剖分应为2D FAILED'
assert tri_cvt.shape[1] == 3, '[TC18] cvt_triangle_uniform 三角剖分每行应为3列 FAILED'
assert np.all(np.isfinite(g_cvt)), '[TC18] 生成点含非有限值 FAILED'

# ---- TC19: place_lipid_bilayer 输出形状正确 ----
import numpy as np
np.random.seed(42)
upper, lower, up_z, low_z = place_lipid_bilayer(
    n_lipids_per_leaflet=10, protein_radius=8.0, box_xy=30.0,
    exclusion_radius=10.0, it_num=10
)
assert upper.shape[1] == 2, '[TC19] upper leaflet 应为 (n,2) FAILED'
assert lower.shape[1] == 2, '[TC19] lower leaflet 应为 (n,2) FAILED'
assert upper.shape[0] == lower.shape[0], '[TC19] 上下 leaflet 脂质数应相等 FAILED'

# ---- TC20: dock_drug_greedy_rotamer 返回有限值与正确长度 ----
import numpy as np
np.random.seed(42)
best_seq, best_energy, best_dih = dock_drug_greedy_rotamer(n_torsions=4, n_bins=8)
assert len(best_seq) == 4, '[TC20] 序列长度错误 FAILED'
assert np.isfinite(best_energy), '[TC20] best_energy 非有限 FAILED'
assert np.all(np.isfinite(best_dih)), '[TC20] best_dihedrals 含非有限值 FAILED'
assert np.all(np.abs(best_dih) <= np.pi + 1e-12), '[TC20] 二面角超出范围 FAILED'

# ---- TC21: backtrack_search 返回解列表 ----
def constraint_check(assignment):
    return len(assignment) < 2 or abs(assignment[-1] - assignment[-2]) <= 2
solutions = backtrack_search(n_vars=3, domain_size=5, constraint_checker=constraint_check, max_solutions=50)
assert isinstance(solutions, list), '[TC21] backtrack_search 返回非列表 FAILED'
assert len(solutions) > 0, '[TC21] backtrack_search 应至少找到1个解 FAILED'
for sol in solutions:
    assert len(sol) == 3, '[TC21] 解长度错误 FAILED'

# ---- TC22: thermodynamic_integration_binding_free_energy 返回有限值 ----
import numpy as np
np.random.seed(42)
dG, lam, dU = thermodynamic_integration_binding_free_energy(
    n_lambda=5, temperature=300.0, dim_conformational=2, sg_level=1
)
assert np.isfinite(dG), '[TC22] Delta_G 非有限 FAILED'
assert np.all(np.isfinite(lam)), '[TC22] lam 含非有限值 FAILED'
assert np.all(np.isfinite(dU)), '[TC22] dU 含非有限值 FAILED'

# ---- TC23: membrane_surface_free_energy 返回有限值 ----
tri_list = [np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]], dtype=float)]
def surf_ed(pts):
    return 0.03 * np.ones(pts.shape[0])
g_surf = membrane_surface_free_energy(tri_list, surf_ed, rule_index=2)
assert np.isscalar(g_surf), '[TC23] membrane_surface_free_energy 返回非标量 FAILED'
assert np.isfinite(g_surf), '[TC23] membrane_surface_free_energy 返回非有限值 FAILED'
assert g_surf > 0, '[TC23] 表面自由能应为正 FAILED'

# ---- TC24: validate_distance_matrix 返回正确字典与属性验证 ----
d_mat = np.array([[0.0, 3.8, 7.6], [3.8, 0.0, 3.8], [7.6, 3.8, 0.0]], dtype=float)
res_dm = validate_distance_matrix(d_mat)
assert isinstance(res_dm, dict), '[TC24] 返回非字典 FAILED'
assert res_dm['is_nonnegative'] == True, '[TC24] 非负性检测失败 FAILED'
assert res_dm['zero_diagonal'] == True, '[TC24] 零对角线检测失败 FAILED'
assert res_dm['is_symmetric'] == True, '[TC24] 对称性检测失败 FAILED'
assert res_dm['triangle_inequality'] == True, '[TC24] 三角不等式检测失败 FAILED'

# ---- TC25: validate_backbone_distances 完美坐标应100%通过 ----
coords = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0]], dtype=float)
expected = np.array([3.8, 3.8, 3.8])
pairs = [(0, 1), (1, 2), (2, 3)]
res_bb = validate_backbone_distances(coords, expected, pairs, tolerance=1.0)
assert isinstance(res_bb, dict), '[TC25] 返回非字典 FAILED'
assert res_bb['pass_rate'] == 1.0, '[TC25] 通过率应为100% FAILED'

# ---- TC26: lennard_jones_potential 输出有限且形状正确 ----
r = np.array([3.0, 4.0, 5.0])
u_lj = lennard_jones_potential(r)
assert u_lj.shape == (3,), '[TC26] LJ 势形状错误 FAILED'
assert np.all(np.isfinite(u_lj)), '[TC26] LJ 势含非有限值 FAILED'

# ---- TC27: debye_huckel_potential 输出有限且形状正确 ----
r2 = np.array([3.0, 5.0, 10.0])
u_dh = debye_huckel_potential(r2)
assert u_dh.shape == (3,), '[TC27] DH 势形状错误 FAILED'
assert np.all(np.isfinite(u_dh)), '[TC27] DH 势含非有限值 FAILED'

# ---- TC28: sawtooth_driver 返回有限标量 ----
st = sawtooth_driver(1.5, omega=2.0)
assert np.isscalar(st), '[TC28] sawtooth_driver 返回非标量 FAILED'
assert np.isfinite(st), '[TC28] sawtooth_driver 返回非有限值 FAILED'

# ---- TC29: coarse_grained_md_simulation 返回正确字典结构 ----
md_results = coarse_grained_md_simulation(
    n_steps=100, dt=0.001, temperature=300.0,
    n_protein_atoms=10, n_drug_atoms=4, n_lipid_atoms=10,
    box_size=np.array([30.0, 30.0, 30.0]),
    random_seed=42
)
assert isinstance(md_results, dict), '[TC29] results 非字典 FAILED'
assert 'avg_temperature' in md_results, '[TC29] results 缺 avg_temperature FAILED'
assert 'avg_potential' in md_results, '[TC29] results 缺 avg_potential FAILED'
assert np.isfinite(md_results['avg_temperature']), '[TC29] avg_temperature 非有限 FAILED'
assert np.isfinite(md_results['avg_potential']), '[TC29] avg_potential 非有限 FAILED'

# ---- TC30: cvt_3d_lumping 能量单调不减 ----
def wd(sx, sy, sz):
    r2 = sx**2 + sy**2
    return np.exp(-r2 / 400.0) * (1.0 + 0.5 * np.abs(sz))
import numpy as np
np.random.seed(42)
g_3d, e_3d, m_3d = cvt_3d_lumping(
    n=8, it_num=8, s_num=10,
    mu_fun=wd,
    box=(-25.0, 25.0, -25.0, 25.0, -30.0, 30.0),
)
assert e_3d.shape == (8,), '[TC30] 能量数组形状错误 FAILED'
assert np.all(np.isfinite(e_3d)), '[TC30] 能量含非有限值 FAILED'
assert e_3d[-1] <= e_3d[0] + 1e-10, '[TC30] 能量应单调不减 FAILED'
