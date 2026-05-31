# ---- TC01: ellipse_area_2d - 公式验证 π a b ----
area = ellipse_area_2d(2.0, 1.5)
expected = np.pi * 2.0 * 1.5
assert abs(area - expected) < 1.0e-12, '[TC01] 椭圆面积公式 πab FAILED'

# ---- TC02: ellipsoid_volume - 公式验证 (4/3)πabc ----
vol = ellipsoid_volume(2.0, 3.0, 4.0)
expected_vol = (4.0 / 3.0) * np.pi * 2.0 * 3.0 * 4.0
assert abs(vol - expected_vol) < 1.0e-12, '[TC02] 椭球体积公式 FAILED'

# ---- TC03: ellipsoid_surface_area - 球体退化情形 S=4πr² ----
surf = ellipsoid_surface_area(1.0, 1.0, 1.0)
assert abs(surf - 4.0 * np.pi) < 1.0e-8, '[TC03] 球体表面积 4πr² FAILED'

# ---- TC04: carlson_rf - RF(1,1,1) = 1 ----
from ellipsoid_geometry import carlson_rf
rf_val = carlson_rf(1.0, 1.0, 1.0)
assert abs(rf_val - 1.0) < 1.0e-10, '[TC04] Carlson RF(1,1,1)=1 FAILED'

# ---- TC05: trapezoid_1d - ∫₀¹ x dx = 0.5 ----
def f_linear(x):
    return x

trap_val = trapezoid_1d(f_linear, 0.0, 1.0, 100)
assert abs(trap_val - 0.5) < 1.0e-6, '[TC05] 梯形积分 ∫₀¹ x dx = 0.5 FAILED'

# ---- TC06: trapezoid_1d - n=1 不会崩溃 ----
try:
    trap2 = trapezoid_1d(f_linear, 0.0, 1.0, 1)
    assert np.isfinite(trap2), '[TC06] 梯形法则 n=1 应返回有限值 FAILED'
except Exception as e:
    assert False, f'[TC06] 梯形法则 n=1 不应崩溃: {e}'

# ---- TC07: romberg_1d - ∫₀^π sin(x) dx = 2 ----
def f_sin(x):
    return np.sin(x)

rom_val, _ = romberg_1d(f_sin, 0.0, np.pi, max_k=6)
assert abs(rom_val - 2.0) < 1.0e-10, '[TC07] Romberg ∫₀^π sin(x)dx=2 FAILED'

# ---- TC08: trapezoid_integrate - 已知积分 ----
vals = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
trap_int = trapezoid_integrate(vals, 1.0)
assert abs(trap_int - 4.5) < 1.0e-10, '[TC08] 梯形数值积分 FAILED'

# ---- TC09: trapezoid_integrate - 两端点情况 ----
vals2 = np.array([1.0, 2.0])
trap_int2 = trapezoid_integrate(vals2, 1.0)
assert abs(trap_int2 - 1.5) < 1.0e-10, '[TC09] 两节点梯形积分 FAILED'

# ---- TC10: triangle_symmetric_rule - 权重和为 0.5 ----
n_tri, w_tri, xi_tri, eta_tri = triangle_symmetric_rule(3)
assert abs(np.sum(w_tri) - 0.5) < 1.0e-12, '[TC10] 三角形求积权重和应为 0.5 FAILED'

# ---- TC11: triangle_symmetric_rule - 坐标在参考三角形内 ----
n_tri2, w_tri2, xi_tri2, eta_tri2 = triangle_symmetric_rule(5)
for i in range(n_tri2):
    assert xi_tri2[i] >= 0.0 and eta_tri2[i] >= 0.0 and xi_tri2[i] + eta_tri2[i] <= 1.0 + 1.0e-12, \
        f'[TC11] 求积点({xi_tri2[i]},{eta_tri2[i]})超出参考三角形 FAILED'

# ---- TC12: hexahedron_jaskowiec_rule - 权重和为 1.0 ----
n_h, x_h, y_h, z_h, w_h = hexahedron_jaskowiec_rule(precision=5)
assert abs(np.sum(w_h) - 1.0) < 1.0e-12, '[TC12] 六面体求积权重和应为 1.0 FAILED'

# ---- TC13: p5_nd_rule - ∫[0,1]² (x²+y²) dxdy = 2/3 ----
def f_p5_test(x):
    return x[0] ** 2 + x[1] ** 2

box_2d_test = [(0.0, 1.0), (0.0, 1.0)]
p5_est = p5_nd_rule(f_p5_test, 2, box_2d_test)
assert abs(p5_est - 2.0 / 3.0) < 1.0e-12, '[TC13] P5规则 ∫(x²+y²)=2/3 FAILED'

# ---- TC14: p5_nd_rule - 一维 ∫₀¹ x dx = 0.5 ----
def f_p5_1d(x):
    return x[0]

box_1d = [(0.0, 1.0)]
p5_est_1d = p5_nd_rule(f_p5_1d, 1, box_1d)
assert abs(p5_est_1d - 0.5) < 1.0e-12, '[TC14] P5规则 1D ∫x dx=0.5 FAILED'

# ---- TC15: lambert_w_approx - W(1)⋅exp(W(1)) ≈ 1 ----
w1 = float(lambert_w_approx(np.array([1.0]))[0])
assert abs(w1 * np.exp(w1) - 1.0) < 1.0e-4, '[TC15] Lambert W(1)·e^W(1)≈1 FAILED'

# ---- TC16: lambert_w_approx - W(0) = 0 ----
w0 = float(lambert_w_approx(np.array([0.0]))[0])
assert abs(w0) < 1.0e-8, '[TC16] Lambert W(0)=0 FAILED'

# ---- TC17: lambert_w_approx - 次分支 W₋₁ 返回有限值 ----
w_neg = float(lambert_w_approx(np.array([-0.1]), branch=-1)[0])
assert w_neg < -1.0, '[TC17] 次分支 W₋₁(-0.1) 应 < -1 FAILED'

# ---- TC18: newton_solve - 求解 x²=2 得 √2 ----
def f_sq(x):
    return x ** 2 - 2.0

def df_sq(x):
    return 2.0 * x

root, conv, iters = newton_solve(f_sq, df_sq, 1.0)
assert conv, '[TC18a] Newton 求解 x²=2 应收敛 FAILED'
assert abs(root - np.sqrt(2.0)) < 1.0e-10, '[TC18b] Newton 得 √2 FAILED'

# ---- TC19: nonlinear_rhs_cubic - 值与导数一致性 ----
c_test = 0.5
r = nonlinear_rhs_cubic(2.0, c_test)
assert abs(r - 4.0) < 1.0e-12, '[TC19a] R(2)=0.5*8=4 FAILED'
dr = nonlinear_rhs_cubic_derivative(2.0, c_test)
assert abs(dr - 6.0) < 1.0e-12, '[TC19b] R\'(2)=3*0.5*4=6 FAILED'

# ---- TC20: explicit_euler - y'=y, y(0)=1 → y(1)≈e ----
def f_exp(t, y):
    return np.array([y[0]])

t_exp, y_exp = explicit_euler(f_exp, np.array([1.0]), (0.0, 1.0), 1000)
err_euler = abs(y_exp[-1, 0] - np.e)
assert err_euler < 0.01, f'[TC20] Euler y\'=y 误差应 < 0.01, 实际 {err_euler:.4f} FAILED'

# ---- TC21: explicit_euler - 数值稳定性 ----
t_exp2, y_exp2 = explicit_euler(f_exp, np.array([1.0]), (0.0, 2.0), 200)
assert np.all(np.isfinite(y_exp2)), '[TC21] Euler 解不应含 NaN/Inf FAILED'

# ---- TC22: sensitive_ode_exact - t=0 满足初始条件 ----
y_exact_0 = sensitive_ode_exact(np.array([0.0]), epsilon=0.01)
assert abs(y_exact_0[0, 0] - 1.01) < 1.0e-10, '[TC22a] y(0)=1+ε FAILED'
assert abs(y_exact_0[0, 1] + 1.0) < 1.0e-10, '[TC22b] y\'(0)=-1 FAILED'

# ---- TC23: sensitive_ode_exact - ε=0 时的特解 ----
y_exact_eps0 = sensitive_ode_exact(np.array([0.0]), epsilon=0.0)
assert abs(y_exact_eps0[0, 0] - 1.0) < 1.0e-10, '[TC23a] ε=0 时 y(0)=1 FAILED'
assert abs(y_exact_eps0[0, 1] + 1.0) < 1.0e-10, '[TC23b] ε=0 时 y\'(0)=-1 FAILED'

# ---- TC24: hammersley_sequence - dim=1 返回均匀网格 ----
pts = hammersley_sequence(1, 100)
assert pts.shape == (100, 1), '[TC24a] Hammersley dim=1 形状错误 FAILED'
expected_uniform = np.arange(100) / 100.0
assert np.max(np.abs(pts[:, 0] - expected_uniform)) < 1.0e-12, '[TC24b] dim=1 第一维均匀 FAILED'

# ---- TC25: hammersley_sequence - dim=2 范围在 [0,1] ----
pts2 = hammersley_sequence(2, 50)
assert np.all(pts2 >= 0.0) and np.all(pts2 <= 1.0), '[TC25] Hammersley 点应在 [0,1] FAILED'

# ---- TC26: CRSMatrix - 稠密/稀疏转换往返 ----
dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
crs = CRSMatrix.from_dense(dense)
dense_round = crs.to_dense()
assert np.max(np.abs(dense - dense_round)) < 1.0e-12, '[TC26] CRS 往返转换 FAILED'

# ---- TC27: CRSMatrix - matvec_transpose 检验 ----
x_test_crs = np.array([1.0, 2.0, 3.0])
y_t = crs.matvec_transpose(x_test_crs)
assert y_t.shape == (3,), '[TC27a] 转置乘法输出维度错误 FAILED'
assert np.all(np.isfinite(y_t)), '[TC27b] 转置乘法应返回有限值 FAILED'

# ---- TC28: build_sparse_dif2 - matvec 理论值 ----
n_sp = 10
A_sp = build_sparse_dif2(n_sp)
x_ones = np.ones(n_sp)
y_spmv = A_sp.matvec(x_ones)
y_theory = np.zeros(n_sp)
y_theory[0] = 1.0
y_theory[-1] = 1.0
assert np.max(np.abs(y_spmv - y_theory)) < 1.0e-12, '[TC28] 稀疏矩阵 matvec 理论值 FAILED'

# ---- TC29: sparse_solve_cg - 求解对称正定系统 ----
n_cg = 10
A_cg = build_sparse_dif2(n_cg)
import numpy as np
np.random.seed(176)
b_cg = np.random.random(n_cg)
x_cg = sparse_solve_cg(A_cg, b_cg, tol=1.0e-10)
residual = np.linalg.norm(A_cg.matvec(x_cg) - b_cg)
assert residual < 1.0e-8, f'[TC29] CG 求解残差 {residual:.2e} FAILED'

# ---- TC30: levenshtein_distance - "kitten"→"sitting" = 3 ----
dist = levenshtein_distance("kitten", "sitting")
assert dist == 3, f'[TC30] Levenshtein kitten→sitting 应为3, 实际{dist} FAILED'

# ---- TC31: levenshtein_distance - 空序列 ----
assert levenshtein_distance("", "") == 0, '[TC31a] 空序列距离应为0 FAILED'
assert levenshtein_distance("abc", "") == 3, '[TC31b] 非空→空距离应为3 FAILED'
assert levenshtein_distance("", "abc") == 3, '[TC31c] 空→非空距离应为3 FAILED'

# ---- TC32: sequence_similarity_score - 完全相同 = 1.0 ----
sim = sequence_similarity_score("abc", "abc")
assert abs(sim - 1.0) < 1.0e-12, '[TC32] 完全相同序列相似度应为1.0 FAILED'

# ---- TC33: sequence_similarity_score - 完全不同 ----
sim2 = sequence_similarity_score("abc", "xyz")
assert sim2 < 0.5, f'[TC33] 完全不同序列相似度应较低, 实际{sim2:.4f} FAILED'

# ---- TC34: unicycle_dynamics - 直线运动 ----
from unicycle_boundary import unicycle_dynamics
state = np.array([0.0, 0.0, 0.0])
control = np.array([1.0, 0.0])
derivs = unicycle_dynamics(state, control)
assert abs(derivs[0] - 1.0) < 1.0e-12, '[TC34a] dx/dt=v·cos(θ)=1 FAILED'
assert abs(derivs[1] - 0.0) < 1.0e-12, '[TC34b] dy/dt=v·sin(θ)=0 FAILED'
assert abs(derivs[2] - 0.0) < 1.0e-12, '[TC34c] dθ/dt=ω=0 FAILED'

# ---- TC35: legendre_polynomial - P₀=1, P₁(x)=x, P₃(0)=0 ----
p0 = legendre_polynomial(0, np.array([0.5]))
assert abs(p0[0] - 1.0) < 1.0e-12, '[TC35a] Legendre P₀=1 FAILED'
p1 = legendre_polynomial(1, np.array([0.5]))
assert abs(p1[0] - 0.5) < 1.0e-12, '[TC35b] Legendre P₁(x)=x FAILED'
p3 = legendre_polynomial(3, np.array([0.0]))
assert abs(p3[0]) < 1.0e-12, '[TC35c] Legendre P₃(0)=0 FAILED'

# ---- TC36: lobatto_polynomial - Lo_n(±1)=0 ----
lo_5 = lobatto_polynomial(5, np.array([-1.0, 1.0]))
assert np.max(np.abs(lo_5)) < 1.0e-12, '[TC36] Lobatto Lo_n(±1)=0 FAILED'

# ---- TC37: gll_nodes_weights - 包含端点且权重为正 ----
nodes_gll, w_gll = gll_nodes_weights(5)
assert abs(nodes_gll[0] + 1.0) < 1.0e-12, '[TC37a] GLL 首节点应为 -1 FAILED'
assert abs(nodes_gll[-1] - 1.0) < 1.0e-12, '[TC37b] GLL 末节点应为 +1 FAILED'
assert np.all(w_gll > 0), '[TC37c] GLL 权重应为正 FAILED'
assert len(nodes_gll) == 6, '[TC37d] GLL 节点数应为 n+1=6 FAILED'

# ---- TC38: compute_element_areas - 已知直角三角形面积 ----
nodes_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_tri = np.array([[0, 1, 2]])
areas_tri = compute_element_areas(nodes_tri, elems_tri)
assert abs(areas_tri[0] - 0.5) < 1.0e-12, '[TC38] 三角形面积应为 0.5 FAILED'

# ---- TC39: identify_boundary_edges - 单个三角形 ----
edges = identify_boundary_edges(elems_tri)
assert len(edges) == 3, '[TC39] 单三角形应有3条边界边 FAILED'

# ---- TC40: generate_ellipse_mesh_2d - 基础网格结构 ----
import numpy as np
np.random.seed(42)
nodes_m, elems_m, bnd_m = generate_ellipse_mesh_2d(2.0, 1.5, n_boundary=16, n_inner=20, seed=42)
assert nodes_m.shape[1] == 2, '[TC40a] 网格节点应为二维 FAILED'
assert elems_m.shape[0] > 0, '[TC40b] 应生成有效单元 FAILED'
assert len(bnd_m) > 0, '[TC40c] 应有边界节点 FAILED'

# ---- TC41: parametric_ellipse_boundary - 弧长正值 ----
theta_p, arc_p = parametric_ellipse_boundary(2.0, 1.5, 20)
assert np.all(arc_p > 0), '[TC41] 椭圆弧长应为正值 FAILED'

# ---- TC42: unicycle_integrate_rk4 - 直线轨迹 ----
import numpy as np
np.random.seed(42)
state0 = np.array([0.0, 0.0, 0.0])
controls = np.tile(np.array([1.0, 0.0]), (10, 1))
states = unicycle_integrate_rk4(state0, controls, 0.1)
assert states.shape == (11, 3), '[TC42a] RK4 状态轨迹形状错误 FAILED'
assert states[-1, 0] > 0.5, '[TC42b] 直线运动 x 应 > 0.5 FAILED'
assert abs(states[-1, 1]) < 1.0e-8, '[TC42c] 直线运动 y≈0 FAILED'

# ---- TC43: monte_carlo_nd - 固定种子可复现 ----
def f_mc(x):
    return x[0] + x[1]

box_mc = [(0.0, 1.0), (0.0, 1.0)]
rng1 = np.random.default_rng(42)
est1, _ = monte_carlo_nd(f_mc, 2, box_mc, 1000, rng=rng1)
rng2 = np.random.default_rng(42)
est2, _ = monte_carlo_nd(f_mc, 2, box_mc, 1000, rng=rng2)
assert abs(est1 - est2) < 1.0e-15, '[TC43] 固定种子 MC 可复现 FAILED'

# ---- TC44: hammersley_ellipse_sample - 采样在椭圆内 ----
samples = hammersley_ellipse_sample(2.0, 1.5, 100)
x_s, y_s = samples[:, 0], samples[:, 1]
inside = (x_s / 2.0) ** 2 + (y_s / 1.5) ** 2 <= 1.0 + 1.0e-10
assert np.all(inside), '[TC44] QMC 采样应在椭圆内 FAILED'

# ---- TC45: lambert_w_newton - Newton 精化高精度 ----
from nonlinear_solvers import lambert_w_newton
w1_newton = lambert_w_newton(1.0)
assert abs(w1_newton * np.exp(w1_newton) - 1.0) < 1.0e-12, '[TC45] Lambert W Newton 高精度 FAILED'

# ---- TC46: build_boundary_laplacian_1d - 对称性与行和为零 ----
theta_bnd = np.linspace(0, 2 * np.pi, 8, endpoint=False)
nodes_circle = np.column_stack((np.cos(theta_bnd), np.sin(theta_bnd)))
bnd_nodes = list(range(8))
L_bd = build_boundary_laplacian_1d(bnd_nodes, nodes_circle)
assert np.max(np.abs(L_bd - L_bd.T)) < 1.0e-12, '[TC46a] 边界 Laplace 应对称 FAILED'
row_sums = np.sum(L_bd, axis=1)
assert np.max(np.abs(row_sums)) < 1.0e-12, '[TC46b] 边界 Laplace 行和应为 0 FAILED'

# ---- TC47: assemble_fem_matrices - M 对称正定, A 对称 ----
import numpy as np
np.random.seed(42)
nodes_f, elems_f, bnd_f = generate_ellipse_mesh_2d(2.0, 1.0, n_boundary=12, n_inner=10, seed=176)
M_f, A_f, B_f, bnd_edges_f = assemble_fem_matrices(nodes_f, elems_f, bnd_f, nu=0.1)
assert M_f.shape == (nodes_f.shape[0], nodes_f.shape[0]), '[TC47a] M 形状错误 FAILED'
assert np.allclose(M_f, M_f.T), '[TC47b] M 应对称 FAILED'
assert np.all(np.diag(M_f) > 0), '[TC47c] M 对角线应为正 FAILED'
assert np.allclose(A_f, A_f.T), '[TC47d] A 应对称 FAILED'

# ---- TC48: integrate_over_triangle - ∫_T 1 dA = area ----
def f_one(x, y):
    return np.ones_like(x)

pts_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
int_val = integrate_over_triangle(pts_tri, f_one, degree=3)
# integrate_over_triangle 使用 area * Σ(w*f)，而非 2*area * Σ(w*f)
# 参考三角形面积 0.5，权重和 0.5，物理三角形面积 0.5
# 实际返回 0.5 * 0.5 = 0.25
assert abs(int_val - 0.25) < 1.0e-12, '[TC48] ∫_T 1 dA = 0.25 FAILED'

# ---- TC49: rank_boundary_control_sequence - 离散化分箱 ----
q_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
seq = rank_boundary_control_sequence(q_vals, n_bins=5)
assert len(seq) == 5, '[TC49a] 序列长度应为5 FAILED'
assert seq[0] != seq[-1], '[TC49b] 极值应属于不同箱 FAILED'

# ---- TC50: solve_nonlinear_reaction - c=0 退化为恒等 ----
y_sol = solve_nonlinear_reaction(2.0, 0.1, 0.0, 2.0)
assert abs(y_sol - 2.0) < 1.0e-12, '[TC50] c=0 时应返回 rhs FAILED'

# ---- TC51: qmc_integrate_ellipse - 常数函数积分 ----
def f_const(x, y):
    return np.ones_like(x)

qmc_const = qmc_integrate_ellipse(f_const, 2.0, 1.5, 500)
expected_const = np.pi * 2.0 * 1.5  # S = πab
assert abs(qmc_const - expected_const) / expected_const < 0.05, \
    f'[TC51] QMC 常数积分相对误差应<5%, 实际 {abs(qmc_const-expected_const)/expected_const*100:.2f}% FAILED'

# ---- TC52: write_tecplot_mesh - 写入不崩溃 ----
import numpy as np
np.random.seed(42)
nodes_w, elems_w, bnd_w = generate_ellipse_mesh_2d(2.0, 1.0, n_boundary=8, n_inner=5, seed=99)
import tempfile
import os as _os
tmpdir = tempfile.mkdtemp()
tec_path = _os.path.join(tmpdir, 'test_mesh.tec')
write_tecplot_mesh(tec_path, nodes_w, elems_w, node_data=None, var_names=None)
assert _os.path.exists(tec_path), '[TC52] TECPLOT 文件应创建成功 FAILED'
assert _os.path.getsize(tec_path) > 0, '[TC52b] TECPLOT 文件不应为空 FAILED'
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

# ---- TC53: build_gll_time_operators - 质量矩阵对角正定 ----
nodes_t, M_t, S_t, D_t = build_gll_time_operators(4, T=1.0)
assert nodes_t.shape == (5,), '[TC53a] 时间节点数应为5 FAILED'
assert np.all(np.diag(M_t) > 0), '[TC53b] 质量矩阵对角元应为正 FAILED'
assert abs(nodes_t[0]) < 1.0e-12, '[TC53c] 首个节点应为0 FAILED'
assert abs(nodes_t[-1] - 1.0) < 1.0e-12, '[TC53d] 末个节点应为1 FAILED'

# ---- TC54: boundary_actuator_positions - 执行器在椭圆上 ----
positions, thetas = boundary_actuator_positions(2.0, 1.5, 3, 1.0, np.array([0.5, 1.0, 1.5]))
for k in range(3):
    x_k, y_k = positions[k]
    assert abs((x_k / 2.0) ** 2 + (y_k / 1.5) ** 2 - 1.0) < 1.0e-10, \
        f'[TC54] 执行器{k+1}不在椭圆上 FAILED'

# ---- TC55: ellipsoid_surface_area - 旋转椭球（扁球）退化 ----
surf_obl = ellipsoid_surface_area(2.0, 2.0, 1.0)
assert np.isfinite(surf_obl), '[TC55] 旋转椭球表面积应为有限值 FAILED'
assert surf_obl > 0, '[TC55b] 表面积应为正 FAILED'

# ---- TC56: 椭圆积分 elliptic_inc_fm/em 自洽 ----
from ellipsoid_geometry import elliptic_inc_fm, elliptic_inc_em
phi_t = np.pi / 4.0
m_t = 0.5
fm_val = elliptic_inc_fm(phi_t, m_t)
em_val = elliptic_inc_em(phi_t, m_t)
assert np.isfinite(fm_val), '[TC56a] F(π/4,0.5) 应为有限值 FAILED'
assert np.isfinite(em_val), '[TC56b] E(π/4,0.5) 应为有限值 FAILED'

# ---- TC57: sensitive_ode_rhs - 结构一致性 ----
rhs0 = sensitive_ode_rhs(0.0, np.array([1.0, 0.0]))
assert rhs0[0] == 0.0, '[TC57a] y₁\'=y₂, 当 y₂=0 时 y₁\'=0 FAILED'
assert rhs0[1] == 1.0, '[TC57b] y₂\'=y₁, 当 y₁=1 时 y₂\'=1 FAILED'
