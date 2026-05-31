# ---- TC01: WeylHamiltonian 线性模型本征值解析验证 ----
ham_lin = WeylHamiltonian(model_type="linear", hbar=1.0, v_f=1.0)
k_test = np.array([0.3, 0.4, 0.0])
energies, eigvecs = ham_lin.eigenproblem(k_test)
k_norm = np.linalg.norm(k_test)
assert np.allclose(energies, np.array([-k_norm, k_norm])), '[TC01] 线性模型本征值与解析解不符 FAILED'

# ---- TC02: WeylHamiltonian 紧束缚模型能隙为正 ----
ham_tb = WeylHamiltonian(model_type="tight_binding", hbar=1.0, v_f=1.0)
k_test_tb = np.array([0.5, 0.5, 0.5])
energies_tb, _ = ham_tb.eigenproblem(k_test_tb)
gap_tb = band_gap(energies_tb.reshape(1, -1))[0]
assert gap_tb > 0, '[TC02] 紧束缚模型能隙必须为正 FAILED'

# ---- TC03: band_gap 返回值非负 ----
import numpy as np
np.random.seed(42)
energies_rand = np.random.randn(10, 2)
energies_rand[:, 1] = energies_rand[:, 0] + np.abs(energies_rand[:, 1] - energies_rand[:, 0]) + 0.01
gaps = band_gap(energies_rand)
assert np.all(gaps >= 0), '[TC03] band_gap 返回值含负数 FAILED'

# ---- TC04: velocity_operator 输出尺寸正确 ----
v_ops = velocity_operator(ham_lin, np.array([0.1, 0.2, 0.3]))
assert v_ops.shape == (3, 2, 2), '[TC04] velocity_operator 输出形状错误 FAILED'

# ---- TC05: berry_curvature_analytic_linear 反对称性验证 ----
k_berry = np.array([0.2, 0.3, 0.4])
Omega_ana = berry_curvature_analytic_linear(k_berry, chirality=1)
assert np.allclose(Omega_ana, -Omega_ana.T), '[TC05] Berry曲率解析解非反对称 FAILED'

# ---- TC06: berry_curvature_numeric 与解析解比较 ----
k_berry2 = np.array([0.5, 0.0, 0.0])
Omega_num = berry_curvature_numeric(ham_lin, k_berry2, band_index=0)
Omega_ana2 = berry_curvature_analytic_linear(k_berry2, chirality=-1)
# band_index=0 对应价带，chirality=-1
rel_diff = np.abs(Omega_num[0, 1] - Omega_ana2[0, 1]) / (np.abs(Omega_ana2[0, 1]) + 1e-10)
assert rel_diff < 0.05, '[TC06] Berry曲率数值解与解析解偏差过大 FAILED'

# ---- TC07: Berry相位沿闭合圆环约为 pi ----
import numpy as np
np.random.seed(42)
theta = np.linspace(0, 2 * np.pi, 80)
radius = 0.4
path_circle = np.array([[radius * np.cos(t), radius * np.sin(t), 0.0] for t in theta])
phase = berry_phase_1d(ham_lin, path_circle, band_index=0)
assert np.abs(np.abs(phase) - np.pi) < 0.2, '[TC07] Berry相位不接近 pi FAILED'

# ---- TC08: 线性模型Weyl节点在原点 ----
node_pos = ham_lin.weyl_node_position_linear()
assert np.allclose(node_pos, np.zeros(3)), '[TC08] 线性模型Weyl节点不在原点 FAILED'

# ---- TC09: ellipse_area 解析验证 ----
a_eye = np.eye(3)
vol_3d = ellipse_area(a_eye, 1.0)
expected_vol = 4.0 / 3.0 * np.pi  # 单位球体积
assert np.abs(vol_3d - expected_vol) < 1e-6, '[TC09] 椭球体积与单位球不符 FAILED'

# ---- TC10: uniform_kpoint_grid 输出点数与尺寸 ----
import numpy as np
np.random.seed(42)
bounds_test = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
grid = uniform_kpoint_grid(bounds_test, grid_size=5)
assert grid.shape == (125, 3), '[TC10] 均匀网格输出形状错误 FAILED'

# ---- TC11: sparse_grid_quadrature 权重和为积分域体积 ----
import numpy as np
np.random.seed(42)
pts, wts = sparse_grid_quadrature(dim=2, max_level=2)
# 2D [-1,1]^2 体积 = 4
assert np.abs(np.sum(wts) - 4.0) < 1e-10, '[TC11] 稀疏网格权重和不等于积分域体积 FAILED'

# ---- TC12: integrate_sparse_grid 对常数函数精确 ----
import numpy as np
np.random.seed(42)
result_const = integrate_sparse_grid(2, 2, lambda x: np.ones(len(x)))
assert np.abs(result_const - 4.0) < 1e-10, '[TC12] 常数函数积分结果错误 FAILED'

# ---- TC13: delaunay_triangulate_2d 输出类型与尺寸 ----
pts_2d = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, 0.3]])
tris = delaunay_triangulate_2d(pts_2d)
assert tris.dtype == np.int64 or tris.dtype == np.int32, '[TC13] 三角剖分索引类型错误 FAILED'
assert tris.shape[1] == 3, '[TC13] 三角剖分列数不为3 FAILED'

# ---- TC14: triangle_area_2d 解析验证 ----
tri_right = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
area_right = triangle_area_2d(tri_right)
assert np.abs(area_right - 0.5) < 1e-10, '[TC14] 直角三角形面积错误 FAILED'

# ---- TC15: triangulation_total_area 非负 ----
import numpy as np
np.random.seed(42)
rand_pts = np.random.rand(10, 2)
rand_tris = delaunay_triangulate_2d(rand_pts)
if len(rand_tris) > 0:
    total_area = triangulation_total_area(rand_pts, rand_tris)
    assert total_area >= 0, '[TC15] 三角剖分总面积为负 FAILED'

# ---- TC16: rk12_adaptive 演化到终止时间 ----
import numpy as np
np.random.seed(42)
t, y, e = rk12_adaptive(lambda t, y: np.array([1.0, -1.0]), tspan=(0.0, 1.0), y0=np.array([0.0, 0.0]), dt=0.1, tol=1e-4)
assert t[-1] >= 0.99, '[TC16] RK12未演化到终止时间 FAILED'
assert y.shape[0] == t.shape[0], '[TC16] RK12输出维度不匹配 FAILED'

# ---- TC17: periodic_lattice_dynamics 输出尺寸匹配输入 ----
import numpy as np
np.random.seed(42)
y_test = np.array([1.0, 2.0, 3.0, 4.0])
dydt = periodic_lattice_dynamics(y_test, force=8.0)
assert dydt.shape == y_test.shape, '[TC17] 周期性动力学输出维度不匹配 FAILED'

# ---- TC18: Nielsen-Ninomiya定理对成对荷为零 ----
import numpy as np
np.random.seed(42)
charges_pair = np.array([1, -1, 1, -1])
assert nielsen_ninomiya_theorem_check(charges_pair), '[TC18] 成对Weyl荷未通过NN定理 FAILED'

# ---- TC19: knapsack_rational 预算约束满足 ----
import numpy as np
np.random.seed(42)
gains_test = np.array([10.0, 8.0, 6.0, 4.0])
costs_test = np.array([2.0, 3.0, 4.0, 5.0])
budget_test = 7.0
x_opt, tc, tg = knapsack_rational(4, budget_test, gains_test, costs_test)
assert tc <= budget_test + 1e-10, '[TC19] 背包问题超出预算 FAILED'
assert np.all((x_opt >= 0) & (x_opt <= 1)), '[TC19] 背包选择比例越界 FAILED'

# ---- TC20: sort_by_profit_density 降序排列验证 ----
import numpy as np
np.random.seed(42)
from transport_optimizer import sort_by_profit_density
gains_s = np.array([3.0, 1.0, 2.0])
costs_s = np.array([1.0, 1.0, 1.0])
g_sorted, c_sorted = sort_by_profit_density(gains_s, costs_s)
assert g_sorted[0] >= g_sorted[1] >= g_sorted[2], '[TC20] 增益密度未按降序排列 FAILED'

# ---- TC21: alnorm 标准正态累积概率边界 ----
p_low = alnorm(-3.0, upper=False)
p_high = alnorm(3.0, upper=False)
assert 0.0 <= p_low <= 1.0, '[TC21] alnorm返回值超出[0,1] FAILED'
assert 0.0 <= p_high <= 1.0, '[TC21] alnorm返回值超出[0,1] FAILED'
assert p_high > p_low, '[TC21] alnorm单调性违反 FAILED'

# ---- TC22: dos_weyl_semimetal_analytic 零能量处为零 ----
e_test = np.array([0.0, 1.0, 2.0])
dos_vals = dos_weyl_semimetal_analytic(e_test, v_f=1.0)
assert dos_vals[0] == 0.0, '[TC22] Weyl DOS在E=0处不为零 FAILED'
assert dos_vals[1] < dos_vals[2], '[TC22] Weyl DOS单调性违反 FAILED'

# ---- TC23: test_hermite_exactness 偶次多项式误差小 ----
import numpy as np
np.random.seed(42)
errors = test_hermite_exactness(alpha=0.0, max_degree=5, n_points=5)
# 5点Gauss-Hermite应精确到9次，低次误差应极小
assert errors[0] < 1e-14, '[TC23] 0次多项式Hermite求积不精确 FAILED'
assert errors[2] < 1e-14, '[TC23] 2次多项式Hermite求积不精确 FAILED'

# ---- TC24: safe_normalize 零矢量安全处理 ----
import numpy as np
np.random.seed(42)
v_zero = np.zeros(3)
v_norm = safe_normalize(v_zero)
assert not np.any(np.isnan(v_norm)), '[TC24] safe_normalize对零矢量输出NaN FAILED'

# ---- TC25: rotation_matrix_from_axis_angle 旋转矩阵行列式为1 ----
import numpy as np
np.random.seed(42)
axis = np.array([0.0, 0.0, 1.0])
R = rotation_matrix_from_axis_angle(axis, np.pi / 2)
assert np.abs(np.linalg.det(R) - 1.0) < 1e-10, '[TC25] 旋转矩阵行列式不为1 FAILED'

# ---- TC26: fermi_dirac 高温极限趋近0.5 ----
import numpy as np
np.random.seed(42)
E_test = np.array([0.0, 0.0, 0.0])
f_vals = fermi_dirac(E_test, mu=0.0, kbt=100.0)
assert np.all(np.abs(f_vals - 0.5) < 0.01), '[TC26] Fermi-Dirac高温极限不为0.5 FAILED'

# ---- TC27: mesh2d_extract 字典字段完整 ----
import numpy as np
np.random.seed(42)
nodeco_2d = np.random.rand(5, 2)
elnode_2d = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
mesh2d = mesh2d_extract(nodeco_2d, elnode_2d)
assert mesh2d['dim'] == 2, '[TC27] 2D网格dim字段错误 FAILED'
assert mesh2d['node_num'] == 5, '[TC27] 2D网格节点数错误 FAILED'
assert mesh2d['element_num'] == 3, '[TC27] 2D网格元素数错误 FAILED'

# ---- TC28: node_values_to_elements 平均值正确 ----
import numpy as np
np.random.seed(42)
node_vals = np.array([1.0, 2.0, 3.0, 4.0])
elnode_test = np.array([[0, 1, 2], [1, 2, 3]])
elem_vals = node_values_to_elements(node_vals, elnode_test)
assert np.abs(elem_vals[0] - 2.0) < 1e-10, '[TC28] 元素平均值计算错误 FAILED'
assert np.abs(elem_vals[1] - 3.0) < 1e-10, '[TC28] 元素平均值计算错误 FAILED'

# ---- TC29: berry_curvature_significance 零均值返回p=1 ----
import numpy as np
np.random.seed(42)
omega_zeros = np.zeros(50)
z_score, p_val = berry_curvature_significance(omega_zeros)
assert p_val > 0.99, '[TC29] 零Berry曲率样本p值不接近1 FAILED'

# ---- TC30: chyper 参数错误返回ifault非零 ----
val, ifault = chyper(point=True, kk=10, ll=20, mm=100, nn=30)
assert ifault != 0, '[TC30] chyper无效参数未返回错误码 FAILED'
