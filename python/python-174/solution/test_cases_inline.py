# 测试所需额外导入 (部分函数仅在main.py函数体内导入)
from nbody_kernel import monomial_value, coulomb_force_direct, pwc_kernel_approx, evaluate_pwc_kernel, build_transition_matrix_from_neighbors, kernel_gradient_laplacian
from spherical_geometry import cartesian_to_spherical, spherical_to_cartesian, great_circle_distance, spherical_cap_area, solid_angle, quadrilateral_area_2d, legendre_associated_normalized, spherical_harmonic_basis, circle_points_on_plane
from monte_carlo_sampler import walker_build, walker_sampler, disk01_positive_sample, coin_biased, alnorm, mc_estimate_integral
from error_analysis import kolmogorov_smirnov_statistic, markov_chain_steady_state, second_eigenvalue_rate
from translation_operators import m2m_translate, l2l_translate, compute_translation_matrix
from adaptive_mesh import refine_triangle_midpoint

# ---- TC01: monomial_value 返回标量ndarray ----
import numpy as np
x = monomial_value([1, 0, 0], np.array([[2.0, 0.0, 0.0]]))
assert isinstance(x, np.ndarray), '[TC01] monomial_value 应返回 ndarray FAILED'
assert abs(x[0] - 2.0) < 1e-12, '[TC01] monomial_value x^1 在 (2,0,0) 应为 2.0 FAILED'

# ---- TC02: monomial_value 多指数 x^2*y^1 ----
import numpy as np
x = monomial_value([2, 1, 0], np.array([[3.0, 2.0, 1.0]]))
assert abs(x[0] - 18.0) < 1e-12, '[TC02] monomial_value x^2*y 在 (3,2,1) 应为 18 FAILED'

# ---- TC03: coulomb_potential_direct 两粒子解析验证 ----
import numpy as np
np.random.seed(42)
pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
chg = np.array([1.0, -1.0])
pot = coulomb_potential_direct(pts, chg)
assert abs(pot[0] + 1.0) < 1e-10, '[TC03] 两粒子势能点0应为 -1.0 FAILED'
assert abs(pot[1] - 1.0) < 1e-10, '[TC03] 两粒子势能点1应为 1.0 FAILED'

# ---- TC04: coulomb_potential_direct 软化参数 ----
import numpy as np
np.random.seed(42)
pts_eq = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
chg_eq = np.array([1.0, 1.0])
pot_eq = coulomb_potential_direct(pts_eq, chg_eq, epsilon=0.5)
assert abs(pot_eq[0] - 2.0) < 1e-10, '[TC04] 软化参数0.5下势能应为 2.0 FAILED'

# ---- TC05: coulomb_potential_direct 有限性 ----
import numpy as np
np.random.seed(42)
pts_10 = np.random.uniform(-1, 1, size=(10, 3))
chg_10 = np.random.uniform(-1, 1, size=10)
pot_10 = coulomb_potential_direct(pts_10, chg_10)
assert np.all(np.isfinite(pot_10)), '[TC05] 势能应全为有限值 FAILED'

# ---- TC06: coulomb_force_direct 输出形状 ----
import numpy as np
np.random.seed(42)
pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
chg = np.array([1.0, -1.0])
f = coulomb_force_direct(pts, chg)
assert f.shape == (2, 3), '[TC06] 力输出形状应为 (2,3) FAILED'
assert np.all(np.isfinite(f)), '[TC06] 力应全为有限值 FAILED'

# ---- TC07: pwc_kernel_approx 输出形状与单调性 ----
import numpy as np
nb, nv = pwc_kernel_approx(0.1, 2.0, 10)
assert len(nb) == 11, '[TC07] 断点数应为 n_segments+1=11 FAILED'
assert len(nv) == 10, '[TC07] 常数值数应为 10 FAILED'
assert nb[0] == 0.1, '[TC07] 第一个断点应为 r_min=0.1 FAILED'
assert nb[-1] == 2.0, '[TC07] 最后一个断点应为 r_max=2.0 FAILED'

# ---- TC08: evaluate_pwc_kernel 单值 ----
import numpy as np
nb, nv = pwc_kernel_approx(0.5, 2.0, 3)
val = evaluate_pwc_kernel(0.6, nb, nv)
assert float(val) > 0, '[TC08] 分段核函数值应为正 FAILED'

# ---- TC09: build_transition_matrix_from_neighbors 行和为1 ----
import numpy as np
np.random.seed(42)
nc = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
T = build_transition_matrix_from_neighbors(nc, 3)
row_sums = np.sum(T, axis=1)
assert np.all(np.abs(row_sums - 1.0) < 1e-12), '[TC09] 转移矩阵行和应为 1 FAILED'

# ---- TC10: kernel_gradient_laplacian 源外拉普拉斯量为0 ----
import numpy as np
np.random.seed(42)
pts_k = np.array([[0.0, 0.0, 0.0]])
chg_k = np.array([1.0])
target_k = np.array([1.0, 0.0, 0.0])
pot_k, grad_k, lap_k = kernel_gradient_laplacian(pts_k, chg_k, target_k)
assert abs(lap_k - 0.0) < 1e-12, '[TC10] 源外拉普拉斯量应为 0 FAILED'
assert abs(pot_k - 1.0) < 1e-10, '[TC10] 1/r势能在r=1处应为 1.0 FAILED'

# ---- TC11: cartesian_to_spherical 基本值 ----
import numpy as np
np.random.seed(42)
xyz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
r, theta, phi = cartesian_to_spherical(xyz)
assert abs(r[0] - 1.0) < 1e-12, '[TC11] (1,0,0) 半径应为 1 FAILED'
assert abs(theta[1]) < 1e-12, '[TC11] (0,0,1) theta 应为 0 FAILED'
assert abs(r[1] - 1.0) < 1e-12, '[TC11] (0,0,1) 半径应为 1 FAILED'

# ---- TC12: spherical_to_cartesian 往返 ----
import numpy as np
np.random.seed(42)
xyz2 = spherical_to_cartesian(1.0, np.pi / 2, 0.0)
assert abs(xyz2[0, 0] - 1.0) < 1e-10, '[TC12] sin(pi/2)*cos(0) 应为 1.0 FAILED'
assert abs(xyz2[0, 2]) < 1e-10, '[TC12] cos(pi/2) 应为 0 FAILED'

# ---- TC13: great_circle_distance 赤道对跖点 ----
import numpy as np
d = great_circle_distance(1.0, np.pi / 2, 0.0, np.pi / 2, np.pi)
assert abs(d - np.pi) < 1e-8, '[TC13] 赤道对跖点大圆距离应为 pi FAILED'

# ---- TC14: spherical_cap_area 半球 ----
import numpy as np
np.random.seed(42)
area = spherical_cap_area(1.0, np.pi / 2)
assert abs(area - 2.0 * np.pi) < 1e-10, '[TC14] 半球冠面积应为 2*pi FAILED'

# ---- TC15: solid_angle 全空间 ----
import numpy as np
omega = solid_angle(1.0, 4.0 * np.pi)
assert abs(omega - 4.0 * np.pi) < 1e-10, '[TC15] 全空间立体角应为 4*pi FAILED'

# ---- TC16: quadrilateral_area_2d 单位正方形 ----
import numpy as np
np.random.seed(42)
quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
area_q = quadrilateral_area_2d(quad)
assert abs(area_q - 1.0) < 1e-10, '[TC16] 单位正方形面积应为 1.0 FAILED'

# ---- TC17: legendre_associated_normalized P_0^0 归一化 ----
import numpy as np
cx = legendre_associated_normalized(0, 0, 0.0)
assert abs(cx[0] - 1.0 / np.sqrt(4.0 * np.pi)) < 1e-12, '[TC17] P_0^0(0) 归一化值应为 1/sqrt(4pi) FAILED'

# ---- TC18: legendre_associated_normalized 高阶有限性 ----
import numpy as np
cx = legendre_associated_normalized(5, 2, 0.5)
assert np.all(np.isfinite(cx)), '[TC18] 归一化连带Legendre应全有限 FAILED'

# ---- TC19: spherical_harmonic_basis 输出结构 ----
import numpy as np
np.random.seed(42)
c, s = spherical_harmonic_basis(3, np.pi / 2, 0.0)
assert len(c) == 4, '[TC19] 球谐基 c_all 长度应为 L+1=4 FAILED'
assert len(s) == 4, '[TC19] 球谐基 s_all 长度应为 L+1=4 FAILED'

# ---- TC20: uniform_sphere_sample 单位球面 ----
import numpy as np
np.random.seed(42)
samples = uniform_sphere_sample(100)
norms = np.linalg.norm(samples, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-12), '[TC20] 球面采样点应在单位球面上 FAILED'

# ---- TC21: circle_points_on_plane 输出形状 ----
import numpy as np
np.random.seed(42)
pts_circ = circle_points_on_plane(np.array([0.0, 0.0, 0.0]), 1.0, np.array([0.0, 0.0, 1.0]), num_points=16)
assert pts_circ.shape == (16, 3), '[TC21] 圆周点形状应为 (16,3) FAILED'

# ---- TC22: compute_bounding_sphere 基本 ----
import numpy as np
np.random.seed(42)
pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
center_cbs, radius_cbs = compute_bounding_sphere(pts)
assert radius_cbs > 0, '[TC22] 包围球半径应为正 FAILED'
assert center_cbs.shape == (3,), '[TC22] 球心形状应为 (3,) FAILED'

# ---- TC23: walker_build 输出类型 ----
import numpy as np
np.random.seed(42)
prob = np.array([0.1, 0.2, 0.3, 0.4])
y, a = walker_build(prob)
assert len(y) == 4 and len(a) == 4, '[TC23] Walker表长度应为 4 FAILED'
assert np.all(y >= 0) and np.all(y <= 1.0 + 1e-12), '[TC23] 阈值应在 [0,1] 范围内 FAILED'

# ---- TC24: walker_sampler 确定性种子 ----
import numpy as np
np.random.seed(42)
prob = np.array([1.0, 0.0])
y, a = walker_build(prob)
np.random.seed(42)
idx = walker_sampler(y, a)
assert 0 <= idx <= 1, '[TC24] Walker采样索引应在范围内 FAILED'

# ---- TC25: disk01_positive_sample 正半平面与圆内 ----
import numpy as np
np.random.seed(42)
pts_disk = disk01_positive_sample(50)
assert pts_disk.shape == (50, 2), '[TC25] 圆盘采样形状应为 (50,2) FAILED'
assert np.all(pts_disk >= 0), '[TC25] 正半圆盘所有坐标应 >= 0 FAILED'
assert np.all(np.linalg.norm(pts_disk, axis=1) <= 1.0 + 1e-12), '[TC25] 采样点应在单位圆内 FAILED'

# ---- TC26: coin_biased 全正面 ----
import numpy as np
np.random.seed(42)
coins = coin_biased(10, 1.0)
assert np.all(coins == 1.0), '[TC26] heads_prob=1 应全为 +1 FAILED'

# ---- TC27: coin_biased 全反面 ----
import numpy as np
np.random.seed(42)
coins2 = coin_biased(10, 0.0)
assert np.all(coins2 == -1.0), '[TC27] heads_prob=0 应全为 -1 FAILED'

# ---- TC28: alnorm P(X<0)=0.5 ----
import numpy as np
p = alnorm(0.0, upper=False)
assert abs(p - 0.5) < 1e-10, '[TC28] 标准正态 P(X<0) 应为 0.5 FAILED'

# ---- TC29: alnorm 对称性 P(X<z) = P(X>-z) upper ----
import numpy as np
p1 = alnorm(1.0, upper=False)
p2 = alnorm(-1.0, upper=True)
assert abs(p1 - p2) < 1e-10, '[TC29] alnorm 对称性 P(X<1)=P(X>1) upper FAILED'

# ---- TC30: mc_estimate_integral 常函数 ----
import numpy as np
np.random.seed(42)
def const_sampler():
    return 1.0
mean, std_err = mc_estimate_integral(const_sampler, 100)
assert abs(mean - 1.0) < 1e-6, '[TC30] 常函数均值应为 1 FAILED'

# ---- TC31: relative_l2_error 零误差 ----
import numpy as np
np.random.seed(42)
a = np.array([1.0, 2.0, 3.0])
err = relative_l2_error(a, a)
assert abs(err) < 1e-14, '[TC31] 完全相同输入 relative_l2_error 应为 0 FAILED'

# ---- TC32: relative_inf_error 有误差 ----
import numpy as np
np.random.seed(42)
exact = np.array([1.0, 2.0, 3.0])
approx = np.array([1.1, 2.0, 3.0])
err = relative_inf_error(approx, exact)
assert err > 0, '[TC32] 有误差时 relative_inf_error 应 > 0 FAILED'

# ---- TC33: convergence_order 线性收敛 ----
import numpy as np
errors = np.array([0.1, 0.05, 0.025])
params = np.array([1.0, 2.0, 4.0])
p = convergence_order(errors, params)
assert len(p) == 2, '[TC33] 收敛阶数数组长度应为 K-1=2 FAILED'

# ---- TC34: fmm_error_budget 分量非负 ----
import numpy as np
budget = fmm_error_budget(200, 5)
assert budget['truncation_error'] >= 0, '[TC34] 截断误差应为非负 FAILED'
assert budget['translation_error'] >= 0, '[TC34] 转换误差应为非负 FAILED'
assert budget['total_error_estimate'] > 0, '[TC34] 总误差估计应为正 FAILED'

# ---- TC35: estimate_truncation_order 返回 rate ----
import numpy as np
np.random.seed(42)
errors_fmm = np.array([0.1, 0.01, 0.001])
orders = np.array([1, 2, 3])
result = estimate_truncation_order(errors_fmm, 1e-10, orders)
assert 'rate' in result, '[TC35] estimate_truncation_order 应返回 rate FAILED'
assert result['rate'] > 0, '[TC35] 截断误差衰减率应为正 FAILED'

# ---- TC36: kolmogorov_smirnov_statistic 范围 ----
import numpy as np
np.random.seed(42)
samples = np.random.randn(1000)
ks = kolmogorov_smirnov_statistic(samples, mu=0.0, sigma=1.0)
assert 0.0 <= ks <= 1.0, '[TC36] KS统计量应在 [0,1] 范围内 FAILED'

# ---- TC37: markov_chain_steady_state 均匀矩阵 ----
import numpy as np
np.random.seed(42)
T = np.ones((3, 3)) / 3.0
pi = markov_chain_steady_state(T)
assert pi.shape == (3,), '[TC37] 稳态分布形状应为 (3,) FAILED'
assert abs(np.sum(pi) - 1.0) < 1e-10, '[TC37] 稳态分布之和应为 1 FAILED'

# ---- TC38: second_eigenvalue_rate 基本 ----
import numpy as np
np.random.seed(42)
T = np.array([[0.5, 0.5], [0.5, 0.5]])
rate = second_eigenvalue_rate(T)
assert rate >= 0.0, '[TC38] 第二特征值绝对值应为非负 FAILED'

# ---- TC39: triangle_area 等边三角形 ----
import numpy as np
np.random.seed(42)
area = triangle_area(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                     np.array([0.5, np.sqrt(3) / 2, 0.0]))
assert abs(area - np.sqrt(3) / 4) < 1e-10, '[TC39] 等边三角形面积应为 sqrt(3)/4 FAILED'

# ---- TC40: refine_triangle_midpoint 产生4子三角形 ----
import numpy as np
np.random.seed(42)
pts_ref, tris = refine_triangle_midpoint(np.array([0.0, 0.0]),
                                          np.array([1.0, 0.0]),
                                          np.array([0.0, 1.0]))
assert len(pts_ref) == 6, '[TC40] 细化后应有 6 个点 FAILED'
assert len(tris) == 4, '[TC40] 细化后应有 4 个三角形 FAILED'

# ---- TC41: project_3d_to_2d 默认投影 xy 平面 ----
import numpy as np
np.random.seed(42)
pts3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
proj = project_3d_to_2d(pts3d)
assert proj.shape == (2, 2), '[TC41] 3D投影到2D形状应为 (2,2) FAILED'
assert abs(proj[0, 0] - 1.0) < 1e-12, '[TC41] 第一点 x 坐标应为 1.0 FAILED'

# ---- TC42: AdaptiveTriMesh element_area ----
import numpy as np
np.random.seed(42)
pts_mesh = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
elements = [(0, 1, 2), (1, 3, 2)]
mesh = AdaptiveTriMesh(pts_mesh, elements)
a = mesh.element_area(0)
assert abs(a - 0.5) < 1e-12, '[TC42] 三角形(0,1,2)面积应为 0.5 FAILED'

# ---- TC43: AdaptiveTriMesh element_diameter ----
import numpy as np
np.random.seed(42)
d = mesh.element_diameter(0)
assert d > 0, '[TC43] 三角形直径应为正 FAILED'

# ---- TC44: generate_test_particles uniform 输出形状 ----
import numpy as np
np.random.seed(42)
pts, chg = generate_test_particles(100, "uniform")
assert pts.shape == (100, 3), '[TC44] uniform 粒子坐标形状应为 (100,3) FAILED'
assert chg.shape == (100,), '[TC44] uniform 电荷形状应为 (100,) FAILED'

# ---- TC45: generate_test_particles gaussian ----
import numpy as np
np.random.seed(42)
pts_g, chg_g = generate_test_particles(50, "gaussian")
assert pts_g.shape == (50, 3), '[TC45] gaussian 粒子坐标形状应为 (50,3) FAILED'

# ---- TC46: generate_test_particles dipole 总电荷为0 ----
import numpy as np
np.random.seed(42)
pts_d, chg_d = generate_test_particles(50, "dipole")
assert pts_d.shape == (50, 3), '[TC46] dipole 粒子坐标形状应为 (50,3) FAILED'
assert abs(np.sum(chg_d)) < 1e-10, '[TC46] 偶极子总电荷应为 0 FAILED'

# ---- TC47: m2m_translate 零位移恒等 ----
import numpy as np
np.random.seed(42)
child_r = [np.array([1.0]), np.array([0.0, 0.0])]
child_i = [np.array([0.0]), np.array([0.0, 0.0])]
parent_r, parent_i = m2m_translate(child_r, child_i,
                                    np.array([0.0, 0.0, 0.0]),
                                    np.array([0.0, 0.0, 0.0]), 1)
assert abs(parent_r[0][0] - 1.0) < 1e-12, '[TC47] M2M零位移应保持矩不变 FAILED'

# ---- TC48: l2l_translate 零位移恒等 ----
import numpy as np
np.random.seed(42)
p_r = [np.array([1.0]), np.array([0.5, 0.0])]
p_i = [np.array([0.0]), np.array([0.0, 0.0])]
c_r, c_i = l2l_translate(p_r, p_i,
                           np.array([0.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 0.0]), 1)
assert abs(c_r[0][0] - 1.0) < 1e-12, '[TC48] L2L零位移应保持系数不变 FAILED'

# ---- TC49: compute_translation_matrix 距离 ----
import numpy as np
np.random.seed(42)
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
matrices = compute_translation_matrix(nodes, 2)
assert (0, 1) in matrices, '[TC49] 转换矩阵字典应包含键 (0,1) FAILED'
assert abs(matrices[(0, 1)]['distance'] - 1.0) < 1e-12, '[TC49] 距离应为 1.0 FAILED'

# ---- TC50: FMM积分测试 run_fmm_experiment 小规模 ----
import numpy as np
np.random.seed(42)
result = run_fmm_experiment(n_particles=50, expansion_order=3, max_depth=3)
assert 'fmm_potential' in result, '[TC50] FMM实验应返回 fmm_potential FAILED'
assert result['fmm_potential'] is not None, '[TC50] FMM势能不应为 None FAILED'
assert np.all(np.isfinite(result['fmm_potential'])), '[TC50] FMM势能应全为有限值 FAILED'
assert result['fmm_potential'].shape == (50,), '[TC50] FMM势能形状应为 (50,) FAILED'

# ---- TC51: nonuniform_particle_sample disk_positive domain ----
import numpy as np
np.random.seed(42)
def density(v):
    return 1.0
samples_nu = nonuniform_particle_sample(density, 30, domain="disk_positive")
assert samples_nu.shape == (30, 3), '[TC51] 非均匀采样disk形状应为 (30,3) FAILED'
