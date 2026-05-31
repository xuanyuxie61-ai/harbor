# ---- TC01: normalize_vector 将非零向量归一化为单位长度 ----
v = np.array([3.0, 4.0])
v_norm = normalize_vector(v)
assert np.isclose(np.linalg.norm(v_norm), 1.0), '[TC01] 归一化后长度不为1 FAILED'

# ---- TC02: normalize_vector 零向量返回零向量 ----
v_zero = np.zeros(5)
v_norm_zero = normalize_vector(v_zero)
assert np.allclose(v_norm_zero, 0.0), '[TC02] 零向量归一化不为零 FAILED'

# ---- TC03: safe_divide 正常除法 ----
result = safe_divide(10.0, 2.0)
assert np.isclose(result, 5.0), '[TC03] 正常除法结果错误 FAILED'

# ---- TC04: safe_divide 被零除返回默认值 ----
result = safe_divide(10.0, 0.0, default=42.0)
assert np.isclose(result, 42.0), '[TC04] 除零未返回默认值 FAILED'

# ---- TC05: is_power_of_two 检测2的幂 ----
from utils import is_power_of_two
assert is_power_of_two(1) == True, '[TC05] 1应为2的幂 FAILED'
assert is_power_of_two(2) == True, '[TC05] 2应为2的幂 FAILED'
assert is_power_of_two(3) == False, '[TC05] 3不应为2的幂 FAILED'
assert is_power_of_two(256) == True, '[TC05] 256应为2的幂 FAILED'

# ---- TC06: chebyshev_nodes 端点与数量 ----
nodes = chebyshev_nodes(10, a=-1.0, b=1.0)
assert len(nodes) == 10, '[TC06] Chebyshev节点数量错误 FAILED'
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC06] Chebyshev节点超出范围 FAILED'

# ---- TC07: hadamard_matrix 正交性 H @ H.T = I ----
H = hadamard_matrix(4)
prod = H @ H.T
assert np.allclose(prod, np.eye(4), atol=1e-12), '[TC07] Hadamard矩阵不正交 FAILED'

# ---- TC08: grover_coin 酉性 ----
C = grover_coin(4)
prod = C @ C.T
assert np.allclose(prod, np.eye(4), atol=1e-12), '[TC08] Grover coin不酉 FAILED'

# ---- TC09: discrete_laplacian_1d 对称性 ----
L = discrete_laplacian_1d(5, periodic=False)
assert np.allclose(L, L.T), '[TC09] 离散Laplacian不对称 FAILED'

# ---- TC10: discrete_laplacian_1d 周期版本行和为零 ----
Lp = discrete_laplacian_1d(5, periodic=True)
row_sums = np.sum(Lp, axis=1)
assert np.allclose(row_sums, 0.0), '[TC10] 周期Laplacian行和不为零 FAILED'

# ---- TC11: r83_mv 三对角矩阵向量乘 ----
from matrix_solvers import r83_mv
n11 = 3
a11 = np.zeros((3, n11))
a11[0, :] = [-1.0, -1.0, 0.0]
a11[1, :] = [2.0, 2.0, 2.0]
a11[2, :] = [0.0, -1.0, -1.0]
x11 = np.array([1.5, 2.0, 1.5])
b11_expected = np.array([1.0, 1.0, 1.0])
b11_computed = r83_mv(n11, a11, x11)
assert np.allclose(b11_computed, b11_expected), '[TC11] 三对角MV结果错误 FAILED'

# ---- TC12: r83_cg 求解精度 ----
n12 = 3
a12 = np.zeros((3, n12))
a12[0, :] = [-1.0, -1.0, 0.0]
a12[1, :] = [2.0, 2.0, 2.0]
a12[2, :] = [0.0, -1.0, -1.0]
b12 = np.ones(n12)
x12_cg = r83_cg(n12, a12, b12)
L12 = np.diag(a12[1, :]) + np.diag(a12[0, :-1], 1) + np.diag(a12[2, 1:], -1)
res12 = np.linalg.norm(L12 @ x12_cg - b12)
assert res12 < 1e-8, '[TC12] r83_cg残差过大 FAILED'

# ---- TC13: r8vm_sl Vandermonde 求解精度 ----
from matrix_solvers import r8vm_mv
n13 = 3
x13_nodes = np.array([1.0, 2.0, 3.0])
true13_coeffs = np.array([1.0, 2.0, 3.0])
b13_vm = r8vm_mv(n13, n13, x13_nodes, true13_coeffs)
recovered13 = r8vm_sl(n13, x13_nodes, b13_vm)
assert np.allclose(recovered13, true13_coeffs, atol=1e-8), '[TC13] Vandermonde求解不精确 FAILED'

# ---- TC14: hexagon_area 已知值 ----
from geometry_mesh import hexagon_area
area14 = hexagon_area()
expected14 = 3.0 * np.sqrt(3.0) / 2.0
assert np.isclose(area14, expected14), '[TC14] 六边形面积错误 FAILED'

# ---- TC15: hexagon_monomial_integral (0,0) = 面积 ----
int00 = hexagon_monomial_integral(0, 0)
assert np.isclose(int00, 3.0 * np.sqrt(3.0) / 2.0), '[TC15] 单项式积分(0,0)不等面积 FAILED'

# ---- TC16: hexagon_monomial_integral 奇次幂为零 ----
int10 = hexagon_monomial_integral(1, 0)
int01 = hexagon_monomial_integral(0, 1)
assert np.isclose(int10, 0.0), '[TC16] x^1积分不为零(对称性) FAILED'
assert np.isclose(int01, 0.0), '[TC16] y^1积分不为零(对称性) FAILED'

# ---- TC17: generate_hexagonal_lattice 顶点数量 ----
pts1ring = generate_hexagonal_lattice(1)
assert pts1ring.shape[0] == 7, '[TC17] 1环六边形晶格顶点数不为7 FAILED'

# ---- TC18: hadamard_coin 酉性 ----
Hc = hadamard_coin(4)
assert np.allclose(Hc @ Hc.T, np.eye(4), atol=1e-12), '[TC18] hadamard_coin不酉 FAILED'

# ---- TC19: fourier_coin 酉性 ----
Fc = fourier_coin(4)
assert np.allclose(Fc @ Fc.conj().T, np.eye(4), atol=1e-12), '[TC19] fourier_coin不酉 FAILED'

# ---- TC20: graph_laplacian 对称性 ----
adj20 = [[1], [0, 2], [1]]
Lg = graph_laplacian(adj20)
assert np.allclose(Lg, Lg.T), '[TC20] 图Laplacian不对称 FAILED'

# ---- TC21: DiscreteTimeQuantumWalk 状态范数守恒 ----
import numpy as np
np.random.seed(42)
qw21 = DiscreteTimeQuantumWalk(16, coin_dim=2, coin_type="hadamard", periodic=True)
qw21.set_initial_state(position=8)
qw21.step(num_steps=30)
norm21 = qw21.get_state_norm()
assert np.isclose(norm21, 1.0, atol=1e-10), '[TC21] DTQW范数不守恒 FAILED'

# ---- TC22: ContinuousTimeQuantumWalk 状态范数守恒 ----
adj22 = [[1], [0, 2], [1]]
ctqw22 = ContinuousTimeQuantumWalk(adj22, gamma=1.0)
ctqw22.set_initial_state(1)
ctqw22.evolve(t=2.0)
norm22 = ctqw22.get_state_norm()
assert np.isclose(norm22, 1.0, atol=1e-10), '[TC22] CTQW范数不守恒 FAILED'

# ---- TC23: newton_raphson 求 sin(x)=0.5 的根 ----
def _f23(x): return np.sin(x) - 0.5
def _df23(x): return np.cos(x)
root23, conv23, iters23 = newton_raphson(_f23, _df23, 0.3)
assert conv23, '[TC23] Newton法未收敛 FAILED'
assert np.isclose(_f23(root23), 0.0, atol=1e-8), '[TC23] Newton法根不满足方程 FAILED'

# ---- TC24: coupon_collector_simulation 理论期望在合理范围内 ----
import numpy as np
np.random.seed(42)
res_cc = coupon_collector_simulation(10, num_trials=2000, seed=42)
emp_mean24 = res_cc['empirical_mean']
theo_mean24 = res_cc['theoretical_expected']
rel_diff24 = abs(emp_mean24 - theo_mean24) / theo_mean24
assert rel_diff24 < 0.20, '[TC24] 优惠券收集器经验均值偏离理论过大 FAILED'

# ---- TC25: steinerberger_integral01_exact 已知值 ----
exact25 = steinerberger_integral01_exact(1)
expected25 = 2.0 / np.pi
assert np.isclose(exact25, expected25), '[TC25] Steinerberger积分精确值错误 FAILED'

# ---- TC26: integrate_simpson 二次函数精确积分 ----
x_quad = np.linspace(0.0, 1.0, 101)
y_quad = x_quad ** 2
approx26 = integrate_simpson(y_quad, x_quad)
exact26 = 1.0 / 3.0
assert abs(approx26 - exact26) < 1e-6, '[TC26] Simpson积分不精确 FAILED'

# ---- TC27: estimate_search_complexity 单调性 ----
est27_1 = estimate_search_complexity(100, 1, graph_degree=4.0)
est27_2 = estimate_search_complexity(100, 4, graph_degree=4.0)
assert est27_1['optimal_steps'] > est27_2['optimal_steps'], '[TC27] 搜索复杂度不单调 FAILED'

# ---- TC28: diophantine_nd_nonnegative 已知解 ----
a28 = np.array([1, 2, 3])
sols28 = diophantine_nd_nonnegative(a28, 4)
assert len(sols28) == 4, '[TC28] Diophantine解数量错误 FAILED'
for sol28 in sols28:
    assert np.dot(a28, sol28) == 4, '[TC28] Diophantine解不满足方程 FAILED'

# ---- TC29: spectral_gap 对已知矩阵正确 ----
H29 = np.diag([0.0, 2.0, 2.0, 5.0])
gap29 = spectral_gap(H29)
assert np.isclose(gap29, 2.0), '[TC29] 谱隙计算错误 FAILED'

# ---- TC30: MultiDimensionalQuantumWalk 范数守恒 ----
import numpy as np
np.random.seed(42)
mdqw30 = MultiDimensionalQuantumWalk((3, 3, 3), coin_type="grover", periodic=True)
mdqw30.set_initial_state()
mdqw30.step(num_steps=5)
prob30_sum = np.sum(mdqw30.get_position_distribution())
assert np.isclose(prob30_sum, 1.0, atol=1e-10), '[TC30] 多维量子行走概率和不守恒 FAILED'

# ---- TC31: eigenstate_localization 对均匀态为 1/N ----
uniform_vec = np.ones(10) / np.sqrt(10)
ipr31 = eigenstate_localization(uniform_vec)
assert np.isclose(ipr31, 0.1, atol=1e-10), '[TC31] 均匀态IPR不为1/N FAILED'

# ---- TC32: graph_adjacency_matrix 对称性 ----
from quantum_operators import graph_adjacency_matrix
adj32 = [[1, 2], [0], [0]]
A32 = graph_adjacency_matrix(adj32)
assert np.allclose(A32, A32.T), '[TC32] 图邻接矩阵不对称 FAILED'

# ---- TC33: find_optimal_coin_angle 在已知凸函数上正确 ----
def _prob_func33(angle):
    return np.sin(2.0 * angle) ** 2
opt33 = find_optimal_coin_angle(_prob_func33, angle0=0.5)
assert 0.6 < opt33['optimal_angle'] < 1.0, '[TC33] 最优角度异常 FAILED'
assert opt33['success_probability'] > 0.5, '[TC33] 最优成功概率过低 FAILED'

# ---- TC34: convergence_rate_analysis 线性收敛检测 ----
import numpy as np
np.random.seed(42)
errors34 = 2.0 ** (-np.arange(1, 11, dtype=float))
conv34 = convergence_rate_analysis(errors34)
assert conv34['estimated_order'] is not None, '[TC34] 收敛阶估计为空 FAILED'
assert conv34['estimated_order'] > 0.5, '[TC34] 收敛阶估计异常低 FAILED'

# ---- TC35: clamp 值检查 ----
assert clamp(1.5, 0.0, 1.0) == 1.0, '[TC35] clamp上限错误 FAILED'
assert clamp(-0.5, 0.0, 1.0) == 0.0, '[TC35] clamp下限错误 FAILED'
assert clamp(0.5, 0.0, 1.0) == 0.5, '[TC35] clamp范围内错误 FAILED'
