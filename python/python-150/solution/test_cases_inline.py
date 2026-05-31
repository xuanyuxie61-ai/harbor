# ---- TC01: build_demo_molecules returns 3 molecules ----
mols = build_demo_molecules()
assert len(mols) == 3, '[TC01] build_demo_molecules 应返回3个分子 FAILED'

# ---- TC02: H2O molecular graph correct atom/bond count ----
mols = build_demo_molecules()
assert mols[0].n_atoms == 3, '[TC02] H2O应有3个原子 FAILED'
assert mols[0].n_bonds == 2, '[TC02] H2O应有2个键 FAILED'

# ---- TC03: adjacency_dense produces symmetric matrix ----
mols = build_demo_molecules()
A = mols[0].adjacency_dense()
assert np.allclose(A, A.T), '[TC03] 邻接矩阵应对称 FAILED'

# ---- TC04: degree matrix has all positive entries ----
mols = build_demo_molecules()
deg = mols[0].degree
assert np.all(deg > 0), '[TC04] 度矩阵所有元素应为正 FAILED'

# ---- TC05: apply_normalized_laplacian returns correct shape ----
mols = build_demo_molecules()
x = np.ones((mols[0].n_atoms, 3))
y = mols[0].apply_normalized_laplacian(x)
assert y.shape == x.shape, '[TC05] 归一化拉普拉斯乘法输出shape应匹配 FAILED'

# ---- TC06: chebyshev_coefficients_1d returns correct shapes ----
xd = np.linspace(-1, 1, 11)
yd = np.sin(2 * np.pi * xd)
c, xmin, xmax = chebyshev_coefficients_1d(11, xd, yd)
assert len(c) == 11, '[TC06] Chebyshev系数应返回11个 FAILED'
assert xmin <= xmax, '[TC06] xmin ≤ xmax FAILED'

# ---- TC07: chebyshev_value_1d near-accurate for sin(2πx) ----
import numpy as np
np.random.seed(42)
xd = np.linspace(-1, 1, 15)
yd = np.sin(2 * np.pi * xd)
c, xmin, xmax = chebyshev_coefficients_1d(15, xd, yd)
xi = np.linspace(-1, 1, 51)
yi = chebyshev_value_1d(c, xmin, xmax, xi)
err = np.max(np.abs(yi - np.sin(2 * np.pi * xi)))
assert err < 0.2, '[TC07] Chebyshev插值误差应小于0.2 FAILED'

# ---- TC08: ChebyshevGraphConv forward pass correct shape ----
import numpy as np
np.random.seed(42)
mols = build_demo_molecules()
x = np.random.randn(mols[0].n_atoms, 4)
conv = ChebyshevGraphConv(4, 8, K=4)
y = conv(x, mols[0].apply_normalized_laplacian)
assert y.shape == (mols[0].n_atoms, 8), '[TC08] ChebyshevGraphConv输出shape应为(n_atoms, 8) FAILED'

# ---- TC09: ChebyshevGraphConv K=1 output has no NaN/Inf ----
mols = build_demo_molecules()
x = np.ones((mols[0].n_atoms, 3))
conv = ChebyshevGraphConv(3, 5, K=1)
y = conv(x, mols[0].apply_normalized_laplacian)
assert not np.any(np.isnan(y)), '[TC09] K=1卷积输出无NaN FAILED'
assert not np.any(np.isinf(y)), '[TC09] K=1卷积输出无Inf FAILED'

# ---- TC10: ElectrostaticSolver deposit_charge preserves total charge ----
import numpy as np
np.random.seed(42)
solver = ElectrostaticSolver(box=(5.0, 5.0, 5.0), grid=(8, 8, 8))
atoms = np.array([[2.5, 2.5, 2.5], [3.2, 2.5, 2.5]], dtype=np.float64)
charges = np.array([-0.8, 0.8], dtype=np.float64)
rho = solver.deposit_charge(atoms, charges)
total_rho = np.sum(rho) * solver.dx * solver.dy * solver.dz
assert abs(total_rho) < 1e-10, '[TC10] 电荷沉积应保持总电荷为零 FAILED'

# ---- TC11: solve_poisson yields zero phi for zero rho ----
solver = ElectrostaticSolver(box=(5.0, 5.0, 5.0), grid=(8, 8, 8))
rho_zero = np.zeros((8, 8, 8))
phi = solver.solve_poisson(rho_zero)
assert np.max(np.abs(phi)) < 1e-10, '[TC11] 零电荷密度应产生零电势 FAILED'

# ---- TC12: maxwell_boltzmann_velocity produces correct count ----
import numpy as np
np.random.seed(42)
v = maxwell_boltzmann_velocity(temperature=300.0, mass_amu=16.0, n_samples=10)
assert len(v) == 10, '[TC12] Maxwell-Boltzmann采样应返回10个速度 FAILED'

# ---- TC13: integrate_triangle exact for ∫∫_T xy dxdy = 1/24 ----
verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
val = integrate_triangle(lambda r: r[0] * r[1], verts, rule="o03")
assert abs(val - 1.0 / 24.0) < 1e-10, '[TC13] 三角形积分 ∫∫xy dxdy 应=1/24 FAILED'

# ---- TC14: integrate_line ∫_0^1 x^2 dx = 1/3 ----
val_line = integrate_line(lambda x: x ** 2, 0.0, 1.0, n=5)
assert abs(val_line - 1.0 / 3.0) < 1e-4, '[TC14] Newton-Cotes ∫x²dx 应≈1/3 FAILED'

# ---- TC15: triangle_unit_o07 produces 7 quadrature points ----
w, xy = triangle_unit_o07()
assert len(w) == 7, '[TC15] o07规则应有7个求积点 FAILED'
assert xy.shape == (7, 2), '[TC15] 求积点坐标shape应为(7,2) FAILED'

# ---- TC16: generate_monomials correct count for m=3, d=2 ----
monos = generate_monomials(m=3, degree=2)
assert monos.shape[0] == 10, '[TC16] 3元2次单项式个数应为10 FAILED'

# ---- TC17: evaluate_polynomial matches expected value ----
monos = generate_monomials(m=3, degree=2)
coeffs = np.array([0, 3, 0, 0, 0, 0, 0, 0, 2, 1], dtype=np.float64)
point = np.array([1.0, 2.0, 0.5])
val = evaluate_polynomial(coeffs, monos, point)
expected = 1.0 ** 2 + 2.0 * 1.0 * 2.0 + 3.0 * 0.5
assert abs(val - expected) < 1e-10, '[TC17] 多项式求值应与期望值一致 FAILED'

# ---- TC18: falling_factorial [5.5]_3 = 5.5*4.5*3.5 ----
ff = falling_factorial(5.5, 3)
assert abs(ff - 5.5 * 4.5 * 3.5) < 1e-10, '[TC18] 下降阶乘[5.5]_3应=86.625 FAILED'

# ---- TC19: lennard_jones_potential returns finite negative value ----
coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
E0 = lennard_jones_potential(coords, epsilon=1.0, sigma=1.0)
assert E0 < 1.0, '[TC19] LJ势在r=σ时应较小 FAILED'
assert np.isfinite(E0), '[TC19] LJ势应有限 FAILED'

# ---- TC20: damped_gradient_flow runs and returns finite results ----
import numpy as np
np.random.seed(42)
coords = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.3, 0.0]], dtype=np.float64)
coords_opt, E_opt = damped_gradient_flow(
    coords, lennard_jones_potential, lennard_jones_gradient, lennard_jones_hessian,
    n_steps=10, h=0.05, tol=1e-2
)
assert coords_opt.shape == coords.shape, '[TC20] 梯度流输出坐标shape匹配 FAILED'
assert np.all(np.isfinite(coords_opt)), '[TC20] 梯度流输出坐标应有限 FAILED'
assert np.isfinite(E_opt), '[TC20] 梯度流输出能量应有限 FAILED'

# ---- TC21: error_function(0) = 0 ----
assert abs(error_function(0.0)) < 1e-8, '[TC21] erf(0)≈0 FAILED'

# ---- TC22: error_function large positive approaches 1 ----
assert error_function(5.0) > 0.9999, '[TC22] erf(5)应接近1 FAILED'

# ---- TC23: gammaln(5.0) ≈ ln(24) ----
assert abs(gammaln(5.0) - np.log(24.0)) < 1e-6, '[TC23] lnΓ(5)应=ln(24) FAILED'

# ---- TC24: digamma(5.0) is finite and positive ----
psi_val = digamma(5.0)
assert np.isfinite(psi_val), '[TC24] ψ(5)应有限 FAILED'
assert psi_val > 0, '[TC24] ψ(5)应为正 FAILED'

# ---- TC25: EvidentialRegressor predict returns positive parameters ----
import numpy as np
np.random.seed(42)
reg = EvidentialRegressor(input_dim=10, hidden_dim=16)
x = np.random.randn(5, 10)
gamma, nu, alpha, beta = reg.predict(x)
assert len(gamma) == 5, '[TC25] gamma应有5个元素 FAILED'
assert np.all(nu > 0), '[TC25] nu应全为正 FAILED'
assert np.all(alpha > 1), '[TC25] alpha应全>1 FAILED'
assert np.all(beta > 0), '[TC25] beta应全为正 FAILED'

# ---- TC26: diophantine_nonnegative_solutions count matches formula ----
from math import comb
sols = diophantine_nonnegative_solutions(target=4, n_vars=3)
assert len(sols) == comb(4 + 3 - 1, 3 - 1), '[TC26] Diophantine解个数应=C(6,2)=15 FAILED'

# ---- TC27: parity_violation_check ----
assert parity_violation_check(np.array([2, 3, 4]), required_parity=0), '[TC27] 奇数和违反检查应为True FAILED'
assert not parity_violation_check(np.array([2, 4, 4]), required_parity=0), '[TC27] 偶数和违反检查应为False FAILED'

# ---- TC28: greedy_graph_partition sums equal total weight ----
weights = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
adj = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
], dtype=np.float64)
part, s0, s1 = greedy_graph_partition(weights, adj)
assert abs(s0 + s1 - np.sum(weights)) < 1e-10, '[TC28] 划分权重和应等于总权重 FAILED'

# ---- TC29: threshold_binarize correctly separates at threshold ----
feats = np.array([0.3, 0.5, 1.2, 2.0])
b = threshold_binarize(feats, threshold=1.0)
assert np.array_equal(b, np.array([0., 0., 1., 1.])), '[TC29] 阈值二值化输出不正确 FAILED'

# ---- TC30: coulomb_matrix diagonal = 0.5 * Z^2.4 ----
atoms = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
Z = np.array([1, 1])
cm = coulomb_matrix(atoms, Z, max_size=4)
cm_mat = cm.reshape(4, 4)
assert abs(cm_mat[0, 0] - 0.5) < 1e-10, '[TC30] Coulomb矩阵对角元素Z=1应为0.5 FAILED'

# ---- TC31: compute_atom_features returns (n, 5) shape ----
feats = compute_atom_features(np.array([6, 1, 1, 1, 1]))
assert feats.shape == (5, 5), '[TC31] 原子特征shape应为(5,5) FAILED'

# ---- TC32: charge_conservation_loss is 0 when charges sum to target ----
pil = PhysicsInformedLoss()
loss = pil.charge_conservation_loss(np.array([1.0, -0.5, -0.5]), 0.0)
assert loss < 1e-12, '[TC32] 电荷守恒损失在守恒时应为零 FAILED'

# ---- TC33: schrodinger_residual_energy returns non-negative finite float ----
psi = np.array([0.5, 0.7, 0.5])
V = np.array([-1.0, -1.0, -1.0])
lapl = np.array([-0.2, 0.1, -0.2])
res = schrodinger_residual_energy(psi, V, lapl, hbar=1.0, mass=1.0)
assert res >= 0, '[TC33] 薛定谔残差能量应非负 FAILED'
assert np.isfinite(res), '[TC33] 薛定谔残差应为有限值 FAILED'

# ---- TC34: MolecularMPNN forward returns expected keys ----
import numpy as np
np.random.seed(42)
dataset = SyntheticMoleculeDataset(n_samples=5, seed=42)
mol, target, Z = dataset[0]
model = MolecularMPNN(node_in=5, edge_in=4, hidden=16, n_layers=2)
out = model.forward(mol, Z)
expected_keys = {"atom_energies", "total_energy", "atom_charges", "gamma", "nu", "alpha", "beta", "node_embeddings"}
assert expected_keys.issubset(set(out.keys())), '[TC34] MPNN输出缺少必要键 FAILED'

# ---- TC35: SyntheticMoleculeDataset has correct length ----
import numpy as np
np.random.seed(42)
ds = SyntheticMoleculeDataset(n_samples=10, seed=42)
assert len(ds) == 10, '[TC35] 数据集长度应为10 FAILED'

# ---- TC36: AdamOptimizer step changes parameters ----
param = np.array([1.0, 2.0, 3.0], dtype=np.float64)
opt = AdamOptimizer([param], lr=0.1)
grad = np.array([0.1, -0.2, 0.3])
param_copy = param.copy()
opt.step([grad])
assert not np.allclose(param, param_copy), '[TC36] Adam优化器应更新参数 FAILED'

# ---- TC37: backward_euler_step returns finite result ----
from dynamics_integrator import backward_euler_step
import numpy as np
np.random.seed(42)
y0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
f_ode = lambda y: -0.5 * y
df_ode = lambda y: -0.5 * np.eye(len(y))
y1 = backward_euler_step(y0, h=0.1, f=f_ode, df=df_ode, max_iter=20, tol=1e-8)
assert np.all(np.isfinite(y1)), '[TC37] 后向Euler步输出应有限 FAILED'

# ---- TC38: integrate_line on constant gives exact result ----
val_const = integrate_line(lambda x: 5.0, 0.0, 2.0, n=4)
assert abs(val_const - 10.0) < 1e-10, '[TC38] 常数积分应为10 FAILED'

# ---- TC39: graph_hash_fingerprint returns non-negative int ----
fp = graph_hash_fingerprint(6, 6)
assert isinstance(fp, int), '[TC39] 图哈希指纹应为int类型 FAILED'
assert fp >= 0, '[TC39] 图哈希指纹应非负 FAILED'

# ---- TC40: spherical_basis_angles returns unit vectors ----
dirs = spherical_basis_angles(n_points=8, rotation=0.0)
norms = np.linalg.norm(dirs, axis=1)
assert np.allclose(norms, 1.0, atol=1e-10), '[TC40] 球基角度向量应为单位向量 FAILED'
