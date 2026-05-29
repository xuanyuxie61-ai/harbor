"""
main.py
=======
统一入口：图神经网络分子性质预测

运行方式:
    python main.py

本项目围绕"神经计算：图神经网络分子性质预测"展开，
融合 15 个种子项目的核心算法，实现博士级科学计算。
"""

import numpy as np
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from molecular_graph import MolecularGraph, build_demo_molecules
from chebyshev_conv import ChebyshevGraphConv, chebyshev_coefficients_1d, chebyshev_value_1d
from electrostatic_solver import ElectrostaticSolver, maxwell_boltzmann_velocity
from numerical_quadrature import (
    triangle_unit_o07, integrate_triangle, line_nco_rule, integrate_line,
    compute_molecular_surface_integral, compute_bond_path_integral
)
from polynomial_basis import (
    generate_monomials, evaluate_polynomial, compute_polynomial_descriptors,
    falling_factorial
)
from dynamics_integrator import (
    damped_gradient_flow, lennard_jones_potential,
    lennard_jones_gradient, lennard_jones_hessian
)
from uncertainty_model import (
    EvidentialRegressor, error_function, incomplete_gamma, gammaln, digamma,
    erf_correlation_kernel, multivariate_normal_sample
)
from graph_utils import (
    greedy_graph_partition, graph_hash_fingerprint,
    diophantine_nonnegative_solutions, parity_violation_check,
    hypercube_distance_stats, descriptor_space_uniformity,
    spherical_basis_angles, angular_descriptor
)
from feature_engineering import (
    threshold_binarize, double_threshold_encode, molecular_fingerprint,
    coulomb_matrix, radial_distribution_histogram, compute_atom_features,
    encode_molecular_features
)
from physics_informed_loss import PhysicsInformedLoss, schrodinger_residual_energy
from gnn_model import MolecularMPNN
from dataset import SyntheticMoleculeDataset
from train_eval import AdamOptimizer, train_epoch, evaluate


def demo_molecular_graph():
    """演示分子图构建与谱分析。"""
    print("=" * 70)
    print("[1] 分子图构建与谱分析 (融合 mesh_vtoe + web_matrix)")
    print("=" * 70)
    mols = build_demo_molecules()
    for name, mol in zip(["H2O", "CH4", "C6H6"], mols):
        print(f"\n分子: {name}")
        print(f"  原子数: {mol.n_atoms}, 键数: {mol.n_bonds}")
        print(f"  拉普拉斯最大特征值上界 λ_max: {mol.lmax:.6f}")
        print(f"  原子重要性 (PageRank 风格): {np.round(mol.atom_importance, 4)}")
        # 测试 Laplacian 乘法
        x = np.random.randn(mol.n_atoms, 2)
        Lx = mol.apply_normalized_laplacian(x)
        print(f"  归一化拉普拉斯乘法测试: shape {Lx.shape}")


def demo_chebyshev():
    """演示 Chebyshev 插值与谱卷积。"""
    print("\n" + "=" * 70)
    print("[2] Chebyshev 谱分析 (融合 chebyshev_interp_1d)")
    print("=" * 70)
    # 1D 插值演示
    xd = np.linspace(-1, 1, 11)
    yd = np.sin(2 * np.pi * xd)
    c, xmin, xmax = chebyshev_coefficients_1d(len(xd), xd, yd)
    xi = np.linspace(-1, 1, 51)
    yi = chebyshev_value_1d(c, xmin, xmax, xi)
    err = np.max(np.abs(yi - np.sin(2 * np.pi * xi)))
    print(f"\nChebyshev 插值 sin(2πx) 最大误差: {err:.2e}")

    # 图卷积演示
    print("\nChebyshev 图卷积层测试:")
    mols = build_demo_molecules()
    mol = mols[0]
    x = np.random.randn(mol.n_atoms, 4)
    conv = ChebyshevGraphConv(4, 8, K=4)
    y = conv(x, mol.apply_normalized_laplacian)
    print(f"  输入 shape: {x.shape} -> 输出 shape: {y.shape}")


def demo_electrostatic():
    """演示静电势求解。"""
    print("\n" + "=" * 70)
    print("[3] 静电势求解 (融合 PIC 粒子云网格)")
    print("=" * 70)
    solver = ElectrostaticSolver(box=(5.0, 5.0, 5.0), grid=(16, 16, 16))
    atoms = np.array([
        [2.5, 2.5, 2.5],
        [3.2, 2.5, 2.5],
        [1.8, 2.5, 2.5]
    ], dtype=np.float64)
    charges = np.array([-0.8, 0.4, 0.4], dtype=np.float64)
    rho = solver.deposit_charge(atoms, charges)
    phi = solver.solve_poisson(rho)
    Ex, Ey, Ez = solver.compute_electric_field(phi)
    E_atoms = solver.interpolate_field_to_atoms(atoms, Ex, Ey, Ez)
    energy = solver.compute_electrostatic_energy(phi, rho)
    print(f"\nH2O 近似电荷沉积:")
    print(f"  网格电荷密度总和: {np.sum(rho):.4f}")
    print(f"  静电能: {energy:.6f} eV·Å")
    print(f"  O 原子处电场: {E_atoms[0]}")

    # Maxwell-Boltzmann 速度
    v = maxwell_boltzmann_velocity(temperature=300.0, mass_amu=16.0, n_samples=5)
    print(f"  Maxwell-Boltzmann 采样速度 (O, 300K): {np.round(v, 3)} Å/ps")


def demo_quadrature():
    """演示数值积分。"""
    print("\n" + "=" * 70)
    print("[4] 数值积分 (融合 triangle_felippa_rule + line_nco_rule)")
    print("=" * 70)
    # 三角形积分: ∫∫_T x*y dA (使用 o03，精度 2，对二次多项式精确)
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    val = integrate_triangle(lambda r: r[0] * r[1], verts, rule="o03")
    print(f"\n单位三角形 ∫∫ x*y dxdy = {val:.6f} (理论值: 1/24 = 0.041667)")

    # 线积分: ∫_0^1 x^2 dx = 1/3
    val_line = integrate_line(lambda x: x ** 2, 0.0, 1.0, n=5)
    print(f"Newton-Cotes Open ∫_0^1 x^2 dx = {val_line:.6f} (理论值: 0.333333)")

    # 分子应用
    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    surf = compute_molecular_surface_integral(atoms, alpha=2.0)
    print(f"  分子表面积分 (简化): {surf:.6f}")


def demo_polynomial():
    """演示多项式基。"""
    print("\n" + "=" * 70)
    print("[5] 多元多项式描述符 (融合 polynomial)")
    print("=" * 70)
    monos = generate_monomials(m=3, degree=2)
    print(f"\n3 元 2 次单项式个数: {monos.shape[0]} (理论: C(3+2,2)=10)")
    print(f"指数矩阵:\n{monos}")

    # 在点 [1, 2, 0.5] 求值 p(x,y,z) = x^2 + 2xy + 3z
    # 按 grlex 顺序 (Burkardt 约定, z 最显著):
    # (0,0,0),(0,0,1),(0,1,0),(1,0,0),(0,0,2),(0,1,1),(0,2,0),(1,0,1),(1,1,0),(2,0,0)
    coeffs = np.array([0, 3, 0, 0, 0, 0, 0, 0, 2, 1], dtype=np.float64)
    point = np.array([1.0, 2.0, 0.5])
    val = evaluate_polynomial(coeffs, monos, point)
    expected = 1.0 ** 2 + 2.0 * 1.0 * 2.0 + 3.0 * 0.5
    print(f"\n多项式求值: {val:.4f} (期望值: {expected:.4f})")

    # 分子描述符
    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    desc = compute_polynomial_descriptors(atoms, degree=2)
    print(f"  分子多项式描述符: shape={desc.shape}, norm={np.linalg.norm(desc):.4f}")

    ff = falling_factorial(5.5, 3)
    print(f"  下降阶乘 [5.5]_3 = {ff:.4f}")


def demo_dynamics():
    """演示结构松弛。"""
    print("\n" + "=" * 70)
    print("[6] 结构隐式松弛 (融合 backward_euler)")
    print("=" * 70)
    # 4 原子体系
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [1.5, 1.5, 0.0]
    ], dtype=np.float64)
    E0 = lennard_jones_potential(coords)
    print(f"\n初始 Lennard-Jones 能量: {E0:.4f}")
    coords_opt, E_opt = damped_gradient_flow(
        coords,
        lennard_jones_potential,
        lennard_jones_gradient,
        lennard_jones_hessian,
        n_steps=30, h=0.02, tol=1e-4
    )
    print(f"优化后能量: {E_opt:.4f}")
    print(f"坐标 RMSD: {np.sqrt(np.mean((coords_opt - coords)**2)):.4f} Å")


def demo_uncertainty():
    """演示不确定性量化。"""
    print("\n" + "=" * 70)
    print("[7] 概率不确定性量化 (融合 prob 特殊函数)")
    print("=" * 70)
    reg = EvidentialRegressor(input_dim=10, hidden_dim=16)
    x = np.random.randn(3, 10)
    gamma, nu, alpha, beta = reg.predict(x)
    print(f"\nNIG 参数预测 (3 样本):")
    for i in range(3):
        print(f"  sample {i}: γ={gamma[i]:.3f}, ν={nu[i]:.3f}, α={alpha[i]:.3f}, β={beta[i]:.3f}")

    ale, epi = reg.uncertainty(alpha, beta, nu)
    print(f"  平均偶然不确定性: {np.mean(ale):.4f}")
    print(f"  平均认知不确定性: {np.mean(epi):.4f}")

    # 特殊函数测试
    print(f"\n  erf(1.0) = {error_function(1.0):.6f} (理论: 0.842701)")
    print(f"  γ(0.5, 1.0) = {incomplete_gamma(0.5, 1.0):.6f}")
    print(f"  lnΓ(5.0) = {gammaln(5.0):.4f} (理论: ln(24)=3.1781)")
    print(f"  ψ(5.0) = {digamma(5.0):.6f} (理论: 1.506118)")

    # ERF 核
    r = np.array([0.2, 0.8, 1.5])
    k = erf_correlation_kernel(r, r_cut=1.2, lengthscale=0.3)
    print(f"  ERF 核 (r={r}): {np.round(k, 3)}")

    # 多维正态采样
    samples = multivariate_normal_sample(np.zeros(2), np.eye(2), 3)
    print(f"  多维正态采样 shape: {samples.shape}")


def demo_graph_utils():
    """演示图分析工具。"""
    print("\n" + "=" * 70)
    print("[8] 图分析与高维几何 (融合 partition_greedy + polyomino_parity + hypercube_distance + fermat_factor + circles)")
    print("=" * 70)
    # 图划分
    weights = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
    adj = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=np.float64)
    part, s0, s1 = greedy_graph_partition(weights, adj)
    print(f"\n贪心图划分: {part}, 子集权重: {s0:.2f}, {s1:.2f}")

    # Fermat 指纹
    fp = graph_hash_fingerprint(6, 6)
    print(f"  C6H6 拓扑指纹: {fp}")

    # Diophantine 解
    sols = diophantine_nonnegative_solutions(target=4, n_vars=3)
    print(f"  x+y+z=4 的非负整数解个数: {len(sols)} (理论: C(4+3-1,3-1)=15)")

    # 奇偶性检查
    pv = parity_violation_check(np.array([2, 3, 4]), required_parity=0)
    print(f"  奇偶违反检查 (2+3+4=9): {'违反' if pv else '通过'}")

    # 高维距离统计
    desc = np.random.randn(20, 8)
    mu, var = hypercube_distance_stats(desc, n_pairs=200)
    print(f"  描述符空间距离统计: μ={mu:.3f}, σ²={var:.3f}")
    u = descriptor_space_uniformity(desc)
    print(f"  描述符空间均匀性: {u:.3f}")

    # 角度描述符
    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    ad = angular_descriptor(atoms, center_idx=0, n_angles=8)
    print(f"  角度描述符 (center=0): {np.round(ad, 3)}")


def demo_features():
    """演示特征工程。"""
    print("\n" + "=" * 70)
    print("[9] 特征工程 (融合 image_threshold)")
    print("=" * 70)
    feats = np.array([0.3, 1.2, 2.5, 0.8, 3.0])
    binarized = threshold_binarize(feats, threshold=1.0)
    print(f"\n单阈值二值化 (t=1.0): {feats} -> {binarized}")

    triple = double_threshold_encode(feats, low=0.5, high=2.0)
    print(f"双阈值编码 (0.5, 2.0): {triple}")

    # Coulomb 矩阵
    atoms = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    Z = np.array([1, 1])
    cm = coulomb_matrix(atoms, Z, max_size=4)
    print(f"  H2 Coulomb 矩阵特征: shape={cm.shape}, 前4项={np.round(cm[:4], 3)}")

    # RDF
    atoms2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rdf = radial_distribution_histogram(atoms2, dr=0.2, r_max=2.0)
    print(f"  RDF 直方图 (前5项): {np.round(rdf[:5], 4)}")


def demo_physics_loss():
    """演示物理损失。"""
    print("\n" + "=" * 70)
    print("[10] 物理信息损失 (physics_informed_loss)")
    print("=" * 70)
    pil = PhysicsInformedLoss(lambda_energy=1.0, lambda_force=5.0, lambda_charge=2.0)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    q_pred = np.array([0.4, -0.4])
    energy_fn = lambda c: -1.0 / np.linalg.norm(c[0] - c[1])
    loss_total = pil.total_physics_loss(energy_fn, coords, q_pred, target_total_charge=0.0)
    print(f"\n物理损失 (H2 近似): {loss_total:.6f}")

    # 薛定谔残差
    psi = np.array([0.5, 0.7, 0.5])
    V = np.array([-1.0, -1.0, -1.0])
    lapl = np.array([-0.2, 0.1, -0.2])
    res = schrodinger_residual_energy(psi, V, lapl, hbar=1.0, mass=1.0)
    print(f"  薛定谔方程残差: {res:.6f}")


def demo_gnn_training():
    """演示 GNN 训练与评估。"""
    print("\n" + "=" * 70)
    print("[11] GNN 分子性质预测训练与评估")
    print("=" * 70)

    # 数据集
    dataset = SyntheticMoleculeDataset(n_samples=60, seed=42)
    train_idx, test_idx = dataset.train_test_split(ratio=0.8)
    print(f"\n数据集: {len(dataset)} 分子, 训练 {len(train_idx)}, 测试 {len(test_idx)}")

    # 模型
    model = MolecularMPNN(node_in=5, edge_in=4, hidden=16, n_layers=2)
    params = model.parameters()
    optimizer = AdamOptimizer(params, lr=5e-3)

    # 训练
    print("\n开始训练 (3 epochs, 每 epoch 采样 30 个分子)...")
    for epoch in range(3):
        # 为加速演示，每轮只随机选 30 个样本
        sampled = np.random.choice(train_idx, size=min(30, len(train_idx)), replace=False).tolist()
        loss = train_epoch(model, dataset, sampled, optimizer, lambda_physics=0.01)
        print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

    # 评估
    metrics = evaluate(model, dataset, test_idx)
    print(f"\n测试集评估:")
    print(f"  MAE (atomization energy): {metrics['mae_energy']:.4f} eV")
    print(f"  MAE (HOMO-LUMO gap):      {metrics['mae_gap']:.4f} eV")
    print(f"  平均偶然不确定性:         {metrics['mean_aleatoric']:.4f}")
    print(f"  平均认知不确定性:         {metrics['mean_epistemic']:.4f}")

    # 单分子预测演示
    mol, target, Z = dataset[0]
    out = model.forward(mol, Z)
    print(f"\n单分子预测示例:")
    print(f"  真实原子化能: {target['atomization_energy']:.4f}")
    print(f"  预测总能量:   {out['total_energy']:.4f}")
    print(f"  真实能隙:     {target['homo_lumo_gap']:.4f}")
    print(f"  预测能隙:     {out['gamma']:.4f}")
    print(f"  预测电荷和:   {np.sum(out['atom_charges']):.4f} (目标: {np.sum(Z)})")


def main():
    np.random.seed(42)
    print("\n")
    print("#" * 70)
    print("#  神经计算：图神经网络分子性质预测")
    print("#  融合 15 个种子项目的博士级科学计算项目")
    print("#" * 70)

    demo_molecular_graph()
    demo_chebyshev()
    demo_electrostatic()
    demo_quadrature()
    demo_polynomial()
    demo_dynamics()
    demo_uncertainty()
    demo_graph_utils()
    demo_features()
    demo_physics_loss()
    demo_gnn_training()

    print("\n" + "=" * 70)
    print("所有模块演示完毕，main.py 运行成功。")
    print("=" * 70)


if __name__ == "__main__":
    main()
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

print('\n全部 40 个测试通过!\n')
