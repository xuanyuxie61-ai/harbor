
import numpy as np
import sys
import os


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
    print("=" * 70)
    print("[1] 分子图构建与谱分析 (融合 mesh_vtoe + web_matrix)")
    print("=" * 70)
    mols = build_demo_molecules()
    for name, mol in zip(["H2O", "CH4", "C6H6"], mols):
        print(f"\n分子: {name}")
        print(f"  原子数: {mol.n_atoms}, 键数: {mol.n_bonds}")
        print(f"  拉普拉斯最大特征值上界 λ_max: {mol.lmax:.6f}")
        print(f"  原子重要性 (PageRank 风格): {np.round(mol.atom_importance, 4)}")

        x = np.random.randn(mol.n_atoms, 2)
        Lx = mol.apply_normalized_laplacian(x)
        print(f"  归一化拉普拉斯乘法测试: shape {Lx.shape}")


def demo_chebyshev():
    print("\n" + "=" * 70)
    print("[2] Chebyshev 谱分析 (融合 chebyshev_interp_1d)")
    print("=" * 70)

    xd = np.linspace(-1, 1, 11)
    yd = np.sin(2 * np.pi * xd)
    c, xmin, xmax = chebyshev_coefficients_1d(len(xd), xd, yd)
    xi = np.linspace(-1, 1, 51)
    yi = chebyshev_value_1d(c, xmin, xmax, xi)
    err = np.max(np.abs(yi - np.sin(2 * np.pi * xi)))
    print(f"\nChebyshev 插值 sin(2πx) 最大误差: {err:.2e}")


    print("\nChebyshev 图卷积层测试:")
    mols = build_demo_molecules()
    mol = mols[0]
    x = np.random.randn(mol.n_atoms, 4)
    conv = ChebyshevGraphConv(4, 8, K=4)
    y = conv(x, mol.apply_normalized_laplacian)
    print(f"  输入 shape: {x.shape} -> 输出 shape: {y.shape}")


def demo_electrostatic():
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


    v = maxwell_boltzmann_velocity(temperature=300.0, mass_amu=16.0, n_samples=5)
    print(f"  Maxwell-Boltzmann 采样速度 (O, 300K): {np.round(v, 3)} Å/ps")


def demo_quadrature():
    print("\n" + "=" * 70)
    print("[4] 数值积分 (融合 triangle_felippa_rule + line_nco_rule)")
    print("=" * 70)

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    val = integrate_triangle(lambda r: r[0] * r[1], verts, rule="o03")
    print(f"\n单位三角形 ∫∫ x*y dxdy = {val:.6f} (理论值: 1/24 = 0.041667)")


    val_line = integrate_line(lambda x: x ** 2, 0.0, 1.0, n=5)
    print(f"Newton-Cotes Open ∫_0^1 x^2 dx = {val_line:.6f} (理论值: 0.333333)")


    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    surf = compute_molecular_surface_integral(atoms, alpha=2.0)
    print(f"  分子表面积分 (简化): {surf:.6f}")


def demo_polynomial():
    print("\n" + "=" * 70)
    print("[5] 多元多项式描述符 (融合 polynomial)")
    print("=" * 70)
    monos = generate_monomials(m=3, degree=2)
    print(f"\n3 元 2 次单项式个数: {monos.shape[0]} (理论: C(3+2,2)=10)")
    print(f"指数矩阵:\n{monos}")




    coeffs = np.array([0, 3, 0, 0, 0, 0, 0, 0, 2, 1], dtype=np.float64)
    point = np.array([1.0, 2.0, 0.5])
    val = evaluate_polynomial(coeffs, monos, point)
    expected = 1.0 ** 2 + 2.0 * 1.0 * 2.0 + 3.0 * 0.5
    print(f"\n多项式求值: {val:.4f} (期望值: {expected:.4f})")


    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    desc = compute_polynomial_descriptors(atoms, degree=2)
    print(f"  分子多项式描述符: shape={desc.shape}, norm={np.linalg.norm(desc):.4f}")

    ff = falling_factorial(5.5, 3)
    print(f"  下降阶乘 [5.5]_3 = {ff:.4f}")


def demo_dynamics():
    print("\n" + "=" * 70)
    print("[6] 结构隐式松弛 (融合 backward_euler)")
    print("=" * 70)

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


    print(f"\n  erf(1.0) = {error_function(1.0):.6f} (理论: 0.842701)")
    print(f"  γ(0.5, 1.0) = {incomplete_gamma(0.5, 1.0):.6f}")
    print(f"  lnΓ(5.0) = {gammaln(5.0):.4f} (理论: ln(24)=3.1781)")
    print(f"  ψ(5.0) = {digamma(5.0):.6f} (理论: 1.506118)")


    r = np.array([0.2, 0.8, 1.5])
    k = erf_correlation_kernel(r, r_cut=1.2, lengthscale=0.3)
    print(f"  ERF 核 (r={r}): {np.round(k, 3)}")


    samples = multivariate_normal_sample(np.zeros(2), np.eye(2), 3)
    print(f"  多维正态采样 shape: {samples.shape}")


def demo_graph_utils():
    print("\n" + "=" * 70)
    print("[8] 图分析与高维几何 (融合 partition_greedy + polyomino_parity + hypercube_distance + fermat_factor + circles)")
    print("=" * 70)

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


    fp = graph_hash_fingerprint(6, 6)
    print(f"  C6H6 拓扑指纹: {fp}")


    sols = diophantine_nonnegative_solutions(target=4, n_vars=3)
    print(f"  x+y+z=4 的非负整数解个数: {len(sols)} (理论: C(4+3-1,3-1)=15)")


    pv = parity_violation_check(np.array([2, 3, 4]), required_parity=0)
    print(f"  奇偶违反检查 (2+3+4=9): {'违反' if pv else '通过'}")


    desc = np.random.randn(20, 8)
    mu, var = hypercube_distance_stats(desc, n_pairs=200)
    print(f"  描述符空间距离统计: μ={mu:.3f}, σ²={var:.3f}")
    u = descriptor_space_uniformity(desc)
    print(f"  描述符空间均匀性: {u:.3f}")


    atoms = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    ad = angular_descriptor(atoms, center_idx=0, n_angles=8)
    print(f"  角度描述符 (center=0): {np.round(ad, 3)}")


def demo_features():
    print("\n" + "=" * 70)
    print("[9] 特征工程 (融合 image_threshold)")
    print("=" * 70)
    feats = np.array([0.3, 1.2, 2.5, 0.8, 3.0])
    binarized = threshold_binarize(feats, threshold=1.0)
    print(f"\n单阈值二值化 (t=1.0): {feats} -> {binarized}")

    triple = double_threshold_encode(feats, low=0.5, high=2.0)
    print(f"双阈值编码 (0.5, 2.0): {triple}")


    atoms = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    Z = np.array([1, 1])
    cm = coulomb_matrix(atoms, Z, max_size=4)
    print(f"  H2 Coulomb 矩阵特征: shape={cm.shape}, 前4项={np.round(cm[:4], 3)}")


    atoms2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rdf = radial_distribution_histogram(atoms2, dr=0.2, r_max=2.0)
    print(f"  RDF 直方图 (前5项): {np.round(rdf[:5], 4)}")


def demo_physics_loss():
    print("\n" + "=" * 70)
    print("[10] 物理信息损失 (physics_informed_loss)")
    print("=" * 70)
    pil = PhysicsInformedLoss(lambda_energy=1.0, lambda_force=5.0, lambda_charge=2.0)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    q_pred = np.array([0.4, -0.4])
    energy_fn = lambda c: -1.0 / np.linalg.norm(c[0] - c[1])
    loss_total = pil.total_physics_loss(energy_fn, coords, q_pred, target_total_charge=0.0)
    print(f"\n物理损失 (H2 近似): {loss_total:.6f}")


    psi = np.array([0.5, 0.7, 0.5])
    V = np.array([-1.0, -1.0, -1.0])
    lapl = np.array([-0.2, 0.1, -0.2])
    res = schrodinger_residual_energy(psi, V, lapl, hbar=1.0, mass=1.0)
    print(f"  薛定谔方程残差: {res:.6f}")


def demo_gnn_training():
    print("\n" + "=" * 70)
    print("[11] GNN 分子性质预测训练与评估")
    print("=" * 70)


    dataset = SyntheticMoleculeDataset(n_samples=60, seed=42)
    train_idx, test_idx = dataset.train_test_split(ratio=0.8)
    print(f"\n数据集: {len(dataset)} 分子, 训练 {len(train_idx)}, 测试 {len(test_idx)}")


    model = MolecularMPNN(node_in=5, edge_in=4, hidden=16, n_layers=2)
    params = model.parameters()
    optimizer = AdamOptimizer(params, lr=5e-3)


    print("\n开始训练 (3 epochs, 每 epoch 采样 30 个分子)...")
    for epoch in range(3):

        sampled = np.random.choice(train_idx, size=min(30, len(train_idx)), replace=False).tolist()
        loss = train_epoch(model, dataset, sampled, optimizer, lambda_physics=0.01)
        print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")


    metrics = evaluate(model, dataset, test_idx)
    print(f"\n测试集评估:")
    print(f"  MAE (atomization energy): {metrics['mae_energy']:.4f} eV")
    print(f"  MAE (HOMO-LUMO gap):      {metrics['mae_gap']:.4f} eV")
    print(f"  平均偶然不确定性:         {metrics['mean_aleatoric']:.4f}")
    print(f"  平均认知不确定性:         {metrics['mean_epistemic']:.4f}")


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
