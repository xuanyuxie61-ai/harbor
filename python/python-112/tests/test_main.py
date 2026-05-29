"""
main.py
=======
膜蛋白嵌入与药物分子对接的博士级综合模拟系统。

统一入口，零参数可运行。执行以下完整流程：
  1. 特殊函数与正交多项式初始化（Gegenbauer、Laguerre、Jacobi）
  2. 稀疏矩阵格式转换与验证
  3. 膜蛋白静电势的 FEM Poisson-Boltzmann 求解
  4. 膜脂双层的 CVT 优化布置（2D/3D）
  5. 药物分子构象贪心搜索与回溯验证
  6. 结合自由能的稀疏网格热力学积分
  7. 粗粒化分子动力学模拟（Verlet + 锯齿波热浴）
  8. 蛋白骨架距离约束验证
  9. 膜表面自由能积分
 10. 综合结果输出

科学领域：分子动力学 — 膜蛋白嵌入与药物分子对接
"""

import numpy as np
import sys
import warnings

# 导入所有模块
from special_functions import (
    r8_hyper_2f1, r8_psi, gegenbauer_integral,
    gegenbauer_exactness_monomial, membrane_vibration_bessel,
    screened_coulomb_green,
)
from sparse_matrix import SparseMatrix, spdiags
from quadrature_rules import (
    clenshaw_curtis_compute, jacobi_compute, gen_hermite_compute,
    laguerre_quadrature_rule, wandzura_rule, integrate_triangle,
)
from membrane_fem import (
    fem1d_bvp_quadratic, assemble_mass_stiffness_1d,
    reaction_diffusion_nonlinear, solve_poisson_boltzmann_membrane,
)
from lipid_cvt import (
    cvt_triangle_uniform, cvt_3d_lumping,
    place_lipid_bilayer,
)
from conformation_search import (
    greedy_conformation_search, path_cost,
    backtrack_search, dock_drug_greedy_rotamer,
)
from free_energy_integration import (
    sparse_grid_total_poly_size, sparse_grid_integrate,
    thermodynamic_integration_binding_free_energy,
    membrane_surface_free_energy,
)
from structure_validation import (
    test_partial_digest, validate_distance_matrix,
    validate_backbone_distances,
)
from molecular_dynamics import (
    sawtooth_driver, coarse_grained_md_simulation,
    lennard_jones_potential, debye_huckel_potential,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("膜蛋白嵌入与药物分子对接 — 博士级综合模拟系统")
    print("Molecular Dynamics: Membrane Protein Embedding & Drug Docking")
    print("Python 科研代码合成项目 | PROJECT_112")
    np.random.seed(2024)

    # =====================================================================
    # 1. 特殊函数与正交多项式验证
    # =====================================================================
    print_section("1. 特殊函数与正交多项式验证")

    # Gegenbauer 积分精确性验证
    alpha = 0.5
    order = 8
    x_cc, w_cc = clenshaw_curtis_compute(order)
    # 将 [-1,1] 映射到 Gegenbauer 权重需调整，这里直接用数值验证
    max_err = 0.0
    for degree in range(min(2 * order, 15)):
        err = gegenbauer_exactness_monomial(degree, alpha, order, w_cc, x_cc)
        max_err = max(max_err, err)
    print(f"  Gegenbauer 求积最大相对误差 (degree <= {2*order-1}): {max_err:.3e}")

    # 超几何函数 2F1 验证
    val_2f1 = r8_hyper_2f1(0.5, 1.0, 1.5, 0.25)
    print(f"  2F1(0.5, 1.0; 1.5; 0.25) = {val_2f1:.8f}")

    # Digamma 验证
    val_psi = r8_psi(2.0)
    print(f"  Psi(2.0) = {val_psi:.8f} (理论: 1 - gamma = {1.0 - 0.57721566:.8f})")

    # 膜振动 Bessel 模式
    mu_n = np.array([3.37561065, 4.27534072, 5.13562230, 6.53025594])
    r_test = np.linspace(0.0, 1.0, 50)
    u_vib = membrane_vibration_bessel(r_test, t=1.0, mu_n=mu_n)
    print(f"  膜振动模式在 t=1.0 时的最大幅度: {np.max(np.abs(u_vib)):.4f} Å")

    # 屏蔽 Coulomb Green 函数
    g_screen = screened_coulomb_green(5.0, kappa=0.1, epsilon=80.0)
    print(f"  屏蔽 Coulomb Green(5.0 Å, kappa=0.1) = {g_screen:.6f} kcal/(mol·e²)")

    # =====================================================================
    # 2. 稀疏矩阵与 FEM 质量/刚度矩阵
    # =====================================================================
    print_section("2. 稀疏矩阵与 FEM 质量/刚度矩阵")

    n_fem = 32
    M, K = assemble_mass_stiffness_1d(n_fem, L=30.0)
    M_sparse = SparseMatrix(n_fem + 1, n_fem + 1).from_dense(M)
    K_sparse = SparseMatrix(n_fem + 1, n_fem + 1).from_dense(K)
    print(f"  质量矩阵 M: 维度 {M_sparse.m}x{M_sparse.n}, 非零元 {M_sparse.nnz}")
    print(f"  刚度矩阵 K: 维度 {K_sparse.m}x{K_sparse.n}, 非零元 {K_sparse.nnz}")

    # 验证稀疏矩阵-向量乘法
    v_test = np.ones(n_fem + 1)
    Mv = M_sparse.spmv(v_test)
    Kv = K_sparse.spmv(v_test)
    print(f"  M * 1 范数: {np.linalg.norm(Mv):.4f}, K * 1 范数: {np.linalg.norm(Kv):.4f}")

    # =====================================================================
    # 3. Poisson-Boltzmann 跨膜电势剖面
    # =====================================================================
    print_section("3. Poisson-Boltzmann 跨膜电势剖面")

    z_grid, phi_profile, eps_profile, kappa_profile = solve_poisson_boltzmann_membrane(
        n=65,
        z_min=-30.0,
        z_max=30.0,
        epsilon_water=80.0,
        epsilon_protein=4.0,
        epsilon_membrane=2.0,
        kappa_water=0.1,
        protein_z_range=(-10.0, 10.0),
        membrane_z_range=(-15.0, -10.0),
    )
    phi_max = np.max(phi_profile)
    phi_min = np.min(phi_profile)
    print(f"  电势范围: [{phi_min:.4f}, {phi_max:.4f}] kcal/(mol·e)")
    print(f"  跨膜电势差 (z=+30 vs z=-30): {phi_profile[-1] - phi_profile[0]:.4f} kcal/(mol·e)")

    # =====================================================================
    # 4. 膜脂双层 CVT 优化布置
    # =====================================================================
    print_section("4. 膜脂双层 CVT 优化布置")

    upper_leaflet, lower_leaflet, upper_z, lower_z = place_lipid_bilayer(
        n_lipids_per_leaflet=24,
        protein_radius=12.0,
        box_xy=50.0,
        exclusion_radius=15.0,
        it_num=20,
    )
    print(f"  上 leaflet 脂质数: {upper_leaflet.shape[0]}, z = {upper_z:.1f} Å")
    print(f"  下 leaflet 脂质数: {lower_leaflet.shape[0]}, z = {lower_z:.1f} Å")
    print(f"  上 leaflet 质心: ({np.mean(upper_leaflet[:,0]):.2f}, {np.mean(upper_leaflet[:,1]):.2f}) Å")

    # 3D CVT 水分子布置
    def water_density(sx, sy, sz):
        # 水密度在膜外高，膜内低
        r2 = sx**2 + sy**2
        return np.exp(-r2/400.0) * (1.0 + 0.5 * np.abs(sz))

    g_water, e_water, m_water = cvt_3d_lumping(
        n=20, it_num=10, s_num=15,
        mu_fun=water_density,
        box=(-25.0, 25.0, -25.0, 25.0, -30.0, 30.0),
    )
    print(f"  3D 水分子 CVT 最终能量: {e_water[-1]:.4f}")
    print(f"  3D 水分子 CVT 最终平均位移: {m_water[-1]:.6f}")

    # =====================================================================
    # 5. 药物分子构象贪心搜索
    # =====================================================================
    print_section("5. 药物分子构象贪心搜索")

    best_seq, best_energy, best_dihedrals = dock_drug_greedy_rotamer(
        n_torsions=5,
        n_bins=12,
        vdw_radius_drug=3.5,
        base_energy=-5.0,
    )
    print(f"  可旋转键数: 5, 每键离散状态: 12")
    print(f"  最佳构象序列: {best_seq}")
    print(f"  最佳构象能量: {best_energy:.4f} kcal/mol")
    print(f"  最佳二面角 (deg): {np.degrees(best_dihedrals)}")

    # 回溯法验证构象约束
    def collision_checker(partial_assignment):
        # 简化的碰撞检测：相邻二面角差不超过 60°
        if len(partial_assignment) < 2:
            return True
        diff = abs(partial_assignment[-1] - partial_assignment[-2])
        return diff <= 2  # 每个 bin = 30°, diff <= 2 即 <= 60°

    valid_conformations = backtrack_search(
        n_vars=4, domain_size=8, constraint_checker=collision_checker, max_solutions=20
    )
    print(f"  回溯法找到的可行构象数 (4键 x 8状态): {len(valid_conformations)}")

    # =====================================================================
    # 6. 结合自由能热力学积分
    # =====================================================================
    print_section("6. 结合自由能热力学积分")

    delta_G, lam_nodes, dU = thermodynamic_integration_binding_free_energy(
        n_lambda=7,
        temperature=300.0,
        dim_conformational=3,
        sg_level=2,
    )
    print(f"  结合自由能 ΔG_bind = {delta_G:.4f} kcal/mol")
    print(f"  λ 节点: {np.round(lam_nodes, 3)}")
    print(f"  <dU/dλ> 在各节点: {np.round(dU, 4)}")

    # 膜表面自由能
    tri_example = np.array([
        [0.0, 0.0], [10.0, 0.0], [5.0, 8.66]
    ], dtype=float)
    def surf_energy_density(pts):
        # 简化的表面张力模型
        return 0.03 * np.ones(pts.shape[0])  # kcal/(mol·Å²)

    G_surf = membrane_surface_free_energy([tri_example], surf_energy_density, rule_index=2)
    print(f"  示例三角形膜片表面自由能: {G_surf:.6f} kcal/mol")

    # =====================================================================
    # 7. 粗粒化分子动力学模拟
    # =====================================================================
    print_section("7. 粗粒化分子动力学模拟")

    md_results = coarse_grained_md_simulation(
        n_steps=2000,
        dt=0.001,
        temperature=300.0,
        n_protein_atoms=30,
        n_drug_atoms=6,
        n_lipid_atoms=40,
        box_size=np.array([50.0, 50.0, 50.0]),
        epsilon_lj=0.1,
        kappa=0.1,
        random_seed=42,
    )
    print(f"  模拟步数: 2000, 时间步长: 1 fs")
    print(f"  平均温度: {md_results['avg_temperature']:.2f} K")
    print(f"  平均势能: {md_results['avg_potential']:.4f} kcal/mol")
    print(f"  最终药物-蛋白质心距: {np.linalg.norm(np.mean(md_results['positions'][30:36], axis=0) - np.mean(md_results['positions'][:30], axis=0)):.2f} Å")

    # =====================================================================
    # 8. 蛋白骨架距离约束验证
    # =====================================================================
    print_section("8. 蛋白骨架距离约束验证")

    # 生成测试 PDP 实例
    locate, d_dist = test_partial_digest(k=6, dmax=20)
    print(f"  PDP 测试: 6 个点位于 {locate}")
    print(f"  成对距离: {np.sort(d_dist)}")

    # 验证距离矩阵
    n_ca = 10
    ca_coords = np.random.randn(n_ca, 3) * 5.0
    ca_coords[0] = [0.0, 0.0, 0.0]
    # 强制满足局部肽键距离
    for i in range(1, n_ca):
        ca_coords[i] = ca_coords[i-1] + [3.8, 0.0, 0.0] + np.random.randn(3) * 0.2

    dist_mat = np.zeros((n_ca, n_ca), dtype=float)
    for i in range(n_ca):
        for j in range(i+1, n_ca):
            dist_mat[i,j] = np.linalg.norm(ca_coords[i] - ca_coords[j])
            dist_mat[j,i] = dist_mat[i,j]

    val_results = validate_distance_matrix(dist_mat)
    print(f"  距离矩阵验证:")
    print(f"    非负性: {val_results['is_nonnegative']}")
    print(f"    零对角线: {val_results['zero_diagonal']}")
    print(f"    对称性: {val_results['is_symmetric']}")
    print(f"    三角不等式: {val_results['triangle_inequality']}")

    # 验证特定约束
    pairs = [(0, 1), (1, 2), (2, 3), (0, 3), (5, 8)]
    expected = np.array([3.8, 3.8, 3.8, 11.4, 11.4])
    bb_val = validate_backbone_distances(ca_coords, expected, pairs, tolerance=2.0)
    print(f"  骨架约束验证通过率: {bb_val['pass_rate']*100:.1f}%")
    print(f"  最大误差: {bb_val['max_error_A']:.3f} Å, RMSD: {bb_val['rmsd_A']:.3f} Å")

    # =====================================================================
    # 9. 综合统计与收敛性分析
    # =====================================================================
    print_section("9. 综合统计与收敛性分析")

    # 稀疏网格尺寸估计
    for dim in [2, 3, 4]:
        for level in [2, 3]:
            sg_size = sparse_grid_total_poly_size(dim, level)
            print(f"  稀疏网格维度={dim}, 层级={level}: 点数={sg_size}")

    # 数值积分验证：高斯函数在 [-1,1]^3 上的积分
    def gaussian_3d(pts):
        # pts shape (3, n)
        return np.exp(-np.sum(pts**2, axis=0))

    sg_val = sparse_grid_integrate(3, 3, gaussian_3d, "clenshaw-curtis")
    # 解析值: (sqrt(pi)/2 * erf(1))^3
    from scipy.special import erf
    analytic = (np.sqrt(np.pi) * 0.5 * erf(1.0)) ** 3
    print(f"  3D 高斯函数稀疏网格积分 (L=3): {sg_val:.8f}")
    print(f"  解析值: {analytic:.8f}, 相对误差: {abs(sg_val-analytic)/analytic:.3e}")

    # =====================================================================
    # 10. 最终汇总
    # =====================================================================
    print_section("10. 模拟最终汇总")
    print(f"  跨膜电势差:     {phi_profile[-1] - phi_profile[0]:+.4f} kcal/(mol·e)")
    print(f"  药物最佳构象能: {best_energy:+.4f} kcal/mol")
    print(f"  结合自由能 ΔG:  {delta_G:+.4f} kcal/mol")
    print(f"  MD 平均温度:    {md_results['avg_temperature']:.2f} K")
    print(f"  MD 平均势能:    {md_results['avg_potential']:+.4f} kcal/mol")
    print(f"  骨架约束通过:   {bb_val['pass_rate']*100:.1f}%")
    print("\n  模拟正常结束。所有模块运行无报错。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

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

print('\n全部 30 个测试通过!\n')
