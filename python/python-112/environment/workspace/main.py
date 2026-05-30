
import numpy as np
import sys
import warnings


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




    print_section("1. 特殊函数与正交多项式验证")


    alpha = 0.5
    order = 8
    x_cc, w_cc = clenshaw_curtis_compute(order)

    max_err = 0.0
    for degree in range(min(2 * order, 15)):
        err = gegenbauer_exactness_monomial(degree, alpha, order, w_cc, x_cc)
        max_err = max(max_err, err)
    print(f"  Gegenbauer 求积最大相对误差 (degree <= {2*order-1}): {max_err:.3e}")


    val_2f1 = r8_hyper_2f1(0.5, 1.0, 1.5, 0.25)
    print(f"  2F1(0.5, 1.0; 1.5; 0.25) = {val_2f1:.8f}")


    val_psi = r8_psi(2.0)
    print(f"  Psi(2.0) = {val_psi:.8f} (理论: 1 - gamma = {1.0 - 0.57721566:.8f})")


    mu_n = np.array([3.37561065, 4.27534072, 5.13562230, 6.53025594])
    r_test = np.linspace(0.0, 1.0, 50)
    u_vib = membrane_vibration_bessel(r_test, t=1.0, mu_n=mu_n)
    print(f"  膜振动模式在 t=1.0 时的最大幅度: {np.max(np.abs(u_vib)):.4f} Å")


    g_screen = screened_coulomb_green(5.0, kappa=0.1, epsilon=80.0)
    print(f"  屏蔽 Coulomb Green(5.0 Å, kappa=0.1) = {g_screen:.6f} kcal/(mol·e²)")




    print_section("2. 稀疏矩阵与 FEM 质量/刚度矩阵")

    n_fem = 32
    M, K = assemble_mass_stiffness_1d(n_fem, L=30.0)
    M_sparse = SparseMatrix(n_fem + 1, n_fem + 1).from_dense(M)
    K_sparse = SparseMatrix(n_fem + 1, n_fem + 1).from_dense(K)
    print(f"  质量矩阵 M: 维度 {M_sparse.m}x{M_sparse.n}, 非零元 {M_sparse.nnz}")
    print(f"  刚度矩阵 K: 维度 {K_sparse.m}x{K_sparse.n}, 非零元 {K_sparse.nnz}")


    v_test = np.ones(n_fem + 1)
    Mv = M_sparse.spmv(v_test)
    Kv = K_sparse.spmv(v_test)
    print(f"  M * 1 范数: {np.linalg.norm(Mv):.4f}, K * 1 范数: {np.linalg.norm(Kv):.4f}")




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


    def water_density(sx, sy, sz):

        r2 = sx**2 + sy**2
        return np.exp(-r2/400.0) * (1.0 + 0.5 * np.abs(sz))

    g_water, e_water, m_water = cvt_3d_lumping(
        n=20, it_num=10, s_num=15,
        mu_fun=water_density,
        box=(-25.0, 25.0, -25.0, 25.0, -30.0, 30.0),
    )
    print(f"  3D 水分子 CVT 最终能量: {e_water[-1]:.4f}")
    print(f"  3D 水分子 CVT 最终平均位移: {m_water[-1]:.6f}")




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


    def collision_checker(partial_assignment):

        if len(partial_assignment) < 2:
            return True
        diff = abs(partial_assignment[-1] - partial_assignment[-2])
        return diff <= 2

    valid_conformations = backtrack_search(
        n_vars=4, domain_size=8, constraint_checker=collision_checker, max_solutions=20
    )
    print(f"  回溯法找到的可行构象数 (4键 x 8状态): {len(valid_conformations)}")




    print_section("6. 结合自由能热力学积分")




    delta_G = 0.0
    lam_nodes = np.array([])
    dU = np.array([])
    print("  [Hole 3a] 结合自由能热力学积分结果待修复")


    tri_example = np.array([
        [0.0, 0.0], [10.0, 0.0], [5.0, 8.66]
    ], dtype=float)
    def surf_energy_density(pts):

        return 0.03 * np.ones(pts.shape[0])

    G_surf = membrane_surface_free_energy([tri_example], surf_energy_density, rule_index=2)
    print(f"  示例三角形膜片表面自由能: {G_surf:.6f} kcal/mol")




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




    print_section("8. 蛋白骨架距离约束验证")


    locate, d_dist = test_partial_digest(k=6, dmax=20)
    print(f"  PDP 测试: 6 个点位于 {locate}")
    print(f"  成对距离: {np.sort(d_dist)}")


    n_ca = 10
    ca_coords = np.random.randn(n_ca, 3) * 5.0
    ca_coords[0] = [0.0, 0.0, 0.0]

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


    pairs = [(0, 1), (1, 2), (2, 3), (0, 3), (5, 8)]
    expected = np.array([3.8, 3.8, 3.8, 11.4, 11.4])
    bb_val = validate_backbone_distances(ca_coords, expected, pairs, tolerance=2.0)
    print(f"  骨架约束验证通过率: {bb_val['pass_rate']*100:.1f}%")
    print(f"  最大误差: {bb_val['max_error_A']:.3f} Å, RMSD: {bb_val['rmsd_A']:.3f} Å")




    print_section("9. 综合统计与收敛性分析")


    for dim in [2, 3, 4]:
        for level in [2, 3]:
            sg_size = sparse_grid_total_poly_size(dim, level)
            print(f"  稀疏网格维度={dim}, 层级={level}: 点数={sg_size}")




    sg_val = 0.0
    analytic = 1.0
    print("  [Hole 3b] 高斯函数稀疏网格积分验证待修复")




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
    sys.exit(main())
