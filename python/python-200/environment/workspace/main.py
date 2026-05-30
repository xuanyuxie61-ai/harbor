
import numpy as np
import sys


from autodiff_core import (
    DualScalar, HyperDualScalar,
    grad_scalar_func_ad, hessian_scalar_func_fd,
    mixed_partial_hyperdual, directional_derivative_ad
)
from potential_models import (
    lennard_jones_potential, lennard_jones_force,
    lennard_jones_dual, lennard_jones_hyperdual,
    gaussian_potential_2d, total_potential_lj,
    total_potential_with_gaussian, virial_stress_lj
)
from md_engine import MDEngine
from tetrahedral_fem import TetrahedralMesh, compute_local_density_field
from chebyshev_spectral import (
    chebyshev_nodes, divided_differences, newton_interpolate,
    chebyshev_interpolate, chebyshev_derivative,
    chebyshev_differentiation_matrix
)
from cvt_optimizer import CVTOptimizer, cvt_1d_nonuniform_python
from biharmonic_elasticity import (
    solve_biharmonic_fd1d, compute_strain_energy,
    thermal_load_from_gradient
)
from sampling_methods import (
    latin_hypercube_sampling, latin_center_sampling,
    triangle_grid_points, set_partition_equivalence,
    stratified_sampling, sobol_like_sampling
)
from thermodynamics import (
    heat_capacity_cv, elastic_constants_from_fluctuations,
    elastic_constants_strain_derivative,
    radial_distribution_function,
    thermal_expansion_estimate, entropy_from_energy_distribution
)
from lattice_geometry import (
    boundary_range_hex, boundary_trace_hex, pram_boundary_word,
    diophantine_nd_nonnegative, frobenius_number_2d,
    generate_hcp_lattice_2d, generate_square_lattice_2d,
    coordination_number, voronoi_cell_area_hex
)
from utils import (
    Timer, benchmark_function, HistogramStats,
    convergence_rate, relative_error, format_time_interval,
    condition_number_estimate, is_symmetric_positive_definite
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_autodiff_engine():
    print_section("模块 1: 自动微分引擎验证")


    def test_func_dual(vars):
        x, y = vars[0], vars[1]
        from autodiff_core import dual_sin, dual_exp
        return dual_sin(x) * dual_exp(y) + (x ** 2) * y

    x0 = np.array([1.0, 0.5])


    grad_ad = grad_scalar_func_ad(test_func_dual, x0)



    grad_exact = np.array([
        np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0] * x0[1],
        np.sin(x0[0]) * np.exp(x0[1]) + x0[0] ** 2
    ])

    print(f"  测试函数: f(x,y) = sin(x)·exp(y) + x²·y")
    print(f"  测试点: x = {x0}")
    print(f"  自动微分梯度:  [{grad_ad[0]:.10f}, {grad_ad[1]:.10f}]")
    print(f"  解析梯度:      [{grad_exact[0]:.10f}, {grad_exact[1]:.10f}]")
    print(f"  相对误差:      [{relative_error(grad_ad[0], grad_exact[0]):.2e}, "
          f"{relative_error(grad_ad[1], grad_exact[1]):.2e}]")


    direction = np.array([0.6, 0.8])
    dir_der = directional_derivative_ad(test_func_dual, x0, direction)
    dir_der_exact = np.dot(grad_exact, direction)
    print(f"  方向导数 (v=[0.6,0.8]): AD={dir_der:.10f}, 解析={dir_der_exact:.10f}")


    def test_func_hd(vars):
        from autodiff_core import hdual_sin, hdual_exp
        x, y = vars[0], vars[1]
        return hdual_sin(x) * hdual_exp(y) + (x ** 2) * y

    mixed = mixed_partial_hyperdual(test_func_hd, x0, 0, 1)

    mixed_exact = np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0]
    print(f"  混合偏导 ∂²f/∂x∂y: HyperDual={mixed:.10f}, 解析={mixed_exact:.10f}")

    return grad_ad, mixed


def demo_lattice_geometry():
    print_section("模块 2: 晶格几何与 Diophantine 约束")


    word = "123456"
    imin, imax, jmin, jmax = boundary_range_hex(word)
    print(f"  六方边界词 '{word}' 的范围: i∈[{imin},{imax}], j∈[{jmin},{jmax}]")

    trace = boundary_trace_hex(word)
    print(f"  边界顶点数: {len(trace)}")


    w_pram, p_pram = pram_boundary_word()
    print(f"  PRAM 边界词长度: {len(w_pram)}")


    a_coeff = [3, 5, 7]
    b_rhs = 20
    solutions = diophantine_nd_nonnegative(a_coeff, b_rhs)
    print(f"  Diophantine 方程 3x+5y+7z={b_rhs}")
    print(f"  非负整数解数量: {len(solutions)}")
    if len(solutions) > 0:
        print(f"  前 5 个解: {solutions[:5].tolist()}")


    g = frobenius_number_2d(5, 7)
    print(f"  Frobenius 数 g(5,7) = {g}")


    hcp_pts = generate_hcp_lattice_2d(3, 3, lattice_constant=1.0)
    print(f"  HCP 晶格 3×3 点数: {len(hcp_pts)}")


    cn = coordination_number("hcp")
    print(f"  HCP 结构配位数: {cn}")
    print(f"  HCP Voronoi 单元面积: {voronoi_cell_area_hex(1.0):.6f}")

    return solutions


def demo_sampling_methods():
    print_section("模块 3: 高级采样方法")


    lhs_pts = latin_center_sampling(2, 16)
    print(f"  Latin Center 采样: {len(lhs_pts)} 点，2D")


    lhs_pts2 = latin_hypercube_sampling(2, 16)
    print(f"  Latin Hypercube 采样: {len(lhs_pts2)} 点，2D")


    tri_verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    tri_pts = triangle_grid_points(4, tri_verts)
    print(f"  三角形网格 (n=4): {len(tri_pts)} 点")


    n = 8

    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i % 2) == (j % 2):
                R[i, j] = 1.0
    classes = set_partition_equivalence(n, R)
    print(f"  集合划分 (8元素按奇偶): {classes}")


    sobol_pts = sobol_like_sampling(2, 16)
    print(f"  准随机采样 (Halton-like): {len(sobol_pts)} 点")

    return lhs_pts


def demo_chebyshev_spectral():
    print_section("模块 4: Chebyshev 谱插值与微分")


    def runge_func(x):
        return 1.0 / (1.0 + 25.0 * x ** 2)

    n_nodes = 17
    a, b = -1.0, 1.0
    xd = chebyshev_nodes(a, b, n_nodes)
    yd = runge_func(xd)
    dd = divided_differences(xd, yd)


    xp = np.linspace(a, b, 101)
    yp_interp = newton_interpolate(xd, dd, xp)
    yp_exact = runge_func(xp)

    max_err = np.max(np.abs(yp_interp - yp_exact))
    print(f"  Runge 函数 Chebyshev 插值 (n={n_nodes})")
    print(f"  最大绝对误差: {max_err:.6e}")


    D = chebyshev_differentiation_matrix(n_nodes)
    cond = condition_number_estimate(D)
    print(f"  谱微分矩阵条件数: {cond:.3e}")


    f_sin = np.sin(xd)
    df_ad = chebyshev_derivative(f_sin)
    df_exact = np.cos(xd)
    der_err = np.max(np.abs(df_ad - df_exact))
    print(f"  sin(x) 谱微分最大误差: {der_err:.6e}")


    from chebyshev_spectral import chebyshev_spectral_solve_ode_bvp
    x_bvp, u_bvp = chebyshev_spectral_solve_ode_bvp(
        lambda x: (-1.0, 0.0, 0.0),
        lambda x: np.sin(x),
        n=16,
        bc_left=(0.0, None),
        bc_right=(0.0, None)
    )
    u_exact = np.sin(x_bvp)
    bvp_err = np.max(np.abs(u_bvp - u_exact))
    print(f"  BVP 谱方法误差: {bvp_err:.6e}")

    return max_err


def demo_cvt_optimizer():
    print_section("模块 5: CVT 非均匀采样优化")


    cvt_1d = cvt_1d_nonuniform_python(
        n_generators=10,
        density_type=6,
        n_steps=50,
        n_samples_per_step=500
    )
    print(f"  1D CVT (密度类型 6) 生成点: {np.round(cvt_1d, 4)}")


    cvt_2d = CVTOptimizer(
        dim=2, n_generators=16,
        domain=(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        density_func=lambda x: 1.0 + 2.0 * np.sin(np.pi * x[0]) ** 2,
        max_iter=30, tol=1e-4
    )
    cvt_2d.initialize_generators("latin_hypercube")
    final_pts = cvt_2d.optimize(sample_multiplier=30)
    print(f"  2D CVT 生成点数量: {len(final_pts)}")
    print(f"  2D CVT 能量历史 (最后5项): {cvt_2d.energy_history[-5:]}")

    weights = cvt_2d.get_sampling_weights()
    print(f"  采样权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")

    return final_pts


def demo_md_simulation():
    print_section("模块 6: 分子动力学模拟")


    np.random.seed(42)
    md = MDEngine(
        n_particles=36,
        dim=2,
        mass=1.0,
        dt=0.001,
        box_size=8.0,
        epsilon=1.0,
        sigma=1.0,
        rcut=2.5,
        temperature=1.0,
        tau_thermostat=0.05
    )


    md.initialize_positions_lattice("square")
    md.initialize_velocities_maxwell_boltzmann()


    n_steps = 500
    print(f"  粒子数: {md.n}, 维度: {md.dim}")
    print(f"  时间步长: {md.dt}, 总步数: {n_steps}")

    with Timer() as t:
        results = md.run(n_steps=n_steps, equilibration_steps=200,
                         apply_thermostat=True)

    print(f"  模拟耗时: {t.elapsed:.4f} 秒")


    e_total = results['total_energy']
    e_mean = np.mean(e_total)
    e_std = np.std(e_total, ddof=1)
    e_drift = abs(e_total[-1] - e_total[0]) / (abs(e_total[0]) + 1e-12)

    print(f"  总能量均值: {e_mean:.6f}, 标准差: {e_std:.6f}")
    print(f"  能量漂移: {e_drift:.6e}")
    print(f"  最终温度: {results['temperature'][-1]:.4f}")
    print(f"  最终压强: {results['pressure'][-1]:.4f}")

    return results, md


def demo_autodiff_forces():
    print_section("模块 7: 自动微分计算力场高阶导数")


    pos = np.array([
        [0.0, 0.0],
        [1.2, 0.0],
        [0.6, 1.0]
    ])


    def potential_flat(x_flat):

        n = len(x_flat)
        v = DualScalar(0.0, 0.0)
        from potential_models import lennard_jones_dual
        from autodiff_core import dual_sqrt
        for i in range(0, n, 2):
            for j in range(i + 2, n, 2):
                dx = x_flat[i] - x_flat[j]
                dy = x_flat[i+1] - x_flat[j+1]
                r_sq = dx * dx + dy * dy
                if hasattr(r_sq, 'val'):
                    r = dual_sqrt(r_sq)
                else:
                    r = DualScalar(np.sqrt(float(r_sq)), 0.0)
                v = v + lennard_jones_dual(r)
        return v

    x_flat = pos.flatten().astype(float)
    grad_f = grad_scalar_func_ad(potential_flat, x_flat)
    print(f"  3-粒子系统 (6D 坐标)")
    print(f"  自动微分力向量: {np.round(grad_f, 6)}")


    from potential_models import total_forces_lj
    forces_ana = total_forces_lj(pos)
    forces_flat = -forces_ana.flatten()
    diff = np.max(np.abs(grad_f - forces_flat))
    print(f"  解析力向量:     {np.round(forces_flat, 6)}")
    print(f"  最大差异: {diff:.6e}")


    H = hessian_scalar_func_fd(lambda x: total_potential_lj(x.reshape(-1, 2)),
                                x_flat, h=1e-5)
    print(f"  Hessian 矩阵 (6×6) 条件数: {condition_number_estimate(H):.3e}")
    print(f"  Hessian 对称性检查: {np.max(np.abs(H - H.T)):.6e}")

    return grad_f, H


def demo_tetrahedral_fem():
    print_section("模块 8: 四面体网格有限元积分")


    mesh = TetrahedralMesh.generate_uniform_box_mesh(
        nx=3, ny=3, nz=3,
        xlim=(0.0, 2.0), ylim=(0.0, 2.0), zlim=(0.0, 2.0)
    )
    print(f"  网格节点数: {mesh.n_nodes}")
    print(f"  网格单元数: {mesh.n_elements}")


    qstats = mesh.mesh_quality_stats()
    print(f"  网格质量: min={qstats['min']:.4f}, mean={qstats['mean']:.4f}, "
          f"max={qstats['max']:.4f}, var={qstats['var']:.6f}")


    def f_test(x, y, z):
        return x * x + y * y + z * z

    integral_val, total_vol = mesh.integrate_over_mesh(f_test, order=2)

    exact_integral = 32.0
    print(f"  积分测试: ∫(x²+y²+z²)dV")
    print(f"  数值积分: {integral_val:.6f}, 解析值: {exact_integral:.6f}")
    print(f"  积分误差: {abs(integral_val - exact_integral):.6e}")
    print(f"  总体积: {total_vol:.6f} (预期: 8.0)")


    test_positions = np.array([
        [0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5],
        [1.0, 1.0, 1.0]
    ])
    density = compute_local_density_field(test_positions, mesh, smoothing_width=0.2)
    print(f"  局部密度场 (前5节点): {np.round(density[:5], 4)}")

    return mesh


def demo_biharmonic():
    print_section("模块 9: 双调和弹性方程")


    def load_func(x):
        return 1.0

    x, u = solve_biharmonic_fd1d(
        load_func, n=65,
        xlim=(-1.0, 1.0),
        bc_displacement=(0.0, 0.0),
        bc_slope=(0.0, 0.0)
    )

    h = x[1] - x[0]
    u_max = np.max(np.abs(u))
    print(f"  固支梁, 均布载荷 f(x)=1")
    print(f"  最大挠度: {u_max:.6f}")


    EI = 1.0
    strain_energy = compute_strain_energy(u, h, young_modulus=EI, moment_of_inertia=1.0)
    print(f"  弯曲应变能: {strain_energy:.6f}")


    def temp_profile(x):
        return 0.5 * (x + 1.0) ** 2

    thermal_load = thermal_load_from_gradient(
        temp_profile, x,
        alpha=1.0, young_modulus=1.0, thickness=0.1, poisson_ratio=0.3
    )
    x_therm, u_therm = solve_biharmonic_fd1d(
        lambda xi: np.interp(xi, x, thermal_load),
        n=65, xlim=(-1.0, 1.0),
        bc_displacement=(0.0, 0.0), bc_slope=(0.0, 0.0)
    )
    print(f"  热载荷最大挠度: {np.max(np.abs(u_therm)):.6f}")

    return x, u


def demo_thermodynamics(results: dict, md: MDEngine):
    print_section("模块 10: 热力学量分析")

    energies = results['total_energy']
    temps = results['temperature']
    pressures = results['pressure']


    cv, cv_dulong = heat_capacity_cv(
        energies, np.mean(temps), md.n, md.dim
    )
    print(f"  定容热容 C_V: {cv:.6f} (Dulong-Petit: {cv_dulong:.6f})")


    pos_final = md.pos.copy()
    volume = md.box ** md.dim
    C_elastic = elastic_constants_strain_derivative(
        pos_final, md.epsilon, md.sigma, md.rcut, volume,
        strain_perturbation=1e-4
    )
    print(f"  弹性常数矩阵 (有限应变法):")
    print(f"    C_11 = {C_elastic[0,0]:.4f}")
    if C_elastic.shape[0] > 1:
        print(f"    C_22 = {C_elastic[1,1]:.4f}")


    r_bins, g_r = radial_distribution_function(
        pos_final, md.box, n_bins=30, rcut=md.box / 2.0
    )
    print(f"  径向分布函数: r∈[{r_bins[0]:.3f},{r_bins[-1]:.3f}], "
          f"g_max={np.max(g_r):.3f}")


    entropy = entropy_from_energy_distribution(energies, np.mean(temps), n_bins=15)
    print(f"  能量分布熵: {entropy:.6f}")


    hist = HistogramStats(energies, n_bins=15)
    stats = hist.summary()
    print(f"  能量统计: 均值={stats['mean']:.4f}, 方差={stats['variance']:.6f}, "
          f"偏度={stats['skewness']:.4f}, 峰度={stats['kurtosis']:.4f}")

    return cv, g_r


def demo_full_pipeline():
    print_section("模块 11: 完整多尺度计算流程")


    print("  [Step 1] 初始化采样...")
    init_pts = latin_center_sampling(2, 36, domain=(np.array([0.0, 0.0]), np.array([6.0, 6.0])))


    print("  [Step 2] 构建晶格...")
    hcp = generate_hcp_lattice_2d(4, 4, 1.1)
    word = "12345612"
    trace = boundary_trace_hex(word)


    print("  [Step 3] 分子动力学模拟...")
    np.random.seed(123)
    md = MDEngine(n_particles=25, dim=2, dt=0.001, box_size=6.0,
                  temperature=0.8, tau_thermostat=0.05)
    md.initialize_positions_lattice("square")
    md.initialize_velocities_maxwell_boltzmann()
    results = md.run(n_steps=300, equilibration_steps=100)


    print("  [Step 4] 自动微分力计算...")
    pos = md.pos.copy()
    def pot_flat(xf):

        n = len(xf)
        v = DualScalar(0.0, 0.0)
        from potential_models import lennard_jones_dual
        from autodiff_core import dual_sqrt
        for i in range(0, n, 2):
            for j in range(i + 2, n, 2):
                dx = xf[i] - xf[j]
                dy = xf[i+1] - xf[j+1]
                r_sq = dx * dx + dy * dy
                if hasattr(r_sq, 'val'):
                    r = dual_sqrt(r_sq)
                else:
                    r = DualScalar(np.sqrt(float(r_sq)), 0.0)
                v = v + lennard_jones_dual(r)
        return v
    grad_ad = grad_scalar_func_ad(pot_flat, pos.flatten())


    print("  [Step 5] Chebyshev 谱插值...")
    def pot_1d(r):
        return lennard_jones_potential(r, 1.0, 1.0)
    r_test = np.linspace(0.9, 3.0, 50)
    r_cheb = chebyshev_nodes(0.9, 3.0, 12)
    y_cheb = np.array([pot_1d(ri) for ri in r_cheb])
    dd = divided_differences(r_cheb, y_cheb)
    y_interp = newton_interpolate(r_cheb, dd, r_test)
    y_exact = np.array([pot_1d(ri) for ri in r_test])
    cheb_err = np.max(np.abs(y_interp - y_exact))
    print(f"    LJ 势能 Chebyshev 插值误差: {cheb_err:.6e}")


    print("  [Step 6] 四面体有限元积分...")
    mesh = TetrahedralMesh.generate_uniform_box_mesh(2, 2, 2,
                                                        xlim=(0.0, 3.0),
                                                        ylim=(0.0, 3.0),
                                                        zlim=(0.0, 3.0))
    def density_3d(x, y, z):
        return np.exp(-(x**2 + y**2 + z**2) / 2.0)
    val_3d, vol_3d = mesh.integrate_over_mesh(density_3d, order=2)
    print(f"    高斯密度积分: {val_3d:.6f}")


    print("  [Step 7] 双调和热弹性分析...")
    x_bh, u_bh = solve_biharmonic_fd1d(
        lambda x: np.sin(np.pi * x), n=33,
        xlim=(-1.0, 1.0), bc_displacement=(0.0, 0.0), bc_slope=(0.0, 0.0)
    )
    se_bh = compute_strain_energy(u_bh, x_bh[1] - x_bh[0])
    print(f"    热应变能: {se_bh:.6f}")


    print("  [Step 8] 热力学统计...")
    cv, _ = heat_capacity_cv(results['total_energy'], 0.8, 25, 2)
    r_bins, g_r = radial_distribution_function(md.pos, md.box, n_bins=20)
    print(f"    C_V = {cv:.6f}")
    print(f"    g(r) 峰值 = {np.max(g_r):.4f}")


    print("  [Step 9] Diophantine 晶格约束...")
    sols = diophantine_nd_nonnegative([2, 3, 5], 15)
    print(f"    2x+3y+5z=15 的解数: {len(sols)}")

    print("  多尺度流程完成。")


def main():
    print("=" * 70)
    print("  AutoDiff-MD: 基于自动微分的分子动力学高阶导数计算平台")
    print("  科学领域: 高性能计算 — 自动微分与梯度计算")
    print("=" * 70)

    overall_timer = Timer()
    overall_timer.start()

    try:

        demo_autodiff_engine()


        demo_lattice_geometry()


        demo_sampling_methods()


        demo_chebyshev_spectral()


        demo_cvt_optimizer()


        results, md = demo_md_simulation()


        demo_autodiff_forces()


        demo_tetrahedral_fem()


        demo_biharmonic()


        demo_thermodynamics(results, md)


        demo_full_pipeline()

    except Exception as e:
        print(f"\n[ERROR] 计算过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    overall_timer.stop()
    print("\n" + "=" * 70)
    print(f"  全部计算完成，总耗时: {format_time_interval(overall_timer.elapsed)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
