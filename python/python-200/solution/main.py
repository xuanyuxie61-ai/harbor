"""
main.py
=======
统一入口：基于自动微分的分子动力学高阶导数计算与材料热力学响应预测系统。

科学问题
--------
在高性能计算框架下，构建一个从原子势能到宏观热力学量的精确梯度传播系统。
传统分子动力学中，力场导数通常依赖有限差分近似，引入 O(h²) 截断误差；
本系统通过前向模式自动微分（Dual Number）和超对偶数（Hyper-Dual）实现
势能函数的一阶、二阶导数的机器精度计算，并融合谱方法、有限元、CVT 优化
采样、双调和弹性理论，完成多尺度材料响应分析。

核心数学框架
------------
1. 自动微分引擎
   Dual Number:    z = a + ε·b,   ε² = 0
   Hyper-Dual:     z = f₀ + ε₁f₁ + ε₂f₂ + ε₁ε₂f₁₂
   
   前向传播规则：
   (a+εa') + (b+εb') = (a+b) + ε(a'+b')
   (a+εa') · (b+εb') = ab + ε(ab'+a'b)

2. 势能模型
   Lennard-Jones:   V_LJ(r) = 4ε[(σ/r)^12 - (σ/r)^6]
   Gaussian:        V_G(r) = A·exp(-½ v^T Σ^{-1} v)

3. 分子动力学
   Velocity Verlet:
     r(t+Δt) = r(t) + v(t)Δt + ½a(t)Δt²
     v(t+Δt) = v(t) + ½[a(t)+a(t+Δt)]Δt

4. 双调和弹性方程
   d⁴u/dx⁴ = f(x), 固支边界条件
   应变能: U = (EI/2) ∫ (d²u/dx²)² dx

5. 热力学量
   C_V = var(E) / (k_B T²)
   g(r) = (V/N²) ⟨ Σ δ(r-r_ij) / shell_volume ⟩
   C_{ij} = (1/Ω) ∂²E/∂ε_i∂ε_j
"""

import numpy as np
import sys

# 导入所有模块
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
    """打印格式化的章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_autodiff_engine():
    """
    演示自动微分引擎的精度与功能。
    融合项目：autodiff_core（源自 interp_chebyshev 的数值微分思想）
    """
    print_section("模块 1: 自动微分引擎验证")

    # 测试函数: f(x,y) = sin(x) * exp(y) + x^2 * y
    def test_func_dual(vars):
        x, y = vars[0], vars[1]
        from autodiff_core import dual_sin, dual_exp
        return dual_sin(x) * dual_exp(y) + (x ** 2) * y

    x0 = np.array([1.0, 0.5])

    # 计算梯度（自动微分）
    grad_ad = grad_scalar_func_ad(test_func_dual, x0)
    # 解析梯度:
    # ∂f/∂x = cos(x)*exp(y) + 2*x*y
    # ∂f/∂y = sin(x)*exp(y) + x^2
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

    # 方向导数测试
    direction = np.array([0.6, 0.8])
    dir_der = directional_derivative_ad(test_func_dual, x0, direction)
    dir_der_exact = np.dot(grad_exact, direction)
    print(f"  方向导数 (v=[0.6,0.8]): AD={dir_der:.10f}, 解析={dir_der_exact:.10f}")

    # Hyper-Dual 二阶导数测试
    def test_func_hd(vars):
        from autodiff_core import hdual_sin, hdual_exp
        x, y = vars[0], vars[1]
        return hdual_sin(x) * hdual_exp(y) + (x ** 2) * y

    mixed = mixed_partial_hyperdual(test_func_hd, x0, 0, 1)
    # ∂²f/∂x∂y = cos(x)*exp(y) + 2*x
    mixed_exact = np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0]
    print(f"  混合偏导 ∂²f/∂x∂y: HyperDual={mixed:.10f}, 解析={mixed_exact:.10f}")

    return grad_ad, mixed


def demo_lattice_geometry():
    """
    晶格几何与 Diophantine 约束演示。
    融合项目: boundary_word_hexagon, pram, mcnuggets
    """
    print_section("模块 2: 晶格几何与 Diophantine 约束")

    # 六方边界词追踪
    word = "123456"
    imin, imax, jmin, jmax = boundary_range_hex(word)
    print(f"  六方边界词 '{word}' 的范围: i∈[{imin},{imax}], j∈[{jmin},{jmax}]")

    trace = boundary_trace_hex(word)
    print(f"  边界顶点数: {len(trace)}")

    # PRAM 边界词
    w_pram, p_pram = pram_boundary_word()
    print(f"  PRAM 边界词长度: {len(w_pram)}")

    # Diophantine 方程: 3x + 5y + 7z = 20
    a_coeff = [3, 5, 7]
    b_rhs = 20
    solutions = diophantine_nd_nonnegative(a_coeff, b_rhs)
    print(f"  Diophantine 方程 3x+5y+7z={b_rhs}")
    print(f"  非负整数解数量: {len(solutions)}")
    if len(solutions) > 0:
        print(f"  前 5 个解: {solutions[:5].tolist()}")

    # Frobenius 数
    g = frobenius_number_2d(5, 7)
    print(f"  Frobenius 数 g(5,7) = {g}")

    # 生成 HCP 晶格
    hcp_pts = generate_hcp_lattice_2d(3, 3, lattice_constant=1.0)
    print(f"  HCP 晶格 3×3 点数: {len(hcp_pts)}")

    # 配位数
    cn = coordination_number("hcp")
    print(f"  HCP 结构配位数: {cn}")
    print(f"  HCP Voronoi 单元面积: {voronoi_cell_area_hex(1.0):.6f}")

    return solutions


def demo_sampling_methods():
    """
    多维采样方法演示。
    融合项目: latin_center, triangle_grid, set_theory
    """
    print_section("模块 3: 高级采样方法")

    # Latin Center 采样
    lhs_pts = latin_center_sampling(2, 16)
    print(f"  Latin Center 采样: {len(lhs_pts)} 点，2D")

    # Latin Hypercube 采样
    lhs_pts2 = latin_hypercube_sampling(2, 16)
    print(f"  Latin Hypercube 采样: {len(lhs_pts2)} 点，2D")

    # 三角形网格
    tri_verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    tri_pts = triangle_grid_points(4, tri_verts)
    print(f"  三角形网格 (n=4): {len(tri_pts)} 点")

    # 集合划分
    n = 8
    # 定义等价关系：偶数一类，奇数一类
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i % 2) == (j % 2):
                R[i, j] = 1.0
    classes = set_partition_equivalence(n, R)
    print(f"  集合划分 (8元素按奇偶): {classes}")

    # Sobol-like 采样
    sobol_pts = sobol_like_sampling(2, 16)
    print(f"  准随机采样 (Halton-like): {len(sobol_pts)} 点")

    return lhs_pts


def demo_chebyshev_spectral():
    """
    Chebyshev 谱方法演示。
    融合项目: interp_chebyshev
    """
    print_section("模块 4: Chebyshev 谱插值与微分")

    # 测试函数：Runge 函数
    def runge_func(x):
        return 1.0 / (1.0 + 25.0 * x ** 2)

    n_nodes = 17
    a, b = -1.0, 1.0
    xd = chebyshev_nodes(a, b, n_nodes)
    yd = runge_func(xd)
    dd = divided_differences(xd, yd)

    # 在测试点求值
    xp = np.linspace(a, b, 101)
    yp_interp = newton_interpolate(xd, dd, xp)
    yp_exact = runge_func(xp)

    max_err = np.max(np.abs(yp_interp - yp_exact))
    print(f"  Runge 函数 Chebyshev 插值 (n={n_nodes})")
    print(f"  最大绝对误差: {max_err:.6e}")

    # 谱微分
    D = chebyshev_differentiation_matrix(n_nodes)
    cond = condition_number_estimate(D)
    print(f"  谱微分矩阵条件数: {cond:.3e}")

    # 测试 sin(x) 的导数
    f_sin = np.sin(xd)
    df_ad = chebyshev_derivative(f_sin)
    df_exact = np.cos(xd)
    der_err = np.max(np.abs(df_ad - df_exact))
    print(f"  sin(x) 谱微分最大误差: {der_err:.6e}")

    # 解 BVP: u'' = -sin(x), u(-1)=u(1)=0
    from chebyshev_spectral import chebyshev_spectral_solve_ode_bvp
    x_bvp, u_bvp = chebyshev_spectral_solve_ode_bvp(
        lambda x: (-1.0, 0.0, 0.0),  # a(x)=-1, b(x)=0, c(x)=0
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
    """
    CVT 优化采样演示。
    融合项目: cvt_1d_nonuniform
    """
    print_section("模块 5: CVT 非均匀采样优化")

    # 1D CVT
    cvt_1d = cvt_1d_nonuniform_python(
        n_generators=10,
        density_type=6,  # sin 密度
        n_steps=50,
        n_samples_per_step=500
    )
    print(f"  1D CVT (密度类型 6) 生成点: {np.round(cvt_1d, 4)}")

    # 2D CVT
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
    """
    分子动力学模拟演示。
    融合项目: md (分子动力学), lennard_jones (LJ 势), gaussian_2d
    """
    print_section("模块 6: 分子动力学模拟")

    # 创建 MD 引擎
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

    # 初始化
    md.initialize_positions_lattice("square")
    md.initialize_velocities_maxwell_boltzmann()

    # 运行模拟
    n_steps = 500
    print(f"  粒子数: {md.n}, 维度: {md.dim}")
    print(f"  时间步长: {md.dt}, 总步数: {n_steps}")

    with Timer() as t:
        results = md.run(n_steps=n_steps, equilibration_steps=200,
                         apply_thermostat=True)

    print(f"  模拟耗时: {t.elapsed:.4f} 秒")

    # 能量统计
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
    """
    使用自动微分计算 LJ 力的导数（Hessian）。
    融合项目: autodiff_core + potential_models
    """
    print_section("模块 7: 自动微分计算力场高阶导数")

    # 3 粒子测试系统
    pos = np.array([
        [0.0, 0.0],
        [1.2, 0.0],
        [0.6, 1.0]
    ])

    # 将 6D 坐标展平为向量，定义总势能为标量函数
    def potential_flat(x_flat):
        # x_flat 可能是 list[DualScalar] 或 ndarray
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

    # 与解析力对比
    from potential_models import total_forces_lj
    forces_ana = total_forces_lj(pos)
    forces_flat = -forces_ana.flatten()
    diff = np.max(np.abs(grad_f - forces_flat))
    print(f"  解析力向量:     {np.round(forces_flat, 6)}")
    print(f"  最大差异: {diff:.6e}")

    # Hessian（有限差分，作为参考）
    H = hessian_scalar_func_fd(lambda x: total_potential_lj(x.reshape(-1, 2)),
                                x_flat, h=1e-5)
    print(f"  Hessian 矩阵 (6×6) 条件数: {condition_number_estimate(H):.3e}")
    print(f"  Hessian 对称性检查: {np.max(np.abs(H - H.T)):.6e}")

    return grad_f, H


def demo_tetrahedral_fem():
    """
    四面体网格与有限元积分演示。
    融合项目: tet_mesh
    """
    print_section("模块 8: 四面体网格有限元积分")

    # 生成均匀盒状网格
    mesh = TetrahedralMesh.generate_uniform_box_mesh(
        nx=3, ny=3, nz=3,
        xlim=(0.0, 2.0), ylim=(0.0, 2.0), zlim=(0.0, 2.0)
    )
    print(f"  网格节点数: {mesh.n_nodes}")
    print(f"  网格单元数: {mesh.n_elements}")

    # 网格质量统计
    qstats = mesh.mesh_quality_stats()
    print(f"  网格质量: min={qstats['min']:.4f}, mean={qstats['mean']:.4f}, "
          f"max={qstats['max']:.4f}, var={qstats['var']:.6f}")

    # 积分测试：f(x,y,z) = x² + y² + z²
    def f_test(x, y, z):
        return x * x + y * y + z * z

    integral_val, total_vol = mesh.integrate_over_mesh(f_test, order=2)
    # 解析积分: ∫_0^2 ∫_0^2 ∫_0^2 (x²+y²+z²) dx dy dz = 3 * (8/3) * 4 = 32
    exact_integral = 32.0
    print(f"  积分测试: ∫(x²+y²+z²)dV")
    print(f"  数值积分: {integral_val:.6f}, 解析值: {exact_integral:.6f}")
    print(f"  积分误差: {abs(integral_val - exact_integral):.6e}")
    print(f"  总体积: {total_vol:.6f} (预期: 8.0)")

    # 局部密度场
    test_positions = np.array([
        [0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5],
        [1.0, 1.0, 1.0]
    ])
    density = compute_local_density_field(test_positions, mesh, smoothing_width=0.2)
    print(f"  局部密度场 (前5节点): {np.round(density[:5], 4)}")

    return mesh


def demo_biharmonic():
    """
    双调和方程求解演示。
    融合项目: biharmonic_fd1d
    """
    print_section("模块 9: 双调和弹性方程")

    # 求解 d⁴u/dx⁴ = 1, u(±1)=0, u'(±1)=0
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

    # 应变能
    EI = 1.0
    strain_energy = compute_strain_energy(u, h, young_modulus=EI, moment_of_inertia=1.0)
    print(f"  弯曲应变能: {strain_energy:.6f}")

    # 热-弹耦合：温度梯度引起的变形
    def temp_profile(x):
        return 0.5 * (x + 1.0) ** 2  # 线性温度梯度

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
    """
    热力学量计算演示。
    融合项目: md 数据 + histogram_display 统计思想
    """
    print_section("模块 10: 热力学量分析")

    energies = results['total_energy']
    temps = results['temperature']
    pressures = results['pressure']

    # 比热
    cv, cv_dulong = heat_capacity_cv(
        energies, np.mean(temps), md.n, md.dim
    )
    print(f"  定容热容 C_V: {cv:.6f} (Dulong-Petit: {cv_dulong:.6f})")

    # 弹性常数（有限应变法）
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

    # 径向分布函数
    r_bins, g_r = radial_distribution_function(
        pos_final, md.box, n_bins=30, rcut=md.box / 2.0
    )
    print(f"  径向分布函数: r∈[{r_bins[0]:.3f},{r_bins[-1]:.3f}], "
          f"g_max={np.max(g_r):.3f}")

    # 熵
    entropy = entropy_from_energy_distribution(energies, np.mean(temps), n_bins=15)
    print(f"  能量分布熵: {entropy:.6f}")

    # 直方图统计
    hist = HistogramStats(energies, n_bins=15)
    stats = hist.summary()
    print(f"  能量统计: 均值={stats['mean']:.4f}, 方差={stats['variance']:.6f}, "
          f"偏度={stats['skewness']:.4f}, 峰度={stats['kurtosis']:.4f}")

    return cv, g_r


def demo_full_pipeline():
    """
    完整多尺度计算流程演示。
    整合所有模块，展示从微观到宏观的梯度传播。
    """
    print_section("模块 11: 完整多尺度计算流程")

    # Step 1: 初始化采样（Latin + CVT）
    print("  [Step 1] 初始化采样...")
    init_pts = latin_center_sampling(2, 36, domain=(np.array([0.0, 0.0]), np.array([6.0, 6.0])))

    # Step 2: 晶格初始化
    print("  [Step 2] 构建晶格...")
    hcp = generate_hcp_lattice_2d(4, 4, 1.1)
    word = "12345612"
    trace = boundary_trace_hex(word)

    # Step 3: MD 模拟
    print("  [Step 3] 分子动力学模拟...")
    np.random.seed(123)
    md = MDEngine(n_particles=25, dim=2, dt=0.001, box_size=6.0,
                  temperature=0.8, tau_thermostat=0.05)
    md.initialize_positions_lattice("square")
    md.initialize_velocities_maxwell_boltzmann()
    results = md.run(n_steps=300, equilibration_steps=100)

    # Step 4: 自动微分力验证
    print("  [Step 4] 自动微分力计算...")
    pos = md.pos.copy()
    def pot_flat(xf):
        # xf 可能是 list[DualScalar]
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

    # Step 5: Chebyshev 插值势能面
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

    # Step 6: 四面体积分
    print("  [Step 6] 四面体有限元积分...")
    mesh = TetrahedralMesh.generate_uniform_box_mesh(2, 2, 2,
                                                        xlim=(0.0, 3.0),
                                                        ylim=(0.0, 3.0),
                                                        zlim=(0.0, 3.0))
    def density_3d(x, y, z):
        return np.exp(-(x**2 + y**2 + z**2) / 2.0)
    val_3d, vol_3d = mesh.integrate_over_mesh(density_3d, order=2)
    print(f"    高斯密度积分: {val_3d:.6f}")

    # Step 7: 双调和热弹性
    print("  [Step 7] 双调和热弹性分析...")
    x_bh, u_bh = solve_biharmonic_fd1d(
        lambda x: np.sin(np.pi * x), n=33,
        xlim=(-1.0, 1.0), bc_displacement=(0.0, 0.0), bc_slope=(0.0, 0.0)
    )
    se_bh = compute_strain_energy(u_bh, x_bh[1] - x_bh[0])
    print(f"    热应变能: {se_bh:.6f}")

    # Step 8: 热力学统计
    print("  [Step 8] 热力学统计...")
    cv, _ = heat_capacity_cv(results['total_energy'], 0.8, 25, 2)
    r_bins, g_r = radial_distribution_function(md.pos, md.box, n_bins=20)
    print(f"    C_V = {cv:.6f}")
    print(f"    g(r) 峰值 = {np.max(g_r):.4f}")

    # Step 9: Diophantine 晶格约束
    print("  [Step 9] Diophantine 晶格约束...")
    sols = diophantine_nd_nonnegative([2, 3, 5], 15)
    print(f"    2x+3y+5z=15 的解数: {len(sols)}")

    print("  多尺度流程完成。")


def main():
    """
    主函数：零参数运行，执行全部计算流程。
    """
    print("=" * 70)
    print("  AutoDiff-MD: 基于自动微分的分子动力学高阶导数计算平台")
    print("  科学领域: 高性能计算 — 自动微分与梯度计算")
    print("=" * 70)

    overall_timer = Timer()
    overall_timer.start()

    try:
        # 模块 1: 自动微分引擎
        demo_autodiff_engine()

        # 模块 2: 晶格几何
        demo_lattice_geometry()

        # 模块 3: 采样方法
        demo_sampling_methods()

        # 模块 4: Chebyshev 谱方法
        demo_chebyshev_spectral()

        # 模块 5: CVT 优化
        demo_cvt_optimizer()

        # 模块 6: 分子动力学
        results, md = demo_md_simulation()

        # 模块 7: AD 力场导数
        demo_autodiff_forces()

        # 模块 8: 四面体 FEM
        demo_tetrahedral_fem()

        # 模块 9: 双调和方程
        demo_biharmonic()

        # 模块 10: 热力学分析
        demo_thermodynamics(results, md)

        # 模块 11: 完整流程
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
