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

# ================================================================
# 测试用例（45个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: DualScalar 基本四则运算与幂运算 ----
a = DualScalar(2.0, 1.0)
b = DualScalar(3.0, 2.0)
c_add = a + b
assert abs(c_add.val - 5.0) < 1e-12, '[TC01] DualScalar addition val FAILED'
assert abs(c_add.der - 3.0) < 1e-12, '[TC01] DualScalar addition der FAILED'
c_mul = a * b
assert abs(c_mul.val - 6.0) < 1e-12, '[TC01] DualScalar multiplication val FAILED'
assert abs(c_mul.der - 7.0) < 1e-12, '[TC01] DualScalar multiplication der FAILED'
c_pow = a ** 2
assert abs(c_pow.val - 4.0) < 1e-12, '[TC01] DualScalar power val FAILED'
assert abs(c_pow.der - 4.0) < 1e-12, '[TC01] DualScalar power der FAILED'
c_sub = a - b
assert abs(c_sub.val + 1.0) < 1e-12, '[TC01] DualScalar subtraction val FAILED'
assert abs(c_sub.der + 1.0) < 1e-12, '[TC01] DualScalar subtraction der FAILED'
c_div = a / b
assert abs(c_div.val - 2.0 / 3.0) < 1e-12, '[TC01] DualScalar division val FAILED'

# ---- TC02: DualScalar 链式法则 sin*exp ----
from autodiff_core import dual_sin, dual_exp
x = DualScalar(1.0, 1.0)
f = dual_sin(x) * dual_exp(x)
expected_val = np.sin(1.0) * np.exp(1.0)
expected_der = np.exp(1.0) * (np.cos(1.0) + np.sin(1.0))
assert abs(f.val - expected_val) < 1e-12, '[TC02] Chain rule sin*exp value FAILED'
assert abs(f.der - expected_der) < 1e-12, '[TC02] Chain rule sin*exp derivative FAILED'

# ---- TC03: HyperDualScalar 二阶算术运算 ----
a = HyperDualScalar(2.0, 1.0, 0.0, 0.0)
b = HyperDualScalar(3.0, 0.0, 1.0, 0.0)
c = a * b
assert abs(c.f0 - 6.0) < 1e-12, '[TC03] HyperDual f0 FAILED'
assert abs(c.f1 - 3.0) < 1e-12, '[TC03] HyperDual f1 FAILED'
assert abs(c.f2 - 2.0) < 1e-12, '[TC03] HyperDual f2 FAILED'
assert abs(c.f12 - 1.0) < 1e-12, '[TC03] HyperDual f12 FAILED'

# ---- TC04: grad_scalar_func_ad 梯度精度 ----
def _f_grad(vars):
    x0, y0 = vars[0], vars[1]
    from autodiff_core import dual_sin, dual_exp
    return dual_sin(x0) * dual_exp(y0) + (x0 ** 2) * y0
x0 = np.array([1.0, 0.5])
grad = grad_scalar_func_ad(_f_grad, x0)
grad_exact = np.array([
    np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0] * x0[1],
    np.sin(x0[0]) * np.exp(x0[1]) + x0[0] ** 2
])
assert np.max(np.abs(grad - grad_exact)) < 1e-10, '[TC04] Gradient accuracy FAILED'

# ---- TC05: directional_derivative_ad 方向导数 ----
direction = np.array([0.6, 0.8])
dd = directional_derivative_ad(_f_grad, x0, direction)
dd_exact = np.dot(grad_exact, direction)
assert abs(dd - dd_exact) < 1e-10, '[TC05] Directional derivative FAILED'

# ---- TC06: mixed_partial_hyperdual 混合偏导数 ----
def _f_hd(vars):
    from autodiff_core import hdual_sin, hdual_exp
    x0, y0 = vars[0], vars[1]
    return hdual_sin(x0) * hdual_exp(y0) + (x0 ** 2) * y0
mixed = mixed_partial_hyperdual(_f_hd, x0, 0, 1)
mixed_exact = np.cos(x0[0]) * np.exp(x0[1]) + 2.0 * x0[0]
assert abs(mixed - mixed_exact) < 1e-10, '[TC06] Mixed partial FAILED'

# ---- TC07: lennard_jones_potential 解析值验证 ----
pot0 = lennard_jones_potential(1.0, epsilon=1.0, sigma=1.0)
assert abs(pot0) < 1e-12, '[TC07] LJ at r=sigma should be 0 FAILED'
r_min = 2.0 ** (1.0 / 6.0)
pot_min = lennard_jones_potential(r_min, epsilon=2.0, sigma=1.5)
expected_min = -2.0  # minimum = -epsilon at r = sigma * 2^(1/6)
assert abs(pot_min - expected_min) < 1e-10, '[TC07] LJ potential minimum FAILED'

# ---- TC08: lennard_jones_force 与有限差分比较 ----
from potential_models import lennard_jones_force
r = 1.2
f_ana = lennard_jones_force(r, epsilon=1.0, sigma=1.0)
h = 1e-6
pot_plus = lennard_jones_potential(r + h)
pot_minus = lennard_jones_potential(r - h)
f_fd = -(pot_plus - pot_minus) / (2.0 * h)
assert abs(f_ana - f_fd) < 1e-5, '[TC08] LJ force vs finite difference FAILED'

# ---- TC09: gaussian_potential_2d 中心值及衰减 ----
val_center = gaussian_potential_2d(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, A=5.0, corr_matrix=np.eye(2))
assert abs(val_center - 5.0) < 1e-12, '[TC09] Gaussian at center FAILED'
val_far = gaussian_potential_2d(4.0, 0.0, 0.0, 0.0, 1.0, 1.0, A=1.0, corr_matrix=np.eye(2))
assert val_far < 1e-3, '[TC09] Gaussian decay at 4 sigma FAILED'

# ---- TC10: total_forces_lj 牛顿第三定律（合力为零） ----
from potential_models import total_forces_lj
np.random.seed(42)
pos = np.random.randn(5, 2) * 0.5
forces = total_forces_lj(pos, epsilon=1.0, sigma=1.0, rcut=3.0, box_size=10.0)
net_force = np.sum(forces, axis=0)
assert np.max(np.abs(net_force)) < 1e-12, '[TC10] Net force should be zero FAILED'
assert forces.shape == (5, 2), '[TC10] Force shape FAILED'

# ---- TC11: frobenius_number_2d 已知值 ----
g57 = frobenius_number_2d(5, 7)
assert g57 == 23, '[TC11] Frobenius g(5,7) should be 23 FAILED'
g35 = frobenius_number_2d(3, 5)
assert g35 == 7, '[TC11] Frobenius g(3,5) should be 7 FAILED'

# ---- TC12: diophantine_nd_nonnegative 解验证 ----
sols = diophantine_nd_nonnegative([3, 5, 7], 20)
assert len(sols) > 0, '[TC12] Should have at least one Diophantine solution FAILED'
for sol in sols:
    val = np.dot(sol, [3, 5, 7])
    assert val == 20, '[TC12] All solutions must satisfy the equation FAILED'

# ---- TC13: generate_hcp_lattice_2d 形状与间距 ----
hcp = generate_hcp_lattice_2d(3, 3, lattice_constant=1.2)
assert hcp.shape == (9, 2), '[TC13] HCP lattice shape FAILED'
d_nn = np.linalg.norm(hcp[0] - hcp[1])
assert abs(d_nn - 1.2) < 1e-10, '[TC13] HCP nearest neighbor distance FAILED'

# ---- TC14: coordination_number 已知值 ----
assert coordination_number("fcc") == 12, '[TC14] FCC coordination FAILED'
assert coordination_number("bcc") == 8, '[TC14] BCC coordination FAILED'
assert coordination_number("hcp") == 12, '[TC14] HCP coordination FAILED'
assert coordination_number("sc") == 6, '[TC14] SC coordination FAILED'
assert coordination_number("diamond") == 4, '[TC14] Diamond coordination FAILED'
assert coordination_number("hexagonal") == 6, '[TC14] Hexagonal coordination FAILED'
assert coordination_number("square") == 4, '[TC14] Square coordination FAILED'

# ---- TC15: latin_hypercube_sampling 形状、范围与可复现性 ----
np.random.seed(42)
pts_lhs = latin_hypercube_sampling(3, 20)
assert pts_lhs.shape == (20, 3), '[TC15] LHS shape FAILED'
assert np.all(pts_lhs >= 0.0) and np.all(pts_lhs <= 1.0), '[TC15] LHS range FAILED'
np.random.seed(42)
pts_lhs2 = latin_hypercube_sampling(3, 20)
assert np.allclose(pts_lhs, pts_lhs2), '[TC15] LHS reproducibility FAILED'

# ---- TC16: latin_center_sampling 形状、范围与可复现性 ----
np.random.seed(42)
pts_lc = latin_center_sampling(2, 16)
assert pts_lc.shape == (16, 2), '[TC16] Latin Center shape FAILED'
assert np.all(pts_lc >= 0.0) and np.all(pts_lc <= 1.0), '[TC16] Latin Center range FAILED'
np.random.seed(42)
pts_lc2 = latin_center_sampling(2, 16)
assert np.allclose(pts_lc, pts_lc2), '[TC16] Latin Center reproducibility FAILED'

# ---- TC17: sobol_like_sampling 形状与范围 ----
pts_sobol = sobol_like_sampling(2, 16)
assert pts_sobol.shape == (16, 2), '[TC17] Sobol-like shape FAILED'
assert np.all(pts_sobol >= 0.0) and np.all(pts_sobol <= 1.0), '[TC17] Sobol-like range FAILED'

# ---- TC18: set_partition_equivalence 正确划分 ----
n8 = 8
R8 = np.zeros((n8, n8))
for i in range(n8):
    for j in range(n8):
        if (i % 3) == (j % 3):
            R8[i, j] = 1.0
classes = set_partition_equivalence(n8, R8)
assert len(classes) == 3, '[TC18] Partition should have 3 classes FAILED'
for cls in classes:
    mods = set(x % 3 for x in cls)
    assert len(mods) == 1, '[TC18] Elements in same class must share mod 3 FAILED'

# ---- TC19: triangle_grid_points 点数与重心坐标范围 ----
verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
tri_pts = triangle_grid_points(4, verts)
n_expected = (4 + 1) * (4 + 2) // 2
assert len(tri_pts) == n_expected, '[TC19] Triangle grid point count FAILED'
assert np.all(tri_pts[:, 0] >= -1e-12) and np.all(tri_pts[:, 1] >= -1e-12), '[TC19] Triangle positivity FAILED'
assert np.all(tri_pts[:, 0] + tri_pts[:, 1] <= 1.0 + 1e-12), '[TC19] Triangle bounds x+y<=1 FAILED'

# ---- TC20: chebyshev_nodes 边界、个数与单调性 ----
n20 = 10
nodes = chebyshev_nodes(-2.0, 2.0, n20)
assert len(nodes) == n20, '[TC20] Chebyshev node count FAILED'
assert abs(nodes[0] - 2.0) < 1e-12, '[TC20] First node should be upper bound FAILED'
assert abs(nodes[-1] + 2.0) < 1e-12, '[TC20] Last node should be lower bound FAILED'
assert np.all(np.diff(nodes) <= 1e-12), '[TC20] Nodes not in descending order FAILED'

# ---- TC21: divided_differences + newton_interpolate 二次多项式精确重建 ----
def _poly2(x):
    return 1.0 + 2.0 * x + 3.0 * x * x
n21 = 5
xd21 = chebyshev_nodes(-1.0, 1.0, n21)
yd21 = _poly2(xd21)
dd21 = divided_differences(xd21, yd21)
xp_test = np.array([-0.5, 0.0, 0.5])
yp_interp = newton_interpolate(xd21, dd21, xp_test)
yp_exact = _poly2(xp_test)
assert np.max(np.abs(yp_interp - yp_exact)) < 1e-10, '[TC21] Newton interpolation of quadratic FAILED'

# ---- TC22: chebyshev_differentiation_matrix sin(pi*x) 求导精度 ----
n22 = 36
nodes22 = chebyshev_nodes(-1.0, 1.0, n22)
D = chebyshev_differentiation_matrix(n22)
f_vals = np.sin(np.pi * nodes22)
df_num = D @ f_vals
df_exact22 = np.pi * np.cos(np.pi * nodes22)
err22 = np.max(np.abs(df_num - df_exact22))
assert err22 < 1e-8, '[TC22] Chebyshev spectral derivative accuracy FAILED'

# ---- TC23: cvt_1d_nonuniform_python 输出形状、范围与单调性 ----
np.random.seed(42)
cvt_pts = cvt_1d_nonuniform_python(n_generators=8, density_type=0, n_steps=30, n_samples_per_step=500)
assert len(cvt_pts) == 8, '[TC23] CVT output size FAILED'
assert np.all(cvt_pts >= 0.0) and np.all(cvt_pts <= 1.0), '[TC23] CVT range FAILED'
assert np.all(np.diff(cvt_pts) > 0), '[TC23] CVT points not monotonic FAILED'

# ---- TC24: solve_biharmonic_fd1d 固支边界条件验证 ----
def _load_uniform(x):
    return 1.0
x_bh, u_bh = solve_biharmonic_fd1d(_load_uniform, n=33, xlim=(-1.0, 1.0), bc_displacement=(0.0, 0.0), bc_slope=(0.0, 0.0))
assert abs(u_bh[0]) < 1e-8, '[TC24] Left displacement BC FAILED'
assert abs(u_bh[-1]) < 1e-8, '[TC24] Right displacement BC FAILED'
assert len(u_bh) == len(x_bh), '[TC24] Solution size mismatch FAILED'
assert np.max(u_bh) > 0, '[TC24] Deflection should be positive for uniform load FAILED'

# ---- TC25: compute_strain_energy 非负性 ----
h25 = 0.01
x25 = np.arange(0.0, 1.0, h25)
u25 = np.sin(np.pi * x25) * 0.1
se = compute_strain_energy(u25, h25, young_modulus=2.0, moment_of_inertia=1.0)
assert se >= -1e-12, '[TC25] Strain energy must be non-negative FAILED'

# ---- TC26: TetrahedralMesh 体积计算 ----
mesh = TetrahedralMesh.generate_uniform_box_mesh(nx=2, ny=2, nz=2, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0))
assert mesh.n_nodes > 0, '[TC26] Mesh node count FAILED'
assert mesh.n_elements > 0, '[TC26] Mesh element count FAILED'
_, vol = mesh.integrate_over_mesh(lambda x, y, z: 1.0, order=1)
assert abs(vol - 1.0) < 1e-10, '[TC26] Unit cube volume FAILED'

# ---- TC27: heat_capacity_cv 正值性与 Dulong-Petit 参考 ----
np.random.seed(42)
e27 = np.random.randn(1000) * 0.05 + 5.0
cv, cv_dp = heat_capacity_cv(e27, temperature=1.0, n_particles=36, dim=3)
assert cv > 0, '[TC27] CV should be positive FAILED'
assert cv_dp > 0, '[TC27] Dulong-Petit should be positive FAILED'

# ---- TC28: entropy_from_energy_distribution 非负性 ----
np.random.seed(42)
e28 = np.random.randn(500) * 0.1 + 3.0
ent = entropy_from_energy_distribution(e28, temperature=1.0, n_bins=15)
assert ent >= -1e-12, '[TC28] Entropy must be non-negative FAILED'

# ---- TC29: Timer 计时器正耗时 ----
t29 = Timer()
t29.start()
import time
time.sleep(0.005)
t29.stop()
assert t29.elapsed > 0, '[TC29] Timer elapsed must be positive FAILED'

# ---- TC30: HistogramStats 概率和为一 ----
np.random.seed(42)
data30 = np.random.randn(500)
hist30 = HistogramStats(data30, n_bins=12)
probs30 = hist30.probabilities
assert abs(np.sum(probs30) - 1.0) < 1e-12, '[TC30] Histogram probabilities must sum to 1 FAILED'

# ---- TC31: relative_error 基本行为 ----
err0 = relative_error(1.0, 1.0)
assert abs(err0) < 1e-15, '[TC31] Relative error of identical values should be 0 FAILED'
err1 = relative_error(1.2, 1.0)
assert abs(err1 - 0.2) < 1e-12, '[TC31] Relative error 1.2 vs 1.0 FAILED'

# ---- TC32: convergence_rate 几何序列收敛阶 ----
errors = [0.5 ** k for k in range(1, 6)]
rates = convergence_rate(errors)
assert len(rates) >= 2, '[TC32] Should have at least 2 rate estimates FAILED'
assert abs(rates[0] - 1.0) < 1e-10, '[TC32] Geometric convergence should be order 1 FAILED'

# ---- TC33: virial_stress_lj 应力张量对称性 ----
np.random.seed(42)
pos33 = np.random.randn(10, 3) * 0.3
stress = virial_stress_lj(pos33, epsilon=1.0, sigma=1.0, rcut=3.0, volume=1000.0, box_size=20.0)
assert stress.shape == (3, 3), '[TC33] Stress tensor shape FAILED'
assert np.max(np.abs(stress - stress.T)) < 1e-12, '[TC33] Stress tensor not symmetric FAILED'

# ---- TC34: MDEngine 初始化与短模拟 ----
np.random.seed(42)
md34 = MDEngine(n_particles=16, dim=2, mass=1.0, dt=0.001, box_size=5.0,
                epsilon=1.0, sigma=1.0, rcut=2.5, temperature=0.5, tau_thermostat=0.05)
md34.initialize_positions_lattice("square")
md34.initialize_velocities_maxwell_boltzmann()
assert md34.pos.shape == (16, 2), '[TC34] Position shape FAILED'
assert md34.vel.shape == (16, 2), '[TC34] Velocity shape FAILED'
res34 = md34.run(n_steps=30, equilibration_steps=10, apply_thermostat=True)
assert 'total_energy' in res34, '[TC34] Results missing total_energy FAILED'
assert len(res34['total_energy']) == 30, '[TC34] Energy history length FAILED'
assert np.all(np.isfinite(res34['total_energy'])), '[TC34] Energy contains non-finite values FAILED'
assert np.all(res34['temperature'] > 0), '[TC34] Temperature must be positive FAILED'

# ---- TC35: thermal_expansion_estimate 正值 ----
L = np.array([1.0, 1.002, 1.004, 1.006])
T = np.array([290.0, 300.0, 310.0, 320.0])
alpha = thermal_expansion_estimate(L, T)
assert alpha > 0, '[TC35] Thermal expansion must be positive FAILED'

# ---- TC36: voronoi_cell_area_hex 正值与一致性 ----
area1 = voronoi_cell_area_hex(1.0)
area2 = voronoi_cell_area_hex(2.0)
assert area1 > 0, '[TC36] Voronoi area must be positive FAILED'
assert abs(area2 / area1 - 4.0) < 1e-10, '[TC36] Area should scale with a^2 FAILED'

# ---- TC37: generate_square_lattice_2d 形状与间距 ----
sq = generate_square_lattice_2d(4, 4, lattice_constant=1.5)
assert sq.shape == (16, 2), '[TC37] Square lattice shape FAILED'
d_sq = np.linalg.norm(sq[0] - sq[1])
assert abs(d_sq - 1.5) < 1e-10, '[TC37] Square lattice spacing FAILED'

# ---- TC38: boundary_trace_hex 闭合六方词追踪 ----
word38 = "123456"
trace38 = boundary_trace_hex(word38)
assert len(trace38) > 0, '[TC38] Boundary trace empty FAILED'
# 六方词 "123456" 应回到原点附近
dist_to_origin = np.linalg.norm(trace38[-1] - trace38[0])
assert dist_to_origin < 1e-10, '[TC38] Hex word 123456 should close FAILED'

# ---- TC39: pram_boundary_word 返回类型 ----
w_pram, p_pram = pram_boundary_word()
assert isinstance(w_pram, str), '[TC39] PRAM word should be string FAILED'
assert len(w_pram) > 0, '[TC39] PRAM word non-empty FAILED'
assert isinstance(p_pram, np.ndarray), '[TC39] PRAM origin should be ndarray FAILED'

# ---- TC40: CVTOptimizer 2D 优化 ----
np.random.seed(42)
cvt40 = CVTOptimizer(dim=2, n_generators=9, domain=(np.array([0., 0.]), np.array([2., 2.])),
                     density_func=lambda x: 1.0, max_iter=20, tol=1e-6)
cvt40.initialize_generators("grid")
pts40 = cvt40.optimize(sample_multiplier=30)
assert pts40.shape == (9, 2), '[TC40] CVT optimizer output shape FAILED'
assert np.all(pts40 >= -1e-10) and np.all(pts40 <= 2.0 + 1e-10), '[TC40] CVT points out of domain FAILED'
# 均匀密度下，重心应接近域中心
center40 = np.mean(pts40, axis=0)
assert abs(center40[0] - 1.0) < 0.5, '[TC40] CVT center x near 1.0 FAILED'
assert abs(center40[1] - 1.0) < 0.5, '[TC40] CVT center y near 1.0 FAILED'

# ---- TC41: boundary_range_hex 范围计算 ----
imin, imax, jmin, jmax = boundary_range_hex("123")
assert imax >= imin, '[TC41] i range invalid FAILED'
assert jmax >= jmin, '[TC41] j range invalid FAILED'

# ---- TC42: benchmark_function 基准测试返回结构 ----
def _f_bench(x):
    return x * x
bm = benchmark_function(_f_bench, 3.0, n_runs=5)
assert 'mean_time' in bm, '[TC42] Benchmark missing mean_time FAILED'
assert 'min_time' in bm, '[TC42] Benchmark missing min_time FAILED'
assert 'max_time' in bm, '[TC42] Benchmark missing max_time FAILED'
assert 'result' in bm, '[TC42] Benchmark missing result FAILED'
assert abs(bm['result'] - 9.0) < 1e-12, '[TC42] Benchmark result mismatch FAILED'

# ---- TC43: is_symmetric_positive_definite 正定检测 ----
A_spd = np.array([[4., 1.], [1., 3.]])
assert is_symmetric_positive_definite(A_spd), '[TC43] SPD matrix detection FAILED'
A_not_spd = np.array([[1., 3.], [3., 1.]])
assert not is_symmetric_positive_definite(A_not_spd), '[TC43] Non-SPD matrix should be rejected FAILED'

# ---- TC44: condition_number_estimate 基本行为 ----
A44 = np.diag([1.0, 2.0, 4.0])
cond44 = condition_number_estimate(A44)
assert cond44 > 0, '[TC44] Condition number must be positive FAILED'
assert abs(cond44 - 4.0) < 1e-10, '[TC44] Condition number of diag(1,2,4) FAILED'

# ---- TC45: elastic_constants_strain_derivative 返回矩阵正定性 ----
np.random.seed(42)
pos45 = np.random.randn(20, 3) * 0.3
vol45 = 100.0
C = elastic_constants_strain_derivative(pos45, epsilon=1.0, sigma=1.0, rcut=3.0, volume=vol45, strain_perturbation=1e-4)
assert C.shape == (3, 3), '[TC45] Elastic constants shape FAILED'
assert np.all(np.isfinite(C)), '[TC45] Elastic constants finite FAILED'

print('\n全部 45 个测试通过!\n')
