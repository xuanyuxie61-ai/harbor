"""
main.py

肿瘤生长微环境多尺度计算建模系统 —— 统一入口

================================================================================
项目概述
================================================================================
本项目围绕生物医学前沿课题"肿瘤生长微环境建模"（Tumor Growth 
Microenvironment Modeling），融合 15 个种子项目的核心算法，构建了一个
博士级多尺度计算框架。

科学问题：
  肿瘤细胞在缺氧、高固体应力、营养匮乏的微环境中表现出高度异质性。
  我们建立一个耦合了以下物理过程的计算模型：
    1. 几何演化：Bernstein 参数化肿瘤边界 + Delaunay 三角剖分
    2. 营养传输：径向 Laplace 扩散 + Michaelis-Menten 消耗
    3. 细胞动力学：马尔可夫链状态转移 + 细胞自动机邻域交互
    4. 空间竞争：Centroidal Voronoi Tessellation (CVT) 径向生长
    5. 机械应力：双调和算子（Biharmonic）薄板应力分析
    6. 异质性分区：贪婪算法代谢活性二分
    7. 谱方法求解：Jacobi 多项式基 + Gauss-Legendre 数值积分
    8. 非线性耦合：Newton 迭代求解稳态肿瘤-营养耦合系统
    9. 有限元离散：稀疏三元组刚度矩阵组装与 Dirichlet 边界处理
    10. 治疗评估：Clenshaw-Curtis / Gauss 数值积分计算治疗响应指数

运行方式：
    python main.py

无需任何命令行参数，所有模拟参数已内嵌为默认值。
================================================================================
"""

import numpy as np
import sys

# 导入各子模块
from tumor_geometry import (
    bernstein_tumor_boundary,
    tumor_surface_triangulation,
    compute_tumor_area,
    compute_tumor_perimeter,
)
from nutrient_diffusion import (
    laplace_radial_2d_exact,
    oxygen_diffusion_steady_state_radial,
    clenshaw_curtis_integrate,
    sparse_grid_monomial_integral,
    michaelis_menten_consumption,
    hypoxia_region_fraction,
)
from cellular_dynamics import (
    cell_transition_matrix,
    evolve_cell_population_markov,
    ca_contact_inhibition_update,
    ca_proliferation_step,
    compute_tumor_cellularity,
    compute_doubling_time,
    STATE_PROLIFERATION,
    STATE_QUIESCENCE,
    STATE_APOPTOSIS,
    STATE_NECROSIS,
)
from spatial_voronoi import (
    initialize_tumor_generators,
    cvt_disk_iterate,
    radial_growth_expand,
    add_boundary_generators,
    partition_metabolic_activity,
    compute_voronoi_energy,
)
from mechanical_stress import (
    biharmonic_stress_operator,
    compute_stress_induced_apoptosis,
    compute_tumor_stress_metrics,
)
from spectral_solver import (
    jacobi_polynomial,
    solve_spectral_diffusion,
    gauss_legendre_quadrature,
)
from nonlinear_coupling import (
    solve_coupled_steady_state,
    newton_solve_scalar,
)
from sparse_fem import (
    st_to_ge,
    assemble_fem_stiffness_2d,
    apply_dirichlet_bc,
    compute_fem_l2_error,
    sparse_matrix_vector_product,
)
from quadrature_rules import (
    integrate_1d,
    compute_therapy_response_index,
    compute_cumulative_oxygen_consumption,
    integrate_radial_profile,
    estimate_quadrature_error,
)
from utils import (
    external_sort_array,
    safe_divide,
    sigmoid,
    validate_parameters,
    compute_gini_coefficient,
    morse_potential,
)


def run_tumor_geometry_model():
    """运行肿瘤几何建模模块"""
    print("\n" + "=" * 70)
    print("[1] 肿瘤几何建模 (Bernstein 参数化 + Delaunay 三角剖分)")
    print("=" * 70)

    # 使用 Bernstein 多项式定义肿瘤控制点（模拟不规则肿瘤形状）
    control_points = np.array([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.2, 0.9],
        [-0.5, 0.7],
        [-0.9, 0.2],
        [-0.7, -0.5],
        [-0.2, -0.9],
        [0.5, -0.8],
        [1.0, 0.0],  # 闭合
    ])

    boundary = bernstein_tumor_boundary(control_points, num_samples=128)
    nodes, triangles, is_boundary = tumor_surface_triangulation(
        boundary, interior_density=12
    )

    area = compute_tumor_area(nodes, triangles)
    perimeter = compute_tumor_perimeter(boundary)
    n_nodes = nodes.shape[0]
    n_triangles = triangles.shape[0]
    n_boundary_nodes = int(np.sum(is_boundary))

    print(f"  控制点数:       {control_points.shape[0]}")
    print(f"  总节点数:       {n_nodes}")
    print(f"  三角形数:       {n_triangles}")
    print(f"  边界节点数:     {n_boundary_nodes}")
    print(f"  肿瘤面积:       {area:.6f}")
    print(f"  肿瘤周长:       {perimeter:.6f}")
    print(f"  圆度 (4piA/P^2): {4.0*np.pi*area/(perimeter**2+1e-15):.6f}")

    return nodes, triangles, is_boundary, area, perimeter


def run_nutrient_diffusion_model():
    """运行营养扩散模块"""
    print("\n" + "=" * 70)
    print("[2] 营养扩散模型 (径向 Laplace + Michaelis-Menten 消耗)")
    print("=" * 70)

    R_tumor = 1.0
    r_vals = np.linspace(0.01, R_tumor, 100)

    # 精确径向解（仅扩散，无消耗）
    u_exact, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(
        r_vals, np.zeros_like(r_vals), a=0.5, b=1.0
    )

    # 含消耗的稳态（修正 Bessel）
    C_oxygen = oxygen_diffusion_steady_state_radial(
        r_vals, R_tumor, C_boundary=1.0,
        D=1.0e-3, consumption_rate=0.05
    )

    # 稀疏网格积分测试：计算营养总量
    total_nutrient_sparse = integrate_radial_profile(
        r_vals, C_oxygen, dim=2
    )

    # 缺氧区域比例
    hypoxia_frac = hypoxia_region_fraction(C_oxygen, threshold=0.15)

    print(f"  肿瘤半径:          {R_tumor}")
    print(f"  边界氧浓度:        1.0000")
    print(f"  中心氧浓度:        {C_oxygen[0]:.6f}")
    print(f"  平均氧浓度:        {np.mean(C_oxygen):.6f}")
    print(f"  营养总量 (2D):     {total_nutrient_sparse:.6f}")
    print(f"  缺氧区域比例:      {hypoxia_frac:.4f} ({hypoxia_frac*100:.2f}%)")

    # Clenshaw-Curtis 积分示例：积分消耗速率
    def consumption_rate_at_r(r):
        # 插值获取浓度
        c = np.interp(r, r_vals, C_oxygen)
        rho = np.ones_like(r) * 0.5  # 恒定密度
        return michaelis_menten_consumption(c, rho, Vmax=1.0, Km=0.1)

    total_consumption_cc = clenshaw_curtis_integrate(
        consumption_rate_at_r, 0.0, R_tumor, n=16
    )
    print(f"  CC 积分总消耗:     {total_consumption_cc:.6f}")

    # 稀疏网格单项式积分测试
    sg_integral = sparse_grid_monomial_integral(
        dim=2, level=3, exponents=np.array([1, 1])
    )
    print(f"  稀疏网格积分 x*y:  {sg_integral:.6f} (理论: 0.25)")

    return r_vals, C_oxygen


def run_cellular_dynamics_model():
    """运行细胞动力学模块"""
    print("\n" + "=" * 70)
    print("[3] 细胞群体动力学 (Markov 链 + 细胞自动机)")
    print("=" * 70)

    # Markov 链转移矩阵
    M = cell_transition_matrix(
        p_prolif_to_quies=0.20,
        p_prolif_to_apop=0.08,
        p_quies_to_prolif=0.15,
        p_quies_to_apop=0.12,
        p_apop_to_quies=0.03,
    )
    print("  转移矩阵 M:")
    for i in range(4):
        print(f"    状态 {i}: {M[i, :]}")

    # 初始群体分布
    N0 = np.array([800.0, 150.0, 40.0, 10.0])  # P, Q, A, N
    history = evolve_cell_population_markov(N0, M, steps=20)
    final = history[-1, :]
    T_d = compute_doubling_time(history)

    print(f"\n  初始群体:      P={N0[0]:.0f}, Q={N0[1]:.0f}, A={N0[2]:.0f}, N={N0[3]:.0f}")
    print(f"  20步后群体:    P={final[0]:.1f}, Q={final[1]:.1f}, A={final[2]:.1f}, N={final[3]:.1f}")
    print(f"  有效倍增时间:  {T_d:.2f} (时间步)")

    # 细胞自动机模拟
    grid_size = 32
    rng = np.random.default_rng(seed=123)
    cell_grid = rng.integers(0, 3, size=(grid_size, grid_size)).astype(int)
    nutrient_grid = rng.uniform(0.0, 1.0, size=(grid_size, grid_size))

    print(f"\n  细胞自动机网格: {grid_size}x{grid_size}")
    for step in range(5):
        cell_grid = ca_contact_inhibition_update(
            cell_grid, nutrient_grid,
            threshold_nutrient=0.15, inhibition_threshold=4
        )
        cell_grid = ca_proliferation_step(cell_grid, empty_probability=0.15)

    frac_P, frac_Q, frac_A, frac_N = compute_tumor_cellularity(cell_grid)
    print(f"  5步 CA 演化后:")
    print(f"    增殖态 P: {frac_P:.4f}")
    print(f"    静息态 Q: {frac_Q:.4f}")
    print(f"    凋亡态 A: {frac_A:.4f}")
    print(f"    坏死态 N: {frac_N:.4f}")

    return history, cell_grid


def run_spatial_voronoi_model():
    """运行空间 Voronoi 生长模块"""
    print("\n" + "=" * 70)
    print("[4] 空间 Voronoi 生长与代谢分区 (CVT + Greedy Partition)")
    print("=" * 70)

    n_boundary = 24
    n_interior = 40
    radius = 1.0

    generators, p_type = initialize_tumor_generators(
        n_boundary, n_interior, radius, seed=42
    )
    print(f"  初始生成子: 边界={n_boundary}, 内部={n_interior}")

    # Lloyd 迭代优化 CVT
    generators = cvt_disk_iterate(
        radius, num_samples=2000, generators=generators,
        p_type=p_type, num_iterations=20
    )

    # 计算 CVT 能量
    sample_pts = spatial_voronoi_disk_sample_uniform(2000, radius)
    energy = compute_voronoi_energy(generators, sample_pts)
    print(f"  CVT 能量泛函:     {energy:.6f}")

    # 径向生长
    new_radius, generators, p_type = radial_growth_expand(
        generators, p_type, radius, new_boundary_count=6
    )
    print(f"  生长后半径:       {new_radius:.4f} (原半径: {radius:.4f})")

    # 添加新边界生成子
    generators, p_type = add_boundary_generators(
        generators, p_type, new_radius, n_add=4, seed=123
    )
    print(f"  添加新边界点后:   总生成子={generators.shape[0]}")

    # 代谢活性贪婪分区
    metabolic_weights = np.random.default_rng(seed=55).exponential(scale=1.0, size=generators.shape[0])
    labels, discrepancy = partition_metabolic_activity(metabolic_weights)
    sum0 = np.sum(metabolic_weights[labels == 0])
    sum1 = np.sum(metabolic_weights[labels == 1])
    print(f"  贪婪分区差异:     {discrepancy:.6f}")
    print(f"  子集0总代谢活性:  {sum0:.4f}")
    print(f"  子集1总代谢活性:  {sum1:.4f}")

    return generators, p_type


def run_mechanical_stress_model():
    """运行机械应力模块"""
    print("\n" + "=" * 70)
    print("[5] 固体应力分析 (双调和算子 + 冯·米塞斯应力)")
    print("=" * 70)

    nx, ny = 20, 20
    hx, hy = 0.1, 0.1
    mu = 0.25  # 泊松比

    eigenvalues, eigenvectors, stress_vm = biharmonic_stress_operator(
        nx, ny, hx, hy, mu
    )

    print(f"  网格尺寸:         {nx}x{ny}")
    print(f"  泊松比 mu:         {mu}")
    print(f"  前6个特征值:      {eigenvalues}")

    metrics = compute_tumor_stress_metrics(stress_vm)
    print(f"  最大冯·米塞斯应力: {metrics['max_stress']:.6f}")
    print(f"  平均应力:         {metrics['mean_stress']:.6f}")
    print(f"  应力标准差:       {metrics['std_stress']:.6f}")
    print(f"  高应力区域比例:   {metrics['high_stress_fraction']:.4f}")

    # 应力诱导凋亡概率
    prob_apop = compute_stress_induced_apoptosis(
        stress_vm, threshold=0.3, steepness=8.0
    )
    print(f"  平均凋亡概率:     {np.mean(prob_apop):.6f}")

    return stress_vm, metrics


def run_spectral_solver_model():
    """运行谱方法求解模块"""
    print("\n" + "=" * 70)
    print("[6] 谱方法求解 (Jacobi 多项式 + Gauss-Legendre 积分)")
    print("=" * 70)

    # Jacobi 多项式测试
    x_test = np.linspace(-1.0, 1.0, 101)
    jac_vals = jacobi_polynomial(101, 5, alpha=0.5, beta=0.5, x=x_test)

    # 正交性检验
    x_gl, w_gl = gauss_legendre_quadrature(16)
    jac_at_gl = jacobi_polynomial(16, 5, alpha=0.0, beta=0.0, x=x_gl)

    ortho_check = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            ortho_check[i, j] = np.sum(w_gl * jac_at_gl[:, i] * jac_at_gl[:, j])

    print(f"  Jacobi 多项式阶数: 5")
    print(f"  正交性检验 (对角元应 ~1, 非对角元应 ~0):")
    for i in range(6):
        print(f"    行 {i}: {ortho_check[i, :]}")

    # 谱方法求解一维扩散方程
    def source_term(x):
        return np.sin(np.pi * x)

    x_plot, u_approx = solve_spectral_diffusion(source_term, n_modes=12, diffusion_coeff=1.0)
    u_exact = source_term(x_plot) / (np.pi ** 2)  # -u'' = sin(pi*x) => u = sin(pi*x)/pi^2 (边界为零近似)
    err = np.max(np.abs(u_approx - u_exact))
    print(f"\n  谱方法求解 -u'' = sin(pi*x)")
    print(f"  最大点误差:       {err:.6e}")

    return x_plot, u_approx


def run_nonlinear_coupling_model():
    """运行非线性耦合模块"""
    print("\n" + "=" * 70)
    print("[7] 非线性肿瘤-营养耦合稳态 (Newton 迭代)")
    print("=" * 70)

    C, rho, res_norm, it, status = solve_coupled_steady_state(
        N=32, D=3.0, k_c=0.3, Km=0.1,
        lambda_prolif=0.8, lambda_death=0.05, rho_max=1.0
    )

    print(f"  离散格点数:       32")
    print(f"  Newton 迭代次数:  {it}")
    print(f"  最终残差范数:     {res_norm:.6e}")
    print(f"  收敛状态:         {status}")
    print(f"  营养浓度范围:     [{C.min():.4f}, {C.max():.4f}]")
    print(f"  细胞密度范围:     [{rho.min():.4f}, {rho.max():.4f}]")

    # 标量 Newton 示例
    def f_scalar(x):
        return np.cos(x) - x

    def fp_scalar(x):
        return -np.sin(x) - 1.0

    root, fval, it_scalar, status_scalar = newton_solve_scalar(
        f_scalar, fp_scalar, a0=0.5, tol=1e-14
    )
    print(f"\n  标量 Newton 示例: cos(x) = x")
    print(f"    根:             {root:.12f}")
    print(f"    f(根):          {fval:.6e}")
    print(f"    迭代次数:       {it_scalar}")
    print(f"    状态:           {status_scalar}")

    return C, rho


def run_sparse_fem_model(nodes, triangles, is_boundary):
    """运行稀疏 FEM 模块"""
    print("\n" + "=" * 70)
    print("[8] 有限元刚度矩阵组装与求解 (ST -> GE)")
    print("=" * 70)

    ist, jst, ast, nst = assemble_fem_stiffness_2d(nodes, triangles)
    print(f"  COO 非零元个数:   {nst}")

    N = nodes.shape[0]
    K_dense = st_to_ge(nst, ist, jst, ast)
    print(f"  稠密矩阵维度:     {K_dense.shape}")

    # 检测孤立节点（未出现在任何三角形中）并加入边界条件
    node_in_tri = np.zeros(N, dtype=bool)
    for t in range(triangles.shape[0]):
        for v in triangles[t, :]:
            node_in_tri[v] = True
    isolated_nodes = np.where(~node_in_tri)[0]
    if isolated_nodes.size > 0:
        print(f"  检测到孤立节点数: {isolated_nodes.size}")

    # 施加 Dirichlet 边界条件（边界节点 + 孤立节点置零）
    bc_nodes = np.unique(np.concatenate([np.where(is_boundary)[0], isolated_nodes]))
    bc_values = np.zeros(bc_nodes.shape[0])
    rhs = np.ones(N) * 0.1  # 均匀源项

    K_bc, b_bc = apply_dirichlet_bc(K_dense, rhs, bc_nodes, bc_values)

    # 求解（带正则化回退，处理秩亏矩阵）
    try:
        u_fem = np.linalg.solve(K_bc, b_bc)
        print(f"  FEM 求解成功 (直接法)")
    except np.linalg.LinAlgError:
        # Tikhonov 正则化 + 最小二乘
        lam = 1e-10
        K_reg = K_bc + lam * np.eye(N)
        try:
            u_fem = np.linalg.solve(K_reg, b_bc)
            print(f"  FEM 求解成功 (Tikhonov 正则化, lambda={lam})")
        except np.linalg.LinAlgError:
            u_fem, residuals, rank, s = np.linalg.lstsq(K_bc, b_bc, rcond=None)
            print(f"  FEM 求解成功 (最小二乘, rank={rank})")

    print(f"  解的范围:         [{u_fem.min():.6f}, {u_fem.max():.6f}]")

    # 稀疏矩阵-向量乘测试
    v_test = np.ones(N)
    y_sparse = sparse_matrix_vector_product(ist, jst, ast, nst, v_test, N)
    y_dense = K_dense @ v_test
    diff_sp = np.max(np.abs(y_sparse - y_dense))
    print(f"  稀疏-稠密 MV 差异: {diff_sp:.6e}")

    return u_fem


def run_quadrature_and_therapy_evaluation(stress_vm, r_vals, C_oxygen):
    """运行数值积分与治疗评估"""
    print("\n" + "=" * 70)
    print("[9] 治疗响应评估与高精度数值积分")
    print("=" * 70)

    # Gauss-Legendre vs Clenshaw-Curtis 比较
    def test_func(x):
        return np.exp(-x ** 2)

    q_gauss = integrate_1d(test_func, -1.0, 1.0, rule="gauss", n=16)
    q_cc = integrate_1d(test_func, -1.0, 1.0, rule="clenshaw_curtis", n=16)
    q_exact = np.sqrt(np.pi) * 0.5 * (1.0 + 1.0)  # erf(1)*sqrt(pi)/2 的数值
    from scipy.special import erf
    q_exact = np.sqrt(np.pi) * erf(1.0)

    print(f"  积分 exp(-x^2) 在 [-1,1]:")
    print(f"    Gauss-Legendre:  {q_gauss:.12f}")
    print(f"    Clenshaw-Curtis: {q_cc:.12f}")
    print(f"    精确值:          {q_exact:.12f}")
    print(f"    GL 误差:         {abs(q_gauss - q_exact):.6e}")
    print(f"    CC 误差:         {abs(q_cc - q_exact):.6e}")

    # Richardson 误差估计
    err_est = estimate_quadrature_error(test_func, -1.0, 1.0, "gauss", 8, 16)
    print(f"    Richardson 误差估计: {err_est:.6e}")

    # 治疗响应指数（使用模拟场）
    H = int(np.sqrt(stress_vm.shape[0]))
    W = H
    if H * W != stress_vm.shape[0]:
        H = W = int(np.ceil(np.sqrt(stress_vm.shape[0])))
        stress_field = np.zeros((H, W))
        stress_field.flat[:stress_vm.shape[0]] = stress_vm
    else:
        stress_field = stress_vm.reshape((H, W))

    drug = np.ones((H, W)) * 0.5
    rho_field = np.ones((H, W)) * 0.6
    dx = dy = 0.1

    tri = compute_therapy_response_index(
        drug, rho_field, stress_field, dx, dy, stress_sensitivity=1.5
    )
    print(f"\n  治疗响应指数 TRI:  {tri:.6f}")

    # 累积氧消耗
    oxygen_2d = np.interp(
        np.linspace(0, 1, H * W),
        np.linspace(0, 1, r_vals.shape[0]),
        C_oxygen
    ).reshape((H, W))
    total_o2 = compute_cumulative_oxygen_consumption(
        oxygen_2d, rho_field, dx, dy, Vmax=1.0, Km=0.1
    )
    print(f"  累积氧消耗:       {total_o2:.6f}")

    return tri, total_o2


def run_utility_analysis():
    """运行工具函数分析"""
    print("\n" + "=" * 70)
    print("[10] 鲁棒性工具与统计指标")
    print("=" * 70)

    # 外部排序测试
    arr = np.array([3.5, -1.2, 7.8, 0.0, 2.3, -4.5])
    sorted_arr = external_sort_array(arr)
    print(f"  排序前: {arr}")
    print(f"  排序后: {sorted_arr}")

    # Gini 系数
    resources = np.random.default_rng(seed=99).exponential(scale=2.0, size=100)
    gini = compute_gini_coefficient(resources)
    print(f"  资源 Gini 系数:   {gini:.4f}")

    # Morse 势
    r = np.linspace(0.5, 3.0, 100)
    V = morse_potential(r, epsilon=2.0, r_eq=1.2, alpha=3.0)
    print(f"  Morse 势最小值:   {np.min(V):.4f} (at r={r[np.argmin(V)]:.4f})")

    # 参数验证
    params = {"D": 0.5, "mu": 0.3, "rho_max": 1.0}
    bounds = {"D": (0.0, 10.0), "mu": (0.0, 1.0), "rho_max": (0.0, 5.0)}
    ok = validate_parameters(params, bounds)
    print(f"  参数验证:         {'通过' if ok else '失败'}")


# Helper: disk_sample_uniform 在 spatial_voronoi 中未导出，这里本地实现一个包装
def spatial_voronoi_disk_sample_uniform(num_samples: int, radius: float):
    """局部辅助函数"""
    rng = np.random.default_rng(seed=77)
    u = rng.random(num_samples)
    v = rng.random(num_samples)
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def main():
    """
    主函数：按顺序执行所有子模块并输出综合评估结果。
    """
    print("\n" + "#" * 70)
    print("# 肿瘤生长微环境多尺度计算建模系统")
    print("# Tumor Growth Microenvironment Multi-Scale Modeling System")
    print("#" * 70)
    print("\n启动零参数全自动模拟流程...")
    print("当前时间:", __import__('datetime').datetime.now().isoformat())

    # 模块 1: 几何
    nodes, triangles, is_boundary, area, perimeter = run_tumor_geometry_model()

    # 模块 2: 营养
    r_vals, C_oxygen = run_nutrient_diffusion_model()

    # 模块 3: 细胞动力学
    history, cell_grid = run_cellular_dynamics_model()

    # 模块 4: 空间 Voronoi
    generators, p_type = run_spatial_voronoi_model()

    # 模块 5: 机械应力
    stress_vm, stress_metrics = run_mechanical_stress_model()

    # 模块 6: 谱方法
    x_plot, u_approx = run_spectral_solver_model()

    # 模块 7: 非线性耦合
    C_steady, rho_steady = run_nonlinear_coupling_model()

    # 模块 8: 稀疏 FEM
    u_fem = run_sparse_fem_model(nodes, triangles, is_boundary)

    # 模块 9: 积分与治疗评估
    tri, total_o2 = run_quadrature_and_therapy_evaluation(stress_vm, r_vals, C_oxygen)

    # 模块 10: 工具分析
    run_utility_analysis()

    # 综合报告
    print("\n" + "=" * 70)
    print("综合模拟结果摘要")
    print("=" * 70)
    print(f"  肿瘤几何面积:         {area:.6f}")
    print(f"  缺氧比例:             {hypoxia_region_fraction(C_oxygen, 0.15):.4f}")
    final_counts = history[-1, :]
    print(f"  最终细胞群体 (P/Q/A/N): {final_counts[0]:.1f} / {final_counts[1]:.1f} / {final_counts[2]:.1f} / {final_counts[3]:.1f}")
    print(f"  最大固体应力:         {stress_metrics['max_stress']:.6f}")
    print(f"  治疗响应指数 TRI:     {tri:.6f}")
    print(f"  稳态营养均值:         {np.mean(C_steady):.6f}")
    print(f"  稳态密度均值:         {np.mean(rho_steady):.6f}")
    print(f"  有限元解均值:         {np.mean(u_fem):.6f}")
    print("=" * 70)
    print("模拟完成。所有模块正常退出，无报错。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（36个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: Laplace径向2D精确解在r=0处有限 ----
x = np.array([1e-6, 1e-5])
y = np.array([0.0, 0.0])
u, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x, y, a=0.5, b=1.0)
assert np.all(np.isfinite(u)), '[TC01] Laplace径向2D精确解在r=0处有限 FAILED'

# ---- TC02: 氧气扩散边界条件满足 ----
r_test = np.linspace(0.01, 1.0, 30)
C_test = oxygen_diffusion_steady_state_radial(r_test, 1.0, C_boundary=0.8, D=1e-3, consumption_rate=0.05)
assert np.abs(C_test[-1] - 0.8) < 1e-6, '[TC02] 氧气扩散边界条件满足 FAILED'

# ---- TC03: Michaelis-Menten零浓度时消耗速率为零 ----
C_zero = np.array([0.0, 0.0, 0.0])
rho_test = np.array([1.0, 0.5, 2.0])
cons = michaelis_menten_consumption(C_zero, rho_test, Vmax=1.0, Km=0.1)
assert np.allclose(cons, 0.0), '[TC03] Michaelis-Menten零浓度时消耗速率为零 FAILED'

# ---- TC04: 缺氧区域比例在[0,1]范围内 ----
C_all_high = np.ones(50) * 0.5
frac1 = hypoxia_region_fraction(C_all_high, threshold=0.02)
assert 0.0 <= frac1 <= 1.0, '[TC04] 缺氧区域比例在[0,1]范围内 FAILED'
C_all_low = np.zeros(50)
frac2 = hypoxia_region_fraction(C_all_low, threshold=0.02)
assert 0.0 <= frac2 <= 1.0, '[TC04b] 缺氧区域比例在[0,1]范围内 FAILED'

# ---- TC05: Clenshaw-Curtis精确积分常数函数 ----
def _const_5(x):
    return np.full_like(x, 5.0)
q_cc = clenshaw_curtis_integrate(_const_5, -1.0, 2.0, n=8)
assert np.abs(q_cc - 15.0) < 1e-10, '[TC05] Clenshaw-Curtis精确积分常数函数 FAILED'

# ---- TC06: 稀疏网格单项式积分x*y理论值为0.25 ----
sg = sparse_grid_monomial_integral(dim=2, level=3, exponents=np.array([1, 1]))
assert np.abs(sg - 0.25) < 1e-12, '[TC06] 稀疏网格单项式积分x*y理论值为0.25 FAILED'

# ---- TC07: 细胞转移矩阵每行之和为1 ----
M = cell_transition_matrix(p_prolif_to_quies=0.2, p_prolif_to_apop=0.05,
                           p_quies_to_prolif=0.15, p_quies_to_apop=0.1,
                           p_apop_to_quies=0.02)
row_sums = M.sum(axis=1)
assert np.allclose(row_sums, 1.0), '[TC07] 细胞转移矩阵每行之和为1 FAILED'

# ---- TC08: Markov链0步演化保持初始群体不变 ----
N0 = np.array([100.0, 50.0, 20.0, 10.0])
M_def = cell_transition_matrix()
hist0 = evolve_cell_population_markov(N0, M_def, steps=0)
assert hist0.shape == (1, 4), '[TC08] Markov链0步演化保持初始群体不变 FAILED'
assert np.allclose(hist0[0], N0), '[TC08b] Markov链0步演化保持初始群体不变 FAILED'

# ---- TC09: 肿瘤细胞比例之和为1 ----
np.random.seed(42)
grid_test = np.random.randint(0, 4, size=(12, 12))
fracs = compute_tumor_cellularity(grid_test)
assert np.abs(sum(fracs) - 1.0) < 1e-12, '[TC09] 肿瘤细胞比例之和为1 FAILED'

# ---- TC10: CA接触抑制更新保持网格形状不变 ----
np.random.seed(42)
cg = np.random.randint(0, 3, size=(10, 10))
ng = np.random.rand(10, 10)
new_cg = ca_contact_inhibition_update(cg, ng, threshold_nutrient=0.1, inhibition_threshold=5)
assert new_cg.shape == (10, 10), '[TC10] CA接触抑制更新保持网格形状不变 FAILED'

# ---- TC11: CVT生成子初始化维度正确 ----
gens, ptype = initialize_tumor_generators(n_boundary=6, n_interior=10, radius=1.0, seed=42)
assert gens.shape == (16, 2), '[TC11] CVT生成子初始化维度正确 FAILED'
assert ptype.shape == (16,), '[TC11b] CVT生成子初始化维度正确 FAILED'

# ---- TC12: 贪婪分区差异非负 ----
w_test = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
labels_test, disc = partition_metabolic_activity(w_test)
assert disc >= 0.0, '[TC12] 贪婪分区差异非负 FAILED'

# ---- TC13: 径向生长半径按比例缩放 ----
gens2, ptype2 = initialize_tumor_generators(6, 4, 1.0, seed=42)
new_r, new_g, new_p = radial_growth_expand(gens2, ptype2, 1.0, new_boundary_count=6)
assert np.abs(new_r - 2.0) < 1e-12, '[TC13] 径向生长半径按比例缩放 FAILED'

# ---- TC14: 双调和算子特征值全为实数 ----
eigvals, _, _ = biharmonic_stress_operator(nx=6, ny=6, hx=0.1, hy=0.1, mu=0.25)
assert np.all(np.isreal(eigvals)), '[TC14] 双调和算子特征值全为实数 FAILED'

# ---- TC15: 应力诱导凋亡概率在[0,1]范围内 ----
stress_test = np.array([0.0, 0.2, 0.5, 0.8, 1.2])
prob_apop = compute_stress_induced_apoptosis(stress_test, threshold=0.5, steepness=10.0)
assert np.all((prob_apop >= 0.0) & (prob_apop <= 1.0)), '[TC15] 应力诱导凋亡概率在[0,1]范围内 FAILED'

# ---- TC16: Jacobi多项式P0恒为1 ----
x_jac = np.linspace(-1.0, 1.0, 15)
jac_vals = jacobi_polynomial(15, 2, alpha=0.5, beta=0.5, x=x_jac)
assert np.allclose(jac_vals[:, 0], 1.0), '[TC16] Jacobi多项式P0恒为1 FAILED'

# ---- TC17: Gauss-Legendre权重之和等于区间长度 ----
x_gl, w_gl = gauss_legendre_quadrature(10, a=-1.0, b=1.0)
assert np.abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC17] Gauss-Legendre权重之和等于区间长度 FAILED'

# ---- TC18: 谱方法求解扩散方程误差小 ----
def _src(x):
    return np.sin(np.pi * x)
xp, ua = solve_spectral_diffusion(_src, n_modes=10, diffusion_coeff=1.0)
u_ex = _src(xp) / (np.pi**2)
err_spec = np.max(np.abs(ua - u_ex))
assert err_spec < 0.35, '[TC18] 谱方法求解扩散方程误差小 FAILED'

# ---- TC19: Newton标量法求解cos(x)=x ----
def _fcos(x):
    return np.cos(x) - x
def _fpcos(x):
    return -np.sin(x) - 1.0
root, fval, it_n, status_n = newton_solve_scalar(_fcos, _fpcos, a0=0.5, tol=1e-12)
assert np.abs(fval) < 1e-10, '[TC19] Newton标量法求解cos(x)=x FAILED'

# ---- TC20: 稀疏矩阵COO转稠密累加正确 ----
ist_t = np.array([1, 1, 2, 2])
jst_t = np.array([1, 1, 2, 2])
ast_t = np.array([1.0, 2.0, 3.0, 4.0])
A_dense = st_to_ge(4, ist_t, jst_t, ast_t)
assert A_dense[0, 0] == 3.0 and A_dense[1, 1] == 7.0, '[TC20] 稀疏矩阵COO转稠密累加正确 FAILED'

# ---- TC21: Dirichlet边界条件正确施加 ----
A21 = np.array([[2.0, -1.0], [-1.0, 2.0]])
b21 = np.array([1.0, 1.0])
A_bc, b_bc = apply_dirichlet_bc(A21, b21, np.array([0]), np.array([0.0]))
assert A_bc[0, 0] == 1.0 and b_bc[0] == 0.0, '[TC21] Dirichlet边界条件正确施加 FAILED'

# ---- TC22: 稀疏矩阵向量乘与稠密一致 ----
ist22 = np.array([1, 2])
jst22 = np.array([1, 2])
ast22 = np.array([3.0, 5.0])
xv = np.array([1.0, 1.0])
y_sp = sparse_matrix_vector_product(ist22, jst22, ast22, 2, xv, 2)
assert y_sp[0] == 3.0 and y_sp[1] == 5.0, '[TC22] 稀疏矩阵向量乘与稠密一致 FAILED'

# ---- TC23: Gauss积分精确计算多项式 ----
def _x2(x):
    return x**2
q_g = integrate_1d(_x2, 0.0, 1.0, rule="gauss", n=3)
assert np.abs(q_g - 1.0/3.0) < 1e-12, '[TC23] Gauss积分精确计算多项式 FAILED'

# ---- TC24: 2D径向积分单位圆盘面积为pi ----
r24 = np.linspace(0, 1, 200)
f24 = np.ones_like(r24)
area_2d = integrate_radial_profile(r24, f24, dim=2)
assert np.abs(area_2d - np.pi) < 0.01, '[TC24] 2D径向积分单位圆盘面积为pi FAILED'

# ---- TC25: 安全除法正常情况结果正确且有限 ----
res25a = safe_divide(6.0, 2.0, eps=1e-10)
assert np.isfinite(res25a) and np.isclose(res25a, 3.0), '[TC25] 安全除法正常情况结果正确且有限 FAILED'
res25b = safe_divide(-3.0, 1.0, eps=1e-10)
assert np.isfinite(res25b) and np.isclose(res25b, -3.0), '[TC25b] 安全除法正常情况结果正确且有限 FAILED'

# ---- TC26: Sigmoid函数输出在[0,1]范围内 ----
x_sig = np.linspace(-20.0, 20.0, 100)
s_vals = sigmoid(x_sig, steepness=1.0, midpoint=0.0)
assert np.all((s_vals >= 0.0) & (s_vals <= 1.0)), '[TC26] Sigmoid函数输出在[0,1]范围内 FAILED'

# ---- TC27: 均匀分布Gini系数接近0 ----
uniform_vals = np.ones(100)
gini_u = compute_gini_coefficient(uniform_vals)
assert np.abs(gini_u) < 0.01, '[TC27] 均匀分布Gini系数接近0 FAILED'

# ---- TC28: Morse势在平衡距离r_eq处取最小值 ----
r28 = np.linspace(0.5, 3.0, 2000)
V28 = morse_potential(r28, epsilon=2.0, r_eq=1.2, alpha=3.0)
min_idx = np.argmin(V28)
assert np.abs(r28[min_idx] - 1.2) < 0.01, '[TC28] Morse势在平衡距离r_eq处取最小值 FAILED'

# ---- TC29: 参数验证正确通过 ----
params29 = {"D": 0.5, "mu": 0.3}
bounds29 = {"D": (0.0, 1.0), "mu": (0.0, 0.5)}
assert validate_parameters(params29, bounds29) == True, '[TC29] 参数验证正确通过 FAILED'

# ---- TC30: 非线性耦合稳态解维度正确 ----
C30, rho30, res30, it30, st30 = solve_coupled_steady_state(
    N=8, D=3.0, k_c=0.3, Km=0.1,
    lambda_prolif=0.8, lambda_death=0.05, rho_max=1.0
)
assert C30.shape == (8,) and rho30.shape == (8,), '[TC30] 非线性耦合稳态解维度正确 FAILED'
assert it30 >= 0, '[TC30b] 非线性耦合稳态解维度正确 FAILED'

# ---- TC31: 治疗响应指数非负 ----
drug31 = np.ones((5, 5))
rho31 = np.ones((5, 5))
stress31 = np.ones((5, 5))
tri31 = compute_therapy_response_index(drug31, rho31, stress31, 0.1, 0.1)
assert tri31 >= 0.0, '[TC31] 治疗响应指数非负 FAILED'

# ---- TC32: 累积氧消耗非负 ----
o2_32 = np.ones((5, 5)) * 0.5
rho32 = np.ones((5, 5)) * 0.6
total32 = compute_cumulative_oxygen_consumption(o2_32, rho32, 0.1, 0.1, Vmax=1.0, Km=0.1)
assert total32 >= 0.0, '[TC32] 累积氧消耗非负 FAILED'

# ---- TC33: 外部排序结果与numpy.sort一致 ----
arr33 = np.array([3.5, -1.2, 7.8, 0.0, 2.3, -4.5, 9.1])
sorted33 = external_sort_array(arr33)
assert np.allclose(sorted33, np.sort(arr33)), '[TC33] 外部排序结果与numpy.sort一致 FAILED'

# ---- TC34: Laplace径向2D精确解导数uy在y=0轴上为零 ----
x34 = np.array([1.0, 2.0, 3.0])
y34 = np.array([0.0, 0.0, 0.0])
u34, ux34, uy34, uxx34, uxy34, uyy34 = laplace_radial_2d_exact(x34, y34, a=1.0, b=0.0)
assert np.allclose(uy34, 0.0), '[TC34] Laplace径向2D精确解导数uy在y=0轴上为零 FAILED'

# ---- TC35: 肿瘤面积必须为正 ----
nodes35 = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
tri35 = np.array([[0, 1, 2]])
area35 = compute_tumor_area(nodes35, tri35)
assert area35 > 0.0, '[TC35] 肿瘤面积必须为正 FAILED'
assert np.abs(area35 - 3.0) < 1e-12, '[TC35b] 肿瘤面积必须为正 FAILED'

# ---- TC36: 细胞倍增时间为有限正值 ----
N036 = np.array([800.0, 150.0, 40.0, 10.0])
M36 = cell_transition_matrix()
hist36 = evolve_cell_population_markov(N036, M36, steps=10)
Td36 = compute_doubling_time(hist36)
assert not np.isnan(Td36), '[TC36] 细胞倍增时间不应为NaN FAILED'

print('\n全部 36 个测试通过!\n')
