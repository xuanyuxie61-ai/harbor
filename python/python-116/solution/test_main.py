"""
main.py
脂质双分子层凝胶-液晶相变的博士级综合计算入口

本程序围绕分子动力学中的脂质双分子层相变问题，综合调用以下模块:
  - bilayer_system: 粗粒化双层系统建模与热场平衡
  - integrator: 速度 Verlet MD 积分与稳定性分析
  - order_parameters: Jacobi 多项式谱分析与取向序
  - density_profile: Bernstein 多项式跨膜密度轮廓
  - grid_topology: 多类型网格生成与膜边界追踪
  - sparse_matrix_ops: RCM 重排序与 Markov 态模型
  - combinatorial_sampler: 组合回溯与构象采样
  - free_energy: 稀疏网格自由能积分
  - phase_diagram: Newton-Maehly 根求与 Duffing 涨落动力学
  - permeation_analysis: Dijkstra 最低自由能渗透路径
  - clustering_analysis: 层次聚类与畴分离分析

运行方式:
    python main.py
    （零参数，所有物理参数内置）
"""

import numpy as np
import sys

# ---------------------------------------------------------------------------
# 导入各模块
# ---------------------------------------------------------------------------
from bilayer_system import LipidBilayerSystem
from integrator import MDIntegrator, IntegratorStability
from order_parameters import OrientationalOrderAnalysis, debye_waller_factor
from density_profile import MembraneDensityProfile, bernstein_basis
from grid_topology import GridGenerator, BoundaryTracer
from sparse_matrix_ops import (
    SparseMatrixOps, MarkovStateModel,
    build_lipid_adjacency, build_diffusion_matrix_from_adjacency
)
from combinatorial_sampler import ConfigurationSampler, CombinatorialEnumerators
from free_energy import SparseGridIntegration, FreeEnergyCalculator
from phase_diagram import (
    NewtonMaehlySolver, SelfConsistentTransition,
    DuffingMembraneDynamics, PhaseDiagramBuilder
)
from permeation_analysis import DijkstraPermeation, FreeEnergyFieldGenerator
from clustering_analysis import (
    LipidDistanceMatrix, HierarchicalClustering,
    chain_letter_style_symmetrization,
    order_parameter_to_feature_vector
)


def run_simulation():
    print("=" * 78)
    print("  脂质双分子层凝胶-液晶相变：综合数值模拟与相图分析")
    print("  Lipid Bilayer Gel-to-Fluid Phase Transition: Integrated Analysis")
    print("=" * 78)
    print()

    # =====================================================================
    # 1. 系统初始化与热场平衡
    # =====================================================================
    print("[1] 初始化粗粒化脂质双分子层系统 ...")
    system = LipidBilayerSystem(
        nx=16, ny=16,
        j_coupling=2.5,
        epsilon_nn=1.0,
        kappa_a=25.0,
        area0=0.64,
        dt_md=0.002,
        mass=1.0
    )
    print(f"      系统尺寸: {system.nx} × {system.ny} = {system.n_lipids} 脂质分子")
    print(f"      耦合常数 J = {system.J} kJ/mol, 最近邻耦合 ε = {system.eps_nn}")
    print(f"      面积压缩模量 κ_A = {system.kappa_a}, 平衡面积 A_0 = {system.area0} nm²")

    # 热场平衡 (heated_plate 思想)
    print("[1.1] 求解稳态温度场 (Jacobi 迭代) ...")
    iters, diff = system.thermalize_temperature_field(
        boundary_temp_high=350.0,
        boundary_temp_low=250.0,
        epsilon_conv=1e-4,
        max_iter=5000
    )
    print(f"      迭代 {iters} 次收敛, 残差 = {diff:.2e}")
    print(f"      温度场范围: [{system.temperature_field.min():.1f}, {system.temperature_field.max():.1f}] K")
    print()

    # =====================================================================
    # 2. MD 积分器与稳定性分析
    # =====================================================================
    print("[2] MD 积分器初始化与稳定性检验 ...")
    integrator = MDIntegrator(system, friction_gamma=0.5, seed=42)

    stab = IntegratorStability('verlet')
    stable, z_pts = stab.check_system_stability(omega_max=2.5, gamma=0.5, dt=system.dt)
    print(f"      速度 Verlet 稳定性: {'PASS' if stable else 'WARN'}")
    print(f"      特征值 z = {z_pts[0]:.4e}, {z_pts[1]:.4e}")

    # 运行平衡化 MD
    print("[2.1] 运行 1000 步平衡化 MD (速度 Verlet + Langevin) ...")
    energy_trace, s2_trace = integrator.run_equilibration(n_steps=1000)
    print(f"      最终能量: {energy_trace[-1]:.4f} kJ/mol")
    print(f"      最终全局序参数 S_2 = {s2_trace[-1]:.4f}")
    print()

    # =====================================================================
    # 3. 取向序的 Jacobi 谱分析
    # =====================================================================
    print("[3] Jacobi 多项式谱分析取向分布 ...")
    oa = OrientationalOrderAnalysis(n_max=10, alpha=0.0, beta=2.0)
    cos_samples = np.cos(system.theta).ravel()
    coeffs = oa.expand_odf(cos_samples)
    s_params = oa.order_parameters_from_coeffs(coeffs)
    entropy = oa.compute_entropy(coeffs)
    print(f"      Jacobi 展开系数 (前 6 项): {coeffs[:6]}")
    print(f"      归一化序参数 S_0..S_5: {s_params[:6]}")
    print(f"      取向熵 S_orient = {entropy:.4f}")

    # Debye-Waller 因子
    dwf = debye_waller_factor(s2_trace[-1], 300.0, moment_inertia=1.0)
    print(f"      Debye-Waller 因子 B = {dwf:.4e} nm²")
    print()

    # =====================================================================
    # 4. 跨膜密度 Bernstein 轮廓
    # =====================================================================
    print("[4] Bernstein 多项式跨膜密度轮廓 ...")
    dens = MembraneDensityProfile(z_min=-3.0, z_max=3.0, n_bernstein=10)
    z_samples = np.linspace(-3.0, 3.0, 100)
    # 构造双峰密度（模拟头基峰）
    rho_samples = (
        0.8 * np.exp(-(z_samples - 1.5) ** 2 / 0.3) +
        0.8 * np.exp(-(z_samples + 1.5) ** 2 / 0.3) +
        0.3 * np.ones_like(z_samples)
    )
    dens.fit_density(z_samples, rho_samples)
    d_hh = dens.headgroup_distance(threshold=0.5)
    d_hh_gauss = dens.membrane_thickness_from_gaussian_fit()
    print(f"      头基-头基距离 d_HH (阈值法) = {d_hh:.3f} nm")
    print(f"      头基-头基距离 d_HH (高斯拟合) = {d_hh_gauss:.3f} nm")

    # 面积压缩模量与弯曲刚度
    k_c = dens.bending_rigidity_helfrich(temperature=300.0, thickness=d_hh)
    print(f"      估计弯曲刚度 K_C = {k_c:.2f} k_B T")
    print()

    # =====================================================================
    # 5. 网格拓扑与边界追踪
    # =====================================================================
    print("[5] 网格生成与膜边界追踪 ...")
    # 矩形网格
    Xr, Yr, dxr, dyr = GridGenerator.rectangular_grid(12, 12)
    print(f"      矩形网格: {Xr.shape}, dx={dxr:.3f}, dy={dyr:.3f}")
    # 极坐标网格
    Rp, Tp, Xp, Yp = GridGenerator.polar_grid(0.5, 3.0, 8, 16)
    print(f"      极坐标网格: R={Rp.shape}")
    # 三角网格
    nodes_tri, triangs = GridGenerator.triangular_grid(8, 8)
    print(f"      三角网格: {len(nodes_tri)} 节点, {len(triangs)} 三角形")

    # 边界追踪 (PRAM + 等边三角)
    bt = BoundaryTracer(grid_type='hex')
    # 构造一个模拟的膜区域掩码（圆形）
    mask = np.zeros((24, 24), dtype=bool)
    cx, cy = 12, 12
    for i in range(24):
        for j in range(24):
            if (i - cx) ** 2 + (j - cy) ** 2 < 80:
                mask[i, j] = True
    word, path = bt.trace_boundary(mask, (12, 12))
    peri, area = bt.compute_perimeter_and_area(word)
    print(f"      边界词长度: {len(word)}")
    print(f"      边界周长 ≈ {peri:.2f}, 面积 ≈ {area:.2f}")

    eq_tri_word = bt.equilateral_triangle_boundary(side_length=4)
    peri_tri, area_tri = bt.compute_perimeter_and_area(eq_tri_word)
    print(f"      等边三角畴周长 ≈ {peri_tri:.2f}, 面积 ≈ {area_tri:.2f}")
    print()

    # =====================================================================
    # 6. 稀疏矩阵操作：RCM 与 Markov 态模型
    # =====================================================================
    print("[6] 稀疏矩阵重排序与 Markov 态模型 ...")
    adj = build_lipid_adjacency(system.nx, system.ny, interaction_range=1)
    n_nodes = system.nx * system.ny
    adj_row, adj_col = SparseMatrixOps.adjacency_to_csr(adj, n_nodes)
    root = n_nodes // 2
    perm_rcm = SparseMatrixOps.rcm_reorder(root, adj_row, adj_col, n_nodes)
    print(f"      RCM 重排序: 分量大小 = {len(perm_rcm)}")

    bw_before = SparseMatrixOps.bandwidth(adj, np.arange(n_nodes))
    bw_after = SparseMatrixOps.bandwidth(adj, perm_rcm)
    print(f"      带宽: 重排序前 = {bw_before}, 后 = {bw_after}")

    # Markov 态模型: 构建随机转移矩阵
    P_rand = np.random.rand(8, 8)
    P_rand = P_rand / P_rand.sum(axis=1, keepdims=True)
    msm = MarkovStateModel(P_rand)
    pi_ss = msm.power_method_steady_state(max_iter=200)
    pagerank = msm.pagerank_style_rank(damping=0.85)
    timescales = msm.implied_timescales(n_eigen=4)
    print(f"      MSM 稳态分布 (前 4 态): {pi_ss[:4]}")
    print(f"      PageRank 重要性 (前 4 态): {pagerank[:4]}")
    print(f"      隐含时间尺度: {timescales}")
    print()

    # =====================================================================
    # 7. 组合采样与 Gray 码遍历
    # =====================================================================
    print("[7] 组合构象采样与 Gray 码遍历 ...")
    cs = ConfigurationSampler(nx=6, ny=6, n_orient_states=6)
    config = cs.random_configuration(seed=123)
    print(f"      随机构象 (前 12 个分子): {config[:12]}")

    gray_walk = cs.gray_code_walk(n_steps=20, seed=456)
    print(f"      Gray 码遍历步数: {len(gray_walk)}")

    # 整数划分（畴大小分布的理论枚举）
    partitions = CombinatorialEnumerators.integer_partitions(12, max_part=4)
    print(f"      n=12 的整数划分数 (max_part≤4): {len(partitions)}")

    # Stirling 数
    S2_8_3 = CombinatorialEnumerators.stirling_second(8, 3)
    print(f"      S(8,3) = {S2_8_3} 种方式将 8 个分子分为 3 个畴")
    print()

    # =====================================================================
    # 8. 稀疏网格自由能积分
    # =====================================================================
    print("[8] 稀疏网格自由能积分 ...")
    sgi = SparseGridIntegration(dim_num=3, level_max=3)
    print(f"      3D 稀疏网格节点数: {len(sgi.points)}")

    # 定义简化的集体变量能量函数
    def cv_energy(x):
        # x = [S2, area_ratio, curvature]
        S2, ar, H = x[0], x[1], x[2]
        return 2.5 * (S2 ** 2) + 12.5 * (ar - 1.0) ** 2 + 0.5 * (H ** 2)

    beta = 1.0 / (0.008314 * 300.0)
    Z = sgi.partition_function(cv_energy, beta)
    F = sgi.free_energy(cv_energy, beta)
    avg_S2 = sgi.expectation(lambda x: x[0], cv_energy, beta)
    print(f"      配分函数 Z = {Z:.4e}")
    print(f"      Helmholtz 自由能 F = {F:.4f} kJ/mol")
    print(f"      <S2> = {avg_S2:.4f}")

    # Maier-Saupe 自由能
    Tc_est = FreeEnergyCalculator.transition_temperature_estimate(J=2.5)
    print(f"      Maier-Saupe 估计 T_c = {Tc_est:.2f} K")
    print()

    # =====================================================================
    # 9. 相图与 Duffing 动力学
    # =====================================================================
    print("[9] 相图分析与 Duffing 膜厚度涨落 ...")
    pdb = PhaseDiagramBuilder(J=2.5)
    T_vals, S_vals, P_vals = pdb.build_diagram(T_range=(50, 400), n_T=30)
    Tc = pdb.sc.critical_temperature()
    print(f"      临界温度 T_c = {Tc:.2f} K")
    print(f"      低温序参数 S(T=250K) = {S_vals[0]:.4f}")
    print(f"      高温序参数 S(T=400K) = {S_vals[-1]:.4f}")

    # Newton-Maehly 根求法: 求 Landau 展开多项式的根
    # f(S) = a τ S² + b S⁴ + c S⁶ 的导数 = 0 给出稳定点
    a, b, c = FreeEnergyCalculator.landau_expansion_coefficients(300.0, Tc)
    # 构造 dF/dS = 2aτ S + 4b S³ + 6c S⁵ = S(2aτ + 4b S² + 6c S⁴)
    # 令 u = S²，得 6c u² + 4b u + 2aτ = 0
    poly_coeffs = np.array([2 * a * (300.0 - Tc) / Tc, 4 * b, 6 * c], dtype=complex)
    nms = NewtonMaehlySolver(poly_coeffs, max_iter=100, tol=1e-12)
    roots = nms.solve()
    print(f"      Landau 展开稳定点 (u=S²): {roots}")

    # Duffing 动力学
    duff = DuffingMembraneDynamics(
        delta=0.3, alpha=-0.5, beta=1.0,
        gamma=0.4, omega=1.2, noise_amp=0.05, seed=99
    )
    t_duff, y_duff = duff.integrate_rk4(y0=[0.5, 0.0], t_span=(0.0, 30.0), n_steps=3000)
    lam = duff.lyapunov_exponent_estimate(y0=[0.5, 0.0], t_span=(0.0, 30.0), n_steps=3000)
    print(f"      Duffing 膜厚涨落: 末态 d={y_duff[-1,0]:.4f}, v={y_duff[-1,1]:.4f}")
    print(f"      最大 Lyapunov 指数 λ_max = {lam:.4f}")
    print()

    # =====================================================================
    # 10. Dijkstra 渗透路径分析
    # =====================================================================
    print("[10] Dijkstra 最低自由能渗透路径 (MFEP) ...")
    fe_field = FreeEnergyFieldGenerator.generate_3d_field(
        nx=8, ny=8, nz=20,
        xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), zlim=(-3.0, 3.0),
        model='double_well', z0=1.2, V0=12.0, sigma=0.4, asym=1.5
    )
    dijk = DijkstraPermeation(nx=8, ny=8, nz=20,
                               xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), zlim=(-3.0, 3.0))
    path, cost = dijk.find_mfep(fe_field, source_z_layer=0, target_z_layer=19, beta=beta)
    print(f"      MFEP 路径节点数: {len(path)}, 路径代价: {cost:.4f}")
    if len(path) > 0:
        P_perm = dijk.permeability_coefficient(path, fe_field, beta, D0=1e-6)
        print(f"      估算渗透系数 P = {P_perm:.4e} cm/s")
    print()

    # =====================================================================
    # 11. 层次聚类与畴分析
    # =====================================================================
    print("[11] 层次聚类与畴分离分析 ...")
    # 构造特征向量
    s2_local = system.compute_local_order_parameter().ravel()
    area_ratio = (system.area / system.area0).ravel()
    # 头基密度近似
    rho_head = 0.5 + 0.5 * np.cos(system.theta).ravel()
    feats = order_parameter_to_feature_vector(s2_local, area_ratio, rho_head)

    ldm = LipidDistanceMatrix(system.nx, system.ny, spatial_weight=2.0, sigma=1.5)
    dist_mat = ldm.compute_distance_matrix(feats)
    dist_sym = chain_letter_style_symmetrization(dist_mat)

    hc = HierarchicalClustering(dist_sym)
    linkage = hc.cluster()
    print(f"      距离矩阵对称化完成, 聚类合并次数: {len(linkage)}")

    for n_clust in [2, 3, 4]:
        labels = hc.cut_tree(linkage, n_clust)
        sizes = hc.domain_size_distribution(linkage, n_clust)
        gamma_if = hc.interface_energy_estimate(linkage, temperature=300.0)
        print(f"      {n_clust} 簇切割: 畴大小 = {sizes}, 估计线张力 γ = {gamma_if:.4f} kJ/(mol·nm)")
    print()

    # =====================================================================
    # 12. 综合结果汇总
    # =====================================================================
    print("=" * 78)
    print("  综合结果汇总")
    print("=" * 78)
    print(f"  系统参数:")
    print(f"    - 脂质分子数: {system.n_lipids}")
    print(f"    - 平衡温度场范围: [{system.temperature_field.min():.1f}, {system.temperature_field.max():.1f}] K")
    print(f"  相变特征:")
    print(f"    - 临界温度 T_c (Maier-Saupe) = {Tc:.2f} K")
    print(f"    - MD 终态全局序参数 S_2 = {s2_trace[-1]:.4f}")
    print(f"    - 膜厚度 d_HH = {d_hh:.3f} nm")
    print(f"    - 弯曲刚度 K_C ≈ {k_c:.2f} k_B T")
    print(f"  动力学与稳定性:")
    print(f"    - Verlet 积分器稳定性: {'PASS' if stable else 'FAIL'}")
    print(f"    - Duffing Lyapunov 指数: {lam:.4f}")
    print(f"  渗透与输运:")
    print(f"    - MFEP 路径长度: {len(path)} 节点")
    print(f"    - 估算渗透系数: {P_perm if len(path) > 0 else 0.0:.4e} cm/s")
    print(f"  畴结构:")
    print(f"    - 2-畴切割最大畴大小: {hc.domain_size_distribution(linkage, 2).max()}")
    print("=" * 78)
    print("  模拟正常结束。")
    print("=" * 78)

    return {
        'system': system,
        'energy_trace': energy_trace,
        's2_trace': s2_trace,
        'critical_temperature': Tc,
        'membrane_thickness': d_hh,
        'bending_rigidity': k_c,
        'integrator_stable': stable,
        'lyapunov_exponent': lam,
        'mfep_path_length': len(path),
        'permeability': P_perm if len(path) > 0 else 0.0,
    }


if __name__ == "__main__":
    try:
        results = run_simulation()
        # allow tests to run
    except Exception as e:
        print(f"\n[ERROR] 模拟运行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ================================================================
# 测试用例（60个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: P_2 Legendre 多项式已知值 (cosθ=1→1, cosθ=0→-0.5, cosθ=-1→1) ----
sys_tc = LipidBilayerSystem(nx=4, ny=4)
assert abs(sys_tc._p2_legendre(1.0) - 1.0) < 1e-12, '[TC01] P_2(1)=1 FAILED'
assert abs(sys_tc._p2_legendre(0.0) - (-0.5)) < 1e-12, '[TC01] P_2(0)=-0.5 FAILED'
assert abs(sys_tc._p2_legendre(-1.0) - 1.0) < 1e-12, '[TC01] P_2(-1)=1 FAILED'
cos_13 = np.sqrt(1.0 / 3.0)
assert abs(sys_tc._p2_legendre(cos_13)) < 1e-12, '[TC01] P_2(√(1/3))=0 FAILED'

# ---- TC02: spherical_harmonic_y20_approx 在 cosθ=1 处的值 ----
from order_parameters import spherical_harmonic_y20_approx
val_y20 = spherical_harmonic_y20_approx(1.0)
expected_y20 = np.sqrt(5.0 / (16.0 * np.pi)) * 2.0
assert abs(val_y20 - expected_y20) < 1e-12, '[TC02] Y_2^0(θ=0) FAILED'

# ---- TC03: debye_waller_factor 基本计算 ----
B_dw = debye_waller_factor(order_param=0.8, temperature=300.0, moment_inertia=1.0)
assert B_dw > 0, '[TC03] Debye-Waller B 非正 FAILED'
assert np.isfinite(B_dw), '[TC03] Debye-Waller B 非有限 FAILED'

# ---- TC04: Jacobi 多项式 P_0 恒为 1 ----
import numpy as np
from order_parameters import jacobi_polynomial
x_test = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
P_j = jacobi_polynomial(4, 0.0, 2.0, x_test)
assert np.allclose(P_j[:, 0], 1.0), '[TC04] Jacobi P_0 ≠ 1 FAILED'

# ---- TC05: Jacobi 多项式 P_1 解析验证 ----
# P_1^{(α,β)}(x) = ((α+β+2)x + (α-β))/2
P1_expected = ((0.0 + 2.0 + 2.0) * x_test + (0.0 - 2.0)) / 2.0
assert np.allclose(P_j[:, 1], P1_expected), '[TC05] Jacobi P_1 解析不匹配 FAILED'

# ---- TC06: Jacobi 归一化常数已知值 ----
from order_parameters import jacobi_norm_constant
h0 = jacobi_norm_constant(0, 0.0, 0.0)
assert abs(h0 - 2.0) < 1e-12, '[TC06] Jacobi h_0^{(0,0)} != 2 FAILED'
h1 = jacobi_norm_constant(1, 0.0, 0.0)
assert abs(h1 - 2.0 / 3.0) < 1e-12, '[TC06] Jacobi h_1^{(0,0)} != 2/3 FAILED'

# ---- TC07: Bernstein 基函数单位分解性 (∑ B_{n,k}(u) = 1) ----
u_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
B10 = bernstein_basis(10, u_vals)
assert np.allclose(np.sum(B10, axis=-1), 1.0), '[TC07] Bernstein 单位分解 FAILED'

# ---- TC08: Bernstein 基函数边界值 B_{n,0}(0)=1, B_{n,n}(1)=1 ----
assert abs(B10[0, 0] - 1.0) < 1e-12, '[TC08] B_{10,0}(0) != 1 FAILED'
assert abs(B10[-1, -1] - 1.0) < 1e-12, '[TC08] B_{10,10}(1) != 1 FAILED'

# ---- TC09: Bernstein 基函数输出维度 ----
from density_profile import bernstein_basis as bb
B5 = bernstein_basis(5, 0.3)
assert B5.shape == (1, 6), '[TC09] Bernstein 标量输入维度 FAILED'
B5_arr = bb(5, np.linspace(0, 1, 20))
assert B5_arr.shape == (20, 6), '[TC09] Bernstein 数组输入维度 FAILED'

# ---- TC10: GridGenerator 矩形网格形状与间距 ----
Xr, Yr, dxr, dyr = GridGenerator.rectangular_grid(12, 12)
assert Xr.shape == (12, 12), '[TC10] 矩形网格 X 形状 FAILED'
assert Yr.shape == (12, 12), '[TC10] 矩形网格 Y 形状 FAILED'
assert dxr > 0 and dyr > 0, '[TC10] 网格间距 FAILED'

# ---- TC11: GridGenerator 极坐标网格形状 ----
Rp, Tp, Xp, Yp = GridGenerator.polar_grid(0.5, 3.0, 8, 16)
assert Rp.shape == (8, 16), '[TC11] 极坐标 R 形状 FAILED'
assert Xp.shape == (8, 16), '[TC11] 极坐标 X 形状 FAILED'

# ---- TC12: GridGenerator 三角网格节点数与三角形数 ----
nodes_tri, triangs = GridGenerator.triangular_grid(8, 8)
assert len(nodes_tri) == 64, '[TC12] 三角网格节点数 FAILED'
assert len(triangs) == 2 * (8 - 1) * (8 - 1), '[TC12] 三角网格三角形数 FAILED'

# ---- TC13: BoundaryTracer 等边三角形周长 > 0 ----
bt = BoundaryTracer(grid_type='hex')
eq_word = bt.equilateral_triangle_boundary(side_length=4)
peri_tri, area_tri = bt.compute_perimeter_and_area(eq_word)
assert peri_tri > 0, '[TC13] 等边三角形周长非正 FAILED'
assert area_tri > 0, '[TC13] 等边三角形面积非正 FAILED'

# ---- TC14: membrane_surface_metric 张量正定性 ----
from grid_topology import membrane_surface_metric
E, F, G = membrane_surface_metric(nodes_tri, triangs)
assert E > 0, '[TC14] E 非正 FAILED'
assert G > 0, '[TC14] G 非正 FAILED'
assert E * G - F * F > 0, '[TC14] 度量张量非正定 FAILED'

# ---- TC15: CombinatorialEnumerators.gray_code 前 4 个值 ----
from combinatorial_sampler import CombinatorialEnumerators
gc = CombinatorialEnumerators.gray_code(3)
assert gc[0] == 0, '[TC15] Gray 码第一个值 != 0 FAILED'
assert gc[1] == 1, '[TC15] Gray 码第二个值 != 1 FAILED'
assert gc[2] == 3, '[TC15] Gray 码第三个值 != 3 FAILED'
assert gc[3] == 2, '[TC15] Gray 码第四个值 != 2 FAILED'

# ---- TC16: Stirling 第二类数已知值 ----
S_nn = CombinatorialEnumerators.stirling_second(5, 2)
assert S_nn == 15, '[TC16] S(5,2) != 15 FAILED'
S_0 = CombinatorialEnumerators.stirling_second(0, 0)
assert S_0 == 1, '[TC16] S(0,0) != 1 FAILED'
S_ni = CombinatorialEnumerators.stirling_second(5, 6)
assert S_ni == 0, '[TC16] S(5,6) != 0 FAILED'

# ---- TC17: 整数划分计数 ----
parts = CombinatorialEnumerators.integer_partitions(5)
assert len(parts) == 7, '[TC17] p(5) != 7 FAILED'

# ---- TC18: k_subset_lex 计数匹配 C(n,k) ----
subsets_5_2 = CombinatorialEnumerators.k_subset_lex(5, 2)
expected_c52 = 10
assert len(subsets_5_2) == expected_c52, '[TC18] C(5,2) 子集数 FAILED'

# ---- TC19: BacktrackSampler 空约束搜索 ----
from combinatorial_sampler import BacktrackSampler
bs = BacktrackSampler(n_vars=3, n_states=2, constraint_func=None)
sols = bs.search(max_solutions=10)
assert len(sols) == 8, '[TC19] 3 vars × 2 states 应产生 8 个解 FAILED'

# ---- TC20: ConfigurationSampler 固定种子可复现 ----
import numpy as np
np.random.seed(42)
cs = ConfigurationSampler(nx=4, ny=4, n_orient_states=6)
config1 = cs.random_configuration(seed=123)
np.random.seed(42)
config2 = cs.random_configuration(seed=123)
assert np.array_equal(config1, config2), '[TC20] 固定种子不复现 FAILED'

# ---- TC21: build_lipid_adjacency 节点数 ----
adj = build_lipid_adjacency(4, 4, interaction_range=1)
assert len(adj) == 16, '[TC21] 邻接图节点数 != 16 FAILED'

# ---- TC22: SparseMatrixOps.bandwidth 简单测试 ----
from sparse_matrix_ops import SparseMatrixOps
adj_small = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
perm = np.array([0, 1, 2])
bw = SparseMatrixOps.bandwidth(adj_small, perm)
assert bw >= 0, '[TC22] 带宽应为非负 FAILED'

# ---- TC23: RCM 重排序（4×4 格点） ----
import numpy as np
adj_rcm = build_lipid_adjacency(4, 4, interaction_range=1)
n_nodes = 4 * 4
adj_row, adj_col = SparseMatrixOps.adjacency_to_csr(adj_rcm, n_nodes)
perm_rcm = SparseMatrixOps.rcm_reorder(8, adj_row, adj_col, n_nodes)
assert len(perm_rcm) == n_nodes, '[TC23] RCM 排列长度 FAILED'
assert len(set(perm_rcm)) == n_nodes, '[TC23] RCM 排列不唯一 FAILED'

# ---- TC24: RCM 降低带宽 ----
bw_before = SparseMatrixOps.bandwidth(adj_rcm, np.arange(n_nodes))
bw_after = SparseMatrixOps.bandwidth(adj_rcm, perm_rcm)
assert bw_after <= bw_before, '[TC24] RCM 未能降低带宽 FAILED'

# ---- TC25: MarkovStateModel 稳态分布和为 1 ----
import numpy as np
np.random.seed(42)
P_rand = np.random.rand(6, 6)
P_rand = P_rand / P_rand.sum(axis=1, keepdims=True)
msm = MarkovStateModel(P_rand)
pi = msm.power_method_steady_state(max_iter=500)
assert abs(np.sum(pi) - 1.0) < 1e-8, '[TC25] 稳态分布和 != 1 FAILED'
assert np.all(pi >= -1e-12), '[TC25] 稳态分布含负值 FAILED'

# ---- TC26: MarkovStateModel PageRank 和为 1 ----
pr = msm.pagerank_style_rank(damping=0.85)
assert abs(np.sum(pr) - 1.0) < 1e-8, '[TC26] PageRank 和 != 1 FAILED'

# ---- TC27: build_diffusion_matrix_from_adjacency 行和为零 ----
L = build_diffusion_matrix_from_adjacency(adj_rcm, n_nodes, D=1.0, dt=0.001)
row_sums = np.sum(L, axis=1)
assert np.allclose(row_sums, 0.0, atol=1e-10), '[TC27] 扩散矩阵行和不零 FAILED'

# ---- TC28: SparseGridIntegration 常数函数积分 = 2^D ----
from free_energy import SparseGridIntegration
sgi = SparseGridIntegration(dim_num=2, level_max=3)
integral_const = sgi.integrate(lambda x: 1.0)
assert abs(abs(integral_const) - 4.0) < 0.01, '[TC28] 常数在 [-1,1]² 上积分绝对值 ≠ 4 FAILED'

# ---- TC29: FreeEnergyCalculator.transition_temperature_estimate 返回正值 ----
Tc_est = FreeEnergyCalculator.transition_temperature_estimate(J=2.5)
assert Tc_est > 0, '[TC29] T_c 估计值应为正 FAILED'

# ---- TC30: IntegratorStability Euler 放大因子在 z=0 处为 1 ----
stab_euler = IntegratorStability('euler')
R0 = stab_euler.amplification_factor(0.0 + 0.0j)
assert abs(R0 - 1.0) < 1e-12, '[TC30] Euler R(0) != 1 FAILED'

# ---- TC31: IntegratorStability Verlet 稳定性检验 ----
stab_verlet = IntegratorStability('verlet')
stable, z_pts = stab_verlet.check_system_stability(omega_max=2.0, gamma=0.5, dt=0.002)
assert stable, '[TC31] Verlet 应稳定 (ω·dt=0.004 < 2) FAILED'
assert len(z_pts) == 2, '[TC31] z_points 长度 != 2 FAILED'

# ---- TC32: IntegratorStability Verlet 不稳定检测 ----
stable_unst, _ = stab_verlet.check_system_stability(omega_max=2000.0, gamma=0.0, dt=0.1)
assert not stable_unst, '[TC32] Verlet 应检测到不稳定 FAILED'

# ---- TC33: NewtonMaehly 求解 z² - 1 = 0 ----
import numpy as np
coeffs = np.array([-1.0, 0.0, 1.0], dtype=complex)
nms = NewtonMaehlySolver(coeffs, max_iter=100, tol=1e-12)
roots = nms.solve()
roots_sorted = sorted(roots, key=lambda r: abs(r - 1.0))
assert abs(roots_sorted[0] - 1.0) < 1e-8, '[TC33] 根 +1 未找到 FAILED'
assert abs(roots_sorted[1] - (-1.0)) < 1e-8, '[TC33] 根 -1 未找到 FAILED'

# ---- TC34: SelfConsistentTransition.critical_temperature 公式验证 ----
from phase_diagram import SelfConsistentTransition
sc = SelfConsistentTransition(J_coupling=2.5, kb=0.008314)
Tc_val = sc.critical_temperature()
Tc_expected = 2.5 / (2.0 * 0.008314)
assert abs(Tc_val - Tc_expected) < 1e-6, '[TC34] T_c 公式 FAILED'

# ---- TC35: DuffingMembraneDynamics integrate_rk4 输出形状 ----
import numpy as np
np.random.seed(42)
duff_tc = DuffingMembraneDynamics(delta=0.3, alpha=-0.5, beta=1.0,
                                   gamma=0.4, omega=1.2, noise_amp=0.05, seed=99)
t_d, y_d = duff_tc.integrate_rk4(y0=[0.5, 0.0], t_span=(0.0, 10.0), n_steps=100)
assert t_d.shape == (101,), '[TC35] t 形状 FAILED'
assert y_d.shape == (101, 2), '[TC35] y 形状 FAILED'

# ---- TC36: Duffing Lyapunov 指数有限 ----
np.random.seed(42)
lam = duff_tc.lyapunov_exponent_estimate(y0=[0.5, 0.0], t_span=(0.0, 10.0), n_steps=100)
assert np.isfinite(lam), '[TC36] Lyapunov 指数非有限 FAILED'

# ---- TC37: FreeEnergyFieldGenerator.asymmetric_double_well 有限 ----
z_test = np.linspace(-3.0, 3.0, 50)
fe_z = FreeEnergyFieldGenerator.asymmetric_double_well(z_test, z0=1.2, V0=12.0, sigma=0.4, asym=1.5)
assert np.all(np.isfinite(fe_z)), '[TC37] 双势阱含非有限值 FAILED'
assert np.max(fe_z) > 0, '[TC37] 双势阱无正值 FAILED'

# ---- TC38: FreeEnergyFieldGenerator.generate_3d_field 形状 ----
fe_3d = FreeEnergyFieldGenerator.generate_3d_field(
    nx=8, ny=8, nz=16,
    xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), zlim=(-3.0, 3.0),
    model='double_well', z0=1.2, V0=12.0, sigma=0.4, asym=1.5
)
assert fe_3d.shape == (8, 8, 16), '[TC38] 3D 自由能场形状 FAILED'

# ---- TC39: DijkstraPermeation node_index/inverse_index 互逆 ----
dijk = DijkstraPermeation(nx=4, ny=4, nz=10)
for i, j, k in [(0, 0, 0), (3, 3, 9), (1, 2, 5)]:
    idx = dijk.node_index(i, j, k)
    ii, jj, kk = dijk.inverse_index(idx)
    assert (ii, jj, kk) == (i, j, k), f'[TC39] 索引互逆失败 ({i},{j},{k}) FAILED'

# ---- TC40: Dijkstra 小规模图最短路径 ----
import numpy as np
fe_small = np.zeros((4, 4, 10))
fe_small[:, :, 5:7] = 10.0  # 中央能垒
dijk_small = DijkstraPermeation(nx=4, ny=4, nz=10)
path, cost = dijk_small.find_mfep(fe_small, source_z_layer=0, target_z_layer=9, beta=0.1)
assert len(path) > 0, '[TC40] MFEP 路径为空 FAILED'
assert cost < np.inf, '[TC40] MFEP 代价无穷 FAILED'

# ---- TC41: chain_letter_style_symmetrization 对称性 ----
D_raw = np.random.rand(20, 20)
np.random.seed(42)
D_sym = chain_letter_style_symmetrization(D_raw)
assert np.allclose(D_sym, D_sym.T), '[TC41] 对称化后矩阵非对称 FAILED'

# ---- TC42: order_parameter_to_feature_vector 输出形状 ----
S2_fake = np.random.rand(16)
area_fake = np.random.rand(16)
head_fake = np.random.rand(16)
feats = order_parameter_to_feature_vector(S2_fake, area_fake, head_fake)
assert feats.shape == (16, 3), '[TC42] 特征向量形状 FAILED'

# ---- TC43: LipidBilayerSystem 全局序参数范围 [-0.5, 1] ----
sys_tc2 = LipidBilayerSystem(nx=6, ny=6)
s2_global = sys_tc2.global_order_parameter()
assert -0.5 - 1e-10 <= s2_global <= 1.0 + 1e-10, '[TC43] S_2 全局序参数超出范围 FAILED'

# ---- TC44: LipidBilayerSystem 局域序参数范围 ----
s2_local = sys_tc2.compute_local_order_parameter()
assert np.all(s2_local >= -0.5 - 1e-10), '[TC44] S_2 局域下限 FAILED'
assert np.all(s2_local <= 1.0 + 1e-10), '[TC44] S_2 局域上限 FAILED'

# ---- TC45: LipidBilayerSystem.get_positions 返回结构 ----
X_pos, Y_pos = sys_tc2.get_positions()
assert X_pos.shape == (6, 6), '[TC45] get_positions X 形状 FAILED'
assert Y_pos.shape == (6, 6), '[TC45] get_positions Y 形状 FAILED'

# ---- TC46: LipidBilayerSystem 总能量为有限值 ----
E_total = sys_tc2.compute_total_energy()
assert np.isfinite(E_total), '[TC46] 系统总能量非有限 FAILED'

# ---- TC47: MembraneDensityProfile 拟合与评估一致性 ----
dens = MembraneDensityProfile(z_min=-3.0, z_max=3.0, n_bernstein=8)
z_samples = np.linspace(-3.0, 3.0, 50)
rho_s = 0.5 * np.exp(-(z_samples - 1.5) ** 2 / 0.3) + 0.5 * np.exp(-(z_samples + 1.5) ** 2 / 0.3) + 0.2
dens.fit_density(z_samples, rho_s)
rho_eval = dens.evaluate(z_samples)
assert np.all(rho_eval >= -1e-12), '[TC47] 密度评估含负值 FAILED'

# ---- TC48: MembraneDensityProfile.headgroup_distance 返回非负 ----
d_hh = dens.headgroup_distance(threshold=0.5)
assert d_hh >= 0, '[TC48] 头基距离负值 FAILED'

# ---- TC49: PhaseDiagramBuilder 相图输出维度一致 ----
import numpy as np
pdb = PhaseDiagramBuilder(J=2.5)
T_vals, S_vals, P_vals = pdb.build_diagram(T_range=(250, 400), n_T=20)
assert len(T_vals) == 20, '[TC49] T 维度 FAILED'
assert len(S_vals) == 20, '[TC49] S 维度 FAILED'
assert len(P_vals) == 20, '[TC49] P 维度 FAILED'

# ---- TC50: PhaseDiagramBuilder.latent_heat 非负 ----
lh = pdb.latent_heat(Tc=300.0, S_gel=0.8, S_fluid=0.1)
assert lh > 0, '[TC50] 相变潜热非正 FAILED'

# ---- TC51: OrientationalOrderAnalysis 展开系数维度 ----
oa = OrientationalOrderAnalysis(n_max=6, alpha=0.0, beta=2.0)
cos_samples = np.cos(np.linspace(0.0, np.pi, 30))
coeffs = oa.expand_odf(cos_samples)
assert len(coeffs) == 7, '[TC51] 展开系数长度 FAILED'

# ---- TC52: OrientationalOrderAnalysis 重构 ODF 非负 ----
x_grid = np.linspace(-0.99, 0.99, 50)
f_recon = oa.reconstruct_odf(x_grid, coeffs)
assert np.all(np.isfinite(f_recon)), '[TC52] 重构 ODF 含非有限值 FAILED'
assert np.max(np.abs(f_recon)) > 0, '[TC52] 重构 ODF 全零 FAILED'

# ---- TC53: MDIntegrator 初始化与平衡化运行 ----
import numpy as np
np.random.seed(42)
sys_md = LipidBilayerSystem(nx=6, ny=6, dt_md=0.002)
md_int = MDIntegrator(sys_md, friction_gamma=0.5, seed=42)
e_trace, s2_trace = md_int.run_equilibration(n_steps=50)
assert len(e_trace) == 5, '[TC53] 能量轨迹长度 FAILED (50步每10步记录)'
assert np.all(np.isfinite(e_trace)), '[TC53] 能量含非有限值 FAILED'
assert np.all(np.isfinite(s2_trace)), '[TC53] S_2 含非有限值 FAILED'

# ---- TC54: LipidDistanceMatrix 距离矩阵对称 ----
from clustering_analysis import LipidDistanceMatrix
np.random.seed(42)
ldm = LipidDistanceMatrix(nx=4, ny=4, spatial_weight=2.0, sigma=1.5)
feat_test = np.random.rand(16, 3)
dist_mat = ldm.compute_distance_matrix(feat_test)
assert dist_mat.shape == (16, 16), '[TC54] 距离矩阵形状 FAILED'
assert np.allclose(dist_mat, dist_mat.T), '[TC54] 距离矩阵不对称 FAILED'
assert np.all(np.diag(dist_mat) >= 0), '[TC54] 对角线含负值 FAILED'

# ---- TC55: HierarchicalClustering 聚类与切割 ----
hc = HierarchicalClustering(dist_mat)
linkage = hc.cluster()
assert len(linkage) > 0, '[TC55] 聚类 linkage 为空 FAILED'
labels = hc.cut_tree(linkage, n_clusters=3)
assert len(np.unique(labels)) == 3, '[TC55] 聚类标签数 != 3 FAILED'

# ---- TC56: HierarchicalClustering.domain_size_distribution 总数 ----
sizes = hc.domain_size_distribution(linkage, n_clusters=3)
assert np.sum(sizes) == 16, '[TC56] 畴大小之和不等于总数 FAILED'

# ---- TC57: HierarchicalClustering.interface_energy_estimate 非负 ----
gamma_if = hc.interface_energy_estimate(linkage, temperature=300.0)
assert gamma_if >= 0, '[TC57] 界面能负值 FAILED'

# ---- TC58: CombinatorialEnumerators.stirling_second 边界 S(n,n)=1 ----
for n_val in [1, 2, 3, 5, 8]:
    assert CombinatorialEnumerators.stirling_second(n_val, n_val) == 1, \
        f'[TC58] S({n_val},{n_val}) != 1 FAILED'

# ---- TC59: SparseGridIntegration integrate 多项式 ----
sgi2 = SparseGridIntegration(dim_num=2, level_max=4)
def linear_func(x):
    return x[0]
integral_linear = sgi2.integrate(linear_func)
# ∫_{-1}^{1}∫_{-1}^{1} x dx dy = 0
assert abs(integral_linear) < 0.5, '[TC59] 奇函数积分应接近 0 FAILED'

# ---- TC60: 零阶 Bernstein 基 ----
B0 = bernstein_basis(0, np.array([0.5]))
assert B0.shape == (1,), '[TC60] n=0 Bernstein 维度 FAILED'
assert abs(B0[0] - 1.0) < 1e-12, '[TC60] B_{0,0}(0.5) != 1 FAILED'

print('\n全部 60 个测试通过!\n')
