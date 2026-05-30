
import numpy as np
import sys




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




    print("[2] MD 积分器初始化与稳定性检验 ...")
    integrator = MDIntegrator(system, friction_gamma=0.5, seed=42)

    stab = IntegratorStability('verlet')
    stable, z_pts = stab.check_system_stability(omega_max=2.5, gamma=0.5, dt=system.dt)
    print(f"      速度 Verlet 稳定性: {'PASS' if stable else 'WARN'}")
    print(f"      特征值 z = {z_pts[0]:.4e}, {z_pts[1]:.4e}")


    print("[2.1] 运行 1000 步平衡化 MD (速度 Verlet + Langevin) ...")
    energy_trace, s2_trace = integrator.run_equilibration(n_steps=1000)
    print(f"      最终能量: {energy_trace[-1]:.4f} kJ/mol")
    print(f"      最终全局序参数 S_2 = {s2_trace[-1]:.4f}")
    print()




    print("[3] Jacobi 多项式谱分析取向分布 ...")
    oa = OrientationalOrderAnalysis(n_max=10, alpha=0.0, beta=2.0)
    cos_samples = np.cos(system.theta).ravel()
    coeffs = oa.expand_odf(cos_samples)
    s_params = oa.order_parameters_from_coeffs(coeffs)
    entropy = oa.compute_entropy(coeffs)
    print(f"      Jacobi 展开系数 (前 6 项): {coeffs[:6]}")
    print(f"      归一化序参数 S_0..S_5: {s_params[:6]}")
    print(f"      取向熵 S_orient = {entropy:.4f}")


    dwf = debye_waller_factor(s2_trace[-1], 300.0, moment_inertia=1.0)
    print(f"      Debye-Waller 因子 B = {dwf:.4e} nm²")
    print()




    print("[4] Bernstein 多项式跨膜密度轮廓 ...")
    dens = MembraneDensityProfile(z_min=-3.0, z_max=3.0, n_bernstein=10)
    z_samples = np.linspace(-3.0, 3.0, 100)

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


    k_c = dens.bending_rigidity_helfrich(temperature=300.0, thickness=d_hh)
    print(f"      估计弯曲刚度 K_C = {k_c:.2f} k_B T")
    print()




    print("[5] 网格生成与膜边界追踪 ...")

    Xr, Yr, dxr, dyr = GridGenerator.rectangular_grid(12, 12)
    print(f"      矩形网格: {Xr.shape}, dx={dxr:.3f}, dy={dyr:.3f}")

    Rp, Tp, Xp, Yp = GridGenerator.polar_grid(0.5, 3.0, 8, 16)
    print(f"      极坐标网格: R={Rp.shape}")

    nodes_tri, triangs = GridGenerator.triangular_grid(8, 8)
    print(f"      三角网格: {len(nodes_tri)} 节点, {len(triangs)} 三角形")


    bt = BoundaryTracer(grid_type='hex')

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




    print("[7] 组合构象采样与 Gray 码遍历 ...")
    cs = ConfigurationSampler(nx=6, ny=6, n_orient_states=6)
    config = cs.random_configuration(seed=123)
    print(f"      随机构象 (前 12 个分子): {config[:12]}")

    gray_walk = cs.gray_code_walk(n_steps=20, seed=456)
    print(f"      Gray 码遍历步数: {len(gray_walk)}")


    partitions = CombinatorialEnumerators.integer_partitions(12, max_part=4)
    print(f"      n=12 的整数划分数 (max_part≤4): {len(partitions)}")


    S2_8_3 = CombinatorialEnumerators.stirling_second(8, 3)
    print(f"      S(8,3) = {S2_8_3} 种方式将 8 个分子分为 3 个畴")
    print()




    print("[8] 稀疏网格自由能积分 ...")
    sgi = SparseGridIntegration(dim_num=3, level_max=3)
    print(f"      3D 稀疏网格节点数: {len(sgi.points)}")


    def cv_energy(x):

        S2, ar, H = x[0], x[1], x[2]
        return 2.5 * (S2 ** 2) + 12.5 * (ar - 1.0) ** 2 + 0.5 * (H ** 2)

    beta = 1.0 / (0.008314 * 300.0)
    Z = sgi.partition_function(cv_energy, beta)
    F = sgi.free_energy(cv_energy, beta)
    avg_S2 = sgi.expectation(lambda x: x[0], cv_energy, beta)
    print(f"      配分函数 Z = {Z:.4e}")
    print(f"      Helmholtz 自由能 F = {F:.4f} kJ/mol")
    print(f"      <S2> = {avg_S2:.4f}")


    Tc_est = FreeEnergyCalculator.transition_temperature_estimate(J=2.5)
    print(f"      Maier-Saupe 估计 T_c = {Tc_est:.2f} K")
    print()




    print("[9] 相图分析与 Duffing 膜厚度涨落 ...")
    pdb = PhaseDiagramBuilder(J=2.5)
    T_vals, S_vals, P_vals = pdb.build_diagram(T_range=(50, 400), n_T=30)
    Tc = pdb.sc.critical_temperature()
    print(f"      临界温度 T_c = {Tc:.2f} K")
    print(f"      低温序参数 S(T=250K) = {S_vals[0]:.4f}")
    print(f"      高温序参数 S(T=400K) = {S_vals[-1]:.4f}")










    a, b, c = FreeEnergyCalculator.landau_expansion_coefficients(300.0, Tc)
    poly_coeffs = None
    nms = NewtonMaehlySolver(poly_coeffs, max_iter=100, tol=1e-12)
    roots = nms.solve()
    print(f"      Landau 展开稳定点 (u=S²): {roots}")


    duff = DuffingMembraneDynamics(
        delta=0.3, alpha=-0.5, beta=1.0,
        gamma=0.4, omega=1.2, noise_amp=0.05, seed=99
    )
    t_duff, y_duff = duff.integrate_rk4(y0=[0.5, 0.0], t_span=(0.0, 30.0), n_steps=3000)
    lam = duff.lyapunov_exponent_estimate(y0=[0.5, 0.0], t_span=(0.0, 30.0), n_steps=3000)
    print(f"      Duffing 膜厚涨落: 末态 d={y_duff[-1,0]:.4f}, v={y_duff[-1,1]:.4f}")
    print(f"      最大 Lyapunov 指数 λ_max = {lam:.4f}")
    print()




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




    print("[11] 层次聚类与畴分离分析 ...")

    s2_local = system.compute_local_order_parameter().ravel()
    area_ratio = (system.area / system.area0).ravel()

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
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 模拟运行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
