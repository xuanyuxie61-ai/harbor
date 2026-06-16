#!/usr/bin/env python3
"""
main.py
=======
惯性约束聚变 (ICF) 内爆对称性模拟
高阶有限差分与稳定性分析 — 统一入口

本项目融合 15 个种子项目的核心算法, 围绕 ICF 内爆物理展开:
  [1002] 弹性张量 → 靶丸材料力学性质
  [1215] ML 解码/SVM → 不稳定性分类代理模型
  [756] mesh_vtoe → 顶点-单元邻接
  [1310] triangle_io → 三角形网格 I/O
  [1324] wandzura_rule → 三角形求积
  [910] prime → 素数筛/低差异序列
  [1331] triangulation_boundary → 边界检测
  [972] r8but → 带状上三角矩阵
  [1297] FormalCellular → 配置空间验证/CDF
  [1413] welzl/icosahedron → 球面网格/包围球
  [919] product_rule → 乘积求积
  [405] fem2d_heat_sparse → 稀疏 FEM/后向 Euler
  [619] kepler_perturbed_ode → 辛积分/哈密顿守恒
  [042] asa144 → 随机列联表/扰动分布
  [982] r8ge_np → 稠密 LU 分解

运行: python main.py (零参数)
"""

import numpy as np
import sys
import os
import time

# 确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from icf_physics import (ICFPhysics, compute_sedov_solution,
                          DEFAULT_GAMMA, DEFAULT_CAPSULE_RADIUS,
                          DEFAULT_FUEL_DENSITY)
from mesh_generator import (generate_icf_mesh, MeshVtoe, BoundaryDetector,
                             IcosahedronSphereMesh, WelzlMinBoundingSphere,
                             TriangleIO)
from high_order_fd import (HighOrderFD, WENO5, CompactDifference,
                            CylindricalOperators, compute_fd_error_convergence)
from stability_analysis import (VonNeumannAnalysis, MatrixStabilityAnalysis,
                                 CFLCondition, analyze_icf_stability)
from quadrature_rules import (WandzuraRule, ProductRule, SphericalQuadrature)
from time_integrator import (ForwardEuler, SSPRK3, ClassicRK4, BackwardEuler,
                              StormerVerlet, integrate_ode)
from linear_solver import (BandedUpperTriangular, DenseNonPivotingLU,
                            SparseFEMAssembler, solve_linear_system)
from symmetry_decomposition import (LegendreDecomposition, RTGrowthModel,
                                     ImplosionSymmetryAnalyzer)
from perturbation_generator import (PrimeBasedSeeding, RandomContingencyPerturbation,
                                     PerturbationSpectrum, InitialPerturbationFactory)
from ml_surrogate import (SVMClassifier, FeatureScaler, CrossValidator,
                           ElasticTensorAnalyzer, ICFSurrogateModel)
from verification import (ManufacturedSolutionVerifier, ConservationChecker,
                           HamiltonianConservationChecker, ConfigurationSpaceAnalyzer,
                           run_full_verification)


def print_header():
    """打印项目标题"""
    print("=" * 70)
    print("  惯性约束聚变内爆对称性模拟")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("  Inertial Confinement Fusion Implosion Symmetry Simulation")
    print("  High-Order Finite Difference & Stability Analysis")
    print("=" * 70)


def section(title):
    """打印分节标题"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run_physics_setup():
    """
    第一阶段: ICF 物理参数设置与状态方程验证。
    融合 [1002] 弹性张量分析 → 靶丸材料力学性质。
    """
    section("阶段 1: ICF 物理参数设置")

    phys = ICFPhysics()
    print(f"  多方指数 γ = {phys.gamma:.4f}")
    print(f"  平均粒子质量 = {phys.mean_mass:.4e} g (DT)")
    print(f"  靶丸半径 = {phys.capsule_radius:.4e} cm")
    print(f"  初始燃料密度 = {phys.fuel_density:.4e} g/cm³")

    # 状态方程测试
    rho = np.array([0.25, 1.0, 10.0, 100.0])
    e_int = np.array([1e12, 1e13, 1e14, 1e15])
    p = phys.pressure(rho, e_int)
    T = phys.temperature(rho, p)
    cs = phys.sound_speed(rho, p)
    s = phys.specific_entropy(rho, p)

    print("\n  状态方程测试:")
    print(f"  {'ρ [g/cm³]':>12} {'p [dyn/cm²]':>14} {'T [K]':>12} {'c_s [cm/s]':>12} {'s':>10}")
    for i in range(len(rho)):
        print(f"  {rho[i]:12.4f} {p[i]:14.4e} {T[i]:12.4e} {cs[i]:12.4e} {s[i]:10.4f}")

    # Rankine-Hugoniot 测试
    print("\n  Rankine-Hugoniot 激波关系:")
    for mach in [1.5, 2.0, 3.0, 5.0]:
        rh = phys.rankine_hugoniot(mach)
        print(f"  M_s={mach:.1f}: ρ₂/ρ₁={rh['rho_ratio']:.4f}, "
              f"p₂/p₁={rh['p_ratio']:.4f}, T₂/T₁={rh['T_ratio']:.4f}")

    # Sedov-Taylor 自相似解
    print("\n  Sedov-Taylor 点爆炸解:")
    E0 = 1e15
    rho0 = 1.0
    for t_us in [0.1, 0.5, 1.0, 5.0]:
        t = t_us * 1e-6
        r_s, v_s = compute_sedov_solution(E0, rho0, DEFAULT_GAMMA, t)
        print(f"  t={t_us:.1f} μs: r_s={r_s:.4e} cm, v_s={v_s:.4e} cm/s")

    # 输运系数
    print("\n  Braginskii 粘性与 Spitzer 热导率:")
    for T_val in [100, 1000, 10000]:
        eta = phys.braginskii_viscosity(1.0, T_val)
        kappa = phys.electron_thermal_conductivity(1.0, T_val)
        Re = phys.reynolds_number(1.0, 1e7, 0.01, eta)
        Kn = phys.knudsen_number(1.0, T_val, 0.01)
        print(f"  T={T_val:5d} K: η={eta:.4e} P, κ={kappa:.4e} erg/(s·cm·K), "
              f"Re={Re:.2e}, Kn={Kn:.4e}")

    # 弹性张量 (融合 [1002])
    print("\n  靶丸材料弹性张量分析 (融合 [1002] LLZO):")
    # 假设 DT 冰的弹性常数 (简化)
    C11, C12, C44 = 50.0, 30.0, 15.0  # GPa
    moduli = ElasticTensorAnalyzer.cubic_moduli(C11, C12, C44)
    stable, checks = ElasticTensorAnalyzer.check_mechanical_stability(C11, C12, C44)
    print(f"  C11={C11}, C12={C12}, C44={C44} GPa")
    print(f"  K_VRH={moduli['K_VRH']:.2f}, G_VRH={moduli['G_VRH']:.2f} GPa")
    print(f"  E={moduli['Young_modulus']:.2f} GPa, ν={moduli['Poisson_ratio']:.4f}")
    print(f"  力学稳定性: {stable}, 判据: {checks}")

    # 临界密度
    nc = phys.critical_density()
    print(f"\n  临界密度 n_c = {nc:.4e} cm⁻³ (351 nm 激光)")

    return phys


def run_mesh_generation():
    """
    第二阶段: 网格生成与拓扑分析。
    融合 [756] [1310] [1331] [1413]。
    """
    section("阶段 2: 网格生成与拓扑分析")

    # 矩形网格
    print("  [矩形轴对称网格]")
    mesh_rect = generate_icf_mesh('rectangular', nr=15, nz=15,
                                   r_min=0.0, r_max=0.2, z_min=-0.2, z_max=0.2)
    print(f"    节点数: {mesh_rect['node_xy'].shape[0]}")
    print(f"    三角形单元数: {mesh_rect['tri_elements'].shape[0]}")
    print(f"    边界节点数: {len(mesh_rect['boundary_nodes'])}")
    print(f"    边界边数: {mesh_rect['boundary_edges'].shape[0]}")

    # 顶点-单元邻接
    vtoe = mesh_rect['vtoe']
    degrees = [vtoe.vertex_degree(v) for v in range(min(5, mesh_rect['node_xy'].shape[0]))]
    print(f"    前 5 个节点度: {degrees}")

    # 球面网格
    print("\n  [球面正二十面体网格] (融合 [1413])")
    verts, faces = IcosahedronSphereMesh.generate(tessellation_level=2)
    print(f"    细分层级: 2")
    print(f"    顶点数: {verts.shape[0]}")
    print(f"    面片数: {faces.shape[0]}")

    # 验证球面
    norms = np.linalg.norm(verts, axis=1)
    print(f"    顶点到原点距离: [{norms.min():.8f}, {norms.max():.8f}] (应为 1.0)")

    # 最小包围球 (融合 [1413] Welzl)
    print("\n  [Welzl 最小包围球] (融合 [1413])")
    rng = np.random.default_rng(42)
    test_points = rng.standard_normal((50, 3))
    center, radius = WelzlMinBoundingSphere.compute(test_points)
    print(f"    50 个 3D 随机点")
    print(f"    包围球心: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"    包围半径: {radius:.4f}")
    # 验证所有点在球内
    dists = np.linalg.norm(test_points - center, axis=1)
    print(f"    最大距离: {dists.max():.4f} ≤ {radius:.4f}: {dists.max() <= radius + 1e-10}")

    # 三角形网格 I/O
    print("\n  [TRIANGLE 格式 I/O] (融合 [1310])")
    io_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_temp_mesh')
    os.makedirs(io_dir, exist_ok=True)
    TriangleIO.write_node(os.path.join(io_dir, 'test.node'), mesh_rect['node_xy'],
                          mesh_rect['boundary_mask'])
    TriangleIO.write_element(os.path.join(io_dir, 'test.ele'), mesh_rect['tri_elements'])
    node_read, marker_read = TriangleIO.read_node(os.path.join(io_dir, 'test.node'))
    print(f"    写入/读取节点: {mesh_rect['node_xy'].shape[0]} → {node_read.shape[0]}")
    print(f"    节点坐标一致性: {np.allclose(node_read, mesh_rect['node_xy'])}")
    # 清理临时文件
    import shutil
    shutil.rmtree(io_dir, ignore_errors=True)

    return mesh_rect


def run_fd_analysis():
    """
    第三阶段: 高阶有限差分分析。
    验证收敛阶数和色散/耗散特性。
    """
    section("阶段 3: 高阶有限差分算子分析")

    # 收敛性测试
    print("  [中心差分收敛性]")
    x_range = (0.0, 2 * np.pi)
    for order in [2, 4, 6]:
        N_list = [21, 41, 81, 161]
        result = compute_fd_error_convergence(
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            x_range, N_list, fd_order=order
        )
        rates = result['rate']
        avg_rate = np.mean(rates) if rates else 0
        print(f"    {order}阶中心差分: 平均收敛率 = {avg_rate:.2f} (理论: {order})")

    # WENO5 测试
    print("\n  [WENO5 重构]")
    N = 64
    x = np.linspace(0, 2 * np.pi, N)
    dx = x[1] - x[0]
    f = np.sin(x)
    div_f = WENO5.flux_divergence(f, dx)
    exact = np.cos(x)
    # 只比较内部点
    err = np.max(np.abs(div_f[3:-3] - exact[3:-3]))
    print(f"    N={N}: max error (内部) = {err:.4e}")

    # 紧致差分测试
    print("\n  [4阶紧致差分]")
    df_compact = CompactDifference.derivative(f, dx, order=4)
    err_compact = np.max(np.abs(df_compact[2:-2] - exact[2:-2]))
    print(f"    N={N}: max error = {err_compact:.4e}")

    # 柱坐标算子
    print("\n  [柱坐标 Laplace 算子]")
    nr, nz = 21, 21
    r = np.linspace(0.01, 1.0, nr)
    z = np.linspace(-1.0, 1.0, nz)
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    RR, ZZ = np.meshgrid(r, z, indexing='ij')
    phi = RR ** 2 * np.cos(ZZ)  # 测试函数
    lap = CylindricalOperators.laplacian_axisym(phi, r, dr, dz)
    # 精确 Laplace: ∇²(r² cos z) = 4 cos z - r² cos z + (1/r)(2r cos z)
    lap_exact = 4 * np.cos(ZZ) - RR ** 2 * np.cos(ZZ) + 2 * np.cos(ZZ)
    err_lap = np.max(np.abs(lap[2:-2, 2:-2] - lap_exact[2:-2, 2:-2]))
    print(f"    测试函数: φ = r² cos(z)")
    print(f"    max error = {err_lap:.4e}")

    return True


def run_stability_analysis():
    """
    第四阶段: 稳定性分析。
    """
    section("阶段 4: 数值稳定性分析")

    # Von Neumann 分析
    print("  [Von Neumann 稳定性分析]")
    vna = VonNeumannAnalysis()
    c = 1.0
    dx = 0.01

    methods_data = {}
    for dt_frac in [0.5, 0.9, 1.0, 1.1]:
        dt = dt_frac * dx / c
        theta, g_lw, nu_lw = vna.amplification_factor_lax_wendroff(c, dt, dx)
        is_stable, max_g = vna.check_stability(g_lw)
        methods_data[dt_frac] = (is_stable, max_g, nu_lw)
        status = "STABLE" if is_stable else "UNSTABLE"
        print(f"    Lax-Wendroff ν={nu_lw:.2f}: max|g|={max_g:.6f} [{status}]")

    # CFL 条件
    print("\n  [CFL 条件计算]")
    cfl = CFLCondition()
    cs = 1e7  # cm/s
    u = 3e7   # cm/s
    kappa = 1e4  # cm²/s
    dx = dz = 0.001  # cm
    dt_a = cfl.acoustic_cfl(cs, dx, dz)
    dt_v = cfl.advective_cfl(u, dx, dz)
    dt_d = cfl.diffusive_cfl(kappa, dx, dz)
    dt_c = cfl.combined_cfl(cs, u, kappa, dx, dz)
    print(f"    声速 CFL: dt ≤ {dt_a:.4e} s")
    print(f"    对流 CFL: dt ≤ {dt_v:.4e} s")
    print(f"    扩散 CFL: dt ≤ {dt_d:.4e} s")
    print(f"    综合 CFL: dt ≤ {dt_c:.4e} s")

    # 矩阵特征值谱
    print("\n  [矩阵特征值谱分析]")
    msa = MatrixStabilityAnalysis()
    for N in [16, 32, 64]:
        A = msa.fd_matrix_1d(N, dx, order=4, bc_type='periodic')
        eigs, spec_rad, stiffness = msa.eigenvalue_spectrum(A)
        dt_max = msa.max_stable_dt(A, method='rk4')
        print(f"    N={N}: ρ(A)={spec_rad:.4e}, 刚性比={stiffness:.2e}, "
              f"dt_max(RK4)={dt_max:.4e}")

    # 综合 ICF 稳定性
    print("\n  [ICF 综合稳定性分析]")
    result = analyze_icf_stability(N=32, dx=0.001, gamma=5/3, rho0=1.0, p0=1e12)
    print(f"    声速: {result['sound_speed']:.4e} cm/s")
    print(f"    dt_acoustic: {result['dt_acoustic']:.4e} s")
    print(f"    dt_matrix: {result['dt_matrix']:.4e} s")
    print(f"    Lax-Wendroff 稳定: {result['lax_wendroff_stable']}")

    return result


def run_quadrature_verification():
    """
    第五阶段: 求积规则验证。
    融合 [1324] Wandzura, [919] product_rule。
    """
    section("阶段 5: 高斯求积规则验证")

    # Wandzura 规则精度测试
    print("  [Wandzura 三角形求积] (融合 [1324])")
    def f_poly(x, y):
        return x ** 3 * y ** 2  # 5次多项式
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.0, 1.0])
    exact = 1.0 / 840.0  # ∫∫ x³y² dA over unit triangle

    for order in [1, 3, 5, 7]:
        val = WandzuraRule.integrate_on_triangle(f_poly, v1, v2, v3, order=order)
        err = abs(val - exact)
        print(f"    order={order}: ∫∫ x³y² = {val:.10f}, error = {err:.2e}")

    # 乘积规则
    print("\n  [2D 乘积求积] (融合 [919])")
    def f_2d(r, z):
        return np.exp(-(r ** 2 + z ** 2))
    val = ProductRule.integrate_2d(f_2d, (-2, 2), (-2, 2), n_r=5, n_z=5)
    exact_gauss = np.pi * (1 - np.exp(-4)) ** 2 / 4  # 近似
    from scipy.special import erf
    exact_gauss = np.pi * erf(2) ** 2
    print(f"    ∫∫ exp(-(r²+z²)) dr dz = {val:.8f}")
    print(f"    精确值 = {exact_gauss:.8f}, error = {abs(val - exact_gauss):.2e}")

    # 球面积分
    print("\n  [球面积分]")
    sq = SphericalQuadrature()
    val_sphere = sq.integrate_sphere(lambda th, ph: 1.0, n_theta=10, n_phi=20)
    print(f"    ∫∫ sin θ dθ dφ = {val_sphere:.6f} (精确: 4π = {4 * np.pi:.6f})")

    return True


def run_time_integration():
    """
    第六阶段: 时间积分与哈密顿守恒。
    融合 [405] 后向 Euler, [619] 辛积分。
    """
    section("阶段 6: 时间积分方法比较")

    # 简单 ODE: du/dt = -u, u(0)=1, 精确解: u=exp(-t)
    print("  [ODE du/dt = -u, u(0)=1]")
    rhs = lambda t, u: -u
    u0 = np.array([1.0])

    for method in ['euler', 'ssprk3', 'rk4']:
        result = integrate_ode(rhs, u0, (0, 2), 0.01, method=method)
        u_final = result['u'][-1][0]
        u_exact = np.exp(-2)
        err = abs(u_final - u_exact)
        print(f"    {method:>8s}: u(2) = {u_final:.8f}, error = {err:.4e}, steps = {result['n_steps']}")

    # 受扰开普勒轨道 (辛积分)
    print("\n  [受扰开普勒辛积分] (融合 [619])")
    delta = 0.005

    def grad_V(q):
        r = np.sqrt(q[0] ** 2 + q[1] ** 2)
        r3 = max(r ** 3, 1e-30)
        r5 = max(r ** 5, 1e-30)
        return q / r3 + 3 * delta * q / (2 * r5)

    sv = StormerVerlet(grad_V, mass=1.0, dt=0.01)
    q = np.array([1.0, 0.0])
    p = np.array([0.0, 1.0])
    state = (q, p)
    hchecker = HamiltonianConservationChecker()

    H_hist = []
    q_hist = [q.copy()]
    for step in range(2000):
        H = hchecker.perturbed_kepler_hamiltonian(state[0], state[1], delta)
        H_hist.append(H)
        state, _ = sv.step(0, state)
        q_hist.append(state[0].copy())

    energy_report = hchecker.check_energy_drift(H_hist, 'Stormer-Verlet')
    print(f"    H(0) = {energy_report['H_initial']:.10f}")
    print(f"    H(end) = {energy_report['H_final']:.10f}")
    print(f"    最大漂移: {energy_report['max_drift']:.4e}")
    print(f"    相对漂移: {energy_report['relative_max_drift']:.4e}")
    print(f"    能量有界: {energy_report['bounded']}")

    # 后向 Euler (融合 [405])
    print("\n  [后向 Euler 求解热方程] (融合 [405])")
    from scipy.sparse import diags
    N = 50
    dx = 1.0 / (N - 1)
    alpha = 0.1
    # 构建 Laplace 矩阵
    main_diag = -2.0 * np.ones(N) / dx ** 2
    off_diag = np.ones(N - 1) / dx ** 2
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    A = alpha * A

    x = np.linspace(0, 1, N)
    u = np.sin(np.pi * x)
    dt = 0.001
    be = BackwardEuler(lambda t, u: A @ u, dt, A_sparse=A)
    t = 0.0
    for _ in range(100):
        u, t = be.step(t, u)
    u_exact = np.sin(np.pi * x) * np.exp(-alpha * np.pi ** 2 * t)
    err_be = np.sqrt(np.mean((u - u_exact) ** 2))
    print(f"    t={t:.3f}: RMSE = {err_be:.4e}")

    return {'energy_report': energy_report, 'be_error': err_be}


def run_linear_solver_demo():
    """
    第七阶段: 线性代数求解器。
    融合 [972] r8but, [982] r8ge_np。
    """
    section("阶段 7: 线性代数求解器")

    # 带状上三角矩阵 (融合 [972])
    print("  [带状上三角矩阵] (融合 [972])")
    N = 10
    mu = 2
    but = BandedUpperTriangular(N, mu)
    diag = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    but.set_diagonal(diag)
    but.set_superdiagonal(1, np.ones(N - 1))
    but.set_superdiagonal(2, 0.5 * np.ones(N - 2))

    b = np.ones(N) * 10.0
    x = but.solve(b)
    Ax = but.matvec(x)
    residual = np.max(np.abs(Ax - b))
    det = but.determinant()
    print(f"    N={N}, μ={mu}")
    print(f"    残差: {residual:.4e}")
    print(f"    det(A) = {det:.4e}")

    # 稠密 LU 分解 (融合 [982])
    print("\n  [稠密非主元 LU 分解] (融合 [982])")
    N = 8
    rng = np.random.default_rng(42)
    # 对角占优矩阵
    A = rng.standard_normal((N, N))
    A = A + N * np.eye(N)  # 确保对角占优
    b = rng.standard_normal(N)

    lu = DenseNonPivotingLU(N)
    info = lu.factor(A)
    if info == 0:
        x = lu.solve(b)
        residual = np.max(np.abs(A @ x - b))
        det_lu = lu.determinant()
        det_np = np.linalg.det(A)
        print(f"    N={N}")
        print(f"    分解状态: 成功 (info=0)")
        print(f"    残差: {residual:.4e}")
        print(f"    det(LU) = {det_lu:.6e}")
        print(f"    det(np) = {det_np:.6e}")
        print(f"    行列式相对误差: {abs(det_lu - det_np) / abs(det_np):.4e}")
    else:
        print(f"    分解失败 (info={info})")

    # 稀疏 FEM 组装
    print("\n  [稀疏 FEM 组装] (融合 [405])")
    N = 20
    dx = 1.0 / (N - 1)
    K = SparseFEMAssembler.assemble_1d_diffusion(N, dx, kappa=1.0)
    M = SparseFEMAssembler.assemble_1d_mass(N, dx, rho_cv=1.0)
    print(f"    刚度矩阵: {K.shape}, nnz={K.nnz}")
    print(f"    质量矩阵: {M.shape}, nnz={M.nnz}")

    # 求解 K u = f
    x_grid = np.linspace(0, 1, N)
    f = np.sin(np.pi * x_grid)
    f[0] = 0.0
    f[-1] = 0.0
    K_mod, f_mod = SparseFEMAssembler.apply_dirichlet_bc(K, f.copy(), [0, N - 1], [0.0, 0.0])
    u_sol = solve_linear_system(K_mod, f_mod)
    print(f"    解的最大值: {np.max(u_sol):.6f}")

    return True


def run_symmetry_and_perturbation():
    """
    第八阶段: 内爆对称性与扰动分析。
    融合 [910] prime, [042] asa144。
    """
    section("阶段 8: ICF 内爆对称性与扰动分析")

    # 素数序列生成 (融合 [910])
    print("  [素数筛与低差异序列] (融合 [910])")
    primes = PrimeBasedSeeding.prime_sieve(100)
    print(f"    ≤100 的素数: {len(primes)} 个")
    print(f"    前 20 个: {primes[:20]}")

    # Halton 序列
    halton = PrimeBasedSeeding.halton_sequence(10, dimensions=2)
    print(f"    Halton 序列 (前 5 个 2D 点):")
    for i in range(5):
        print(f"      {i}: ({halton[i, 0]:.6f}, {halton[i, 1]:.6f})")

    # 随机列联表扰动 (融合 [042])
    print("\n  [Boyett 随机列联表] (融合 [042] ASA144)")
    row_totals = np.array([10, 20, 15])
    col_totals = np.array([12, 18, 15])
    rcont = RandomContingencyPerturbation.boyett_rcont(row_totals, col_totals)
    print(f"    行边际: {row_totals}")
    print(f"    列边际: {col_totals}")
    print(f"    随机列联表:")
    print(f"      {rcont}")
    print(f"    行和: {rcont.sum(axis=1)}, 列和: {rcont.sum(axis=0)}")
    print(f"    行和一致: {np.array_equal(rcont.sum(axis=1), row_totals)}")
    print(f"    列和一致: {np.array_equal(rcont.sum(axis=0), col_totals)}")

    # 靶丸表面扰动
    print("\n  [靶丸表面扰动生成]")
    pert = InitialPerturbationFactory.surface_roughness_perturbation(
        n_theta=16, n_phi=32, rms_roughness_cm=5e-6, seed=42
    )
    rms_actual = np.sqrt(np.mean(pert ** 2))
    print(f"    网格: {pert.shape}")
    print(f"    目标 RMS: 5.00e-06 cm")
    print(f"    实际 RMS: {rms_actual:.4e} cm")

    # Legendre 分解
    print("\n  [Legendre 球谐分解]")
    ld = LegendreDecomposition()
    theta = np.linspace(0, np.pi, 64)
    # 构造已知模式的 R(θ)
    R_test = 1.0 + 0.005 * (3 * np.cos(theta) ** 2 - 1) / 2  # P2 模式
    modes = ld.decompose(R_test, theta, max_mode=6)
    print(f"    测试 R(θ) = 1 + 0.005 P₂(cos θ)")
    for l in range(7):
        print(f"      a_{l} = {modes[l]:.6f}")

    metrics = ld.symmetry_metrics(modes)
    print(f"    P₂ 不对称度: {metrics['P2_asymmetry']:.6f}")
    print(f"    P₄ 不对称度: {metrics['P4_asymmetry']:.6f}")
    print(f"    总不对称度: {metrics['total_asymmetry']:.6f}")

    # RT 增长模型
    print("\n  [Rayleigh-Taylor 增长率]")
    rt = RTGrowthModel()
    for l in [2, 4, 6, 8, 10]:
        gamma = rt.growth_rate(l, R=0.01, A=0.3, g_eff=1e14, sigma=0.0, rho_h=100.0)
        print(f"    l={l:2d}: γ = {gamma:.4e} /s")

    # 对称性评估
    print("\n  [内爆对称性综合评估]")
    analyzer = ImplosionSymmetryAnalyzer()
    # 构造多时刻数据
    R_data = np.zeros((5, 64))
    for t_idx in range(5):
        asym_factor = 0.005 * (1 + 0.5 * t_idx)
        R_data[t_idx, :] = 1.0 + asym_factor * (3 * np.cos(theta) ** 2 - 1) / 2
    result = analyzer.analyze_implosion_symmetry(R_data, theta)
    assessment = analyzer.assess_symmetry_quality(result['metrics_history'])
    print(f"    对称性评估: {assessment['details']}")
    print(f"    P₂ = {assessment['P2']:.4f}, P₄ = {assessment['P4']:.4f}")

    return True


def run_ml_surrogate_demo():
    """
    第九阶段: ML 代理模型。
    融合 [1215] SVM/交叉验证, [1002] 弹性张量。
    """
    section("阶段 9: ML 代理模型 (不稳定性预测)")

    print("  [SVM 不稳定性分类] (融合 [1215])")
    result = ICFSurrogateModel.train_and_validate(n_samples=200, k_folds=5, seed=42)
    print(f"    训练样本: {result['n_samples']}")
    print(f"    特征维度: {result['n_features']}")
    print(f"    5-fold 精度: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")

    # 预测示例
    scaler = result['scaler']
    clf = result['classifier']
    test_sample = np.array([[0.01, 5e-5, 3e7, 0.3, 0.01]])
    test_scaled = scaler.transform(test_sample)
    pred = clf.predict(test_scaled)
    score = clf.decision_function(test_scaled)
    label = "不稳定" if pred[0] < 0 else "稳定"
    print(f"    测试样本: drive_asym=1%, rough=50nm, v=3e7, ρR=0.3")
    print(f"    预测: {label} (score={score[0]:.4f})")

    # 交叉验证详细结果
    print("\n  [交叉验证详情]")
    X, y = ICFSurrogateModel.generate_training_data(200, 42)
    X_scaled = scaler.fit_transform(X)
    splits = CrossValidator.k_fold_split(200, k=5)
    for fold, (train_idx, val_idx) in enumerate(splits):
        clf_k = SVMClassifier(C=1.0, n_iter=500)
        clf_k.fit(X_scaled[train_idx], y[train_idx])
        acc = clf_k.accuracy(X_scaled[val_idx], y[val_idx])
        print(f"    Fold {fold + 1}: accuracy = {acc:.4f}")

    return result


def run_verification():
    """
    第十阶段: 完整验证。
    融合 [1297] 形式化验证, [619] 哈密顿守恒。
    """
    section("阶段 10: 数值验证与形式化检验")

    result = run_full_verification()

    # 配置空间覆盖 (融合 [1297])
    print("\n  [配置空间覆盖分析] (融合 [1297])")
    csa = ConfigurationSpaceAnalyzer()
    # 模拟参数扫描结果
    rng = np.random.default_rng(42)
    results_sim = rng.exponential(0.02, 100)  # 模拟对称性偏离
    values, cdf = csa.compute_coverage_cdf(results_sim)
    robustness = csa.robustness_analysis(results_sim, threshold=0.01)
    print(f"    模拟次数: {robustness['n_total']}")
    print(f"    通过次数: {robustness['n_pass']}")
    print(f"    覆盖率: {robustness['coverage']:.4f}")
    print(f"    阈值: {robustness['threshold']}")
    print(f"    CDF 分位数 (50%): {values[len(values) // 2]:.6f}")

    return result


def run_final_summary():
    """打印最终总结"""
    section("计算完成 — 总结")

    print("""
  本项目成功融合了 15 个种子项目的核心算法, 构建了完整的 ICF 内爆
  对称性模拟与高阶有限差分稳定性分析框架。

  种子项目映射:
    [1002] LLZO 分子动力学 → 弹性张量/材料力学性质
    [1215] 风险决策解码 → SVM 分类/交叉验证/ML 代理
    [756] mesh_vtoe → 顶点-单元邻接关系
    [1310] triangle_io → TRIANGLE 格式网格 I/O
    [1324] wandzura_rule → 三角形高对称求积
    [910] prime → 素数筛/Halton 低差异序列
    [1331] triangulation_boundary → 边界边检测
    [972] r8but → 带状上三角矩阵运算
    [1297] FormalCellular → 配置空间覆盖/CDF 分析
    [1413] welzl/icosahedron → 最小包围球/球面网格
    [919] product_rule → 多维乘积求积
    [405] fem2d_heat_sparse → 稀疏 FEM/后向 Euler
    [619] kepler_perturbed → 辛积分/哈密顿守恒
    [042] asa144 → 随机列联表/扰动分布
    [982] r8ge_np → 非主元 LU 分解

  科学计算内容:
    • 理想气体状态方程 + Rankine-Hugoniot 激波关系
    • Braginskii 粘性 + Spitzer-Härm 热传导
    • 高阶有限差分 (2/4/6/8阶中心, WENO5, 紧致差分)
    • Von Neumann 稳定性 + CFL 条件 + 矩阵谱分析
    • 多种时间积分 (FE, SSP-RK, RK4, 后向 Euler, 辛积分)
    • Legendre 球谐分解 + RT 不稳定性增长
    • ML 代理模型 (SVM + 交叉验证)
    • MMS 验证 + 守恒律检验 + 形式化配置空间分析
""")


def main():
    """主入口函数"""
    t_start = time.time()
    print_header()

    try:
        # 阶段 1: 物理参数
        phys = run_physics_setup()

        # 阶段 2: 网格生成
        mesh = run_mesh_generation()

        # 阶段 3: 有限差分
        run_fd_analysis()

        # 阶段 4: 稳定性分析
        run_stability_analysis()

        # 阶段 5: 求积规则
        run_quadrature_verification()

        # 阶段 6: 时间积分
        run_time_integration()

        # 阶段 7: 线性代数
        run_linear_solver_demo()

        # 阶段 8: 对称性与扰动
        run_symmetry_and_perturbation()

        # 阶段 9: ML 代理
        run_ml_surrogate_demo()

        # 阶段 10: 验证
        run_verification()

        # 总结
        run_final_summary()

        elapsed = time.time() - t_start
        print(f"  总计算时间: {elapsed:.2f} 秒")
        print("=" * 70)
        print("  所有阶段成功完成, 无报错。")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
