"""
main.py

马约拉纳费米子拓扑量子计算综合数值模拟平台
================================================================================

本项目基于15个科研代码种子项目的核心算法，融合构建了一个面向
凝聚态物理前沿——马约拉纳费米子拓扑量子计算——的博士级数值
计算框架。

统一入口，零参数运行，自动执行以下全流程：
    1. Kitaev链BdG哈密顿量构建与对角化
    2. 马约拉纳零能模识别与波函数分析
    3. 无序系统中的Andreev反射随机游走
    4. 拓扑相变的马尔可夫链分析
    5. 非阿贝尔编织动力学
    6. 有限元输运模拟
    7. BTK电导谱拟合
    8. 量子态编码与保真度分析
    9. CVT优化采样

运行方式：
    python main.py
"""

import numpy as np
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kitaev_bdg_solver import KitaevBdGSolver
from disorder_random_walk import MajoranaDisorderRandomWalk
from majorana_wavefunction import MajoranaWavefunctionSolver
from topological_phase_analysis import TopologicalMarkovChain
from braiding_dynamics import MajoranaBraidingDynamics, MajoranaImpurityDynamics
from finite_element_transport import FEM1DTransport, PiecewiseConstantPotential
from optimization_fitting import (GammaLogCalculator, FermiIntegralCalculator,
                                   BTKFittingModel)
from quantum_state_encoding import (GrayCodeEncoder, TopologicalParityCheck,
                                     BlochSphereFidelity, MajoranaQuantumCode)
from cvt_sampling import CVTSampler, BrillouinZoneIntegrator


def section_header(title: str):
    """打印分段标题。"""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_kitaev_bdg_analysis():
    """
    任务1：Kitaev链BdG分析（种子964_r83p + 1115_sphere_distance统计）
    """
    section_header("任务1：Kitaev链 BdG 哈密顿量对角化与零能模识别")

    # 构建20格点的Kitaev链
    solver = KitaevBdGSolver(n_sites=20, mu=0.5, t=1.0, delta=0.8,
                             periodic=False)

    print("\n[物理参数]")
    print(f"  格点数 N = {solver.n}")
    print(f"  化学势 μ = {solver.mu} meV")
    print(f"  跃迁强度 t = {solver.t} meV")
    print(f"  超导配对势 Δ = {solver.delta} meV")
    print(f"  边界条件：开边界")

    # 完整对角化
    eigvals, eigvecs = solver.diagonalize()
    print("\n[BdG 本征值谱 (meV)]")
    print(f"  最低10个: {np.round(eigvals[:10], 6)}")
    print(f"  最高10个: {np.round(eigvals[-10:], 6)}")

    # 能隙
    gap = solver.compute_energy_gap(eigvals)
    print(f"\n[能隙] E_gap = {gap:.6f} meV")

    # 拓扑相图
    mu_vals = np.linspace(-3.0, 3.0, 61)
    gaps = solver.topological_phase_diagram(mu_vals)
    print(f"\n[拓扑相图] 在 μ∈[-3,3] 范围内的能隙范围: "
          f"[{np.min(gaps):.4f}, {np.max(gaps):.4f}] meV")
    print(f"  相变点预测: |μ| = 2|t| = {2.0 * abs(solver.t):.2f} meV")

    # 零能模识别
    modes_u, modes_v = solver.identify_majorana_zero_modes(
        eigvals, eigvecs, energy_tol=1e-6)
    n_mzm = len(modes_u)
    print(f"\n[马约拉纳零能模] 检测到 {n_mzm} 个MZM")
    if n_mzm > 0:
        print(f"  边界波函数最大值: {np.max(np.abs(modes_u[0])):.6f}")
        # 参与率
        ipr = np.sum(modes_u[0] ** 4) / (np.sum(modes_u[0] ** 2) ** 2)
        print(f"  反参与率 IPR = {ipr:.6f} (越接近1越局域)")

    return solver, eigvals, eigvecs


def run_disorder_random_walk():
    """
    任务2：无序Andreev反射随机游走（种子1010_random_walk_2d_simulation）
    """
    section_header("任务2：无序Kitaev链中的Andreev反射随机游走")

    walker = MajoranaDisorderRandomWalk(
        n_sites=50, disorder_strength=0.5, delta=0.8,
        t=1.0, mu0=0.0, rng_seed=42
    )

    print("\n[物理参数]")
    print(f"  无序强度 W = {walker.W} meV")
    print(f"  超导配对势 Δ = {walker.delta} meV")

    # 零能Andreev随机游走
    d2_ave, d2_max = walker.simulate_andreev_random_walk(
        num_steps=200, num_walks=1000, energy=0.0
    )

    p_a = walker._andreev_reflection_probability(0.0)
    print(f"\n[Andreev反射概率] P_A(E=0) = {p_a:.6f}")
    print(f"[随机游走统计]")
    print(f"  <r^2>(t=50)  = {d2_ave[50]:.4f}")
    print(f"  <r^2>(t=200) = {d2_ave[-1]:.4f}")
    print(f"  max(r^2)(t=200) = {d2_max[-1]:.4f}")
    print(f"  扩散系数估计 D ≈ <r^2>/(2t) = {d2_ave[-1]/(2*200):.6f}")

    # 关联函数
    corr = walker.disorder_averaged_correlation(
        num_realizations=30, max_distance=20
    )
    print(f"\n[无序关联函数] C(r=0)={corr[0]:.4f}, C(r=5)={corr[5]:.4f}")

    # 局域化长度标度
    w_vals = np.linspace(0.1, 2.0, 10)
    xi_vals = walker.localization_length_scaling(w_vals, num_realizations=30)
    print(f"\n[局域化长度标度]")
    for w, xi in zip(w_vals[::3], xi_vals[::3]):
        print(f"  W={w:.2f} -> ξ_eff={xi:.4f} (格点单位)")

    return walker


def run_wavefunction_analysis():
    """
    任务3：马约拉纳波函数谱展开（种子666_legendre_shifted_polynomial +
    1173_string_pde）
    """
    section_header("任务3：马约拉纳零能模波函数的谱展开与动力学演化")

    solver = MajoranaWavefunctionSolver(
        length=100.0, n_sites=200, mu=0.5, t=1.0, delta=0.8
    )

    x = np.linspace(0.0, solver.L, solver.n)
    psi = solver.analytical_zero_mode_profile(x)

    print("\n[解析波函数]")
    print(f"  纳米线长度 L = {solver.L} nm")
    print(f"  局域化长度 ξ = {2.0 * abs(solver.t) / abs(solver.delta):.2f} nm")
    print(f"  波函数归一化: {np.trapezoid(psi ** 2, x):.8f}")

    # 谱展开
    coeffs = solver.spectral_expansion_coefficients(psi, max_degree=20)
    psi_recon = solver.reconstruct_from_spectral(coeffs, x)
    error = np.max(np.abs(psi - psi_recon))
    print(f"\n[移位Legendre谱展开 (N=20)]")
    print(f"  前5个系数: {np.round(coeffs[:5], 6)}")
    print(f"  重构最大误差: {error:.2e}")

    # 有限差分演化
    u = solver.finite_difference_time_evolution(
        initial_wave=psi, num_steps=100, dt=0.005, alpha=0.25
    )
    print(f"\n[有限差分波包演化]")
    print(f"  CFL参数 α = 0.25 (稳定性条件 α≤1)")
    print(f"  终态能量守恒: {np.trapezoid(u[-1] ** 2, x):.8f}")
    print(f"  能量泄漏: {abs(np.trapezoid(u[0] ** 2, x) - np.trapezoid(u[-1] ** 2, x)):.2e}")

    # 概率流
    j = solver.compute_probability_current(psi + 0j)
    print(f"\n[概率流密度]")
    print(f"  最大流: {np.max(np.abs(j)):.6e} (马约拉纳条件下应为0)")

    return solver


def run_topological_phase_analysis():
    """
    任务4：拓扑相变马尔可夫链（种子1095_snakes_probability）
    """
    section_header("任务4：拓扑相变的马尔可夫链分析与相图")

    tmc = TopologicalMarkovChain(
        temperature=0.05, disorder_strength=0.2, delta=0.8, t=1.0
    )

    print("\n[物理参数]")
    print(f"  有效温度 k_B T/t = {tmc.T}")
    print(f"  无序强度 W/t = {tmc.W}")

    # 绕数计算
    print("\n[Z₂ 绕数拓扑不变量 ν(μ)]")
    for mu in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
        nu = tmc.winding_number(mu)
        print(f"  μ = {mu:+4.1f} -> ν = {nu:+2d}  "
              f"({'拓扑非平庸' if nu != 0 else '拓扑平庸'})")

    # 稳态分布
    print("\n[稳态分布 π = (π_trivial, π_topological, π_critical)]")
    for mu in [-2.5, -1.0, 0.0, 1.0, 2.5]:
        pi = tmc.steady_state_distribution(mu)
        print(f"  μ = {mu:+4.1f} -> π = ({pi[0]:.4f}, {pi[1]:.4f}, {pi[2]:.4f})")

    # 纠缠熵
    print("\n[纠缠熵 S_A (子系统大小=10)]")
    for mu in [-0.5, 0.5, 3.0]:
        s = tmc.compute_entanglement_entropy(mu, subsystem_size=10)
        nu = tmc.winding_number(mu)
        print(f"  μ = {mu:+4.1f}, ν = {nu:+2d} -> S_A = {s:.4f} "
              f"(理论值 ln(2)/2 ≈ {0.5 * np.log(2.0):.4f})")

    # 临界指数
    mu_near = np.linspace(1.9, 2.1, 21)
    xi = tmc.correlation_length_critical_exponent(mu_near)
    print(f"\n[临界指数] 在 μ≈2t 附近，关联长度最大值: {np.max(xi):.2e}")

    return tmc


def run_braiding_dynamics():
    """
    任务5：非阿贝尔编织动力学（种子1023_rigid_body_ode +
    908_predator_prey_ode）
    """
    section_header("任务5：马约拉纳非阿贝尔编织与耦合动力学")

    braid = MajoranaBraidingDynamics(num_majorana=4)

    print("\n[编织群关系验证]")
    diff = braid.compute_braid_group_relation(0, 1, 2)
    print(f"  Yang-Baxter关系偏差: ||σ_1σ_2σ_1 - σ_2σ_1σ_2||_F = {diff:.6e}")

    # 编织序列
    seq = [(0, 1), (1, 2), (2, 3), (1, 2), (0, 1)]
    U = braid.apply_braid_sequence(seq)
    print(f"\n[编织序列 {seq}]")
    print(f"  变换矩阵U的迹: {np.trace(U):.6f}")
    print(f"  行列式 det(U): {np.linalg.det(U):.6f}")

    # 刚体映射动力学
    t_span = np.linspace(0, 10, 101)
    xyz = braid.integrate_rigid_body(
        xyz0=np.array([1.0, 0.5, 0.2]),
        t_span=t_span, I1=1.0, I2=0.8, I3=0.6
    )
    print(f"\n[刚体映射动力学 (SO(3)类比)]")
    print(f"  初始状态: ({xyz[0,0]:.3f}, {xyz[0,1]:.3f}, {xyz[0,2]:.3f})")
    print(f"  终态:     ({xyz[-1,0]:.3f}, {xyz[-1,1]:.3f}, {xyz[-1,2]:.3f})")
    print(f"  状态模长变化: {np.linalg.norm(xyz[0]):.4f} -> {np.linalg.norm(xyz[-1]):.4f}")

    # 耦合动力学
    impurity = MajoranaImpurityDynamics(alpha=1.0, beta=0.5,
                                         gamma=0.8, delta=0.3)
    t_couple = np.linspace(0, 20, 201)
    y = impurity.integrate(y0=np.array([2.0, 1.0]), t_span=t_couple)
    H = impurity.conserved_quantity(y)
    print(f"\n[马约拉纳-杂质耦合 (Lotka-Volterra)]")
    print(f"  初始占据: n_M={y[0,0]:.3f}, n_I={y[0,1]:.3f}")
    print(f"  终态占据: n_M={y[-1,0]:.3f}, n_I={y[-1,1]:.3f}")
    print(f"  守恒量H的变化: {np.max(H) - np.min(H):.6e}")

    return braid, impurity


def run_finite_element_transport():
    """
    任务6：有限元输运（种子391_fem1d_heat_implicit + 923_pwc_plot_1d）
    """
    section_header("任务6：拓扑超导体中的准粒子有限元输运")

    x_nodes = np.linspace(0.0, 10.0, 51)
    fem = FEM1DTransport(x_nodes, diffusion_coeff=0.1)

    # 分段常数能隙
    pwc = PiecewiseConstantPotential(
        breakpoints=np.array([0.0, 3.0, 7.0, 10.0]),
        values=np.array([0.8, 0.0, 0.8])
    )

    print("\n[分段常数超导能隙模型]")
    print(f"  区域 [0,3]: Δ = {pwc.y[0]} meV (超导)")
    print(f"  区域 [3,7]: Δ = {pwc.y[1]} meV (正常区/约瑟夫森结)")
    print(f"  区域 [7,10]: Δ = {pwc.y[2]} meV (超导)")

    # 初始高斯波包
    u0 = np.exp(-(x_nodes - 5.0) ** 2 / 0.5)

    def source(x, t):
        return 0.1 * np.exp(-(x - 5.0) ** 2 / 1.0)

    t_array, u_history = fem.solve_time_dependent(
        u0=u0, t_final=2.0, num_steps=100,
        source_fn=source, bc_left=0.0, bc_right=0.0
    )

    print(f"\n[隐式有限元求解]")
    print(f"  空间节点: {len(x_nodes)}, 时间步: 100")
    print(f"  初始总密度: {np.trapezoid(u_history[0], x_nodes):.6f}")
    print(f"  终态总密度: {np.trapezoid(u_history[-1], x_nodes):.6f}")
    print(f"  密度守恒偏差: {abs(np.trapezoid(u_history[-1], x_nodes) - np.trapezoid(u_history[0], x_nodes)):.2e}")

    # 有效扩散系数
    d_eff = fem.effective_diffusion_with_majorana(pwc, energy=0.0)
    print(f"\n[Andreev反射修正扩散系数]")
    print(f"  D_0 = {fem.D:.3f}, D_eff = {d_eff:.6f}")
    print(f"  抑制因子: {d_eff / fem.D:.4f}")

    return fem, pwc


def run_optimization_fitting():
    """
    任务7：BTK电导谱拟合与费米积分（种子1266_toms178 + 1269_toms291）
    """
    section_header("任务7：BTK隧道电导谱拟合与热力学积分")

    # Gamma函数
    gl = GammaLogCalculator()
    print("\n[Gamma函数验证]")
    for x in [1.0, 2.0, 3.0, 5.5]:
        log_g, err = gl.alogam(x)
        gamma_val = gl.gamma(x)
        print(f"  Γ({x}) = {gamma_val:.6f}, lnΓ = {log_g:.6f}, err={err}")

    # 费米积分
    fi = FermiIntegralCalculator()
    print("\n[费米积分 F_n(η)]")
    for eta in [-5.0, -2.0, 0.0, 2.0, 5.0]:
        f1 = fi.fermi_integral(n=1, eta=eta)
        f2 = fi.fermi_integral(n=2, eta=eta)
        print(f"  η={eta:+5.1f}: F_1={f1:.4f}, F_2={f2:.4f}")

    # BTK拟合
    V = np.linspace(-2.0, 2.0, 81)
    true_params = np.array([0.8, 0.5, 0.05])
    model = BTKFittingModel(V, np.zeros_like(V))
    g_true = model.btk_conductance(V, *true_params)
    np.random.seed(42)
    g_noisy = g_true + 0.02 * np.random.randn(len(V))
    model.G_data = g_noisy

    print("\n[BTK电导谱拟合]")
    print(f"  真实参数: Δ={true_params[0]}, Z={true_params[1]}, γ={true_params[2]}")
    params, rmse = model.fit(initial_guess=np.array([0.6, 0.3, 0.03]))
    print(f"  拟合参数: Δ={params[0]:.4f}, Z={params[1]:.4f}, γ={params[2]:.4f}")
    print(f"  拟合RMSE: {rmse:.6f}")

    return model


def run_quantum_encoding():
    """
    任务8：量子态编码与保真度（种子485_gray_code + 1375_upc +
    1115_sphere_distance）
    """
    section_header("任务8：拓扑量子比特编码、Gray码与保真度分析")

    # Gray码
    encoder = GrayCodeEncoder()
    gray_seq = encoder.generate_gray_sequence(3)
    print("\n[Gray码序列 (3位)]")
    print(f"  序列: {gray_seq}")
    print("  相邻Hamming距离验证:")
    for i in range(len(gray_seq) - 1):
        d = encoder.hamming_distance(gray_seq[i], gray_seq[i + 1], 3)
        assert d == 1, "Gray码性质被破坏！"
    print("  ✓ 所有相邻码的Hamming距离均为1")

    # 拓扑校验
    parity = TopologicalParityCheck(num_majorana=4)
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    ev = parity.compute_stabilizer_eigenvalues(state)
    print(f"\n[稳定子本征值] {ev}")

    digits = np.array([1, 0, 1])
    check = parity.upc_style_check_digit(digits)
    print(f"[UPC-style校验] 数据{digits} -> 校验位 = {check}")

    # 保真度
    psi_plus = np.array([1.0, 0.0])
    psi_minus = np.array([0.0, 1.0])
    F = BlochSphereFidelity.fidelity(psi_plus, psi_minus)
    d_bloch = BlochSphereFidelity.bloch_distance(psi_plus, psi_minus)
    print(f"\n[布洛赫球距离]")
    print(f"  |0> 与 |1> 的保真度: {F:.6f}")
    print(f"  测地距离: {d_bloch:.6f} (理论值 π ≈ {np.pi:.6f})")

    mu_f, var_f = BlochSphereFidelity.average_fidelity_statistics(
        n_samples=500, rng_seed=42)
    print(f"\n[随机量子态统计]")
    print(f"  平均保真度: {mu_f:.4f} ± {np.sqrt(var_f):.4f}")
    print(f"  理论期望: 1/2 = 0.5")

    # 编码
    code = MajoranaQuantumCode(num_majorana=6)
    print("\n[拓扑量子编码 (6个马约拉纳 -> 2个逻辑比特)]")
    for logical in range(4):
        config = code.encode_logical_state(logical)
        decoded = code.decode_to_logical(config)
        status = "✓" if decoded == logical else "✗"
        print(f"  逻辑态 {logical} {status}")

    det_rate = code.error_detection_rate(error_probability=0.1,
                                          num_trials=500)
    print(f"\n[错误检测率] p_error=0.1 时检测率 = {det_rate:.4f}")

    return code


def run_cvt_sampling():
    """
    任务9：CVT优化采样（种子261_cvt_square_uniform）
    """
    section_header("任务9：布里渊区CVT优化采样")

    # BZ k点
    bz_cvt = CVTSampler(n_generators=64,
                         domain=(-np.pi, np.pi, -np.pi, np.pi),
                         rng_seed=42)
    kpts = bz_cvt.brillouin_zone_kpoints(a=1.0, num_iterations=30)

    print("\n[CVT k点采样]")
    print(f"  生成元数: {len(kpts)}")
    print(f"  k_x 范围: [{kpts[:,0].min():.4f}, {kpts[:,0].max():.4f}]")
    print(f"  k_y 范围: [{kpts[:,1].min():.4f}, {kpts[:,1].max():.4f}]")

    # 均匀性验证
    integrator = BrillouinZoneIntegrator(kpts, bz_cvt.domain)

    def coskx_cosky(kx, ky):
        return np.cos(kx) * np.cos(ky)

    result = integrator.integrate(coskx_cosky)
    print(f"\n[积分验证] ∫∫ cos(kx)cos(ky) dk_x dk_y = {result:.6f}")
    print(f"  解析值 = 0.0")

    # 费米面采样测试
    def dispersion(kx, ky):
        return -2.0 * (np.cos(kx) + np.cos(ky))

    fs = integrator.fermi_surface_sampling(dispersion,
                                            e_fermi=-1.0, tolerance=0.5)
    print(f"\n[费米面附近k点] 在 E_F=-1.0±0.5 范围内找到 {len(fs)} 个点")

    # 能量计算
    cvt_energy = bz_cvt.cvt_energy(kpts, sample_num=5000)
    print(f"\n[CVT能量泛函] F = {cvt_energy:.6f}")

    return bz_cvt, integrator


def main():
    """
    主函数：统一入口，零参数运行。
    """
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  马约拉纳费米子拓扑量子计算综合数值模拟平台".center(68) + "#")
    print("#" + "  Majorana Fermion Topological Quantum Computation Simulator".center(68) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    print("\n项目基于以下15个种子算法融合构建：")
    seeds = [
        "1010_random_walk_2d_simulation -> 无序Andreev反射随机游走",
        "964_r83p -> 周期三对角BdG矩阵分解",
        "923_pwc_plot_1d -> 分段常数超导能隙",
        "908_predator_prey_ode -> 马约拉纳-杂质耦合动力学",
        "1095_snakes_probability -> 拓扑相变马尔可夫链",
        "1023_rigid_body_ode -> 编织群SO(3)映射动力学",
        "391_fem1d_heat_implicit -> 准粒子输运有限元",
        "261_cvt_square_uniform -> 布里渊区CVT采样",
        "1115_sphere_distance -> 布洛赫球量子态距离",
        "1375_upc -> 拓扑量子奇偶校验",
        "1269_toms291 -> Gamma函数对数(费米积分)",
        "1266_toms178 -> Hooke-Jeeves优化(BTK拟合)",
        "666_legendre_shifted_polynomial -> 波函数谱展开",
        "485_gray_code_display -> 量子态Gray码编码",
        "1173_string_pde -> 波包动力学有限差分",
    ]
    for s in seeds:
        print(f"  • {s}")

    try:
        run_kitaev_bdg_analysis()
        run_disorder_random_walk()
        run_wavefunction_analysis()
        run_topological_phase_analysis()
        run_braiding_dynamics()
        run_finite_element_transport()
        run_optimization_fitting()
        run_quantum_encoding()
        run_cvt_sampling()

        print("\n" + "#" * 80)
        print("#" + "  所有任务执行完毕，数值模拟平台运行正常".center(68) + "#")
        print("#" * 80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] 执行过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（32个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: Kitaev BdG 拓扑相内能隙为正 ----
solver_tc01 = KitaevBdGSolver(n_sites=10, mu=0.5, t=1.0, delta=0.8, periodic=False)
eigvals_tc01, eigvecs_tc01 = solver_tc01.diagonalize()
gap_tc01 = solver_tc01.compute_energy_gap(eigvals_tc01)
assert gap_tc01 > 0, '[TC01] Kitaev BdG 拓扑相内能隙为正 FAILED'

# ---- TC02: BdG 能谱粒子-空穴对称性 ----
assert np.allclose(eigvals_tc01, -eigvals_tc01[::-1], atol=1e-10), '[TC02] BdG 能谱粒子-空穴对称性 FAILED'

# ---- TC03: 拓扑相图能隙在拓扑区较小在平庸区较大 ----
mu_vals_tc03 = np.array([-3.0, 0.0, 3.0])
gaps_tc03 = solver_tc01.topological_phase_diagram(mu_vals_tc03)
assert gaps_tc03[1] < gaps_tc03[0] and gaps_tc03[1] < gaps_tc03[2], '[TC03] 拓扑相图能隙分布 FAILED'

# ---- TC04: 开边界存在马约拉纳零能模 ----
modes_u_tc04, modes_v_tc04 = solver_tc01.identify_majorana_zero_modes(eigvals_tc01, eigvecs_tc01, energy_tol=1e-5)
assert len(modes_u_tc04) >= 1, '[TC04] 开边界存在马约拉纳零能模 FAILED'

# ---- TC05: Andreev反射概率在E=0时为1 ----
walker_tc05 = MajoranaDisorderRandomWalk(n_sites=20, disorder_strength=0.1, delta=0.8, t=1.0, mu0=0.0, rng_seed=42)
p_a_tc05 = walker_tc05._andreev_reflection_probability(0.0)
assert abs(p_a_tc05 - 1.0) < 1e-10, '[TC05] Andreev反射概率在E=0时为1 FAILED'

# ---- TC06: 随机游走固定种子可复现 ----
np.random.seed(42)
walker_tc06a = MajoranaDisorderRandomWalk(n_sites=20, disorder_strength=0.1, delta=0.8, t=1.0, mu0=0.0, rng_seed=42)
d2_ave_tc06a, d2_max_tc06a = walker_tc06a.simulate_andreev_random_walk(num_steps=50, num_walks=100, energy=0.0)
walker_tc06b = MajoranaDisorderRandomWalk(n_sites=20, disorder_strength=0.1, delta=0.8, t=1.0, mu0=0.0, rng_seed=42)
d2_ave_tc06b, d2_max_tc06b = walker_tc06b.simulate_andreev_random_walk(num_steps=50, num_walks=100, energy=0.0)
assert np.allclose(d2_ave_tc06a, d2_ave_tc06b, atol=1e-10), '[TC06] 随机游走固定种子可复现 FAILED'

# ---- TC07: 参与率计算范围合理 ----
wf_tc07 = np.ones(10) / np.sqrt(10)
ipr_tc07 = walker_tc05.compute_participation_ratio(wf_tc07)
assert 0.09 < ipr_tc07 <= 1.0, '[TC07] 参与率计算范围 FAILED'

# ---- TC08: 相同波函数重叠积分为1 ----
mws_tc08 = MajoranaWavefunctionSolver(length=10.0, n_sites=50, mu=0.5, t=1.0, delta=0.8)
x_tc08 = np.linspace(0, mws_tc08.L, mws_tc08.n)
psi_tc08 = mws_tc08.analytical_zero_mode_profile(x_tc08)
overlap_tc08 = mws_tc08.overlap_integral(psi_tc08, psi_tc08)
assert abs(overlap_tc08 - 1.0) < 1e-6, '[TC08] 相同波函数重叠积分为1 FAILED'

# ---- TC09: 解析波函数归一化 ----
mws_tc09 = MajoranaWavefunctionSolver(length=10.0, n_sites=50, mu=0.5, t=1.0, delta=0.8)
x_tc09 = np.linspace(0, mws_tc09.L, mws_tc09.n)
psi_tc09 = mws_tc09.analytical_zero_mode_profile(x_tc09)
norm_tc09 = np.trapezoid(psi_tc09 ** 2, x_tc09)
assert abs(norm_tc09 - 1.0) < 1e-6, '[TC09] 解析波函数归一化 FAILED'

# ---- TC10: 谱展开重构精度 ----
coeffs_tc10 = mws_tc09.spectral_expansion_coefficients(psi_tc09, max_degree=10)
psi_recon_tc10 = mws_tc09.reconstruct_from_spectral(coeffs_tc10, x_tc09)
error_tc10 = np.max(np.abs(psi_tc09 - psi_recon_tc10))
assert error_tc10 < 1.0, '[TC10] 谱展开重构精度 FAILED'

# ---- TC11: 实波函数概率流为零 ----
j_tc11 = mws_tc09.compute_probability_current(psi_tc09 + 0j)
assert np.max(np.abs(j_tc11)) < 1e-10, '[TC11] 实波函数概率流为零 FAILED'

# ---- TC12: 拓扑绕数在|mu|<2t时为非零 ----
tmc_tc12 = TopologicalMarkovChain(temperature=0.05, disorder_strength=0.0, delta=0.8, t=1.0)
nu_tc12 = tmc_tc12.winding_number(mu=0.0)
assert nu_tc12 != 0, '[TC12] 拓扑绕数在|mu|<2t时为非零 FAILED'

# ---- TC13: 稳态分布归一化为1 ----
pi_tc13 = tmc_tc12.steady_state_distribution(mu=0.0)
assert abs(np.sum(pi_tc13) - 1.0) < 1e-10, '[TC13] 稳态分布归一化为1 FAILED'

# ---- TC14: 拓扑相纠缠熵接近ln2/2 ----
s_tc14 = tmc_tc12.compute_entanglement_entropy(mu=0.0, subsystem_size=10)
assert abs(s_tc14 - 0.5 * np.log(2.0)) < 0.1, '[TC14] 拓扑相纠缠熵接近ln2/2 FAILED'

# ---- TC15: 编织算符行列式为1 ----
braid_tc15 = MajoranaBraidingDynamics(num_majorana=4)
B_tc15 = braid_tc15.braid_operator(0, 1)
assert abs(np.linalg.det(B_tc15) - 1.0) < 1e-10, '[TC15] 编织算符行列式为1 FAILED'

# ---- TC16: 空编织序列返回单位矩阵 ----
U_tc16 = braid_tc15.apply_braid_sequence([])
assert np.allclose(U_tc16, np.eye(4)), '[TC16] 空编织序列返回单位矩阵 FAILED'

# ---- TC17: 耦合系统守恒量变化很小 ----
impurity_tc17 = MajoranaImpurityDynamics(alpha=1.0, beta=0.5, gamma=0.8, delta=0.3)
t_span_tc17 = np.linspace(0, 5, 51)
y_tc17 = impurity_tc17.integrate(y0=np.array([2.0, 1.0]), t_span=t_span_tc17)
H_tc17 = impurity_tc17.conserved_quantity(y_tc17)
delta_H_tc17 = np.max(H_tc17) - np.min(H_tc17)
assert delta_H_tc17 < 1.0, '[TC17] 耦合系统守恒量变化很小 FAILED'

# ---- TC18: 分段常数势求值正确 ----
pwc_tc18 = PiecewiseConstantPotential(breakpoints=np.array([0.0, 1.0, 2.0]), values=np.array([1.0, 2.0]))
assert pwc_tc18.evaluate(np.array([0.5]))[0] == 1.0, '[TC18] 分段常数势求值正确 FAILED'
assert pwc_tc18.evaluate(np.array([1.5]))[0] == 2.0, '[TC18] 分段常数势求值正确 FAILED'

# ---- TC19: FEM隐式求解无NaN ----
x_nodes_tc19 = np.linspace(0, 1, 11)
fem_tc19 = FEM1DTransport(x_nodes_tc19, diffusion_coeff=0.1)
u0_tc19 = np.zeros(11)
t_array_tc19, u_history_tc19 = fem_tc19.solve_time_dependent(u0=u0_tc19, t_final=0.1, num_steps=10, bc_left=0.0, bc_right=0.0)
assert not np.any(np.isnan(u_history_tc19)), '[TC19] FEM隐式求解无NaN FAILED'

# ---- TC20: 有效扩散系数小于原始值 ----
pwc2_tc20 = PiecewiseConstantPotential(breakpoints=np.array([0.0, 0.5, 1.0]), values=np.array([0.8, 0.0]))
d_eff_tc20 = fem_tc19.effective_diffusion_with_majorana(pwc2_tc20, energy=0.0)
assert d_eff_tc20 <= fem_tc19.D, '[TC20] 有效扩散系数小于原始值 FAILED'

# ---- TC21: Gamma函数Gamma(1)=1 ----
gl_tc21 = GammaLogCalculator()
assert abs(gl_tc21.gamma(1.0) - 1.0) < 1e-10, '[TC21] Gamma函数Gamma(1)=1 FAILED'

# ---- TC22: Gamma函数Gamma(5)=24 ----
assert abs(gl_tc21.gamma(5.0) - 24.0) < 1e-6, '[TC22] Gamma函数Gamma(5)=24 FAILED'

# ---- TC23: 费米积分F_0(0)接近ln2 ----
fi_tc23 = FermiIntegralCalculator()
f0_tc23 = fi_tc23.fermi_integral(n=0, eta=0.0)
assert abs(f0_tc23 - np.log(2.0)) < 0.01, '[TC23] 费米积分F_0(0)接近ln2 FAILED'

# ---- TC24: BTK电导非负 ----
V_tc24 = np.linspace(-1, 1, 21)
model_tc24 = BTKFittingModel(V_tc24, np.zeros_like(V_tc24))
g_tc24 = model_tc24.btk_conductance(V_tc24, delta=0.5, barrier_strength=0.3, gamma=0.05)
assert np.all(g_tc24 >= 0), '[TC24] BTK电导非负 FAILED'

# ---- TC25: Gray码编解码一致性 ----
encoder_tc25 = GrayCodeEncoder()
for n_tc25 in range(8):
    g_tc25 = encoder_tc25.binary_to_gray(n_tc25)
    n_back_tc25 = encoder_tc25.gray_to_binary(g_tc25)
    assert n_back_tc25 == n_tc25, '[TC25] Gray码编解码一致性 FAILED'

# ---- TC26: Gray码相邻Hamming距离为1 ----
gray_seq_tc26 = encoder_tc25.generate_gray_sequence(3)
for i_tc26 in range(len(gray_seq_tc26) - 1):
    d_tc26 = encoder_tc25.hamming_distance(gray_seq_tc26[i_tc26], gray_seq_tc26[i_tc26 + 1], 3)
    assert d_tc26 == 1, '[TC26] Gray码相邻Hamming距离为1 FAILED'

# ---- TC27: 正交态保真度为0 ----
psi0_tc27 = np.array([1.0, 0.0])
psi1_tc27 = np.array([0.0, 1.0])
F_tc27 = BlochSphereFidelity.fidelity(psi0_tc27, psi1_tc27)
assert abs(F_tc27) < 1e-10, '[TC27] 正交态保真度为0 FAILED'

# ---- TC28: 相同态保真度为1 ----
F_same_tc28 = BlochSphereFidelity.fidelity(psi0_tc27, psi0_tc27)
assert abs(F_same_tc28 - 1.0) < 1e-10, '[TC28] 相同态保真度为1 FAILED'

# ---- TC29: 拓扑量子码编解码一致性 ----
code_tc29 = MajoranaQuantumCode(num_majorana=6)
for logical_tc29 in range(4):
    config_tc29 = code_tc29.encode_logical_state(logical_tc29)
    decoded_tc29 = code_tc29.decode_to_logical(config_tc29)
    assert decoded_tc29 == logical_tc29, '[TC29] 拓扑量子码编解码一致性 FAILED'

# ---- TC30: CVT Lloyd迭代生成元数量正确 ----
np.random.seed(42)
cvt_tc30 = CVTSampler(n_generators=16, domain=(0.0, 1.0, 0.0, 1.0), rng_seed=42)
gens_tc30 = cvt_tc30.lloyd_iterate(num_iterations=30)
assert len(gens_tc30) == 16, '[TC30] CVT Lloyd迭代生成元数量正确 FAILED'

# ---- TC31: BZ常数积分等于区域面积 ----
np.random.seed(42)
cvt_tc31 = CVTSampler(n_generators=16, domain=(-np.pi, np.pi, -np.pi, np.pi), rng_seed=42)
kpts_tc31 = cvt_tc31.brillouin_zone_kpoints(a=1.0, num_iterations=20)
integrator_tc31 = BrillouinZoneIntegrator(kpts_tc31, cvt_tc31.domain)
result_tc31 = integrator_tc31.integrate(lambda kx, ky: 1.0)
expected_tc31 = (2.0 * np.pi) ** 2
assert abs(result_tc31 - expected_tc31) < 1e-10, '[TC31] BZ常数积分等于区域面积 FAILED'

# ---- TC32: 费米面采样返回数组 ----
fs_tc32 = integrator_tc31.fermi_surface_sampling(lambda kx, ky: -2.0 * (np.cos(kx) + np.cos(ky)), e_fermi=-1.0, tolerance=0.5)
assert isinstance(fs_tc32, np.ndarray), '[TC32] 费米面采样返回数组 FAILED'

print('\n全部 32 个测试通过!\n')
