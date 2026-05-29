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

    # === HOLE 3 (Part A) START ===
    # TODO: 实现马约拉纳零能模识别结果的处理逻辑
    #
    # 说明：solver.identify_majorana_zero_modes() 的返回值格式已从
    # 元组 (modes_u, modes_v) 改为字典格式。同时，BdG求解器中的
    # MZM识别科学逻辑（Hole 1）和拓扑相分析中的绕数公式（Hole 2）
    # 均已被挖空，需要协同修复。
    #
    # 需要完成的任务：
    # 1. 调用 solver.identify_majorana_zero_modes(eigvals, eigvecs, energy_tol=1e-6)
    # 2. 根据新的字典返回值格式（包含 'u_modes', 'v_modes', 'count' 等键）提取数据
    # 3. 计算并输出MZM的数量、边界波函数最大值和反参与率IPR
    raise NotImplementedError("Hole 3 Part A: 请实现零能模识别结果的处理逻辑")
    # === HOLE 3 (Part A) END ===

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

    # === HOLE 3 (Part B) START ===
    # TODO: 实现Z₂绕数拓扑不变量的计算与输出
    #
    # 说明：tmc.winding_number() 的返回值类型已从 int 改为 float。
    # 同时，绕数计算公式本身（Hole 2）已被挖空，需要协同修复。
    # 此处需要适配新的返回值类型并正确格式化输出。
    #
    # 需要完成的任务：
    # 1. 对给定的 mu 列表 [-3.0, -2.0, ..., 3.0] 调用 tmc.winding_number(mu)
    # 2. 适配 float 返回值，正确格式化输出绕数 ν 和相位判断
    # 3. 注意：绕数 ν 可能为 0.0（平庸相）或 ±1.0（非平庸相）
    raise NotImplementedError("Hole 3 Part B: 请实现绕数计算与输出适配逻辑")
    # === HOLE 3 (Part B) END ===

    # 稳态分布
    print("\n[稳态分布 π = (π_trivial, π_topological, π_critical)]")
    for mu in [-2.5, -1.0, 0.0, 1.0, 2.5]:
        pi = tmc.steady_state_distribution(mu)
        print(f"  μ = {mu:+4.1f} -> π = ({pi[0]:.4f}, {pi[1]:.4f}, {pi[2]:.4f})")

    # === HOLE 3 (Part C) START ===
    # TODO: 实现纠缠熵计算与绕数输出的协同适配
    #
    # 说明：tmc.winding_number() 返回 float 类型（原 int），
    # 需要同步适配格式化输出。纠缠熵计算依赖于绕数结果。
    #
    # 需要完成的任务：
    # 1. 对 mu 列表 [-0.5, 0.5, 3.0] 计算纠缠熵和绕数
    # 2. 适配 float 类型的绕数返回值，正确格式化输出
    raise NotImplementedError("Hole 3 Part C: 请实现纠缠熵与绕数的协同输出")
    # === HOLE 3 (Part C) END ===

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
