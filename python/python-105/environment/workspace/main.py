"""
main.py
=======
量子光源纠缠态产生全数值模拟平台 —— 统一入口，零参数运行。

本程序执行以下博士级科学计算流程：
1. 构建周期性极化 KTP 晶体的 Sellmeier 色散模型。
2. 利用有限元方法 (FEM) 求解泵浦光在晶体中的非线性传播。
3. 通过后向欧拉隐式积分求解 SPDC 三波耦合刚性 ODE。
4. 计算联合光谱振幅 (JSA)，进行 Schmidt 分解。
5. 在动量空间使用格点规则、金字塔高阶求积与三角形 Monte Carlo
   三种方法积分耦合效率并交叉验证。
6. 利用球谐函数与 Jacobi 多项式展开横向模式，Bernstein 多项式逼近光谱包络。
7. 在离散参数空间（极化周期、晶体长度、温度）中搜索最优参数组合。
8. 构建级联晶体网络的邻接矩阵，分析多段结构的光子数演化。
9. 计算 Concurrence、纠缠熵、HOM 可见度、态保真度与 CHSH 参数。

运行方式：
    python main.py
"""

import numpy as np
import sys

# ---------------------------------------------------------------------------
# Import synthesized modules
# ---------------------------------------------------------------------------
from linear_solver import (
    gauss_elimination_partial_pivot, plu_decomposition,
    solve_plu, condition_number_estimate
)
from mode_analysis import (
    spherical_harmonic_basis, jacobi_polynomial,
    bernstein_basis, bernstein_approximate
)
from pump_propagation import (
    solve_pump_envelope_fem, burgers_like_pump_solution
)
from quantum_evolution import (
    spdc_derivative, backward_euler_spdc, robertson_like_conservation
)
from joint_spectrum import (
    normalized_sinc, phase_mismatch, pump_envelope_gaussian,
    phase_matching_function, compute_jsa, schmidt_decomposition_jsa
)
from phase_space_integral import (
    lattice_rule_2d_periodic, integrate_pyramid_felippa,
    triangle_monte_carlo, phase_space_coupling_efficiency
)
from parameter_optimizer import (
    gray_code_subsets, diophantine_nd_nonnegative,
    subset_sum_backtrack_all, optimize_polling_period_and_length
)
from network_coupling import (
    build_coupling_digraph, adjacency_to_transition,
    network_photon_number_evolution, transitive_closure_digraph
)
from entanglement_metrics import (
    concurrence_from_purity, von_neumann_entropy_schmidt,
    hom_visibility, state_fidelity_target, chsh_parameter
)


def print_section(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    np.random.seed(42)
    print("=" * 72)
    print("  量子光源纠缠态产生 —— 博士级全数值模拟平台")
    print("  Quantum Entangled Photon Source Simulation (QEPSS)")
    print("=" * 72)

    # =====================================================================
    # 1. Crystal Physics & Sellmeier Dispersion
    # =====================================================================
    print_section("1. 晶体色散模型与物理参数 (Crystal Physics)")

    c = 2.99792458e8  # m/s
    lambda_p0 = 405.0e-9  # pump central wavelength, m
    omega_p0 = 2.0 * np.pi * c / lambda_p0

    # PPKTP Sellmeier equations (temperature-dependent, simplified)
    def sellmeier_p(omega):
        r"""Pump (e-polarized) refractive index."""
        lam = 2.0 * np.pi * c / (omega + 1e-20)
        lam_um = np.abs(lam) * 1e6
        n2 = 2.19229 + 0.83547 / (1.0 - 0.04970 / (lam_um ** 2)) \
             - 0.01696 * lam_um ** 2
        return np.sqrt(np.maximum(1.0, n2))

    def sellmeier_s(omega):
        """Signal (o-polarized) refractive index."""
        lam = 2.0 * np.pi * c / (omega + 1e-20)
        lam_um = np.abs(lam) * 1e6
        n2 = 2.10468 + 0.89342 / (1.0 - 0.04456 / (lam_um ** 2)) \
             - 0.01020 * lam_um ** 2
        return np.sqrt(np.maximum(1.0, n2))

    def sellmeier_i(omega):
        """Idler (e-polarized) refractive index."""
        return sellmeier_p(omega)

    print(f"  泵浦中心波长: {lambda_p0*1e9:.1f} nm")
    print(f"  泵浦中心角频率: {omega_p0:.4e} rad/s")
    print(f"  泵浦折射率 n_p: {sellmeier_p(omega_p0):.6f}")
    print(f"  信号折射率 n_s: {sellmeier_s(omega_p0/2):.6f}")

    # =====================================================================
    # 2. FEM Pump Propagation
    # =====================================================================
    print_section("2. 泵浦光 FEM 非线性传播 (Pump Propagation)")

    k_p = sellmeier_p(omega_p0) * omega_p0 / c
    L_crystal = 10.0e-3  # 10 mm
    alpha_p = 0.05  # 1/m absorption
    n_nodes = 21

    def gamma_eff(z, A_p):
        # Periodic poling: chi^(2) eff = (2/pi) d33 * sign(sin(2pi z / Lambda))
        Lambda_qpm = 9.5e-6  # 9.5 um poling period
        chi2 = 14.6e-12  # m/V (KTP d33 approx)
        return (2.0 / np.pi) * chi2 * np.sign(np.sin(2.0 * np.pi * z / Lambda_qpm))

    def source_spdc(z):
        return 0.0 + 0.0j

    try:
        A_p_fem = solve_pump_envelope_fem(
            n_nodes=n_nodes,
            z_domain=(0.0, L_crystal),
            k_p=k_p,
            alpha_p=alpha_p,
            gamma_eff=gamma_eff,
            source_spdc=source_spdc,
            nonlinear_tol=1e-8,
            max_iter=30
        )
        print(f"  FEM 求解成功，节点数: {n_nodes}")
        print(f"  泵浦入口振幅 |A(0)|: {np.abs(A_p_fem[0]):.4e}")
        print(f"  泵浦出口振幅 |A(L)|: {np.abs(A_p_fem[-1]):.4e}")
        print(f"  功率衰减比: {np.abs(A_p_fem[-1]/A_p_fem[0])**2:.4f}")
    except Exception as e:
        print(f"  FEM 求解异常（可能为刚性迭代不收敛）: {e}")
        A_p_fem = np.ones(n_nodes, dtype=np.complex128) * 1.0e4

    # Burgers-like approximate solution for comparison
    z_grid = np.linspace(-1.0, 1.0, 51)
    t_grid = np.linspace(0.0, 0.5, 11)
    U_burgers = burgers_like_pump_solution(nu_eff=0.01 / np.pi, z_grid=z_grid, t_grid=t_grid)
    print(f"  Burgers 近似解计算完成，形状: {U_burgers.shape}")

    # =====================================================================
    # 3. SPDC Quantum Evolution (Stiff ODE)
    # =====================================================================
    print_section("3. SPDC 量子态刚性 ODE 演化 (Quantum Evolution)")

    gamma = np.array([1.0e6, 1.0e6, 1.0e10], dtype=np.float64)  # s^-1
    kappa0 = 5.0e6  # coupling rad/s

    def kappa_func(t):
        return kappa0 * (1.0 + 0.1 * np.sin(2.0 * np.pi * 1.0e6 * t))

    def f_noise(t):
        return np.array([0.0, 0.0, 0.0], dtype=np.complex128)

    y0 = np.array([0.0, 0.0, 1.0e3], dtype=np.complex128)
    t_span = (0.0, 5.0e-6)
    n_steps = 200

    t_ode, y_ode = backward_euler_spdc(
        y0=y0, t_span=t_span, n_steps=n_steps,
        gamma=gamma, kappa_func=kappa_func, f_noise=f_noise,
        newton_tol=1e-10, max_newton=20
    )
    C_conserved = robertson_like_conservation(y_ode)
    print(f"  后向欧拉积分完成，步数: {n_steps}")
    print(f"  初始信号幅度 |a_s(0)|: {np.abs(y_ode[0,0]):.4e}")
    print(f"  终态信号幅度 |a_s(T)|: {np.abs(y_ode[-1,0]):.4e}")
    print(f"  光子数守恒波动: {np.std(C_conserved)/np.mean(C_conserved):.4e}")

    # =====================================================================
    # 4. Joint Spectral Amplitude & Schmidt Decomposition
    # =====================================================================
    print_section("4. 联合光谱振幅与 Schmidt 分解 (Joint Spectrum)")

    n_omega = 64
    omega_s = np.linspace(omega_p0 * 0.45, omega_p0 * 0.55, n_omega)
    omega_i = np.linspace(omega_p0 * 0.45, omega_p0 * 0.55, n_omega)
    sigma_p = omega_p0 * 0.015
    Lambda_qpm = 9.5e-6

    # TODO (Hole 3): 计算联合光谱振幅 JSA，进行 Schmidt 分解，并计算纠缠度量
    # 1. 调用 compute_jsa 生成 JSA 矩阵
    # 2. 对 JSA 进行 Schmidt 分解，正确解包返回值
    # 3. 由纯度计算 Concurrence，由 Schmidt 系数计算纠缠熵
    raise NotImplementedError("Hole 3: 请计算 JSA 并完成纠缠度量评估")

    # =====================================================================
    # 5. Phase Space Integrals (Lattice / Pyramid / Triangle MC)
    # =====================================================================
    print_section("5. 动量空间高维积分 (Phase Space Integrals)")

    def kz_range(kx, ky):
        k0 = sellmeier_p(omega_p0) * omega_p0 / c
        kz_min = 0.0
        kz_max = np.sqrt(np.maximum(0.0, k0**2 - kx**2 - ky**2))
        return kz_min, kz_max

    def integrand_phase(kx, ky, kz):
        # simplified coupling efficiency kernel
        k0 = sellmeier_p(omega_p0) * omega_p0 / c
        return np.exp(-(kx**2 + ky**2) / (k0 * 0.05)**2) * np.sinc(kz / k0 * 2.0)**2

    eta_lattice = phase_space_coupling_efficiency(
        kx_max=1.0e6, ky_max=1.0e6,
        kz_func=kz_range, integrand=integrand_phase, method="lattice"
    )
    eta_pyramid = phase_space_coupling_efficiency(
        kx_max=1.0e6, ky_max=1.0e6,
        kz_func=kz_range, integrand=integrand_phase, method="pyramid"
    )
    eta_triangle = phase_space_coupling_efficiency(
        kx_max=1.0e6, ky_max=1.0e6,
        kz_func=kz_range, integrand=integrand_phase, method="triangle_mc"
    )

    print(f"  耦合效率 (Lattice rule):    {eta_lattice:.6e}")
    print(f"  耦合效率 (Pyramid Felippa): {eta_pyramid:.6e}")
    print(f"  耦合效率 (Triangle MC):     {eta_triangle:.6e}")
    print(f"  三种方法相对偏差: {np.std([eta_lattice, eta_pyramid, eta_triangle]):.4e}")

    # =====================================================================
    # 6. Mode Analysis (Spherical Harmonics, Jacobi, Bernstein)
    # =====================================================================
    print_section("6. 横向模式与光谱包络分析 (Mode Analysis)")

    theta = np.linspace(0.0, np.pi, 25)
    phi = np.linspace(0.0, 2.0 * np.pi, 50)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    Y_real, Y_imag = spherical_harmonic_basis(l_max=3, theta=Theta.flatten(), phi=Phi.flatten())
    print(f"  球谐基底维度: l_max=3, 总模式数={(3+1)**2}")

    x_jacobi = np.linspace(-1.0, 1.0, 100)
    P_jacobi = jacobi_polynomial(n=5, alpha=0.5, beta=-0.3, x=x_jacobi)
    print(f"  Jacobi 多项式 P_5^{{(0.5,-0.3)}}(0): {float(jacobi_polynomial(5,0.5,-0.3,0.0)[0]):.6f}")

    # Bernstein approximation of pump envelope
    x_bern = np.linspace(0.0, 1.0, 9)
    f_vals = np.exp(-4.0 * (x_bern - 0.5)**2)
    x_eval = np.linspace(0.0, 1.0, 101)
    y_approx = bernstein_approximate(f_vals, 0.0, 1.0, 8, x_eval)
    err_bern = np.max(np.abs(y_approx - np.exp(-4.0 * (x_eval - 0.5)**2)))
    print(f"  Bernstein 8次逼近最大误差: {err_bern:.6e}")

    # =====================================================================
    # 7. Parameter Optimization (Diophantine + Gray code + Backtracking)
    # =====================================================================
    print_section("7. 离散参数优化 (Parameter Optimization)")

    # Diophantine: find allowed (T, L) combinations from quantization steps
    a_coeff = np.array([50, 100], dtype=int)  # T step 0.5 C, L step 1 mm quantized
    b_target = 3000  # e.g. 30.0 * 100
    sols_dioph = diophantine_nd_nonnegative(a_coeff, b_target)
    print(f"  丢番图方程 a·x={b_target} 非负解数目: {sols_dioph.shape[0]}")

    # Subset sum backtrack: select pump power budget allocation
    power_budgets = np.array([100, 200, 300, 400, 500], dtype=int)
    target_power = 600
    power_solutions = subset_sum_backtrack_all(target_power, power_budgets)
    print(f"  子集和回溯: 和为 {target_power} 的子集数目: {len(power_solutions)}")

    # Gray code subsets for switching configurations
    subsets_gray, iadds_gray = gray_code_subsets(4)
    print(f"  Gray 码枚举: 4 位配置空间子集数: {len(subsets_gray)}")

    # Direct optimization over discrete parameter grid
    periods = np.array([9.0, 9.25, 9.5, 9.75, 10.0]) * 1e-6
    lengths = np.array([5.0, 10.0, 15.0, 20.0]) * 1e-3
    temperatures = np.array([20.0, 25.0, 30.0, 35.0])

    def dummy_objective(period, length, temp):
        # simplified purity proxy
        dk = 2.0 * np.pi * (1.0 / period - 1.0 / Lambda_qpm)
        phi = float(normalized_sinc(dk * length / (2.0 * np.pi))[0])
        temp_penalty = np.exp(-((temp - 25.0) / 10.0) ** 2)
        return phi * temp_penalty

    best_p, best_l, best_t, best_obj = optimize_polling_period_and_length(
        periods, lengths, temperatures, dummy_objective, max_evals=40
    )
    print(f"  最优极化周期: {best_p*1e6:.2f} um")
    print(f"  最优晶体长度: {best_l*1e3:.1f} mm")
    print(f"  最优工作温度: {best_t:.1f} C")
    print(f"  最优目标值:   {best_obj:.6f}")

    # =====================================================================
    # 8. Network Coupling (Cascade Graph)
    # =====================================================================
    print_section("8. 级联晶体网络耦合 (Network Coupling)")

    n_stages = 3
    A_net = build_coupling_digraph(n_stages, coupling_strength=0.15, phase_noise_std=0.1)
    T_net = adjacency_to_transition(A_net)
    n_initial = np.zeros(2 * n_stages, dtype=np.float64)
    n_initial[0] = 1.0e4  # pump into first stage signal channel
    source_terms = np.zeros((n_stages, 2 * n_stages), dtype=np.float64)
    source_terms[:, 0] = 500.0  # SPDC generation in signal
    source_terms[:, 1] = 500.0  # SPDC generation in idler

    n_history = network_photon_number_evolution(n_stages, n_initial, source_terms, A_net)
    C_closure = transitive_closure_digraph(A_net)
    print(f"  级联段数: {n_stages}")
    print(f"  邻接矩阵条件数估计: {condition_number_estimate(np.abs(A_net)):.4e}")
    print(f"  最终级信号光子数: {n_history[-1, 2*(n_stages-1)]:.2f}")
    print(f"  最终级闲置光子数: {n_history[-1, 2*(n_stages-1)+1]:.2f}")
    print(f"  传递闭包非零元: {np.sum(C_closure)}")

    # =====================================================================
    # 9. Entanglement Metrics
    # =====================================================================
    print_section("9. 纠缠度量综合评估 (Entanglement Metrics)")

    # HOM visibility
    delay_grid = np.linspace(-5.0e-13, 5.0e-13, 51)
    R_tau, V_hom = hom_visibility(jsa.real, jsa.imag, delay_grid, omega_s, omega_i)
    print(f"  HOM 可见度 V: {V_hom:.4f}")

    # State fidelity
    F_singlet = state_fidelity_target(jsa, target_type="singlet")
    F_triplet = state_fidelity_target(jsa, target_type="triplet")
    # Approximate fidelity from JSA self-overlap symmetry
    jsa_sym = 0.5 * (jsa - jsa.T)
    F_sym = np.abs(np.sum(np.conj(jsa) * jsa_sym)) ** 2
    print(f"  与单态保真度 F_singlet: {F_singlet:.4f}")
    print(f"  与三重态保真度 F_triplet: {F_triplet:.4f}")
    print(f"  JSA 反对称重叠保真度: {F_sym:.4f}")

    # CHSH parameter (simulated correlation matrix for near-maximal entanglement)
    corr = np.array([
        [495, 5, 470, 30],
        [5, 495, 30, 470],
        [470, 30, 480, 20],
        [30, 470, 20, 480]
    ], dtype=np.float64)
    S_chsh = chsh_parameter(corr)
    print(f"  CHSH 参数 S: {S_chsh:.4f}")
    print(f"  贝尔不等式违反: {'是' if S_chsh > 2.0 else '否'}")

    # =====================================================================
    # 10. Linear Solver Stress Test (PLU)
    # =====================================================================
    print_section("10. 线性求解器验证 (Linear Solver Verification)")

    n_test = 20
    A_test = np.random.randn(n_test, n_test) + np.eye(n_test) * 5.0
    x_exact = np.random.randn(n_test)
    b_test = A_test @ x_exact

    x_gauss = gauss_elimination_partial_pivot(A_test, b_test)
    err_gauss = np.linalg.norm(x_gauss - x_exact) / np.linalg.norm(x_exact)

    P, L, U = plu_decomposition(A_test)
    x_plu = solve_plu(P, L, U, b_test)
    err_plu = np.linalg.norm(x_plu - x_exact) / np.linalg.norm(x_exact)

    print(f"  Gauss 消元相对误差: {err_gauss:.4e}")
    print(f"  PLU 分解相对误差:   {err_plu:.4e}")
    print(f"  条件数估计:         {condition_number_estimate(A_test):.4e}")

    # =====================================================================
    # Summary
    # =====================================================================
    print_section("综合结果摘要 (Summary)")
    print(f"  Schmidt 数 K          = {K_schmidt:.4f}")
    print(f"  态纯度 P              = {purity:.6f}")
    print(f"  Concurrence C         = {C_conc:.6f}")
    print(f"  纠缠熵 S              = {S_entropy:.4f} bits")
    print(f"  HOM 可见度 V          = {V_hom:.4f}")
    print(f"  单态保真度 F          = {F_singlet:.4f}")
    print(f"  CHSH 参数 S           = {S_chsh:.4f}")
    print(f"  耦合效率 (Pyramid)    = {eta_pyramid:.6e}")
    print(f"  最优极化周期          = {best_p*1e6:.2f} um")
    print(f"  最优晶体长度          = {best_l*1e3:.1f} mm")
    print("=" * 72)
    print("  模拟完成。所有模块已通过集成验证。")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
