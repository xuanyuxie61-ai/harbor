
import numpy as np
import sys


from jacobi_spectral import jacobi_polynomial, jacobi_quadrature_rule, spectral_expand_pulse, dispersion_operator_spectral
from pulse_overlap import pwl_product_integral, pulse_nonlinear_overlap, pulse_inner_product, raman_response_convolution
from fiber_geometry import (create_fiber_triangulation, identify_boundary_nodes, triangle_integrand_gauss,
                            compute_effective_area, compute_nonlinear_coefficient, triangle_area)
from monte_carlo_sampler import hyperball01_sample, sphere01_quad_llm, sphere_cvt_step, monte_carlo_uncertainty_quantification, tp_to_xyz
from noise_model import (brownian_motion_simulation, generate_ase_noise, bose_einstein_distribution,
                         photon_number_fluctuation, parrondo_inspired_noise_coupling)
from sparse_solver import mgmres, build_dispersion_matrix_crs
from mcmc_inversion import dream_mcmc
from rootfinder import zero_laguerre, find_fiber_mode_roots
from phase_coding import (caesar_shift_phase, magic_matrix, magic_phase_mask, apply_phase_mask_to_pulse,
                          four_fifths_search, wdm_channel_search)
from gnlse_solver import (ssfm_solve, raman_response_function, dispersion_operator, nonlinear_operator,
                          soliton_order, spectral_width, temporal_width)


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_jacobi_spectral():
    print_section("1. Jacobi谱方法: 脉冲包络的谱展开")


    T0 = 1e-12
    t = np.linspace(-5e-12, 5e-12, 512)
    A0 = np.exp(-(t / T0) ** 2 / 2.0)

    coeffs, A_recon = spectral_expand_pulse(t, A0, alpha_jac=-0.5, beta_jac=-0.5, n_modes=32)

    error = np.max(np.abs(A0 - np.real(A_recon)))
    print(f"  初始脉冲: 高斯型, T0 = {T0*1e12:.2f} ps")
    print(f"  谱展开系数数量: {coeffs.size}")
    print(f"  重构最大误差: {error:.6e}")
    print(f"  前5个系数模: {np.abs(coeffs[:5])}")


    beta2 = -20e-27
    beta3 = 0.1e-39
    disp_coeffs = dispersion_operator_spectral(coeffs, -0.5, -0.5, coeffs.size, beta2, beta3, 1.0)
    print(f"  色散算子作用后系数变化量: {np.linalg.norm(disp_coeffs - coeffs):.6e}")


def demo_pulse_overlap():
    print_section("2. 分段线性重叠积分: XPM相互作用强度")

    t = np.linspace(-5e-12, 5e-12, 256)
    A1 = np.exp(-(t / 1e-12) ** 2 / 2.0)
    A2 = np.exp(-((t - 0.5e-12) / 1.2e-12) ** 2 / 2.0)

    overlap = pulse_nonlinear_overlap(t, A1, A2)
    inner = pulse_inner_product(t, A1 + 0j, A2 + 0j)

    print(f"  脉冲1: 高斯, T0 = 1.0 ps")
    print(f"  脉冲2: 高斯, T0 = 1.2 ps, 时延 0.5 ps")
    print(f"  非线性重叠积分 (|A1|²|A2|²): {overlap:.6e} W²·s")
    print(f"  复内积 <A1|A2>: {inner:.6e}")


    h_R = raman_response_function(t)
    conv = raman_response_convolution(t, A1 + 0j, h_R)
    print(f"  Raman卷积峰值: {np.max(np.abs(conv)):.6e}")


def demo_fiber_geometry():
    print_section("3. 光纤截面几何: 三角剖分与有效模场面积")

    r_core = 2e-6
    r_cladding = 62.5e-6
    nodes, triangles, boundary_flags = create_fiber_triangulation(r_core, r_cladding, n_theta=24, n_radial_core=3, n_radial_clad=4)


    boundary_detected = identify_boundary_nodes(triangles, nodes.shape[0])
    n_boundary = np.sum(boundary_detected)

    print(f"  芯径: {r_core*1e6:.1f} μm, 包层半径: {r_cladding*1e6:.1f} μm")
    print(f"  节点数: {nodes.shape[0]}, 三角形数: {triangles.shape[0]}")
    print(f"  边界节点数: {n_boundary}")


    w0 = 3e-6
    def mode_field(x, y):
        return np.exp(-(x ** 2 + y ** 2) / (2 * w0 ** 2))

    A_eff = compute_effective_area(nodes, triangles, mode_field)
    n2 = 2.6e-20
    omega0 = 2.0 * np.pi * 2.99792458e8 / 1550e-9
    gamma = compute_nonlinear_coefficient(n2, omega0, A_eff)

    print(f"  有效模场面积 A_eff: {A_eff*1e12:.3f} μm²")
    print(f"  非线性系数 γ: {gamma:.6e} 1/(W·m)")


def demo_monte_carlo():
    print_section("4. 蒙特卡洛采样: 参数不确定性与远场积分")


    samples = hyperball01_sample(5, 1000)
    print(f"  5维超球采样: {samples.shape[1]} 个样本")
    print(f"  样本均值范数: {np.mean(np.linalg.norm(samples, axis=0)):.4f} (理论: 5/6≈0.833)")


    def far_field_pattern(x):

        theta = np.arccos(np.clip(x[2], -1.0, 1.0))
        return (1.0 + np.cos(theta) ** 2) / 2.0

    integral, n_eval = sphere01_quad_llm(far_field_pattern, h=0.3)
    print(f"  远场积分点数: {n_eval}")
    print(f"  球面积分结果: {integral:.6f} (理论: 4π/3·2 ≈ 8.378)")


    n_points = 20
    xyz = np.random.randn(3, n_points)
    xyz = xyz / np.linalg.norm(xyz, axis=0)
    centroid = sphere_cvt_step(n_points, xyz)
    print(f"  球面CVT步进后点间最小距离: {np.min(np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=0)):.4f}")


def demo_noise_model():
    print_section("5. 噪声模型: ASE与光子统计")


    traj = brownian_motion_simulation(2, 501, 1e-3, 1.0, seed=42)
    print(f"  2D布朗运动轨迹终点: ({traj[0,-1]:.4f}, {traj[1,-1]:.4f})")
    print(f"  终点位移: {np.linalg.norm(traj[:, -1]):.4f}")


    t = np.linspace(-5e-12, 5e-12, 512)
    noise = generate_ase_noise(t, n_sp=2.0, G=10.0, h_nu=1.28e-19, bw=1e12, seed=42)
    noise_power = np.mean(np.abs(noise) ** 2)
    print(f"  ASE噪声平均功率: {noise_power:.6e} W")


    probs, n = bose_einstein_distribution(n_avg=5.0, n_max=20)
    print(f"  玻色-爱因斯坦分布 (⟨n⟩=5): P(0)={probs[0]:.4f}, P(5)={probs[5]:.4f}")


    A_test = np.exp(-(t / 1e-12) ** 2 / 2.0)
    A_noisy = parrondo_inspired_noise_coupling(t, A_test + 0j, epsilon=0.005)
    corr = np.abs(np.vdot(A_test, A_noisy)) / (np.linalg.norm(A_test) * np.linalg.norm(A_noisy))
    print(f"  Parrondo噪声耦合后信号相关系数: {corr:.4f}")


def demo_sparse_solver():
    print_section("6. GMRES稀疏求解: 色散算子线性系统")

    n = 64


    rows = []
    cols = []
    vals = []
    h = 1.0 / (n + 1)
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(2.0 / (h ** 2) + 0.1)
        if i > 0:
            rows.append(i)
            cols.append(i - 1)
            vals.append(-1.0 / (h ** 2))
        if i < n - 1:
            rows.append(i)
            cols.append(i + 1)
            vals.append(-1.0 / (h ** 2))

    nz_num = len(vals)
    a_crs = np.array(vals, dtype=float)
    ia_crs = np.array(rows, dtype=int)
    ja_crs = np.array(cols, dtype=int)


    x_exact = np.sin(np.pi * np.linspace(0, 1, n))
    rhs = ax_crs_for_demo(a_crs, ia_crs, ja_crs, x_exact, n, nz_num)

    x0 = np.zeros(n)
    x_sol = mgmres(a_crs, ia_crs, ja_crs, x0, rhs, n, nz_num,
                   itr_max=100, mr=30, tol_abs=1e-12, tol_rel=1e-10, verbose=False)

    error = np.linalg.norm(x_sol - x_exact) / np.linalg.norm(x_exact)
    residual = np.linalg.norm(rhs - ax_crs_for_demo(a_crs, ia_crs, ja_crs, x_sol, n, nz_num))
    print(f"  矩阵维数: {n}, 非零元: {nz_num}")
    print(f"  GMRES相对误差: {error:.6e}")
    print(f"  残差范数: {residual:.6e}")


def ax_crs_for_demo(a, ia, ja, x, n, nz_num):
    y = np.zeros(n, dtype=complex)
    for k in range(nz_num):
        y[ia[k]] += a[k] * x[ja[k]]
    return y


def demo_mcmc_inversion():
    print_section("7. DREAM MCMC: 光纤参数贝叶斯反演")


    true_params = np.array([1.5e-3, -20e-27, 0.1e-39, 0.2e-3, 3.0e-15])
    param_names = ["gamma (1/W/m)", "beta2 (s²/m)", "beta3 (s³/m)", "alpha (1/m)", "T_R (s)"]
    par_num = 5
    chain_num = 6
    gen_num = 300

    limits = np.array([
        [0.5e-3, -50e-27, -1.0e-39, 0.05e-3, 1.0e-15],
        [3.0e-3, 0.0, 1.0e-39, 0.5e-3, 6.0e-15]
    ])


    def log_likelihood(p):
        diff = (p - true_params) / (0.1 * np.abs(true_params))
        return -0.5 * np.sum(diff ** 2)

    def log_prior(p):
        for j in range(par_num):
            if p[j] < limits[0, j] or p[j] > limits[1, j]:
                return -np.inf
        return 0.0

    z, fit, rate = dream_mcmc(log_likelihood, log_prior, par_num, chain_num, gen_num,
                              limits, pair_num=2, cr_num=3, jumpstep=10,
                              gr_threshold=1.5, printstep=1000, seed=42)


    burn_in = gen_num // 3
    posterior_mean = np.mean(z[:, :, burn_in:], axis=(1, 2))

    print(f"  参数数量: {par_num}, 链数: {chain_num}, 代数: {gen_num}")
    print(f"  接受率: {rate:.4f}")
    for j in range(par_num):
        print(f"  {param_names[j]}: 真值={true_params[j]:.6e}, 估计={posterior_mean[j]:.6e}")


def demo_rootfinder():
    print_section("8. Laguerre根查找: 光纤模式传播常数")

    n_core = 1.45
    n_clad = 1.444
    a_core = 4e-6
    lambda0 = 1550e-9
    k0 = 2.0 * np.pi / lambda0
    V = a_core * k0 * np.sqrt(n_core ** 2 - n_clad ** 2)

    print(f"  归一化频率 V: {V:.4f}")
    print(f"  芯径 a: {a_core*1e6:.1f} μm")

    for l_mode in [0, 1]:
        roots, betas = find_fiber_mode_roots(V, l_mode, n_core, n_clad, n_roots=3)
        print(f"  LP_{l_mode}m 模式:")
        for idx, (u, beta) in enumerate(zip(roots, betas)):
            n_eff = beta / k0
            print(f"    m={idx+1}: u={u:.6f}, n_eff={n_eff:.6f}")


def demo_phase_coding():
    print_section("9. 相位编码与整数搜索")


    M = magic_matrix(5)
    magic_sum = np.sum(M[0, :])
    print(f"  5阶幻方矩阵:")
    print(M)
    print(f"  幻和: {magic_sum} (理论: 5(25+1)/2 = 65)")


    phase = magic_phase_mask(5)
    print(f"  相位掩码范围: [{phase.min():.4f}, {phase.max():.4f}] rad")


    n = 64
    spectrum = np.random.randn(n) + 1j * np.random.randn(n)
    shifted = caesar_shift_phase(spectrum, 8)
    print(f"  频谱循环移位: 能量守恒检查 {np.abs(np.sum(np.abs(spectrum)**2) - np.sum(np.abs(shifted)**2)):.6e}")


    result = four_fifths_search(30, exponent=2)
    print(f"  整数搜索 (平方): {result}")


    channels, fwm = wdm_channel_search()
    print(f"  WDM最优信道 (nm): {['%.2f' % c for c in channels]}")
    print(f"  FWM效率指标: {fwm:.6e}")


def demo_gnlse_propagation():
    print_section("10. GNLSE仿真: 超短脉冲非线性传输")


    lambda0 = 1550e-9
    T0 = 100e-15
    P0 = 1000.0
    z_max = 0.5
    n_steps = 500


    beta2 = -20e-27
    beta3 = 0.1e-39
    gamma = 1.5e-3
    alpha = 0.2e-3


    T_window = 20 * T0
    n_t = 2048
    t = np.linspace(-T_window / 2, T_window / 2, n_t)
    dt = t[1] - t[0]


    A0 = np.sqrt(P0) / np.cosh(t / T0)


    N_sol, L_D, L_NL = soliton_order(A0, t, gamma, beta2, T0)
    print(f"  脉冲宽度 T0: {T0*1e15:.1f} fs")
    print(f"  峰值功率 P0: {P0:.1f} W")
    print(f"  孤子阶数 N: {N_sol:.3f}")
    print(f"  色散长度 L_D: {L_D*1e3:.3f} mm")
    print(f"  非线性长度 L_NL: {L_NL*1e3:.3f} mm")


    h = 6.62607015e-34
    c = 2.99792458e8
    nu0 = c / lambda0
    noise = generate_ase_noise(t, n_sp=2.0, G=10.0, h_nu=h * nu0, bw=1.0 / dt, seed=123)






    raise NotImplementedError("Hole 3: demo_gnlse_propagation SSFM调用与结果分析待实现")


def main():
    print("=" * 70)
    print("  光纤非线性脉冲传输 — 博士级综合科学计算项目")
    print("  领域: 光学工程 — 光纤非线性脉冲传输")
    print("=" * 70)


    demo_jacobi_spectral()
    demo_pulse_overlap()
    demo_fiber_geometry()
    demo_monte_carlo()
    demo_noise_model()
    demo_sparse_solver()
    demo_mcmc_inversion()
    demo_rootfinder()
    demo_phase_coding()
    demo_gnlse_propagation()

    print("\n" + "=" * 70)
    print("  所有计算模块运行完毕。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
