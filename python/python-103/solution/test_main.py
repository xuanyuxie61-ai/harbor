"""
main.py
光纤非线性脉冲传输博士级综合计算项目

统一入口：零参数运行，执行完整的仿真流程。

科学问题概述：
  本项目研究光子晶体光纤中超短光脉冲的非线性传输动力学，
  综合运用广义非线性薛定谔方程（GNLSE）、谱方法、稀疏线性求解、
  蒙特卡洛不确定性量化、MCMC参数反演等前沿数值方法，
  系统分析色散、非线性、Raman散射和ASE噪声对脉冲演化的影响。

运行方式:
  python main.py
"""

import numpy as np
import sys

# 导入所有子模块
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
    """演示Jacobi多项式谱方法在脉冲展开中的应用。"""
    print_section("1. Jacobi谱方法: 脉冲包络的谱展开")

    # 构造高斯初始脉冲
    T0 = 1e-12  # 1 ps脉冲宽度
    t = np.linspace(-5e-12, 5e-12, 512)
    A0 = np.exp(-(t / T0) ** 2 / 2.0)

    coeffs, A_recon = spectral_expand_pulse(t, A0, alpha_jac=-0.5, beta_jac=-0.5, n_modes=32)

    error = np.max(np.abs(A0 - np.real(A_recon)))
    print(f"  初始脉冲: 高斯型, T0 = {T0*1e12:.2f} ps")
    print(f"  谱展开系数数量: {coeffs.size}")
    print(f"  重构最大误差: {error:.6e}")
    print(f"  前5个系数模: {np.abs(coeffs[:5])}")

    # 演示色散算子在谱空间的作用
    beta2 = -20e-27  # s²/m
    beta3 = 0.1e-39  # s³/m
    disp_coeffs = dispersion_operator_spectral(coeffs, -0.5, -0.5, coeffs.size, beta2, beta3, 1.0)
    print(f"  色散算子作用后系数变化量: {np.linalg.norm(disp_coeffs - coeffs):.6e}")


def demo_pulse_overlap():
    """演示分段线性脉冲重叠积分。"""
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

    # Raman响应卷积
    h_R = raman_response_function(t)
    conv = raman_response_convolution(t, A1 + 0j, h_R)
    print(f"  Raman卷积峰值: {np.max(np.abs(conv)):.6e}")


def demo_fiber_geometry():
    """演示光纤截面几何建模。"""
    print_section("3. 光纤截面几何: 三角剖分与有效模场面积")

    r_core = 2e-6
    r_cladding = 62.5e-6
    nodes, triangles, boundary_flags = create_fiber_triangulation(r_core, r_cladding, n_theta=24, n_radial_core=3, n_radial_clad=4)

    # 验证边界识别
    boundary_detected = identify_boundary_nodes(triangles, nodes.shape[0])
    n_boundary = np.sum(boundary_detected)

    print(f"  芯径: {r_core*1e6:.1f} μm, 包层半径: {r_cladding*1e6:.1f} μm")
    print(f"  节点数: {nodes.shape[0]}, 三角形数: {triangles.shape[0]}")
    print(f"  边界节点数: {n_boundary}")

    # 计算有效模场面积（使用高斯近似模式场）
    w0 = 3e-6  # 模场半径
    def mode_field(x, y):
        return np.exp(-(x ** 2 + y ** 2) / (2 * w0 ** 2))

    A_eff = compute_effective_area(nodes, triangles, mode_field)
    n2 = 2.6e-20  # m²/W
    omega0 = 2.0 * np.pi * 2.99792458e8 / 1550e-9
    gamma = compute_nonlinear_coefficient(n2, omega0, A_eff)

    print(f"  有效模场面积 A_eff: {A_eff*1e12:.3f} μm²")
    print(f"  非线性系数 γ: {gamma:.6e} 1/(W·m)")


def demo_monte_carlo():
    """演示蒙特卡洛采样与球面积分。"""
    print_section("4. 蒙特卡洛采样: 参数不确定性与远场积分")

    # 超球采样（5维参数空间）
    samples = hyperball01_sample(5, 1000)
    print(f"  5维超球采样: {samples.shape[1]} 个样本")
    print(f"  样本均值范数: {np.mean(np.linalg.norm(samples, axis=0)):.4f} (理论: 5/6≈0.833)")

    # 球面积分（远场辐射模式）
    def far_field_pattern(x):
        # 简化的偶极子辐射模式
        theta = np.arccos(np.clip(x[2], -1.0, 1.0))
        return (1.0 + np.cos(theta) ** 2) / 2.0

    integral, n_eval = sphere01_quad_llm(far_field_pattern, h=0.3)
    print(f"  远场积分点数: {n_eval}")
    print(f"  球面积分结果: {integral:.6f} (理论: 4π/3·2 ≈ 8.378)")

    # 球面CVT
    n_points = 20
    xyz = np.random.randn(3, n_points)
    xyz = xyz / np.linalg.norm(xyz, axis=0)
    centroid = sphere_cvt_step(n_points, xyz)
    print(f"  球面CVT步进后点间最小距离: {np.min(np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=0)):.4f}")


def demo_noise_model():
    """演示噪声模型。"""
    print_section("5. 噪声模型: ASE与光子统计")

    # 布朗运动模拟
    traj = brownian_motion_simulation(2, 501, 1e-3, 1.0, seed=42)
    print(f"  2D布朗运动轨迹终点: ({traj[0,-1]:.4f}, {traj[1,-1]:.4f})")
    print(f"  终点位移: {np.linalg.norm(traj[:, -1]):.4f}")

    # ASE噪声
    t = np.linspace(-5e-12, 5e-12, 512)
    noise = generate_ase_noise(t, n_sp=2.0, G=10.0, h_nu=1.28e-19, bw=1e12, seed=42)
    noise_power = np.mean(np.abs(noise) ** 2)
    print(f"  ASE噪声平均功率: {noise_power:.6e} W")

    # 玻色-爱因斯坦分布
    probs, n = bose_einstein_distribution(n_avg=5.0, n_max=20)
    print(f"  玻色-爱因斯坦分布 (⟨n⟩=5): P(0)={probs[0]:.4f}, P(5)={probs[5]:.4f}")

    # Parrondo-inspired噪声耦合
    A_test = np.exp(-(t / 1e-12) ** 2 / 2.0)
    A_noisy = parrondo_inspired_noise_coupling(t, A_test + 0j, epsilon=0.005)
    corr = np.abs(np.vdot(A_test, A_noisy)) / (np.linalg.norm(A_test) * np.linalg.norm(A_noisy))
    print(f"  Parrondo噪声耦合后信号相关系数: {corr:.4f}")


def demo_sparse_solver():
    """演示GMRES稀疏求解器。"""
    print_section("6. GMRES稀疏求解: 色散算子线性系统")

    n = 64
    # 使用一个实对称正定三对角矩阵（二阶差分算子的负值）来测试GMRES
    # -u'' = f, 离散化为三对角系统
    rows = []
    cols = []
    vals = []
    h = 1.0 / (n + 1)
    for i in range(n):
        rows.append(i)
        cols.append(i)
        vals.append(2.0 / (h ** 2) + 0.1)  # 加0.1保证正定
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

    # 精确解：正弦函数
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
    """辅助函数：CRS矩阵向量乘。"""
    y = np.zeros(n, dtype=complex)
    for k in range(nz_num):
        y[ia[k]] += a[k] * x[ja[k]]
    return y


def demo_mcmc_inversion():
    """演示DREAM MCMC参数反演。"""
    print_section("7. DREAM MCMC: 光纤参数贝叶斯反演")

    # 生成合成观测数据
    true_params = np.array([1.5e-3, -20e-27, 0.1e-39, 0.2e-3, 3.0e-15])
    param_names = ["gamma (1/W/m)", "beta2 (s²/m)", "beta3 (s³/m)", "alpha (1/m)", "T_R (s)"]
    par_num = 5
    chain_num = 6
    gen_num = 300

    limits = np.array([
        [0.5e-3, -50e-27, -1.0e-39, 0.05e-3, 1.0e-15],
        [3.0e-3, 0.0, 1.0e-39, 0.5e-3, 6.0e-15]
    ])

    # 简化的似然函数（高斯）
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

    # 后验均值
    burn_in = gen_num // 3
    posterior_mean = np.mean(z[:, :, burn_in:], axis=(1, 2))

    print(f"  参数数量: {par_num}, 链数: {chain_num}, 代数: {gen_num}")
    print(f"  接受率: {rate:.4f}")
    for j in range(par_num):
        print(f"  {param_names[j]}: 真值={true_params[j]:.6e}, 估计={posterior_mean[j]:.6e}")


def demo_rootfinder():
    """演示Laguerre根查找用于光纤模式分析。"""
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
    """演示相位编码与整数搜索。"""
    print_section("9. 相位编码与整数搜索")

    # 幻方矩阵
    M = magic_matrix(5)
    magic_sum = np.sum(M[0, :])
    print(f"  5阶幻方矩阵:")
    print(M)
    print(f"  幻和: {magic_sum} (理论: 5(25+1)/2 = 65)")

    # 相位掩码
    phase = magic_phase_mask(5)
    print(f"  相位掩码范围: [{phase.min():.4f}, {phase.max():.4f}] rad")

    # 频谱循环移位
    n = 64
    spectrum = np.random.randn(n) + 1j * np.random.randn(n)
    shifted = caesar_shift_phase(spectrum, 8)
    print(f"  频谱循环移位: 能量守恒检查 {np.abs(np.sum(np.abs(spectrum)**2) - np.sum(np.abs(shifted)**2)):.6e}")

    # four_fifths搜索
    result = four_fifths_search(30, exponent=2)
    print(f"  整数搜索 (平方): {result}")

    # WDM信道搜索
    channels, fwm = wdm_channel_search()
    print(f"  WDM最优信道 (nm): {['%.2f' % c for c in channels]}")
    print(f"  FWM效率指标: {fwm:.6e}")


def demo_gnlse_propagation():
    """演示GNLSE脉冲传输仿真。"""
    print_section("10. GNLSE仿真: 超短脉冲非线性传输")

    # 参数设置
    lambda0 = 1550e-9
    T0 = 100e-15  # 100 fs
    P0 = 1000.0   # 1 kW峰值功率
    z_max = 0.5   # 0.5 m光纤
    n_steps = 500

    # 标准单模光纤参数
    beta2 = -20e-27   # s²/m
    beta3 = 0.1e-39   # s³/m
    gamma = 1.5e-3    # 1/(W·m)
    alpha = 0.2e-3    # 1/m

    # 时间网格
    T_window = 20 * T0
    n_t = 2048
    t = np.linspace(-T_window / 2, T_window / 2, n_t)
    dt = t[1] - t[0]

    # 初始脉冲：sech型（基态孤子近似）
    A0 = np.sqrt(P0) / np.cosh(t / T0)

    # 孤子参数
    N_sol, L_D, L_NL = soliton_order(A0, t, gamma, beta2, T0)
    print(f"  脉冲宽度 T0: {T0*1e15:.1f} fs")
    print(f"  峰值功率 P0: {P0:.1f} W")
    print(f"  孤子阶数 N: {N_sol:.3f}")
    print(f"  色散长度 L_D: {L_D*1e3:.3f} mm")
    print(f"  非线性长度 L_NL: {L_NL*1e3:.3f} mm")

    # ASE噪声
    h = 6.62607015e-34
    c = 2.99792458e8
    nu0 = c / lambda0
    noise = generate_ase_noise(t, n_sp=2.0, G=10.0, h_nu=h * nu0, bw=1.0 / dt, seed=123)

    # 运行SSFM
    print(f"  开始SSFM传播: z_max={z_max} m, n_steps={n_steps}")
    try:
        A_final, z_hist, A_hist = ssfm_solve(A0 + 0j, t, z_max, n_steps, alpha, beta2, beta3, gamma,
                                               lambda0=lambda0, f_R=0.18, beta4=0.0,
                                               noise_ase=noise, use_implicit=False)
    except RuntimeError as e:
        print(f"  传播错误: {e}")
        return

    # 结果分析
    width_in = temporal_width(t, A0)
    width_out = temporal_width(t, A_final)
    spec_in = spectral_width(t, A0)
    spec_out = spectral_width(t, A_final)
    energy_in = np.sum(np.abs(A0) ** 2) * dt
    energy_out = np.sum(np.abs(A_final) ** 2) * dt

    print(f"  时域宽度变化: {width_in*1e15:.1f} fs -> {width_out*1e15:.1f} fs")
    print(f"  频谱宽度变化: {spec_in*1e-12:.3f} THz -> {spec_out*1e-12:.3f} THz")
    print(f"  脉冲能量变化: {energy_in*1e12:.3f} pJ -> {energy_out*1e12:.3f} pJ")
    print(f"  能量损耗: {(1-energy_out/energy_in)*100:.2f}%")

    # 验证能量守恒（考虑损耗）
    expected_energy = energy_in * np.exp(-alpha * z_max)
    print(f"  预期能量 (含损耗): {expected_energy*1e12:.3f} pJ")


def main():
    print("=" * 70)
    print("  光纤非线性脉冲传输 — 博士级综合科学计算项目")
    print("  领域: 光学工程 — 光纤非线性脉冲传输")
    print("=" * 70)

    # 运行所有演示模块
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
    main()


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
np.random.seed(42)
from sparse_solver import ax_crs

# ---- TC01: jacobi_polynomial P0恒为1 ----
x_test = np.array([-0.5, 0.0, 0.5])
v_test = jacobi_polynomial(3, 0, -0.5, -0.5, x_test)
assert np.allclose(v_test[:, 0], 1.0), '[TC01] jacobi_polynomial P0恒为1 FAILED'

# ---- TC02: jacobi_quadrature_rule权重和等于零阶矩 ----
x_q, w_q = jacobi_quadrature_rule(8, -0.5, -0.5)
assert np.isclose(np.sum(w_q), np.pi), '[TC02] jacobi_quadrature_rule权重和 FAILED'

# ---- TC03: spectral_expand_pulse重构尺寸匹配 ----
t_test = np.linspace(-1, 1, 64)
A_test = np.exp(-(t_test / 0.3) ** 2)
coeffs, A_recon = spectral_expand_pulse(t_test, A_test, alpha_jac=-0.5, beta_jac=-0.5, n_modes=16)
assert coeffs.size == 16 and A_recon.size == t_test.size, '[TC03] spectral_expand_pulse重构尺寸匹配 FAILED'

# ---- TC04: dispersion_operator_spectral输出尺寸匹配 ----
coeffs_test = np.ones(8, dtype=complex)
disp_coeffs = dispersion_operator_spectral(coeffs_test, -0.5, -0.5, 8, -20e-27, 0.1e-39, 1.0)
assert disp_coeffs.size == 8, '[TC04] dispersion_operator_spectral输出尺寸匹配 FAILED'

# ---- TC05: triangle_area解析值验证 ----
area = triangle_area(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0]))
assert np.isclose(area, 1.0), '[TC05] triangle_area解析值验证 FAILED'

# ---- TC06: create_fiber_triangulation输出结构正确 ----
nodes, triangles, boundary_flags = create_fiber_triangulation(2e-6, 10e-6, n_theta=8, n_radial_core=2, n_radial_clad=2)
assert nodes.ndim == 2 and nodes.shape[1] == 2, '[TC06] create_fiber_triangulation输出结构正确 FAILED'
assert triangles.ndim == 2 and triangles.shape[1] == 3, '[TC06] create_fiber_triangulation输出结构正确 FAILED'
assert boundary_flags.size == nodes.shape[0], '[TC06] create_fiber_triangulation输出结构正确 FAILED'

# ---- TC07: identify_boundary_nodes检测到外边界 ----
boundary_nodes = identify_boundary_nodes(triangles, nodes.shape[0])
assert np.sum(boundary_nodes) >= 8, '[TC07] identify_boundary_nodes检测到外边界 FAILED'

# ---- TC08: compute_nonlinear_coefficient公式解析验证 ----
gamma_test = compute_nonlinear_coefficient(2.6e-20, 1.0e15, 1.0e-12)
c_light = 2.99792458e8
expected_gamma = 2.6e-20 * 1.0e15 / (c_light * 1.0e-12)
assert np.isclose(gamma_test, expected_gamma), '[TC08] compute_nonlinear_coefficient公式解析验证 FAILED'

# ---- TC09: pwl_product_integral常数函数解析验证 ----
x_pwl = np.linspace(0.0, 1.0, 5)
f_pwl = np.ones(5) * 3.0
g_pwl = np.ones(5) * 4.0
integral = pwl_product_integral(0.0, 1.0, x_pwl, f_pwl, x_pwl, g_pwl)
assert np.isclose(integral, 12.0), '[TC09] pwl_product_integral常数函数解析验证 FAILED'

# ---- TC10: pulse_inner_product交换对称性 ----
t_pwl = np.linspace(-1e-12, 1e-12, 64)
A1 = np.exp(-(t_pwl / 0.5e-12) ** 2)
A2 = np.exp(-((t_pwl - 0.2e-12) / 0.6e-12) ** 2)
ip12 = pulse_inner_product(t_pwl, A1 + 0j, A2 + 0j)
ip21 = pulse_inner_product(t_pwl, A2 + 0j, A1 + 0j)
assert np.isclose(ip12, np.conj(ip21)), '[TC10] pulse_inner_product交换对称性 FAILED'

# ---- TC11: raman_response_convolution输出尺寸匹配 ----
t_r = np.linspace(-1e-12, 3e-12, 256)
A_r = np.exp(-(t_r / 0.5e-12) ** 2) + 0j
h_R = np.ones_like(t_r)
conv = raman_response_convolution(t_r, A_r, h_R)
assert conv.size == t_r.size, '[TC11] raman_response_convolution输出尺寸匹配 FAILED'

# ---- TC12: hyperball01_sample样本在单位球内 ----
np.random.seed(42)
samples = hyperball01_sample(3, 500)
normss = np.linalg.norm(samples, axis=0)
assert np.all(normss <= 1.0 + 1e-12), '[TC12] hyperball01_sample样本在单位球内 FAILED'

# ---- TC13: sphere01_quad_llm常数函数积分接近4π ----
integral, n_eval = sphere01_quad_llm(lambda x: 1.0, h=0.5)
assert np.isclose(integral, 4.0 * np.pi, atol=0.5), '[TC13] sphere01_quad_llm常数函数积分接近4π FAILED'

# ---- TC14: brownian_motion_simulation固定种子可复现 ----
traj1 = brownian_motion_simulation(2, 101, 1e-3, 1.0, seed=42)
traj2 = brownian_motion_simulation(2, 101, 1e-3, 1.0, seed=42)
assert np.allclose(traj1, traj2), '[TC14] brownian_motion_simulation固定种子可复现 FAILED'

# ---- TC15: generate_ase_noise G<=1时返回全零 ----
t_ase = np.linspace(0.0, 1e-12, 64)
noise = generate_ase_noise(t_ase, n_sp=2.0, G=1.0, h_nu=1e-19, bw=1e12, seed=42)
assert np.all(noise == 0.0), '[TC15] generate_ase_noise G<=1时返回全零 FAILED'

# ---- TC16: bose_einstein_distribution概率和为1 ----
probs, n_arr = bose_einstein_distribution(n_avg=3.0, n_max=30)
assert np.isclose(np.sum(probs), 1.0), '[TC16] bose_einstein_distribution概率和为1 FAILED'

# ---- TC17: ax_crs稀疏矩阵向量乘解析验证 ----
a_crs = np.array([1.0, 2.0, 3.0])
ia_crs = np.array([0, 1, 1])
ja_crs = np.array([0, 0, 1])
x_vec = np.array([1.0, 1.0])
y_vec = ax_crs(a_crs, ia_crs, ja_crs, x_vec, 2, 3)
assert np.allclose(y_vec, np.array([1.0, 5.0])), '[TC17] ax_crs稀疏矩阵向量乘解析验证 FAILED'

# ---- TC18: mgmres求解三对角系统精确 ----
n = 32
h = 1.0 / (n + 1)
rows = []
cols = []
vals = []
for i in range(n):
    rows.append(i); cols.append(i); vals.append(2.0 / (h ** 2) + 0.1)
    if i > 0:
        rows.append(i); cols.append(i - 1); vals.append(-1.0 / (h ** 2))
    if i < n - 1:
        rows.append(i); cols.append(i + 1); vals.append(-1.0 / (h ** 2))
nz = len(vals)
a_crs_m = np.array(vals, dtype=float)
ia_crs_m = np.array(rows, dtype=int)
ja_crs_m = np.array(cols, dtype=int)
x_exact = np.sin(np.pi * np.linspace(0, 1, n))
rhs = ax_crs(a_crs_m, ia_crs_m, ja_crs_m, x_exact, n, nz)
x0 = np.zeros(n)
x_sol = mgmres(a_crs_m, ia_crs_m, ja_crs_m, x0, rhs, n, nz, itr_max=100, mr=20, tol_abs=1e-12, tol_rel=1e-10, verbose=False)
rel_err = np.linalg.norm(x_sol - x_exact) / np.linalg.norm(x_exact)
assert rel_err < 1e-6, '[TC18] mgmres求解三对角系统精确 FAILED'

# ---- TC19: build_dispersion_matrix_crs输出结构正确 ----
a_d, ia_d, ja_d, nz_d = build_dispersion_matrix_crs(16, 1e-15, -20e-27, 0.1e-39, beta4=0.0)
assert nz_d > 0 and a_d.size == nz_d, '[TC19] build_dispersion_matrix_crs输出结构正确 FAILED'

# ---- TC20: magic_matrix幻和正确 ----
M = magic_matrix(5)
magic_sum = np.sum(M[0, :])
assert np.allclose(np.sum(M, axis=1), magic_sum) and np.allclose(np.sum(M, axis=0), magic_sum), '[TC20] magic_matrix幻和正确 FAILED'

# ---- TC21: caesar_shift_phase能量守恒 ----
np.random.seed(42)
spectrum = np.random.randn(64) + 1j * np.random.randn(64)
shifted = caesar_shift_phase(spectrum, 8)
assert np.isclose(np.sum(np.abs(spectrum) ** 2), np.sum(np.abs(shifted) ** 2)), '[TC21] caesar_shift_phase能量守恒 FAILED'

# ---- TC22: four_fifths_search返回有效组合 ----
result = four_fifths_search(20, exponent=2)
assert result is not None and len(result) == 5, '[TC22] four_fifths_search返回有效组合 FAILED'

# ---- TC23: wdm_channel_search返回正确长度 ----
channels, fwm = wdm_channel_search(n_channels=4)
assert len(channels) == 4 and fwm >= 0.0, '[TC23] wdm_channel_search返回正确长度 FAILED'

# ---- TC24: raman_response_function因果性验证 ----
t_raman = np.linspace(-1e-12, 3e-12, 512)
h_r = raman_response_function(t_raman)
assert np.all(h_r[t_raman < 0] == 0.0), '[TC24] raman_response_function因果性验证 FAILED'

# ---- TC25: dispersion_operator零频验证 ----
omega_test = np.zeros(10)
D_test = dispersion_operator(omega_test, 0.2, -20e-27, 0.1e-39)
assert np.allclose(D_test, -0.1), '[TC25] dispersion_operator零频验证 FAILED'

# ---- TC26: soliton_order输出有限正数 ----
t_sol = np.linspace(-5e-12, 5e-12, 256)
A0_sol = np.sqrt(1000.0) / np.cosh(t_sol / 100e-15)
N_sol, L_D, L_NL = soliton_order(A0_sol, t_sol, 1.5e-3, -20e-27)
assert np.isfinite(N_sol) and N_sol > 0.0, '[TC26] soliton_order输出有限正数 FAILED'

# ---- TC27: temporal_width零输入返回0 ----
t_zero = np.zeros(64)
w_zero = temporal_width(t_zero, t_zero)
assert w_zero == 0.0, '[TC27] temporal_width零输入返回0 FAILED'

# ---- TC28: spectral_width对称脉冲非零 ----
t_sym = np.linspace(-5e-12, 5e-12, 256)
A_sym = np.exp(-(t_sym / 1e-12) ** 2)
sw = spectral_width(t_sym, A_sym)
assert sw > 0.0 and np.isfinite(sw), '[TC28] spectral_width对称脉冲非零 FAILED'

# ---- TC29: ssfm_solve基本传播输出有限 ----
t_ss = np.linspace(-2e-12, 2e-12, 256)
A0_ss = np.exp(-(t_ss / 0.5e-12) ** 2) + 0j
A_final, z_hist, A_hist = ssfm_solve(A0_ss, t_ss, 0.01, 10, 0.2e-3, -20e-27, 0.1e-39, 1.5e-3)
assert A_final.size == A0_ss.size and np.all(np.isfinite(A_final)), '[TC29] ssfm_solve基本传播输出有限 FAILED'

# ---- TC30: dream_mcmc输出尺寸正确 ----
np.random.seed(42)
logL_test = lambda p: -0.5 * np.sum(p ** 2)
logP_test = lambda p: 0.0 if np.all((p >= -1.0) & (p <= 1.0)) else -np.inf
z, fit, rate = dream_mcmc(logL_test, logP_test, par_num=2, chain_num=3, gen_num=20, limits=np.array([[-1.0, -1.0], [1.0, 1.0]]), seed=42)
assert z.shape == (2, 3, 20), '[TC30] dream_mcmc输出尺寸正确 FAILED'
assert 0.0 <= rate <= 1.0, '[TC30] dream_mcmc接受率范围 FAILED'

print('\n全部 30 个测试通过!\n')
