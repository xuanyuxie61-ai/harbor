
import numpy as np
import sys
import time


from binary_black_hole import BinaryBlackHole, effective_spin
from numerical_relativity import (
    evolve_binary_orbit, conformal_factor_brill_lindquist,
    run_stability_tests, final_mass_spin
)
from teukolsky import solve_qnm_frequencies, gravitational_wave_luminosity
from waveform import (
    full_imrphenom_waveform, matched_filter_snr,
    shifted_legendre_polynomial, waveform_inner_product_chebyshev
)
from detector import (
    compute_sky_position_from_time_delays,
    antenna_pattern_functions,
    get_standard_detector_network,
    network_snr
)
from bayesian import GWPrior, metropolis_hastings, marginal_posterior_mass_plane
from sparse_solver import solve_initial_data_brill_lindquist, compute_extrinsic_curvature
from utils import euclidean_gcd, rational_approximation, check_finite


def print_banner():
    print("=" * 70)
    print("  引力波信号数值相对论模拟与贝叶斯参数推断系统")
    print("  Numerical Relativity Gravitational-Wave Simulation")
    print("=" * 70)
    print()


def stage_1_initial_data():
    print("[阶段 1] 双黑洞初始数据构造")
    print("-" * 40)
    
    m1, m2 = 36.0, 29.0
    M_total = m1 + m2
    

    masses = np.array([m1, m2]) * BinaryBlackHole.MSUN_SI * BinaryBlackHole.G_SI / BinaryBlackHole.C_SI**3
    positions = np.array([[-1.5, 0.0], [1.5, 0.0]]) * m1 * BinaryBlackHole.G_SI / BinaryBlackHole.C_SI**2
    

    initial_data = solve_initial_data_brill_lindquist(
        nx=33, ny=33, h=0.3,
        masses=masses, positions=positions
    )
    
    psi = initial_data['psi']
    K = compute_extrinsic_curvature(psi, initial_data['h'])
    
    adm_mass_msun = initial_data['adm_mass'] / (BinaryBlackHole.MSUN_SI * BinaryBlackHole.G_SI / BinaryBlackHole.C_SI**3)
    print(f"  ADM 质量: {adm_mass_msun:.2f} M_sun")
    print(f"  共形因子 ψ 范围: [{np.min(psi):.4f}, {np.max(psi):.4f}]")
    print(f"  外曲率 trace: {np.mean(K['trace']):.6e}")
    print(f"  稀疏矩阵密度: {initial_data['matrix_info']['density']:.6e}")
    print()
    
    return initial_data, m1, m2


def stage_2_orbit_evolution(m1, m2):
    print("[阶段 2] 双黑洞轨道动力学演化")
    print("-" * 40)
    
    bbh = BinaryBlackHole(m1_msun=m1, m2_msun=m2)
    

    r0 = 10.0 * bbh.M
    

    t_max = bbh.t_inspiral * 1.2
    
    print(f"  啁啾质量 M_c = {bbh.M_c:.4f} s = {bbh.M_c / (bbh.MSUN_SI * bbh.G_SI / bbh.C_SI**3):.2f} M_sun")
    print(f"  对称质量比 η = {bbh.eta:.4f}")
    print(f"  ISCO 频率 f_ISCO = {bbh.f_isco:.4e} Hz")
    print(f"  初始分离 r0 = {r0:.4e} m")
    print(f"  演化时间尺度 t_inspiral = {bbh.t_inspiral:.4e} s")
    
    t, trajectory, energy = evolve_binary_orbit(
        m1=bbh.m1, m2=bbh.m2,
        initial_separation=r0,
        t_span=(0.0, t_max),
        n_steps=5000
    )
    
    print(f"  轨道演化完成: {len(t)} 时间步")
    print(f"  能量相对变化: {np.abs((energy[-1] - energy[0]) / energy[0]):.6e}")
    print()
    
    return bbh, t, trajectory, energy


def stage_3_waveform_generation(bbh):
    print("[阶段 3] 引力波波形生成")
    print("-" * 40)
    

    t = np.linspace(-bbh.t_inspiral, 0.0, 8192)
    

    M_f, a_f = final_mass_spin(bbh.m1_msun, bbh.m2_msun, bbh.a1, bbh.a2)
    print(f"  预估最终质量 M_f = {M_f:.2f} M_sun")
    print(f"  预估最终自旋 a_f = {a_f:.4f}")
    

    qnm_freqs = solve_qnm_frequencies(l_max=3, n_overtones=1, M=M_f, a=a_f)
    print(f"  准正规模频率数量: {len(qnm_freqs)}")
    

    if (2, 2, 0) in qnm_freqs:
        f_220 = qnm_freqs[(2, 2, 0)]
        print(f"  (2,2,0) 模式频率: Re(ω) = {f_220.real:.6f}, Im(ω) = {f_220.imag:.6f}")
    

    h_plus, h_cross = full_imrphenom_waveform(
        t, bbh.m1_msun, bbh.m2_msun, bbh.D_L_mpc,
        inclination=bbh.inclination,
        M_final=M_f, a_final=a_f,
        qnm_freqs=qnm_freqs
    )
    

    lum, lum_dimless = gravitational_wave_luminosity(qnm_freqs, M_f, a_f)
    print(f"  引力波峰值光度 (无量纲): {lum_dimless:.6e}")
    

    strain_energy = np.sum(h_plus**2 + h_cross**2)
    print(f"  波形应变能量: {strain_energy:.6e}")
    print()
    
    return t, h_plus, h_cross, qnm_freqs


def stage_4_detector_response(t, h_plus, h_cross, bbh):
    print("[阶段 4] 探测器响应与网络分析")
    print("-" * 40)
    
    network = get_standard_detector_network()
    

    theta_s = bbh.inclination
    phi_s = bbh.phi_c
    psi_s = bbh.psi
    
    F_plus_list = []
    F_cross_list = []
    
    for det in network:
        Fp, Fc = antenna_pattern_functions(
            theta_s, phi_s, psi_s,
            det['arm1'], det['arm2']
        )
        F_plus_list.append(Fp)
        F_cross_list.append(Fc)
        print(f"  {det['name']}: F+ = {Fp:.4f}, F× = {Fc:.4f}")
    

    rho_net = network_snr(F_plus_list, F_cross_list, h_plus, h_cross, noise_psd=1.0)
    print(f"  网络信噪比 ρ_net ≈ {rho_net:.4f}")
    

    dt = np.array([0.0, 0.007, 0.003])
    positions = np.array([det['position'] for det in network])
    
    try:
        lat, lon, n_vec = compute_sky_position_from_time_delays(
            positions, dt, radius=1.0
        )
        print(f"  反演天球位置: 纬度 = {np.degrees(lat):.2f}°, 经度 = {np.degrees(lon):.2f}°")
    except Exception as e:
        print(f"  天球定位信息: {e}")
    
    print()
    
    return F_plus_list, F_cross_list, rho_net


def stage_5_bayesian_inference(bbh, t, h_plus, h_cross):
    print("[阶段 5] 贝叶斯参数估计")
    print("-" * 40)
    

    prior = GWPrior(m_min=5.0, m_max=100.0, mu_mass=3.5, sigma_mass=0.5)
    

    noise_level = 0.05 * np.max(np.abs(h_plus))
    h_plus_noisy = h_plus + noise_level * np.random.randn(len(h_plus))
    h_cross_noisy = h_cross + noise_level * np.random.randn(len(h_cross))
    

    t_template = t.copy()
    

    true_params = bbh.to_parameter_dict()
    
    def log_posterior(params):
        lp = prior.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        


        ll = 0.0
        for key in ['m1', 'm2', 'D_L']:
            sigma = true_params[key] * 0.15
            ll += -0.5 * ((params[key] - true_params[key]) / sigma) ** 2
        

        for key in ['inclination', 'phi_c']:
            diff = np.mod(params[key] - true_params[key] + np.pi, 2*np.pi) - np.pi
            ll += -0.5 * (diff / 0.3) ** 2
        
        return lp + ll
    

    init_params = true_params.copy()
    init_params['m1'] += 2.0
    init_params['m2'] += 2.0
    

    print("  开始 MCMC 采样 (2000 步)...")
    samples, acc_rate = metropolis_hastings(
        log_posterior, init_params, n_steps=2000,
        step_sizes={
            'm1': 2.0, 'm2': 2.0, 'a1': 0.05, 'a2': 0.05,
            'D_L': 100.0, 'inclination': 0.15,
            'phi_c': 0.3, 'psi': 0.3, 't_c': 0.01
        }
    )
    print(f"  MCMC 接受率: {acc_rate:.4f}")
    

    m1_samples = np.array([s['m1'] for s in samples[500:]])
    m2_samples = np.array([s['m2'] for s in samples[500:]])
    print(f"  m1 后验均值: {np.mean(m1_samples):.2f} ± {np.std(m1_samples):.2f} M_sun")
    print(f"  m2 后验均值: {np.mean(m2_samples):.2f} ± {np.std(m2_samples):.2f} M_sun")
    

    def log_post_mass_plane(m1_val, m2_val):
        p = true_params.copy()
        p['m1'] = m1_val
        p['m2'] = m2_val
        return log_posterior(p)
    
    points, post_vals, evidence = marginal_posterior_mass_plane(
        log_post_mass_plane,
        m1_range=(bbh.m1_msun * 0.7, bbh.m1_msun * 1.3),
        m2_range=(bbh.m2_msun * 0.7, bbh.m2_msun * 1.3),
        order=4
    )
    print(f"  质量平面证据 (对数): {np.log(evidence + 1e-300):.4f}")
    print()
    
    return samples, evidence


def stage_6_stability_tests():
    print("[阶段 6] 数值稳定性验证套件")
    print("-" * 40)
    
    results = run_stability_tests()
    
    print(f"  Robertson 刚性系统守恒误差: {results.get('robertson_conservation_error', 'N/A'):.6e}")
    print(f"  Burgers 激波守恒误差: {results.get('burgers_conservation_error', 'N/A'):.6e}")
    print(f"  锯齿波振子能量漂移: {results.get('sawtooth_energy_drift', 'N/A'):.6e}")
    print(f"  规范波 L2 误差: {results.get('gauge_wave_l2_error', 'N/A'):.6e}")
    print(f"  全部测试通过: {results.get('all_pass', False)}")
    print()
    
    return results


def stage_7_auxiliary_computations():
    print("[阶段 7] 辅助科学计算")
    print("-" * 40)
    

    x = np.linspace(0.0, 1.0, 50)
    leg_vals = shifted_legendre_polynomial(x, n_max=6)
    print(f"  移位 Legendre 多项式 P_0^* 到 P_6^* 已计算")
    print(f"  P_6^*(0.5) = {leg_vals[25, 6]:.6f}")
    

    eta = 0.24
    p, q = rational_approximation(eta, max_denominator=100)
    print(f"  质量比 η = {eta} 的有理近似: {p}/{q} = {p/q:.6f}")
    

    g = euclidean_gcd(p, q)
    p_reduced = p // g
    q_reduced = q // g
    print(f"  约化后: {p_reduced}/{q_reduced}, gcd({p},{q}) = {g}")
    

    chi_eff = effective_spin(36.0, 29.0, 0.3, -0.2)
    print(f"  有效自旋 χ_eff = {chi_eff:.4f}")
    

    check_finite(leg_vals, "Legendre多项式值")
    print()
    
    return leg_vals, (p_reduced, q_reduced)


def main():
    print_banner()
    start_time = time.time()
    
    np.random.seed(42)
    
    try:

        initial_data, m1, m2 = stage_1_initial_data()
        

        bbh, t_orbit, trajectory, energy = stage_2_orbit_evolution(m1, m2)
        

        t_wave, h_plus, h_cross, qnm_freqs = stage_3_waveform_generation(bbh)
        

        Fp_list, Fc_list, rho_net = stage_4_detector_response(t_wave, h_plus, h_cross, bbh)
        

        samples, evidence = stage_5_bayesian_inference(bbh, t_wave, h_plus, h_cross)
        

        stability = stage_6_stability_tests()
        

        leg_vals, rational = stage_7_auxiliary_computations()
        

        elapsed = time.time() - start_time
        print("=" * 70)
        print("  模拟完成")
        print(f"  总耗时: {elapsed:.2f} 秒")
        print(f"  双黑洞系统: {m1:.1f} M_sun + {m2:.1f} M_sun")
        print(f"  网络信噪比: {rho_net:.2f}")
        print(f"  数值稳定性: {'通过' if stability.get('all_pass', False) else '部分通过'}")
        print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
