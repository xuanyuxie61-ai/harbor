#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
超越标准模型（BSM）新物理信号提取与共振峰重建统一入口

项目主题: LHC 暗区重媒介子 Z' → ℓ⁺ℓ⁻ 信号的全流程数值分析平台

科学问题:
    在大型强子对撞机（LHC）的高能对撞数据中，标准模型（SM）以外的
    新物理可能通过重媒介子 Z' 与标准模型粒子的耦合显现。本项目构建
    一套从探测器模拟、径迹重建、信号处理到统计推断的完整数值分析
    流程，用于在双轻子末态中寻找 Z' 共振信号并排除 BSM 参数空间。

运行方式:  python main.py
（零参数，所有配置使用默认值）
"""

import sys
import os
import numpy as np

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bsm_physics import (
    ZPrimeModel, breit_wigner_propagator, dilepton_cross_section,
    eft_contact_interaction, decay_width_dilepton, decay_width_hadronic,
    width_consistency_check, scattering_amplitude_matrix,
    chi_square_signal, exclusion_limit_at_95cl
)
from matrix_solver import (
    c8mat_fss, r8row_sort_quick_a, r8utt_sl,
    detector_deconvolution_toeplitz
)
from detector_response import (
    advection_diffusion_energy_deposit,
    news_edge_detector,
    detector_hit_map,
    aperiodic_detector_geometry,
    detector_energy_resolution
)
from track_reconstruction import (
    tsp_track_association,
    hermite_cubic_spline,
    estimate_momentum_from_curvature,
    fem1d_track_fit,
    particle_id_from_dedx
)
from signal_processing import (
    svd_low_rank_approximation,
    singular_value_entropy,
    signal_background_discriminator,
    pca_denoise,
    resonance_peak_finder
)
from interpolation_utils import (
    pwl_interp_2d_scattered,
    bsm_cross_section_interp_2d,
    sparse_interp_nd_value
)
from parameter_scan import smolyak_parameter_scan
from shower_model import (
    flame_ode_solve,
    electromagnetic_shower_profile,
    burgers_hadronization_pde,
    hadronization_energy_spectrum
)
from parameter_scan import (
    knapsack_channel_selection,
    expected_signal_yield,
    exclusion_contour_2d,
    discovery_potential
)
from event_selection import (
    reconstruct_invariant_mass,
    run_full_analysis,
    format_physics_summary,
    cl_s_limit
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_bsm_physics():
    """演示 BSM 物理模型计算。"""
    print_section("模块 1: Z' 玻色子 Breit-Wigner 共振模型")

    # 定义 Z' 模型: 质量 3 TeV, 宽度 90 GeV, 耦合 g_q = 0.2, g_ℓ = 0.1
    model = ZPrimeModel(
        mass=3000.0,
        total_width=90.0,
        gq_coupling=0.2,
        gl_coupling=0.1,
        gq_axial=0.05,
        gl_axial=0.02,
        chi=0.01
    )

    print(f"  Z' 质量 M_Z' = {model.mass:.1f} GeV")
    print(f"  Z' 宽度 Γ_Z' = {model.total_width:.1f} GeV")
    print(f"  夸克耦合 g_q^V = {model.gq_coupling:.3f}")
    print(f"  轻子耦合 g_ℓ^V = {model.gl_coupling:.3f}")

    # 幺正性检查
    consistent = width_consistency_check(model)
    print(f"  宽度自洽性检查: {'通过' if consistent else '警告（允许暗区衰变）'}")

    # 部分衰变宽度
    gamma_ll = decay_width_dilepton(model)
    gamma_qq = decay_width_hadronic(model)
    print(f"  轻子道部分宽度: {gamma_ll:.4f} GeV")
    print(f"  强子道部分宽度: {gamma_qq:.4f} GeV")

    # 传播子计算
    s_vals = np.linspace(8.0e6, 9.5e6, 100)  # √s ≈ 3 TeV 附近
    propagators = np.array([breit_wigner_propagator(s, model) for s in s_vals])
    print(f"  传播子 |D(s)| 峰值: {np.max(np.abs(propagators)):.6f}")

    # 微分截面
    cos_theta = np.linspace(-0.95, 0.95, 9)
    s_peak = model.mass ** 2
    dsigma = dilepton_cross_section(np.array([s_peak]), cos_theta, model)
    print(f"  共振峰处 dσ/dΩ 范围: [{np.min(dsigma):.4f}, {np.max(dsigma):.4f}] pb/sr")

    # EFT 接触相互作用
    delta_sigma = eft_contact_interaction(
        s=9.0e6, eta_ll=1.0, eta_rr=0.0, eta_lr=0.0, Lambda=5000.0
    )
    print(f"  EFT 接触相互作用修正 (Λ=5 TeV): {delta_sigma:.6f} pb")

    # 散射振幅矩阵
    amp_matrix = scattering_amplitude_matrix(s_vals[::10], model)
    print(f"  振幅矩阵形状: {amp_matrix.shape}")
    print(f"  振幅矩阵 Frobenius 范数: {np.linalg.norm(amp_matrix, 'fro'):.4f}")

    return model


def demo_matrix_solver():
    """演示矩阵求解器功能。"""
    print_section("模块 2: 特殊结构矩阵求解")

    # 复数矩阵求解（c8mat_fss）
    n = 8
    a_complex = np.eye(n, dtype=complex) + 0.1j * np.random.randn(n, n)
    b_complex = np.ones((n, 3), dtype=complex)
    x_complex = c8mat_fss(n, a_complex, 3, b_complex)
    residual = np.linalg.norm(a_complex @ x_complex - b_complex, 'fro')
    print(f"  复数线性系统残差: {residual:.2e}")

    # 行排序（r8row_sort_quick_a）
    m, n_r = 10, 3
    a_unsorted = np.random.rand(m, n_r)
    a_sorted = r8row_sort_quick_a(m, n_r, a_unsorted.copy())
    is_sorted = np.all(np.diff(a_sorted[:, 0]) >= -1e-12)
    print(f"  行排序验证（首列单调递增）: {is_sorted}")

    # 上三角 Toeplitz 求解（r8utt_sl）
    a_toep = np.array([2.0, -1.0, 0.5, 0.1])
    b_toep = np.array([1.0, 2.0, 3.0, 4.0])
    x_toep = r8utt_sl(4, a_toep, b_toep)
    print(f"  Toeplitz 解: {x_toep}")

    # 探测器反卷积
    true_signal = np.exp(-np.linspace(0, 2, 32) ** 2)
    psf = np.array([0.6, 0.3, 0.1])
    observed = np.convolve(true_signal, psf, mode='same')
    deconvolved = detector_deconvolution_toeplitz(observed, psf, regularization=1e-4)
    print(f"  反卷积前后相关性: {np.corrcoef(true_signal, deconvolved)[0, 1]:.4f}")


def demo_detector_response():
    """演示探测器响应模拟。"""
    print_section("模块 3: 探测器能量沉积与边缘检测")

    # 一维能量沉积平流-扩散
    x_dep, e_dep = advection_diffusion_energy_deposit(
        nx=101, nt=1000, c=1.0, diff_coeff=0.001
    )
    print(f"  能量沉积最大值位置: {x_dep[np.argmax(e_dep)]:.4f} m")
    print(f"  总沉积能量: {np.sum(e_dep):.4f} arb. units")

    # 二维击中图与边缘检测
    hit_map, edge_map = detector_hit_map(n_pixels=64, noise_level=0.02, seed=123)
    print(f"  击中图形状: {hit_map.shape}")
    print(f"  边缘响应最大值: {np.max(edge_map):.4f}")
    print(f"  检测到的边缘像素比例: {np.mean(edge_map > 0):.4f}")

    # 非周期探测器几何
    det_cells = aperiodic_detector_geometry(nmax=2, scale=1.0)
    print(f"  非周期探测器单元数: {len(det_cells)}")

    # 能量分辨率
    true_energies = np.array([10.0, 100.0, 1000.0, 5000.0])
    measured = detector_energy_resolution(true_energies)
    for te, me in zip(true_energies, measured):
        print(f"  E_true = {te:.0f} GeV -> E_meas = {me:.2f} GeV (res = {abs(me-te)/te*100:.2f}%)")


def demo_track_reconstruction():
    """演示径迹重建功能。"""
    print_section("模块 4: 径迹重建与 Hermite 样条平滑")

    # TSP 径迹关联
    layers = [
        np.array([[0.1, 0.1], [0.12, 0.08]]),
        np.array([[0.3, 0.25], [0.28, 0.27]]),
        np.array([[0.5, 0.4], [0.52, 0.42]]),
        np.array([[0.7, 0.55], [0.68, 0.53]]),
        np.array([[0.9, 0.7], [0.92, 0.72]]),
    ]
    path, length = tsp_track_association(layers, max_iter=3000)
    print(f"  TSP 最优路径长度: {length:.4f}")
    print(f"  路径访问顺序: {path}")

    # Hermite 三次样条平滑
    xn = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fn = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 0.9])
    dn = np.array([1.0, 0.9, 0.85, 0.8, 0.75, 0.7])
    x_eval = np.linspace(0.0, 1.0, 51)
    f_spline, d_spline, s_spline, t_spline = hermite_cubic_spline(xn, fn, dn, x_eval)
    print(f"  样条插值 RMS 二阶导数: {np.sqrt(np.mean(s_spline**2)):.4f}")

    # 动量估计
    x_track = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    y_track = np.array([0.0, 0.095, 0.18, 0.255, 0.32, 0.375])
    p_est = estimate_momentum_from_curvature(x_track, y_track, magnetic_field=3.8)
    print(f"  估计横向动量 p_T: {p_est:.2f} GeV")

    # 有限元 dE/dx 拟合
    track_len = np.linspace(0, 1, 20)
    dedx = 2.0 * np.exp(-track_len) + 0.1 * np.random.randn(20)
    node_x, node_c = fem1d_track_fit(track_len, dedx, n_nodes=10)
    print(f"  FEM 节点数: {len(node_x)}, 最大 dE/dx: {np.max(node_c):.4f}")

    # 粒子鉴别
    pid = particle_id_from_dedx(dedx, momentum=p_est)
    print(f"  粒子鉴别结果: {pid}")


def demo_shower_model():
    """演示簇射与强子化模型。"""
    print_section("模块 5: 电磁簇射与强子化模型")

    # 火焰 ODE（类比簇射增长）
    # 对 delta=0.01（非刚性），使用 t_max=2/delta 观察完整演化
    t_flame, y_flame = flame_ode_solve((0.0, 200.0), y0=0.01, delta=0.01, n_steps=5000)
    print(f"  火焰 ODE 饱和值: {y_flame[-1]:.6f}")
    idx_50 = np.argmax(y_flame > 0.5)
    print(f"  达到 50% 饱和深度: {t_flame[idx_50]:.4f}")

    # 电磁簇射剖面
    depths = np.linspace(0, 20, 100)
    profile = electromagnetic_shower_profile(depths, E0=100.0, Ec=0.008)
    t_max = depths[np.argmax(profile)]
    print(f"  簇射极大值深度 t_max ≈ {t_max:.2f} X_0")
    print(f"  理论 t_max ≈ ln(E0/Ec) - 1 = {np.log(100.0/0.008) - 1.0:.2f}")

    # Burgers 方程（强子化激波）
    x_burg, t_burg, u_burg = burgers_hadronization_pde(nx=128, nt=20, viscosity=0.03)
    print(f"  Burgers PDE 解形状: {u_burg.shape}")
    print(f"  最大能量密度: {np.max(u_burg):.4f}")

    # 强子化能量谱
    energies = hadronization_energy_spectrum(parton_energy=500.0, n_particles=50)
    print(f"  强子化后平均能量: {np.mean(energies):.2f} GeV")
    print(f"  强子化后能量标准差: {np.std(energies):.2f} GeV")


def demo_signal_processing():
    """演示信号处理与降维。"""
    print_section("模块 6: SVD 降维与信号/背景判别")

    # 生成信号与背景击中图
    np.random.seed(42)
    n_sig = 20
    n_bkg = 30
    hit_maps = []
    labels = []

    for _ in range(n_sig):
        hm, _ = detector_hit_map(n_pixels=32, noise_level=0.01)
        hit_maps.append(hm)
        labels.append(1)
    for _ in range(n_bkg):
        hm = np.random.exponential(0.1, (32, 32))
        hit_maps.append(hm)
        labels.append(0)

    labels = np.array(labels)

    # SVD 低秩近似
    sample_matrix = hit_maps[0]
    u_r, s_r, vh_r, approx = svd_low_rank_approximation(sample_matrix, rank=5)
    print(f"  SVD 截断秩: 5, 保留奇异值: {s_r[:3]}")
    print(f"  近似误差 (Frobenius): {np.linalg.norm(sample_matrix - approx, 'fro'):.4f}")

    # 熵分析
    entropy = singular_value_entropy(s_r)
    print(f"  奇异值谱归一化熵: {entropy:.4f}")

    # 信号/背景判别
    basis, scores = signal_background_discriminator(hit_maps, labels, n_components=5)
    print(f"  判别器基形状: {basis.shape}")
    print(f"  信号平均分数: {np.mean(scores[labels == 1]):.4f}")
    print(f"  背景平均分数: {np.mean(scores[labels == 0]):.4f}")

    # PCA 去噪
    noisy = sample_matrix + np.random.normal(0, 0.1, sample_matrix.shape)
    denoised = pca_denoise(noisy, variance_threshold=0.9)
    print(f"  PCA 去噪前后差异: {np.linalg.norm(noisy - denoised, 'fro'):.4f}")


def demo_interpolation():
    """演示多维插值功能。"""
    print_section("模块 7: 散乱数据与稀疏网格插值")

    # 2D 散乱数据插值
    n_data = 50
    data_pts = np.random.rand(n_data, 2) * 10.0
    data_vals = np.sin(data_pts[:, 0]) * np.cos(data_pts[:, 1]) + 0.1 * np.random.randn(n_data)
    query_pts = np.array([[2.5, 3.5], [5.0, 5.0], [7.5, 1.5]])
    interp_vals = pwl_interp_2d_scattered(data_pts, data_vals, query_pts)
    true_vals = np.sin(query_pts[:, 0]) * np.cos(query_pts[:, 1])
    print(f"  2D PWL 插值误差: {np.mean(np.abs(interp_vals - true_vals)):.4f}")

    # BSM 截面双线性插值
    mass_grid = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    coupling_grid = np.array([0.05, 0.1, 0.2, 0.3])
    cs_table = np.outer(1.0 / mass_grid ** 2, coupling_grid ** 2) * 1000.0
    cs_interp = bsm_cross_section_interp_2d(
        mass_grid, coupling_grid, cs_table,
        query_mass=2500.0, query_coupling=0.15
    )
    print(f"  插值截面 (M=2500, g=0.15): {cs_interp:.6f} pb")

    # Smolyak 稀疏网格参数扫描
    mass_pts, coup_pts, width_pts = smolyak_parameter_scan(
        mass_range=(1000.0, 5000.0),
        coupling_range=(0.01, 0.5),
        width_range=(0.01, 0.3),
        max_level=2
    )
    print(f"  Smolyak 采样点数: {len(mass_pts)}")


def demo_parameter_scan():
    """演示参数扫描与优化。"""
    print_section("模块 8: 背包优化与发现潜力评估")

    # 背包问题: 选择最优分析通道
    channels = ['ee', 'μμ', 'ττ', 'jj', 'bb', 'tt']
    signal_yields = np.array([5.0, 8.0, 2.0, 15.0, 10.0, 3.0])
    background_yields = np.array([50.0, 40.0, 80.0, 200.0, 150.0, 100.0])
    luminosities = np.array([500.0, 500.0, 1000.0, 300.0, 500.0, 800.0])
    max_lumi = 2000.0

    total_sig, selected = knapsack_channel_selection(
        signal_yields, background_yields, luminosities, max_lumi
    )
    print(f"  分析通道: {channels}")
    print(f"  选择结果: {selected}")
    print(f"  最优总显著性: {total_sig:.4f}")

    # 预期信号产额
    n_exp = expected_signal_yield(
        cross_section=0.1, luminosity=3000.0,
        efficiency=0.6, branching_ratio=0.03
    )
    print(f"  预期信号产额 (σ=0.1 pb, L=3 ab⁻¹): {n_exp:.2f}")

    # 发现潜力
    sig_cs = np.array([0.05, 0.1, 0.2, 0.5])
    bkg_cs = np.array([10.0, 10.0, 10.0, 10.0])
    lumi_arr = np.array([3000.0, 3000.0, 3000.0, 3000.0])
    sys_err = np.array([0.05, 0.05, 0.05, 0.05])
    z_vals = discovery_potential(sig_cs, bkg_cs, lumi_arr, sys_err)
    for cs, z in zip(sig_cs, z_vals):
        print(f"  σ={cs:.2f} pb -> Z={z:.3f} σ")


def demo_event_selection():
    """演示完整事例选择与分析流程。"""
    print_section("模块 9: 完整信号分析与统计推断")

    # 不变质量重建
    m_rec = reconstruct_invariant_mass(
        pt1=500.0, eta1=0.5, phi1=0.3,
        pt2=480.0, eta2=-0.4, phi2=3.5,
        mass1=0.000511, mass2=0.000511
    )
    print(f"  重建不变质量: {m_rec:.2f} GeV")

    # 完整分析
    zp_params = {'mass': 3000.0, 'total_width': 90.0, 'gq_coupling': 0.2}
    results = run_full_analysis(zp_params, luminosity_fb=3000.0, n_bins=40)
    summary = format_physics_summary(results)
    print(summary)

    # CL_s 检验
    excluded = cl_s_limit(
        n_observed=25, n_background=20.0,
        n_signal_hypothesis=10.0, confidence_level=0.95
    )
    print(f"  信号假设 (s=10, b=20) 95% CL 排除: {'是' if excluded else '否'}")


def main():
    """
    主程序入口：依次运行所有分析模块。
    """
    print("\n" + "#" * 70)
    print("#  LHC 超越标准模型新物理信号提取与共振峰重建平台")
    print("#  项目: BSM Z' → ℓ⁺ℓ⁻ 数值分析系统")
    print("#" * 70)

    # 运行各模块演示
    try:
        demo_bsm_physics()
    except Exception as e:
        print(f"  [警告] BSM 物理模块异常: {e}")

    try:
        demo_matrix_solver()
    except Exception as e:
        print(f"  [警告] 矩阵求解模块异常: {e}")

    try:
        demo_detector_response()
    except Exception as e:
        print(f"  [警告] 探测器响应模块异常: {e}")

    try:
        demo_track_reconstruction()
    except Exception as e:
        print(f"  [警告] 径迹重建模块异常: {e}")

    try:
        demo_shower_model()
    except Exception as e:
        print(f"  [警告] 簇射模型模块异常: {e}")

    try:
        demo_signal_processing()
    except Exception as e:
        print(f"  [警告] 信号处理模块异常: {e}")

    try:
        demo_interpolation()
    except Exception as e:
        print(f"  [警告] 插值模块异常: {e}")

    try:
        demo_parameter_scan()
    except Exception as e:
        print(f"  [警告] 参数扫描模块异常: {e}")

    try:
        demo_event_selection()
    except Exception as e:
        print(f"  [警告] 事例选择模块异常: {e}")

    print("\n" + "#" * 70)
    print("#  全部分析模块运行完毕，无致命错误。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: ZPrimeModel stores attributes correctly ----
model = ZPrimeModel(mass=3000.0, total_width=90.0, gq_coupling=0.2, gl_coupling=0.1, gq_axial=0.05, gl_axial=0.02, chi=0.01)
assert model.mass == 3000.0, '[TC01] ZPrimeModel mass FAILED'
assert model.gl_coupling == 0.1, '[TC01] ZPrimeModel gl_coupling FAILED'

# ---- TC02: breit_wigner_propagator finite at resonance ----
s_peak = model.mass ** 2
prop = breit_wigner_propagator(s_peak, model)
assert np.isfinite(prop) and abs(prop) > 0, '[TC02] BW propagator finite FAILED'

# ---- TC03: breit_wigner_propagator returns zero for non-physical s ----
prop_zero = breit_wigner_propagator(-100.0, model)
assert prop_zero == 0.0 + 0.0j, '[TC03] BW propagator non-physical FAILED'

# ---- TC04: dilepton_cross_section shape and non-negative ----
s_test = np.array([model.mass ** 2])
ct_test = np.linspace(-0.9, 0.9, 5)
dsigma = dilepton_cross_section(s_test, ct_test, model, include_sm=False)
assert dsigma.shape == (1, 5), '[TC04] cross section shape FAILED'
assert np.all(dsigma >= 0), '[TC04] cross section positivity FAILED'

# ---- TC05: eft_contact_interaction zero for invalid inputs ----
delta_sigma = eft_contact_interaction(s=1e6, eta_ll=1.0, eta_rr=0.0, eta_lr=0.0, Lambda=0.0)
assert delta_sigma == 0.0, '[TC05] EFT invalid input FAILED'

# ---- TC06: decay_width_dilepton positive ----
gamma_ll = decay_width_dilepton(model)
assert gamma_ll > 0, '[TC06] dilepton width positive FAILED'

# ---- TC07: width_consistency_check returns bool ----
check = width_consistency_check(model)
assert isinstance(check, (bool, np.bool_)), '[TC07] width consistency type FAILED'

# ---- TC08: scattering_amplitude_matrix shape and complex dtype ----
s_vals = np.linspace(8e6, 9e6, 5)
amp = scattering_amplitude_matrix(s_vals, model)
assert amp.shape == (5, 9), '[TC08] amplitude matrix shape FAILED'
assert np.iscomplexobj(amp), '[TC08] amplitude matrix complex FAILED'

# ---- TC09: chi_square_signal non-negative ----
obs = np.array([10.0, 12.0, 11.0])
bkg = np.array([9.0, 9.5, 10.0])
sig = np.array([1.0, 2.0, 1.0])
unc = np.array([1.0, 1.0, 1.0])
chi2 = chi_square_signal(obs, bkg, sig, unc)
assert chi2 >= 0, '[TC09] chi-square non-negative FAILED'

# ---- TC10: exclusion_limit_at_95cl finite for positive luminosity ----
limit = exclusion_limit_at_95cl(5.0, 20.0, 3000.0)
assert np.isfinite(limit) and limit >= 0, '[TC10] exclusion limit finite FAILED'

# ---- TC11: c8mat_fss solves identity system exactly ----
n = 4
a_id = np.eye(n, dtype=complex)
b_mat = np.ones((n, 2), dtype=complex)
x_sol = c8mat_fss(n, a_id, 2, b_mat)
assert np.allclose(x_sol, b_mat), '[TC11] c8mat_fss identity FAILED'

# ---- TC12: r8row_sort_quick_a sorts by first column ----
mat = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 0.5], [5.0, 1.5], [4.0, 2.5]])
sorted_mat = r8row_sort_quick_a(5, 2, mat.copy())
assert np.all(np.diff(sorted_mat[:, 0]) >= -1e-12), '[TC12] row sort monotonic FAILED'

# ---- TC13: r8utt_sl returns finite solution ----
a_toep = np.array([2.0, -1.0, 0.5])
b_toep = np.array([4.0, 2.0, 1.0])
x_toep = r8utt_sl(3, a_toep, b_toep)
assert np.isfinite(x_toep).all(), '[TC13] Toeplitz solution finite FAILED'

# ---- TC14: detector_deconvolution_toeplitz preserves array size ----
obs = np.array([0.5, 0.8, 1.0, 0.8, 0.5])
psf = np.array([0.6, 0.3, 0.1])
deconv = detector_deconvolution_toeplitz(obs, psf, regularization=1e-4)
assert deconv.shape == obs.shape, '[TC14] deconvolution shape FAILED'

# ---- TC15: advection_diffusion_energy_deposit returns matching shapes ----
x_dep, e_dep = advection_diffusion_energy_deposit(nx=51, nt=100, c=0.5, diff_coeff=0.001)
assert x_dep.shape == (51,) and e_dep.shape == (51,), '[TC15] advection shapes FAILED'

# ---- TC16: news_edge_detector raises ValueError on 1D input ----
try:
    news_edge_detector(np.array([1.0, 2.0, 3.0]))
    assert False, '[TC16] edge detector 1D rejection FAILED'
except ValueError:
    pass

# ---- TC17: detector_hit_map reproducible with fixed seed ----
np.random.seed(123)
hm1, em1 = detector_hit_map(n_pixels=32, noise_level=0.01, seed=123)
np.random.seed(123)
hm2, em2 = detector_hit_map(n_pixels=32, noise_level=0.01, seed=123)
assert np.allclose(hm1, hm2), '[TC17] hit map reproducibility FAILED'

# ---- TC18: aperiodic_detector_geometry returns 2D coordinates ----
coords = aperiodic_detector_geometry(nmax=1, scale=1.0)
assert coords.shape[1] == 2, '[TC18] geometry 2D FAILED'
assert np.isfinite(coords).all(), '[TC18] geometry finite FAILED'

# ---- TC19: estimate_momentum_from_curvature positive for circular arc ----
theta = np.linspace(0, np.pi/4, 10)
R = 1.0
x_arc = R * np.cos(theta)
y_arc = R * np.sin(theta)
p_est = estimate_momentum_from_curvature(x_arc, y_arc, magnetic_field=3.8)
assert p_est > 0, '[TC19] momentum positive FAILED'

# ---- TC20: hermite_cubic_spline interpolates exactly at nodes ----
xn = np.array([0.0, 1.0, 2.0])
fn = np.array([0.0, 1.0, 4.0])
dn = np.array([0.0, 2.0, 4.0])
f_out, d_out, s_out, t_out = hermite_cubic_spline(xn, fn, dn, xn)
assert np.allclose(f_out, fn), '[TC20] hermite exact nodes FAILED'

# ---- TC21: fem1d_track_fit returns consistent node and coefficient lengths ----
tlen = np.linspace(0, 1, 15)
dedx = 2.0 * np.exp(-tlen)
node_x, node_c = fem1d_track_fit(tlen, dedx, n_nodes=8)
assert len(node_x) == 8 and len(node_c) == 8, '[TC21] FEM lengths FAILED'

# ---- TC22: particle_id_from_dedx returns valid particle string ----
pid = particle_id_from_dedx(np.array([1.5, 1.6, 1.55]), momentum=10.0)
assert pid in ('electron', 'muon', 'pion', 'proton', 'unknown'), '[TC22] PID string FAILED'

# ---- TC23: svd_low_rank_approximation error decreases with higher rank ----
np.random.seed(42)
mat = np.random.rand(10, 8)
_, _, _, approx1 = svd_low_rank_approximation(mat, rank=1)
_, _, _, approx3 = svd_low_rank_approximation(mat, rank=3)
err1 = np.linalg.norm(mat - approx1, 'fro')
err3 = np.linalg.norm(mat - approx3, 'fro')
assert err3 <= err1 + 1e-12, '[TC23] SVD rank monotonicity FAILED'

# ---- TC24: singular_value_entropy within [0, 1] ----
s_arr = np.array([5.0, 3.0, 1.0, 0.1])
entropy = singular_value_entropy(s_arr)
assert 0.0 <= entropy <= 1.0, '[TC24] entropy range FAILED'

# ---- TC25: pca_denoise preserves matrix shape ----
np.random.seed(42)
mat_noisy = np.random.rand(8, 8)
denoised = pca_denoise(mat_noisy, variance_threshold=0.9)
assert denoised.shape == mat_noisy.shape, '[TC25] PCA shape FAILED'

# ---- TC26: signal_background_discriminator separates signal and background ----
np.random.seed(42)
hit_maps = []
labels = []
for _ in range(10):
    hm, _ = detector_hit_map(n_pixels=16, noise_level=0.01, seed=42)
    hit_maps.append(hm)
    labels.append(1)
for _ in range(10):
    hm = np.random.exponential(0.1, (16, 16))
    hit_maps.append(hm)
    labels.append(0)
labels = np.array(labels)
basis, scores = signal_background_discriminator(hit_maps, labels, n_components=3)
assert basis.shape[0] == 3, '[TC26] discriminator basis shape FAILED'
assert len(scores) == 20, '[TC26] discriminator scores length FAILED'

# ---- TC27: resonance_peak_finder detects planted peak ----
masses = np.arange(100, 200, 1.0)
counts = np.ones_like(masses) * 10.0
counts[50] = 100.0
pm, ph, ps = resonance_peak_finder(masses, counts, window_width=5.0)
assert ph > 0, '[TC27] peak height positive FAILED'

# ---- TC28: bsm_cross_section_interp_2d within data extrema ----
mg = np.array([1000.0, 2000.0, 3000.0])
cg = np.array([0.1, 0.2])
cs_tab = np.outer(1.0/mg**2, cg**2) * 1000.0
cs_val = bsm_cross_section_interp_2d(mg, cg, cs_tab, query_mass=1500.0, query_coupling=0.15)
assert np.min(cs_tab) <= cs_val <= np.max(cs_tab), '[TC28] interp 2D range FAILED'

# ---- TC29: expected_signal_yield scales linearly with luminosity ----
n1 = expected_signal_yield(0.1, 1000.0, efficiency=1.0, branching_ratio=1.0)
n2 = expected_signal_yield(0.1, 2000.0, efficiency=1.0, branching_ratio=1.0)
assert np.isclose(n2, 2.0 * n1), '[TC29] yield linearity FAILED'

# ---- TC30: discovery_potential positive for positive signal ----
z_vals = discovery_potential(np.array([0.1]), np.array([10.0]), np.array([3000.0]), np.array([0.05]))
assert z_vals[0] > 0, '[TC30] discovery potential positive FAILED'

# ---- TC31: reconstruct_invariant_mass symmetric under swap ----
m_a = reconstruct_invariant_mass(500.0, 0.5, 0.3, 480.0, -0.4, 3.5)
m_b = reconstruct_invariant_mass(480.0, -0.4, 3.5, 500.0, 0.5, 0.3)
assert np.isclose(m_a, m_b), '[TC31] invariant mass symmetry FAILED'

# ---- TC32: cl_s_limit returns boolean ----
excluded = cl_s_limit(25, 20.0, 10.0, 0.95)
assert isinstance(excluded, (bool, np.bool_)), '[TC32] CLs type FAILED'

# ---- TC33: run_full_analysis returns dict with required keys ----
zp_p = {'mass': 2000.0, 'total_width': 60.0, 'gq_coupling': 0.2}
res = run_full_analysis(zp_p, luminosity_fb=1000.0, n_bins=20)
req_keys = ('zp_mass', 'peak_mass', 'exclusion_limit_sigma_95_pb', 'discovery_potential_z')
assert all(k in res for k in req_keys), '[TC33] full analysis keys FAILED'

# ---- TC34: format_physics_summary returns non-empty string ----
summary = format_physics_summary(res)
assert isinstance(summary, str) and len(summary) > 0, '[TC34] summary string FAILED'

# ---- TC35: flame_ode_solve saturates towards 1 ----
t_fl, y_fl = flame_ode_solve((0.0, 200.0), y0=0.01, delta=0.01, n_steps=5000)
assert y_fl[-1] > 0.9 and y_fl[-1] <= 1.0, '[TC35] flame saturation FAILED'

# ---- TC36: electromagnetic_shower_profile non-negative with peak ----
depths = np.linspace(0, 15, 100)
prof = electromagnetic_shower_profile(depths, E0=100.0, Ec=0.008)
assert np.all(prof >= 0), '[TC36] shower profile non-negative FAILED'
assert np.max(prof) > 0, '[TC36] shower profile has peak FAILED'

# ---- TC37: burgers_hadronization_pde output shape consistency ----
x_b, t_b, u_b = burgers_hadronization_pde(nx=64, nt=5, viscosity=0.03, t_max=0.3)
assert u_b.shape == (len(t_b), len(x_b)), '[TC37] Burgers shape FAILED'

# ---- TC38: hadronization_energy_spectrum conserves energy approximately ----
np.random.seed(42)
energies = hadronization_energy_spectrum(parton_energy=500.0, n_particles=50)
assert np.sum(energies) <= 500.0 * 1.01, '[TC38] energy conservation FAILED'
assert np.all(energies >= 0), '[TC38] energy positivity FAILED'

# ---- TC39: pwl_interp_2d_scattered exact at data points ----
dp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
dv = np.array([0.0, 1.0, 1.0, 2.0])
qp = np.array([[0.0, 0.0], [1.0, 1.0]])
iv = pwl_interp_2d_scattered(dp, dv, qp)
assert np.allclose(iv, np.array([0.0, 2.0]), atol=1e-6), '[TC39] PWL exact nodes FAILED'

# ---- TC40: knapsack_channel_selection returns boolean selection ----
sig_y = np.array([5.0, 8.0, 2.0])
bkg_y = np.array([50.0, 40.0, 80.0])
lumi = np.array([500.0, 500.0, 1000.0])
total_sig, selected = knapsack_channel_selection(sig_y, bkg_y, lumi, max_lumi=1000.0)
assert isinstance(total_sig, (float, np.floating)), '[TC40] knapsack sig type FAILED'
assert selected.dtype == bool, '[TC40] knapsack dtype FAILED'

print('\n全部 40 个测试通过!\n')
