"""
main.py
超连续谱产生（Supercontinuum Generation）全波形数值仿真系统

统一入口，零参数运行。

项目概述:
  本程序基于广义非线性薛定谔方程（GNLSE），对光子晶体光纤（PCF）中的
  超连续谱产生过程进行全波形数值仿真。涵盖以下核心物理过程:
    - 高阶色散（至 6 阶 Taylor 展开）
    - 自相位调制（SPM）
    - 自陡峭（Self-steepening）
    - 延迟 Raman 响应（Blow-Wood 模型）
    - 光纤几何结构（六边形空气孔晶格）
    - 自适应时间/频率采样（CVT 优化）
    - 输出端 Fresnel 衍射
    - 多模耦合动力学
    - 插值稳定性分析（Lebesgue 常数）

运行方式:
    python main.py

输出:
    在标准输出打印超连续谱特征参数、光谱带宽、孤子阶数等。
"""

import numpy as np
import sys

# 导入各科学计算模块
from pcf_geometry import (
    pcf_air_holes_geometry,
    effective_mode_area,
    nonlinear_coefficient,
    pcf_transverse_triangle_grid,
    circle_segment_area_from_angle,
    circle_segment_centroid_from_angle,
)
from mesh_utils import (
    pcf_triangular_mesh,
    tri_mesh_edge_neighbors,
    mesh_bounding_box,
)
from dispersion_calculus import (
    beta_from_sellmeier,
    dispersion_taylor_coefficients,
    chebyshev_coefficients,
    chebyshev_interpolant,
    cubic_spline_coefficients,
    cubic_spline_eval,
    lebesgue_constant,
    chebyshev_zeros,
    sellmeier_equation_silica,
)
from adaptive_grid import (
    adaptive_cvt_grid,
    spectral_boundary_detect,
    log_spaced_grid,
)
from nonlinear_response import (
    raman_response_blow_wood,
    nonlinear_response_full,
    self_steepening_factor,
)
from gnlse_propagator import (
    ssfm_propagate,
    sech_pulse,
    gaussian_pulse,
    arclength_parameterization,
    adaptive_step_size_estimate,
)
from fresnel_output import (
    fresnel_integrals,
    fresnel_diffraction_1d,
    fresnel_number,
)
from multimode_coupling import (
    mode_coupling_matrix,
    xpm_coefficients,
    multimode_propagation_verlet,
    mode_power_orbits,
)
from spectrum_analysis import (
    spectral_bandwidth,
    spectral_flatness,
    dispersion_length,
    nonlinear_length,
    soliton_order,
    fourier_limit_duration,
    spectral_snr,
)


def print_header(title: str) -> None:
    """打印带分隔线的标题。"""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_pcf_geometry_analysis() -> dict:
    """
    任务1: 光子晶体光纤几何分析
    融合原项目 184_circle_segment + 1305_triangle_grid
    """
    print_header("任务1: 光子晶体光纤几何结构分析")
    pitch = 3.0e-6      # 晶格常数 3 um
    hole_radius = 0.9e-6
    n_rings = 3
    geo = pcf_air_holes_geometry(pitch, n_rings, hole_radius)
    a_eff = effective_mode_area(pitch, hole_radius, n_rings)
    gamma = nonlinear_coefficient(pitch, hole_radius, n_rings)
    print(f"  空气孔数量: {geo['n_holes']}")
    print(f"  填充率: {geo['filling_fraction']:.4f}")
    print(f"  等效模场面积: {a_eff*1e12:.4f} um^2")
    print(f"  非线性系数 gamma: {gamma:.4f} W^{-1} m^{-1}")
    # 圆段几何示例
    theta = np.pi / 3.0
    seg_area = circle_segment_area_from_angle(hole_radius, theta)
    seg_centroid = circle_segment_centroid_from_angle(hole_radius, theta)
    print(f"  圆段面积(theta=pi/3): {seg_area:.4e} m^2")
    print(f"  圆段质心偏移: {seg_centroid[0]:.4e} m")
    # 三角形网格
    tg = pcf_transverse_triangle_grid(pitch, n_rings, subdivisions=5)
    print(f"  代表性三角形单元网格点数: {tg.shape[1]}")
    return {
        "pitch": pitch,
        "hole_radius": hole_radius,
        "n_rings": n_rings,
        "a_eff": a_eff,
        "gamma": gamma,
        "filling_fraction": geo["filling_fraction"],
    }


def run_mesh_topology() -> dict:
    """
    任务2: 网格拓扑与邻接关系
    融合原项目 1292_tri_surface_display + 1168_stla_to_tri_surface_fast + 1239_tet_mesh_tet_neighbors
    """
    print_header("任务2: 网格拓扑与邻接关系")
    pitch = 3.0e-6
    hole_radius = 0.9e-6
    n_rings = 2
    nodes, elements = pcf_triangular_mesh(pitch, hole_radius, n_rings)
    bbox_min, bbox_max = mesh_bounding_box(nodes)
    boundary_edges = tri_mesh_edge_neighbors(nodes, elements)
    print(f"  节点数: {nodes.shape[0]}")
    print(f"  三角形数: {elements.shape[0]}")
    print(f"  包围盒: [{bbox_min[0]:.3e}, {bbox_min[1]:.3e}] -> [{bbox_max[0]:.3e}, {bbox_max[1]:.3e}]")
    print(f"  边界边数: {boundary_edges.shape[0]}")
    # 四面体邻接示例（简化四面体）
    tetra_order = 4
    tetra_num = 5
    np.random.seed(7)
    tetra_node = np.random.randint(1, 20, size=(tetra_order, tetra_num))
    from mesh_utils import tet_mesh_tet_neighbors
    neighbors = tet_mesh_tet_neighbors(tetra_order, tetra_num, tetra_node)
    print(f"  四面体邻接示例 (5 tets): 邻居矩阵 shape={neighbors.shape}")
    return {
        "n_nodes": nodes.shape[0],
        "n_triangles": elements.shape[0],
        "n_boundary_edges": boundary_edges.shape[0],
    }


def run_dispersion_analysis() -> dict:
    """
    任务3: 色散曲线插值与稳定性分析
    融合原项目 159_chebyshev + 1388_vanloan + 658_lebesgue
    """
    print_header("任务3: 色散曲线高精度插值与 Lebesgue 稳定性")
    wavelength_um = np.linspace(0.5, 2.0, 100)
    beta = beta_from_sellmeier(wavelength_um)
    omega0_rad = 2.0 * np.pi * 2.99792458e8 / (1.55e-6)
    omega_rad = 2.0 * np.pi * 2.99792458e8 / (wavelength_um * 1e-6)
    # 按 omega 递增排序（omega 随波长增加而递减，需翻转）
    sort_idx = np.argsort(omega_rad)
    omega_rad = omega_rad[sort_idx]
    beta = beta[sort_idx]
    # Chebyshev 插值
    a, b = float(omega_rad.min()), float(omega_rad.max())
    c_cheb = chebyshev_coefficients(a, b, 12, lambda x: np.interp(x, omega_rad, beta))
    x_test = np.linspace(a, b, 200)
    beta_cheb = chebyshev_interpolant(c_cheb, a, b, x_test)
    beta_true = np.interp(x_test, omega_rad, beta)
    # 使用相对误差评估
    cheb_err = np.max(np.abs(beta_cheb - beta_true) / (np.abs(beta_true) + 1.0))
    print(f"  Chebyshev 插值最大相对误差: {cheb_err:.4e}")
    # 三次样条
    a_spl, b_spl, c_spl, d_spl = cubic_spline_coefficients(omega_rad, beta)
    beta_spline = cubic_spline_eval(x_test, a_spl, b_spl, c_spl, d_spl, omega_rad)
    spline_err = np.max(np.abs(beta_spline - beta_true) / (np.abs(beta_true) + 1.0))
    print(f"  三次样条最大相对误差: {spline_err:.4e}")
    # Lebesgue 常数
    x_nodes_cheb = chebyshev_zeros(12, a, b)
    x_dense = np.linspace(a, b, 1000)
    lambda_cheb = lebesgue_constant(12, x_nodes_cheb, x_dense)
    x_nodes_eq = np.linspace(a, b, 12)
    lambda_eq = lebesgue_constant(12, x_nodes_eq, x_dense)
    print(f"  Chebyshev 节点 Lebesgue 常数: {lambda_cheb:.4f}")
    print(f"  等距节点 Lebesgue 常数: {lambda_eq:.4f}")
    print(f"  -> Chebyshev 稳定性优势: {lambda_eq / lambda_cheb:.2f}x")
    # 提取 Taylor 系数
    beta_coeffs = dispersion_taylor_coefficients(omega_rad, beta, omega0_rad, order=6)
    print(f"  beta2 (GVD): {beta_coeffs[2]:.4e} s^2/m")
    print(f"  beta3 (TOD): {beta_coeffs[3]:.4e} s^3/m")
    print(f"  beta4: {beta_coeffs[4]:.4e} s^4/m")
    return {
        "beta_coeffs": beta_coeffs,
        "omega0": omega0_rad,
        "cheb_err": cheb_err,
        "spline_err": spline_err,
        "lambda_cheb": lambda_cheb,
        "lambda_eq": lambda_eq,
    }


def run_adaptive_sampling() -> dict:
    """
    任务4: 自适应 CVT 采样与光谱边界检测
    融合原项目 238_cvt + 585_image_sample
    """
    print_header("任务4: 自适应采样点优化（CVT）与光谱边界检测")
    # 模拟高斯型光谱功率密度
    omega_grid = np.linspace(-50e12, 50e12, 1000)
    power = np.exp(-(omega_grid / 10e12) ** 2) + 0.1 * np.exp(-((omega_grid - 30e12) / 5e12) ** 2)
    # 光谱边界检测
    left, right = spectral_boundary_detect(power, omega_grid, threshold_db=-20.0)
    print(f"  -20 dB 光谱边界: [{left/1e12:.2f}, {right/1e12:.2f}] THz")
    # CVT 自适应采样
    density = lambda x: np.interp(x, omega_grid, power) + 1e-6
    cvt_points = adaptive_cvt_grid(32, density, (left, right),
                                    n_samples=3000, it_max=20)
    print(f"  CVT 优化采样点数: {len(cvt_points)}")
    print(f"  采样点范围: [{cvt_points[0]/1e12:.2f}, {cvt_points[-1]/1e12:.2f}] THz")
    return {
        "cvt_points": cvt_points,
        "spectral_left": left,
        "spectral_right": right,
    }


def run_gnlse_simulation(disp_data: dict, geo_data: dict) -> dict:
    """
    任务5: GNLSE 分步傅里叶传播仿真
    融合原项目 1135_spiral_pde_movie + 016_arclength
    """
    print_header("任务5: GNLSE 超连续谱产生数值仿真（SSFM）")
    # 物理参数
    gamma = geo_data["gamma"]
    beta_coeffs = disp_data["beta_coeffs"]
    omega0 = disp_data["omega0"]
    alpha = 0.2e-3  # 损耗 0.2 dB/km -> 1/m
    # 脉冲参数
    T0 = 50e-15      # 50 fs
    P0 = 10e3        # 10 kW 峰值功率
    z_target = 0.02   # 2 cm 光纤
    # 时间窗口
    n_t = 2 ** 12
    T_window = 10e-12  # 10 ps
    t = np.linspace(-T_window / 2, T_window / 2, n_t)
    # 初始脉冲
    A0 = sech_pulse(t, T0, P0, C=0.0)
    print(f"  初始脉宽: {T0*1e15:.1f} fs, 峰值功率: {P0/1e3:.1f} kW")
    print(f"  传播距离: {z_target*1e2:.1f} cm")
    # 弧长参数化检查
    S, s_param = arclength_parameterization(A0, t)
    print(f"  初始脉冲弧长: {S:.4e}")
    # SSFM 传播
    # TODO Hole 3: 调用 ssfm_propagate 执行 GNLSE 分步傅里叶传播仿真
    # 需要正确传递以下参数:
    #   - 初始脉冲: A0, 时间窗口: t
    #   - 传播距离: z_target, 损耗: alpha, 非线性系数: gamma
    #   - 色散系数: beta_coeffs (来自 disp_data), 中心频率: omega0
    #   - Raman 参数: f_R, tau1, tau2
    #   - 步进参数: dz_initial, n_z_records, use_symmetrized
    raise NotImplementedError("Hole 3: 请调用 ssfm_propagate 完成 GNLSE 传播仿真")
    # 最终光谱
    spec_final = spec_z[-1, :]
    omega = 2.0 * np.pi * np.fft.fftfreq(n_t, t[1] - t[0])
    # 排序到自然顺序
    omega_sorted = np.fft.fftshift(omega)
    spec_sorted = np.fft.fftshift(spec_final)
    # 光谱分析
    bw_fwhm = spectral_bandwidth(omega_sorted, spec_sorted, method="fwhm")
    bw_20db = spectral_bandwidth(omega_sorted, spec_sorted, method="twenty_db")
    flatness = spectral_flatness(spec_sorted)
    print(f"  最终 -3 dB 带宽: {bw_fwhm/1e12:.2f} THz")
    print(f"  最终 -20 dB 带宽: {bw_20db/1e12:.2f} THz")
    print(f"  光谱平坦度: {flatness:.4f}")
    # 孤子参数
    beta2 = beta_coeffs[2]
    N_sol = soliton_order(beta2, gamma, T0, P0)
    L_D = dispersion_length(T0, beta2)
    L_NL = nonlinear_length(gamma, P0)
    print(f"  孤子阶数 N: {N_sol:.2f}")
    print(f"  色散长度 L_D: {L_D*1e2:.3f} cm")
    print(f"  非线性长度 L_NL: {L_NL*1e2:.3f} cm")
    return {
        "z_out": z_out,
        "A_z": A_z,
        "spec_z": spec_z,
        "omega": omega_sorted,
        "spec_final": spec_sorted,
        "N_soliton": N_sol,
        "L_D": L_D,
        "L_NL": L_NL,
        "bandwidth_fwhm": bw_fwhm,
        "bandwidth_20db": bw_20db,
        "flatness": flatness,
    }


def run_fresnel_output(gnlse_data: dict, geo_data: dict) -> dict:
    """
    任务6: Fresnel 衍射输出场计算
    融合原项目 448_fresnel
    """
    print_header("任务6: 输出端 Fresnel 衍射分析")
    # 取最终时域包络（峰值附近切片近似为孔径场）
    A_final = gnlse_data["A_z"][-1, :]
    t = np.linspace(-5e-12, 5e-12, len(A_final))
    # 1D Fresnel 衍射（将时间映射为空间）
    x_aperture = t * 2.99792458e8  # 光速映射（示意性）
    wavelength = 1.55e-6
    z = 1e-3  # 1 mm
    x_obs = np.linspace(-2e-3, 2e-3, 200)
    from fresnel_output import fresnel_diffraction_1d
    E_out = fresnel_diffraction_1d(A_final, x_aperture, x_obs, wavelength, z)
    intensity_out = np.abs(E_out) ** 2
    print(f"  衍射距离: {z*1e3:.1f} mm")
    print(f"  波长: {wavelength*1e9:.1f} nm")
    print(f"  输出峰值强度: {np.max(intensity_out):.4e} a.u.")
    # Fresnel 数
    a = np.max(np.abs(x_aperture))
    N_f = fresnel_number(a, wavelength, z)
    print(f"  Fresnel 数: {N_f:.4f} ({'近场' if N_f > 1 else '远场'})")
    # Fresnel 积分示例
    C_val, S_val = fresnel_integrals(2.0)
    print(f"  Fresnel 积分 C(2.0)={C_val:.6f}, S(2.0)={S_val:.6f}")
    return {
        "x_obs": x_obs,
        "intensity_out": intensity_out,
        "fresnel_number": N_f,
    }


def run_multimode_coupling(gnlse_data: dict, geo_data: dict) -> dict:
    """
    任务7: 多模耦合轨道动力学
    融合原项目 345_exm/orbits
    """
    print_header("任务7: 多模耦合轨道动力学")
    n_modes = 3
    delta_beta = 100.0  # 1/m
    kappa = 10.0        # 1/m
    K = mode_coupling_matrix(n_modes, delta_beta, kappa)
    gamma_mm = geo_data["gamma"]
    overlap = np.ones((n_modes, n_modes)) * 0.5
    np.fill_diagonal(overlap, 1.0)
    chi = xpm_coefficients(n_modes, gamma_mm, overlap)
    # 初始能量分布
    P_total = 1e3  # W
    A0 = np.sqrt(P_total / n_modes) * np.ones(n_modes, dtype=complex)
    A0[0] *= np.sqrt(2.0)  # 基模占优
    z_target = 0.05
    dz = 1e-4
    z_array, A_hist = multimode_propagation_verlet(A0, z_target, K, chi, dz)
    P_hist = mode_power_orbits(A_hist)
    print(f"  模式数: {n_modes}")
    print(f"  传播距离: {z_target*1e2:.1f} cm")
    print(f"  初始功率分布: {[f'{p:.1f}' for p in P_hist[0, :]]} W")
    print(f"  最终功率分布: {[f'{p:.1f}' for p in P_hist[-1, :]]} W")
    print(f"  总功率守恒误差: {abs(np.sum(P_hist[-1,:]) - np.sum(P_hist[0,:])) / np.sum(P_hist[0,:]) * 100:.4f}%")
    return {
        "z_array": z_array,
        "P_hist": P_hist,
    }


def run_spectrum_summary(gnlse_data: dict) -> dict:
    """
    任务8: 光谱综合分析
    """
    print_header("任务8: 超连续谱综合分析")
    omega = gnlse_data["omega"]
    spec = gnlse_data["spec_final"]
    # 多种带宽指标
    bw_rms = spectral_bandwidth(omega, spec, method="rms")
    print(f"  RMS 带宽: {bw_rms/1e12:.4f} THz")
    # 傅里叶极限
    fourier_lim = fourier_limit_duration(bw_rms / (2.0 * np.pi), pulse_shape="sech")
    print(f"  对应傅里叶极限脉宽: {fourier_lim*1e15:.2f} fs")
    # SNR
    mid = len(spec) // 2
    snr = spectral_snr(spec, (mid - 100, mid + 100))
    print(f"  光谱 SNR: {snr:.2f} dB")
    print(f"  光谱平坦度: {gnlse_data['flatness']:.4f}")
    print(f"  孤子阶数: {gnlse_data['N_soliton']:.2f}")
    return {
        "bw_rms": bw_rms,
        "fourier_limit": fourier_lim,
        "snr_db": snr,
    }


def main():
    """统一入口函数。"""
    print("\n" + "#" * 70)
    print("#  超连续谱产生数值仿真系统")
    print("#  Supercontinuum Generation Simulation in Photonic Crystal Fiber")
    print("#" * 70 + "\n")

    np.random.seed(42)

    # 任务1: 光纤几何
    geo_data = run_pcf_geometry_analysis()

    # 任务2: 网格拓扑
    mesh_data = run_mesh_topology()

    # 任务3: 色散分析
    disp_data = run_dispersion_analysis()

    # 任务4: 自适应采样
    samp_data = run_adaptive_sampling()

    # 任务5: GNLSE 仿真（核心）
    gnlse_data = run_gnlse_simulation(disp_data, geo_data)

    # 任务6: Fresnel 输出
    fresnel_data = run_fresnel_output(gnlse_data, geo_data)

    # 任务7: 多模耦合
    mode_data = run_multimode_coupling(gnlse_data, geo_data)

    # 任务8: 综合分析
    summary = run_spectrum_summary(gnlse_data)

    print("\n" + "=" * 70)
    print("  仿真完成。所有计算模块已成功执行。")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
