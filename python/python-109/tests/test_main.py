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
    hexagonal_lattice_points,
    triangle_grid,
)
from mesh_utils import (
    pcf_triangular_mesh,
    tri_mesh_edge_neighbors,
    mesh_bounding_box,
    tet_mesh_tet_neighbors,
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
    dispersion_operator_fft,
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
    z_out, A_z, spec_z = ssfm_propagate(
        A0, t, z_target, alpha, gamma, beta_coeffs, omega0,
        f_R=0.18, tau1=12.2e-15, tau2=32.0e-15,
        dz_initial=1e-4, n_z_records=50, use_symmetrized=True
    )
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


# 在测试文件中，main() 由 TC44 调用，此处不再重复执行
if __name__ == "__main__":
    pass

# ================================================================
# 测试用例（44个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: circle_segment_area_from_angle 半圆面积 ----
np.random.seed(42)
r = 1.0e-6
theta = np.pi
area = circle_segment_area_from_angle(r, theta)
assert abs(area - np.pi * r * r / 2.0) < 1e-20, '[TC01] 半圆面积计算 FAILED'

# ---- TC02: circle_segment_area_from_angle 零角度面积 ----
area0 = circle_segment_area_from_angle(r, 0.0)
assert abs(area0) < 1e-30, '[TC02] 零角度面积 FAILED'

# ---- TC03: circle_segment_centroid_from_angle 质心偏移为正 ----
centroid = circle_segment_centroid_from_angle(r, np.pi / 2.0)
assert centroid[0] > 0.0, '[TC03] 圆段质心偏移应为正 FAILED'
assert centroid[1] == 0.0, '[TC03] 圆段质心y坐标应为0 FAILED'

# ---- TC04: hexagonal_lattice_points 1环孔数 ----
pts = hexagonal_lattice_points(3.0e-6, 1)
assert pts.shape[0] == 7, '[TC04] 1环六边形晶格应有7个孔 FAILED'
assert pts.shape[1] == 2, '[TC04] 晶格点应为2维 FAILED'

# ---- TC05: pcf_air_holes_geometry 填充率在合理范围 ----
geo = pcf_air_holes_geometry(3.0e-6, 2, 0.9e-6)
assert 0.0 <= geo["filling_fraction"] <= 1.0, '[TC05] 填充率应在[0,1] FAILED'
assert geo["n_holes"] > 0, '[TC05] 空气孔数应大于0 FAILED'

# ---- TC06: triangle_grid 输出维度正确 ----
t = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, np.sqrt(3.0)/2.0]])
grid = triangle_grid(4, t)
expected_ng = (4 + 1) * (4 + 2) // 2
assert grid.shape == (2, expected_ng), '[TC06] triangle_grid输出维度 FAILED'

# ---- TC07: effective_mode_area 为正 ----
a_eff = effective_mode_area(3.0e-6, 0.9e-6, 2)
assert a_eff > 0.0, '[TC07] 有效模场面积应大于0 FAILED'

# ---- TC08: nonlinear_coefficient 物理正值 ----
gamma_val = nonlinear_coefficient(3.0e-6, 0.9e-6, 2)
assert gamma_val > 0.0, '[TC08] 非线性系数应为正 FAILED'

# ---- TC09: mesh_bounding_box 包围盒单调性 ----
nodes = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
bbox_min, bbox_max = mesh_bounding_box(nodes)
assert np.all(bbox_max >= bbox_min), '[TC09] 包围盒max应>=min FAILED'
assert bbox_min[0] == 0.0, '[TC09] 包围盒x_min FAILED'

# ---- TC10: tri_mesh_edge_neighbors 边界边为整数 ----
elements = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
boundary = tri_mesh_edge_neighbors(nodes, elements)
assert boundary.dtype == int or np.issubdtype(boundary.dtype, np.integer), '[TC10] 边界边应为整数 FAILED'
assert boundary.shape[1] == 2, '[TC10] 边界边应为2列 FAILED'

# ---- TC11: tet_mesh_tet_neighbors 形状与邻接正确性 ----
tetra_node = np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=int).T
neighbors = tet_mesh_tet_neighbors(4, 2, tetra_node)
assert neighbors.shape == (4, 2), '[TC11] 四面体邻居矩阵形状 FAILED'
assert np.all(neighbors >= -1), '[TC11] 邻居索引应>=-1 FAILED'

# ---- TC12: sellmeier_equation_silica 1.55um折射率 ----
lam = np.array([1.55])
n = sellmeier_equation_silica(lam)
assert 1.4 < n[0] < 1.5, '[TC12] 石英1.55um折射率应在1.4-1.5 FAILED'

# ---- TC13: beta_from_sellmeier 传播常数为正 ----
beta = beta_from_sellmeier(np.array([0.5, 1.0, 1.55, 2.0]))
assert np.all(beta > 0), '[TC13] 传播常数应为正 FAILED'

# ---- TC14: chebyshev_zeros 节点在区间内 ----
nodes_cheb = chebyshev_zeros(10, -2.0, 3.0)
assert np.all(nodes_cheb >= -2.0), '[TC14] Chebyshev节点应>=a FAILED'
assert np.all(nodes_cheb <= 3.0), '[TC14] Chebyshev节点应<=b FAILED'
assert len(nodes_cheb) == 10, '[TC14] Chebyshev节点数 FAILED'

# ---- TC15: chebyshev_coefficients 常数函数c0为2 ----
c_const = chebyshev_coefficients(-1.0, 1.0, 5, lambda x: np.ones_like(x))
assert len(c_const) == 5, '[TC15] Chebyshev系数长度 FAILED'
assert abs(c_const[0] - 2.0) < 1e-12, '[TC15] 常数函数c0应为2 FAILED'

# ---- TC16: chebyshev_interpolant 奇函数在节点处精确 ----
nodes_c = chebyshev_zeros(8, -1.0, 1.0)
c_odd = chebyshev_coefficients(-1.0, 1.0, 8, lambda x: x ** 3)
y_interp = chebyshev_interpolant(c_odd, -1.0, 1.0, nodes_c)
y_true = nodes_c ** 3
assert np.allclose(y_interp, y_true, atol=1e-10), '[TC16] Chebyshev插值对x^3应在节点精确 FAILED'

# ---- TC17: cubic_spline_coefficients 输出四元组 ----
x_spl = np.linspace(0, 1, 6)
y_spl = np.sin(x_spl)
a_spl, b_spl, c_spl, d_spl = cubic_spline_coefficients(x_spl, y_spl)
assert len(a_spl) == len(x_spl) - 1, '[TC17] 样条系数a长度 FAILED'
assert len(b_spl) == len(x_spl) - 1, '[TC17] 样条系数b长度 FAILED'
assert len(c_spl) == len(x_spl) - 1, '[TC17] 样条系数c长度 FAILED'
assert len(d_spl) == len(x_spl) - 1, '[TC17] 样条系数d长度 FAILED'

# ---- TC18: cubic_spline_eval 节点处近似精确 ----
x_eval = np.array([0.0, 0.2, 0.6, 1.0])
y_eval = cubic_spline_eval(x_eval, a_spl, b_spl, c_spl, d_spl, x_spl)
expected = np.sin(x_eval)
assert np.allclose(y_eval, expected, atol=0.1), '[TC18] 样条节点处近似 FAILED'

# ---- TC19: lebesgue_constant Chebyshev优于等距节点 ----
n_l = 8
a_l, b_l = -1.0, 1.0
x_dense = np.linspace(a_l, b_l, 500)
x_cheb = chebyshev_zeros(n_l, a_l, b_l)
x_eq = np.linspace(a_l, b_l, n_l)
lam_cheb = lebesgue_constant(n_l, x_cheb, x_dense)
lam_eq = lebesgue_constant(n_l, x_eq, x_dense)
assert lam_cheb < lam_eq, '[TC19] Chebyshev Lebesgue常数应小于等距节点 FAILED'

# ---- TC20: dispersion_taylor_coefficients 形状与阶数 ----
omega_t = np.linspace(-10e12, 10e12, 50)
beta_t = omega_t ** 2 * 1e-28
coeffs = dispersion_taylor_coefficients(omega_t, beta_t, 0.0, order=4)
assert len(coeffs) == 5, '[TC20] Taylor系数长度应为order+1 FAILED'

# ---- TC21: spectral_boundary_detect 阈值单调性 ----
omega_b = np.linspace(-20e12, 20e12, 400)
pwr_b = np.exp(-(omega_b / 5e12) ** 2)
left_20, right_20 = spectral_boundary_detect(pwr_b, omega_b, threshold_db=-20.0)
left_10, right_10 = spectral_boundary_detect(pwr_b, omega_b, threshold_db=-10.0)
assert left_10 >= left_20, '[TC21] 更高阈值左边界应更靠右 FAILED'
assert right_10 <= right_20, '[TC21] 更高阈值右边界应更靠左 FAILED'

# ---- TC22: log_spaced_grid 单调递增与范围 ----
log_grid = log_spaced_grid(1e9, 1e15, 30)
assert np.all(np.diff(log_grid) > 0), '[TC22] 对数网格应严格单调递增 FAILED'
assert abs(log_grid[0] - 1e9) < 1e-3, '[TC22] 对数网格起点 FAILED'
assert abs(log_grid[-1] - 1e15) < 1.0, '[TC22] 对数网格终点 FAILED'

# ---- TC23: raman_response_blow_wood 负时间为零 ----
t_r = np.linspace(-50e-15, 100e-15, 200)
h_r = raman_response_blow_wood(t_r, tau1=12.2e-15, tau2=32.0e-15)
assert np.all(h_r[t_r < 0] == 0.0), '[TC23] Raman响应负时间应为零 FAILED'
assert np.any(h_r[t_r >= 0] > 0), '[TC23] Raman响应正时间应有正值 FAILED'

# ---- TC24: self_steepening_factor 基本公式 ----
omega_s = np.array([0.0, 1e15, -1e15])
S = self_steepening_factor(omega_s, omega0=2e15)
assert abs(S[0] - 1.0) < 1e-12, '[TC24] 零频自陡峭因子应为1 FAILED'
assert S[1] > 1.0, '[TC24] 正频自陡峭因子应>1 FAILED'
assert S[2] < 1.0, '[TC24] 负频自陡峭因子应<1 FAILED'

# ---- TC25: sech_pulse 峰值振幅 ----
t_p = np.linspace(-200e-15, 200e-15, 512)
A_sech = sech_pulse(t_p, T0=50e-15, P0=100.0)
peak_idx = np.argmax(np.abs(A_sech))
assert abs(np.abs(A_sech[peak_idx]) - np.sqrt(100.0)) < 0.1, '[TC25] sech脉冲峰值振幅 FAILED'

# ---- TC26: gaussian_pulse 峰值与形状 ----
A_gauss = gaussian_pulse(t_p, T0=50e-15, P0=100.0)
assert abs(np.max(np.abs(A_gauss)) - np.sqrt(100.0)) < 0.1, '[TC26] 高斯脉冲峰值振幅 FAILED'
assert abs(A_gauss[0]) < abs(A_gauss[len(A_gauss)//2]), '[TC26] 高斯脉冲中心应为最大 FAILED'

# ---- TC27: arclength_parameterization 归一化与总弧长 ----
y_arc = np.exp(1j * np.linspace(0, 2*np.pi, 100))
t_arc = np.linspace(0, 1, 100)
S_total, s_param = arclength_parameterization(y_arc, t_arc)
assert S_total > 0, '[TC27] 总弧长应为正 FAILED'
assert abs(s_param[0]) < 1e-12, '[TC27] 弧长参数起点应为0 FAILED'
assert abs(s_param[-1] - 1.0) < 1e-12, '[TC27] 弧长参数终点应为1 FAILED'
assert np.all(np.diff(s_param) >= -1e-12), '[TC27] 弧长参数应非减 FAILED'

# ---- TC28: adaptive_step_size_estimate 在界限内 ----
A_test = sech_pulse(t_p, 50e-15, 1000.0)
dz_est = adaptive_step_size_estimate(A_test, t_p, dz_current=1e-4, z=0.0, z_target=1e-2)
assert dz_est >= 1e-6, '[TC28] 自适应步长应>=min_dz FAILED'
assert dz_est <= 1e-2, '[TC28] 自适应步长应<=max_dz FAILED'

# ---- TC29: dispersion_operator_fft 形状与损耗项 ----
omega_d = np.fft.fftfreq(128, 1e-15) * 2 * np.pi
beta_c = np.zeros(6)
beta_c[2] = -1e-26
D_op = dispersion_operator_fft(omega_d, alpha=0.2e-3, beta_coeffs=beta_c)
assert D_op.shape == omega_d.shape, '[TC29] 色散算子形状 FAILED'
assert np.all(np.real(D_op) == -0.1e-3), '[TC29] 色散算子实部应为-alpha/2 FAILED'

# ---- TC30: fresnel_integrals 原点为零 ----
C0, S0 = fresnel_integrals(0.0)
assert abs(C0) < 1e-15, '[TC30] Fresnel C(0)应为0 FAILED'
assert abs(S0) < 1e-15, '[TC30] Fresnel S(0)应为0 FAILED'

# ---- TC31: fresnel_number 近场远场判断 ----
Nf_near = fresnel_number(1e-3, 1.55e-6, 1e-4)
Nf_far = fresnel_number(1e-3, 1.55e-6, 10.0)
assert Nf_near > 1.0, '[TC31] 近场Fresnel数应>1 FAILED'
assert Nf_far < 1.0, '[TC31] 远场Fresnel数应<1 FAILED'

# ---- TC32: fresnel_diffraction_1d 输出形状 ----
x_ap = np.linspace(-1e-3, 1e-3, 64)
aperture = np.ones(64, dtype=complex)
x_obs = np.linspace(-2e-3, 2e-3, 128)
E_out = fresnel_diffraction_1d(aperture, x_ap, x_obs, 1.55e-6, 1e-3)
assert E_out.shape == (128,), '[TC32] Fresnel衍射输出形状 FAILED'
assert np.all(np.isfinite(E_out)), '[TC32] Fresnel衍射输出应为有限值 FAILED'

# ---- TC33: mode_coupling_matrix 厄米性与形状 ----
K = mode_coupling_matrix(4, delta_beta=50.0, coupling_coeff=5.0)
assert K.shape == (4, 4), '[TC33] 耦合矩阵形状 FAILED'
assert np.allclose(K, K.T.conj()), '[TC33] 耦合矩阵应为厄米 FAILED'

# ---- TC34: xpm_coefficients 对角与非对角关系 ----
overlap = np.ones((3, 3)) * 0.5
np.fill_diagonal(overlap, 1.0)
chi = xpm_coefficients(3, gamma=2.0, overlap_factors=overlap)
assert chi.shape == (3, 3), '[TC34] XPM系数形状 FAILED'
assert abs(chi[0, 0] - 2.0) < 1e-12, '[TC34] XPM对角元应为gamma FAILED'
assert abs(chi[0, 1] - 2.0) < 1e-12, '[TC34] XPM非对角元应为2*gamma*overlap FAILED'

# ---- TC35: multimode_propagation_verlet 功率守恒 ----
K_mm = mode_coupling_matrix(3, 100.0, 10.0)
overlap_mm = np.ones((3, 3)) * 0.5
np.fill_diagonal(overlap_mm, 1.0)
chi_mm = xpm_coefficients(3, 1.0, overlap_mm)
A0_mm = np.sqrt(300.0 / 3.0) * np.ones(3, dtype=complex)
z_arr, A_hist_mm = multimode_propagation_verlet(A0_mm, 0.01, K_mm, chi_mm, dz=1e-3)
P_hist_mm = mode_power_orbits(A_hist_mm)
P_total_initial = np.sum(P_hist_mm[0])
P_total_final = np.sum(P_hist_mm[-1])
assert abs(P_total_final - P_total_initial) / P_total_initial < 1e-10, '[TC35] 多模传播功率不守恒 FAILED'

# ---- TC36: spectral_bandwidth fwhm与rms为正 ----
omega_spec = np.linspace(-10e12, 10e12, 1000)
p_spec = np.exp(-(omega_spec / 2e12) ** 2)
bw_fwhm = spectral_bandwidth(omega_spec, p_spec, method="fwhm")
bw_rms = spectral_bandwidth(omega_spec, p_spec, method="rms")
assert bw_fwhm > 0, '[TC36] FWHM带宽应为正 FAILED'
assert bw_rms > 0, '[TC36] RMS带宽应为正 FAILED'

# ---- TC37: spectral_flatness 范围 ----
flat_uniform = spectral_flatness(np.ones(100))
assert abs(flat_uniform - 1.0) < 1e-12, '[TC37] 均匀谱平坦度应为1 FAILED'
flat_peaked = spectral_flatness(np.concatenate([np.ones(10)*100, np.ones(90)*1e-6]))
assert 0.0 <= flat_peaked <= 1.0, '[TC37] 尖峰谱平坦度应在[0,1] FAILED'
assert flat_peaked < flat_uniform, '[TC37] 尖峰谱平坦度应小于均匀谱 FAILED'

# ---- TC38: soliton_order 基本公式 ----
N_sol = soliton_order(beta2=-1e-26, gamma=10.0, T0=50e-15, P0=1e3)
assert N_sol > 0, '[TC38] 孤子阶数应为正 FAILED'
L_D = dispersion_length(T0=50e-15, beta2=-1e-26)
L_NL = nonlinear_length(gamma=10.0, P0=1e3)
expected_N = np.sqrt(L_D / L_NL)
assert abs(N_sol - expected_N) < 1e-10, '[TC38] 孤子阶数公式一致性 FAILED'

# ---- TC39: dispersion_length 极小beta2极限 ----
L_D_large = dispersion_length(T0=50e-15, beta2=1e-40)
assert L_D_large > 1e15, '[TC39] 极小beta2色散长度应极大 FAILED'

# ---- TC40: nonlinear_length 零参数极限 ----
L_NL_zero = nonlinear_length(gamma=1e-20, P0=1e-20)
assert L_NL_zero > 1e15, '[TC40] 极小gamma*P0非线性长度应极大 FAILED'

# ---- TC41: fourier_limit_duration 形状依赖 ----
T_sech = fourier_limit_duration(1e12, pulse_shape="sech")
T_gauss = fourier_limit_duration(1e12, pulse_shape="gaussian")
assert T_sech > 0, '[TC41] sech傅里叶极限应为正 FAILED'
assert T_gauss > 0, '[TC41] gaussian傅里叶极限应为正 FAILED'
assert T_gauss > T_sech, '[TC41] gaussian极限应大于sech FAILED'

# ---- TC42: spectral_snr 基本计算 ----
p_snr = np.concatenate([np.ones(50)*10, np.ones(50)*0.01])
snr_val = spectral_snr(p_snr, (20, 30))
assert snr_val > 0, '[TC42] SNR应为正 FAILED'

# ---- TC43: ssfm_propagate 输出形状与记录点数 ----
np.random.seed(42)
n_t = 2 ** 8
T_win = 2e-12
t_ssfm = np.linspace(-T_win/2, T_win/2, n_t)
A0_ssfm = sech_pulse(t_ssfm, T0=100e-15, P0=100.0)
beta_c_ssfm = np.array([0, 0, -1e-26, 0, 0, 0, 0], dtype=float)
omega0_ssfm = 2.0 * np.pi * 2.99792458e8 / 1.55e-6
z_out_s, A_z_s, spec_z_s = ssfm_propagate(
    A0_ssfm, t_ssfm, z_target=1e-4, alpha=0.2e-3, gamma=1.0,
    beta_coeffs=beta_c_ssfm, omega0=omega0_ssfm,
    dz_initial=1e-5, n_z_records=5, use_symmetrized=True
)
assert z_out_s.shape == (5,), '[TC43] z_out形状 FAILED'
assert A_z_s.shape == (5, n_t), '[TC43] A_z形状 FAILED'
assert spec_z_s.shape == (5, n_t), '[TC43] spec_z形状 FAILED'

# ---- TC44: 主入口函数完整流程 ----
np.random.seed(42)
result_main = main()
assert result_main == 0, '[TC44] main()应返回0 FAILED'

print('\n全部 44 个测试通过!\n')
