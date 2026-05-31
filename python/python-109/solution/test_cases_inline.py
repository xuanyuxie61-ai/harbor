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
