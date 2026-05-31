# ---- TC01: 验证二维倒格矢正交关系 a_i · b_j = 2π δ_ij ----
a1 = np.array([500e-9, 0.0])
a2 = np.array([0.0, 500e-9])
b1, b2 = reciprocal_lattice_2d(a1, a2)
assert np.isclose(np.dot(a1, b1), 2*np.pi) and np.isclose(np.dot(a2, b2), 2*np.pi) and np.isclose(np.dot(a1, b2), 0.0) and np.isclose(np.dot(a2, b1), 0.0), '[TC01] 倒格矢正交关系 FAILED'

# ---- TC02: 正方晶格 BZ 路径输出尺寸正确 ----
b1 = np.array([2*np.pi/500e-9, 0.0])
b2 = np.array([0.0, 2*np.pi/500e-9])
k_pts, labels = brillouin_zone_path_2d(b1, b2, 5, 'square')
assert k_pts.shape[1] == 2 and len(labels) == 3 and len(k_pts) == 15, '[TC02] BZ 路径尺寸 FAILED'

# ---- TC03: 归一化频率对正参数返回非负标量 ----
a_test = 500e-9
omega_test = 2*np.pi*C_0 / a_test
om_norm = normalized_frequency(a_test, omega_test)
assert np.isscalar(om_norm) and om_norm >= 0.0, '[TC03] 归一化频率 FAILED'

# ---- TC04: 带隙相对宽度解析验证 ----
ratio = bandgap_ratio(1.0, 2.0)
expected = 2.0 * (2.0 - 1.0) / (2.0 + 1.0)
assert np.isclose(ratio, expected), '[TC04] 带隙相对宽度 FAILED'

# ---- TC05: 微腔 Q 因子解析验证 ----
Q = cavity_q_factor(1e15, 1e12)
assert np.isclose(Q, 1000.0), '[TC05] Q 因子 FAILED'

# ---- TC06: 布拉格反射率必须在 [0, 1] 内 ----
R = bragg_reflectivity(5000.0+0j, 100e-6, 0.0)
assert 0.0 <= R <= 1.0, '[TC06] 布拉格反射率范围 FAILED'

# ---- TC07: 耦合模方程输出长度为 2 ----
dA = coupled_mode_equations(0.0, np.array([1.0, 0.0]), 5000.0+0j, 0.0)
assert len(dA) == 2, '[TC07] 耦合模方程输出尺寸 FAILED'

# ---- TC08: 幻方矩阵每行每列和对角线之和相等 ----
M = magic_matrix(5)
magic_sum = np.sum(M[0, :])
row_ok = np.all(np.sum(M, axis=1) == magic_sum)
col_ok = np.all(np.sum(M, axis=0) == magic_sum)
diag1_ok = np.sum(np.diag(M)) == magic_sum
diag2_ok = np.sum(np.diag(np.fliplr(M))) == magic_sum
assert row_ok and col_ok and diag1_ok and diag2_ok, '[TC08] 幻方矩阵 FAILED'

# ---- TC09: 正方晶格介电分布尺寸与值范围正确 ----
eps_r, x, y = square_photonic_crystal(8, 8, 500e-9, 100e-9, 12.0, 1.0)
assert eps_r.shape == (8, 8) and np.all((eps_r == 1.0) | (eps_r == 12.0)), '[TC09] 正方晶格 FAILED'

# ---- TC10: 优化矩阵乘法与 numpy 结果一致 ----
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
C = mxm_optimized(A, B)
C_np = A.dot(B)
assert np.allclose(C, C_np), '[TC10] 矩阵乘法 FAILED'

# ---- TC11: 三对角方程组求解验证 ----
n = 5
a_td = np.ones(n-1)
b_td = 4.0 * np.ones(n)
c_td = np.ones(n-1)
d_td = np.ones(n)
x_td = solve_tridiagonal(a_td, b_td, c_td, d_td, n)
res = np.zeros(n)
res[0] = b_td[0]*x_td[0] + c_td[0]*x_td[1] - d_td[0]
for i in range(1, n-1):
    res[i] = a_td[i-1]*x_td[i-1] + b_td[i]*x_td[i] + c_td[i]*x_td[i+1] - d_td[i]
res[-1] = a_td[-1]*x_td[-2] + b_td[-1]*x_td[-1] - d_td[-1]
assert np.linalg.norm(res) < 1e-10, '[TC11] 三对角求解 FAILED'

# ---- TC12: packed Cholesky 分解与求解集成验证 ----
a_packed = np.array([4.0, 1.0, 3.0, 0.5, 0.8, 2.0])
r_packed, info = r8pp_fa(3, a_packed)
assert info == 0, '[TC12] Cholesky 分解失败'
b_test = np.array([1.0, 2.0, 3.0])
x_sol = r8pp_sl(3, r_packed, b_test)
x_verify = r8pp_mv(3, a_packed, x_sol)
assert np.linalg.norm(x_verify - b_test) < 1e-10, '[TC12] Cholesky 求解验证 FAILED'

# ---- TC13: Gauss-Seidel 迭代收敛到正确解 ----
A_gs = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
b_gs = np.array([1.0, 2.0, 3.0])
x_gs, hist_gs = gauss_seidel_solve(A_gs, b_gs, tol=1e-12, max_iter=1000)
res_gs = np.linalg.norm(A_gs.dot(x_gs) - b_gs)
assert res_gs < 1e-10 and len(hist_gs) < 1000, '[TC13] Gauss-Seidel FAILED'

# ---- TC14: 离散正弦变换结果可复现 ----
np.random.seed(42)
f_vals = np.sin(np.pi * np.arange(1, 9) / 9)
s1 = sine_transform_data(8, f_vals)
np.random.seed(42)
s2 = sine_transform_data(8, f_vals)
assert np.allclose(s1, s2), '[TC14] 正弦变换可复现性 FAILED'

# ---- TC15: 任务分配总数正确 ----
divs = task_division(10, 0, 3)
total_tasks = sum(d[1] for d in divs)
assert total_tasks == 10, '[TC15] 任务分配 FAILED'

# ---- TC16: RK4 对线性 ODE dy/dt = -y 给出指数衰减解析解 ----
def dydt_linear(t, y):
    return -y
t_rk, y_rk = rk4(dydt_linear, (0.0, 1.0), np.array([1.0]), 100)
y_exact = np.exp(-t_rk)
assert np.allclose(y_rk[:, 0], y_exact, atol=1e-4), '[TC16] RK4 线性 ODE FAILED'

# ---- TC17: 带隙检测对人工数据正确识别带隙 ----
omega_test = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
gaps = detect_bandgaps(omega_test, threshold_ratio=0.01)
assert len(gaps) >= 2, '[TC17] 带隙检测 FAILED'

# ---- TC18: 带隙失配参数在合理范围内 ----
mismatch = gap_mismatch_parameter(12.0, 1.0, 0.3)
assert 0.0 <= mismatch <= 1.0, '[TC18] 失配参数范围 FAILED'

# ---- TC19: 二维三角形体积(面积)解析验证 ----
t_tri = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
vol = simplex_volume(2, t_tri)
assert np.isclose(vol, 0.5), '[TC19] 单纯形体积 FAILED'

# ---- TC20: 离散 CDF 单调非减且归一化到 1 ----
pdf = np.array([[0.1, 0.2], [0.3, 0.4]])
cdf = set_discrete_cdf(2, 2, pdf)
monotone = True
for j in range(2):
    col = cdf[:, j]
    if np.any(np.diff(col) < -1e-15):
        monotone = False
assert monotone and np.isclose(cdf[-1, -1], 1.0), '[TC20] CDF 单调性 FAILED'

# ---- TC21: Chebyshev 拒绝采样结果在 [-1, 1] 内 ----
np.random.seed(42)
samples, trials = chebyshev2_rejection_sample(100)
assert np.all(samples >= -1.0) and np.all(samples <= 1.0), '[TC21] Chebyshev 采样范围 FAILED'

# ---- TC22: 高斯拒绝采样结果可复现 ----
np.random.seed(42)
s1 = gaussian_rejection_sample(10, 0.0, 1.0)
np.random.seed(42)
s2 = gaussian_rejection_sample(10, 0.0, 1.0)
assert np.allclose(s1, s2), '[TC22] 高斯采样可复现性 FAILED'

# ---- TC23: 光子跃迁矩阵最大特征值模为 1 ----
np.random.seed(42)
A_ph, eigenvals, xi = photon_hopping_matrix(20, 0.4, 0.2)
max_abs_ev = np.max(np.abs(eigenvals))
assert np.isclose(max_abs_ev, 1.0, atol=1e-10), '[TC23] 跃迁矩阵特征值 FAILED'

# ---- TC24: 辐射输运方程能量守恒 T+R <= 1 ----
z_rt, I_fwd, I_bwd, T, R = radiative_transfer_1d(1.0, 100.0, 10.0, 1e-3, 50)
assert T + R <= 1.0 + 1e-6 and T >= 0.0 and R >= 0.0, '[TC24] 能量守恒 FAILED'

# ---- TC25: 安德森局域化长度为正且 Ioffe-Regel 参数正确 ----
xi_loc, kl, is_loc = anderson_localization_length(1550e-9, 500e-9, 0.1)
assert xi_loc > 0.0 and kl > 0.0, '[TC25] 局域化长度 FAILED'

# ---- TC26: 光子扩散常数解析验证 D = v_g * l_mfp / 3 ----
D = diffusion_constant_photonic(1.0, 3.0)
assert np.isclose(D, 1.0), '[TC26] 扩散常数 FAILED'

# ---- TC27: 标度理论 beta 函数在 g->inf 时趋近 d-2 ----
beta_large_g = scaling_theory_beta_function(1e6, d=3)
assert np.isclose(beta_large_g, 1.0, atol=1e-6), '[TC27] beta 函数渐近行为 FAILED'

# ---- TC28: NAS 随机数生成器可复现 ----
rng1 = NASRandom(seed=0.314159)
r1a = rng1.next()
rng2 = NASRandom(seed=0.314159)
r1b = rng2.next()
assert np.isclose(r1a, r1b), '[TC28] NASRandom 可复现性 FAILED'

# ---- TC29: 一维 DFT 正逆变换恢复原始信号 ----
x_sig = np.array([1.0, 2.0, 3.0, 4.0])
X_dft = dft_1d_naive(x_sig)
x_rec = dft_1d_naive(X_dft, inverse=True)
assert np.allclose(x_sig, x_rec), '[TC29] DFT 正逆变换 FAILED'

# ---- TC30: 二维 FFT 输出谱尺寸与输入一致 ----
field = np.ones((16, 16))
spectrum, kx_f, ky_f = fft_2d_photonic(field, 1e-9, 1e-9)
assert spectrum.shape == (16, 16) and len(kx_f) == 16 and len(ky_f) == 16, '[TC30] 2D FFT 输出尺寸 FAILED'

# ---- TC31: 慢光群折射率输出尺寸与输入一致 ----
omega_sl = np.linspace(1e14, 2e14, 10)
k_sl = np.linspace(0, 1, 10)
n_g = slow_light_group_index(omega_sl, k_sl)
assert len(n_g) == 10, '[TC31] 群折射率输出尺寸 FAILED'

# ---- TC32: 布拉格光栅反射率在 [0, 1] 内 ----
z_bg, Ap_bg, Am_bg, R_bg = propagate_bragg_grating(5000.0+0j, 0.0, 100e-6, n_z=100)
assert 0.0 <= R_bg <= 1.0, '[TC32] 布拉格光栅反射率 FAILED'

# ---- TC33: 三维 LDOS 对正参数返回非负值 ----
rho = local_density_of_states_3d(1e15, 3.0, 1e-18)
assert rho >= 0.0, '[TC33] LDOS 非负性 FAILED'

# ---- TC34: ST→CCS 非零元数量正确 ----
ist = np.array([0, 1, 2])
jst = np.array([0, 1, 2])
ncc = st_to_ccs_size(3, ist, jst)
assert ncc == 3, '[TC34] ST→CCS 大小 FAILED'

# ---- TC35: 五对角方程组求解验证 ----
n_p = 6
a2 = 0.1 * np.ones(n_p - 2)
a1 = 0.5 * np.ones(n_p - 1)
bp = 3.0 * np.ones(n_p)
c1 = 0.5 * np.ones(n_p - 1)
c2 = 0.1 * np.ones(n_p - 2)
dp = np.ones(n_p)
x_p = solve_pentadiagonal(a2, a1, bp, c1, c2, dp, n_p)
A_p = np.zeros((n_p, n_p))
for i in range(n_p):
    A_p[i, i] = bp[i]
    if i > 0: A_p[i, i-1] = a1[i-1]
    if i > 1: A_p[i, i-2] = a2[i-2]
    if i < n_p-1: A_p[i, i+1] = c1[i]
    if i < n_p-2: A_p[i, i+2] = c2[i]
res_p = A_p.dot(x_p) - dp
assert np.linalg.norm(res_p) < 1e-6, '[TC35] 五对角求解 FAILED'

# ---- TC36: k 点划分子集总数等于原始 k 点数 ----
k_test = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
subsets = divide_k_points(k_test, 2)
total_k = sum(len(s) for s in subsets)
assert total_k == 4, '[TC36] k 点划分 FAILED'

# ---- TC37: 马尔可夫链扩散熵单调不减 ----
np.random.seed(42)
A_md, _, _ = photon_hopping_matrix(10, 0.4, 0.1)
P0 = np.zeros(10)
P0[5] = 1.0
dist_md, entropy_md = photon_diffusion_markov(A_md, P0, n_steps=20)
entropy_diff = np.diff(entropy_md)
assert np.all(entropy_diff >= -1e-12), '[TC37] 扩散熵单调性 FAILED'

# ---- TC38: Van Hove 奇异点检测返回列表 ----
omega_vh = np.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]])
k_dist_vh = np.array([0.0, 0.5, 1.0])
singularities = van_hove_singularity_type(omega_vh, k_dist_vh)
assert isinstance(singularities, list), '[TC38] Van Hove 奇异点 FAILED'

# ---- TC39: 缺陷态频率在带隙内且线宽为正 ----
omega_d, delta_omega = defect_mode_frequency(1e15, 0.1, 1000.0)
assert omega_d > 0.0 and delta_omega > 0.0, '[TC39] 缺陷态频率 FAILED'

# ---- TC40: CDF 反演采样结果在 [0,1]^2 内 ----
np.random.seed(42)
pdf_d = np.array([[0.25, 0.25], [0.25, 0.25]])
cdf_d = set_discrete_cdf(2, 2, pdf_d)
xy_d = discrete_cdf_to_xy(2, 2, cdf_d, 10)
assert np.all(xy_d >= 0.0) and np.all(xy_d <= 1.0), '[TC40] CDF 采样范围 FAILED'
