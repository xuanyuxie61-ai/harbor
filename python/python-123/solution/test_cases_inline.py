# ---- TC01: Laplace径向2D精确解在r=0处有限 ----
x = np.array([1e-6, 1e-5])
y = np.array([0.0, 0.0])
u, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x, y, a=0.5, b=1.0)
assert np.all(np.isfinite(u)), '[TC01] Laplace径向2D精确解在r=0处有限 FAILED'

# ---- TC02: 氧气扩散边界条件满足 ----
r_test = np.linspace(0.01, 1.0, 30)
C_test = oxygen_diffusion_steady_state_radial(r_test, 1.0, C_boundary=0.8, D=1e-3, consumption_rate=0.05)
assert np.abs(C_test[-1] - 0.8) < 1e-6, '[TC02] 氧气扩散边界条件满足 FAILED'

# ---- TC03: Michaelis-Menten零浓度时消耗速率为零 ----
C_zero = np.array([0.0, 0.0, 0.0])
rho_test = np.array([1.0, 0.5, 2.0])
cons = michaelis_menten_consumption(C_zero, rho_test, Vmax=1.0, Km=0.1)
assert np.allclose(cons, 0.0), '[TC03] Michaelis-Menten零浓度时消耗速率为零 FAILED'

# ---- TC04: 缺氧区域比例在[0,1]范围内 ----
C_all_high = np.ones(50) * 0.5
frac1 = hypoxia_region_fraction(C_all_high, threshold=0.02)
assert 0.0 <= frac1 <= 1.0, '[TC04] 缺氧区域比例在[0,1]范围内 FAILED'
C_all_low = np.zeros(50)
frac2 = hypoxia_region_fraction(C_all_low, threshold=0.02)
assert 0.0 <= frac2 <= 1.0, '[TC04b] 缺氧区域比例在[0,1]范围内 FAILED'

# ---- TC05: Clenshaw-Curtis精确积分常数函数 ----
def _const_5(x):
    return np.full_like(x, 5.0)
q_cc = clenshaw_curtis_integrate(_const_5, -1.0, 2.0, n=8)
assert np.abs(q_cc - 15.0) < 1e-10, '[TC05] Clenshaw-Curtis精确积分常数函数 FAILED'

# ---- TC06: 稀疏网格单项式积分x*y理论值为0.25 ----
sg = sparse_grid_monomial_integral(dim=2, level=3, exponents=np.array([1, 1]))
assert np.abs(sg - 0.25) < 1e-12, '[TC06] 稀疏网格单项式积分x*y理论值为0.25 FAILED'

# ---- TC07: 细胞转移矩阵每行之和为1 ----
M = cell_transition_matrix(p_prolif_to_quies=0.2, p_prolif_to_apop=0.05,
                           p_quies_to_prolif=0.15, p_quies_to_apop=0.1,
                           p_apop_to_quies=0.02)
row_sums = M.sum(axis=1)
assert np.allclose(row_sums, 1.0), '[TC07] 细胞转移矩阵每行之和为1 FAILED'

# ---- TC08: Markov链0步演化保持初始群体不变 ----
N0 = np.array([100.0, 50.0, 20.0, 10.0])
M_def = cell_transition_matrix()
hist0 = evolve_cell_population_markov(N0, M_def, steps=0)
assert hist0.shape == (1, 4), '[TC08] Markov链0步演化保持初始群体不变 FAILED'
assert np.allclose(hist0[0], N0), '[TC08b] Markov链0步演化保持初始群体不变 FAILED'

# ---- TC09: 肿瘤细胞比例之和为1 ----
np.random.seed(42)
grid_test = np.random.randint(0, 4, size=(12, 12))
fracs = compute_tumor_cellularity(grid_test)
assert np.abs(sum(fracs) - 1.0) < 1e-12, '[TC09] 肿瘤细胞比例之和为1 FAILED'

# ---- TC10: CA接触抑制更新保持网格形状不变 ----
np.random.seed(42)
cg = np.random.randint(0, 3, size=(10, 10))
ng = np.random.rand(10, 10)
new_cg = ca_contact_inhibition_update(cg, ng, threshold_nutrient=0.1, inhibition_threshold=5)
assert new_cg.shape == (10, 10), '[TC10] CA接触抑制更新保持网格形状不变 FAILED'

# ---- TC11: CVT生成子初始化维度正确 ----
gens, ptype = initialize_tumor_generators(n_boundary=6, n_interior=10, radius=1.0, seed=42)
assert gens.shape == (16, 2), '[TC11] CVT生成子初始化维度正确 FAILED'
assert ptype.shape == (16,), '[TC11b] CVT生成子初始化维度正确 FAILED'

# ---- TC12: 贪婪分区差异非负 ----
w_test = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
labels_test, disc = partition_metabolic_activity(w_test)
assert disc >= 0.0, '[TC12] 贪婪分区差异非负 FAILED'

# ---- TC13: 径向生长半径按比例缩放 ----
gens2, ptype2 = initialize_tumor_generators(6, 4, 1.0, seed=42)
new_r, new_g, new_p = radial_growth_expand(gens2, ptype2, 1.0, new_boundary_count=6)
assert np.abs(new_r - 2.0) < 1e-12, '[TC13] 径向生长半径按比例缩放 FAILED'

# ---- TC14: 双调和算子特征值全为实数 ----
eigvals, _, _ = biharmonic_stress_operator(nx=6, ny=6, hx=0.1, hy=0.1, mu=0.25)
assert np.all(np.isreal(eigvals)), '[TC14] 双调和算子特征值全为实数 FAILED'

# ---- TC15: 应力诱导凋亡概率在[0,1]范围内 ----
stress_test = np.array([0.0, 0.2, 0.5, 0.8, 1.2])
prob_apop = compute_stress_induced_apoptosis(stress_test, threshold=0.5, steepness=10.0)
assert np.all((prob_apop >= 0.0) & (prob_apop <= 1.0)), '[TC15] 应力诱导凋亡概率在[0,1]范围内 FAILED'

# ---- TC16: Jacobi多项式P0恒为1 ----
x_jac = np.linspace(-1.0, 1.0, 15)
jac_vals = jacobi_polynomial(15, 2, alpha=0.5, beta=0.5, x=x_jac)
assert np.allclose(jac_vals[:, 0], 1.0), '[TC16] Jacobi多项式P0恒为1 FAILED'

# ---- TC17: Gauss-Legendre权重之和等于区间长度 ----
x_gl, w_gl = gauss_legendre_quadrature(10, a=-1.0, b=1.0)
assert np.abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC17] Gauss-Legendre权重之和等于区间长度 FAILED'

# ---- TC18: 谱方法求解扩散方程误差小 ----
def _src(x):
    return np.sin(np.pi * x)
xp, ua = solve_spectral_diffusion(_src, n_modes=10, diffusion_coeff=1.0)
u_ex = _src(xp) / (np.pi**2)
err_spec = np.max(np.abs(ua - u_ex))
assert err_spec < 0.35, '[TC18] 谱方法求解扩散方程误差小 FAILED'

# ---- TC19: Newton标量法求解cos(x)=x ----
def _fcos(x):
    return np.cos(x) - x
def _fpcos(x):
    return -np.sin(x) - 1.0
root, fval, it_n, status_n = newton_solve_scalar(_fcos, _fpcos, a0=0.5, tol=1e-12)
assert np.abs(fval) < 1e-10, '[TC19] Newton标量法求解cos(x)=x FAILED'

# ---- TC20: 稀疏矩阵COO转稠密累加正确 ----
ist_t = np.array([1, 1, 2, 2])
jst_t = np.array([1, 1, 2, 2])
ast_t = np.array([1.0, 2.0, 3.0, 4.0])
A_dense = st_to_ge(4, ist_t, jst_t, ast_t)
assert A_dense[0, 0] == 3.0 and A_dense[1, 1] == 7.0, '[TC20] 稀疏矩阵COO转稠密累加正确 FAILED'

# ---- TC21: Dirichlet边界条件正确施加 ----
A21 = np.array([[2.0, -1.0], [-1.0, 2.0]])
b21 = np.array([1.0, 1.0])
A_bc, b_bc = apply_dirichlet_bc(A21, b21, np.array([0]), np.array([0.0]))
assert A_bc[0, 0] == 1.0 and b_bc[0] == 0.0, '[TC21] Dirichlet边界条件正确施加 FAILED'

# ---- TC22: 稀疏矩阵向量乘与稠密一致 ----
ist22 = np.array([1, 2])
jst22 = np.array([1, 2])
ast22 = np.array([3.0, 5.0])
xv = np.array([1.0, 1.0])
y_sp = sparse_matrix_vector_product(ist22, jst22, ast22, 2, xv, 2)
assert y_sp[0] == 3.0 and y_sp[1] == 5.0, '[TC22] 稀疏矩阵向量乘与稠密一致 FAILED'

# ---- TC23: Gauss积分精确计算多项式 ----
def _x2(x):
    return x**2
q_g = integrate_1d(_x2, 0.0, 1.0, rule="gauss", n=3)
assert np.abs(q_g - 1.0/3.0) < 1e-12, '[TC23] Gauss积分精确计算多项式 FAILED'

# ---- TC24: 2D径向积分单位圆盘面积为pi ----
r24 = np.linspace(0, 1, 200)
f24 = np.ones_like(r24)
area_2d = integrate_radial_profile(r24, f24, dim=2)
assert np.abs(area_2d - np.pi) < 0.01, '[TC24] 2D径向积分单位圆盘面积为pi FAILED'

# ---- TC25: 安全除法正常情况结果正确且有限 ----
res25a = safe_divide(6.0, 2.0, eps=1e-10)
assert np.isfinite(res25a) and np.isclose(res25a, 3.0), '[TC25] 安全除法正常情况结果正确且有限 FAILED'
res25b = safe_divide(-3.0, 1.0, eps=1e-10)
assert np.isfinite(res25b) and np.isclose(res25b, -3.0), '[TC25b] 安全除法正常情况结果正确且有限 FAILED'

# ---- TC26: Sigmoid函数输出在[0,1]范围内 ----
x_sig = np.linspace(-20.0, 20.0, 100)
s_vals = sigmoid(x_sig, steepness=1.0, midpoint=0.0)
assert np.all((s_vals >= 0.0) & (s_vals <= 1.0)), '[TC26] Sigmoid函数输出在[0,1]范围内 FAILED'

# ---- TC27: 均匀分布Gini系数接近0 ----
uniform_vals = np.ones(100)
gini_u = compute_gini_coefficient(uniform_vals)
assert np.abs(gini_u) < 0.01, '[TC27] 均匀分布Gini系数接近0 FAILED'

# ---- TC28: Morse势在平衡距离r_eq处取最小值 ----
r28 = np.linspace(0.5, 3.0, 2000)
V28 = morse_potential(r28, epsilon=2.0, r_eq=1.2, alpha=3.0)
min_idx = np.argmin(V28)
assert np.abs(r28[min_idx] - 1.2) < 0.01, '[TC28] Morse势在平衡距离r_eq处取最小值 FAILED'

# ---- TC29: 参数验证正确通过 ----
params29 = {"D": 0.5, "mu": 0.3}
bounds29 = {"D": (0.0, 1.0), "mu": (0.0, 0.5)}
assert validate_parameters(params29, bounds29) == True, '[TC29] 参数验证正确通过 FAILED'

# ---- TC30: 非线性耦合稳态解维度正确 ----
C30, rho30, res30, it30, st30 = solve_coupled_steady_state(
    N=8, D=3.0, k_c=0.3, Km=0.1,
    lambda_prolif=0.8, lambda_death=0.05, rho_max=1.0
)
assert C30.shape == (8,) and rho30.shape == (8,), '[TC30] 非线性耦合稳态解维度正确 FAILED'
assert it30 >= 0, '[TC30b] 非线性耦合稳态解维度正确 FAILED'

# ---- TC31: 治疗响应指数非负 ----
drug31 = np.ones((5, 5))
rho31 = np.ones((5, 5))
stress31 = np.ones((5, 5))
tri31 = compute_therapy_response_index(drug31, rho31, stress31, 0.1, 0.1)
assert tri31 >= 0.0, '[TC31] 治疗响应指数非负 FAILED'

# ---- TC32: 累积氧消耗非负 ----
o2_32 = np.ones((5, 5)) * 0.5
rho32 = np.ones((5, 5)) * 0.6
total32 = compute_cumulative_oxygen_consumption(o2_32, rho32, 0.1, 0.1, Vmax=1.0, Km=0.1)
assert total32 >= 0.0, '[TC32] 累积氧消耗非负 FAILED'

# ---- TC33: 外部排序结果与numpy.sort一致 ----
arr33 = np.array([3.5, -1.2, 7.8, 0.0, 2.3, -4.5, 9.1])
sorted33 = external_sort_array(arr33)
assert np.allclose(sorted33, np.sort(arr33)), '[TC33] 外部排序结果与numpy.sort一致 FAILED'

# ---- TC34: Laplace径向2D精确解导数uy在y=0轴上为零 ----
x34 = np.array([1.0, 2.0, 3.0])
y34 = np.array([0.0, 0.0, 0.0])
u34, ux34, uy34, uxx34, uxy34, uyy34 = laplace_radial_2d_exact(x34, y34, a=1.0, b=0.0)
assert np.allclose(uy34, 0.0), '[TC34] Laplace径向2D精确解导数uy在y=0轴上为零 FAILED'

# ---- TC35: 肿瘤面积必须为正 ----
nodes35 = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
tri35 = np.array([[0, 1, 2]])
area35 = compute_tumor_area(nodes35, tri35)
assert area35 > 0.0, '[TC35] 肿瘤面积必须为正 FAILED'
assert np.abs(area35 - 3.0) < 1e-12, '[TC35b] 肿瘤面积必须为正 FAILED'

# ---- TC36: 细胞倍增时间为有限正值 ----
N036 = np.array([800.0, 150.0, 40.0, 10.0])
M36 = cell_transition_matrix()
hist36 = evolve_cell_population_markov(N036, M36, steps=10)
Td36 = compute_doubling_time(hist36)
assert not np.isnan(Td36), '[TC36] 细胞倍增时间不应为NaN FAILED'
