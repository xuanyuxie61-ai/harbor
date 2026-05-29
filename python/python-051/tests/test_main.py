"""
main.py
=======
三维热盐环流-浮游生物生态耦合系统的多尺度数值模拟与参数反演
=================================================================

统一入口：零参数运行，执行以下完整流程：
1. 初始化热盐环流模型（流函数-涡度方程组）
2. 初始化 NPZD 生态系统并耦合至物理场
3. 拉格朗日粒子追踪营养盐微团输运
4. 稀疏网格不确定性量化（生态参数敏感性）
5. Nelder-Mead 参数标定
6. TSP 最优观测路径规划
7. 曲边海岸几何积分
8. 离散营养盐预算分配与状态压缩
9. 收敛性与稳定性分析
"""

import numpy as np
import time

from thermohaline_circulation import ThermohalineCirculation
from ecosystem_dynamics import NPZDEcosystem
from particle_transport import LagrangianParticleTransport
from matrix_solvers import bicg_solve, r8sto_inverse, SparseNCF, create_poisson_stencil
from sparse_grid_cubature import sparse_grid_quadrature, stroud_cn_leg_5, test_quadrature_exactness
from optimization_calibration import nelder_mead_optimize, tsp_descent, generate_sampling_stations, build_distance_matrix
from coastal_geometry import circle_segment_area_from_height, quadrature_on_curved_domain, coastal_boundary_length
from uncertainty_quantification import norm_loo, first_order_sobol_pce, gci_refinement_estimator
from discrete_allocation import diophantine_nd_nonnegative, allocate_nutrient_budget, dictionary_encode, snapshot_matrix


def main():
    print("=" * 72)
    print("海洋热盐环流-生态耦合系统多尺度数值模拟")
    print("Thermohaline Circulation - Marine Ecosystem Coupled Simulation")
    print("=" * 72)
    t_start = time.time()

    # =====================================================================
    # 1. 初始化物理与生态模型
    # =====================================================================
    print("\n[1] 初始化热盐环流与生态系统模型...")
    nx, nz = 32, 16
    Lx, Lz = 5.0e6, 4.0e3
    dt = 43200.0  # 12 小时
    n_steps = 20

    ocean = ThermohalineCirculation(nx=nx, nz=nz, Lx=Lx, Lz=Lz, dt=dt)
    eco = NPZDEcosystem(nx=nx, nz=nz, dx=ocean.dx, dz=ocean.dz, dt=dt,
                        V_max=1.0, K_N=0.5, I_opt=50.0, g_max=0.6, K_P=0.5)
    particles = LagrangianParticleTransport(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                                            nparticles=2000, dt=dt)

    # =====================================================================
    # 2. 耦合时间积分
    # =====================================================================
    print("\n[2] 执行耦合时间积分（{} 步，每步 {:.1f} 小时）...".format(n_steps, dt / 3600.0))
    pp_history = []
    total_n_history = []

    for step in range(n_steps):
        # 物理步
        ocean.step()
        u, w = ocean.get_velocity()

        # 生态步（利用物理速度场）
        eco.step(u, w)

        # 粒子步
        particles.step(u, w, omega_bio=1e-7)
        particles.resample_particles(eco.N, eco.P, ocean.T, ocean.S)

        pp = eco.primary_production()
        tn = eco.total_nitrogen()
        pp_history.append(pp)
        total_n_history.append(tn)

        if (step + 1) % 5 == 0:
            print("  Step {:3d}: PP = {:.4e} mmol N/s, Total N = {:.4e} mmol N".format(
                step + 1, pp, tn))

    print("  耦合积分完成。最终 PP = {:.4e}".format(pp_history[-1]))

    # =====================================================================
    # 3. 矩阵求解器测试（BiCG + Toeplitz + Sparse NCF）
    # =====================================================================
    print("\n[3] 稀疏线性代数模块验证...")
    # BiCG 测试：泊松方程
    b_test = np.random.randn(nx * nz)
    A_poisson = create_poisson_stencil(nx, nz, ocean.dx, ocean.dz)
    x_bicg = bicg_solve(A_poisson, b_test, tol=1e-8, max_iter=300)
    res_bicg = np.linalg.norm(A_poisson.dot(x_bicg) - b_test)
    print("  BiCG 残差 ||Ax-b|| = {:.4e}".format(res_bicg))

    # Toeplitz 逆测试
    n_toep = 8
    a_row = np.array([2.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    T_inv = r8sto_inverse(n_toep, a_row)
    # 构造 Toeplitz 矩阵
    T = np.zeros((n_toep, n_toep))
    for i in range(n_toep):
        for j in range(n_toep):
            T[i, j] = a_row[abs(i - j)]
    I_approx = T_inv.dot(T)
    err_toep = np.linalg.norm(I_approx - np.eye(n_toep))
    print("  Toeplitz 逆误差 ||T^{{-1}}T - I|| = {:.4e}".format(err_toep))

    # NCF 稀疏矩阵测试
    rowcol = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]]).T
    a_vals = np.array([4.0, 4.0, 4.0, -1.0, -1.0])
    spm = SparseNCF(3, 3, 5, rowcol, a_vals)
    x_vec = np.array([1.0, 2.0, 3.0])
    y_vec = spm.mv(x_vec)
    print("  NCF 稀疏乘法结果 = {}".format(y_vec))

    # =====================================================================
    # 4. 稀疏网格不确定性量化
    # =====================================================================
    print("\n[4] 稀疏网格不确定性量化（生态参数敏感性）...")

    def biological_flux_func(xi):
        """
        将归一化参数 xi ∈ [-1,1]^d 映射到生态模型，
        计算稳态初级生产力作为标量输出。
        """
        # 两维参数：V_max ∈ [0.5, 1.5], K_N ∈ [0.2, 1.0]
        V_max = 0.5 + (xi[0] + 1.0) * 0.5
        K_N = 0.2 + (xi[1] + 1.0) * 0.4
        eco_test = NPZDEcosystem(nx=16, nz=8, dx=Lx / 15, dz=Lz / 7, dt=dt,
                                 V_max=V_max, K_N=K_N)
        # 简化的稳态近似：直接计算 uptake
        N_tmp = np.ones((16, 8)) * 3.0
        P_tmp = np.ones((16, 8)) * 0.2
        U = eco_test.uptake_rate(N_tmp, P_tmp)
        pp_approx = np.sum(U * P_tmp) * eco_test.dx * eco_test.dz
        return pp_approx

    result_sg, n_pts = sparse_grid_quadrature(dim=2, max_level=3, func=biological_flux_func)
    print("  稀疏网格积分（level=3）: E[PP] = {:.4e}, 节点数 = {}".format(result_sg, n_pts))

    # Stroud 规则测试（dim=4）
    pts, wts = stroud_cn_leg_5(4)
    if pts is not None:
        print("  Stroud CN:5-1 规则（dim=4）: 节点数 = {}".format(len(wts)))

    # Legendre 精确性检验
    x_cc, w_cc = np.polynomial.legendre.leggauss(5)
    max_exact = test_quadrature_exactness(x_cc, w_cc, degree_max=11)
    print("  Gauss-Legendre 5 点规则精确度: p = {}".format(max_exact))

    # =====================================================================
    # 5. Nelder-Mead 参数标定
    # =====================================================================
    print("\n[5] Nelder-Mead 生态系统参数标定...")

    # 构造“观测”初级生产力（以 V_max=1.0, K_N=0.5 为真值）
    true_pp = biological_flux_func(np.array([0.0, 0.0]))

    def loss(theta):
        # theta = [V_max_raw, K_N_raw]，映射到物理值
        V_max = max(0.1, theta[0])
        K_N = max(0.05, theta[1])
        xi = np.array([(V_max - 0.5) / 0.5 - 1.0, (K_N - 0.2) / 0.4 - 1.0])
        try:
            pp = biological_flux_func(xi)
        except Exception:
            pp = 0.0
        return (pp - true_pp) ** 2

    x0 = np.array([[0.8, 0.3], [1.2, 0.7], [1.0, 0.5]])
    theta_opt, fopt, nfev = nelder_mead_optimize(loss, x0, tol=1e-4, max_feval=200)
    print("  最优参数: V_max = {:.4f}, K_N = {:.4f}".format(theta_opt[0], theta_opt[1]))
    print("  目标函数值 = {:.4e}, 评估次数 = {}".format(fopt, nfev))

    # =====================================================================
    # 6. TSP 最优观测路径
    # =====================================================================
    print("\n[6] TSP 最优 AUV 采样路径规划...")
    stations = generate_sampling_stations(n_stations=12, Lx=Lx, Lz=Lz, depth_min=100.0)
    dist_mat = build_distance_matrix(stations)
    best_path, best_cost = tsp_descent(dist_mat, variation_num=1000, seed=42)
    print("  采样站数 = 12, TSP 近似最优成本 = {:.2e} m".format(best_cost))
    print("  路径顺序前5站: {}".format(best_path[:5]))

    # =====================================================================
    # 7. 曲边海岸几何
    # =====================================================================
    print("\n[7] 曲边海岸几何与区域积分...")
    R_bay = 8.0e5  # 海湾半径 800 km
    h_bay = 0.3 * R_bay
    area_bay = circle_segment_area_from_height(R_bay, h_bay)
    print("  海湾圆缺面积 = {:.4e} m²".format(area_bay))

    # 曲边域积分示例
    def biomass_density(x, z):
        return 0.1 * np.exp(-((x - Lx / 2) ** 2) / (2 * (1e6) ** 2)) + 0.05

    arc_centers = [(1.5e6, 1.0e3)]
    arc_radii = [-5.0e5]
    biomass_total = quadrature_on_curved_domain(biomass_density, (0.0, Lx), (0.0, Lz),
                                                arc_centers, arc_radii, n_x=16, n_z=8)
    print("  曲边域生物量积分 = {:.4e} mmol N".format(biomass_total))

    # 海岸边界长度
    boundary_len = coastal_boundary_length(arc_centers, arc_radii, [np.pi / 2])
    print("  海岸边界长度 = {:.2e} m".format(boundary_len))

    # =====================================================================
    # 8. 离散资源分配与字典编码
    # =====================================================================
    print("\n[8] 离散生态资源分配与状态压缩...")
    budget = 100  # 100 mmol N 预算
    demands = np.array([2, 3, 5, 7, 11])
    alloc = allocate_nutrient_budget(budget, demands, objective="min_variance")
    print("  氮预算 {} mmol 分配给 {} 个功能组:".format(budget, len(demands)))
    print("  分配方案 = {}, 总和验证 = {}".format(alloc, np.dot(demands, alloc)))

    # 丢番图解枚举
    sols = diophantine_nd_nonnegative(demands, budget)
    print("  可行解总数 = {}".format(sols.shape[0]))

    # 字典编码：压缩状态快照
    snapshots = []
    for step in range(0, n_steps, 4):
        fields = {
            'T': ocean.T,
            'S': ocean.S,
            'N': eco.N,
            'P': eco.P,
        }
        snapshots.append(snapshot_matrix(fields))
    states_mat = np.vstack(snapshots)
    dictionary, indices, cr = dictionary_encode(states_mat, tol=1e-4)
    print("  状态字典大小 = {}, 压缩比 = {:.2f}".format(dictionary.shape[0], cr))

    # =====================================================================
    # 9. 稳定性与收敛性分析
    # =====================================================================
    print("\n[9] 稳定性与收敛性分析...")
    loo_psi = norm_loo(ocean.psi)
    loo_omega = norm_loo(ocean.omega)
    print("  L∞(ψ) = {:.4e} m²/s, L∞(ω) = {:.4e} 1/s".format(loo_psi, loo_omega))

    # GCI 估计
    if len(pp_history) >= 3:
        p_gci, gci_val = gci_refinement_estimator(pp_history[-1], pp_history[-2], pp_history[-3], r=1.0)
        print("  GCI 估计: p = {:.3f}, GCI = {:.4e}".format(p_gci, gci_val))

    # PCE 一阶 Sobol' 指数（简化示例）
    coeffs = np.array([1.0, 0.3, 0.2, 0.1])
    multi_indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    total_var = np.sum(coeffs[1:] ** 2)
    S1 = first_order_sobol_pce(coeffs, multi_indices, total_var, dim=2)
    print("  示例 Sobol' 一阶指数: S1 = [{:.3f}, {:.3f}]".format(S1[0], S1[1]))

    # =====================================================================
    # 10. 总结
    # =====================================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("模拟完成。总耗时: {:.2f} 秒".format(elapsed))
    print("=" * 72)
    print("\n关键物理量汇总:")
    print("  经向翻转流函数最大值: {:.4e} m²/s".format(np.max(np.abs(ocean.psi))))
    print("  温度异常范围: [{:.2f}, {:.2f}] K".format(np.min(ocean.T), np.max(ocean.T)))
    print("  盐度异常范围: [{:.3f}, {:.3f}] psu".format(np.min(ocean.S), np.max(ocean.S)))
    print("  最终初级生产力: {:.4e} mmol N/s".format(pp_history[-1]))
    print("  氮守恒误差: {:.4e} mmol N".format(abs(total_n_history[-1] - total_n_history[0])))
    print("  活跃粒子数: {} / {}".format(np.sum(particles.active), particles.nparticles))


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: circle_segment_area_from_height 完整圆 ----
area_full = circle_segment_area_from_height(5.0, 10.0)
assert np.isclose(area_full, np.pi * 25.0), '[TC01] circle_segment_area_from_height 完整圆 FAILED'

# ---- TC02: circle_segment_area_from_height 零高度 ----
area_zero = circle_segment_area_from_height(5.0, 0.0)
assert area_zero == 0.0, '[TC02] circle_segment_area_from_height 零高度 FAILED'

# ---- TC03: circle_segment_area_from_height 半圆对称性 ----
area_half = circle_segment_area_from_height(5.0, 5.0)
assert np.isclose(area_half, 0.5 * np.pi * 25.0, rtol=1e-10), '[TC03] circle_segment_area_from_height 半圆 FAILED'

# ---- TC04: quadrature_on_curved_domain 常数函数积分 ----
def const_func(x, z):
    return 2.0
integral_const = quadrature_on_curved_domain(const_func, (0.0, 1.0), (0.0, 1.0), [], [], n_x=20, n_z=20)
assert np.isclose(integral_const, 2.0, rtol=0.05), '[TC04] quadrature_on_curved_domain 常数函数 FAILED'

# ---- TC05: coastal_boundary_length 半圆弧 ----
blen = coastal_boundary_length([(0.0, 0.0)], [1.0], [np.pi])
assert np.isclose(blen, np.pi, rtol=1e-10), '[TC05] coastal_boundary_length 半圆弧 FAILED'

# ---- TC06: diophantine_nd_nonnegative 基本解枚举 ----
sols = diophantine_nd_nonnegative(np.array([2, 3]), 6)
assert sols.shape[0] == 2, '[TC06] diophantine_nd_nonnegative 解个数 FAILED'
assert np.allclose(np.dot(sols, np.array([2, 3])), np.full(sols.shape[0], 6)), '[TC06] diophantine_nd_nonnegative 约束验证 FAILED'

# ---- TC07: allocate_nutrient_budget min_variance 约束满足 ----
alloc = allocate_nutrient_budget(10, np.array([1, 2, 3]), objective="min_variance")
assert np.dot(np.array([1, 2, 3]), alloc) == 10, '[TC07] allocate_nutrient_budget 约束验证 FAILED'

# ---- TC08: dictionary_encode 往返一致性 ----
np.random.seed(42)
vecs = np.random.rand(5, 3)
dictionary, indices, cr = dictionary_encode(vecs, tol=1e-8)
decoded = dictionary[indices]
assert np.allclose(decoded, vecs, atol=1e-7), '[TC08] dictionary_encode 往返一致性 FAILED'

# ---- TC09: snapshot_matrix 展平形状与值 ----
fields = {'N': np.ones((2, 2)), 'P': np.zeros((2, 2))}
flat = snapshot_matrix(fields)
assert flat.shape == (8,), '[TC09] snapshot_matrix 形状 FAILED'
assert np.allclose(flat, np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])), '[TC09] snapshot_matrix 值 FAILED'

# ---- TC10: bicg_solve 单位矩阵精确解 ----
A_eye = np.eye(5)
b_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_bicg = bicg_solve(A_eye, b_vec, tol=1e-10, max_iter=10)
assert np.allclose(x_bicg, b_vec, atol=1e-8), '[TC10] bicg_solve 单位矩阵 FAILED'

# ---- TC11: SparseNCF 转置矩阵向量乘法 ----
rowcol_t = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]]).T
a_vals_t = np.array([4.0, 4.0, 4.0, -1.0, -1.0])
spm_t = SparseNCF(3, 3, 5, rowcol_t, a_vals_t)
x_t_vec = np.array([1.0, 0.0, 0.0])
y_t_vec = spm_t.mtv(x_t_vec)
expected_t = np.array([4.0, -1.0, 0.0])
assert np.allclose(y_t_vec, expected_t, atol=1e-10), '[TC11] SparseNCF 转置矩阵向量乘法 FAILED'

# ---- TC12: SparseNCF 矩阵向量乘法 ----
rowcol = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]]).T
a_vals = np.array([4.0, 4.0, 4.0, -1.0, -1.0])
spm = SparseNCF(3, 3, 5, rowcol, a_vals)
x_vec = np.array([1.0, 2.0, 3.0])
y_vec = spm.mv(x_vec)
expected = np.array([4.0 * 1.0 - 1.0 * 2.0, 4.0 * 2.0 - 1.0 * 1.0, 4.0 * 3.0])
assert np.allclose(y_vec, expected, atol=1e-10), '[TC12] SparseNCF 矩阵向量乘法 FAILED'

# ---- TC13: create_poisson_stencil 输出形状 ----
A_p = create_poisson_stencil(4, 4, 1.0, 1.0)
assert A_p.shape == (16, 16), '[TC13] create_poisson_stencil 输出形状 FAILED'

# ---- TC14: nelder_mead_optimize 二次函数最小值 ----
def quad_loss(x):
    return (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2
x0_nm = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
theta_opt, fopt, nfev = nelder_mead_optimize(quad_loss, x0_nm, tol=1e-8, max_feval=500)
assert fopt < 1e-4, '[TC14] nelder_mead_optimize 二次函数最小值 FAILED'

# ---- TC15: tsp_descent 四城市距离矩阵 ----
np.random.seed(42)
dist_mat = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
best_path, best_cost = tsp_descent(dist_mat, variation_num=500, seed=42)
assert best_cost <= 4.0, '[TC15] tsp_descent 四城市距离矩阵 FAILED'

# ---- TC16: build_distance_matrix 对称性与对角线 ----
stations = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 5.0]])
dist_mat = build_distance_matrix(stations)
assert np.allclose(dist_mat, dist_mat.T, atol=1e-10), '[TC16] build_distance_matrix 对称性 FAILED'
assert np.all(np.diag(dist_mat) == 0.0), '[TC16] build_distance_matrix 对角线 FAILED'

# ---- TC17: test_quadrature_exactness Legendre五点精确度 ----
x_leg, w_leg = np.polynomial.legendre.leggauss(5)
max_exact = test_quadrature_exactness(x_leg, w_leg, degree_max=11)
assert max_exact == 9, '[TC17] test_quadrature_exactness Legendre五点精确度 FAILED'

# ---- TC18: sparse_grid_quadrature 返回值类型与节点数 ----
result_sg, n_pts = sparse_grid_quadrature(dim=2, max_level=2, func=lambda x: 1.0)
assert np.isscalar(result_sg) and np.isfinite(result_sg), '[TC18] sparse_grid_quadrature 返回值类型 FAILED'
assert n_pts > 0, '[TC18] sparse_grid_quadrature 节点数 FAILED'

# ---- TC19: norm_loo 最大值范数 ----
field = np.array([[1.0, -3.0], [2.0, 0.5]])
ninf = norm_loo(field)
assert ninf == 3.0, '[TC19] norm_loo 最大值范数 FAILED'

# ---- TC20: first_order_sobol_pce 范围与非负性 ----
coeffs = np.array([1.0, 0.3, 0.2, 0.1])
multi_indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
total_var = np.sum(coeffs[1:] ** 2)
S1 = first_order_sobol_pce(coeffs, multi_indices, total_var, dim=2)
assert np.all(S1 >= 0.0), '[TC20] first_order_sobol_pce 非负性 FAILED'
assert np.all(S1 <= 1.0), '[TC20] first_order_sobol_pce 上界 FAILED'
assert np.sum(S1) <= 1.0 + 1e-10, '[TC20] first_order_sobol_pce 总和上界 FAILED'

# ---- TC21: gci_refinement_estimator 收敛阶正性 ----
p_gci, gci_val = gci_refinement_estimator(1.0, 1.5, 2.5, r=2.0)
assert p_gci > 0.0, '[TC21] gci_refinement_estimator 收敛阶 FAILED'
assert gci_val >= 0.0, '[TC21] gci_refinement_estimator GCI非负 FAILED'

# ---- TC22: NPZDEcosystem uptake_rate 非负与有限 ----
eco_test = NPZDEcosystem(nx=4, nz=4, dx=1.0, dz=1.0, dt=3600.0, V_max=1.0, K_N=0.5)
N_test = np.ones((4, 4)) * 2.0
P_test = np.ones((4, 4)) * 0.5
U = eco_test.uptake_rate(N_test, P_test)
assert np.all(U >= 0.0), '[TC22] NPZDEcosystem uptake_rate 非负 FAILED'
assert np.all(np.isfinite(U)), '[TC22] NPZDEcosystem uptake_rate 有限 FAILED'

# ---- TC23: NPZDEcosystem grazing_rate 非负与有限 ----
G = eco_test.grazing_rate(P_test)
assert np.all(G >= 0.0), '[TC23] NPZDEcosystem grazing_rate 非负 FAILED'
assert np.all(np.isfinite(G)), '[TC23] NPZDEcosystem grazing_rate 有限 FAILED'

# ---- TC24: NPZDEcosystem total_nitrogen 守恒为正 ----
tn = eco_test.total_nitrogen()
assert tn > 0.0, '[TC24] NPZDEcosystem total_nitrogen 为正 FAILED'

# ---- TC25: ThermohalineCirculation laplacian 零场为零 ----
ocean_test = ThermohalineCirculation(nx=4, nz=4, Lx=1.0, Lz=1.0, dt=3600.0)
zero_field = np.zeros((4, 4))
L_zero = ocean_test.laplacian(zero_field)
assert np.allclose(L_zero, 0.0, atol=1e-10), '[TC25] ThermohalineCirculation laplacian 零场 FAILED'

# ---- TC26: ThermohalineCirculation get_velocity 零流场 ----
ocean_test2 = ThermohalineCirculation(nx=4, nz=4, Lx=1.0, Lz=1.0, dt=3600.0)
u_vel, w_vel = ocean_test2.get_velocity()
assert np.allclose(u_vel, 0.0, atol=1e-10), '[TC26] ThermohalineCirculation get_velocity u FAILED'
assert np.allclose(w_vel, 0.0, atol=1e-10), '[TC26] ThermohalineCirculation get_velocity w FAILED'

# ---- TC27: LagrangianParticleTransport bilinear_weights 边界行为 ----
np.random.seed(42)
pt = LagrangianParticleTransport(nx=4, nz=4, Lx=1.0, Lz=1.0, nparticles=100, dt=1.0)
i, j, hx, hy = pt.bilinear_weights(0.0, 0.0)
assert 0 <= i <= pt.nx - 2 and 0 <= j <= pt.nz - 2, '[TC27] LagrangianParticleTransport bilinear_weights 边界索引 FAILED'
assert 0.0 <= hx <= 1.0 and 0.0 <= hy <= 1.0, '[TC27] LagrangianParticleTransport bilinear_weights 边界权重 FAILED'
assert hx == 0.0 and hy == 0.0, '[TC27] LagrangianParticleTransport bilinear_weights 原点偏移 FAILED'

# ---- TC28: LagrangianParticleTransport 粒子密度场非负 ----
density = pt.get_particle_density_field()
assert np.all(density >= 0.0), '[TC28] LagrangianParticleTransport 粒子密度场非负 FAILED'

# ---- TC29: stroud_cn_leg_5 权重归一化 ----
pts, wts = stroud_cn_leg_5(4)
if pts is not None:
    assert np.isclose(np.sum(wts), 16.0, rtol=1e-10), '[TC29] stroud_cn_leg_5 权重归一化 FAILED'

# ---- TC30: 集成测试 耦合模拟流程可复现 ----
np.random.seed(123)
ocean_int = ThermohalineCirculation(nx=8, nz=4, Lx=1.0e6, Lz=1.0e3, dt=3600.0)
eco_int = NPZDEcosystem(nx=8, nz=4, dx=ocean_int.dx, dz=ocean_int.dz, dt=3600.0)
for _ in range(3):
    ocean_int.step()
    u_int, w_int = ocean_int.get_velocity()
    eco_int.step(u_int, w_int)
pp_final = eco_int.primary_production()
assert np.isfinite(pp_final) and pp_final >= 0.0, '[TC30] 集成测试 耦合模拟流程 FAILED'

print('\n全部 30 个测试通过!\n')
