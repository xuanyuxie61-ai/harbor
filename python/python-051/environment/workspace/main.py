
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




    print("\n[1] 初始化热盐环流与生态系统模型...")
    nx, nz = 32, 16
    Lx, Lz = 5.0e6, 4.0e3
    dt = 43200.0
    n_steps = 20

    ocean = ThermohalineCirculation(nx=nx, nz=nz, Lx=Lx, Lz=Lz, dt=dt)
    eco = NPZDEcosystem(nx=nx, nz=nz, dx=ocean.dx, dz=ocean.dz, dt=dt,
                        V_max=1.0, K_N=0.5, I_opt=50.0, g_max=0.6, K_P=0.5)
    particles = LagrangianParticleTransport(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                                            nparticles=2000, dt=dt)




    print("\n[2] 执行耦合时间积分（{} 步，每步 {:.1f} 小时）...".format(n_steps, dt / 3600.0))
    pp_history = []
    total_n_history = []

    for step in range(n_steps):

        ocean.step()
        u, w = ocean.get_velocity()


        eco.step(u, w)


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




    print("\n[3] 稀疏线性代数模块验证...")

    b_test = np.random.randn(nx * nz)
    A_poisson = create_poisson_stencil(nx, nz, ocean.dx, ocean.dz)
    x_bicg = bicg_solve(A_poisson, b_test, tol=1e-8, max_iter=300)
    res_bicg = np.linalg.norm(A_poisson.dot(x_bicg) - b_test)
    print("  BiCG 残差 ||Ax-b|| = {:.4e}".format(res_bicg))


    n_toep = 8
    a_row = np.array([2.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    T_inv = r8sto_inverse(n_toep, a_row)

    T = np.zeros((n_toep, n_toep))
    for i in range(n_toep):
        for j in range(n_toep):
            T[i, j] = a_row[abs(i - j)]
    I_approx = T_inv.dot(T)
    err_toep = np.linalg.norm(I_approx - np.eye(n_toep))
    print("  Toeplitz 逆误差 ||T^{{-1}}T - I|| = {:.4e}".format(err_toep))


    rowcol = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 0]]).T
    a_vals = np.array([4.0, 4.0, 4.0, -1.0, -1.0])
    spm = SparseNCF(3, 3, 5, rowcol, a_vals)
    x_vec = np.array([1.0, 2.0, 3.0])
    y_vec = spm.mv(x_vec)
    print("  NCF 稀疏乘法结果 = {}".format(y_vec))




    print("\n[4] 稀疏网格不确定性量化（生态参数敏感性）...")

    def biological_flux_func(xi):


        raise NotImplementedError("Hole 2: 请实现 biological_flux_func")

    result_sg, n_pts = sparse_grid_quadrature(dim=2, max_level=3, func=biological_flux_func)
    print("  稀疏网格积分（level=3）: E[PP] = {:.4e}, 节点数 = {}".format(result_sg, n_pts))


    pts, wts = stroud_cn_leg_5(4)
    if pts is not None:
        print("  Stroud CN:5-1 规则（dim=4）: 节点数 = {}".format(len(wts)))


    x_cc, w_cc = np.polynomial.legendre.leggauss(5)
    max_exact = test_quadrature_exactness(x_cc, w_cc, degree_max=11)
    print("  Gauss-Legendre 5 点规则精确度: p = {}".format(max_exact))




    print("\n[5] Nelder-Mead 生态系统参数标定...")


    true_pp = biological_flux_func(np.array([0.0, 0.0]))

    def loss(theta):


        raise NotImplementedError("Hole 3: 请实现 loss")

    x0 = np.array([[0.8, 0.3], [1.2, 0.7], [1.0, 0.5]])
    theta_opt, fopt, nfev = nelder_mead_optimize(loss, x0, tol=1e-4, max_feval=200)
    print("  最优参数: V_max = {:.4f}, K_N = {:.4f}".format(theta_opt[0], theta_opt[1]))
    print("  目标函数值 = {:.4e}, 评估次数 = {}".format(fopt, nfev))




    print("\n[6] TSP 最优 AUV 采样路径规划...")
    stations = generate_sampling_stations(n_stations=12, Lx=Lx, Lz=Lz, depth_min=100.0)
    dist_mat = build_distance_matrix(stations)
    best_path, best_cost = tsp_descent(dist_mat, variation_num=1000, seed=42)
    print("  采样站数 = 12, TSP 近似最优成本 = {:.2e} m".format(best_cost))
    print("  路径顺序前5站: {}".format(best_path[:5]))




    print("\n[7] 曲边海岸几何与区域积分...")
    R_bay = 8.0e5
    h_bay = 0.3 * R_bay
    area_bay = circle_segment_area_from_height(R_bay, h_bay)
    print("  海湾圆缺面积 = {:.4e} m²".format(area_bay))


    def biomass_density(x, z):
        return 0.1 * np.exp(-((x - Lx / 2) ** 2) / (2 * (1e6) ** 2)) + 0.05

    arc_centers = [(1.5e6, 1.0e3)]
    arc_radii = [-5.0e5]
    biomass_total = quadrature_on_curved_domain(biomass_density, (0.0, Lx), (0.0, Lz),
                                                arc_centers, arc_radii, n_x=16, n_z=8)
    print("  曲边域生物量积分 = {:.4e} mmol N".format(biomass_total))


    boundary_len = coastal_boundary_length(arc_centers, arc_radii, [np.pi / 2])
    print("  海岸边界长度 = {:.2e} m".format(boundary_len))




    print("\n[8] 离散生态资源分配与状态压缩...")
    budget = 100
    demands = np.array([2, 3, 5, 7, 11])
    alloc = allocate_nutrient_budget(budget, demands, objective="min_variance")
    print("  氮预算 {} mmol 分配给 {} 个功能组:".format(budget, len(demands)))
    print("  分配方案 = {}, 总和验证 = {}".format(alloc, np.dot(demands, alloc)))


    sols = diophantine_nd_nonnegative(demands, budget)
    print("  可行解总数 = {}".format(sols.shape[0]))


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




    print("\n[9] 稳定性与收敛性分析...")
    loo_psi = norm_loo(ocean.psi)
    loo_omega = norm_loo(ocean.omega)
    print("  L∞(ψ) = {:.4e} m²/s, L∞(ω) = {:.4e} 1/s".format(loo_psi, loo_omega))


    if len(pp_history) >= 3:
        p_gci, gci_val = gci_refinement_estimator(pp_history[-1], pp_history[-2], pp_history[-3], r=1.0)
        print("  GCI 估计: p = {:.3f}, GCI = {:.4e}".format(p_gci, gci_val))


    coeffs = np.array([1.0, 0.3, 0.2, 0.1])
    multi_indices = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    total_var = np.sum(coeffs[1:] ** 2)
    S1 = first_order_sobol_pce(coeffs, multi_indices, total_var, dim=2)
    print("  示例 Sobol' 一阶指数: S1 = [{:.3f}, {:.3f}]".format(S1[0], S1[1]))




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
