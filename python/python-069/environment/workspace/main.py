"""
================================================================================
森林冠层光合与碳通量耦合模拟系统 (Forest Canopy Photosynthesis & Carbon Flux)
================================================================================
统一入口：零参数运行，完成从冠层结构构建到碳通量输出的完整流程。

科学问题：三维异质性森林冠层中，光合-呼吸-碳分配的耦合时空演化。
"""
import numpy as np
import time

# 导入各模块
import canopy_geometry
import radiation_transfer
import leaf_angle_sampling
import microclimate_fem
import photosynthesis_model
import cvt_leaf_distribution
import soil_carbon_flux
import uncertainty_quantification
import co2_diffusion
import carbon_allocation
import environment_response
import data_assimilation
import boundary_flux
import optimization_parser


def main():
    np.random.seed(42)
    t0 = time.time()
    print("=" * 70)
    print("  森林冠层光合与碳通量耦合模拟系统")
    print("  Forest Canopy Photosynthesis & Carbon Flux Model")
    print("=" * 70)

    # ========================================================================
    # 1. 冠层几何结构 (canopy_geometry)
    # ========================================================================
    print("\n[1/14] 构建冠层几何结构 ...")
    canopy_height = 20.0
    crown_radius = 5.0
    n_sub = 8
    lai_max = 4.5
    grid, lai_vals, vertices = canopy_geometry.build_canopy_grid(
        canopy_height, crown_radius, n_sub, lai_max)
    print(f"       冠层网格点数: {grid.shape[0]}, 顶点数: {vertices.shape[0]}")

    # 3D LAI 场
    points_3d, lai_3d = canopy_geometry.canopy_volume_lai_3d(
        canopy_height, crown_radius, n_vert=20, n_horiz=20, lai_max=lai_max)
    print(f"       3D LAI 采样点数: {points_3d.shape[0]}")

    # ========================================================================
    # 2. 辐射传输 (radiation_transfer)
    # ========================================================================
    print("\n[2/14] 计算辐射传输与光环境 ...")
    i0 = 1200.0  # W/m^2 冠层上方入射辐射
    k_ext = 0.5
    sharpness = 1.3
    Xr, Yr, I_enhanced = radiation_transfer.radiation_2d_grid(
        grid, vertices, i0, k_ext, lai_max, canopy_height, sharpness, resolution=30)
    print(f"       辐射场分辨率: {Xr.shape}")
    print(f"       冠层底部光强: {I_enhanced[-1, :].mean():.2f} W/m^2")

    # G 函数表 (leaf_angle_sampling)
    theta_s_range = np.linspace(0.0, np.pi / 2, 10)
    _, g_vals = leaf_angle_sampling.g_function_table(
        theta_s_range, n_samples=5000, theta_l_mean=np.pi / 4, sigma_theta=np.pi / 6)
    print(f"       G(theta_s) 范围: [{g_vals.min():.3f}, {g_vals.max():.3f}]")

    # ========================================================================
    # 3. 微气候 FEM (microclimate_fem)
    # ========================================================================
    print("\n[3/14] 有限元求解冠层微气候 ...")
    # 构建简化三角形网格（仅演示 FEM 流程）
    n_nodes = 16
    node_xy = np.zeros((n_nodes, 2), dtype=float)
    idx = 0
    for j in range(4):
        for i in range(4):
            node_xy[idx, 0] = i * crown_radius / 3.0 - crown_radius / 2.0
            node_xy[idx, 1] = j * canopy_height / 3.0
            idx += 1

    # 4 个 T3 线性三角形单元（16 节点 4x4 网格）
    element_node_t3 = np.array([
        [0, 1, 5],
        [1, 6, 5],
        [1, 2, 6],
        [2, 7, 6],
        [5, 6, 10],
        [6, 11, 10],
        [6, 7, 11],
        [7, 12, 11],
        [10, 11, 15],
        [11, 15, 10],
        [11, 12, 15],
        [12, 15, 11],
        [2, 3, 7],
        [3, 8, 7],
        [7, 8, 12],
        [8, 13, 12],
        [3, 4, 8],
        [4, 9, 8]
    ], dtype=int).T
    # 只保留有效索引
    valid_elements = []
    for e in range(element_node_t3.shape[1]):
        if np.all(element_node_t3[:, e] < n_nodes):
            valid_elements.append(e)
    element_node_t3 = element_node_t3[:, valid_elements]
    node_boundary = np.zeros(n_nodes, dtype=bool)
    for i in range(n_nodes):
        x, y = node_xy[i]
        if abs(x + crown_radius / 2.0) < 0.1 or abs(x - crown_radius / 2.0) < 0.1 or \
           abs(y) < 0.1 or abs(y - canopy_height) < 0.1:
            node_boundary[i] = True
    node_boundary_t3 = node_boundary

    alpha_diff = 1.0e-4
    dt_fem = 60.0
    n_steps_fem = 5
    t_initial = 25.0
    t_ambient = 28.0
    q_rad_profile = [500.0 * (1.0 + 0.3 * np.sin(step * 0.5)) for step in range(n_steps_fem)]
    gs_profile = [0.02] * n_steps_fem
    ea = 1.5

    # 检查 element_node_t3 是否都在范围内
    if np.max(element_node_t3) < n_nodes and np.min(element_node_t3) >= 0:
        try:
            temp_results = microclimate_fem.solve_microclimate(
                node_xy, element_node_t3, node_boundary_t3,
                alpha_diff, dt_fem, n_steps_fem, t_initial, t_ambient,
                q_rad_profile, gs_profile, ea)
            final_temp = temp_results[-1]
            print(f"       最终温度范围: [{final_temp.min():.2f}, {final_temp.max():.2f}] °C")
        except Exception as e:
            print(f"       FEM 求解出现异常（网格简化导致），使用解析近似: {e}")
            final_temp = np.full(n_nodes, t_ambient + 2.0)
    else:
        print("       FEM 网格索引越界，使用解析温度近似")
        final_temp = np.full(n_nodes, t_ambient + 2.0)

    # ========================================================================
    # 4. 光合模型 (photosynthesis_model)
    # ========================================================================
    print("\n[4/14] Farquhar 光合生化模型计算 ...")
    # TODO: 准备光合模型参数并调用 farquhar_photosynthesis，
    # 然后计算净光合速率 A_n、Rubisco 限制 W_c、RuBP 限制 W_j
    # 并计算温度敏感性 dA_n/dT
    raise NotImplementedError("Hole 3a: 请补全光合模型调用与参数准备")

    # ========================================================================
    # 5. CVT 叶片分布优化 (cvt_leaf_distribution)
    # ========================================================================
    print("\n[5/14] CVT 优化叶片空间分布 ...")
    g_opt, e_hist, m_hist = cvt_leaf_distribution.canopy_cvt_optimization(
        canopy_height, crown_radius, n_clusters=30, it_num=10, s_num=15)
    print(f"       优化后叶片簇数: {g_opt.shape[0]}")
    print(f"       最终能量: {e_hist[-1]:.4e}, 平均运动: {m_hist[-1]:.4e}")

    # ========================================================================
    # 6. 土壤碳通量 (soil_carbon_flux)
    # ========================================================================
    print("\n[6/14] 高阶求积计算土壤-冠层碳通量 ...")
    p_rule = 5
    nq, xq, yq, wq = soil_carbon_flux.quadrilateral_witherden_rule(p_rule)
    corners = np.array([[-crown_radius, 0.0],
                        [crown_radius, 0.0],
                        [crown_radius, canopy_height],
                        [-crown_radius, canopy_height]], dtype=float)

    def lai_func(x, y):
        return canopy_geometry.canopy_lai_profile(float(y), canopy_height, lai_max)

    def rd_func(x, y):
        return rd

    # TODO: 基于光合结果计算碳通量
    # 1. 冠层呼吸 = 暗呼吸 R_d * LAI_max
    # 2. 调用 soil_carbon_flux.lloyd_taylor_soil_respiration 计算土壤呼吸
    # 3. GPP = max(A_n * LAI_max, 0)
    # 4. NEE = 冠层呼吸 + 土壤呼吸 - GPP
    raise NotImplementedError("Hole 3b: 请补全碳通量综合计算")

    # ========================================================================
    # 7. CO2 反应扩散 (co2_diffusion)
    # ========================================================================
    print("\n[7/14] CO2 反应扩散模拟 ...")
    J_grid = 16
    h_grid = crown_radius * 2.0 / J_grid
    D_co2 = 0.01
    dt_co2 = 10.0
    n_steps_co2 = 5
    C0 = 400.0
    R_soil_co2 = 2.0
    V_max_co2 = 0.02
    K_m_co2 = 150.0
    LAI_grid = np.random.rand(J_grid, J_grid) * lai_max * 0.5
    co2_results = co2_diffusion.co2_diffusion_solver(
        J_grid, h_grid, D_co2, dt_co2, n_steps_co2,
        C0, R_soil_co2, V_max_co2, K_m_co2, LAI_grid)
    final_co2 = co2_results[-1]
    print(f"       最终 CO2 浓度范围: [{final_co2.min():.2f}, {final_co2.max():.2f}] umol/mol")

    # ========================================================================
    # 8. 不确定性量化 (uncertainty_quantification)
    # ========================================================================
    print("\n[8/14] 稀疏网格不确定性量化 ...")
    def gpp_model(params):
        """params: (n, 2) [V_cmax, J_max]"""
        vc = params[:, 0]
        jm = params[:, 1]
        out = np.zeros(len(vc))
        for i in range(len(vc)):
            try:
                an_i, _, _, _, _ = photosynthesis_model.farquhar_photosynthesis(
                    ci, oi, t_k, i_abs, vcmax_25=vc[i], jmax_25=jm[i])
                out[i] = max(an_i, 0.0) * lai_max
            except Exception:
                out[i] = 0.0
        return out

    mean_gpp, var_gpp, std_gpp = uncertainty_quantification.propagate_uncertainty(
        gpp_model, dim_num=2, level_max=2,
        param_means=[80.0, 136.0], param_stds=[15.0, 25.0])
    print(f"       GPP 均值: {mean_gpp:.3f}, 标准差: {std_gpp:.3f} umol/m^2/s")

    # ========================================================================
    # 9. 碳分配 (carbon_allocation)
    # ========================================================================
    print("\n[9/14] 贪心碳分配 ...")
    c_total = max(an * lai_max, 0.0) * 12.0 * 3600.0 / 1e6  # 转换为 gC/m^2/day
    demands = {'leaf': c_total * 0.4, 'stem': c_total * 0.3,
               'root': c_total * 0.2, 'storage': c_total * 0.2}
    biomass = {'leaf': 500.0, 'stem': 2000.0, 'root': 800.0}
    costs = {'leaf': 1.5, 'stem': 2.0, 'root': 1.8, 'storage': 1.0}
    allocated = carbon_allocation.greedy_carbon_allocation(
        c_total, demands, biomass, costs, eta=0.75)
    eff = carbon_allocation.compute_allocation_efficiency(allocated, demands)
    print(f"       总碳收入: {c_total:.4f} gC/m^2/day")
    print(f"       分配结果: {allocated}")
    print(f"       分配效率: {eff:.3f}")

    # ========================================================================
    # 10. 环境响应插值 (environment_response)
    # ========================================================================
    print("\n[10/14] 环境响应插值 ...")
    tables = environment_response.build_response_tables()
    t_c = 25.0
    vpd_val = 1.5
    theta = 0.30
    env_factor = environment_response.compute_environmental_factor(t_c, vpd_val, theta, tables)
    print(f"       T={t_c}°C, VPD={vpd_val}kPa, SWC={theta}: 环境因子 = {env_factor:.4f}")

    # ========================================================================
    # 11. 数据同化噪声 (data_assimilation)
    # ========================================================================
    print("\n[11/14] 观测数据噪声模拟 ...")
    true_flux = np.array([an] * 10)
    noisy_flux = data_assimilation.add_gaussian_noise(true_flux, alpha=0.05, beta=0.5)
    noisy_flux_spike = data_assimilation.add_spike_noise(noisy_flux, level=0.05, magnitude=5.0)
    # 卡尔曼滤波
    x_a, P_a = data_assimilation.simple_kalman_update(
        an, 4.0, noisy_flux_spike[0], 1.0, 2.0)
    print(f"       真实通量均值: {true_flux.mean():.3f}")
    print(f"       噪声通量均值: {noisy_flux_spike.mean():.3f}")
    print(f"       卡尔曼分析值: {x_a:.3f}, 分析方差: {P_a:.3f}")

    # ========================================================================
    # 12. 边界通量蒙特卡洛 (boundary_flux)
    # ========================================================================
    print("\n[12/14] 六边形边界碳通量蒙特卡洛估算 ...")
    dc_dn = lambda x, y: 5.0 * np.exp(-(x ** 2 + y ** 2))
    lateral_flux = boundary_flux.estimate_lateral_flux(2000, 0.02, dc_dn)
    print(f"       侧边界通量: {lateral_flux:.4f} umol/m^2/s")

    # ========================================================================
    # 13. 优化解析 (optimization_parser)
    # ========================================================================
    print("\n[13/14] 碳分配线性规划解析 ...")
    coeffs = {'leaf': 1.2, 'stem': 1.0, 'root': 0.9, 'storage': 0.7}
    maint = {'leaf': c_total * 0.1, 'stem': c_total * 0.05,
             'root': c_total * 0.05, 'storage': 0.0}
    sol_lp, obj_lp = optimization_parser.solve_carbon_lp(c_total, coeffs, maint)
    shadows = optimization_parser.shadow_prices(c_total, coeffs, maint)
    print(f"       LP 最优解: {sol_lp}")
    print(f"       目标值: {obj_lp:.4f}")
    print(f"       影子价格: {shadows}")

    # ========================================================================
    # 14. 综合输出与总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("  模拟结果汇总")
    print("=" * 70)
    print(f"  冠层高度: {canopy_height} m, 冠幅半径: {crown_radius} m")
    print(f"  最大 LAI: {lai_max}")
    print(f"  净光合速率 (A_n): {an:.3f} umol/m^2/s")
    print(f"  总初级生产力 (GPP): {gpp:.3f} umol/m^2/s")
    print(f"  冠层呼吸 (R_c): {resp_canopy:.3f} umol/m^2/s")
    print(f"  土壤呼吸 (R_s): {resp_soil:.3f} umol/m^2/s")
    print(f"  净生态系统交换量 (NEE): {nee:.3f} umol/m^2/s")
    print(f"  GPP 不确定性 (1-sigma): {std_gpp:.3f} umol/m^2/s")
    print(f"  碳分配效率: {eff:.3f}")
    print(f"  环境限制因子: {env_factor:.3f}")
    print(f"  运行时间: {time.time() - t0:.3f} s")
    print("=" * 70)
    print("  模拟完成，无报错。")
    print("=" * 70)

    # 返回关键结果字典（便于外部调用）
    return {
        'an': an,
        'resp_canopy': resp_canopy,
        'resp_soil': resp_soil,
        'nee': nee,
        'gpp_mean': mean_gpp,
        'gpp_std': std_gpp,
        'allocation': allocated,
        'env_factor': env_factor,
        'lateral_flux': lateral_flux,
        'lp_solution': sol_lp
    }


if __name__ == "__main__":
    main()
