"""
main.py
深部地幔密度异常反演的多尺度重力计算框架 —— 统一入口

================================================================================
科学问题：
    基于地表重力异常观测数据，通过多尺度自适应反演方法，
    重建地球深部（上地幔至核幔边界）的三维密度异常结构，
    并耦合热-化学演化模型评估密度异常的时间稳定性。

核心物理模型与数学公式：
================================================================================

1. 重力正演（Newton位理论）
    重力位：
        V(r) = G * integral_V [ rho(r') / |r - r'| ] dV'
    
    垂直重力异常（一阶径向导数）：
        dg_z(r) = -dV/dz = G * integral_V [ rho(r') * (z - z') / |r - r'|^3 ] dV'
    
    球谐展开（全球参考场）：
        V(r,theta,lambda) = (GM/r) * sum_{l=0}^{\infty} sum_{m=-l}^{l}
                            (R/r)^l * C_{lm} * Y_{lm}(theta, lambda)
    
    其中 Y_{lm} 为球谐函数，C_{lm} 为完全正常化球谐系数。

2. 热-密度耦合（热膨胀状态方程）
    rho(T) = rho_0 * [1 - alpha_th * (T - T_0)]
    
    稳态热传导方程：
        nabla . (k * nabla T) + H = 0
    
    瞬态热传导方程（ADI格式）：
        rho*C_p * dT/dt = nabla . (k * nabla T) + H
    
    对流-扩散方程（Lax-Wendroff）：
        d(rho)/dt + d(u*rho)/dz = D * d^2(rho)/dz^2 + S(z,t)

3. Tikhonov正则化反演
n    min_m { ||G*m - d||_2^2 + alpha^2 * ||L*m||_2^2 }
    
    正规方程：
        (G^T G + alpha^2 L^T L) * m = G^T d
    
    L曲线准则（Morozov）：
        ||G*m - d|| = tau * delta，其中 delta 为噪声水平
    
    广义交叉验证（GCV）：
        GCV(alpha) = ||G*m - d||^2 / (N - tr(H))^2

4. 多孔介质方程（验证基准）
    d(u)/dt = nabla^2(u^m)
    
    Barenblatt-Pattle自相似解：
        u(x,t) = (t+delta)^{-beta} * [C - gamma*(x/(t+delta)^{beta})^2]^{1/(m-1)}
        beta = 1/(m+1), gamma = (m-1)/(2*m*(m+1))

5. 数值积分方法
    Keast四面体规则（精度阶 p）：
        integral_{T_ref} f(r) dV = sum_{q=1}^{N_q} w_q * f(r_q) * |det(J)|
    
    Hammersley准蒙特卡洛：
        星差异 D_N^* = O((log N)^d / N)

================================================================================
运行方式：
    python main.py
    （零参数运行，自动执行完整计算流程）
================================================================================
"""

import numpy as np
import sys
import time

# ------------------------------------------------------------------------------
# 导入所有子模块
# ------------------------------------------------------------------------------
from forward_model import (
    prism_gravity_anomaly,
    tetrahedron_gravity_anomaly,
    qmc_gravity_anomaly,
    keast_tetrahedron_nodes_weights,
    composite_forward_model,
    G_CONST
)
from matrix_kernels import (
    r8utt_solve,
    r8utt_inverse,
    build_toeplitz_green_matrix,
    football_combination_count,
    invmod_matrix
)
from inverse_solver import (
    build_sensitivity_matrix,
    tikhonov_solve_dense,
    l_curve_criterion,
    gcv_criterion,
    iterative_tikhonov_cg,
    resolution_matrix_analysis
)
from thermal_solver import (
    jacobi_thermal_2d,
    adi_thermal_2d,
    stiff_thermal_decay,
    euler_explicit_thermal,
    thermal_expansion_density,
    coupled_thermal_density_evolution
)
from convection_model import (
    lax_wendroff_density_convection,
    porous_medium_barenblatt,
    porous_medium_verification,
    density_anomaly_evolution_full
)
from mesh_generator import (
    voronoi_nearest_neighbor,
    cvt_lloyd_iteration,
    adaptive_metric_from_gravity,
    spherical_triangle_histogram,
    adaptive_gravity_mesh
)
from sampling_utils import (
    halton_sequence,
    hammersley_sequence_nd,
    sphere_surface_uniform,
    sphere_distance_statistics,
    generate_gravity_station_network,
    great_circle_distance
)
from parallel_dispatcher import (
    simulate_cuda_indexing,
    task_scheduler_static,
    task_scheduler_dynamic
)


# ------------------------------------------------------------------------------
# 物理常数与地球参考模型
# ------------------------------------------------------------------------------
EARTH_RADIUS = 6371e3          # m
MANTLE_DENSITY_REF = 3300.0    # kg/m^3
CORE_DENSITY_REF = 10500.0     # kg/m^3
THERMAL_EXPANSION = 3e-5       # K^-1
THERMAL_CONDUCTIVITY = 3.0     # W/(m K)
SPECIFIC_HEAT = 1000.0         # J/(kg K)
RADIOGENIC_HEAT = 1e-6         # W/m^3
GRAVITY_ACC = 9.81             # m/s^2


def print_section(title):
    """打印分节标题。"""
    print("\n" + "=" * 78)
    print("  {}".format(title))
    print("=" * 78 + "\n")


def main():
    """主计算流程：零参数运行。"""
    total_start = time.time()
    
    print("\n")
    print("*" * 78)
    print("*  深部地幔密度异常反演的多尺度重力计算框架")
    print("*  Multi-Scale Gravity Computational Framework for Deep Mantle")
    print("*  Density Anomaly Inversion")
    print("*" * 78)
    print("\n")
    
    # ==========================================================================
    # 阶段 1：生成合成地球物理模型与观测网络
    # ==========================================================================
    print_section("阶段 1：合成地球模型与重力测站网络生成")
    
    # 全球重力测站网络（Fibonacci均匀采样）
    n_stations = 120
    stations = generate_gravity_station_network(
        n_stations, method='fibonacci',
        lat_range=(-70, 70), lon_range=(0, 360)
    )
    
    # 转换为笛卡尔坐标（简化平面近似，局部切平面）
    # 投影到以 (0,0) 为中心的局部坐标系，单位 m
    obs_x = np.radians(stations[:, 1]) * EARTH_RADIUS * np.cos(np.radians(stations[:, 0]))
    obs_y = np.radians(stations[:, 0]) * EARTH_RADIUS
    obs_z = np.zeros(n_stations)  # 地表
    obs_points = np.column_stack([obs_x, obs_y, obs_z])
    
    print("  生成 {} 个全球重力测站".format(n_stations))
    
    # 评估测站空间分布均匀性
    mean_d, std_d, min_d, max_d = sphere_distance_statistics(stations)
    print("  测站球面距离统计:")
    print("    平均弦长距离 = {:.4f} (归一化)".format(mean_d))
    print("    标准差       = {:.4f}".format(std_d))
    print("    最小距离     = {:.4f}".format(min_d))
    print("    最大距离     = {:.4f}".format(max_d))
    
    # 球面三角形直方图
    histo, uniformity, expected = spherical_triangle_histogram(
        stations[:, 0], stations[:, 1], n_divisions=4
    )
    print("  球面三角形直方图均匀性指数 = {:.4f} (越小越均匀)".format(uniformity))
    
    # 定义三维密度模型（合成异常体）
    # 模型1：上地幔低速带（深度 100-400 km，密度降低 30 kg/m^3）
    # 模型2：核幔边界热柱（深度 2800-3000 km，密度降低 50 kg/m^3）
    # 模型3：俯冲板片（倾斜高密体，密度增加 80 kg/m^3）
    
    prisms = []
    # TODO(Hole_3a): 定义合成三维密度异常体（棱柱体列表）
    # 每个棱柱体为 tuple: (x1, x2, y1, y2, z1, z2, density_anomaly)
    # 需要构造至少3个具有地球物理意义的异常体：
    #   1. 上地幔低密度柱（深度100-400km，密度降低约30 kg/m^3）
    #   2. 核幔边界热柱（深度2800-3000km，密度降低约50 kg/m^3）
    #   3. 俯冲板片高密体（倾斜分布，密度增加约80 kg/m^3）
    # 注意：异常体的空间范围应与反演网格（Hole_3b）的覆盖范围协调
    raise NotImplementedError("Hole_3a: 合成密度模型棱柱体参数待定义")
    
    # 四面体异常体（模拟不规则地幔柱头部）
    tetra_verts = np.array([
        [0.0, 0.0, -500e3],
        [1e5, 0.0, -600e3],
        [0.0, 1e5, -600e3],
        [0.0, 0.0, -700e3]
    ], dtype=float)
    tetras = [(tetra_verts, -40.0)]
    
    print("  合成密度模型包含:")
    print("    - {} 个棱柱体异常".format(len(prisms)))
    print("    - {} 个四面体异常".format(len(tetras)))
    
    # ==========================================================================
    # 阶段 2：重力正演计算
    # ==========================================================================
    print_section("阶段 2：重力正演计算")
    
    # 使用组合正演模型
    dg_true = composite_forward_model(prisms, tetras, None, obs_points, qmc_samples=1000)
    
    print("  真值重力异常统计:")
    print("    均值  = {:.4f} mGal".format(np.mean(dg_true)))
    print("    标准差= {:.4f} mGal".format(np.std(dg_true)))
    print("    最小值= {:.4f} mGal".format(np.min(dg_true)))
    print("    最大值= {:.4f} mGal".format(np.max(dg_true)))
    
    # 添加合成观测噪声（高斯白噪声，标准差 0.05 mGal，模拟高精度重力仪）
    noise_level = 0.05
    noise = np.random.normal(0.0, noise_level, n_stations)
    dg_obs = dg_true + noise
    
    print("  添加观测噪声: sigma = {:.2f} mGal".format(noise_level))
    snr_val = np.std(dg_true) / (noise_level + 1e-15)
    print("  观测信噪比 SNR = {:.2f} dB".format(
        20.0 * np.log10(snr_val) if snr_val > 0 else -999
    ))
    
    # ==========================================================================
    # 阶段 3：反演网格离散化与灵敏度矩阵构造
    # ==========================================================================
    print_section("阶段 3：反演网格与灵敏度矩阵")
    
    # TODO(Hole_3b): 定义三维反演网格参数
    # 需要定义：
    #   - nx, ny, nz: 网格维度（建议 8x8x6 = 384 个参数）
    #   - dx, dy, dz: 网格间距 [m]（建议 dx=dy=150km, dz=500km）
    #   - grid_centers: (N_param, 3) 网格单元中心坐标 [m]
    #   - grid_volumes: (N_param,) 网格单元体积 [m^3]
    # 注意：
    #   1. 网格中心坐标应覆盖合成异常体的空间范围（与 Hole_3a 协调）
    #   2. 深度坐标 zc 应向下为负
    #   3. grid_centers 和 grid_volumes 应为 numpy 数组
    raise NotImplementedError("Hole_3b: 反演网格离散化参数待定义")
    
    print("  反演网格: {} x {} x {} = {} 个参数".format(nx, ny, nz, n_param))
    print("  网格间距: dx={:.0f}km, dy={:.0f}km, dz={:.0f}km".format(
        dx/1e3, dy/1e3, dz/1e3))
    
    # 构造灵敏度矩阵
    t0 = time.time()
    G = build_sensitivity_matrix(obs_points, grid_centers, grid_volumes)
    t1 = time.time()
    
    print("  灵敏度矩阵构造耗时: {:.3f} s".format(t1 - t0))
    print("  矩阵维度: {} x {}".format(G.shape[0], G.shape[1]))
    print("  矩阵条件数估计: {:.2e}".format(np.linalg.cond(G) if n_param <= 100 else np.inf))
    
    # ==========================================================================
    # 阶段 4：Tikhonov正则化反演
    # ==========================================================================
    print_section("阶段 4：Tikhonov正则化反演")
    
    # L曲线选择正则化参数
    alphas = np.logspace(-4, 2, 20)
    best_alpha_l, res_l, reg_l, curv_l = l_curve_criterion(G, dg_obs, alphas, order=1)
    print("  L曲线最优正则化参数: alpha = {:.4e}".format(best_alpha_l))
    
    # GCV选择正则化参数
    best_alpha_gcv, gcv_vals, res_gcv = gcv_criterion(G, dg_obs, alphas, order=1)
    print("  GCV最优正则化参数: alpha = {:.4e}".format(best_alpha_gcv))
    
    # 使用L曲线参数进行反演
    alpha = best_alpha_l
    
    # 稠密矩阵直接求解
    m_inv_dense, res_dense, reg_dense = tikhonov_solve_dense(G, dg_obs, alpha, order=1)
    print("  稠密反演结果:")
    print("    残差范数     = {:.4f} mGal".format(res_dense))
    print("    正则化项范数 = {:.4f}".format(reg_dense))
    print("    密度异常范围 = [{:.2f}, {:.2f}] kg/m^3".format(
        np.min(m_inv_dense), np.max(m_inv_dense)))
    
    # 共轭梯度迭代求解
    m_inv_cg, cg_iter, cg_res = iterative_tikhonov_cg(G, dg_obs, alpha, order=1,
                                                        max_iter=500, tol=1e-6)
    print("  CG迭代反演:")
    print("    迭代次数     = {}".format(cg_iter))
    print("    最终残差     = {:.4e}".format(cg_res[-1] if len(cg_res) > 0 else 0))
    
    # 分辨率矩阵分析
    R_m, R_d, spread_m, trace_d = resolution_matrix_analysis(G, alpha, order=1)
    print("  分辨率分析:")
    print("    模型分辨率展布 = {:.2f}".format(spread_m))
    print("    数据分辨率迹   = {:.2f}".format(trace_d))
    
    # ==========================================================================
    # 阶段 5：热场计算与热-密度耦合
    # ==========================================================================
    print_section("阶段 5：热场计算与热-密度耦合")
    
    # 二维热传导求解（模拟横截面）
    nx_th, ny_th = 21, 21
    dx_th = 50e3  # 50 km
    dy_th = 50e3
    
    # 非均匀热导率场（地幔过渡带低热导率）
    k_field = np.ones((nx_th, ny_th)) * THERMAL_CONDUCTIVITY
    for i in range(nx_th):
        depth = i * dx_th
        if 400e3 < depth < 700e3:
            k_field[i, :] *= 0.7  # 过渡带低热导率
    
    # 热源场（放射性生热 + 底部热流）
    # 调整为更合理的地球物理参数（uW/m^3量级）
    H_field = np.ones((nx_th, ny_th)) * 2e-8  # 2e-8 W/m^3 = 0.02 uW/m^3
    H_field[-3:, :] += 1e-7  # 核幔边界稍高
    
    T_boundary = {
        'top': 300.0,      # 地表温度 K
        'bottom': 3000.0,  # 核幔边界温度 K
        'left': 1600.0,
        'right': 1600.0
    }
    
    T_steady, iter_jacobi = jacobi_thermal_2d(
        nx_th, ny_th, dx_th, dy_th, k_field, H_field, T_boundary,
        epsilon=1e-6, max_iter=10000
    )
    
    print("  稳态热场 (Jacobi迭代):")
    print("    收敛迭代次数 = {}".format(iter_jacobi))
    print("    温度范围     = [{:.1f}, {:.1f}] K".format(np.min(T_steady), np.max(T_steady)))
    
    # 热膨胀密度修正
    rho_thermal = thermal_expansion_density(
        MANTLE_DENSITY_REF, T_steady, 300.0, THERMAL_EXPANSION
    )
    print("  热膨胀修正密度:")
    print("    密度范围 = [{:.1f}, {:.1f}] kg/m^3".format(np.min(rho_thermal), np.max(rho_thermal)))
    
    # 刚性热衰减（模拟热异常随时间衰减）
    t, T_decay, T_exact = stiff_thermal_decay(
        (0.0, 1e15),  # 0 到 ~30 Myr (以秒计)
        T0=2000.0, lambda_stiff=1e-15, T_ambient=1600.0, n_steps=1000
    )
    error_stiff = np.max(np.abs(T_decay - T_exact))
    print("  刚性热衰减ODE:")
    print("    解析解与数值解最大误差 = {:.6e} K".format(error_stiff))
    
    # ==========================================================================
    # 阶段 6：地幔对流-扩散演化
    # ==========================================================================
    print_section("阶段 6：地幔对流-扩散密度异常演化")
    
    # 垂向密度异常演化
    nz_conv = 41
    z_max = 3e6  # 3000 km
    z = np.linspace(0, z_max, nz_conv)
    dz_conv = z[1] - z[0]
    
    # 密度异常演化（相对于背景密度）
    rho_background = MANTLE_DENSITY_REF
    delta_rho_init = -50.0  # kg/m^3 初始密度异常
    
    # Stokes速度
    from convection_model import stokes_velocity_profile
    eta_mantle = 1e21  # Pa s
    u = stokes_velocity_profile(z, eta_mantle, delta_rho_init, z_max)
    u = np.clip(u, -1e-9, 1e-9)  # 限制速度防止数值不稳定
    print("  Stokes特征速度: u_max = {:.4e} m/s".format(np.max(np.abs(u))))
    
    # 对流-扩散演化（使用密度异常作为变量，不是绝对密度）
    dt_conv = 5e12  # ~150 kyr
    n_steps_conv = 50
    
    def source_func(z_pos, t_pos):
        # 底部热源导致的密度异常生成（极小量级）
        if z_pos > 0.9 * z_max:
            return -1e-4 * np.exp(-((z_pos - z_max)**2) / (2e10))
        return 0.0
    
    # 初始密度异常场
    rho_anomaly_init = np.zeros(nz_conv)
    z_center = z_max / 2.0
    sigma_z = z_max / 10.0
    rho_anomaly_init += delta_rho_init * np.exp(-((z - z_center)**2) / (2.0 * sigma_z**2))
    
    rho_anomaly_final, history_conv = lax_wendroff_density_convection(
        nz_conv, dz_conv, dt_conv, n_steps_conv,
        rho_anomaly_init, u, D_diff=1e-6, source=source_func,
        bc_type='zero_gradient'
    )
    
    # 数值保护：限制异常范围
    # 数值稳定性保护：限制极端异常值
    rho_anomaly_final = np.clip(rho_anomaly_final, -150.0, 150.0)
    
    print("  Lax-Wendroff对流-扩散演化:")
    print("    时间步长   = {:.3e} s (~{:.1f} kyr)".format(dt_conv, dt_conv / (365.25 * 24 * 3600) / 1000))
    print("    总演化时间 = {:.3e} s (~{:.1f} Myr)".format(
        dt_conv * n_steps_conv, dt_conv * n_steps_conv / (365.25 * 24 * 3600) / 1e6))
    print("    初始密度异常范围 = [{:.2f}, {:.2f}] kg/m^3".format(np.min(rho_anomaly_init), np.max(rho_anomaly_init)))
    print("    最终密度异常范围 = [{:.2f}, {:.2f}] kg/m^3".format(np.min(rho_anomaly_final), np.max(rho_anomaly_final)))
    
    # 多孔介质方程验证
    l2_err, linf_err, u_num, u_exact = porous_medium_verification(
        nz=81, z_max=10.0, t_test=2.0, m=2.0, C=1.0, delta=0.1
    )
    print("  多孔介质Barenblatt验证:")
    print("    L2误差   = {:.6e}".format(l2_err))
    print("    Linf误差 = {:.6e}".format(linf_err))
    
    # ==========================================================================
    # 阶段 7：自适应网格与Voronoi分析
    # ==========================================================================
    print_section("阶段 7：自适应网格与Voronoi分析")
    
    # 变度量CVT优化测站分布
    def simple_metric(p):
        return np.eye(2) * (1.0 + 0.5 * np.sin(p[0] / 1e6)**2)
    
    initial_points = np.random.rand(20, 2) * np.array([140.0, 360.0]) + np.array([-70.0, 0.0])
    opt_points, energies = cvt_lloyd_iteration(
        initial_points, simple_metric, n_samples=2000, n_iter=10
    )
    print("  CVT Lloyd迭代能量收敛:")
    print("    初始能量 = {:.4f}".format(energies[0]))
    print("    最终能量 = {:.4f}".format(energies[-1]))
    print("    能量下降比 = {:.2%}".format(1.0 - energies[-1] / (energies[0] + 1e-15)))
    
    # Voronoi最近邻分析
    query_pts = np.random.rand(50, 2) * np.array([140.0, 360.0]) + np.array([-70.0, 0.0])
    dists_vor, idx_vor = voronoi_nearest_neighbor(opt_points, query_pts)
    print("  Voronoi最近邻平均距离 = {:.2f} deg".format(np.mean(dists_vor)))
    
    # ==========================================================================
    # 阶段 8：Keast积分验证与QMC积分
    # ==========================================================================
    print_section("阶段 8：数值积分验证")
    
    # Keast四面体积分验证：积分 f(x,y,z)=1 应等于参考四面体体积 1/6
    nodes, weights = keast_tetrahedron_nodes_weights(order=3)
    integral_1 = np.sum(weights)
    print("  Keast规则体积积分验证:")
    print("    计算值 = {:.10f}".format(integral_1))
    print("    理论值 = {:.10f} (1/6)".format(1.0 / 6.0))
    print("    误差   = {:.2e}".format(abs(integral_1 - 1.0 / 6.0)))
    
    # 积分 f(x,y,z)=x 在参考四面体上，理论值 = 1/24
    integral_x = np.sum(nodes[:, 0] * weights)
    print("  Keast规则 <x> 积分验证:")
    print("    计算值 = {:.10f}".format(integral_x))
    print("    理论值 = {:.10f} (1/24)".format(1.0 / 24.0))
    print("    误差   = {:.2e}".format(abs(integral_x - 1.0 / 24.0)))
    
    # Hammersley序列统计验证
    hseq = hammersley_sequence_nd(3, 1000)
    print("  Hammersley序列统计:")
    print("    均值  = [{:.4f}, {:.4f}, {:.4f}]".format(*np.mean(hseq, axis=0)))
    print("    方差  = [{:.4f}, {:.4f}, {:.4f}]".format(*np.var(hseq, axis=0)))
    
    # ==========================================================================
    # 阶段 9：模运算矩阵与Toeplitz求解验证
    # ==========================================================================
    print_section("阶段 9：特殊矩阵算法验证")
    
    # Toeplitz求解验证
    n_test = 10
    a_test = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b_test = np.ones(n_test)
    x_test = r8utt_solve(n_test, a_test, b_test)
    A_dense = np.zeros((n_test, n_test))
    for i in range(n_test):
        for j in range(i, n_test):
            A_dense[i, j] = a_test[j - i]
    x_exact = np.linalg.solve(A_dense, b_test)
    toeplitz_err = np.max(np.abs(x_test - x_exact))
    print("  上三角Toeplitz求解验证:")
    print("    最大误差 = {:.2e}".format(toeplitz_err))
    
    # 模运算矩阵求逆验证
    mat_mod = np.array([[1, 0], [0, 1]], dtype=int)
    rmod = np.array([5, 5])
    cmod = np.array([5, 5])
    imat, ifault = invmod_matrix(mat_mod, rmod, cmod)
    print("  模运算矩阵求逆验证:")
    print("    输入矩阵 = [[1,0],[0,1]] mod 5")
    print("    逆矩阵   = {}".format(imat.tolist()))
    print("    错误标志 = {} (0=成功)".format(ifault))
    
    # 足球组合计数（球谐参数组合数）
    combo_counts = football_combination_count(12)
    print("  球谐阶数组合数 (football_dynamic迁移):")
    print("    最大12阶的组合方式数 = {}".format(combo_counts[-1]))
    
    # ==========================================================================
    # 阶段 10：CUDA并行调度模拟
    # ==========================================================================
    print_section("阶段 10：并行调度策略模拟")
    
    task_map = simulate_cuda_indexing(
        grid_dim=(2, 2, 1),
        block_dim=(4, 4, 1),
        n_tasks=256
    )
    total_threads = 2 * 2 * 1 * 4 * 4 * 1
    print("  CUDA风格任务分配:")
    print("    Grid 维度  = (2, 2, 1)")
    print("    Block 维度 = (4, 4, 1)")
    print("    总线程数   = {}".format(total_threads))
    print("    总任务数   = 256")
    print("    每个线程平均任务数 = {:.1f}".format(256.0 / total_threads))
    
    static_ranges = task_scheduler_static(256, 4)
    print("  静态任务调度:")
    for w, (s, e) in enumerate(static_ranges):
        print("    Worker {}: 任务 {}-{} (共{}个)".format(w, s, e - 1, e - s))
    
    # ==========================================================================
    # 阶段 11：结果综合与误差分析
    # ==========================================================================
    print_section("阶段 11：综合结果与误差分析")
    
    # 反演结果与真值对比（在观测点处的拟合）
    dg_pred = G @ m_inv_dense
    rms_misfit = np.sqrt(np.mean((dg_obs - dg_pred)**2))
    rms_true_misfit = np.sqrt(np.mean((dg_true - dg_pred)**2))
    
    print("  反演拟合统计:")
    print("    RMS数据拟合残差    = {:.4f} mGal".format(rms_misfit))
    print("    RMS真值拟合残差    = {:.4f} mGal".format(rms_true_misfit))
    print("    数据方差解释率     = {:.2%}".format(
        1.0 - rms_misfit**2 / (np.var(dg_obs) + 1e-15)))
    
    # 反演结果的空间统计
    print("  反演密度异常统计:")
    print("    均值  = {:.2f} kg/m^3".format(np.mean(m_inv_dense)))
    print("    标准差= {:.2f} kg/m^3".format(np.std(m_inv_dense)))
    print("    最小值= {:.2f} kg/m^3".format(np.min(m_inv_dense)))
    print("    最大值= {:.2f} kg/m^3".format(np.max(m_inv_dense)))
    
    # 总耗时
    total_time = time.time() - total_start
    print("\n" + "=" * 78)
    print("  总计算耗时: {:.2f} 秒".format(total_time))
    print("=" * 78)
    print("\n  程序正常结束。")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] 程序运行中出现异常: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
