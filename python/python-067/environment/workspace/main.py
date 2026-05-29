# -*- coding: utf-8 -*-
"""
main.py
裂隙介质渗流与示踪试验数值模拟系统 — 统一入口

本项目基于15个科研代码项目的核心算法融合，
围绕水文地质前沿领域：裂隙介质渗流与示踪试验，
构建了一个博士级科学计算系统。

运行方式：
    python main.py

零参数可运行，内置默认物理参数和模拟配置。
"""

import numpy as np
import sys
import time

from random_generator import MiddleSquareGenerator
from fracture_network import FractureNetwork
from geometry_parser import OBJGeometryParser
from mesh_generator import MeshGenerator
from sinc_interpolator import SincInterpolator
from hydraulic_solver import HydraulicSolver
from flow_integrator import GegenbauerQuadrature, FlowIntegrator
from transport_solver import TransportSolver
from inverse_model import InverseModel
from uncertainty_quant import UncertaintyQuantification


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_fracture_network_generation():
    """步骤1：生成裂隙网络"""
    print_section("步骤 1: 分形裂隙网络生成")

    # 使用 IFS 分形 + 网格状态模型
    network = FractureNetwork(
        domain_size=(50.0, 50.0),
        nx=40, ny=40,
        seed=1234
    )

    result = network.generate_full_network(
        n_fracture_points=8000,
        base_aperture=1.0e-4
    )

    print(f"  模拟域尺寸: {result['domain_size']} m")
    print(f"  网格分辨率: {result['resolution']}")
    print(f"  裂隙孔隙度: {result['porosity']:.4f}")
    print(f"  等效渗透率: {result['equivalent_permeability']:.4e} m²")
    print(f"  是否存在渗流路径: {result['percolates']}")
    print(f"  迂曲度: {result['tortuosity']:.4f}")

    return network, result


def run_geometry_parsing():
    """步骤2：解析三维裂隙几何"""
    print_section("步骤 2: 三维裂隙几何解析")

    parser = OBJGeometryParser()
    obj_text = parser.generate_sample_fracture_obj(
        size=1.0, amplitude=0.01, n_segments=10
    )
    geo = parser.parse_string(obj_text)

    print(f"  顶点数: {geo['n_vertices']}")
    print(f"  面片数: {geo['n_faces']}")
    print(f"  总表面积: {parser.total_surface_area():.6f} m²")
    print(f"  表面粗糙度 (σ_z): {parser.roughness_coefficient():.6f} m")

    b_est = parser.mean_aperture_estimate(volume=0.001)
    print(f"  估算平均开度: {b_est:.6e} m")

    return parser


def run_mesh_generation():
    """步骤3：生成计算网格"""
    print_section("步骤 3: CVT 计算网格生成")

    mesh = MeshGenerator(domain=(0.0, 50.0, 0.0, 50.0))

    # CVT 优化采样
    mesh.cvt_relaxation(n_points=200, n_iterations=8, n_samples=3000)
    mesh.delaunay_triangulation()

    # 网格升级与细化
    mesh.upgrade_to_quadratic()
    mesh.refine_mesh(n_refinements=1)

    stats = mesh.mesh_statistics()
    print(f"  CVT 节点数: {stats['n_points']}")
    print(f"  三角形数: {stats['n_triangles']}")
    print(f"  二次元节点数: {stats['n_quadratic_nodes']}")
    print(f"  平均网格质量: {stats.get('mean_quality', 0):.4f}")

    return mesh


def run_sinc_interpolation():
    """步骤4：Sinc 谱插值验证"""
    print_section("步骤 4: Sinc 谱插值验证")

    interpolator = SincInterpolator()

    # 测试函数: f(x) = exp(-x^2)
    x_grid = np.linspace(-3, 3, 31)
    f_vals = np.exp(-x_grid ** 2)
    x_query = np.linspace(-2.5, 2.5, 50)

    f_interp = SincInterpolator.interpolate_1d(x_grid, f_vals, x_query)
    f_exact = np.exp(-x_query ** 2)
    error = np.max(np.abs(f_interp - f_exact))

    print(f"  网格点数: {len(x_grid)}")
    print(f"  查询点数: {len(x_query)}")
    print(f"  最大插值误差: {error:.4e}")

    # 导数插值验证
    df_exact = -2.0 * x_query * np.exp(-x_query ** 2)
    df_interp = SincInterpolator.derivative_1d(x_grid, f_vals, x_query)
    derror = np.max(np.abs(df_interp - df_exact))
    print(f"  导数插值误差: {derror:.4e}")

    return interpolator


def run_hydraulic_simulation(network: FractureNetwork):
    """步骤5：水力压力场求解"""
    print_section("步骤 5: 水力压力场求解")

    ny, nx = network.ny, network.nx
    dx = network.dx
    dy = network.dy

    solver = HydraulicSolver(nx=nx, ny=ny, dx=dx, dy=dy)

    # 传导系数场
    T = network.transmissivity.copy()
    # 避免零传导系数导致奇异，设置背景传导系数
    T = np.clip(T, 1.0e-12, None)

    # Gauss-Seidel 求解
    h_gs = solver.solve_gauss_seidel(
        T,
        h_boundary={'left': 10.0, 'right': 0.0, 'top': 5.0, 'bottom': 5.0},
        max_iter=5000,
        tol=1.0e-8,
        omega=1.5
    )

    print(f"  网格: {nx} x {ny}")
    print(f"  水头范围: [{h_gs.min():.4f}, {h_gs.max():.4f}] m")

    # 共轭梯度法求解对比
    h_cg = solver.solve_conjugate_gradient(
        T,
        h_boundary={'left': 10.0, 'right': 0.0, 'top': 5.0, 'bottom': 5.0},
        tol=1.0e-10
    )
    cg_error = np.max(np.abs(h_cg - h_gs))
    print(f"  GS 与 CG 解的最大差异: {cg_error:.4e} m")

    # 流速计算
    porosity = network.connectivity.astype(float) * 0.3 + 0.01
    vx, vy = solver.compute_velocity(T, porosity)
    v_mag = np.sqrt(vx ** 2 + vy ** 2)
    print(f"  最大达西流速: {v_mag.max():.4e} m/s")

    # 边界流量
    flow = solver.compute_flow_rate(T)
    print(f"  左侧流入: {flow['Q_left']:.4e} m³/s")
    print(f"  右侧流出: {flow['Q_right']:.4e} m³/s")
    print(f"  净流量: {flow['Q_net']:.4e} m³/s")

    return solver, vx, vy


def run_flow_integration(network: FractureNetwork, solver: HydraulicSolver):
    """步骤6：流量数值积分与通量分析"""
    print_section("步骤 6: 高斯-盖根堡尔流量数值积分")

    # 盖根堡尔求积验证：积分抛物线速度剖面
    quad = GegenbauerQuadrature(order=16, alpha=0.0, a=0.0, b=1.0e-4)

    dp_dx = -100.0  # Pa/m
    b = 1.0e-4      # m
    Q_per_width = quad.integrate_parabolic_profile(dp_dx, b)

    # 解析解
    mu = 1.0e-3
    Q_analytical = -(b ** 3 / (12.0 * mu)) * dp_dx

    print(f"  求积阶数: {quad.order}")
    print(f"  数值积分结果: {Q_per_width:.4e} m²/s")
    print(f"  解析解: {Q_analytical:.4e} m²/s")
    print(f"  相对误差: {abs(Q_per_width - Q_analytical) / abs(Q_analytical):.4e}")

    # 突破曲线矩计算
    integrator = FlowIntegrator()
    times = np.linspace(0, 3600, 100)
    # 模拟高斯型突破曲线
    t_mean = 1800.0
    sigma_t = 300.0
    C = np.exp(-0.5 * ((times - t_mean) / sigma_t) ** 2)

    moments = integrator.breakthrough_curve_moments(times, C)
    print(f"  突破曲线零阶矩 (总质量): {moments['M0']:.4e}")
    print(f"  平均突破时间: {moments['t_mean']:.2f} s")
    print(f"  时间标准差: {moments['std']:.2f} s")
    print(f"  偏度: {moments['skewness']:.4f}")

    # 弥散度估算
    L = 50.0
    v = 0.001
    alpha_L = integrator.dispersivity_from_moments(
        moments['t_mean'], moments['variance'], L, v
    )
    print(f"  估算纵向弥散度: {alpha_L:.4e} m")

    # Peclet 数
    D = alpha_L * v + 1.0e-9
    Pe = integrator.peclet_number(v, L, D)
    print(f"  Peclet 数: {Pe:.2f}")

    # Reynolds 数
    Re = integrator.reynolds_number(v, b)
    print(f"  Reynolds 数: {Re:.4f}")

    return quad, integrator


def run_transport_simulation(network: FractureNetwork, vx: np.ndarray, vy: np.ndarray):
    """步骤7：示踪剂迁移方程求解"""
    print_section("步骤 7: 示踪剂对流-弥散迁移模拟")

    ny, nx = network.ny, network.nx
    dx = network.dx
    dy = network.dy
    dt = 10.0  # s

    transport = TransportSolver(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
        R=1.0, lambda_decay=1.0e-5
    )

    # 设置流速场
    # 使用简化流速（内部区域）
    vx_simplified = np.zeros((ny, nx))
    vy_simplified = np.zeros((ny, nx))
    vx_simplified[1:-1, 1:-1] = np.abs(vx[1:-1, 1:-1])
    vy_simplified[1:-1, 1:-1] = np.abs(vy[1:-1, 1:-1])

    # 最小非零流速
    vx_simplified = np.clip(vx_simplified, 1.0e-8, None)
    vy_simplified = np.clip(vy_simplified, 1.0e-8, None)

    transport.set_velocity_field(vx_simplified, vy_simplified)
    transport.set_dispersivity(alpha_L=0.1, alpha_T=0.01, D_m=1.0e-9)

    # 稳定性检查
    stability = transport.stability_check()
    print(f"  CFL 数: {stability['CFL']:.4f}")
    print(f"  弥散数: {stability['diffusion_number']:.4f}")
    print(f"  稳定性: {'满足' if stability['stable'] else '不满足'}")

    if stability['CFL'] > 1.0:
        # 自适应时间步长
        dt_new = 0.8 * dt / max(stability['CFL'], 1e-10)
        transport.dt = dt_new
        print(f"  自适应调整时间步长: {dt_new:.4f} s")

    # 求解
    n_steps = 200
    injection = (ny // 2 - 2, ny // 2 + 2, 2, 6)
    outlet = (ny // 2 - 2, ny // 2 + 2, nx - 6, nx - 2)

    result = transport.solve(
        n_steps=n_steps,
        injection_zone=injection,
        C_inject=1.0,
        check_mass=True
    )

    print(f"  时间步数: {n_steps}")
    print(f"  最终总质量: {result['final_mass']:.4e} kg")
    print(f"  质量守恒相对误差: {result.get('mass_conservation_error', 0):.4e}")

    # 突破曲线
    bc = transport.breakthrough_curve(
        outlet_zone=outlet,
        n_steps=n_steps,
        injection_zone=injection,
        C_inject=1.0
    )

    C_max = np.max(bc['concentrations'])
    t_peak = bc['times'][np.argmax(bc['concentrations'])]
    print(f"  出口峰值浓度: {C_max:.4e} kg/m³")
    print(f"  峰值时间: {t_peak:.2f} s")

    return transport, result, bc


def run_inverse_modeling(network: FractureNetwork):
    """步骤8：渗透率参数反演"""
    print_section("步骤 8: 渗透率参数反演")

    inverse = InverseModel()

    # 从穿透时间反演裂隙开度
    t_obs = 50000.0  # s
    L = 50.0
    i_hydraulic = 0.01
    n_e = 0.1

    result_perm = inverse.invert_permeability_from_travel_time(
        t_obs=t_obs, L=L, i_hydraulic=i_hydraulic, n_e=n_e
    )

    print(f"  观测穿透时间: {t_obs:.1f} s")
    print(f"  反演裂隙开度: {result_perm['aperture']:.4e} m")
    print(f"  反演等效渗透率: {result_perm['permeability']:.4e} m²")
    print(f"  迭代次数: {result_perm['iterations']}")
    print(f"  收敛: {result_perm['converged']}")

    # 从突破曲线反演弥散度
    times = np.linspace(0, 3600, 100)
    v = L / t_obs
    D_true = 0.1 * v + 1.0e-9
    from scipy.special import erfc
    C_obs = 0.5 * erfc((L - v * times) / (2.0 * np.sqrt(D_true * np.maximum(times, 1.0))))
    C_obs += 1.0e-10  # 避免零值

    result_disp = inverse.invert_dispersivity_from_breakthrough(
        times=times, concentrations=C_obs, L=L, v=v
    )

    print(f"  反演弥散系数: {result_disp['dispersion_coefficient']:.4e} m²/s")
    print(f"  反演纵向弥散度: {result_disp['longitudinal_dispersivity']:.4e} m")

    # Dirichlet 分布估计（裂隙方向比例）
    rng = np.random.default_rng(42)
    n_samples = 100
    k = 3  # 三个方向组
    alpha_true = np.array([2.0, 3.0, 5.0])
    from scipy.stats import dirichlet
    samples = dirichlet.rvs(alpha_true, size=n_samples, random_state=rng)

    result_dirichlet = InverseModel.dirichlet_estimate_moments(samples)
    print(f"  Dirichlet 参数估计: [{', '.join([f'{a:.3f}' for a in result_dirichlet['alpha']])}]")
    print(f"  真实参数: [{', '.join([f'{a:.3f}' for a in alpha_true])}]")
    print(f"  对数似然: {result_dirichlet['log_likelihood']:.4f}")

    # 双孔隙模型标定
    result_dp = inverse.calibrate_dual_porosity(
        t_obs=times, C_obs=C_obs, L=L
    )
    print(f"  质量交换系数: {result_dp['mass_transfer_rate']:.4e} 1/s")

    return inverse


def run_uncertainty_quantification(network: FractureNetwork):
    """步骤9：不确定性量化"""
    print_section("步骤 9: 不确定性量化")

    uq = UncertaintyQuantification()

    # 非中心 Beta 分布 CDF 计算
    x_test = 0.5
    a, b_param, lam = 2.0, 3.0, 1.5
    cdf_val = uq.noncentral_beta_cdf(x_test, a, b_param, lam)
    print(f"  非中心 Beta CDF({x_test}; {a}, {b_param}, {lam}): {cdf_val:.6f}")

    # Gamma 分布统计
    gamma_stats = uq.gamma_sample_stats(alpha=3.0, beta_param=0.001)
    print(f"  Gamma(3, 0.001) 均值: {gamma_stats['mean']:.2f} s")
    print(f"  方差: {gamma_stats['variance']:.2e} s²")
    print(f"  偏度: {gamma_stats['skewness']:.4f}")

    # 渗透率置信区间
    K_est = network.equivalent_permeability()
    K_std = K_est * 0.3
    ci_low, ci_high = uq.permeability_confidence_interval(K_est, K_std)
    print(f"  渗透率估计: {K_est:.4e} m²")
    print(f"  95% 置信区间: [{ci_low:.4e}, {ci_high:.4e}] m²")

    # 蒙特卡洛不确定性传播
    def forward_model(params):
        b = params['aperture']
        i_grad = params['gradient']
        rho = 1000.0
        g = 9.81
        mu = 1.0e-3
        K = (rho * g * b ** 2) / (12.0 * mu)
        v = K * i_grad / 0.1
        return v

    param_dists = [
        {'name': 'aperture', 'dist': 'lognormal', 'params': {'mu': np.log(5e-5), 'sigma': 0.3}},
        {'name': 'gradient', 'dist': 'uniform', 'params': {'low': 0.005, 'high': 0.02}}
    ]

    mc_result = uq.monte_carlo_uncertainty(
        forward_model, param_dists, n_samples=500, seed=42
    )
    print(f"  蒙特卡洛流速均值: {mc_result['mean']:.4e} m/s")
    print(f"  标准差: {mc_result['std']:.4e} m/s")
    print(f"  95% CI: [{mc_result['ci_95'][0]:.4e}, {mc_result['ci_95'][1]:.4e}] m/s")

    # 敏感性分析
    base_params = {'aperture': 5.0e-5, 'gradient': 0.01}
    sens = uq.sensitivity_analysis(forward_model, base_params)
    print(f"  开度敏感性系数: {sens.get('aperture', 0):.4f}")
    print(f"  梯度敏感性系数: {sens.get('gradient', 0):.4f}")

    # 一阶可靠性方法
    def g_limit(x):
        # 极限状态: v > 0.002 m/s 为失效
        b, i_grad = x[0], x[1]
        rho, g, mu = 1000.0, 9.81, 1.0e-3
        K = (rho * g * b ** 2) / (12.0 * mu)
        v = K * i_grad / 0.1
        return 0.002 - v

    mean_p = np.array([5.0e-5, 0.01])
    cov = np.array([[1.0e-10, 0.0], [0.0, 1.0e-5]])
    form_result = uq.first_order_reliability(g_limit, mean_p, cov)
    print(f"  可靠性指标 β: {form_result['reliability_index']:.4f}")
    print(f"  失效概率 P_f: {form_result['failure_probability']:.4e}")

    return uq


def main():
    """主程序入口"""
    print("\n" + "#" * 70)
    print("#  裂隙介质渗流与示踪试验数值模拟系统")
    print("#  水文地质博士级科学计算项目")
    print("#" * 70)

    start_time = time.time()

    try:
        # 步骤1: 裂隙网络生成
        network, net_result = run_fracture_network_generation()

        # 步骤2: 三维几何解析
        parser = run_geometry_parsing()

        # 步骤3: 计算网格生成
        mesh = run_mesh_generation()

        # 步骤4: Sinc 插值
        interpolator = run_sinc_interpolation()

        # 步骤5: 水力求解
        solver, vx, vy = run_hydraulic_simulation(network)

        # 步骤6: 流量积分
        quad, integrator = run_flow_integration(network, solver)

        # 步骤7: 示踪剂迁移
        transport, trans_result, bc = run_transport_simulation(network, vx, vy)

        # 步骤8: 参数反演
        inverse = run_inverse_modeling(network)

        # 步骤9: 不确定性量化
        uq = run_uncertainty_quantification(network)

        elapsed = time.time() - start_time

        print("\n" + "#" * 70)
        print(f"#  模拟完成，总耗时: {elapsed:.2f} 秒")
        print("#" * 70)
        print("\n所有计算步骤成功执行，无报错。")

        return 0

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
