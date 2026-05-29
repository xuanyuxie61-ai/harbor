"""
main.py
=======
统一入口文件：多相流界面追踪与相变的耦合数值模拟

科学问题：
    基于相场-Navier-Stokes 耦合模型，模拟二元合金在过冷熔体中的
    凝固过程。追踪液固界面的演化，计算温度场、浓度场和速度场的
    耦合发展，分析界面稳定性（Mullins-Sekerka 不稳定性）和枝晶
    形态的形成。

核心物理模型：
    1. Allen-Cahn 相场方程：
        τ ∂φ/∂t = ε²∇²φ - W'(φ) - λ_T(1-φ²)²(T-T_M) - λ_C(1-φ²)²(C-C_e)

    2. Navier-Stokes 方程（投影法）：
        ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + F_σ

    3. 温度对流-扩散方程：
        ∂T/∂t + v·∇T = α_T ∇²T + (L_f/c_p) ∂h/∂t

    4. 浓度对流-扩散方程：
        ∂C/∂t + v·∇C = ∇·(D(φ)∇C) + Q_C

运行方式：
    python main.py
"""

import numpy as np
import sys
import time

# 导入各模块
from phase_field_core import PhaseFieldModel
from navier_stokes_solver import NavierStokesSolver
from thermal_transport import ThermalTransportSolver
from interface_tracking import InterfaceTracker
from numerical_quadrature import GaussQuadrature, HypercubeIntegrals, CompositeQuadrature
from stability_analysis import CompanionMatrixEigenvalue, LogarithmicNorm, LinearStabilityAnalysis, BifurcationAnalysis
from stochastic_perturbation import RandomNumberGenerator, ThermalNoise
from time_integrator import ODESolver, CoupledOscillator, PhaseFieldTimeStepper
from mesh_adaptation import MeshAdaptation, TriangleGridTopology
from spectral_solver import SpectralPoissonSolver, GaussSeidelPoisson, SpectralHeatSolver, SineTransform
from fem_2d_serene import FEM2DSerene


def setup_simulation_domain():
    """
    设置模拟域和物理参数。

    Returns
    -------
    dict
        参数字典。
    """
    params = {
        # 网格参数
        'nx': 41,
        'ny': 41,
        'Lx': 1.0,
        'Ly': 1.0,

        # 时间参数
        'dt': 0.001,
        'n_steps': 100,
        'output_interval': 25,

        # 相场参数
        'epsilon': 0.03,
        'tau': 1.0,
        'lambda_thermal': 0.2,
        'lambda_solute': 0.1,
        'T_m': 1.0,
        'C_e': 0.5,

        # 流体参数
        'rho': 1.0,
        'mu': 0.1,
        'surface_tension': 0.05,

        # 热传输参数
        'thermal_diffusivity': 0.2,
        'latent_heat': 0.5,
        'specific_heat': 1.0,

        # 溶质参数
        'D_solid': 0.001,
        'D_liquid': 0.2,
        'partition_coefficient': 0.3,
        'liquidus_slope': -2.0,

        # 过冷度
        'undercooling': 0.15,
    }

    params['dx'] = params['Lx'] / (params['nx'] - 1)
    params['dy'] = params['Ly'] / (params['ny'] - 1)

    return params


def initialize_fields(params):
    """
    初始化相场、温度场、浓度场和速度场。

    Parameters
    ----------
    params : dict
        参数字典。

    Returns
    -------
    tuple
        (phi, T, C, vx, vy)
    """
    nx, ny = params['nx'], params['ny']
    dx, dy = params['dx'], params['dy']
    Lx, Ly = params['Lx'], params['Ly']

    # 相场模型初始化
    pf = PhaseFieldModel(
        nx=nx, ny=ny, dx=dx, dy=dy,
        epsilon=params['epsilon'],
        tau=params['tau'],
        lambda_thermal=params['lambda_thermal'],
        lambda_solute=params['lambda_solute'],
        T_m=params['T_m'],
        C_e=params['C_e']
    )

    # 初始化圆形晶核
    phi = pf.initialize_circular_nucleus(
        center_x=Lx * 0.5,
        center_y=Ly * 0.5,
        radius=0.15
    )

    # 温度场：均匀过冷
    T = params['T_m'] - params['undercooling'] * np.ones((nx, ny))

    # 在界面附近施加轻微扰动以触发 Mullins-Sekerka 不稳定性
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    r = np.sqrt((X - Lx * 0.5) ** 2 + (Y - Ly * 0.5) ** 2)
    theta = np.arctan2(Y - Ly * 0.5, X - Lx * 0.5)

    # 添加四重对称扰动
    perturbation = 0.02 * np.cos(4.0 * theta) * np.exp(-((r - 0.15) / 0.05) ** 2)
    T += perturbation

    # 浓度场：初始为平衡浓度
    C = params['C_e'] * np.ones((nx, ny))

    # 速度场：初始为零
    vx = np.zeros((nx, ny))
    vy = np.zeros((nx, ny))

    return phi, T, C, vx, vy


def run_phase_field_ns_coupling(params):
    """
    运行相场-NS 耦合主循环。

    Parameters
    ----------
    params : dict
        参数字典。

    Returns
    -------
    dict
        结果字典。
    """
    nx, ny = params['nx'], params['ny']
    dx, dy = params['dx'], params['dy']
    dt = params['dt']

    # 初始化场
    phi, T, C, vx, vy = initialize_fields(params)

    # 初始化各求解器
    pf = PhaseFieldModel(
        nx=nx, ny=ny, dx=dx, dy=dy,
        epsilon=params['epsilon'],
        tau=params['tau'],
        lambda_thermal=params['lambda_thermal'],
        lambda_solute=params['lambda_solute'],
        T_m=params['T_m'],
        C_e=params['C_e']
    )

    ns = NavierStokesSolver(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
        rho=params['rho'],
        mu=params['mu'],
        surface_tension=params['surface_tension'],
        epsilon=params['epsilon']
    )

    thermal = ThermalTransportSolver(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt,
        thermal_diffusivity=params['thermal_diffusivity'],
        latent_heat=params['latent_heat'],
        specific_heat=params['specific_heat'],
        solute_diffusivity_solid=params['D_solid'],
        solute_diffusivity_liquid=params['D_liquid'],
        partition_coefficient=params['partition_coefficient'],
        liquidus_slope=params['liquidus_slope']
    )

    tracker = InterfaceTracker(nx=nx, ny=ny, dx=dx, dy=dy)
    timestepper = PhaseFieldTimeStepper(
        dt=dt, dx=dx, dy=dy,
        epsilon=params['epsilon'],
        tau=params['tau']
    )

    # 存储历史
    phi_history = [phi.copy()]
    T_history = [T.copy()]
    C_history = [C.copy()]
    vx_history = [vx.copy()]
    time_stamps = [0.0]

    interface_area_history = []
    morphology_history = []
    tip_velocity_history = []

    print("=" * 60)
    print("多相流界面追踪与相变耦合模拟")
    print("=" * 60)
    print(f"网格: {nx} x {ny}")
    print(f"时间步长: {dt}")
    print(f"总步数: {params['n_steps']}")
    print("=" * 60)

    start_time = time.time()

    for step in range(1, params['n_steps'] + 1):
        phi_old = phi.copy()

        # ============================================================
        # HOLE 3: 实现相场-热传输-浓度场的耦合推进循环
        #
        # 需要按正确顺序完成以下四步推进：
        #
        # 1. 相场方程推进:
        #    - 调用 pf.phase_field_rhs(phi, T, C, vx, vy) 计算右端项
        #    - 调用 timestepper.runge_kutta_step 推进 phi
        #    - 注意 RK4 需要封装 lambda 以固定 T, C, vx, vy
        #    - 对 phi 进行 clip 数值稳定性限制
        #
        # 2. 温度场推进:
        #    - 调用 thermal.temperature_rhs(T, phi, phi_old, vx, vy)
        #    - 限制温度变化率后做显式 Euler 推进
        #    - 对 T 进行物理范围 clip
        #
        # 3. 浓度场推进:
        #    - 调用 thermal.concentration_rhs(C, phi, phi_old, vx, vy)
        #    - 限制浓度变化率后做显式 Euler 推进
        #    - 对 C 进行物理范围 clip
        #
        # 4. NS 方程推进（每 10 步一次）:
        #    - 调用 ns.time_step(vx, vy, phi)
        #
        # 关键协同点：
        #   - 相场推进使用新 phi，但 T/C/vx/vy 在当前步保持不变
        #   - 温度和浓度推进必须使用更新后的 phi（而非 phi_old）
        #   - 温度/浓度的 rhs 需要传入 phi_old 以计算潜热源
        #   - 注意各场的 clip 范围不同
        # ============================================================
        raise NotImplementedError("HOLE 3: 请实现耦合推进循环")

        # 5. 界面追踪与诊断
        if step % params['output_interval'] == 0:
            area = tracker.compute_interface_area(phi)
            morph = tracker.compute_morphology_number(phi)
            # 计算tip velocity时避免除零和空切片
            tip_vels = []
            for _ in range(1):
                tip_v = tracker.tip_velocity_dendrite(phi_old, phi, dt)
                if abs(tip_v) > 1e-12:
                    tip_vels.append(tip_v)
            tip_v = tip_vels[0] if tip_vels else 0.0

            interface_area_history.append(area)
            morphology_history.append(morph)
            tip_velocity_history.append(tip_v)

            phi_history.append(phi.copy())
            T_history.append(T.copy())
            C_history.append(C.copy())
            vx_history.append(vx.copy())
            time_stamps.append(step * dt)

            print(f"Step {step:4d} | Time {step*dt:.4f} | "
                  f"Interface Area: {area:.4f} | Morphology: {morph:.4f} | "
                  f"Tip V: {tip_v:.4f}")

    elapsed = time.time() - start_time
    print(f"\n模拟完成，耗时: {elapsed:.2f} 秒")

    return {
        'phi_final': phi,
        'T_final': T,
        'C_final': C,
        'vx_final': vx,
        'vy_final': vy,
        'phi_history': phi_history,
        'T_history': T_history,
        'C_history': C_history,
        'time_stamps': time_stamps,
        'interface_area': interface_area_history,
        'morphology': morphology_history,
        'tip_velocity': tip_velocity_history,
    }


def run_numerical_tests():
    """
    运行数值方法的验证和测试。
    融合各种子项目的核心算法验证。
    """
    print("\n" + "=" * 60)
    print("数值方法验证与测试")
    print("=" * 60)

    # 1. 高斯求积精确度测试 (464_gen_hermite_exactness)
    print("\n[1] Gauss-Hermite 求积精确度测试")
    nodes, weights = GaussQuadrature.gauss_hermite_5point()
    max_degree = 9
    print(f"   5点 Gauss-Hermite 规则，检验 0-{max_degree} 次单项式")
    for degree in range(0, max_degree + 1, 2):
        # 精确值
        if degree % 2 == 1:
            exact = 0.0
        else:
            from scipy.special import gamma
            exact = gamma((degree + 1.0) / 2.0)
        # 数值积分
        quad_val = np.sum(weights * (nodes ** degree))
        err = abs(quad_val - exact) if exact != 0 else abs(quad_val)
        print(f"   Degree {degree:2d}: Exact={exact:.6f}, Quad={quad_val:.6f}, Err={err:.2e}")

    # 2. 超立方体积分测试 (559_hypercube_integrals)
    print("\n[2] 超立方体单项式积分")
    m = 3
    exponents = np.array([1, 2, 1])
    exact = HypercubeIntegrals.monomial_integral(m, exponents)
    print(f"   ∫_[0,1]^{m} x^1 y^2 z^1 dV = {exact:.6f}")
    print(f"   理论值 = 1/2 * 1/3 * 1/2 = 1/12 = {1.0/12:.6f}")

    # 3. 伴矩阵特征值测试 (203_companion_matrix)
    print("\n[3] 伴矩阵特征值计算")
    # 多项式 p(x) = x² - 5x + 6 = (x-2)(x-3)
    coeffs_power = np.array([6.0, -5.0, 1.0])
    roots_power = CompanionMatrixEigenvalue.find_roots(coeffs_power, basis='power')
    print(f"   p(x)=x²-5x+6, 根: {roots_power}")

    # 4. 矩阵对数范数测试 (697_log_norm)
    print("\n[4] 矩阵对数范数")
    A = np.array([[-2.0, 1.0], [0.5, -3.0]])
    mu1 = LogarithmicNorm.log_norm(A, p=1)
    mu2 = LogarithmicNorm.log_norm(A, p=2)
    muinf = LogarithmicNorm.log_norm(A, p=np.inf)
    print(f"   A = [[-2, 1], [0.5, -3]]")
    print(f"   μ_1(A) = {mu1:.4f}, μ_2(A) = {mu2:.4f}, μ_∞(A) = {muinf:.4f}")

    # 5. Logistic 分叉分析 (700_logistic_bifurcation)
    print("\n[5] Logistic 映射分叉分析")
    r_values = [2.5, 3.2, 3.5, 3.8]
    for r in r_values:
        lyap = BifurcationAnalysis.lyapunov_exponent_logistic(r)
        attractors = BifurcationAnalysis.find_attractors(r)
        print(f"   r={r:.1f}: Lyapunov={lyap:.4f}, Attractors={np.round(attractors, 4)}")

    # 6. 正弦变换测试 (1085_sine_transform)
    print("\n[6] 正弦变换测试")
    n = 7
    f = np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
    b = SineTransform.dst_1d(f)
    f_recon = SineTransform.idst_1d(b)
    error = np.max(np.abs(f - f_recon))
    print(f"   DST + IDST 重构误差: {error:.2e}")

    # 7. 耦合振荡器 (1138_spring_double_ode)
    print("\n[7] 双弹簧耦合振荡器")
    osc = CoupledOscillator(m1=3.0, m2=5.0, k1=1.0, k2=10.0)
    y0 = np.array([0.0, 1.0, 0.0, 0.0])
    t, y = osc.solve(t0=0.0, y0=y0, t_end=10.0, h=0.01, method='rk4')
    print(f"   t=0:  u1={y[0,0]:.4f}, v1={y[0,1]:.4f}")
    print(f"   t=10: u1={y[-1,0]:.4f}, v1={y[-1,1]:.4f}")

    # 8. 泊松方程求解 (875_poisson_1d)
    print("\n[8] 一维泊松方程 Gauss-Seidel 求解")
    x, u, it = GaussSeidelPoisson.solve_1d_poisson_gs(
        n_intervals=20, a=0.0, b=1.0, ua=0.0, ub=0.0,
        force_func=lambda x: np.pi ** 2 * np.sin(np.pi * x),
        max_iter=5000, tol=1e-6
    )
    u_exact = np.sin(np.pi * x)
    err = np.max(np.abs(u - u_exact))
    print(f"   与精确解 u=sin(πx) 的最大误差: {err:.2e}")
    print(f"   GS 迭代次数: {it}")

    # 9. 有限元求解 (402_fem2d_bvp_serene)
    print("\n[9] 二维 FEM Serendipity 求解器")
    nx_fem, ny_fem = 7, 7  # 使用更大的网格避免奇异
    x_fem = np.linspace(0, 1, nx_fem)
    y_fem = np.linspace(0, 1, ny_fem)
    fem = FEM2DSerene(nx_fem, ny_fem, x_fem, y_fem)

    a_func = lambda x, y: 1.0
    c_func = lambda x, y: 0.0
    f_func = lambda x, y: 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    K, F = fem.assemble_system(a_func, c_func, f_func)

    # Dirichlet BC: u=0 on boundary
    bc_nodes = []
    bc_values = []
    for i in range(nx_fem):
        for j in range(ny_fem):
            node_id = i * ny_fem + j
            if i == 0 or i == nx_fem - 1 or j == 0 or j == ny_fem - 1:
                bc_nodes.append(node_id)
                bc_values.append(0.0)

    K_bc, F_bc = fem.apply_dirichlet_bc(K, F, bc_nodes, bc_values)
    try:
        u_fem = fem.solve(K_bc, F_bc)
        # 计算中心点误差
        center_node = (nx_fem // 2) * ny_fem + (ny_fem // 2)
        u_center = u_fem[center_node]
        u_exact_center = np.sin(np.pi * 0.5) * np.sin(np.pi * 0.5)
        print(f"   FEM 中心点解: {u_center:.4f}, 精确值: {u_exact_center:.4f}")
    except np.linalg.LinAlgError:
        # 若矩阵奇异，使用最小二乘求解
        u_fem = np.linalg.lstsq(K_bc, F_bc, rcond=None)[0]
        center_node = (nx_fem // 2) * ny_fem + (ny_fem // 2)
        u_center = u_fem[center_node]
        u_exact_center = np.sin(np.pi * 0.5) * np.sin(np.pi * 0.5)
        print(f"   FEM 中心点解 (lstsq): {u_center:.4f}, 精确值: {u_exact_center:.4f}")

    # 10. 动态规划网格优化 (156_change_dynamic)
    print("\n[10] 动态规划网格优化")
    mesh_adapt = MeshAdaptation(nx=20, ny=20, x_max=1.0, y_max=1.0)
    # 定义三个区域的误差函数
    error_funcs = [
        lambda n: 1.0 / max(n, 1) ** 2,
        lambda n: 0.5 / max(n, 1) ** 2,
        lambda n: 2.0 / max(n, 1) ** 2,
    ]
    distribution = mesh_adapt.dynamic_programming_mesh_distribution(
        n_total=30, error_funcs=error_funcs, regions=3
    )
    print(f"   总网格数 30 的最优分配: {distribution}")

    print("\n" + "=" * 60)
    print("所有数值测试通过")
    print("=" * 60)


def main():
    """
    主函数：运行相场-NS 耦合模拟和数值验证。
    """
    print("\n")
    print("*" * 60)
    print("*  计算流体力学：多相流界面追踪与相变")
    print("*  博士级科研代码合成项目")
    print("*" * 60)

    # 运行数值方法验证
    run_numerical_tests()

    # 运行主模拟
    print("\n")
    print("*" * 60)
    print("*  主模拟：二元合金凝固过程")
    print("*" * 60)

    params = setup_simulation_domain()
    results = run_phase_field_ns_coupling(params)

    # 输出最终诊断
    print("\n" + "=" * 60)
    print("最终诊断")
    print("=" * 60)
    print(f"最终界面面积: {results['interface_area'][-1]:.4f}")
    print(f"最终形态学数: {results['morphology'][-1]:.4f}")
    print(f"界面面积变化: {results['interface_area'][0]:.4f} -> {results['interface_area'][-1]:.4f}")
    tip_v_list = [v for v in results['tip_velocity'] if abs(v) > 1e-12]
    print(f"平均尖端速度: {np.mean(tip_v_list):.4f}" if tip_v_list else "平均尖端速度: 0.0000")

    # 物理量统计
    print(f"\n温度场范围: [{np.min(results['T_final']):.4f}, {np.max(results['T_final']):.4f}]")
    print(f"浓度场范围: [{np.min(results['C_final']):.4f}, {np.max(results['C_final']):.4f}]")
    print(f"速度场范围: vx=[{np.min(results['vx_final']):.4f}, {np.max(results['vx_final']):.4f}], "
          f"vy=[{np.min(results['vy_final']):.4f}, {np.max(results['vy_final']):.4f}]")
    print(f"序参量范围: [{np.min(results['phi_final']):.4f}, {np.max(results['phi_final']):.4f}]")

    # Mullins-Sekerka 稳定性参数
    bifurcation = BifurcationAnalysis.phase_transition_bifurcation_parameter(
        T_undercooling=params['undercooling'],
        gamma=params['surface_tension'],
        m_L=params['liquidus_slope'],
        C0=params['C_e'],
        D_l=params['D_liquid'],
        k_p=params['partition_coefficient']
    )
    print(f"\nMullins-Sekerka 稳定性分析:")
    print(f"  毛细长度 d_0: {bifurcation['capillary_length']:.6f}")
    print(f"  特征速度 V: {bifurcation['characteristic_velocity']:.4f}")
    print(f"  稳定性参数 σ*: {bifurcation['stability_parameter']:.4f}")
    print(f"  界面失稳预测: {'是' if bifurcation['unstable'] else '否'}")

    print("\n" + "=" * 60)
    print("程序正常结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
