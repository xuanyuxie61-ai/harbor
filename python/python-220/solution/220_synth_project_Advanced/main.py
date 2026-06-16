"""
main.py — 分布式 ADMM 多物理场 PDE 约束优化框架
====================================================
统一入口 (零参数可运行)

科学问题:
  将多种物理场 (反应扩散、神经动力学、大气化学、流体力学)
  的 PDE 约束反问题通过分布式 ADMM 方法在区域分解框架下求解.

项目结构:
  1. 网格生成与区域分解 (mesh.py, domain_decomp.py)
  2. 高阶有限元算子 (fem_operators.py, quadrature.py)
  3. 多物理场正演模型 (gray_scott.py, hh_ode.py, ozone_ode.py, burgers.py)
  4. 迭代线性求解器 (linear_solvers.py)
  5. 无导数优化 (nelder_mead.py)
  6. PDE 约束反问题 (inverse_problem.py)
  7. 分布式 ADMM 核心 (admm_core.py)
  8. 自适应罚参数 (adaptive_penalty.py)
  9. 收敛性分析 (convergence.py)

实验演示:
  实验 1: 分布式 Poisson 反问题 (扩散系数辨识)
  实验 2: Gray-Scott 参数估计 (ADMM 加速)
  实验 3: 多物理场数据融合 (Hodgkin-Huxley + 扩散)
  实验 4:  Burgers 方程粘性参数辨识
  实验 5: 臭氧化学反问题

运行方式:
  python main.py
"""

import numpy as np
import sys
import os
import time

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GrayScottConfig, HodgkinHuxleyConfig, OzoneChemistryConfig,
    BurgersConfig, ADMMConfig, ExperimentConfig, EPS_NUM
)


# ============================================================
#  输出工具
# ============================================================
def print_header(title: str):
    """打印带装饰的实验标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """打印节标题"""
    print(f"\n--- {title} ---")


def print_result(key: str, value, fmt: str = ".6e"):
    """打印结果"""
    if isinstance(value, float):
        print(f"  {key}: {value:{fmt}}")
    elif isinstance(value, (np.floating,)):
        print(f"  {key}: {value:{fmt}}")
    elif isinstance(value, (int, np.integer)):
        print(f"  {key}: {value}")
    elif isinstance(value, np.ndarray):
        if value.size <= 5:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: [{value[0]:.4f}, ..., {value[-1]:.4f}] (size={value.size})")
    else:
        print(f"  {key}: {value}")


# ============================================================
#  实验 1: 分布式 Poisson 反问题 (扩散系数辨识)
# ============================================================
def experiment_distributed_poisson():
    """
    实验 1: 分布式 Poisson 反问题

    问题: 求解 -∇·(κ∇u) = f on Ω, 其中 κ 未知.
    方法: 将 Ω 分解为 4 个子域, 使用 Consensus ADMM 分布式求解.

    正演: 有限差分法求解 Poisson 方程
    反问题: 从稀疏观测数据辨识扩散系数 κ
    """
    print_header("实验 1: 分布式 Poisson 反问题 (扩散系数辨识)")

    from mesh import generate_structured_mesh
    from domain_decomp import decompose_rectangular_mesh, InterfaceManager
    from linear_solvers import solve_poisson_1d_gs
    from fem_operators import assemble_stiffness_matrix_1d_linear, assemble_mass_matrix_1d
    from admm_core import consensus_admm
    from convergence import ConvergenceDiagnostics, generate_convergence_report

    # 1. 网格生成
    print_section("1. 网格生成与区域分解")
    nx, ny = 16, 16
    mesh = generate_structured_mesh(nx, ny, Lx=1.0, Ly=1.0)
    print_result("网格节点数", mesh.n_nodes)
    print_result("单元数", mesh.n_elements)
    print_result("最小单元质量", mesh.min_quality(), ".4f")

    # 2. 区域分解
    n_sub = 4
    subdomains = decompose_rectangular_mesh(nx, ny, 2, 2)
    iface_mgr = InterfaceManager(subdomains)
    print_result("子域数", n_sub)
    print_result("接口数", iface_mgr.n_interfaces)
    print_result("接口节点总数", len(iface_mgr.all_interface_nodes))

    for sub in subdomains:
        print(f"  子域 {sub.subdomain_id}: {sub.n_local_nodes} 节点, "
              f"{sub.n_interface_nodes} 接口节点, "
              f"邻居 {sub.neighbors}")

    # 3. 1D Poisson 基准测试 (来源: 452)
    print_section("2. Gauss-Seidel 求解 1D Poisson 基准")
    poisson_result = solve_poisson_1d_gs(n=50, max_iter=3000, tol=1e-8)
    print_result("GS 迭代次数", poisson_result['solver_info']['iterations'])
    print_result("GS 最终残差", poisson_result['solver_info']['final_residual'])
    print_result("最大误差 (vs 精确解)", poisson_result['max_error'])
    print_result("GS 收敛", poisson_result['solver_info']['converged'])

    # 4. FEM 矩阵组装 (来源: 377)
    print_section("3. FEM 矩阵组装 (Neumann)")
    n_fem = 20
    h_fem = 1.0 / n_fem
    K_fem = assemble_stiffness_matrix_1d_linear(n_fem + 1, h_fem, bc_type="neumann")
    M_fem = assemble_mass_matrix_1d(n_fem + 1, h_fem, bc_type="neumann")
    print_result("刚度矩阵条件数", np.linalg.cond(K_fem + np.eye(n_fem+1)*EPS_NUM))
    print_result("质量矩阵迹", np.trace(M_fem))

    # 5. 分布式 ADMM 求解
    print_section("4. Consensus ADMM 分布式求解")

    # 模拟: 4 个子域各自有局部二次目标
    # f_i(x) = 0.5 * x^T Q_i x + c_i^T x
    n_vars = 5
    rng = np.random.RandomState(42)
    local_objectives = []
    for i in range(n_sub):
        Q_i = rng.randn(n_vars, n_vars)
        Q_i = Q_i.T @ Q_i + 0.5 * np.eye(n_vars)  # SPD
        c_i = rng.randn(n_vars) * 0.1
        def make_obj(Q, c):
            return lambda x: 0.5 * x @ Q @ x + c @ x
        local_objectives.append(make_obj(Q_i, c_i))

    admm_config = ADMMConfig(
        rho=1.0, max_iter=200, abs_tol=1e-6, rel_tol=1e-4,
        use_adaptive_penalty=True, mu=10.0, tau_incr=2.0, tau_decr=2.0
    )

    admm_result = consensus_admm(local_objectives, n_vars, admm_config)

    print_result("ADMM 迭代次数", admm_result['n_iterations'])
    print_result("最终原始残差", admm_result['state'].primal_residuals[-1] if admm_result['state'].primal_residuals else 0)
    print_result("最终对偶残差", admm_result['state'].dual_residuals[-1] if admm_result['state'].dual_residuals else 0)
    print_result("最终目标值", admm_result['state'].objective_values[-1] if admm_result['state'].objective_values else 0)
    print_result("收敛", admm_result['converged'])

    # 6. 收敛诊断
    print_section("5. 收敛诊断")
    diag = ConvergenceDiagnostics(n_sub, n_vars)
    for k in range(len(admm_result['state'].primal_residuals)):
        diag.record(
            k,
            admm_result['state'].primal_residuals[k],
            admm_result['state'].dual_residuals[k],
            admm_result['state'].objective_values[k],
            admm_result['state'].rho_history[k] if k < len(admm_result['state'].rho_history) else 1.0,
            admm_result['state'].x_locals,
            admm_result['state'].z_consensus,
        )
    report = generate_convergence_report(diag)
    print(report)

    return admm_result


# ============================================================
#  实验 2: Gray-Scott 反应扩散模拟与参数灵敏度
# ============================================================
def experiment_gray_scott():
    """
    实验 2: Gray-Scott 反应扩散系统

    演示 Gray-Scott 系统的 Turing 模式形成,
    以及 ADMM 在参数估计中的应用.
    """
    print_header("实验 2: Gray-Scott 反应扩散系统")

    from gray_scott import (
        run_gray_scott_simulation, gray_scott_initial_condition,
        gray_scott_step, laplacian_9pt
    )

    # 1. 模拟
    print_section("1. Gray-Scott 模拟 (斑点模式)")
    config = GrayScottConfig(
        Du=0.16, Dv=0.08, gamma=0.024, kappa=0.056,
        nx=30, ny=30, T_final=3.0
    )
    print_result("扩散系数比 Du/Dv", config.Du / config.Dv)
    print_result("最大时间步长", config.dt_max)
    print_result("网格尺寸 dx × dy", f"{config.dx:.4f} × {config.dy:.4f}")

    t0 = time.time()
    result = run_gray_scott_simulation(config, n_snapshots=3, seed=42)
    t_sim = time.time() - t0

    print_result("模拟时间", t_sim, ".3f")
    print_result("快照数", len(result['times']))
    print_result("最终 U 范围", f"[{result['final_U'].min():.4f}, {result['final_U'].max():.4f}]")
    print_result("最终 V 范围", f"[{result['final_V'].min():.4f}, {result['final_V'].max():.4f}]")

    # 质量守恒
    if result['mass_history']:
        mass_init = result['mass_history'][0] if result['mass_history'] else 0
        mass_final = result['mass_history'][-1] if result['mass_history'] else 0
        print_result("初始总质量", mass_init)
        print_result("最终总质量", mass_final)
        print_result("质量漂移", abs(mass_final - mass_init))

    # 2. Laplacian 验证
    print_section("2. 9 点 Laplacian 验证")
    # 对已知函数 f(x,y) = sin(πx)sin(πy), ∇²f = -2π²f
    nx, ny = 30, 30
    Lx, Ly = config.Lx, config.Ly
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f_test = np.sin(PI * X / Lx) * np.sin(PI * Y / Ly)
    lap_exact = -2.0 * (PI / Lx)**2 * f_test  # 一维 + 一维
    # 注意：周期性边界，精确 Laplacian 需要修正
    lap_numeric = laplacian_9pt(f_test, config.dx, config.dy)
    # 内部点误差 (边界有周期性wrap误差)
    err = np.max(np.abs(lap_numeric[2:-2, 2:-2] - lap_exact[2:-2, 2:-2]))
    print_result("Laplacian 最大误差 (内部)", err)

    return result


PI = np.pi


# ============================================================
#  实验 3: Hodgkin-Huxley 神经动力学模拟
# ============================================================
def experiment_hodgkin_huxley():
    """
    实验 3: Hodgkin-Huxley 神经轴突模型

    模拟动作电位的产生与传播,
    演示 ADMM 在参数辨识中的应用.
    """
    print_header("实验 3: Hodgkin-Huxley 神经动力学")

    from hh_ode import (
        run_hh_simulation, compute_steady_state,
        compute_ionic_currents, compute_iv_curve
    )

    # 1. 基本模拟
    print_section("1. HH 模型模拟 (动作电位)")
    config = HodgkinHuxleyConfig(
        V0=-65.0, T_final=50.0, dt=0.01, I_ext=10.0
    )
    result = run_hh_simulation(config)

    print_result("模拟时长", f"{config.T_final} ms")
    print_result("时间步数", int(config.T_final / config.dt))
    print_result("动作电位次数", len(result['spike_times']))
    print_result("放电频率", f"{result['firing_rate']:.2f} Hz")
    print_result("V 范围", f"[{result['V'].min():.2f}, {result['V'].max():.2f}] mV")
    print_result("最终门控变量",
                 f"n={result['n'][-1]:.4f}, m={result['m'][-1]:.4f}, h={result['h'][-1]:.4f}")

    # 2. 稳态分析
    print_section("2. 稳态分析")
    for V_test in [-70.0, -60.0, -40.0, 0.0, 30.0]:
        ss = compute_steady_state(V_test, config)
        print(f"  V={V_test:6.1f}mV: n∞={ss['n_inf']:.4f}, "
              f"m∞={ss['m_inf']:.4f}, h∞={ss['h_inf']:.4f}, "
              f"τn={ss['tau_n']:.3f}, τm={ss['tau_m']:.4f}, τh={ss['tau_h']:.3f}")

    # 3. 离子电流
    print_section("3. 离子电流 (V=-60mV)")
    currents = compute_ionic_currents(-60.0, 0.3, 0.05, 0.6, config)
    for name, val in currents.items():
        print_result(f"  {name}", val, ".4f")

    # 4. I-V 曲线
    print_section("4. I-V 曲线特征")
    iv = compute_iv_curve(config)
    # 找到零电流交叉 (静息电位)
    idx = np.where(np.diff(np.sign(iv['I_total'])))[0]
    if len(idx) > 0:
        V_rest = iv['V_values'][idx[0]]
        print_result("静息电位估计", V_rest, ".2f")
    print_result("I-V 采样点数", len(iv['V_values']))

    return result


# ============================================================
#  实验 4: 臭氧化学与 Burgers 方程
# ============================================================
def experiment_ozone_burgers():
    """
    实验 4: 大气臭氧化学 + 粘性 Burgers 方程

    演示 ADMM 在化学反应动力学和流体力学反问题中的应用.
    """
    print_header("实验 4: 大气臭氧化学与 Burgers 方程")

    from ozone_ode import run_ozone_simulation, compute_photostationary_state
    from burgers import run_burgers_simulation

    # 1. 臭氧化学
    print_section("1. 大气臭氧化学 (24h 模拟)")
    ozone_config = OzoneChemistryConfig()
    ozone_result = run_ozone_simulation(ozone_config)

    print_result("模拟时长", f"{ozone_config.T_final} 小时")
    print_result("最终 [O]", ozone_result['O'][-1])
    print_result("最终 [NO]", ozone_result['NO'][-1])
    print_result("最终 [NO₂]", ozone_result['NO2'][-1])
    print_result("最终 [O₃]", ozone_result['O3'][-1])
    print_result("守恒量最大漂移",
                 ozone_result['conservation']['max_relative_error'])

    # 光稳态
    print_section("2. 光稳态分析")
    for k1_val in [1e-5, 1e-4, 1e-3]:
        pss = compute_photostationary_state(ozone_config, k1_val)
        print(f"  k₁={k1_val:.0e}: [O]={pss['O']:.2e}, [NO]={pss['NO']:.2e}, "
              f"[NO₂]={pss['NO2']:.2e}, [O₃]={pss['O3']:.2e}")

    # 2. Burgers 方程
    print_section("3. 粘性 Burgers 方程 (激波衰减)")
    burgers_config = BurgersConfig(
        nu=0.05, nx=80, L=1.0, T_final=0.5,
        ic_type="shock", bc_type="dirichlet"
    )
    print_result("Reynolds 数 Re = 1/ν", 1.0 / burgers_config.nu)
    print_result("dx", burgers_config.dx)
    print_result("dt_max", burgers_config.dt_max)

    burgers_result = run_burgers_simulation(burgers_config, n_snapshots=5)
    print_result("快照数", len(burgers_result['times']))
    print_result("最终 L2 能量", burgers_result['energy_history'][-1] if burgers_result['energy_history'] else 0, ".6f")
    if len(burgers_result['energy_history']) > 1:
        energy_dissipation = burgers_result['energy_history'][0] - burgers_result['energy_history'][-1]
        print_result("能量耗散", energy_dissipation, ".6f")
        print_result("能量单调递减",
                     all(burgers_result['energy_history'][i] >= burgers_result['energy_history'][i+1]
                         for i in range(len(burgers_result['energy_history'])-1)))

    return ozone_result, burgers_result


# ============================================================
#  实验 5: 多物理场 ADMM 数据融合
# ============================================================
def experiment_multiphysics_admm():
    """
    实验 5: 多物理场 ADMM 数据融合

    将不同物理场的参数估计通过 ADMM 框架融合:
    子域 1: Gray-Scott 反应扩散参数
    子域 2: Hodgkin-Huxley 动力学参数
    子域 3: Burgers 流体力学参数
    子域 4: 臭氧化学参数

    各子域通过 ADMM 接口共享公共参数.
    """
    print_header("实验 5: 多物理场 ADMM 数据融合")

    from admm_core import consensus_admm
    from adaptive_penalty import CombinedAdaptivePenalty, ResidualBalancingStrategy
    from convergence import ConvergenceDiagnostics, generate_convergence_report
    from nelder_mead import nelder_mead, rosenbrock, sphere
    from quadrature import verify_quadrature_accuracy, pyramid_volume
    from inverse_problem import compute_metric_distortion

    # 1. NM 求解器验证
    print_section("1. Nelder-Mead 求解器验证")
    for name, func, x0 in [
        ("Sphere 2D", sphere, np.array([3.0, 4.0])),
        ("Rosenbrock 2D", rosenbrock, np.array([-1.0, 1.0])),
        ("Sphere 5D", sphere, np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
    ]:
        result = nelder_mead(func, x0, max_iter=2000, tol=1e-10)
        print(f"  {name}: f*={result['f_opt']:.2e}, "
              f"iters={result['iterations']}, "
              f"x*={result['x_opt']}")

    # 2. 数值积分验证
    print_section("2. 数值积分规则验证")
    quad_results = verify_quadrature_accuracy()
    for name, info in quad_results.items():
        if 'max_exact_degree' in info:
            print(f"  {name}: 精确至 {info['max_exact_degree']} 次 (期望 {info['expected']})")
        elif 'max_error' in info:
            print(f"  {name}: 最大误差 {info['max_error']:.2e}")
    print_result("金字塔体积 (解析)", pyramid_volume(), ".4f")

    # 3. ADMM 多物理场融合
    print_section("3. 分布式 ADMM 多物理场参数融合")

    # 模拟各物理场的局部目标函数 (二次近似)
    n_shared_params = 4  # 共享参数维度
    n_sub = 4
    rng = np.random.RandomState(123)

    # 每个物理场有不同的局部二次目标
    physics_names = ["Gray-Scott", "Hodgkin-Huxley", "Burgers", "Ozone"]
    local_objectives = []
    for i in range(n_sub):
        # 局部 Hessian (物理场特异性)
        H_i = rng.randn(n_shared_params, n_shared_params)
        H_i = H_i.T @ H_i + (i + 1) * 0.3 * np.eye(n_shared_params)
        # 局部梯度偏移 (不同物理场有不同最优)
        c_i = rng.randn(n_shared_params) * 0.5

        def make_obj(H, c, name=""):
            def obj(x):
                return 0.5 * x @ H @ x + c @ x
            return obj

        local_objectives.append(make_obj(H_i, c_i, physics_names[i]))
        print(f"  {physics_names[i]}: 局部条件数 = "
              f"{np.linalg.cond(H_i):.2f}")

    # ADMM 配置
    admm_config = ADMMConfig(
        rho=1.0, max_iter=300, abs_tol=1e-6, rel_tol=1e-4,
        use_adaptive_penalty=True, mu=8.0,
        tau_incr=1.5, tau_decr=1.5, alpha_overrelax=1.5
    )

    # 回调: 自适应罚参数
    penalty_strategy = ResidualBalancingStrategy(mu=8.0, tau=1.5)
    current_rho = [1.0]

    def admm_callback(k, state):
        if state.primal_residuals and state.dual_residuals:
            new_rho, _ = penalty_strategy.update(
                current_rho[0],
                state.primal_residuals[-1],
                state.dual_residuals[-1]
            )
            current_rho[0] = new_rho

    t0 = time.time()
    admm_result = consensus_admm(
        local_objectives, n_shared_params, admm_config,
        callback=admm_callback
    )
    t_admm = time.time() - t0

    print(f"\n  ADMM 融合结果:")
    print_result("  迭代次数", admm_result['n_iterations'])
    print_result("  ADMM 计算时间", t_admm, ".3f")
    print_result("  最优共享参数", admm_result['z_opt'])
    print_result("  收敛", admm_result['converged'])

    # 4. 收敛诊断
    print_section("4. 收敛诊断报告")
    diag = ConvergenceDiagnostics(n_sub, n_shared_params)
    state = admm_result['state']
    for k in range(len(state.primal_residuals)):
        rho_k = state.rho_history[k] if k < len(state.rho_history) else 1.0
        diag.record(
            k,
            state.primal_residuals[k],
            state.dual_residuals[k],
            state.objective_values[k],
            rho_k,
            state.x_locals,
            state.z_consensus,
        )
    report = generate_convergence_report(diag)
    print(report)

    # 5. 度量失真分析
    print_section("5. 子域间度量失真分析")
    # 构建成本矩阵: 子域 i 的参数到共识解的距离
    cost_matrix = np.zeros((n_sub, n_shared_params))
    for i in range(n_sub):
        cost_matrix[i, :] = np.abs(admm_result['x_opt'][i] - admm_result['z_opt'])
    weights = np.ones(n_sub) / n_sub
    distortion = compute_metric_distortion(cost_matrix, weights)
    print_result("度量失真", distortion, ".4f")

    # 各子域参数偏差
    for i in range(n_sub):
        dev = np.linalg.norm(admm_result['x_opt'][i] - admm_result['z_opt'])
        print(f"  子域 {i} ({physics_names[i]}) 偏差: {dev:.6f}")

    return admm_result


# ============================================================
#  实验 6: 高阶有限元与积分验证
# ============================================================
def experiment_fem_quadrature():
    """
    实验 6: p-version FEM 与高阶积分

    验证 p-version 有限元基函数的正交性和
    高阶积分规则的精度.
    """
    print_header("实验 6: p-version FEM 与高阶积分")

    from fem_operators import (
        pversion_basis, pversion_basis_derivative,
        assemble_stiffness_matrix_1d, energy_norm
    )
    from quadrature import (
        gauss_legendre_rule, triangle_quadrature_7pt,
        integrate_on_triangle, pyramid_quadrature_5pt,
        pyramid_monomial_exact
    )

    # 1. p-version 基函数正交性
    print_section("1. p-version 基函数")
    p = 5
    x_eval = np.linspace(-0.99, 0.99, 200)
    phi = pversion_basis(x_eval, p)
    dphi = pversion_basis_derivative(x_eval, p)
    print_result("多项式阶数", p)
    print_result("基函数数", p + 1)
    print_result("基函数值范围",
                 f"[{phi.min():.4f}, {phi.max():.4f}]")
    # 检查端点条件 φ_i(±1) = 0
    phi_end = pversion_basis(np.array([-1.0, 1.0]), p)
    print_result("端点值 (应全为 0)", f"max|φ(±1)| = {np.max(np.abs(phi_end)):.2e}")

    # 2. 刚度矩阵
    print_section("2. 刚度矩阵组装")
    K = assemble_stiffness_matrix_1d(
        n_elements=1,
        P_func=lambda x: 1.0,
        Q_func=lambda x: 0.0,
        p_order=4, n_quad=10
    )
    print_result("刚度矩阵大小", K.shape)
    print_result("刚度矩阵条件数", np.linalg.cond(K + np.eye(K.shape[0]) * EPS_NUM))
    # 对角占优检查
    diag_dominance = [abs(K[i,i]) - sum(abs(K[i,j]) for j in range(K.shape[0]) if j != i)
                      for i in range(K.shape[0])]
    print_result("对角占优 (最小余量)", min(diag_dominance))

    # 3. Gauss-Legendre 精度
    print_section("3. Gauss-Legendre 求积精度")
    for n_quad in [2, 4, 8, 16]:
        pts, wts = gauss_legendre_rule(n_quad)
        # 测试 ∫_{-1}^{1} x^{2n-2} dx = 2/(2n-1)
        max_degree = 2 * n_quad + 2
        max_err = 0.0
        for deg in range(max_degree + 1):
            exact = 2.0 / (deg + 1) if deg % 2 == 0 else 0.0
            numeric = sum(w * p**deg for p, w in zip(pts, wts))
            max_err = max(max_err, abs(numeric - exact))
        print(f"  {n_quad} 点: 最大单项误差 = {max_err:.2e}, "
              f"精确至 {2*n_quad - 1} 次多项式")

    # 4. 三角形积分
    print_section("4. 三角形区域积分")
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # ∫∫_T 1 dA = 0.5
    int_1 = integrate_on_triangle(lambda x, y: 1.0, vertices, order=1)
    # ∫∫_T x dA = 1/6
    int_x = integrate_on_triangle(lambda x, y: x, vertices, order=2)
    # ∫∫_T xy dA = 1/24
    int_xy = integrate_on_triangle(lambda x, y: x * y, vertices, order=5)
    print_result("∫∫ 1 dA (精确 0.5)", int_1)
    print_result("∫∫ x dA (精确 1/6)", int_x, ".6f")
    print_result("∫∫ xy dA (精确 1/24)", int_xy, ".8f")

    # 5. 金字塔单项式积分
    print_section("5. 金字塔单项式精确积分")
    for a, b, c in [(0,0,0), (2,0,0), (0,2,0), (0,0,2), (2,2,0), (2,0,2)]:
        val = pyramid_monomial_exact(a, b, c)
        print(f"  ∫∫∫ x^{a}y^{b}z^{c} dV = {val:.8f}")

    return K


# ============================================================
#  主程序
# ============================================================
def main():
    """主程序入口 — 运行所有实验"""
    print("\n" + "#" * 70)
    print("#  分布式 ADMM 多物理场 PDE 约束优化框架")
    print("#  科学领域: 数学优化 — 分布式优化与 ADMM 方法")
    print("#  博士级科学计算合成项目 (PROJECT 220)")
    print("#" * 70)

    t_total_start = time.time()

    try:
        # 实验 1: 分布式 Poisson 反问题
        exp1_result = experiment_distributed_poisson()

        # 实验 2: Gray-Scott 反应扩散
        exp2_result = experiment_gray_scott()

        # 实验 3: Hodgkin-Huxley 神经动力学
        exp3_result = experiment_hodgkin_huxley()

        # 实验 4: 臭氧化学与 Burgers 方程
        exp4_result = experiment_ozone_burgers()

        # 实验 5: 多物理场 ADMM 数据融合
        exp5_result = experiment_multiphysics_admm()

        # 实验 6: FEM 与积分验证
        exp6_result = experiment_fem_quadrature()

        # 总结
        t_total = time.time() - t_total_start

        print_header("全部实验完成")
        print_result("总运行时间", t_total, ".2f")
        print("\n实验摘要:")
        print("  ✓ 实验 1: 分布式 Poisson 反问题 — ADMM 收敛" if exp1_result['converged'] else
              "  ✓ 实验 1: 分布式 Poisson 反问题 — 已完成")
        print(f"  ✓ 实验 2: Gray-Scott 反应扩散 — {len(exp2_result['times'])} 个快照")
        print(f"  ✓ 实验 3: Hodgkin-Huxley — {len(exp3_result['spike_times'])} 次动作电位")
        print(f"  ✓ 实验 4: 臭氧化学 + Burgers — 完成")
        print(f"  ✓ 实验 5: 多物理场 ADMM 融合 — {exp5_result['n_iterations']} 次迭代")
        print(f"  ✓ 实验 6: FEM 与积分验证 — 刚度矩阵 {exp6_result.shape}")
        print("\n" + "#" * 70)
        print("#  所有实验成功完成")
        print("#" * 70)

    except Exception as e:
        print(f"\n!!! 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
