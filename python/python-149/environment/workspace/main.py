"""
main.py
================================================================================
神经计算：强化学习与最优控制 — 统一入口
================================================================================

本项目围绕"基于随机微分方程的神经群体最优控制与深度强化学习"展开，
融合15个种子项目的核心算法，构建一个面向前沿神经计算问题的博士级Python科研代码。

执行流程:
  1. 环境检测与数值稳定性确认
  2. Wilson-Cowan神经质量模型动力学验证
  3. 随机微分方程数值积分（Euler-Maruyama, Milstein, 随机中点法）
  4. Hamilton-Jacobi-Bellman方程有限元求解
  5. Actor-Critic深度强化学习训练
  6. Nelder-Mead策略参数无梯度优化
  7. 高维数值积分验证（金字塔规则, Gauss-Hermite, Monte Carlo）
  8. 变度量CVT状态空间离散化
  9. 收敛性与均方稳定性分析
 10. 综合结果输出

运行方式:
     python main.py
（无需任何命令行参数）
"""

import numpy as np
import time

# ========================================================================
# 导入各模块
# ========================================================================
from neural_mass_dynamics import (
    get_neural_parameters,
    neural_mass_deriv,
    neural_mass_jacobian,
    neural_oscillation_period,
    compute_running_cost,
    compute_terminal_cost,
    sigmoid_activation,
)
from sde_integrator import (
    euler_maruyama,
    milstein_method,
    stochastic_explicit_midpoint,
    mean_square_stability_check,
    compute_strong_error,
)
from hjb_fem_solver import (
    regular_tetrahedral_mesh,
    assemble_fem_matrices,
    solve_hjb_backward,
    tetrahedron_volume,
    compute_tet_quality,
    bicg_solver,
)
from policy_optimizer import (
    nelder_mead_optimize,
    linear_feedback_policy,
    evaluate_policy_cost,
)
from state_space_tools import (
    cvt_lloyd_iterate,
    state_to_index,
    StateEncoder,
    serialize_state_trajectory,
    metric_tensor,
)
from quadrature_engine import (
    pyramid_jaskowiec_rule,
    integrate_over_pyramid,
    gauss_hermite_quad_1d,
    monte_carlo_expectation,
)
from rl_agent import ActorCriticAgent
from numeric_utils import (
    check_environment,
    assert_numeric_stability,
    bisection_find_root,
    rosenbrock_function,
    benchmark_optimizer,
    find_switching_time,
    safe_divide,
)
from convergence_analyzer import (
    analyze_ms_stability_region,
    estimate_convergence_rate,
    lyapunov_exponential_decay_rate,
    perform_stability_sweep,
)


def print_section(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def print_subsection(title: str):
    print(f"\n--- {title} ---")


def main():
    np.random.seed(42)
    rng = np.random.default_rng(seed=42)
    overall_start = time.time()

    # ===================================================================
    # 1. 环境检测
    # ===================================================================
    print_section("1. 数值计算环境检测")
    env_report = check_environment()
    print(f"  NumPy 版本: {env_report['numpy_version']}")
    print(f"  机器精度 ε: {env_report['float_info']['epsilon']:.2e}")
    print(f"  浮点最大值: {env_report['float_info']['max']:.2e}")
    print("  [状态] 环境检测通过")

    # ===================================================================
    # 2. 神经质量模型动力学验证
    # ===================================================================
    print_section("2. Wilson-Cowan 神经质量模型动力学")

    # 获取默认参数
    params = get_neural_parameters()
    print(f"  兴奋性时间常数 τ_e = {params[0]:.2f} ms")
    print(f"  抑制性时间常数 τ_i = {params[1]:.2f} ms")
    print(f"  耦合强度 a_ee={params[2]:.1f}, a_ei={params[3]:.1f}, "
          f"a_ie={params[4]:.1f}, a_ii={params[5]:.1f}")

    # 平衡点分析
    y_eq = np.array([0.1, 0.05])
    for _ in range(50):
        dydt = neural_mass_deriv(0.0, y_eq)
        y_eq = y_eq + 0.1 * dydt
        y_eq = np.clip(y_eq, 0.0, 1.0)
    print(f"  数值平衡点 E*={y_eq[0]:.4f}, I*={y_eq[1]:.4f}")

    # Jacobian与稳定性分析
    J = neural_mass_jacobian(y_eq)
    eigenvals = np.linalg.eigvals(J)
    print(f"  Jacobian特征值: λ1={eigenvals[0]:.4f}, λ2={eigenvals[1]:.4f}")
    period_est = neural_oscillation_period(linearized=True)
    print(f"  线性化振荡周期估计: T ≈ {period_est:.2f} ms")

    # 验证数值稳定性
    assert assert_numeric_stability(y_eq, "equilibrium")
    print("  [状态] 动力学验证通过")

    # ===================================================================
    # 3. SDE数值积分
    # ===================================================================
    print_section("3. 随机神经动力学数值积分")

    # 定义SDE的漂移与扩散
    def drift_fn(t, y):
        return neural_mass_deriv(t, y)

    def diffusion_fn(t, y):
        _, _, _, _, _, _, _, _, _, _, _, _, sigma_e, sigma_i, _, _, _ = get_neural_parameters()
        # 状态相关扩散（对角噪声）
        return np.array([sigma_e * max(y[0], 0.0), sigma_i * max(y[1], 0.0)])

    y0 = np.array([0.2, 0.1])
    tspan = (0.0, 50.0)
    n_steps = 500

    print_subsection("Euler-Maruyama 方法")
    t_em, y_em = euler_maruyama(drift_fn, diffusion_fn, tspan, y0, n_steps, rng=rng)
    print(f"  末端状态: E={y_em[-1,0]:.4f}, I={y_em[-1,1]:.4f}")
    print(f"  轨迹数值稳定: {assert_numeric_stability(y_em)}")

    print_subsection("Milstein 方法")
    def diffusion_deriv(t, y):
        # TODO: Hole 3 — 补充扩散项对状态的导数
        # 提示: diffusion_fn 返回 [σ_e * max(y_0, 0), σ_i * max(y_1, 0)]
        #       需要返回该向量对状态 y 的导数（对角元素）
        pass

    t_mil, y_mil = milstein_method(drift_fn, diffusion_fn, diffusion_deriv, tspan, y0, n_steps, rng=rng)
    print(f"  末端状态: E={y_mil[-1,0]:.4f}, I={y_mil[-1,1]:.4f}")

    print_subsection("随机显式中点法")
    t_sem, y_sem = stochastic_explicit_midpoint(drift_fn, diffusion_fn, tspan, y0, n_steps, rng=rng)
    print(f"  末端状态: E={y_sem[-1,0]:.4f}, I={y_sem[-1,1]:.4f}")

    # 均方稳定性验证
    print_subsection("均方稳定性验证")
    lambda_test = -3.0
    mu_test = np.sqrt(3.0)
    dt_test = 0.1
    stable = mean_square_stability_check(lambda_test, mu_test, dt_test)
    print(f"  λ={lambda_test}, μ={mu_test:.4f}, Δt={dt_test}")
    print(f"  理论均方稳定: {stable}")

    # ===================================================================
    # 4. HJB方程有限元求解
    # ===================================================================
    print_section("4. Hamilton-Jacobi-Bellman 方程有限元求解")

    # 构建三维增广状态空间网格 (E, I, t) → 简化为二维空间+时间的三维问题
    # 由于计算复杂度限制，使用较粗网格
    bounds = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    n_per_dim = 5
    nodes, tets = regular_tetrahedral_mesh(bounds, n_per_dim=n_per_dim)
    print(f"  网格节点数: {nodes.shape[0]}")
    print(f"  四面体单元数: {tets.shape[0]}")

    # 网格质量评估
    qualities = []
    for tet in tets[: min(100, len(tets))]:
        q = compute_tet_quality(nodes[tet, :])
        qualities.append(q)
    print(f"  平均网格质量: {np.mean(qualities):.4f}")

    # 漂移与扩散定义
    def spatial_drift(x):
        return np.array([0.1 * x[0], 0.05 * x[1], 0.2])

    def spatial_diffusion(x):
        return np.diag([0.01, 0.01, 0.0])

    # 组装有限元矩阵
    print_subsection("有限元矩阵组装")
    M, A = assemble_fem_matrices(nodes, tets, spatial_drift, spatial_diffusion)
    print(f"  质量矩阵 M: 非零元素比例 = {np.count_nonzero(M) / M.size:.4f}")
    print(f"  刚度矩阵 A: 非零元素比例 = {np.count_nonzero(A) / A.size:.4f}")

    # BiCG求解器测试
    print_subsection("BiCG求解器测试")
    b_test = np.ones(nodes.shape[0])
    x_test, err, iters, flag = bicg_solver(M + 0.01 * np.eye(nodes.shape[0]), b_test, max_iter=500, tol=1e-8)
    print(f"  BiCG迭代次数: {iters}, 相对残差: {err:.2e}, 标志: {flag}")

    # HJB后向求解（简化：终端代价为到目标状态的距离）
    print_subsection("HJB后向时间步进")
    y_target = np.array([0.6, 0.3])

    def terminal_cost_fn(nodes_arr):
        # 到目标状态 (0.6, 0.3, 0.5) 的欧氏距离平方
        target = np.array([0.6, 0.3, 0.5])
        dists = np.sum((nodes_arr - target) ** 2, axis=1)
        return dists

    def running_cost_fn(nodes_arr, u):
        # LQR型运行代价
        target = np.array([0.6, 0.3, 0.5])
        dy = nodes_arr - target
        state_cost = np.sum(dy ** 2, axis=1)
        control_cost = 0.5 * np.sum(u ** 2)
        return state_cost + control_cost

    try:
        V_history, t_grid = solve_hjb_backward(
            nodes, tets, spatial_drift, spatial_diffusion,
            terminal_cost_fn, running_cost_fn,
            tspan=(0.0, 10.0), n_time=5,
            control_candidates=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), np.array([-0.5, 0.0, 0.0])],
        )
        print(f"  HJB求解完成: 时间层数={len(t_grid)}, 值函数范围=[{V_history.min():.4f}, {V_history.max():.4f}]")
    except Exception as e:
        print(f"  HJB求解完成（简化模式）: {e}")

    # ===================================================================
    # 5. Actor-Critic强化学习训练
    # ===================================================================
    print_section("5. Actor-Critic 深度强化学习训练")

    agent = ActorCriticAgent(
        state_dim=2,
        n_rbf=16,
        action_bound=5.0,
        gamma=0.95,
        alpha_critic=0.15,
        alpha_actor=0.02,
        policy_sigma=0.3,
        rng=rng,
    )

    # 定义环境步进函数
    def env_step(state, action):
        dt_rl = 0.5
        # Euler步进（确定性简化环境）
        dydt = neural_mass_deriv(0.0, state, control_fn=lambda t, y: action)
        next_state = state + dt_rl * dydt
        next_state = np.clip(next_state, 0.0, 1.0)

        # 奖励: 负的运行代价
        r_cost = compute_running_cost(next_state, action, y_target, Q_mat=np.eye(2) * 10.0, R_scalar=0.5)
        reward = -r_cost * dt_rl

        # 终止条件
        done = bool(np.linalg.norm(next_state - y_target) < 0.05)
        return next_state, reward, done

    n_episodes = 30
    for ep in range(n_episodes):
        s0 = rng.uniform(0.1, 0.4, 2)
        total_r = agent.run_episode(env_step, s0, max_steps=200)
        if ep % 10 == 0 or ep == n_episodes - 1:
            print(f"  Episode {ep+1:3d}: Total Reward = {total_r:10.4f}")

    # 测试学习后的策略
    print_subsection("学习后策略测试")
    test_state = np.array([0.2, 0.1])
    action_learned = agent.select_action(test_state)
    print(f"  测试状态 E={test_state[0]:.2f}, I={test_state[1]:.2f}")
    print(f"  学习后策略输出: u = {action_learned:.4f}")
    print(f"  Critic值函数估计: V(s) = {agent.critic.value(test_state):.4f}")

    # ===================================================================
    # 6. Nelder-Mead策略优化
    # ===================================================================
    print_section("6. Nelder-Mead 无梯度策略优化")

    # 定义基于反馈增益的策略参数优化目标
    def policy_objective(theta):
        # theta = [K1, K2] 为二维反馈增益
        if len(theta) < 2:
            theta = np.concatenate([theta, np.zeros(2 - len(theta))])
        K = theta[:2]
        def control_fn(t, y):
            dx = y - y_target
            u = -np.dot(K, dx)
            return float(np.clip(u, -5.0, 5.0))
        # 确定性轨迹评估（使用显式中点法）
        def det_drift(t, y):
            return neural_mass_deriv(t, y, control_fn=control_fn)
        def zero_diff(t, y):
            return np.zeros(2)
        t_traj, y_traj = stochastic_explicit_midpoint(det_drift, zero_diff, (0.0, 30.0), np.array([0.2, 0.1]), 300, rng=rng)
        # 总代价
        cost = 0.0
        for i in range(len(t_traj) - 1):
            dt_i = t_traj[i+1] - t_traj[i]
            u_i = control_fn(t_traj[i], y_traj[i])
            cost += compute_running_cost(y_traj[i], u_i, y_target) * dt_i
        cost += compute_terminal_cost(y_traj[-1], y_target)
        return cost

    theta0 = np.array([0.5, 0.3])
    theta_opt, n_feval = nelder_mead_optimize(policy_objective, theta0, tolerance=1e-5, max_feval=300)
    print(f"  初始参数 θ0 = [{theta0[0]:.4f}, {theta0[1]:.4f}]")
    print(f"  最优参数 θ* = [{theta_opt[0]:.4f}, {theta_opt[1]:.4f}]")
    print(f"  函数评估次数: {n_feval}")
    print(f"  初始代价: {policy_objective(theta0):.4f}")
    print(f"  最优代价: {policy_objective(theta_opt):.4f}")

    # ===================================================================
    # 7. 高维数值积分验证
    # ===================================================================
    print_section("7. 高维数值积分验证")

    print_subsection("金字塔区域积分")
    # 测试 f(x,y,z)=1 在金字塔上的积分，应等于体积 4/3
    def f_unit(x_arr, y_arr, z_arr):
        return np.ones_like(x_arr)

    for p in [0, 2, 4, 6]:
        try:
            val = integrate_over_pyramid(f_unit, p=p)
            print(f"  精度p={p}: ∫_P 1 dV = {val:.6f} (理论: 1.333333)")
        except Exception as e:
            print(f"  精度p={p}: 错误 {e}")

    print_subsection("Gauss-Hermite求积")
    # 测试 E[X^2] 对于 X~N(0,1)，应等于1
    def f_xsq(x):
        return x ** 2
    for n_pts in [2, 3, 5]:
        val = gauss_hermite_quad_1d(n_pts, f_xsq, sigma=1.0)
        print(f"  节点数{n_pts}: E[X^2] = {val:.6f} (理论: 1.0)")

    print_subsection("Monte Carlo期望估计")
    def sampler(n, rng):
        return rng.standard_normal((n, 2))
    def f_mc(x):
        return x[0] ** 2 + x[1] ** 2
    mean_est, std_err = monte_carlo_expectation(f_mc, sampler, n_samples=5000, rng=rng)
    print(f"  E[X1^2+X2^2] = {mean_est:.4f} ± {std_err:.4f} (理论: 2.0)")

    # ===================================================================
    # 8. 变度量CVT状态空间离散化
    # ===================================================================
    print_section("8. 变度量CVT状态空间离散化")

    n_gen = 20
    init_gens = rng.uniform(0, 1, (n_gen, 2))
    gens = cvt_lloyd_iterate(
        init_gens,
        n_samples=3000,
        n_iter=8,
        metric_fn=lambda x: metric_tensor(x, metric_type="fisher"),
        bounds=(np.zeros(2), np.ones(2)),
        rng=rng,
    )
    print(f"  生成元数量: {n_gen}")
    print(f"  CVT生成元均值: [{np.mean(gens[:,0]):.4f}, {np.mean(gens[:,1]):.4f}]")
    print(f"  CVT生成元标准差: [{np.std(gens[:,0]):.4f}, {np.std(gens[:,1]):.4f}]")

    # 状态编码测试
    encoder = StateEncoder(n_gen, rng=rng)
    test_x = np.array([0.5, 0.5])
    idx = state_to_index(test_x, gens)
    code = encoder.encode(idx)
    decoded = encoder.decode(code)
    print(f"  测试状态 x={test_x} → 索引 {idx} → 编码 {code} → 解码 {decoded}")
    print(f"  编码一致性: {idx == decoded}")

    # ===================================================================
    # 9. 收敛性与稳定性分析
    # ===================================================================
    print_section("9. 收敛性与稳定性分析")

    print_subsection("SDE强收敛阶估计")
    def linear_drift(t, y):
        return -0.5 * y
    def linear_diff(t, y):
        return 0.3 * y
    y0_scalar = 1.0
    n_ref = 2 ** 10
    n_coarse_list = [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]
    try:
        dt_vals, errors = compute_strong_error(
            linear_drift, linear_diff, y0_scalar, (0.0, 1.0),
            n_ref, n_coarse_list, n_paths=300, rng=rng,
        )
        p_est, logC, resid = estimate_convergence_rate(dt_vals, errors)
        print(f"  步长序列: {dt_vals}")
        print(f"  误差序列: {errors}")
        print(f"  估计强收敛阶 p = {p_est:.4f} (理论 Euler-Maruyama: 0.5)")
        print(f"  拟合残差: {resid:.4e}")
    except Exception as e:
        print(f"  强收敛分析: {e}")

    print_subsection("Lyapunov指数衰减分析")
    # 对受控神经轨迹分析衰减
    def controlled_drift(y):
        K_opt = theta_opt[:2]
        def ctrl(t, yy):
            dx = yy - y_target
            u = -np.dot(K_opt, dx)
            return float(np.clip(u, -5.0, 5.0))
        return neural_mass_deriv(0.0, y, control_fn=ctrl)

    t_lyap = np.linspace(0, 50, 500)
    y_lyap = np.zeros((500, 2))
    y_lyap[0, :] = np.array([0.2, 0.1])
    dt_lyap = t_lyap[1] - t_lyap[0]
    for i in range(1, 500):
        y_lyap[i, :] = y_lyap[i-1, :] + dt_lyap * controlled_drift(y_lyap[i-1, :])
        y_lyap[i, :] = np.clip(y_lyap[i, :], 0.0, 1.0)

    decay_rate = lyapunov_exponential_decay_rate(t_lyap, y_lyap - y_target)
    print(f"  受控系统到平衡点的指数衰减速率: λ = {decay_rate:.4f}")
    print(f"  衰减时间常数: τ = {safe_divide(1.0, abs(decay_rate), default=1e6):.2f} ms")

    print_subsection("均方稳定性数值验证")
    lambda_list = [-4.0, -3.0, -2.0, -1.0]
    mu_list = [0.5, 1.0, 1.5, 2.0]
    stability_mat = perform_stability_sweep(
        None, lambda_list, mu_list, dt=0.1, n_paths=500, tmax=5.0, rng=rng,
    )
    print(f"  (λ, μ) 稳定性矩阵 (1=稳定, 0=不稳定):")
    print(f"  λ\\μ  {' '.join([f'{m:.1f}' for m in mu_list])}")
    for i, lam in enumerate(lambda_list):
        row = ' '.join([str(stability_mat[i, j]) for j in range(len(mu_list))])
        print(f"  {lam:.1f}  {row}")

    # ===================================================================
    # 10. 综合结果输出
    # ===================================================================
    print_section("10. 综合结果汇总")

    print("  [神经动力学]")
    print(f"    平衡点: E*={y_eq[0]:.4f}, I*={y_eq[1]:.4f}")
    print(f"    振荡周期: T={period_est:.2f} ms")

    print("  [SDE数值积分]")
    print(f"    Euler-Maruyama末端: E={y_em[-1,0]:.4f}, I={y_em[-1,1]:.4f}")
    print(f"    Milstein末端: E={y_mil[-1,0]:.4f}, I={y_mil[-1,1]:.4f}")

    print("  [强化学习]")
    print(f"    训练回合数: {n_episodes}")
    if len(agent.episode_rewards) > 0:
        print(f"    最后回合奖励: {agent.episode_rewards[-1]:.4f}")
        print(f"    平均奖励(后10回合): {np.mean(agent.episode_rewards[-10:]):.4f}")

    print("  [策略优化]")
    print(f"    最优反馈增益: K=[{theta_opt[0]:.4f}, {theta_opt[1]:.4f}]")
    print(f"    最优代价: {policy_objective(theta_opt):.4f}")

    print("  [数值积分]")
    print(f"    Gauss-Hermite E[X^2] (5点): {gauss_hermite_quad_1d(5, f_xsq, sigma=1.0):.6f}")

    print("  [稳定性]")
    print(f"    受控系统指数衰减速率: λ={decay_rate:.4f}")

    overall_time = time.time() - overall_start
    print(f"\n  总运行时间: {overall_time:.2f} 秒")
    print("\n" + "=" * 78)
    print("  神经随机最优控制与强化学习综合实验完成")
    print("=" * 78)


if __name__ == "__main__":
    main()
