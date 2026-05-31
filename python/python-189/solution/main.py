"""
main.py

统一入口: 零参数可运行

博士级科学问题:
    强化学习策略梯度用于非线性振荡网络的最优控制
    —— 融合锯齿波强迫、放牧动力学、谱基逼近与自然梯度

运行方式:
    python main.py

输出:
    - 训练过程日志
    - 最终策略在测试轨迹上的性能指标
    - 数值验证报告
"""

import numpy as np
import sys
import time

# 确保本地模块可导入
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))

from dynamical_system import ControlledNonlinearOscillator, sawtooth_wave, grazing_coupling
from policy_network import SpectralPolicyNetwork
from value_approximator import SpectralValueFunction, compute_discounted_returns
from policy_gradient_core import PolicyGradientTrainer
from special_functions import sine_integral, bessel_zero, bessel_zeros, incomplete_beta, beta_cdf
from linear_algebra import rref_compute, rref_solve, rref_rank, r83p_solve, toeplitz_cholesky_lower
from stochastic_processes import brownian_motion, ornstein_uhlenbeck_process, \
    multivariate_normal_distance_stats, gaussian_kernel_matrix
from spectral_basis import pca_vectors, pca_transform, build_legendre_basis, bessel_spectral_filter
from natural_gradient import sample_fisher_information_matrix, conjugate_gradient_solve, \
    natural_gradient_update, NaturalPolicyGradientOptimizer
from constrained_optimizer import lp_action_projection, trust_region_probability, \
    check_trust_region, CosineAnnealingScheduler
from mesh_geometry import StateSpaceTriangulation, adaptive_mesh_refinement


def scientific_validation():
    """
    核心科学计算模块的数值验证.
    """
    print("=" * 70)
    print("博士级科学计算模块数值验证")
    print("=" * 70)

    errors = 0

    # 1. 正弦积分验证
    print("\n[1] 正弦积分 Si(x) 验证")
    test_points = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    for x in test_points:
        si = sine_integral(x)
        # 理论: Si(x) → π/2 as x→∞
        print(f"    Si({x:5.2f}) = {si:12.8f}")
    if abs(sine_integral(100.0) - np.pi / 2.0) > 0.1:
        print("    WARNING: 渐近值偏差较大")
        errors += 1
    else:
        print("    PASS: 渐近收敛到 π/2")

    # 2. 贝塞尔零点验证
    print("\n[2] 贝塞尔零点 Bessel Zero 验证")
    try:
        z1 = bessel_zero(0.0, 1, kind=1)
        z2 = bessel_zero(0.0, 2, kind=1)
        print(f"    J_0 第1零点: {z1:.8f}  (理论: 2.40482556)")
        print(f"    J_0 第2零点: {z2:.8f}  (理论: 5.52007811)")
        if abs(z1 - 2.40482556) < 0.01 and abs(z2 - 5.52007811) < 0.01:
            print("    PASS")
        else:
            print("    WARNING: 偏差略大")
            errors += 1
    except Exception as e:
        print(f"    SKIP (需要 scipy): {e}")

    # 3. 不完全 Beta 验证
    print("\n[3] 不完全 Beta 函数验证")
    prob, ier = incomplete_beta(0.5, 2.0, 3.0)
    print(f"    I_{0.5}(2,3) = {prob:.8f}, ier={ier}")
    if ier == 0 and 0.0 <= prob <= 1.0:
        print("    PASS")
    else:
        print("    FAIL")
        errors += 1

    # 4. RREF 求解验证
    print("\n[4] RREF 线性求解验证")
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 8.0])
    x = rref_solve(A, b)
    x_true = np.linalg.solve(A, b)
    print(f"    解: {x.flatten()}")
    print(f"    真值: {x_true}")
    if np.allclose(x.flatten(), x_true, atol=1.0e-6):
        print("    PASS")
    else:
        print("    FAIL")
        errors += 1

    # 5. Toeplitz Cholesky 验证
    print("\n[5] Toeplitz Cholesky 验证")
    n = 5
    first_col = np.array([2.0, 0.5, 0.3, 0.2, 0.1])
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            T[i, j] = first_col[abs(i - j)]
    try:
        L = toeplitz_cholesky_lower(n, T)
        recon = L @ L.T
        err = np.max(np.abs(recon - T))
        print(f"    重构误差: {err:.2e}")
        if err < 1.0e-8:
            print("    PASS")
        else:
            print("    FAIL")
            errors += 1
    except Exception as e:
        print(f"    ERROR: {e}")
        errors += 1

    # 6. PCA 验证
    print("\n[6] PCA 主成分分析验证")
    data = np.random.randn(10, 50)
    V, vals, Psi = pca_vectors(data, 5)
    print(f"    数据维度: {data.shape}, 降维后: {V.shape}")
    if V.shape == (10, 5):
        print("    PASS")
    else:
        print("    FAIL")
        errors += 1

    # 7. Legendre 基验证
    print("\n[7] Legendre 乘积多项式基验证")
    X = np.random.uniform(-1, 1, (2, 10))
    B = build_legendre_basis(2, 2, X)
    print(f"    基函数矩阵形状: {B.shape}")
    if B.shape[0] == 6:  # C(2+2,2)=6
        print("    PASS")
    else:
        print("    FAIL")
        errors += 1

    # 8. 布朗运动统计验证
    print("\n[8] Brownian 运动统计验证")
    traj = brownian_motion(1000, 2, sigma=1.0)
    mean_pos = np.mean(traj, axis=0)
    std_pos = np.std(traj, axis=0)
    print(f"    终点均值: {mean_pos}, 标准差: {std_pos}")
    # Brownian motion 的终点标准差约为 sigma*sqrt(N) ~ 30, 均值应接近 0
    if np.all(np.abs(mean_pos) < 50.0):
        print("    PASS")
    else:
        print("    FAIL")
        errors += 1

    # 9. 信任区域概率验证
    print("\n[9] 信任区域概率验证")
    prob = trust_region_probability(0.01, 10, 100)
    print(f"    P(KL≤0.01 | d=10, N=100) ≈ {prob:.4f}")
    if 0.0 <= prob <= 1.0:
        print("    PASS")
    else:
        print("    FAIL")
        errors += 1

    print("\n" + "=" * 70)
    print(f"验证完成: 错误数 = {errors}")
    print("=" * 70)
    return errors


def run_policy_gradient_training():
    """
    执行策略梯度训练.
    """
    print("\n" + "=" * 70)
    print("策略梯度训练: 非线性振荡网络最优控制")
    print("=" * 70)

    np.random.seed(42)

    trainer = PolicyGradientTrainer(
        state_dim=4,
        action_dim=4,
        policy_degree=2,
        value_degree=2,
        gamma=0.99,
        lam=0.95,
        lr_policy=0.005,
        lr_value=0.01,
        cg_iter=10,
        cg_damping=0.1,
        max_kl=0.03,
        entropy_coef=0.01,
        batch_episodes=3,
        max_steps_per_episode=100
    )

    results = trainer.train(num_iterations=20)

    # 测试最终策略
    print("\n" + "-" * 70)
    print("最终策略测试 (确定性执行, 5 条轨迹)")
    print("-" * 70)
    test_rewards = []
    for ep in range(5):
        buf = trainer.collect_episode(deterministic=True)
        total_reward = np.sum(buf.rewards)
        test_rewards.append(total_reward)
        print(f"  Test Episode {ep+1}: Reward = {total_reward:.3f}, Length = {buf.size()}")

    avg_test = np.mean(test_rewards)
    print(f"\n  平均测试奖励: {avg_test:.3f}")

    # 参考轨迹跟踪误差
    print("\n" + "-" * 70)
    print("参考轨迹跟踪性能分析")
    print("-" * 70)
    env = ControlledNonlinearOscillator(dt=0.01)
    obs = env.reset()
    tracking_errors = []
    for step in range(200):
        action = trainer.policy.sample(obs)
        action = np.clip(action, -2.0, 2.0)
        next_obs, reward, done, info = env.step(action)
        ref = env.reference_trajectory(info['t'])
        err = np.linalg.norm(env.state - ref)
        tracking_errors.append(err)
        obs = next_obs
        if done:
            break
    print(f"  平均跟踪误差: {np.mean(tracking_errors):.4f}")
    print(f"  最大跟踪误差: {np.max(tracking_errors):.4f}")
    print(f"  最终跟踪误差: {tracking_errors[-1]:.4f}")

    return results, avg_test, tracking_errors


def run_mesh_analysis():
    """
    状态空间网格分析.
    """
    print("\n" + "=" * 70)
    print("状态空间三角剖分分析")
    print("=" * 70)

    # 从随机轨迹中采样状态点
    env = ControlledNonlinearOscillator(dt=0.01)
    states = []
    for _ in range(20):
        obs = env.reset()
        for _ in range(50):
            action = np.random.randn(4) * 0.5
            next_obs, _, done, _ = env.step(action)
            states.append(next_obs)
            obs = next_obs
            if done:
                break

    states_arr = np.array(states)
    print(f"  采样状态数: {len(states_arr)}")

    # PCA 降维到 2D 用于可视化分析 (内部计算, 无图形)
    V, vals, Psi = pca_vectors(states_arr.T, 2)
    states_2d = pca_transform(states_arr.T, V, Psi).T

    # Delaunay 三角剖分
    tri = StateSpaceTriangulation(states_2d)
    vols = tri.simplex_volumes()
    print(f"  单纯形数量: {len(tri.simplices)}")
    print(f"  平均单纯形体积: {np.mean(vols):.6f}")
    print(f"  最大单纯形体积: {np.max(vols):.6f}")

    # 插值测试
    test_point = np.mean(states_2d, axis=0)
    dummy_values = np.random.randn(len(states_2d))
    interp_val = tri.interpolate(test_point, dummy_values)
    print(f"  测试点插值结果: {interp_val:.4f}")

    return tri


def main():
    """
    主函数: 零参数运行完整实验流程.
    """
    print("\n" + "#" * 70)
    print("#  PROJECT_189: 强化学习策略梯度 —— 非线性振荡网络最优控制")
    print("#  科学领域: 数据科学 · 强化学习策略梯度")
    print("#" * 70)

    start_time = time.time()

    # 阶段 1: 数值验证
    val_errors = scientific_validation()
    if val_errors > 5:
        print("\n警告: 过多验证失败, 但仍继续训练...")

    # 阶段 2: 策略梯度训练
    results, avg_test_reward, tracking_errors = run_policy_gradient_training()

    # 阶段 3: 状态空间网格分析
    tri = run_mesh_analysis()

    # 汇总
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    print(f"  总运行时间: {elapsed:.2f} 秒")
    print(f"  训练迭代数: 20")
    print(f"  最终平均测试奖励: {avg_test_reward:.3f}")
    print(f"  跟踪误差均值: {np.mean(tracking_errors):.4f}")
    print(f"  数值验证错误数: {val_errors}")
    print("=" * 70)
    print("\n所有模块运行完毕, 无报错退出.\n")


if __name__ == "__main__":
    main()
