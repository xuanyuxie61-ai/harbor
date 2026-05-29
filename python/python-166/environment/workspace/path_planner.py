"""
path_planner.py
动态规划路径规划模块

融合种子项目:
- 156_change_dynamic: 动态规划硬币找零

科学应用: 软体机器人多段构型空间中的最优路径规划
使用动态规划在离散化构型空间中寻找最小能量路径
"""

import numpy as np
from typing import Tuple, List, Callable


def change_dynamic(coin_values: np.ndarray, target: int) -> np.ndarray:
    """
    动态规划硬币找零 — 直接来自种子项目156_change_dynamic

    计算用给定面额凑出1到target所需的最少硬币数

    递推关系:
        dp[0] = 0
        dp[j] = min(dp[j], dp[j - coin[i]] + 1)
    """
    if target < 1:
        return np.array([])

    INF = target + 1
    dp = np.full(target + 1, INF, dtype=int)
    dp[0] = 0

    for j in range(1, target + 1):
        for c in coin_values:
            if c <= j:
                dp[j] = min(dp[j], dp[j - c] + 1)

    return dp[1:]


def discretize_configuration_space(n_segments: int, n_angles: int,
                                   theta_max: float = np.pi / 2.0) -> np.ndarray:
    """
    离散化软体机器人构型空间

    每段有一个弯曲角度 theta_i ∈ [-theta_max, theta_max]
    总构型: theta = [theta_1, ..., theta_n_segments]

    返回所有可能的离散角度值
    """
    if n_angles < 2:
        n_angles = 2
    angles = np.linspace(-theta_max, theta_max, n_angles)
    return angles


def configuration_to_tip(n_segments: int, segment_length: float,
                         angles: np.ndarray) -> Tuple[float, float]:
    """
    从分段弯曲角度计算末端位置

    每段长度 l = segment_length
    第i段末端方向: phi_i = sum_{j=1}^i angles[j]
    末端位置:
        x = sum l * cos(phi_i)
        y = sum l * sin(phi_i)
    """
    x, y = 0.0, 0.0
    phi = 0.0
    for i in range(n_segments):
        phi += angles[i]
        x += segment_length * np.cos(phi)
        y += segment_length * np.sin(phi)
    return x, y


def energy_cost(current: np.ndarray, next_config: np.ndarray,
                stiffness: float = 1.0,
                damping: float = 0.1) -> float:
    """
    计算构型转移的能量代价

    E = 0.5 * k * ||next - current||^2 + 0.5 * c * ||next||^2
    """
    diff = next_config - current
    E = 0.5 * stiffness * np.sum(diff ** 2) + 0.5 * damping * np.sum(next_config ** 2)
    return E


def dp_path_planning_2d(n_segments: int, segment_length: float,
                        target: Tuple[float, float],
                        n_discrete: int = 11,
                        theta_max: float = np.pi / 3.0) -> Tuple[np.ndarray, float]:
    """
    2D软体臂的动态规划路径规划

    将每段角度离散化为 n_discrete 个值，
    使用动态规划寻找从直杆到目标的最小能量路径

    参数:
        n_segments: 段数
        segment_length: 每段长度
        target: (tx, ty) 目标位置
        n_discrete: 每段离散角度数
        theta_max: 最大弯曲角

    返回:
        optimal_angles: (n_segments,) 最优角度序列
        min_cost: 最小代价
    """
    angles = discretize_configuration_space(n_segments, n_discrete, theta_max)
    n_a = len(angles)

    # 对于少量段，使用完全枚举或简化DP
    if n_segments > 5:
        # 段数太多，使用贪心+局部搜索
        return _greedy_path_planning(n_segments, segment_length, target, angles)

    # 动态规划表
    # dp[i][j] = 前i段以角度angles[j]结束的最小代价
    INF = 1e10
    dp = np.full((n_segments, n_a), INF)
    parent = np.full((n_segments, n_a), -1, dtype=int)

    # 初始化第一段
    for j in range(n_a):
        # 从直杆(0)到angles[j]的代价
        dp[0, j] = energy_cost(np.array([0.0]), np.array([angles[j]]))

    # DP递推
    for i in range(1, n_segments):
        for j in range(n_a):
            for k in range(n_a):
                prev_angles = np.full(i, angles[k])
                curr_angles = np.full(i + 1, angles[j])
                # 只考虑最后一段的变化
                cost = dp[i - 1, k] + energy_cost(np.array([angles[k]]), np.array([angles[j]]))
                if cost < dp[i, j]:
                    dp[i, j] = cost
                    parent[i, j] = k

    # 找到最接近目标的最终构型
    best_j = -1
    best_dist = INF
    for j in range(n_a):
        # 假设所有段使用相同角度（简化）
        test_angles = np.full(n_segments, angles[j])
        tx, ty = configuration_to_tip(n_segments, segment_length, test_angles)
        dist = (tx - target[0]) ** 2 + (ty - target[1]) ** 2
        total_cost = dp[n_segments - 1, j] + 10.0 * dist  # 加权目标距离
        if total_cost < best_dist:
            best_dist = total_cost
            best_j = j

    # 回溯最优路径
    optimal_angles = np.zeros(n_segments)
    if best_j >= 0:
        j = best_j
        optimal_angles[n_segments - 1] = angles[j]
        for i in range(n_segments - 1, 0, -1):
            j = parent[i, j]
            if j < 0:
                j = 0
            optimal_angles[i - 1] = angles[j]

    min_cost = dp[n_segments - 1, best_j] if best_j >= 0 else INF
    return optimal_angles, min_cost


def _greedy_path_planning(n_segments: int, segment_length: float,
                          target: Tuple[float, float],
                          angles: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    贪心+局部搜索路径规划（用于段数较多时）
    """
    optimal_angles = np.zeros(n_segments)
    target_x, target_y = target

    # 初始猜测: 指向目标的方向
    target_angle = np.arctan2(target_y, target_x)
    avg_angle = target_angle / n_segments

    for i in range(n_segments):
        # 在离散角度中选择最接近avg_angle的值
        idx = np.argmin(np.abs(angles - avg_angle))
        optimal_angles[i] = angles[idx]

    # 局部搜索优化
    tx, ty = configuration_to_tip(n_segments, segment_length, optimal_angles)
    best_dist = (tx - target_x) ** 2 + (ty - target_y) ** 2

    for _ in range(50):
        improved = False
        for i in range(n_segments):
            for a in angles:
                old = optimal_angles[i]
                optimal_angles[i] = a
                tx, ty = configuration_to_tip(n_segments, segment_length, optimal_angles)
                dist = (tx - target_x) ** 2 + (ty - target_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    improved = True
                else:
                    optimal_angles[i] = old
        if not improved:
            break

    cost = energy_cost(np.zeros(n_segments), optimal_angles)
    return optimal_angles, cost


def multi_target_path_planning(n_segments: int, segment_length: float,
                               targets: List[Tuple[float, float]],
                               n_discrete: int = 9) -> List[np.ndarray]:
    """
    多目标点的路径规划（途经多个工作点）

    使用DP为每个目标点独立规划（使用不同随机种子增加多样性）
    """
    paths = []
    for idx, target in enumerate(targets):
        # 为不同目标使用不同的初始角度猜测，增加多样性
        angles, _ = dp_path_planning_2d(n_segments, segment_length, target, n_discrete)
        # 额外局部优化
        tx, ty = configuration_to_tip(n_segments, segment_length, angles)
        best_dist = (tx - target[0]) ** 2 + (ty - target[1]) ** 2
        # 随机扰动搜索
        rng = np.random.RandomState(idx * 17 + 42)
        angles_test = angles.copy()
        for _ in range(20):
            i_seg = rng.randint(0, n_segments)
            angles_test[i_seg] += rng.uniform(-0.1, 0.1)
            angles_test = np.clip(angles_test, -np.pi / 3.0, np.pi / 3.0)
            tx, ty = configuration_to_tip(n_segments, segment_length, angles_test)
            dist = (tx - target[0]) ** 2 + (ty - target[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                angles = angles_test.copy()
        paths.append(angles.copy())
    return paths
