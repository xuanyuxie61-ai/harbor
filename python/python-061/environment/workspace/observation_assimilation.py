"""
多源观测数据聚合与集合卡尔曼滤波同化模块
==========================================
基于种子项目 118_brc_naive 的大数据分组聚合思想。

核心科学问题：
    台风预报需要同化多源异构观测数据（卫星、雷达、飞机下投、浮标等）。
    本模块实现观测数据的分组聚合、质量控制和集合卡尔曼滤波（EnKF）
    同化算法。

观测数据分组模型（基于 brc_naive）：
    每个观测 o_i = (type, location, value, timestamp, error_variance)
    按类型分组后计算各组的加权平均：
    
    X̄_type = Σ w_i * x_i / Σ w_i
    
    其中权重 w_i = 1 / σ_i²（反方差加权）。

集合卡尔曼滤波（EnKF）：
    分析方程：
        X^a = X^f + K * (y - H * X^f)
    
    其中 Kalman 增益：
        K = P^f * Hᵀ * (H * P^f * Hᵀ + R)^{-1}
    
    在集合框架下，用样本协方差近似：
        P^f ≈ (1/(N-1)) * X' * X'ᵀ
    
    其中 X' 为扰动矩阵。

局地化（Localization）：
    为避免伪相关，引入局地化函数：
        ρ(r) = exp(-(r/L)²/2)  （Gaspari-Cohn 函数）
    
    其中 L 为局地化长度尺度。
"""

import numpy as np


class Observation:
    """单个观测数据点。"""
    def __init__(self, obs_type, lon, lat, value, timestamp, error_var):
        self.type = obs_type
        self.lon = lon
        self.lat = lat
        self.value = value
        self.timestamp = timestamp
        self.error_var = max(error_var, 1e-6)  # 防止零方差


class ObservationAggregator:
    """
    观测数据聚合器（基于 118_brc_naive 的分组聚合思想）。
    """
    def __init__(self):
        self.observations = []
    
    def add_observation(self, obs):
        """添加观测。"""
        self.observations.append(obs)
    
    def group_by_type(self):
        """
        按观测类型分组。
        
        返回:
            groups: dict，键为类型，值为 Observation 列表
        """
        groups = {}
        for obs in self.observations:
            if obs.type not in groups:
                groups[obs.type] = []
            groups[obs.type].append(obs)
        return groups
    
    def aggregate_group(self, obs_list):
        """
        对同一组观测进行反方差加权平均。
        
        公式：
            X̄ = Σ (x_i / σ_i²) / Σ (1/σ_i²)
            σ̄² = 1 / Σ (1/σ_i²)
        
        参数:
            obs_list: 同类型观测列表
        
        返回:
            mean_val: 加权平均值
            combined_var: 组合方差
            n: 观测数
        """
        if len(obs_list) == 0:
            return None, None, 0
        
        weights = np.array([1.0 / obs.error_var for obs in obs_list])
        values = np.array([obs.value for obs in obs_list])
        
        total_weight = np.sum(weights)
        if total_weight < 1e-12:
            return np.mean(values), np.var(values), len(obs_list)
        
        mean_val = np.sum(weights * values) / total_weight
        combined_var = 1.0 / total_weight
        
        return mean_val, combined_var, len(obs_list)
    
    def summarize_all_groups(self):
        """
        汇总所有观测类型的统计信息。
        
        返回:
            summary: dict，键为类型，值为统计信息字典
        """
        groups = self.group_by_type()
        summary = {}
        for obs_type, obs_list in groups.items():
            mean_val, var, n = self.aggregate_group(obs_list)
            summary[obs_type] = {
                'mean': mean_val,
                'variance': var,
                'count': n,
                'min': min(obs.value for obs in obs_list),
                'max': max(obs.value for obs in obs_list)
            }
        return summary


def gaspari_cohn_localization(distance, localization_length):
    """
    Gaspari-Cohn 局地化函数（五次样条）。
    
    定义：
        z = r / L
        
        若 z ≤ 1:
            ρ(z) = -z⁵/4 + z⁴/2 + 5z³/8 - 5z²/3 + 1
        若 1 < z ≤ 2:
            ρ(z) = z⁵/12 - z⁴/2 + 5z³/8 + 5z²/3 - 5z + 4 - 2/(3z)
        若 z > 2:
            ρ(z) = 0
    
    参数:
        distance: 距离（km）
        localization_length: 局地化长度尺度（km）
    
    返回:
        rho: 局地化系数
    """
    z = distance / localization_length
    
    if isinstance(z, np.ndarray):
        rho = np.zeros_like(z)
        mask1 = z <= 1.0
        mask2 = (z > 1.0) & (z <= 2.0)
        
        rho[mask1] = (-0.25 * z[mask1]**5 + 0.5 * z[mask1]**4
                      + 0.625 * z[mask1]**3 - (5.0/3.0) * z[mask1]**2 + 1.0)
        
        rho[mask2] = ((1.0/12.0) * z[mask2]**5 - 0.5 * z[mask2]**4
                      + 0.625 * z[mask2]**3 + (5.0/3.0) * z[mask2]**2
                      - 5.0 * z[mask2] + 4.0 - 2.0/(3.0 * z[mask2]))
        return rho
    else:
        if z <= 1.0:
            return (-0.25 * z**5 + 0.5 * z**4 + 0.625 * z**3
                    - (5.0/3.0) * z**2 + 1.0)
        elif z <= 2.0:
            return ((1.0/12.0) * z**5 - 0.5 * z**4 + 0.625 * z**3
                    + (5.0/3.0) * z**2 - 5.0 * z + 4.0 - 2.0/(3.0 * z))
        else:
            return 0.0


def ensemble_kalman_filter_update(ensemble_states, observations, observation_operator,
                                   obs_errors, localization_length=500.0):
    """
    集合卡尔曼滤波分析步骤。
    
    参数:
        ensemble_states: (n_ens, state_dim) 预报集合
        observations: (n_obs,) 观测向量
        observation_operator: (n_obs, state_dim) 观测算子矩阵 H
        obs_errors: (n_obs,) 观测误差标准差
        localization_length: 局地化长度（km）
    
    返回:
        analysis_states: (n_ens, state_dim) 分析集合
    """
    n_ens, state_dim = ensemble_states.shape
    n_obs = len(observations)
    
    # 集合均值
    x_mean = np.mean(ensemble_states, axis=0)
    
    # 扰动矩阵 X'
    X_prime = ensemble_states - x_mean[np.newaxis, :]
    
    # 样本协方差 P^f = X' * X'ᵀ / (N-1)
    # 在状态空间中直接计算
    
    # 预报观测
    Y_f = np.dot(ensemble_states, observation_operator.T)  # (n_ens, n_obs)
    y_mean = np.mean(Y_f, axis=0)
    Y_prime = Y_f - y_mean[np.newaxis, :]
    
    # 观测误差协方差 R
    R = np.diag(obs_errors**2)
    
    # 计算 Kalman 增益
    # K = P^f Hᵀ (H P^f Hᵀ + R)^{-1}
    # 在集合形式中：
    # P^f Hᵀ ≈ X' * Y'ᵀ / (N-1)
    # H P^f Hᵀ ≈ Y' * Y'ᵀ / (N-1)
    
    P_HT = np.dot(X_prime.T, Y_prime) / (n_ens - 1.0)
    HPH_T = np.dot(Y_prime.T, Y_prime) / (n_ens - 1.0)
    
    # 局地化（对 HPH_T 进行 Schur 乘积）
    # 简化为对角尺度化
    for i in range(n_obs):
        for j in range(n_obs):
            if i != j:
                dist = 0.0  # 简化：假设观测在同一点
                loc = gaspari_cohn_localization(dist, localization_length)
                HPH_T[i, j] *= loc
    
    innov_cov = HPH_T + R
    
    try:
        inv_innov = np.linalg.inv(innov_cov)
    except np.linalg.LinAlgError:
        inv_innov = np.linalg.pinv(innov_cov)
    
    K = np.dot(P_HT, inv_innov)
    
    # 分析更新
    analysis_states = np.zeros_like(ensemble_states)
    for i in range(n_ens):
        innovation = observations - Y_f[i, :]
        analysis_states[i, :] = ensemble_states[i, :] + np.dot(K, innovation)
    
    return analysis_states


def generate_synthetic_observations(true_state, obs_types=None):
    """
    生成模拟观测数据用于测试同化系统。
    
    参数:
        true_state: [x, y, p_min, r_max] 真实状态
        obs_types: 观测类型列表
    
    返回:
        aggregator: ObservationAggregator 对象
    """
    if obs_types is None:
        obs_types = ['satellite_wind', 'dropsonde', 'radar', 'buoy']
    
    aggregator = ObservationAggregator()
    
    # 为每种类型生成若干观测
    for obs_type in obs_types:
        n_obs = np.random.randint(3, 8)
        for _ in range(n_obs):
            # 观测值 = 真值 + 噪声
            if obs_type == 'satellite_wind':
                value = true_state[3] + np.random.normal(0, 10.0)
                error = 15.0
            elif obs_type == 'dropsonde':
                value = true_state[2] + np.random.normal(0, 3.0)
                error = 5.0
            elif obs_type == 'radar':
                value = true_state[2] + np.random.normal(0, 5.0)
                error = 8.0
            else:  # buoy
                value = true_state[2] + np.random.normal(0, 8.0)
                error = 12.0
            
            obs = Observation(
                obs_type=obs_type,
                lon=true_state[0] + np.random.normal(0, 0.5),
                lat=true_state[1] + np.random.normal(0, 0.5),
                value=value,
                timestamp=0.0,
                error_var=error**2
            )
            aggregator.add_observation(obs)
    
    return aggregator
