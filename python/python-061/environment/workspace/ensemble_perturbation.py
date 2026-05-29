"""
集合预报初始扰动生成与概率统计分析
======================================
基于种子项目:
  - 1124_sphere_monte_carlo: 球面上均匀随机采样
  - 189_clock_solitaire_simulation: 随机过程模拟思想
  - 118_brc_naive: 大数据分组聚合统计

核心科学问题：
    台风预报具有高度的不确定性。集合预报通过生成一组有物理意义的
    初始扰动样本，来量化预报不确定性。本模块实现：
    
    1. 球面上正交随机扰动的蒙特卡洛生成（Monte Carlo on S²）
    2. 集合成员的分类聚合统计（类似BRC分组）
    3. 概率密度估计与置信区间计算

数学模型：

=== 1. 球面随机扰动采样 ===

在三维空间中生成标准正态随机向量 ξ ~ N(0, I₃)，然后归一化到单位球面：
    η = ξ / ||ξ||₂

该采样在球面 S² 上是均匀分布的（Muller, 1959; Marsaglia, 1972）。

对台风初始条件的扰动采用 Breeding of Growing Modes (BGM) 方法：
    δX_i = ε * (η_i ∘ σ_b)  for i = 1, ..., N_ens

其中 ∘ 表示逐元素乘积，σ_b 为背景误差标准差向量。

=== 2. 集合统计量 ===

集合均值：
    X̄ = (1/N) Σ_{i=1}^{N} X_i

集合协方差：
    P = (1/(N-1)) Σ_{i=1}^{N} (X_i - X̄)(X_i - X̄)ᵀ

扰动总能量范数：
    E_i = √(δX_iᵀ W δX_i)

其中 W 为能量范数权重矩阵，反映各变量的物理重要性。
"""

import numpy as np


def sphere01_sample(n):
    """
    在单位球面 S² 上均匀采样 n 个点。
    基于种子项目 1124_sphere_monte_carlo 的核心算法：
    
        1. 生成 ξ ~ N(0, I₃)
        2. 归一化 η = ξ / ||ξ||
    
    参数:
        n: 采样点数
    
    返回:
        x: (3, n) 数组，每列为单位球面上一点
    """
    x = np.random.randn(3, n)
    norms = np.sqrt(np.sum(x**2, axis=0))
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms[np.newaxis, :]
    return x


def sphere01_monomial_integral(e):
    """
    计算单位球面上单项式的精确积分。
    
    对于积分 ∫_{S²} x^e[0] * y^e[1] * z^e[2] dS：
    
    若任一 e[i] 为奇数，则积分值为 0。
    若全为偶数，则：
        I = 4π * Π_{i=0}^{2} (e[i] - 1)!! / (Σ e[i] + 1)!!
    
    其中 !! 表示双阶乘。
    
    参数:
        e: 长度为3的整数数组，表示各方向指数
    
    返回:
        integral: 积分值
    """
    if np.any(e < 0):
        return 0.0
    if np.any(e % 2 == 1):
        return 0.0
    
    def double_factorial(k):
        """计算双阶乘 k!!"""
        if k < 0:
            return 1.0
        result = 1.0
        for i in range(k, 0, -2):
            result *= i
        return result
    
    total_sum = np.sum(e)
    numerator = 1.0
    for i in range(3):
        numerator *= double_factorial(e[i] - 1)
    denominator = double_factorial(total_sum + 1)
    
    return 4.0 * np.pi * numerator / denominator


def generate_ensemble_perturbations(n_ens=20, state_dim=4, amplitude=2.0):
    """
    生成集合预报的初始扰动。
    
    参数:
        n_ens: 集合成员数
        state_dim: 状态维度（x, y, p_min, r_max）
        amplitude: 扰动振幅
    
    返回:
        perturbations: (n_ens, state_dim) 扰动矩阵
    """
    # 背景误差标准差（基于台风观测误差经验值）
    sigma_b = np.array([0.5, 0.3, 5.0, 10.0])  # deg, deg, hPa, km
    
    # 球面上采样方向（使用多维球面推广）
    # 这里使用高斯随机向量 + Gram-Schmidt 正交化（类似 BGM 方法）
    raw_pert = np.random.randn(n_ens, state_dim)
    
    # Gram-Schmidt 正交化，使扰动在状态空间中正交
    perturbations = np.zeros((n_ens, state_dim))
    for i in range(n_ens):
        v = raw_pert[i, :].copy()
        for j in range(i):
            proj = np.dot(perturbations[j, :], v) / (np.dot(perturbations[j, :], perturbations[j, :]) + 1e-12)
            v = v - proj * perturbations[j, :]
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            perturbations[i, :] = amplitude * (v / norm_v) * sigma_b
        else:
            perturbations[i, :] = 0.0
    
    return perturbations


class EnsembleStatistics:
    """
    集合预报统计量计算类（基于 118_brc_naive 的分组聚合思想）。
    """
    def __init__(self, ensemble_states):
        """
        参数:
            ensemble_states: (n_ens, n_time, state_dim) 或 (n_ens, state_dim)
        """
        self.states = np.array(ensemble_states)
        self.n_ens = self.states.shape[0]
    
    def ensemble_mean(self):
        """计算集合均值。"""
        return np.mean(self.states, axis=0)
    
    def ensemble_spread(self):
        """
        计算集合离散度（标准差）。
        
        spread = √( (1/(N-1)) Σ (X_i - X̄)² )
        """
        mean = self.ensemble_mean()
        if self.states.ndim == 3:
            # (n_ens, n_time, state_dim)
            diff = self.states - mean[np.newaxis, :, :]
        else:
            diff = self.states - mean[np.newaxis, :]
        return np.sqrt(np.mean(diff**2, axis=0))
    
    def probability_in_interval(self, variable_idx, lower, upper):
        """
        计算指定变量落在区间 [lower, upper] 内的概率。
        
        基于 189_clock_solitaire 的蒙特卡洛概率估计思想：
            P = (满足条件的样本数) / (总样本数)
        """
        if self.states.ndim == 3:
            vals = self.states[:, :, variable_idx]
        else:
            vals = self.states[:, variable_idx]
        
        count = np.sum((vals >= lower) & (vals <= upper), axis=0)
        return count / self.n_ens
    
    def confidence_interval(self, variable_idx, level=0.95):
        """
        计算指定变量的置信区间（百分位法）。
        
        参数:
            variable_idx: 变量索引
            level: 置信水平
        
        返回:
            lower, upper: 置信区间上下界
        """
        alpha = (1.0 - level) / 2.0
        if self.states.ndim == 3:
            vals = self.states[:, :, variable_idx]
        else:
            vals = self.states[:, variable_idx]
        
        lower = np.percentile(vals, alpha * 100.0, axis=0)
        upper = np.percentile(vals, (1.0 - alpha) * 100.0, axis=0)
        return lower, upper
    
    def group_by_intensity(self, pmin_idx=2, thresholds=(980, 960, 940)):
        """
        按台风强度分组统计（基于 118_brc_naive 的分组聚合思想）。
        
        分组标准（按中心气压）：
            TD:  P_min > 980 hPa    (热带低压)
            TS:  960 < P_min ≤ 980  (热带风暴)
            TY:  940 < P_min ≤ 960  (台风)
            STY: P_min ≤ 940        (强台风/超强台风)
        
        返回:
            groups: dict，键为类别名，值为该组的成员索引列表
        """
        if self.states.ndim == 3:
            # 取最后时刻的强度
            pmin = self.states[:, -1, pmin_idx]
        else:
            pmin = self.states[:, pmin_idx]
        
        groups = {
            'TD': [],
            'TS': [],
            'TY': [],
            'STY': []
        }
        
        for i in range(self.n_ens):
            p = pmin[i]
            if p > thresholds[0]:
                groups['TD'].append(i)
            elif p > thresholds[1]:
                groups['TS'].append(i)
            elif p > thresholds[2]:
                groups['TY'].append(i)
            else:
                groups['STY'].append(i)
        
        return groups
    
    def summarize_groups(self, groups):
        """
        汇总各组的统计信息。
        
        返回:
            summary: dict，包含每组的数量和占比
        """
        summary = {}
        total = self.n_ens
        for name, members in groups.items():
            count = len(members)
            summary[name] = {
                'count': count,
                'percentage': count / total * 100.0 if total > 0 else 0.0
            }
        return summary


def run_ensemble_forecast(vortex_solver_class, n_ens=20, t_span=(0.0, 72.0), n_steps=720):
    """
    运行集合预报实验。
    
    参数:
        vortex_solver_class: 台风涡旋求解器类
        n_ens: 集合成员数
        t_span: 预报时长（小时）
        n_steps: 时间步数
    
    返回:
        ensemble_states: (n_ens, n_steps+1, state_dim) 所有成员的状态历史
        stats: EnsembleStatistics 对象
    
    TODO HOLE 2: 实现集合预报扰动叠加与运行
    需要:
    1. 生成初始扰动 (generate_ensemble_perturbations)
    2. 对每个集合成员，将扰动叠加到初始条件 (x0, y0, p_min_initial, r_max_initial)
       注意: 扰动索引 [0,1,2,3] 对应 [x0, y0, p_min_initial, r_max_initial]
       必须与 typhoon_vortex_ode.py 中状态向量 [x, y, p_min, r_max] 的顺序一致
    3. 运行涡旋ODE求解器 (solver.solve)
    4. 收集所有成员状态，构建 EnsembleStatistics
    """
    from typhoon_vortex_ode import TyphoonVortexODE, TyphoonVortexParameters
    
    # HOLE 2 BEGIN: 请补全集合预报实现
    # TODO: 生成扰动并叠加到初始条件，运行每个集合成员
    # 注意: perturbations[i, :] 的索引必须与状态向量维度一致
    ensemble_states = np.zeros((n_ens, n_steps + 1, 4))
    stats = EnsembleStatistics(ensemble_states)
    t_arr = np.linspace(t_span[0], t_span[1], n_steps + 1)
    # HOLE 2 END
    
    return ensemble_states, stats, t_arr
