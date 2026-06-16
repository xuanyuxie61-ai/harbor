"""
蒙特卡罗采样与动力学蒙特卡罗模块
===============================
实现中微子混合参数的超球面采样，
以及物质密度涨落的动力学蒙特卡罗（KMC）随机传播。

核心算法：
1. 正超球面上的均匀采样：
   x = |N(0,1)| / ||N(0,1)||  →  x ∈ Sⁿ⁻¹₊

2. 点对距离统计：
   d(p,q) = ||p-q||,  p,q ∈ Sⁿ⁻¹₊
   理论均值: E[d] = √2 × Γ(n/2) / Γ((n-1)/2) × 某常数

3. Arrhenius温度外推：
   log(rate) = a + b/T
   从高温MD数据外推到物理温度

4. Gillespie KMC算法：
   总倾向性: a_tot = Σ_i h_i × k_i
   时间步长: Δt = -ln(r) / a_tot
   反应选择: 累积概率采样

5. 置换正交性：
   对任意置换P₁,P₂, 向量 Q = P₁·v - P₂·v
   旋转后 X = R(45°)Q 的各分量正交

数据来源：
- 567_hypersphere_positive_distance: 超球面采样与距离统计
- 1066_temperature-extrapolation-KMC: KMC模拟与Arrhenius外推
- 922_puzzles: 置换正交性、蒙特卡罗悖论
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class HypersphereSampler:
    """
    正超球面采样器。

    在Sⁿ⁻¹₊ = {x ∈ ℝⁿ : ||x||=1, x_i ≥ 0}上均匀采样。

    方法：Muller (1959)
    1. 从n维标准正态分布采样: z ~ N(0, I)
    2. 取绝对值: x = |z|
    3. 归一化: x = x / ||x||

    物理应用：
    - 中微子混合矩阵的参数化
    - 幺正矩阵空间的采样
    """

    def __init__(self, dimension: int = 3):
        """
        初始化采样器。

        参数：
            dimension: 空间维度
        """
        self.dimension = dimension

    def sample_uniform(self, n_samples: int, seed: int = None) -> np.ndarray:
        """
        在正超球面上均匀采样。

        参数：
            n_samples: 样本数
            seed: 随机种子

        返回：
            samples: 样本点 (n_samples, dimension)
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # 标准正态采样
        z = rng.randn(n_samples, self.dimension)

        # 取绝对值（限制在正象限）
        x = np.abs(z)

        # 归一化
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)  # 防止除零
        samples = x / norms

        return samples

    def pairwise_distances(self, samples: np.ndarray) -> np.ndarray:
        """
        计算样本点对之间的欧氏距离。

        d(p,q) = ||p-q|| = √(2 - 2p·q)  (因为||p||=||q||=1)

        参数：
            samples: 样本点 (N, D)

        返回：
            distances: 距离数组 (N*(N-1)/2,)
        """
        N = len(samples)
        distances = []

        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(samples[i] - samples[j])
                distances.append(d)

        return np.array(distances)

    def distance_statistics(self, n_samples: int = 1000,
                            seed: int = None) -> dict:
        """
        计算距离的统计量。

        理论结果（n维正超球面）：
        E[d²] = 2(1 - 1/n)  (对于全超球面)
        对于正超球面，需要修正因子。

        参数：
            n_samples: 样本数
            seed: 随机种子

        返回：
            stats: 统计量字典
        """
        samples = self.sample_uniform(n_samples, seed)
        distances = self.pairwise_distances(samples)

        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'mean_squared_distance': np.mean(distances ** 2),
            'n_samples': n_samples,
            'n_pairs': len(distances),
            'dimension': self.dimension,
        }


class ArrheniusExtrapolator:
    """
    Arrhenius温度外推器。

    从高温反应速率外推到物理温度。

    Arrhenius方程：
    k(T) = A × exp(-E_a / (k_B T))

    对数形式：
    log₁₀(k) = log₁₀(A) - E_a / (k_B T × ln(10))
    y = a + b × (1/T)

    使用OLS拟合a, b，然后外推到目标温度。
    """

    def __init__(self):
        """初始化外推器"""
        self.a = None  # 截距
        self.b = None  # 斜率
        self.fitted = False

    def fit(self, temperatures: np.ndarray, rates: np.ndarray):
        """
        拟合Arrhenius参数。

        模型: log₁₀(k) = a + b × (1/T)

        参数：
            temperatures: 温度数组 (K)
            rates: 反应速率数组
        """
        # 过滤无效数据
        mask = (temperatures > 0) & (rates > 0)
        T_valid = temperatures[mask]
        k_valid = rates[mask]

        if len(T_valid) < 2:
            raise ValueError("至少需要2个有效数据点")

        # 对数变换
        inv_T = 1.0 / T_valid
        log_k = np.log10(k_valid)

        # OLS拟合: y = a + b*x
        # b = Cov(x,y) / Var(x)
        # a = mean(y) - b*mean(x)
        n = len(inv_T)
        mean_x = np.mean(inv_T)
        mean_y = np.mean(log_k)

        b = np.sum((inv_T - mean_x) * (log_k - mean_y)) / np.sum((inv_T - mean_x)**2)
        a = mean_y - b * mean_x

        self.a = a
        self.b = b
        self.fitted = True

    def predict(self, temperature: float) -> float:
        """
        预测给定温度下的反应速率。

        参数：
            temperature: 温度 (K)

        返回：
            rate: 反应速率
        """
        if not self.fitted:
            raise RuntimeError("模型未拟合")

        log_k = self.a + self.b / temperature
        return 10.0 ** log_k

    def extrapolate_with_bootstrap(self, temperatures: np.ndarray,
                                    rates: np.ndarray, target_T: float,
                                    n_bootstrap: int = 1000,
                                    seed: int = None) -> dict:
        """
        带Bootstrap误差估计的温度外推。

        方法：
        1. 对残差进行Bootstrap重采样
        2. 对每个Bootstrap样本拟合Arrhenius
        3. 外推到目标温度
        4. 统计预测分布

        参数：
            temperatures: 温度数组
            rates: 速率数组
            target_T: 目标温度
            n_bootstrap: Bootstrap次数
            seed: 随机种子

        返回：
            result: 外推结果和置信区间
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # 原始拟合
        self.fit(temperatures, rates)
        rate_original = self.predict(target_T)

        # 计算残差
        log_k_fit = self.a + self.b / temperatures
        log_k_obs = np.log10(np.maximum(rates, 1e-30))
        residuals = log_k_obs - log_k_fit

        # Bootstrap
        rates_bootstrap = []

        for _ in range(n_bootstrap):
            # 重采样残差
            resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
            log_k_boot = log_k_fit + resampled_residuals
            k_boot = 10.0 ** log_k_boot

            # 重新拟合
            try:
                self.fit(temperatures, k_boot)
                rate_boot = self.predict(target_T)
                rates_bootstrap.append(rate_boot)
            except ValueError:
                continue

        rates_bootstrap = np.array(rates_bootstrap)

        return {
            'rate_mean': np.mean(rates_bootstrap),
            'rate_std': np.std(rates_bootstrap),
            'rate_median': np.median(rates_bootstrap),
            'ci_95_lower': np.percentile(rates_bootstrap, 2.5),
            'ci_95_upper': np.percentile(rates_bootstrap, 97.5),
            'rate_original': rate_original,
            'n_valid_bootstrap': len(rates_bootstrap),
        }


class KineticMonteCarlo:
    """
    动力学蒙特卡罗（KMC）模拟器。

    用于模拟中微子在随机涨落物质密度中的传播。

    Gillespie算法：
    1. 计算所有反应的倾向性 a_i = h_i × k_i
    2. 总倾向性 a_tot = Σ a_i
    3. 时间步长 Δt = -ln(r₁) / a_tot
    4. 选择反应: 累积概率法
    5. 更新系统状态
    6. 重复

    物理应用：
    - 中微子在湍流物质中的传播
    - 随机密度涨落对振荡的影响
    """

    def __init__(self, n_states: int = 3):
        """
        初始化KMC模拟器。

        参数：
            n_states: 状态数（中微子味数）
        """
        self.n_states = n_states
        self.state = None
        self.time = 0.0
        self.history = []

    def initialize(self, initial_state: np.ndarray):
        """
        初始化系统状态。

        参数：
            initial_state: 初始状态向量 (n_states,)
        """
        self.state = initial_state.copy()
        self.time = 0.0
        self.history = [{'time': 0.0, 'state': self.state.copy()}]

    def compute_propensities(self, rates: np.ndarray, multiplicities: np.ndarray = None) -> np.ndarray:
        """
        计算反应倾向性。

        a_i = h_i × k_i

        参数：
            rates: 反应速率数组
            multiplicities: 反应简并度

        返回：
            propensities: 倾向性数组
        """
        if multiplicities is None:
            multiplicities = np.ones_like(rates)

        return multiplicities * rates

    def select_reaction(self, propensities: np.ndarray, rng: np.random.RandomState) -> int:
        """
        按累积概率选择反应。

        P(reaction i) = a_i / a_tot

        参数：
            propensities: 倾向性数组
            rng: 随机数生成器

        返回：
            reaction_idx: 选择的反应索引
        """
        a_tot = np.sum(propensities)
        if a_tot <= 0:
            return -1

        # 累积概率
        cum_prob = np.cumsum(propensities) / a_tot

        # 均匀随机数
        r = rng.random()

        # 选择
        for i, cp in enumerate(cum_prob):
            if r <= cp:
                return i

        return len(propensities) - 1

    def step(self, rates: np.ndarray, transition_matrix: np.ndarray,
             multiplicities: np.ndarray = None, rng: np.random.RandomState = None) -> dict:
        """
        执行一步KMC。

        参数：
            rates: 反应速率
            transition_matrix: 状态转移矩阵 (n_reactions, n_states, n_states)
            multiplicities: 简并度
            rng: 随机数生成器

        返回：
            step_info: 步进信息
        """
        if rng is None:
            rng = np.random.RandomState()

        propensities = self.compute_propensities(rates, multiplicities)
        a_tot = np.sum(propensities)

        if a_tot <= 0:
            return {'dt': 0, 'reaction': -1, 'new_state': self.state.copy()}

        # 时间步长
        r1 = rng.random()
        dt = -np.log(max(r1, 1e-30)) / a_tot

        # 选择反应
        reaction = self.select_reaction(propensities, rng)

        # 更新状态
        if 0 <= reaction < len(transition_matrix):
            new_state = transition_matrix[reaction] @ self.state
            # 归一化
            norm = np.linalg.norm(new_state)
            if norm > 1e-15:
                new_state /= norm
            self.state = new_state

        self.time += dt
        self.history.append({'time': self.time, 'state': self.state.copy()})

        return {
            'dt': dt,
            'reaction': reaction,
            'new_state': self.state.copy(),
            'total_time': self.time,
        }

    def run(self, rates: np.ndarray, transition_matrix: np.ndarray,
            n_steps: int, multiplicities: np.ndarray = None,
            seed: int = None) -> dict:
        """
        运行KMC模拟。

        参数：
            rates: 反应速率
            transition_matrix: 转移矩阵
            n_steps: 步数
            multiplicities: 简并度
            seed: 随机种子

        返回：
            result: 模拟结果
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        for _ in range(n_steps):
            self.step(rates, transition_matrix, multiplicities, rng)

        # 提取历史
        times = [h['time'] for h in self.history]
        states = np.array([h['state'] for h in self.history])

        return {
            'times': np.array(times),
            'states': states,
            'final_state': self.state.copy(),
            'total_time': self.time,
            'n_steps': len(self.history) - 1,
        }


class PermutationOrthogonality:
    """
    置换正交性演示模块。

    定理：对于任意两个置换P₁, P₂ ∈ S_n，
    向量 x = P₁·v - P₂·v 经过45°旋转后
    的分量在ℝⁿ中相互正交。

    旋转矩阵：
    R(45°) = [[cos45, -sin45], [sin45, cos45]]
    = (1/√2) [[1, -1], [1, 1]]
    """

    @staticmethod
    def demonstrate(n: int = 10, seed: int = None) -> dict:
        """
        演示置换正交性。

        参数：
            n: 置换维度
            seed: 随机种子

        返回：
            result: 演示结果
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # 基础向量
        v = np.arange(1, n + 1, dtype=float)

        # 随机置换
        P1 = rng.permutation(n)
        P2 = rng.permutation(n)

        # 置换后的向量
        x_perm = v[P1]
        y_perm = v[P2]

        # 中心化
        mean_val = (n + 1) / 2.0
        x_centered = x_perm - mean_val
        y_centered = y_perm - mean_val

        # 45°旋转
        cos45 = np.cos(np.pi / 4)
        sin45 = np.sin(np.pi / 4)
        X_rot = cos45 * x_centered - sin45 * y_centered
        Y_rot = sin45 * x_centered + cos45 * y_centered

        # 检查正交性
        dot_product = np.dot(X_rot, Y_rot)

        return {
            'n': n,
            'P1': P1,
            'P2': P2,
            'dot_product': dot_product,
            'is_orthogonal': abs(dot_product) < 1e-10,
            'X_rot': X_rot,
            'Y_rot': Y_rot,
        }


class CasinoParadox:
    """
    赌场悖论模拟。

     multipliciative 随机过程：
    S_{n+1} = S_n × X_n
    其中 X_n = 1.2 (概率0.5) 或 0.83 (概率0.5)

    E[X] = 0.5×1.2 + 0.5×0.83 = 1.015 > 1
    但几何平均: √(1.2×0.83) = √0.996 ≈ 0.998 < 1

    结论：期望增长 ≠ 典型增长
    """

    @staticmethod
    def simulate(n_flips: int = 100, n_trials: int = 1000,
                 initial_stake: float = 100.0,
                 up_factor: float = 1.2,
                 down_factor: float = 0.83,
                 seed: int = None) -> dict:
        """
        模拟赌场悖论。

        参数：
            n_flips: 每次试验的抛币次数
            n_trials: 试验次数
            initial_stake: 初始赌注
            up_factor: 正面倍数
            down_factor: 反面倍数
            seed: 随机种子

        返回：
            result: 模拟结果
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        final_stakes = np.zeros(n_trials)

        for trial in range(n_trials):
            stake = initial_stake
            for _ in range(n_flips):
                if rng.random() < 0.5:
                    stake *= up_factor
                else:
                    stake *= down_factor
            final_stakes[trial] = stake

        return {
            'mean_final': np.mean(final_stakes),
            'median_final': np.median(final_stakes),
            'std_final': np.std(final_stakes),
            'geometric_mean': np.exp(np.mean(np.log(final_stakes))),
            'expected_single_flip': 0.5 * up_factor + 0.5 * down_factor,
            'geometric_single_flip': np.sqrt(up_factor * down_factor),
            'n_trials': n_trials,
            'n_flips': n_flips,
        }
