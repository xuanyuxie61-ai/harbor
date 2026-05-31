"""
monte_carlo_sampler.py
贝叶斯参数采样与不确定性量化模块。

融合原始项目：1081_simplex_monte_carlo（单纯形上的 Monte Carlo 采样与积分）
             870_pink_noise（1/f 噪声生成，用于增强参数空间探索）

在系外行星大气光谱反演中，贝叶斯框架下需要：
- 在参数空间的后验分布中采样
- 估计参数不确定性和相关性
- 评估模型证据
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


class SimplexSampler:
    """
    单纯形上的 Monte Carlo 采样器。

    融合 simplex_monte_carlo 的核心思想：
    在 M 维标准单纯形
        Δ_M = { x ∈ R^M | x_i ≥ 0, Σ x_i = 1 }
    上生成均匀或加权随机样本。

    生成方法（Dirichlet 分布）:
        若 y_i ~ Exp(1) i.i.d.，则
        x_i = y_i / Σ_j y_j
        服从 Δ_M 上的均匀分布。
    """

    @staticmethod
    def sample_unit_simplex(m: int, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        在 M 维标准单纯形上生成 N 个均匀随机样本。

        参数:
            m: 单纯形维度
            n: 样本数
            seed: 随机种子

        返回:
            样本数组，形状 (n, m)
        """
        if m <= 0 or n <= 0:
            raise ValueError("维度和样本数必须为正")
        if seed is not None:
            np.random.seed(seed)

        # 生成指数分布随机数并归一化
        y = np.random.exponential(scale=1.0, size=(n, m))
        s = y.sum(axis=1, keepdims=True)
        s = np.maximum(s, 1e-30)
        return y / s

    @staticmethod
    def sample_general_simplex(vertices: np.ndarray, n: int,
                                seed: Optional[int] = None) -> np.ndarray:
        """
        在一般单纯形上采样。

        公式:
            x = V^T · u
        其中 V 是顶点矩阵 (m × (m+1))，u 是标准单纯形上的重心坐标。

        参数:
            vertices: 单纯形顶点，形状 (m, m+1)
            n: 样本数

        返回:
            样本数组，形状 (n, m)
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        m, n_vert = vertices.shape
        if n_vert != m + 1:
            raise ValueError(f"M维单纯形应有 M+1 个顶点，得到 {n_vert} 个顶点")

        u = SimplexSampler.sample_unit_simplex(m + 1, n, seed)
        return u @ vertices.T

    @staticmethod
    def simplex_volume(vertices: np.ndarray) -> float:
        """
        计算 M 维单纯形的体积。

        公式:
            V = |det(v_1 - v_0, v_2 - v_0, ..., v_m - v_0)| / m!
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        m = vertices.shape[0]
        M = vertices[:, 1:] - vertices[:, 0:1]
        return abs(np.linalg.det(M)) / np.math.factorial(m)


class PinkNoiseGenerator:
    """
    1/f 噪声（粉红噪声）生成器。

    融合 pink_noise 的核心算法：
    使用多尺度随机游走叠加生成具有幂律谱 S(f) ∝ 1/f^β 的噪声序列。

    在参数采样中，粉红噪声比白噪声具有更好的长程相关性，
    有助于在参数空间中进行更有效的探索。
    """

    def __init__(self, beta: float = 1.0):
        """
        参数:
            beta: 幂律指数，β=1 对应标准粉红噪声
        """
        if beta < 0 or beta > 2:
            raise ValueError("β 应在 [0, 2] 范围内")
        self.beta = beta

    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        通过频域滤波生成 1/f^β 噪声。

        算法:
            1. 生成白噪声序列
            2. 做 FFT 得到频谱
            3. 乘以 1/f^{β/2} 滤波器
            4. 做逆 FFT

        参数:
            n: 序列长度
            seed: 随机种子

        返回:
            噪声序列，零均值，单位方差
        """
        if seed is not None:
            np.random.seed(seed)

        white = np.random.normal(0.0, 1.0, size=n)
        spectrum = np.fft.rfft(white)
        freq = np.fft.rfftfreq(n)

        # 避免 f=0 处的奇点
        freq[0] = freq[1] if len(freq) > 1 else 1e-10

        # 频域滤波
        filter_resp = freq**(-self.beta / 2.0)
        filter_resp[0] = 0.0  # 去除直流分量

        spectrum *= filter_resp
        pink = np.fft.irfft(spectrum, n=n)

        # 归一化
        std = np.std(pink)
        if std > 1e-15:
            pink = pink / std
        return pink


class MetropolisHastingsSampler:
    """
    Metropolis-Hastings MCMC 采样器。

    用于从后验分布 P(θ|D) ∝ P(D|θ) P(θ) 中采样参数 θ。

    算法:
        1. 初始化 θ_0
        2. 对于 t = 1, ..., N:
            a. 从提议分布 q(θ'|θ_{t-1}) 采样候选 θ'
            b. 计算接受率 α = min(1, P(θ'|D) / P(θ_{t-1}|D) * q(θ_{t-1}|θ') / q(θ'|θ_{t-1}))
            c. 以概率 α 接受 θ'，否则保持 θ_{t-1}
    """

    def __init__(self, log_posterior: Callable[[np.ndarray], float],
                 proposal_cov: np.ndarray,
                 bounds: Optional[List[Tuple[float, float]]] = None):
        """
        参数:
            log_posterior: 对数后验密度函数 log P(θ|D)
            proposal_cov: 提议分布的协方差矩阵 (高斯随机游走)
            bounds: 各参数的边界约束 [(min, max), ...]
        """
        self.log_posterior = log_posterior
        self.proposal_cov = np.asarray(proposal_cov, dtype=np.float64)
        self.bounds = bounds
        self.dim = self.proposal_cov.shape[0]

        if self.proposal_cov.shape != (self.dim, self.dim):
            raise ValueError("提议协方差矩阵必须为方阵")

        # 预计算 Cholesky 分解
        try:
            self.L = np.linalg.cholesky(self.proposal_cov + 1e-12 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            # 若不正定，使用对角近似
            self.L = np.diag(np.sqrt(np.maximum(np.diag(self.proposal_cov), 1e-12)))

    def _propose(self, current: np.ndarray) -> np.ndarray:
        """从高斯随机游走提议分布采样。"""
        noise = self.L @ np.random.normal(0.0, 1.0, size=self.dim)
        candidate = current + noise

        if self.bounds is not None:
            for i, (low, high) in enumerate(self.bounds):
                candidate[i] = np.clip(candidate[i], low, high)
        return candidate

    def sample(self, x0: np.ndarray, n_samples: int,
                burn_in: int = 1000, thin: int = 10,
                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行 MCMC 采样。

        参数:
            x0: 初始参数向量
            n_samples: 最终保留的样本数
            burn_in: 预烧期步数
            thin: 稀释间隔（每 thin 步保留一个样本）
            seed: 随机种子

        返回:
            samples: 采样结果，形状 (n_samples, dim)
            log_probs: 对应的对数后验值
        """
        if seed is not None:
            np.random.seed(seed)

        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.shape[0] != self.dim:
            raise ValueError(f"初始参数维度 {x0.shape[0]} 与提议分布维度 {self.dim} 不匹配")

        total_steps = burn_in + n_samples * thin
        current = x0.copy()
        current_logp = self.log_posterior(current)

        samples = np.zeros((n_samples, self.dim), dtype=np.float64)
        log_probs = np.zeros(n_samples, dtype=np.float64)

        accepted = 0
        sample_idx = 0

        for step in range(total_steps):
            candidate = self._propose(current)
            cand_logp = self.log_posterior(candidate)

            # 计算接受率
            log_alpha = cand_logp - current_logp
            # 边界反射的修正（对于对称提议，q 比值为 1）
            alpha = np.exp(min(log_alpha, 0.0))

            if np.random.uniform() < alpha:
                current = candidate
                current_logp = cand_logp
                accepted += 1

            # 保存样本
            if step >= burn_in and (step - burn_in) % thin == 0:
                samples[sample_idx] = current
                log_probs[sample_idx] = current_logp
                sample_idx += 1

        acceptance_rate = accepted / total_steps
        return samples, log_probs, acceptance_rate


class NestedSampler:
    """
    嵌套采样（Nested Sampling）算法。

    用于贝叶斯模型证据（边缘似然）计算：
        Z = ∫ L(θ) π(θ) dθ

    算法（Skilling, 2006）:
        1. 从先验分布中随机抽取 N 个活点（live points）
        2. 每次迭代：
            a. 找到活点中似然最小的点，记录其似然 L_min
            b. 估计该点对应的先验体积 X_i ≈ exp(-i/N)
            c. 贡献 ΔZ_i = L_min * ΔX_i
            d. 用 MCMC 或拒绝采样从 L(θ) > L_min 的区域生成新点替换之
        3. 当 L_max * X_i < tol * Z 时终止
    """

    def __init__(self, log_likelihood: Callable[[np.ndarray], float],
                 prior_transform: Callable[[np.ndarray], np.ndarray],
                 n_live: int = 100):
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.n_live = n_live

    def run(self, dim: int, max_iter: int = 10000,
            log_z_tol: float = 0.1, seed: Optional[int] = None) -> dict:
        """
        执行嵌套采样。

        参数:
            dim: 参数维度
            max_iter: 最大迭代次数
            log_z_tol: 终止容差（log Z 的相对变化）
            seed: 随机种子

        返回:
            结果字典，包含 logZ、采样点和权重
        """
        if seed is not None:
            np.random.seed(seed)

        # 初始化活点
        live_u = np.random.uniform(0.0, 1.0, size=(self.n_live, dim))
        live_v = np.array([self.prior_transform(u) for u in live_u])
        live_logl = np.array([self.log_likelihood(v) for v in live_v])

        logZ = -1e300
        samples = []
        logls = []
        logws = []

        logX = 0.0  # log 先验体积

        for it in range(max_iter):
            # 找到最小似然点
            min_idx = np.argmin(live_logl)
            min_logl = live_logl[min_idx]

            # 体积估计: log(ΔX) ≈ -it/N - log(N)
            log_dX = -it / self.n_live - np.log(self.n_live)
            log_dZ = min_logl + log_dX

            # 更新证据
            logZ = np.logaddexp(logZ, log_dZ)

            samples.append(live_v[min_idx].copy())
            logls.append(min_logl)
            logws.append(log_dZ)

            # 从剩余活点中随机选一点作为 MCMC 起点
            other_idx = np.random.choice([i for i in range(self.n_live) if i != min_idx])
            proposal = live_v[other_idx].copy()

            # 简单拒绝采样生成新点（在L>L_min的区域）
            max_attempts = 1000
            new_point = None
            for _ in range(max_attempts):
                # 围绕 proposal 随机扰动
                scale = 0.1
                trial_u = np.clip(np.random.normal(0.5, scale, size=dim), 0.0, 1.0)
                trial_v = self.prior_transform(trial_u)
                trial_logl = self.log_likelihood(trial_v)
                if trial_logl > min_logl:
                    new_point = trial_v
                    new_logl = trial_logl
                    break

            if new_point is None:
                # 若无法找到，使用存活点中似然最大的
                new_point = live_v[np.argmax(live_logl)].copy()
                new_logl = np.max(live_logl)

            live_v[min_idx] = new_point
            live_logl[min_idx] = new_logl

            # 终止条件
            logLmax = np.max(live_logl)
            log_remainder = logLmax + logX - it / self.n_live
            if log_remainder < np.log(log_z_tol) + logZ:
                break

        return {
            'logZ': float(logZ),
            'samples': np.array(samples),
            'log_likelihoods': np.array(logls),
            'log_weights': np.array(logws)
        }
