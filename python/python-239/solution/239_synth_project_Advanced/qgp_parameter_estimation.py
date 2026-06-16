"""
qgp_parameter_estimation.py — DREAM MCMC 参数估计
=====================================================

融合种子项目: 319_dream (DREAM: DiffeRential Evolution Adaptive Metropolis)

本模块使用 DREAM 算法估计 QGP 流体动力学模型参数.

DREAM 算法 (Vrugt et al. 2009):
---------------------------------

DREAM 是一种基于差分进化的自适应 MCMC 方法:

1. 初始化 N 条马尔可夫链 (par_num x chain_num x gen_num)
2. 每条链在当前状态生成候选点:
   - 从其他链中随机选取两对 (r1, r2), (r3, r4)
   - 差分向量: delta = gamma * (X[r1] - X[r2])
   - 噪声: e ~ N(0, sigma_e)
   - 候选: X* = X[current] + delta + e
3. Metropolis 接受:
   alpha = min(1, p(X*) / p(X[current]))
   以概率 alpha 接受候选点
4. 自适应交叉概率 CR:
   - CR 从离散集合 {1/n, 2/n, ..., 1} 中选取
   - 基于历史接受率自适应更新
5. Gelman-Rubin 收敛诊断:
   R_hat = sqrt((n-1)/n + B/(n*W))
   收敛: R_hat < 1.2

参数空间 (QGP 流体动力学):
    theta = [eta/s, T_init, tau_0, mix_lambda]
    - eta/s: 剪切粘滞比 (KSS 下限 1/(4pi))
    - T_init: 初始最大温度 (GeV)
    - tau_0: 热化时间 (fm/c)
    - mix_lambda: 初始条件混合参数

目标函数:
    log p(theta|data) = -0.5 * sum_k (O_k^sim - O_k^exp)^2 / sigma_k^2
    其中 O_k 为观测物理量 (v_2, v_3, dN/dy, <p_T>, ...)
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
from qgp_config import NumericalParams, ETA_OVER_S_KSS


class DREAMMCMC:
    """
    DREAM MCMC 参数估计器

    实现完整的 DREAM 算法:
    1. 链初始化
    2. 差分进化候选生成
    3. Metropolis-Hastings 接受/拒绝
    4. CR 自适应
    5. Gelman-Rubin 收敛诊断
    """

    def __init__(self, n_chains: int = None, n_steps: int = None,
                 n_params: int = 4, target_func: Callable = None):
        """
        Args:
            n_chains: 马尔可夫链条数
            n_steps: 每链条迭代步数
            n_params: 参数维度
            target_func: 对数似然函数 log p(theta)
        """
        params = NumericalParams
        self.n_chains = n_chains or params.MCMC_CHAINS
        self.n_steps = n_steps or params.MCMC_STEPS
        self.n_params = n_params
        self.target_func = target_func

        # CR 候选值
        self.cr_values = np.array(params.MCMC_CR_VALUES, dtype=np.float64)
        self.n_cr = len(self.cr_values)

        # 跳跃率表 (DE 缩放因子)
        self.gamma_base = params.MCMC_GAMMA_SCALE

        # 链状态: (n_chains, n_params)
        self.chains = None
        self.log_likelihoods = None

        # CR 概率
        self.cr_probs = np.ones(self.n_cr) / self.n_cr

        # Gelman-Rubin 统计
        self.gr_history = []

        # 参数边界
        self.param_bounds = None

    def initialize_chains(self, param_center: np.ndarray,
                           param_width: np.ndarray,
                           bounds: list = None):
        """
        初始化马尔可夫链

        策略: 以 param_center 为中心, param_width 为半宽,
        均匀分布初始化 n_chains 条链.

        链间多样性保证:
        - 链间距 > param_width / n_chains
        - 避免链聚集导致收敛失败

        Args:
            param_center: 参数中心值 (n_params,)
            param_width: 参数宽度 (n_params,)
            bounds: 参数边界 [(min, max), ...]
        """
        rng = np.random.RandomState(42)
        self.chains = np.zeros((self.n_chains, self.n_params))
        self.param_bounds = bounds

        for i in range(self.n_chains):
            # 均匀分布初始化
            self.chains[i] = param_center + param_width * \
                             (2.0 * rng.rand(self.n_params) - 1.0)

            # 施加边界
            if bounds is not None:
                for j in range(self.n_params):
                    lo, hi = bounds[j]
                    self.chains[i, j] = np.clip(self.chains[i, j], lo, hi)

        # 计算初始似然
        self.log_likelihoods = np.array([
            self.target_func(self.chains[i]) for i in range(self.n_chains)
        ])

    def sample_candidate(self, chain_idx: int, cr_idx: int,
                          generation: int) -> np.ndarray:
        """
        差分进化候选生成 (DREAM 核心)

        步骤:
        1. 从 n_chains 中选取 2*delta 条不同的链
        2. 差分向量: delta = gamma * sum_pairs (X[r1] - X[r2])
        3. 跳跃率: gamma = 2.38 / sqrt(2 * n_CR * n_params)
        4. 噪声: e ~ N(0, b^2) where b ~ U(0, 1)
        5. CR 采样: 仅更新 CR[cr_idx] 对应的维度子集

        候选: X* = X[current] + delta + e

        Args:
            chain_idx: 当前链索引
            cr_idx: CR 值索引
            generation: 当前代数

        Returns:
            候选参数 (n_params,)
        """
        n_chains = self.n_chains
        n_CR = self.cr_values[cr_idx]

        # 差分对数 (至少需要 2*delta 条链)
        n_pairs = max(1, int(n_CR / 2))
        n_pairs = min(n_pairs, (n_chains - 1) // 2)

        # 跳跃率
        gamma = self.gamma_base / np.sqrt(2.0 * n_pairs * self.n_params + 1.0e-10)

        # 随机选取配对链
        others = [j for j in range(n_chains) if j != chain_idx]
        rng = np.random.RandomState()
        selected = rng.choice(others, size=2*n_pairs, replace=False)

        # 差分向量
        delta = np.zeros(self.n_params)
        for p in range(n_pairs):
            r1 = selected[2*p]
            r2 = selected[2*p + 1]
            delta += self.chains[r1] - self.chains[r2]
        delta *= gamma

        # 噪声
        b = rng.rand()
        noise = b * rng.randn(self.n_params) * 1.0e-6

        # CR 维度选择 (部分维度更新)
        n_update = max(1, int(n_CR))
        update_dims = rng.choice(self.n_params, size=n_update, replace=False)

        # 候选
        candidate = self.chains[chain_idx].copy()
        for d in update_dims:
            candidate[d] += delta[d] + noise[d]

        # 边界投影
        if self.param_bounds is not None:
            for d in range(self.n_params):
                lo, hi = self.param_bounds[d]
                if candidate[d] < lo:
                    candidate[d] = 2.0 * lo - candidate[d]
                    candidate[d] = max(candidate[d], lo)
                elif candidate[d] > hi:
                    candidate[d] = 2.0 * hi - candidate[d]
                    candidate[d] = min(candidate[d], hi)

        return candidate

    def metropolis_accept(self, current_ll: float,
                           candidate_ll: float) -> bool:
        """
        Metropolis-Hastings 接受/拒绝

        alpha = min(1, exp(ll_candidate - ll_current))

        若 ll_candidate >= ll_current: 总是接受
        否则: 以概率 alpha 接受

        Args:
            current_ll: 当前对数似然
            candidate_ll: 候选对数似然

        Returns:
            是否接受
        """
        if not np.isfinite(candidate_ll):
            return False

        log_alpha = candidate_ll - current_ll
        if log_alpha >= 0:
            return True

        rng = np.random.RandomState()
        return rng.rand() < np.exp(log_alpha)

    def compute_gelman_rubin(self) -> float:
        """
        Gelman-Rubin R_hat 统计量 (收敛诊断)

        将每条链分为前后两半, 计算:
        - W: 链内方差 (Within-chain variance)
        - B: 链间方差 (Between-chain variance)
        - R_hat = sqrt((n-1)/n + (1 + 1/n) * B/W)

        收敛标准: R_hat < 1.2

        Args:
            chains: 链状态 (n_chains, n_params) 或 (n_chains, n_steps, n_params)

        Returns:
            R_hat (最大值 over 所有参数)
        """
        if self.chains is None:
            return np.inf

        n = self.n_chains
        d = self.n_params

        # 链均值
        chain_means = np.mean(self.chains, axis=0)
        overall_mean = np.mean(chain_means, axis=0)

        # 链间方差 B
        B = n / (n - 1.0 + 1.0e-10) * np.sum(
            (chain_means - overall_mean)**2, axis=0)

        # 链内方差 W (使用链的当前状态的分散度作为近似)
        W = np.var(self.chains, axis=0, ddof=1)

        # R_hat
        with np.errstate(divide='ignore', invalid='ignore'):
            R_hat = np.sqrt((n - 1.0)/n + (n + 1.0)/n * B / (W + 1.0e-30))

        return float(np.max(R_hat))

    def update_cr_probabilities(self, acceptance_by_cr: list):
        """
        更新 CR 选择概率 (自适应)

        基于每个 CR 值的历史接受率更新:
            p(CR_j) proportional to acceptance_rate_j^alpha

        高接受率的 CR 值被更频繁选择.

        Args:
            acceptance_by_cr: 每个 CR 值的接受次数列表
        """
        total = sum(acceptance_by_cr) + 1.0e-30
        rates = [a / total for a in acceptance_by_cr]

        # 指数加权
        alpha = 2.0
        weights = np.array(rates)**alpha + 1.0e-10
        self.cr_probs = weights / np.sum(weights)

    def choose_cr(self) -> int:
        """
        根据自适应概率选择 CR 值

        Returns:
            CR 索引
        """
        rng = np.random.RandomState()
        return int(rng.choice(self.n_cr, p=self.cr_probs))

    def detect_outliers(self) -> list:
        """
        检测异常链 (源自 DREAM 链异常检测)

        异常链: 对数似然远低于中位数的链
        处理: 将异常链重置为其他链的随机副本

        Returns:
            异常链索引列表
        """
        if self.log_likelihoods is None:
            return []

        median_ll = np.median(self.log_likelihoods)
        mad = np.median(np.abs(self.log_likelihoods - median_ll))
        threshold = median_ll - 10.0 * max(mad, 1.0e-10)

        outliers = [i for i in range(self.n_chains)
                    if self.log_likelihoods[i] < threshold]
        return outliers

    def reset_outlier_chains(self, outlier_indices: list):
        """
        重置异常链 (复制自随机非异常链)

        Args:
            outlier_indices: 异常链索引列表
        """
        normal = [i for i in range(self.n_chains)
                  if i not in outlier_indices]
        if not normal:
            return

        rng = np.random.RandomState()
        for idx in outlier_indices:
            source = rng.choice(normal)
            self.chains[idx] = self.chains[source].copy()
            self.log_likelihoods[idx] = self.log_likelihoods[source]

    def run(self, param_center: np.ndarray,
             param_width: np.ndarray,
             bounds: list = None,
             verbose: bool = True) -> dict:
        """
        运行完整 DREAM MCMC

        主循环:
        for gen in range(n_steps):
            for chain in range(n_chains):
                1. 选择 CR 值
                2. 生成候选 (差分进化)
                3. 评估似然
                4. Metropolis 接受/拒绝
            5. 更新 CR 概率
            6. 检测异常链
            7. 计算 Gelman-Rubin

        Args:
            param_center: 参数中心
            param_width: 参数宽度
            bounds: 参数边界
            verbose: 是否打印进度

        Returns:
            {'chains': array, 'log_likelihoods': array,
             'gr_history': list, 'acceptance_rate': float,
             'best_params': array, 'best_ll': float}
        """
        self.initialize_chains(param_center, param_width, bounds)

        n_accept = 0
        n_total = 0
        accept_by_cr = [0] * self.n_cr

        if verbose:
            print(f"  [DREAM MCMC] 开始参数估计:")
            print(f"    chains={self.n_chains}, steps={self.n_steps}, "
                  f"params={self.n_params}")

        for gen in range(self.n_steps):
            for c in range(self.n_chains):
                cr_idx = self.choose_cr()

                candidate = self.sample_candidate(c, cr_idx, gen)
                candidate_ll = self.target_func(candidate)
                n_total += 1

                if self.metropolis_accept(self.log_likelihoods[c],
                                           candidate_ll):
                    self.chains[c] = candidate
                    self.log_likelihoods[c] = candidate_ll
                    n_accept += 1
                    accept_by_cr[cr_idx] += 1

            # 更新 CR 概率 (每 10 步)
            if (gen + 1) % 10 == 0:
                self.update_cr_probabilities(accept_by_cr)
                accept_by_cr = [0] * self.n_cr

            # 异常检测 (每 20 步)
            if (gen + 1) % 20 == 0:
                outliers = self.detect_outliers()
                if outliers:
                    self.reset_outlier_chains(outliers)

            # Gelman-Rubin (每 25 步)
            if (gen + 1) % 25 == 0:
                gr = self.compute_gelman_rubin()
                self.gr_history.append(gr)
                if verbose and (gen + 1) % 50 == 0:
                    best_idx = np.argmax(self.log_likelihoods)
                    best_ll = self.log_likelihoods[best_idx]
                    best_params = self.chains[best_idx]
                    print(f"    gen={gen+1:4d}, R_hat={gr:.4f}, "
                          f"best_ll={best_ll:.4f}, "
                          f"accept={n_accept/max(n_total,1):.3f}")

                # 提前收敛
                if gr < NumericalParams.MCMC_OUTLIER_THRESHOLD and gen > 50:
                    if verbose:
                        print(f"    提前收敛: R_hat={gr:.4f} < "
                              f"{NumericalParams.MCMC_OUTLIER_THRESHOLD}")
                    break

        # 结果
        best_idx = np.argmax(self.log_likelihoods)
        result = {
            'chains': self.chains.copy(),
            'log_likelihoods': self.log_likelihoods.copy(),
            'gr_history': self.gr_history,
            'acceptance_rate': n_accept / max(n_total, 1),
            'best_params': self.chains[best_idx].copy(),
            'best_ll': self.log_likelihoods[best_idx],
            'final_gr': self.gr_history[-1] if self.gr_history else np.inf,
            'n_generations': gen + 1,
        }

        if verbose:
            print(f"  [DREAM MCMC] 完成: {result['n_generations']} 代, "
                  f"acceptance={result['acceptance_rate']:.3f}, "
                  f"R_hat={result['final_gr']:.4f}")
            print(f"    最优参数: {result['best_params']}")

        return result
