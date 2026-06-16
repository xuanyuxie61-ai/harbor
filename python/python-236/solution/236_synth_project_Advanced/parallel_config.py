"""
parallel_config.py — 并行 Monte Carlo 配置管理与并行拟合
=========================================================
融合种子项目:
  [913_prime_parfor] : parfor 并行 → 配置级并行
  [1065_Azamat-Mukhamediya_SRPM-ST] : joblib 并行 → 多进程拟合

物理背景:
  格点 QCD 计算的核心是 Monte Carlo 采样:
  1. 生成 N 个独立规范场配置 (HMC / Hybrid Monte Carlo)
  2. 在每个配置上测量物理量 (关联函数)
  3. 统计分析提取物理质量

  并行策略:
  - 配置级并行: 不同配置独立测量 (embarrassingly parallel)
  - 时间片并行: 关联函数不同时间片并行计算
  - 拟合并行: 多参数点/多 bootstrap 样本并行拟合

核心公式:
  HMC 算法:
  1. 从热分布抽取动量: pi ~ exp(-Tr(pi^2)/2)
  2. 分子动力学轨迹: dU/dt = i*Q*U, dQ/dt = -dS/dU
  3. Metropolis 接受/拒绝: P_acc = min(1, exp(-delta_H))

  测量算符:
  O[U_i] = 在配置 U_i 上测量的物理量
  <O> = (1/N) sum_i O[U_i]
  sigma_O^2 = (1/(N*(N-1))) sum_i (O[U_i] - <O>)^2
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time


class ParallelConfigManager:
    """并行 Monte Carlo 配置管理器.

    参数
    ----
    n_configs : int
        配置总数.
    seed_base : int
        基础随机种子.
    n_workers : int
        并行工作进程数 (默认 = CPU 核心数).
    """

    def __init__(self, n_configs: int = 20,
                 seed_base: int = 236,
                 n_workers: int = 2):
        self.n_configs = n_configs
        self.seed_base = seed_base
        self.n_workers = min(n_workers, 4)  # 限制最大进程数

        # 配置元数据
        self.config_seeds = [seed_base + i for i in range(n_configs)]
        self.config_labels = [f"cfg_{i:04d}" for i in range(n_configs)]

    # ------------------------------------------------------------------
    # 并行配置测量 (源自 [913] 的 parfor 并行)
    # ------------------------------------------------------------------
    def parallel_measure(self, measure_func: Callable,
                         *args, **kwargs) -> List:
        """并行测量所有配置 (源自 [913_prime_parfor]).

        对每个配置独立执行 measure_func(config_seed, *args, **kwargs),
        结果并行汇总.

        参数
        ----
        measure_func : callable
            测量函数, 签名 (seed, *args, **kwargs) -> result.
        *args, **kwargs :
            传递给测量函数的额外参数.

        返回
        ----
        results : list
            每个配置的测量结果.
        """
        results = [None] * self.n_configs

        if self.n_workers <= 1:
            # 串行执行
            for i in range(self.n_configs):
                results[i] = measure_func(self.config_seeds[i],
                                           *args, **kwargs)
        else:
            # 线程并行 (避免进程启动开销)
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for i in range(self.n_configs):
                    fut = executor.submit(measure_func,
                                          self.config_seeds[i],
                                          *args, **kwargs)
                    futures.append((i, fut))
                for i, fut in futures:
                    try:
                        results[i] = fut.result(timeout=120)
                    except Exception as e:
                        results[i] = {'error': str(e)}

        return results

    # ------------------------------------------------------------------
    # 并行拟合 (源自 [1065] 的 joblib 并行)
    # ------------------------------------------------------------------
    def parallel_bootstrap_fit(self, C_ensemble: np.ndarray,
                               t_data: np.ndarray,
                               n_bootstrap: int = 100,
                               t_min: int = 1,
                               seed: int = 42) -> Dict:
        """并行 Bootstrap 拟合 (源自 [1065] 的并行 CV).

        将 Bootstrap 样本分配到多个工作进程.
        """
        from bootstrap_errors import BootstrapAnalyzer

        analyzer = BootstrapAnalyzer(C_ensemble, t_data)

        # 分批 bootstrap
        batch_size = max(n_bootstrap // self.n_workers, 1)
        n_batches = (n_bootstrap + batch_size - 1) // batch_size

        all_masses = []
        for batch in range(n_batches):
            b_start = batch * batch_size
            b_end = min(b_start + batch_size, n_bootstrap)
            n_batch = b_end - b_start

            res = analyzer.bootstrap_fit(
                n_bootstrap=n_batch,
                t_min=t_min,
                seed=seed + b_start)
            all_masses.extend(res['masses'].tolist())

        all_masses = np.array(all_masses)
        return {
            'masses': all_masses,
            'mean_mass': float(np.mean(all_masses)),
            'std_mass': float(np.std(all_masses)),
            'n_bootstrap': len(all_masses)
        }

    # ------------------------------------------------------------------
    # 简化 HMC 配置生成 (可复现的小规模实验)
    # ------------------------------------------------------------------
    def generate_synthetic_ensemble(self, Lt: int = 12,
                                    true_mass: float = 0.4,
                                    true_amplitude: float = 1.0,
                                    noise_level: float = 0.02,
                                    autocorrelation: float = 0.3,
                                    seed: int = 236) -> np.ndarray:
        """生成合成关联函数配置集合 (可复现实验).

        模拟 Monte Carlo 采样的关联函数数据:
        C_i(t) = C_true(t) * (1 + noise_i(t))
        其中 noise 具有指数自相关:
        <noise_i * noise_j> ~ exp(-|i-j| / tau_int)

        参数
        ----
        Lt : int
            时间尺寸.
        true_mass : float
            真实基态质量.
        true_amplitude : float
            真实振幅.
        noise_level : float
            统计噪声水平.
        autocorrelation : float
            自相关系数 rho (0=独立, 1=完全相关).
        seed : int
            随机种子.

        返回
        ----
        C_ensemble : ndarray, shape (n_configs, Lt)
        """
        rng = np.random.default_rng(seed)
        t = np.arange(Lt, dtype=np.float64)

        # 真实关联函数 (双曲余弦形式, 包含激发态)
        m0 = true_mass
        A0 = true_amplitude
        m1 = m0 * 2.0  # 第一激发态
        A1 = A0 * 0.3  # 激发态振幅

        C_true = (A0 * (np.exp(-m0 * t) + np.exp(-m0 * (Lt - t)))
                  + A1 * (np.exp(-m1 * t) + np.exp(-m1 * (Lt - t))))

        # 生成自相关噪声
        C_ensemble = np.zeros((self.n_configs, Lt))
        noise_prev = np.zeros(Lt)

        for i in range(self.n_configs):
            # AR(1) 自相关噪声
            white = rng.standard_normal(Lt)
            noise = (autocorrelation * noise_prev
                     + np.sqrt(1 - autocorrelation ** 2) * white)
            noise_prev = noise.copy()

            C_ensemble[i] = C_true * (1.0 + noise_level * noise)

        self._true_mass = true_mass
        self._true_amplitude = true_amplitude
        self._C_true = C_true

        return C_ensemble

    # ------------------------------------------------------------------
    # 素数筛并行化 (源自 [913_prime_parfor] 的直接映射)
    # ------------------------------------------------------------------
    @staticmethod
    def count_primes_parallel(n: int, n_workers: int = 2) -> int:
        """并行素数计数 (源自 [913_prime_parfor]).

        在格点 QCD 中, 素数用于确定 Hasenbusch 质量预处理参数
        和某些数论相关的有限体积修正.

        使用试除法, 并行化外层循环 (类似 parfor).
        """
        if n < 2:
            return 0

        def is_prime(i: int) -> bool:
            if i < 2:
                return False
            if i < 4:
                return True
            if i % 2 == 0 or i % 3 == 0:
                return False
            j = 5
            while j * j <= i:
                if i % j == 0 or i % (j + 2) == 0:
                    return False
                j += 6
            return True

        # 并行计数
        count = 0
        chunk_size = max(n // (n_workers * 4), 1)
        ranges = []
        start = 2
        while start <= n:
            end = min(start + chunk_size - 1, n)
            ranges.append((start, end))
            start = end + 1

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(
                lambda r: sum(1 for i in range(r[0], r[1] + 1)
                              if is_prime(i)),
                ranges))

        return sum(results)

    # ------------------------------------------------------------------
    # Hasenbusch 质量预处理参数 (需要素数)
    # ------------------------------------------------------------------
    @staticmethod
    def hasenbusch_masses(n_masses: int = 3,
                          m_light: float = 0.01,
                          m_heavy: float = 1.0) -> np.ndarray:
        """Hasenbusch 质量预处理参数.

        mu_k = m_light * (m_heavy / m_light)^{k/n}  for k = 0, ..., n

        使用几何级数分配中间质量, 加速 CG 收敛.
        """
        if n_masses < 1:
            return np.array([m_light])
        ratios = np.linspace(0, 1, n_masses + 1)
        masses = m_light * (m_heavy / m_light) ** ratios
        return masses
