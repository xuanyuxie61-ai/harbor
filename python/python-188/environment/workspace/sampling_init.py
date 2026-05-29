"""
拉丁超立方采样模块：语义嵌入空间的高维均匀采样

原项目映射: 653_latinize
科学背景: Latin Hypercube Sampling (LHS) 是一种统计采样方法，
          确保在每个维度上的投影都是均匀分布的。

数学模型:
    给定 M 个样本和 N 个维度:
        1. 将每个维度 [0,1] 划分为 M 个等宽区间
        2. 在每个维度上，将 1..M 的排列随机分配给 M 个样本
        3. 在每个区间内均匀随机采样
    
    Latinize 操作:
        对已有数据集进行"拉丁化"，使得每行的值在最小值和最大值之间
        均匀分布，同时保持原有的排序关系。
        
        对于每列 j:
            1. 找到最小值 v_min 和最大值 v_max
            2. 对列值排序，得到排序索引 indx
            3. 对排序后的位置赋予均匀分布的值:
                table[indx[i], j] = ((M-i)*v_min + (i-1)*v_max) / (M-1)

在NLP语义嵌入中的应用:
    - 生成高维语义空间的代表性初始样本
    - 确保采样在各个语义维度上均匀覆盖
    - 用于语义嵌入的初始化、贝叶斯优化等
"""

import numpy as np


class SemanticSpaceSampler:
    """
    语义嵌入空间采样器。
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def latin_hypercube_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """
        生成拉丁超立方样本。
        
        Parameters
        ----------
        n_samples : int
            样本数，必须 >= 2。
        n_dims : int
            维度数，必须 >= 1。
            
        Returns
        -------
        np.ndarray
            n_samples x n_dims 的采样矩阵。
        """
        if n_samples < 2:
            raise ValueError(f"n_samples must be at least 2, got {n_samples}")
        if n_dims < 1:
            raise ValueError(f"n_dims must be at least 1, got {n_dims}")

        samples = np.zeros((n_samples, n_dims))

        for d in range(n_dims):
            # 生成随机排列
            perm = self.rng.permutation(n_samples)
            # 在每个区间内均匀采样
            for i in range(n_samples):
                lower = perm[i] / n_samples
                upper = (perm[i] + 1) / n_samples
                samples[i, d] = self.rng.uniform(lower, upper)

        return samples

    def latinize(self, data: np.ndarray) -> np.ndarray:
        """
        对数据集进行拉丁化处理。
        
        对每一列:
            1. 保持最小值和最大值不变
            2. 将值替换为均匀间隔的值
            3. 保持原有的排序关系
        """
        data = np.asarray(data, dtype=float)
        m, n = data.shape

        if m <= 2:
            return data.copy()

        result = data.copy()

        for j in range(n):
            v_min = float(np.min(data[:, j]))
            v_max = float(np.max(data[:, j]))

            # 获取排序索引
            indx = np.argsort(data[:, j])

            for i in range(m):
                # 赋予均匀分布的值
                result[indx[i], j] = ((m - 1 - i) * v_min + i * v_max) / (m - 1)

        return result

    def sobol_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """
        使用Sobol序列采样（如果可用）。
        
        否则退化为LHS。
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.rng.integers(0, 2**31))
            samples = sampler.random(n=n_samples)
            return samples
        except ImportError:
            return self.latin_hypercube_sampling(n_samples, n_dims)

    def project_to_hypersphere(self, samples: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        将样本投影到超球面上。
        
        用于生成单位球面上的语义嵌入初始点。
        """
        samples = np.asarray(samples, dtype=float)
        # 先映射到 [-1, 1]
        samples = 2.0 * samples - 1.0
        # 归一化
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return radius * samples / norms

    def uniform_direction_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """
        在超球面上均匀采样方向。
        
        使用正态分布后归一化（Marsaglia方法）。
        """
        samples = self.rng.standard_normal((n_samples, n_dims))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return samples / norms

    def discrepancy(self, samples: np.ndarray) -> float:
        """
        计算星差异度（Star Discrepancy）。
        
        D* = sup_{x in [0,1]^d} |Vol([0,x]) - #{samples in [0,x]}/N|
        
        使用随机点估计上界。
        """
        samples = np.asarray(samples, dtype=float)
        m, n = samples.shape

        # 随机测试点
        n_test = min(1000, m * 10)
        test_points = self.rng.random((n_test, n))

        max_discrepancy = 0.0
        for tp in test_points:
            # 计算在 [0, tp] 内的样本比例
            in_box = np.all(samples <= tp, axis=1)
            empirical = np.sum(in_box) / m
            volume = np.prod(tp)
            disc = abs(volume - empirical)
            max_discrepancy = max(max_discrepancy, disc)

        return max_discrepancy


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("语义嵌入空间采样演示")
    print("=" * 60)

    sampler = SemanticSpaceSampler(seed=42)

    # LHS
    print("\n--- 拉丁超立方采样 (LHS) ---")
    n_samples, n_dims = 50, 5
    samples = sampler.latin_hypercube_sampling(n_samples, n_dims)
    print(f"样本数: {n_samples}, 维度: {n_dims}")
    print(f"样本范围: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"每维均值: {samples.mean(axis=0)}")
    print(f"每维标准差: {samples.std(axis=0)}")

    # Latinize
    print("\n--- Latinize 变换 ---")
    rng = np.random.default_rng(42)
    raw_data = rng.standard_normal((20, 4))
    latinized = sampler.latinize(raw_data)
    print(f"原始数据范围: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    print(f"Latinized范围: [{latinized.min():.4f}, {latinized.max():.4f}]")

    # 超球面采样
    print("\n--- 超球面方向采样 ---")
    directions = sampler.uniform_direction_sampling(100, 8)
    norms = np.linalg.norm(directions, axis=1)
    print(f"方向向量范数: 均值={norms.mean():.6f}, 标准差={norms.std():.6e}")
    print(f"范数范围: [{norms.min():.6f}, {norms.max():.6f}]")

    # 差异度
    print("\n--- 采样质量评估 ---")
    lhs_disc = sampler.discrepancy(samples)
    random_samples = rng.random((n_samples, n_dims))
    random_disc = sampler.discrepancy(random_samples)
    print(f"LHS 星差异度:     {lhs_disc:.6f}")
    print(f"随机采样差异度:   {random_disc:.6f}")
    print(f"LHS 改进比例:     {(random_disc - lhs_disc) / random_disc * 100:.2f}%")

    print("\n模块运行完成")
    return sampler, samples


if __name__ == "__main__":
    demo()
