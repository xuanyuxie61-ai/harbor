
import numpy as np


class SemanticSpaceSampler:

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def latin_hypercube_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        if n_samples < 2:
            raise ValueError(f"n_samples must be at least 2, got {n_samples}")
        if n_dims < 1:
            raise ValueError(f"n_dims must be at least 1, got {n_dims}")

        samples = np.zeros((n_samples, n_dims))

        for d in range(n_dims):

            perm = self.rng.permutation(n_samples)

            for i in range(n_samples):
                lower = perm[i] / n_samples
                upper = (perm[i] + 1) / n_samples
                samples[i, d] = self.rng.uniform(lower, upper)

        return samples

    def latinize(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        m, n = data.shape

        if m <= 2:
            return data.copy()

        result = data.copy()

        for j in range(n):
            v_min = float(np.min(data[:, j]))
            v_max = float(np.max(data[:, j]))


            indx = np.argsort(data[:, j])

            for i in range(m):

                result[indx[i], j] = ((m - 1 - i) * v_min + i * v_max) / (m - 1)

        return result

    def sobol_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_dims, scramble=True, seed=self.rng.integers(0, 2**31))
            samples = sampler.random(n=n_samples)
            return samples
        except ImportError:
            return self.latin_hypercube_sampling(n_samples, n_dims)

    def project_to_hypersphere(self, samples: np.ndarray, radius: float = 1.0) -> np.ndarray:
        samples = np.asarray(samples, dtype=float)

        samples = 2.0 * samples - 1.0

        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return radius * samples / norms

    def uniform_direction_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        samples = self.rng.standard_normal((n_samples, n_dims))
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return samples / norms

    def discrepancy(self, samples: np.ndarray) -> float:
        samples = np.asarray(samples, dtype=float)
        m, n = samples.shape


        n_test = min(1000, m * 10)
        test_points = self.rng.random((n_test, n))

        max_discrepancy = 0.0
        for tp in test_points:

            in_box = np.all(samples <= tp, axis=1)
            empirical = np.sum(in_box) / m
            volume = np.prod(tp)
            disc = abs(volume - empirical)
            max_discrepancy = max(max_discrepancy, disc)

        return max_discrepancy


def demo():
    print("=" * 60)
    print("语义嵌入空间采样演示")
    print("=" * 60)

    sampler = SemanticSpaceSampler(seed=42)


    print("\n--- 拉丁超立方采样 (LHS) ---")
    n_samples, n_dims = 50, 5
    samples = sampler.latin_hypercube_sampling(n_samples, n_dims)
    print(f"样本数: {n_samples}, 维度: {n_dims}")
    print(f"样本范围: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"每维均值: {samples.mean(axis=0)}")
    print(f"每维标准差: {samples.std(axis=0)}")


    print("\n--- Latinize 变换 ---")
    rng = np.random.default_rng(42)
    raw_data = rng.standard_normal((20, 4))
    latinized = sampler.latinize(raw_data)
    print(f"原始数据范围: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    print(f"Latinized范围: [{latinized.min():.4f}, {latinized.max():.4f}]")


    print("\n--- 超球面方向采样 ---")
    directions = sampler.uniform_direction_sampling(100, 8)
    norms = np.linalg.norm(directions, axis=1)
    print(f"方向向量范数: 均值={norms.mean():.6f}, 标准差={norms.std():.6e}")
    print(f"范数范围: [{norms.min():.6f}, {norms.max():.6f}]")


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
