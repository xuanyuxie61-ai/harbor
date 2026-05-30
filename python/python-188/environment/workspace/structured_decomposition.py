
import numpy as np
from math import isqrt


class SemanticStructuredDecomposition:

    def __init__(self):
        pass

    @staticmethod
    def fermat_factor(n: int) -> tuple:
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if n % 2 == 0:
            return n // 2, 2

        a = isqrt(n)
        if a * a == n:
            return a, a

        a = a + 1
        max_a = n
        while a <= max_a:
            b_sq = a * a - n
            if b_sq < 0:
                a += 1
                continue
            b = isqrt(b_sq)
            if b * b == b_sq:
                f1 = a + b
                f2 = a - b
                return f1, f2
            a += 1


        return n, 1

    @staticmethod
    def prime_factors(n: int) -> list:
        if n < 2:
            return []

        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    @staticmethod
    def all_factorizations(n: int) -> list:
        factors = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        return factors

    def optimal_tensor_shape(self, dim: int, max_rank: int = 4) -> dict:
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")

        factors = self.prime_factors(dim)

        if len(factors) == 0:
            return {'shape': [dim], 'balance_score': 0.0}


        while len(factors) > max_rank:
            factors = sorted(factors)
            merged = factors[0] * factors[1]
            factors = [merged] + factors[2:]


        while len(factors) < max_rank:
            factors.append(1)

        shape = sorted(factors, reverse=True)


        balance = 1.0 / (np.std(shape) + 1e-15)

        return {
            'shape': shape,
            'balance_score': balance,
            'volume': np.prod(shape)
        }

    def embedding_dimension_analysis(self, dims: list) -> dict:
        results = {}
        for dim in dims:
            f1, f2 = self.fermat_factor(dim)
            primes = self.prime_factors(dim)
            tensor_shape = self.optimal_tensor_shape(dim, max_rank=4)

            results[dim] = {
                'fermat_factors': (f1, f2),
                'prime_factors': primes,
                'optimal_tensor_shape': tensor_shape['shape'],
                'balance_score': tensor_shape['balance_score']
            }

        return results

    def hierarchical_decomposition(self, embedding: np.ndarray,
                                   target_shape: tuple = None) -> dict:
        embedding = np.asarray(embedding, dtype=float)
        dim = len(embedding)

        if target_shape is None:
            shape_info = self.optimal_tensor_shape(dim, max_rank=4)
            target_shape = tuple(shape_info['shape'])

        target_volume = int(np.prod(target_shape))
        if target_volume != dim:

            if target_volume > dim:
                padded = np.zeros(target_volume)
                padded[:dim] = embedding
                embedding = padded
            else:
                embedding = embedding[:target_volume]

        tensor = embedding.reshape(target_shape)


        mode_energies = []
        for mode in range(len(target_shape)):

            new_shape = (-1, target_shape[mode])
            matrix = tensor.reshape(new_shape)
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            energy_ratio = s[0] / (np.sum(s) + 1e-15)
            mode_energies.append({
                'mode': mode,
                'shape': target_shape[mode],
                'rank': len(s),
                'dominant_energy_ratio': energy_ratio
            })

        return {
            'original_dim': dim,
            'tensor_shape': target_shape,
            'tensor': tensor,
            'mode_energies': mode_energies
        }


def demo():
    print("=" * 60)
    print("语义嵌入维度结构化分解演示")
    print("=" * 60)

    decomp = SemanticStructuredDecomposition()


    print("\n--- Fermat因式分解 ---")
    test_numbers = [91, 143, 899, 2025, 768, 512]
    for n in test_numbers:
        f1, f2 = decomp.fermat_factor(n)
        primes = decomp.prime_factors(n)
        print(f"{n:4d} = {f1} * {f2}  (质因数: {primes})")


    print("\n--- 常见语义嵌入维度分析 ---")
    common_dims = [128, 256, 384, 512, 768, 1024]
    analysis = decomp.embedding_dimension_analysis(common_dims)
    for dim in common_dims:
        info = analysis[dim]
        print(f"\n维度 {dim}:")
        print(f"  Fermat分解: {info['fermat_factors']}")
        print(f"  质因数分解: {info['prime_factors']}")
        print(f"  最优张量形状: {info['optimal_tensor_shape']}")
        print(f"  平衡度评分: {info['balance_score']:.4f}")


    print("\n--- 语义嵌入分层分解 ---")
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(768)
    result = decomp.hierarchical_decomposition(embedding, target_shape=(8, 8, 12))
    print(f"原始维度: {result['original_dim']}")
    print(f"张量形状: {result['tensor_shape']}")
    print(f"张量元素和: {result['tensor'].sum():.4f}")
    print(f"各模态能量分析:")
    for me in result['mode_energies']:
        print(f"  模态 {me['mode']} (形状 {me['shape']}): "
              f"主导能量比 = {me['dominant_energy_ratio']:.4f}")

    print("\n模块运行完成")
    return decomp, result


if __name__ == "__main__":
    demo()
