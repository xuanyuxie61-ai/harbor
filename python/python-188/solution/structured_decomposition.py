"""
结构化分解模块：语义嵌入维度的因式分解降维

原项目映射: 420_fermat_factor
科学背景: Fermat因式分解方法基于平方差公式:
            N = A^2 - B^2 = (A+B)(A-B)
          
          算法:
            从 A = floor(sqrt(N)) 开始，逐步增大 A，
            检查 A^2 - N 是否为完全平方数。
            
          复杂度: 当两个因子接近时效率最高，O(sqrt(N)) 最坏情况。

数学模型:
    输入: 正整数 N
    输出: 因子 f1, f2 使得 N = f1 * f2
    
    迭代:
        for A = floor(sqrt(N)) to N:
            B^2 = A^2 - N
            if B^2 是完全平方数:
                f1 = A + B
                f2 = A - B
                return

在NLP语义嵌入中的应用:
    将嵌入维度 N 进行结构化因式分解，寻找最优的因子分解形式，
    用于设计高效的张量分解结构和分层语义表示。
    
    例如:
        - 维度 768 = 24 * 32，可设计为 24x32 的矩阵结构
        - 利用因子分解进行分层降维
"""

import numpy as np
from math import isqrt


class SemanticStructuredDecomposition:
    """
    语义嵌入维度结构化分解系统。
    """

    def __init__(self):
        pass

    @staticmethod
    def fermat_factor(n: int) -> tuple:
        """
        Fermat因式分解。
        
        Parameters
        ----------
        n : int
            待分解的正整数，必须 >= 2。
            
        Returns
        -------
        tuple
            (f1, f2) 使得 n = f1 * f2
            
        Raises
        ------
        ValueError
            如果 n 为素数或无法分解。
        """
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

        # n 是素数
        return n, 1

    @staticmethod
    def prime_factors(n: int) -> list:
        """
        质因数分解。
        
        Returns
        -------
        list
            质因数列表（含重复）。
        """
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
        """
        获取 n 的所有因子对。
        """
        factors = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        return factors

    def optimal_tensor_shape(self, dim: int, max_rank: int = 4) -> dict:
        """
        寻找最优的张量分解形状。
        
        目标: 将维度 dim 分解为不超过 max_rank 个因子的乘积，
              使得各因子尽可能均衡。
        """
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")

        factors = self.prime_factors(dim)

        if len(factors) == 0:
            return {'shape': [dim], 'balance_score': 0.0}

        # 贪心合并: 每次合并最小的两个因子
        while len(factors) > max_rank:
            factors = sorted(factors)
            merged = factors[0] * factors[1]
            factors = [merged] + factors[2:]

        # 补1使达到max_rank
        while len(factors) < max_rank:
            factors.append(1)

        shape = sorted(factors, reverse=True)

        # 平衡度评分: 标准差的倒数
        balance = 1.0 / (np.std(shape) + 1e-15)

        return {
            'shape': shape,
            'balance_score': balance,
            'volume': np.prod(shape)
        }

    def embedding_dimension_analysis(self, dims: list) -> dict:
        """
        分析常见语义嵌入维度的结构。
        """
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
        """
        对语义嵌入进行分层结构化分解。
        
        将一维嵌入向量重塑为多维张量。
        """
        embedding = np.asarray(embedding, dtype=float)
        dim = len(embedding)

        if target_shape is None:
            shape_info = self.optimal_tensor_shape(dim, max_rank=4)
            target_shape = tuple(shape_info['shape'])

        target_volume = int(np.prod(target_shape))
        if target_volume != dim:
            # 填充或截断
            if target_volume > dim:
                padded = np.zeros(target_volume)
                padded[:dim] = embedding
                embedding = padded
            else:
                embedding = embedding[:target_volume]

        tensor = embedding.reshape(target_shape)

        # 计算各模态的奇异值分解能量分布
        mode_energies = []
        for mode in range(len(target_shape)):
            # 将张量展开为矩阵
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
    """模块功能演示"""
    print("=" * 60)
    print("语义嵌入维度结构化分解演示")
    print("=" * 60)

    decomp = SemanticStructuredDecomposition()

    # Fermat因式分解
    print("\n--- Fermat因式分解 ---")
    test_numbers = [91, 143, 899, 2025, 768, 512]
    for n in test_numbers:
        f1, f2 = decomp.fermat_factor(n)
        primes = decomp.prime_factors(n)
        print(f"{n:4d} = {f1} * {f2}  (质因数: {primes})")

    # 常见嵌入维度分析
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

    # 分层分解
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
