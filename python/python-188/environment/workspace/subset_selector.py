"""
子集和特征选择模块：语义嵌入的最优特征子集搜索

原项目映射: 1180_subset_sum_brute
科学背景: 子集和问题（Subset Sum Problem）是经典的NP完全问题：
          给定 N 个权重和一个目标和，寻找一个子集使其和恰好等于目标值。
          
          暴力搜索方法:
            遍历所有 2^N 个可能的子集（用N位二进制数表示），
            检查每个子集的和是否等于目标值。

数学模型:
    给定: weight[1..N], target
    求: choose[1..N] \in {0, 1}
    满足: sum(choose[i] * weight[i]) = target
    
    搜索空间大小: 2^N

在NLP语义嵌入中的应用:
    将语义特征的重要性视为权重，寻找最优特征子集使得:
        - 保留的语义信息量最大化
        - 计算成本最小化
    
    具体应用:
        - 降维: 从高维语义嵌入中选择最有信息量的维度
        - 模型压缩: 选择最小的特征子集保持模型性能
        - 可解释性: 识别对特定语义任务最重要的特征
"""

import numpy as np
from itertools import combinations


class SemanticSubsetSelector:
    """
    语义嵌入特征子集选择器。
    """

    def __init__(self, max_brute_force_n: int = 22):
        """
        Parameters
        ----------
        max_brute_force_n : int
            暴力搜索的最大N值（超过则使用启发式方法）。
        """
        self.max_brute_force_n = int(max_brute_force_n)

    def _int_to_binary_vector(self, i: int, n: int) -> np.ndarray:
        """
        将整数 i 转换为 n 位二进制向量。
        """
        vec = np.zeros(n, dtype=int)
        for k in range(n):
            vec[k] = (i >> k) & 1
        return vec

    def brute_force_search(self, weights: np.ndarray, target: float,
                           tolerance: float = 1e-9) -> dict:
        """
        暴力搜索子集和。
        
        Parameters
        ----------
        weights : np.ndarray
            权重数组（语义特征重要性）。
        target : float
            目标和。
        tolerance : float
            数值容差。
            
        Returns
        -------
        dict
            包含最优子集、实际和、误差的结果。
        """
        weights = np.asarray(weights, dtype=float)
        n = len(weights)

        if n > self.max_brute_force_n:
            raise ValueError(f"n={n} too large for brute force (max={self.max_brute_force_n})")

        best_choose = None
        best_error = float('inf')
        best_sum = 0.0
        num_checked = 0

        for i in range(2 ** n):
            choose = self._int_to_binary_vector(i, n)
            s = np.dot(choose, weights)
            error = abs(s - target)
            num_checked += 1

            if error < best_error:
                best_error = error
                best_sum = s
                best_choose = choose.copy()
                if error <= tolerance:
                    break

        return {
            'choose': best_choose,
            'subset_sum': best_sum,
            'target': target,
            'error': best_error,
            'num_checked': num_checked,
            'found_exact': best_error <= tolerance
        }

    def greedy_approximation(self, weights: np.ndarray, target: float) -> dict:
        """
        贪心近似算法（用于大规模问题）。
        
        策略：每次选择使剩余差距最小的权重。
        """
        weights = np.asarray(weights, dtype=float)
        n = len(weights)
        choose = np.zeros(n, dtype=int)
        remaining = float(target)

        # 按权重排序的索引（从大到小）
        sorted_idx = np.argsort(np.abs(weights))[::-1]

        for idx in sorted_idx:
            w = weights[idx]
            if abs(w) < 1e-15:
                continue
            # 选择这个权重是否能减小差距
            if abs(remaining - w) < abs(remaining):
                choose[idx] = 1
                remaining -= w

        s = np.dot(choose, weights)
        return {
            'choose': choose,
            'subset_sum': s,
            'target': target,
            'error': abs(s - target),
            'num_checked': n
        }

    def dynamic_programming(self, weights: np.ndarray, target: float,
                            scale_factor: float = 100.0) -> dict:
        """
        动态规划求解（整数权重版本）。
        
        将权重缩放为整数后使用经典DP。
        """
        weights = np.asarray(weights, dtype=float)
        n = len(weights)

        # 缩放为整数
        int_weights = np.round(weights * scale_factor).astype(int)
        int_target = int(round(target * scale_factor))

        max_sum = np.sum(np.abs(int_weights))
        if max_sum > 1000000:
            raise ValueError("scaled sum too large for DP")

        # dp[s] = 是否可以达到和 s
        offset = max_sum
        dp = np.zeros(2 * max_sum + 1, dtype=bool)
        dp[offset] = True
        parent = {offset: []}

        for i in range(n):
            w = int_weights[i]
            new_dp = dp.copy()
            for s in range(2 * max_sum + 1):
                if dp[s]:
                    new_s = s + w
                    if 0 <= new_s <= 2 * max_sum:
                        new_dp[new_s] = True
                        if new_s not in parent:
                            parent[new_s] = parent.get(s, []) + [i]
            dp = new_dp

        # 找最接近 target 的解
        best_s = offset
        best_error = abs(int_target)
        for s in range(2 * max_sum + 1):
            if dp[s]:
                error = abs(s - int_target)
                if error < best_error:
                    best_error = error
                    best_s = s

        choose = np.zeros(n, dtype=int)
        # 回溯（简化版：直接搜索找到的元素）
        # 这里使用贪心回溯
        remaining = best_s - offset
        for i in range(n - 1, -1, -1):
            w = int_weights[i]
            if remaining >= w and w != 0:
                # 检查是否可以去掉这个权重
                temp_rem = remaining - w
                if dp[offset + temp_rem]:
                    choose[i] = 1
                    remaining = temp_rem

        actual_sum = np.dot(choose, weights)
        return {
            'choose': choose,
            'subset_sum': actual_sum,
            'target': target,
            'error': abs(actual_sum - target),
            'found_exact': best_error == 0
        }

    def feature_selection_for_embedding(self, embedding: np.ndarray,
                                        target_info_ratio: float = 0.8) -> dict:
        """
        为语义嵌入向量选择最优特征子集。
        
        策略: 将各维度的绝对值作为权重，目标是保留总信息量的 target_info_ratio。
        """
        embedding = np.asarray(embedding, dtype=float)
        weights = np.abs(embedding)
        total_weight = np.sum(weights)
        target = target_info_ratio * total_weight

        n = len(weights)
        if n <= self.max_brute_force_n:
            return self.brute_force_search(weights, target)
        else:
            return self.greedy_approximation(weights, target)


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("语义特征子集选择演示")
    print("=" * 60)

    selector = SemanticSubsetSelector(max_brute_force_n=20)

    # 示例1: 小规模精确解
    print("\n--- 小规模暴力搜索 ---")
    weights = np.array([1, 2, 4, 8, 16, 32])
    target = 22.0
    result = selector.brute_force_search(weights, target)
    print(f"权重: {weights}")
    print(f"目标: {target}")
    print(f"选择: {result['choose']}")
    print(f"子集和: {result['subset_sum']}")
    print(f"精确解: {result['found_exact']}")
    print(f"搜索次数: {result['num_checked']}/{2**len(weights)}")

    # 示例2: 语义嵌入特征选择
    print("\n--- 语义嵌入特征选择 ---")
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(16)
    # 模拟衰减的重要性
    embedding = embedding * np.exp(-np.arange(16) / 5.0)

    result = selector.feature_selection_for_embedding(embedding, target_info_ratio=0.8)
    selected_dims = np.where(result['choose'] == 1)[0]
    total_info = np.sum(np.abs(embedding))
    selected_info = np.sum(np.abs(embedding[selected_dims]))

    print(f"嵌入维度: {len(embedding)}")
    print(f"总信息量: {total_info:.4f}")
    print(f"目标保留: {result['target']:.4f} (80%)")
    print(f"实际保留: {selected_info:.4f}")
    print(f"选择维度数: {len(selected_dims)}")
    print(f"选择维度: {selected_dims}")
    print(f"相对误差: {result['error'] / total_info * 100:.2f}%")

    # 示例3: 动态规划
    print("\n--- 动态规划求解 ---")
    weights2 = np.array([3, 7, 11, 13, 17, 19, 23])
    target2 = 50.0
    result_dp = selector.dynamic_programming(weights2, target2, scale_factor=10.0)
    print(f"权重: {weights2}")
    print(f"目标: {target2}")
    print(f"选择: {result_dp['choose']}")
    print(f"子集和: {result_dp['subset_sum']:.4f}")
    print(f"精确解: {result_dp['found_exact']}")

    print("\n模块运行完成")
    return selector, result


if __name__ == "__main__":
    demo()
