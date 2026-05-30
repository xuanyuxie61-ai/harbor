
import numpy as np
from itertools import combinations


class SemanticSubsetSelector:

    def __init__(self, max_brute_force_n: int = 22):
        self.max_brute_force_n = int(max_brute_force_n)

    def _int_to_binary_vector(self, i: int, n: int) -> np.ndarray:
        vec = np.zeros(n, dtype=int)
        for k in range(n):
            vec[k] = (i >> k) & 1
        return vec

    def brute_force_search(self, weights: np.ndarray, target: float,
                           tolerance: float = 1e-9) -> dict:
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
        weights = np.asarray(weights, dtype=float)
        n = len(weights)
        choose = np.zeros(n, dtype=int)
        remaining = float(target)


        sorted_idx = np.argsort(np.abs(weights))[::-1]

        for idx in sorted_idx:
            w = weights[idx]
            if abs(w) < 1e-15:
                continue

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
        weights = np.asarray(weights, dtype=float)
        n = len(weights)


        int_weights = np.round(weights * scale_factor).astype(int)
        int_target = int(round(target * scale_factor))

        max_sum = np.sum(np.abs(int_weights))
        if max_sum > 1000000:
            raise ValueError("scaled sum too large for DP")


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


        best_s = offset
        best_error = abs(int_target)
        for s in range(2 * max_sum + 1):
            if dp[s]:
                error = abs(s - int_target)
                if error < best_error:
                    best_error = error
                    best_s = s

        choose = np.zeros(n, dtype=int)


        remaining = best_s - offset
        for i in range(n - 1, -1, -1):
            w = int_weights[i]
            if remaining >= w and w != 0:

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
    print("=" * 60)
    print("语义特征子集选择演示")
    print("=" * 60)

    selector = SemanticSubsetSelector(max_brute_force_n=20)


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


    print("\n--- 语义嵌入特征选择 ---")
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(16)

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
