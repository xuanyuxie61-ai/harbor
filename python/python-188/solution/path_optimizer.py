"""
TSP路径优化模块：语义嵌入遍历路径优化

原项目映射: 1364_tsp_descent
科学背景: 使用下降法（局部搜索）求解旅行商问题(TSP)，
          通过换位(transpose)和反转(reversal)两种邻域操作
          寻找近似最优路径。

数学模型:
    TSP: 给定 N 个城市的距离矩阵 D，寻找排列 p 使得:
        cost(p) = sum_{i=1}^{N} D(p_i, p_{i+1})
    其中 p_{N+1} = p_1。
    
    邻域操作:
        1. Transpose: 交换两个非相邻城市的位置
           p' = [p_1,...,p_i, p_j, p_{i+1},...,p_{j-1}, p_{j+1},...,p_N]
        
        2. Reversal: 反转一段子路径
           p' = [p_1,...,p_{i-1}, p_j, p_{j-1},...,p_{i+1}, p_i, p_{j+1},...,p_N]
    
    下降法:
        从随机初始路径出发，重复应用邻域操作，
        只接受降低cost的修改，直到局部最优。

在NLP语义嵌入中的应用:
    将语义向量视为"城市"，语义相似度（或距离）作为路径代价，
    寻找遍历所有语义中心的最优路径，用于:
        - 语义漫游和导航
        - 文档摘要的句子排序
        - 知识图谱的最优遍历
"""

import numpy as np


class SemanticPathOptimizer:
    """
    语义嵌入空间路径优化器（TSP下降法）。
    """

    def __init__(self, distance_matrix: np.ndarray, seed: int = 42):
        """
        初始化路径优化器。
        
        Parameters
        ----------
        distance_matrix : np.ndarray
            N x N 对称距离矩阵，对角线为零。
        seed : int
            随机种子。
        """
        self.D = np.asarray(distance_matrix, dtype=float)
        self.n = self.D.shape[0]

        if self.D.shape[0] != self.D.shape[1]:
            raise ValueError("distance_matrix must be square")
        if self.n < 4:
            raise ValueError(f"TSP requires at least 4 cities, got {self.n}")
        if np.max(np.abs(np.diag(self.D))) > 1e-12:
            raise ValueError("distance_matrix diagonal must be zero")
        if np.max(np.abs(self.D - self.D.T)) > 1e-12:
            raise ValueError("distance_matrix must be symmetric")

        self.rng = np.random.default_rng(seed)

    def path_cost(self, p: np.ndarray) -> float:
        """
        计算路径代价。
        
        cost = sum_{i=1}^{n} D(p_i, p_{i+1})
        其中 p_{n+1} = p_1
        """
        cost = 0.0
        for i in range(self.n):
            j = (i + 1) % self.n
            cost += self.D[p[i], p[j]]
        return cost

    def transpose_move(self, p: np.ndarray, i1: int, i2: int) -> np.ndarray:
        """
        换位操作：将 p[i2] 插入到 p[i1] 之后。
        
        要求: i1 + 1 < i2
        """
        p2 = np.concatenate([
            p[:i1 + 1],
            [p[i2]],
            p[i1 + 1:i2],
            p[i2 + 1:]
        ])
        return p2

    def reversal_move(self, p: np.ndarray, i1: int, i2: int) -> np.ndarray:
        """
        反转操作：反转 p[i1:i2+1]。
        """
        if i1 == 0:
            reversed_part = p[i2::-1]
        else:
            reversed_part = p[i2:i1 - 1:-1]
        p2 = np.concatenate([
            p[:i1],
            reversed_part,
            p[i2 + 1:]
        ])
        return p2

    def optimize(self, max_variations: int = 2000,
                 early_stop_rounds: int = 500) -> dict:
        """
        使用下降法优化路径。
        
        Parameters
        ----------
        max_variations : int
            最大变异尝试次数。
        early_stop_rounds : int
            连续多少次没有改进后提前停止。
            
        Returns
        -------
        dict
            包含最优路径、代价、统计信息的结果。
        """
        # 随机初始路径
        p = self.rng.permutation(self.n)
        cost = self.path_cost(p)

        transpose_count = 0
        reversal_count = 0
        transpose_accepted = 0
        reversal_accepted = 0
        no_improve_count = 0

        for _ in range(max_variations):
            improved = False

            # 尝试换位
            for _attempt in range(20):
                c = self.rng.choice(self.n, size=2, replace=False)
                c = np.sort(c)
                i1, i2 = int(c[0]), int(c[1])
                if i1 + 1 < i2:
                    break
            else:
                continue

            transpose_count += 1
            p2 = self.transpose_move(p, i1, i2)
            cost2 = self.path_cost(p2)
            if cost2 < cost - 1e-12:
                p = p2
                cost = cost2
                transpose_accepted += 1
                improved = True

            # 尝试反转
            c = self.rng.choice(self.n, size=2, replace=False)
            c = np.sort(c)
            i1, i2 = int(c[0]), int(c[1])

            reversal_count += 1
            p2 = self.reversal_move(p, i1, i2)
            cost2 = self.path_cost(p2)
            if cost2 < cost - 1e-12:
                p = p2
                cost = cost2
                reversal_accepted += 1
                improved = True

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_rounds:
                    break

        return {
            'path': p,
            'cost': cost,
            'transpose_count': transpose_count,
            'transpose_accepted': transpose_accepted,
            'reversal_count': reversal_count,
            'reversal_accepted': reversal_accepted,
            'iterations': transpose_count + reversal_count
        }

    def multi_start_optimize(self, num_starts: int = 10,
                             max_variations: int = 2000) -> dict:
        """
        多起点优化，取最好结果。
        """
        best_result = None
        best_cost = float('inf')

        for start in range(num_starts):
            result = self.optimize(max_variations=max_variations,
                                   early_stop_rounds=500)
            if result['cost'] < best_cost:
                best_cost = result['cost']
                best_result = result

        return best_result


def generate_semantic_distance_matrix(n: int, dim: int = 10,
                                      seed: int = 42) -> np.ndarray:
    """
    生成语义距离矩阵。
    
    基于高维语义向量的欧氏距离。
    """
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim))
    # 归一化到单位球面
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    embeddings = embeddings / norms

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # 使用欧氏距离
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            D[i, j] = dist
            D[j, i] = dist

    return D


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("语义嵌入TSP路径优化演示")
    print("=" * 60)

    n = 12
    D = generate_semantic_distance_matrix(n, dim=8, seed=42)
    print(f"\n语义节点数: {n}")
    print(f"距离矩阵范围: [{D.min():.4f}, {D.max():.4f}]")

    optimizer = SemanticPathOptimizer(D, seed=42)

    # 随机路径代价
    random_path = np.arange(n)
    random_cost = optimizer.path_cost(random_path)
    print(f"\n顺序路径代价: {random_cost:.4f}")

    # 单起点优化
    result = optimizer.optimize(max_variations=5000, early_stop_rounds=1000)
    print(f"\n优化后路径代价: {result['cost']:.4f}")
    print(f"改进比例: {(random_cost - result['cost']) / random_cost * 100:.2f}%")
    print(f"换位尝试/接受: {result['transpose_count']}/{result['transpose_accepted']}")
    print(f"反转尝试/接受: {result['reversal_count']}/{result['reversal_accepted']}")
    print(f"最优路径: {result['path']}")

    # 多起点优化
    result_ms = optimizer.multi_start_optimize(num_starts=5, max_variations=3000)
    print(f"\n多起点优化后代价: {result_ms['cost']:.4f}")
    print(f"相比单起点改进: {(result['cost'] - result_ms['cost']) / result['cost'] * 100:.2f}%")

    print("\n模块运行完成")
    return optimizer, result_ms


if __name__ == "__main__":
    demo()
