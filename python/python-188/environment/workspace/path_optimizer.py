
import numpy as np


class SemanticPathOptimizer:

    def __init__(self, distance_matrix: np.ndarray, seed: int = 42):
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
        cost = 0.0
        for i in range(self.n):
            j = (i + 1) % self.n
            cost += self.D[p[i], p[j]]
        return cost

    def transpose_move(self, p: np.ndarray, i1: int, i2: int) -> np.ndarray:
        p2 = np.concatenate([
            p[:i1 + 1],
            [p[i2]],
            p[i1 + 1:i2],
            p[i2 + 1:]
        ])
        return p2

    def reversal_move(self, p: np.ndarray, i1: int, i2: int) -> np.ndarray:
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

        p = self.rng.permutation(self.n)
        cost = self.path_cost(p)

        transpose_count = 0
        reversal_count = 0
        transpose_accepted = 0
        reversal_accepted = 0
        no_improve_count = 0

        for _ in range(max_variations):
            improved = False


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
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, dim))

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    embeddings = embeddings / norms

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):

            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            D[i, j] = dist
            D[j, i] = dist

    return D


def demo():
    print("=" * 60)
    print("语义嵌入TSP路径优化演示")
    print("=" * 60)

    n = 12
    D = generate_semantic_distance_matrix(n, dim=8, seed=42)
    print(f"\n语义节点数: {n}")
    print(f"距离矩阵范围: [{D.min():.4f}, {D.max():.4f}]")

    optimizer = SemanticPathOptimizer(D, seed=42)


    random_path = np.arange(n)
    random_cost = optimizer.path_cost(random_path)
    print(f"\n顺序路径代价: {random_cost:.4f}")


    result = optimizer.optimize(max_variations=5000, early_stop_rounds=1000)
    print(f"\n优化后路径代价: {result['cost']:.4f}")
    print(f"改进比例: {(random_cost - result['cost']) / random_cost * 100:.2f}%")
    print(f"换位尝试/接受: {result['transpose_count']}/{result['transpose_accepted']}")
    print(f"反转尝试/接受: {result['reversal_count']}/{result['reversal_accepted']}")
    print(f"最优路径: {result['path']}")


    result_ms = optimizer.multi_start_optimize(num_starts=5, max_variations=3000)
    print(f"\n多起点优化后代价: {result_ms['cost']:.4f}")
    print(f"相比单起点改进: {(result['cost'] - result_ms['cost']) / result['cost'] * 100:.2f}%")

    print("\n模块运行完成")
    return optimizer, result_ms


if __name__ == "__main__":
    demo()
