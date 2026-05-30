
import numpy as np
import warnings


class RobustNumericUtils:

    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        if abs(b) < 1e-15:
            return float(default)
        return float(a) / float(b)

    @staticmethod
    def safe_log(x: float, default: float = -700.0) -> float:
        if x <= 0.0:
            return float(default)
        return float(np.log(x))

    @staticmethod
    def safe_sqrt(x: float, default: float = 0.0) -> float:
        if x < 0.0:
            return float(default)
        return float(np.sqrt(x))

    @staticmethod
    def clip_array(arr: np.ndarray, min_val: float = None,
                   max_val: float = None) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if min_val is not None and max_val is not None:
            if min_val > max_val:
                raise ValueError(f"min_val ({min_val}) must not exceed max_val ({max_val})")
        return np.clip(arr, min_val, max_val)

    @staticmethod
    def normalize_vector(v: np.ndarray, ord: int = 2,
                         default: np.ndarray = None) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        norm = np.linalg.norm(v, ord=ord)
        if norm < 1e-15:
            if default is not None:
                return np.asarray(default, dtype=float)
            return np.zeros_like(v)
        return v / norm

    @staticmethod
    def check_finite(arr: np.ndarray, name: str = "array") -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(arr)):
            n_nan = np.sum(np.isnan(arr))
            n_inf = np.sum(np.isinf(arr))
            raise ValueError(
                f"{name} contains {n_nan} NaN and {n_inf} Inf values"
            )
        return arr

    @staticmethod
    def condition_number_safe(A: np.ndarray, threshold: float = 1e12) -> dict:
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix")

        try:
            cond = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            return {'condition_number': float('inf'), 'is_well_conditioned': False}

        is_well = cond < threshold
        if not is_well:
            warnings.warn(f"Matrix condition number {cond:.2e} exceeds threshold {threshold:.2e}")

        return {
            'condition_number': cond,
            'is_well_conditioned': is_well
        }

    @staticmethod
    def solve_linear_system_safe(A: np.ndarray, b: np.ndarray,
                                  regularization: float = 1e-10) -> np.ndarray:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        try:
            x = np.linalg.solve(A, b)
            return x
        except np.linalg.LinAlgError:

            n = A.shape[0]
            A_reg = A + regularization * np.eye(n)
            x = np.linalg.solve(A_reg, b)
            warnings.warn(f"Matrix was singular; used regularization {regularization}")
            return x

    @staticmethod
    def semantic_similarity_safe(u: np.ndarray, v: np.ndarray) -> float:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        if len(u) != len(v):
            raise ValueError(f"u and v must have same length, got {len(u)} and {len(v)}")

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u < 1e-15 or norm_v < 1e-15:
            return 0.0

        dot = np.dot(u, v)
        return float(dot / (norm_u * norm_v))

    @staticmethod
    def batch_semantic_similarity(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=float)
        n = embeddings.shape[0]
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = RobustNumericUtils.semantic_similarity_safe(
                    embeddings[i], embeddings[j]
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix


def demo():
    print("=" * 60)
    print("数值鲁棒性工具演示")
    print("=" * 60)

    utils = RobustNumericUtils()


    print("\n--- 安全除法 ---")
    print(f"safe_divide(1.0, 2.0) = {utils.safe_divide(1.0, 2.0)}")
    print(f"safe_divide(1.0, 0.0) = {utils.safe_divide(1.0, 0.0)}")
    print(f"safe_divide(1, 2) = {utils.safe_divide(1, 2)}")


    print("\n--- 安全对数 ---")
    print(f"safe_log(2.0) = {utils.safe_log(2.0):.6f}")
    print(f"safe_log(0.0) = {utils.safe_log(0.0)}")
    print(f"safe_log(-1.0) = {utils.safe_log(-1.0)}")


    print("\n--- 安全归一化 ---")
    v1 = np.array([3.0, 4.0])
    v2 = np.array([0.0, 0.0])
    print(f"normalize({v1}) = {utils.normalize_vector(v1)}")
    print(f"normalize({v2}) = {utils.normalize_vector(v2)}")


    print("\n--- 语义相似度 ---")
    e1 = np.array([1.0, 0.0, 1.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([1.0, 0.0, 1.0])
    print(f"sim(e1, e2) = {utils.semantic_similarity_safe(e1, e2):.6f}")
    print(f"sim(e1, e3) = {utils.semantic_similarity_safe(e1, e3):.6f}")


    print("\n--- 批量语义相似度 ---")
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float)
    sim_mat = utils.batch_semantic_similarity(embeddings)
    print(f"相似度矩阵:\n{sim_mat}")


    print("\n--- 病态矩阵处理 ---")
    A = np.array([[1.0, 1.0], [1.0, 1.0000001]])
    b = np.array([2.0, 2.0])
    cond_info = utils.condition_number_safe(A)
    print(f"条件数: {cond_info['condition_number']:.2e}")
    print(f"良态: {cond_info['is_well_conditioned']}")

    x = utils.solve_linear_system_safe(A, b)
    print(f"解: {x}")
    print(f"残差: {np.linalg.norm(A @ x - b):.2e}")

    print("\n模块运行完成")
    return utils


if __name__ == "__main__":
    demo()
