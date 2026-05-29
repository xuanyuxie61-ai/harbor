"""
structure_validation.py
=======================
膜蛋白结构的距离约束验证模块。

核心数学内容：
  - 部分消化问题（Partial Digest Problem, PDP）：
    给定一组距离 multiset $D = \{|x_i - x_j| : 1 \le i < j \le n\}$，
    重构原始点集 $X = \{x_1, \dots, x_n\}$。
  - 在膜蛋白结构中，此问题用于验证 C-alpha 骨架的距离约束：
    给定实验测得的残基间距离（如 NMR NOE 或 FRET 数据），
    验证是否存在满足所有约束的三维排布。
  - 成对距离矩阵的鲁棒性检查：对称性、三角不等式、度量空间性质。

种子项目映射：
  - 1225_test_partial_digest  →  部分消化测试问题生成
  - 459_ge_to_st             →  稀疏距离矩阵存储
"""

import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# 部分消化问题生成（种子项目 1225_test_partial_digest）
# ---------------------------------------------------------------------------
def test_partial_digest(k: int, dmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一个部分消化问题的测试实例。

    数学定义：
      随机选择 $k$ 个整数位置 $X \subset \{0, 1, \dots, d_{\max}\}$，
      且 $0 \in X$, $d_{\max} \in X$。
      计算所有成对距离 $D = \{|x_i - x_j| : i < j\}$。

    参数边界：
        k    >= 2
        dmax >= k - 1
    """
    if k < 2:
        raise ValueError("test_partial_digest: k must be >= 2.")
    if dmax < k - 1:
        raise ValueError("test_partial_digest: dmax must be >= k - 1.")

    # 从 {1, ..., dmax-1} 中随机选取 k-2 个位置
    if dmax - 1 >= k - 2:
        interior = np.random.choice(np.arange(1, dmax), size=k - 2, replace=False)
    else:
        interior = np.array([], dtype=int)

    locate = np.sort(np.concatenate(([0], interior, [dmax])))
    d = i4vec_distances(k, locate)
    return locate, d


def i4vec_distances(n: int, locate: np.ndarray) -> np.ndarray:
    """
    计算 n 个点之间的所有 $n(n-1)/2$ 个成对距离。

    参数边界：
        n >= 2
        locate 长度 == n
    """
    if n < 2:
        raise ValueError("i4vec_distances: n must be >= 2.")
    if locate.shape[0] != n:
        raise ValueError("i4vec_distances: locate length must equal n.")

    nd = n * (n - 1) // 2
    d = np.zeros(nd, dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d[idx] = abs(int(locate[j]) - int(locate[i]))
            idx += 1

    return d


def ksub_random(n: int, k: int) -> np.ndarray:
    """
    从 {1, ..., n} 中随机选取 k 个不同元素。

    参数边界：
        n >= k >= 0
    """
    if n < k:
        raise ValueError("ksub_random: n must be >= k.")
    if k < 0:
        raise ValueError("ksub_random: k must be >= 0.")
    if k == 0:
        return np.array([], dtype=int)
    return np.random.choice(np.arange(1, n + 1), size=k, replace=False)


# ---------------------------------------------------------------------------
# 距离矩阵验证
# ---------------------------------------------------------------------------
def validate_distance_matrix(
    dist: np.ndarray,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-10,
) -> dict:
    """
    验证距离矩阵是否满足度量空间的基本性质。

    检查项目：
      1. 方阵性
      2. 非负性
      3. 零对角线
      4. 对称性
      5. 三角不等式：$d_{ij} \le d_{ik} + d_{kj}$

    参数边界：
        dist : shape (n, n) 的距离矩阵
    """
    if dist.ndim != 2:
        raise ValueError("validate_distance_matrix: dist must be 2D.")
    n = dist.shape[0]
    if dist.shape[1] != n:
        raise ValueError("validate_distance_matrix: dist must be square.")

    results = {
        "is_square": True,
        "is_nonnegative": True,
        "zero_diagonal": True,
        "is_symmetric": True,
        "triangle_inequality": True,
        "violations": [],
    }

    # 非负性
    if np.any(dist < -atol):
        results["is_nonnegative"] = False
        results["violations"].append("Negative distance found.")

    # 零对角线
    diag_norm = np.linalg.norm(np.diag(dist))
    if diag_norm > atol:
        results["zero_diagonal"] = False
        results["violations"].append(f"Non-zero diagonal: norm={diag_norm:.3e}")

    # 对称性
    asym_norm = np.linalg.norm(dist - dist.T)
    if asym_norm > atol:
        results["is_symmetric"] = False
        results["violations"].append(f"Asymmetric matrix: norm={asym_norm:.3e}")

    # 三角不等式
    max_violation = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j] + atol:
                    violation = dist[i, j] - (dist[i, k] + dist[k, j])
                    max_violation = max(max_violation, violation)
    if max_violation > atol:
        results["triangle_inequality"] = False
        results["violations"].append(f"Triangle inequality violated by {max_violation:.3e}")

    return results


# ---------------------------------------------------------------------------
# 蛋白骨架距离约束验证
# ---------------------------------------------------------------------------
def validate_backbone_distances(
    ca_coords: np.ndarray,
    expected_distances: np.ndarray,
    expected_pairs: List[Tuple[int, int]],
    tolerance: float = 1.5,  # Å
) -> dict:
    """
    验证 C-alpha 骨架坐标是否满足给定的距离约束。

    参数：
        ca_coords         : shape (n_residues, 3) 的 C-alpha 坐标
        expected_distances: shape (m,) 的期望距离
        expected_pairs    : m 对残基索引 (i, j)
        tolerance         : 允许误差（Å）

    返回：
        results 字典，包含通过/失败状态及统计信息。
    """
    if ca_coords.ndim != 2 or ca_coords.shape[1] != 3:
        raise ValueError("validate_backbone_distances: ca_coords must be (n, 3).")
    n_res = ca_coords.shape[0]
    m = len(expected_pairs)
    if expected_distances.shape[0] != m:
        raise ValueError("validate_backbone_distances: expected_distances length must match pairs.")

    computed = np.zeros(m, dtype=float)
    passed = np.zeros(m, dtype=bool)

    for idx, (i, j) in enumerate(expected_pairs):
        if not (0 <= i < n_res and 0 <= j < n_res):
            raise ValueError(f"validate_backbone_distances: residue index out of range at pair {idx}.")
        d = np.linalg.norm(ca_coords[i] - ca_coords[j])
        computed[idx] = d
        passed[idx] = abs(d - expected_distances[idx]) <= tolerance

    results = {
        "n_constraints": m,
        "n_passed": int(np.count_nonzero(passed)),
        "n_failed": int(m - np.count_nonzero(passed)),
        "pass_rate": float(np.mean(passed)),
        "max_error_A": float(np.max(np.abs(computed - expected_distances))),
        "rmsd_A": float(np.sqrt(np.mean((computed - expected_distances) ** 2))),
        "passed_mask": passed,
        "computed_distances": computed,
    }
    return results
