"""
sparse_uq.py
高维不确定度量化 (Uncertainty Quantification) 模块

科学背景:
海洋中尺度涡旋模拟存在大量参数不确定性:
  - β 参数 (Coriolis 梯度)
  - 耗散系数 ν
  - Rossby 变形半径 L_d
  - 强迫振幅和尺度

直接蒙特卡洛需要 O(10^4 - 10^6) 次模拟. 稀疏网格 (Sparse Grid)
通过 Smolyak 构造将高维积分复杂度从 O(N^d) 降至 O(N (log N)^{d-1}).

Smolyak 公式:
  A(q,d) = sum_{|i| = q} (\Delta_{i_1} \otimes ... \otimes \Delta_{i_d})
  其中 \Delta_i = Q_i - Q_{i-1} 为差分求积算子.

本模块实现:
  - 整数组合枚举 (comp_next)
  - 一维嵌套/非嵌套求积规则 (Clenshaw-Curtis, Newton-Cotes)
  - 多维稀疏网格点与权重生成
  - 基于稀疏网格的统计矩估计 (均值、方差、灵敏度)

融合来源:
- 1109_sparse_grid_total_poly: 稀疏网格构造、整数组合、Newton-Cotes
"""

import numpy as np
import math
from numerics_core import clenshaw_curtis_nodes_weights
from typing import Tuple, List, Dict, Generator


# ============================================================
# 1. 整数组合枚举 (from 1109_sparse_grid_total_poly)
# ============================================================

def comp_next(n: int, k: int) -> Generator[np.ndarray, None, None]:
    """
    枚举将 n 拆分为 k 个非负整数的所有组合.

    组合按字典序生成, 用于稀疏网格层级索引构造.

    Parameters
    ----------
    n : int
        待拆分的整数
    k : int
        部分数

    Yields
    ------
    np.ndarray
        长度为 k 的组合向量
    """
    if n < 0 or k < 1:
        return
    if k == 1:
        yield np.array([n])
        return

    # 递归生成
    def _recurse(remaining: int, parts: int, prefix: List[int]):
        if parts == 1:
            yield np.array(prefix + [remaining])
            return
        for val in range(remaining + 1):
            yield from _recurse(remaining - val, parts - 1, prefix + [val])

    yield from _recurse(n, k, [])


def comp_all(n: int, k: int) -> np.ndarray:
    """返回所有组合为一个数组."""
    return np.array(list(comp_next(n, k)))


# ============================================================
# 2. 一维嵌套求积规则
# ============================================================

def cc_nested_rule(level: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    嵌套 Clenshaw-Curtis 规则: 层级 l 的节点数为 2^l + 1 (l>=1), 1 (l=0).

    嵌套性质: 层级 l 的节点包含层级 l-1 的所有节点.
    这是 Smolyak 构造的关键.
    """
    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** level
    return clenshaw_curtis_nodes_weights(n)


def trapezoidal_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    复合梯形规则 (等距节点).
    用于 Newton-Cotes 型求积.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.linspace(-1.0, 1.0, n + 1)
    w = np.full(n + 1, 2.0 / n)
    w[0] = 1.0 / n
    w[-1] = 1.0 / n
    return x, w


def newton_cotes_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Newton-Cotes 闭型求积规则 (等距节点, n+1 点具有 n 次代数精度).

    权重通过 Lagrange 基函数积分计算:
      w_j = \int_{-1}^{1} l_j(x) dx
    其中 l_j(x) = prod_{m\neq j} (x - x_m) / (x_j - x_m).
    """
    if n < 1:
        return trapezoidal_rule(1)
    x = np.linspace(-1.0, 1.0, n + 1)
    w = np.zeros(n + 1)
    for j in range(n + 1):
        # 构造 Lagrange 基多项式在节点上的系数
        # 通过积分多项式系数
        coeff = np.zeros(n + 1)
        coeff[0] = 1.0
        for m in range(n + 1):
            if m == j:
                continue
            # 乘以 (x - x_m) / (x_j - x_m)
            p = np.array([1.0, -x[m]]) / (x[j] - x[m])
            coeff = np.convolve(coeff, p)
        # 积分多项式 coeff[0] + coeff[1]*x + ...
        # \int_{-1}^1 x^k dx = [1 - (-1)^{k+1}] / (k+1)
        for k in range(len(coeff)):
            w[j] += coeff[k] * (1.0 - (-1.0) ** (k + 1)) / (k + 1.0)
    return x, w


# ============================================================
# 3. 稀疏网格构造 (Smolyak)
# ============================================================

class SparseGrid:
    """
    d 维稀疏网格求积器.

    Smolyak 公式:
      A(q,d) = sum_{max(d, q-d+1) <= |l| <= q} (-1)^{q-|l|} C(d-1, q-|l|) (Q_{l_1} \otimes ... \otimes Q_{l_d})

    其中 q >= d 为总层级, l = (l_1, ..., l_d) 为各维层级.
    """

    def __init__(self, dim: int, level: int, rule: str = "cc"):
        self.dim = dim
        self.level = max(level, dim)
        self.rule = rule
        self.points = None
        self.weights = None
        self._construct()

    def _oned_rule(self, lev: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.rule == "cc":
            return cc_nested_rule(lev)
        elif self.rule == "nc":
            n = max(1, 2 ** lev)
            return newton_cotes_rule(n)
        elif self.rule == "trap":
            n = max(1, 2 ** lev)
            return trapezoidal_rule(n)
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

    def _construct(self):
        """构造稀疏网格点与权重 (Smolyak 公式)."""
        from itertools import product
        d = self.dim
        q = self.level

        all_points = []
        all_weights = []

        # 遍历所有满足 d <= |l| <= q 的层级组合
        for s in range(d, q + 1):
            for lev_vec in comp_next(s, d):
                # Smolyak 系数
                coeff = (-1) ** (q - s) * math.comb(d - 1, q - s)
                if coeff == 0:
                    continue

                # 各维节点与权重
                nodes_list = []
                weights_list = []
                for l in lev_vec:
                    x, w = self._oned_rule(int(l))
                    nodes_list.append(x)
                    weights_list.append(w)

                # 张量积
                for idx in product(*[range(len(nl)) for nl in nodes_list]):
                    pt = np.array([nodes_list[i][idx[i]] for i in range(d)])
                    w = coeff * np.prod([weights_list[i][idx[i]] for i in range(d)])
                    all_points.append(pt)
                    all_weights.append(w)

        if not all_points:
            self.points = np.zeros((0, d))
            self.weights = np.array([])
            return

        # 合并重复点
        pts = np.array(all_points)
        wts = np.array(all_weights)

        # 使用四舍五入合并
        unique_dict = {}
        for i in range(pts.shape[0]):
            key = tuple(np.round(pts[i], decimals=12))
            if key in unique_dict:
                unique_dict[key] += wts[i]
            else:
                unique_dict[key] = wts[i]

        self.points = np.array([np.array(k) for k in unique_dict.keys()])
        self.weights = np.array(list(unique_dict.values()))

    def integrate(self, f: callable) -> float:
        """对函数 f: R^d → R 求积分."""
        if self.points.shape[0] == 0:
            return 0.0
        vals = np.array([f(p) for p in self.points])
        return float(np.dot(self.weights, vals))

    def size(self) -> int:
        return self.points.shape[0]


# ============================================================
# 4. 参数化 UQ 分析器
# ============================================================

class ParameterizedUQ:
    """
    海洋 QG 模型参数的不确定度量化.

    参数空间 (d=4):
      p1 = β / β_0 ∈ [0.5, 2.0]
      p2 = ν / ν_0 ∈ [0.1, 10.0]
      p3 = L_d / L_{d0} ∈ [0.5, 2.0]
      p4 = forcing_amp / f_0 ∈ [0.0, 2.0]
    """

    def __init__(self, dim: int = 4, level: int = 3):
        self.dim = dim
        self.level = level
        self.grid = SparseGrid(dim, level, rule="cc")

    def map_to_physical(self, p_norm: np.ndarray) -> np.ndarray:
        """将 [-1,1]^d 映射到物理参数空间."""
        p = np.zeros(self.dim)
        # p0: beta in [0.5, 2.0]
        p[0] = 1.25 + 0.75 * p_norm[0]
        # p1: nu in [0.1, 10.0] (对数尺度)
        p[1] = 10.0 ** (-1.0 + 1.1 * (p_norm[1] + 1.0) / 2.0)
        # p2: Ld in [0.5, 2.0]
        p[2] = 1.25 + 0.75 * p_norm[2]
        # p3: forcing in [0.0, 2.0]
        p[3] = 1.0 + 1.0 * p_norm[3]
        return p

    def estimate_statistics(self, model_evaluator: callable) -> Dict[str, float]:
        """
        估计模型输出的统计量.

        model_evaluator(p_norm) → scalar
        """
        vals = []
        for pt in self.grid.points:
            try:
                v = float(model_evaluator(pt))
                if not np.isfinite(v):
                    v = np.nan
            except Exception:
                v = np.nan
            vals.append(v)
        vals = np.array(vals)

        w = self.grid.weights
        # 过滤 nan 值及其对应权重
        valid = np.isfinite(vals)
        if not np.any(valid):
            return {
                "mean": np.nan, "variance": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan, "grid_size": int(self.grid.size())
            }

        vals_valid = vals[valid]
        w_valid = w[valid]
        w_sum = np.sum(w_valid)
        if abs(w_sum) < 1e-15:
            w_sum = 1.0

        mean = np.dot(w_valid, vals_valid) / w_sum
        variance = np.dot(w_valid, (vals_valid - mean) ** 2) / w_sum

        return {
            "mean": float(mean),
            "variance": float(variance),
            "std": float(np.sqrt(max(variance, 0.0))),
            "min": float(np.min(vals_valid)),
            "max": float(np.max(vals_valid)),
            "grid_size": int(self.grid.size())
        }


if __name__ == "__main__":
    # 测试稀疏网格
    sg = SparseGrid(dim=2, level=3, rule="cc")
    print(f"Sparse grid 2D level 3: {sg.size()} points")

    # 测试积分
    f_test = lambda p: np.exp(-(p[0]**2 + p[1]**2))
    val = sg.integrate(f_test)
    print("Integral approx:", val)

    # 测试组合枚举
    combos = comp_all(4, 3)
    print("Combinations of 4 into 3 parts:", len(combos))
