"""
stability.py
============

有限差分稳定性分析 —— 数值诊断核心。

种子项目 658 (Lebesgue constant) 与 159 (chebyshev interpolation) 启发:
利用 Lebesgue 常数和条件数分析有限差分算子的数值稳定性。

1. Lebesgue 常数 (seed 658):
    Λ_n(x) = Σ_j |l_j(x)|
    其中 l_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k) 为 Lagrange 基函数

    对等距节点: Λ_n ~ 2^{n+1} / (e · n · ln n)  → 指数增长, 不稳定
    对 Chebyshev 节点: Λ_n ~ (2/π) ln n + O(1)  → 对数增长, 稳定

2. 差分算子条件数 (矩阵形式):
    D_h = W · diag(1/h^m)
    κ(D_h) = ||D_h|| · ||D_h^{-1}||

    对 m 阶导数、n 节点: 条件数 ~ O(h^{-m})

3. 截断误差 vs 舍入误差平衡:
    总误差 E(h) = C_T h^p + C_R ε / h^m
    最优步长 h* = (m C_R ε / (p C_T))^{1/(p+m)}

4. 稳定性区域图:
    扫描 h ∈ [h_min, h_max], 绘制 |D_h f - D_exact| 随 h 变化,
    识别 "稳定平台" 区域。

5. 数值稳定性诊断指标:
    - 相对误差: |D_h f - D_exact| / |D_exact|
    - 放大因子: ||D_h||_∞ = Σ |w_j| / h^m
    - 条件数: κ = h^m · Σ|w_j| · h^{-m} = Σ|w_j| (与 h 无关!)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, List, Optional, Dict

from finite_diff import FiniteDiff, fornberg_weights


# ---------------------------------------------------------------------------
# Lebesgue 常数计算 (seed 658)
# ---------------------------------------------------------------------------
def lagrange_basis(
    nodes: np.ndarray, j: int, x: float
) -> float:
    """
    Lagrange 基函数:
        l_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)
    """
    n = len(nodes)
    result = 1.0
    for k in range(n):
        if k != j:
            denom = nodes[j] - nodes[k]
            if abs(denom) < 1e-15:
                raise ValueError(f"节点重合: nodes[{j}]={nodes[j]}, nodes[{k}]={nodes[k]}")
            result *= (x - nodes[k]) / denom
    return result


def lebesgue_function(
    nodes: np.ndarray, x: float
) -> float:
    """
    Lebesgue 函数: Λ(x) = Σ_j |l_j(x)|
    """
    n = len(nodes)
    return sum(abs(lagrange_basis(nodes, j, x)) for j in range(n))


def lebesgue_constant(
    nodes: np.ndarray, n_eval: int = 10000
) -> float:
    """
    Lebesgue 常数: Λ_n = max_x Λ(x), x ∈ [-1, 1]

    通过在密集网格上采样 Lebesgue 函数并取最大值估计。

    Parameters
    ----------
    nodes : ndarray
        插值节点
    n_eval : int
        评估点数

    Returns
    -------
    float
        Lebesgue 常数
    """
    x_grid = np.linspace(-1.0, 1.0, n_eval)
    values = np.array([lebesgue_function(nodes, x) for x in x_grid])
    return float(np.max(values))


def chebyshev_nodes(n: int, kind: int = 2) -> np.ndarray:
    """
    Chebyshev 节点 (seed 159, 658):

    Kind 1 (零点):
        x_k = cos((2k - 1) π / (2n)), k = 1, ..., n

    Kind 2 (极值点):
        x_k = cos(k π / (n - 1)), k = 0, ..., n - 1
    """
    if n < 1:
        raise ValueError(f"节点数必须 ≥ 1: n={n}")
    if kind == 1:
        k = np.arange(1, n + 1)
        return np.cos((2 * k - 1) * np.pi / (2 * n))
    elif kind == 2:
        k = np.arange(n)
        return np.cos(k * np.pi / max(n - 1, 1))
    else:
        raise ValueError(f"不支持的 Chebyshev 类型: kind={kind}")


def equidistant_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """等距节点。"""
    return np.linspace(a, b, n)


# ---------------------------------------------------------------------------
# 差分算子条件数
# ---------------------------------------------------------------------------
def fd_operator_norm(
    weights: np.ndarray, h: float, m: int
) -> float:
    """
    有限差分算子 ∞-范数:

        ||D_h||_∞ = Σ_j |w_j| / h^m

    控制函数扰动对导数近似的最大放大。
    """
    if h <= 0.0:
        raise ValueError(f"步长必须为正: h={h}")
    return float(np.sum(np.abs(weights)) / h**m)


def fd_condition_number(
    weights: np.ndarray, m: int
) -> float:
    """
    有限差分权重条件数 (与 h 无关):

        κ(W) = Σ_j |w_j|

    该值衡量差分模板的固有数值稳定性:
    - 小 κ → 稳定 (如中心差分 κ ~ O(1))
    - 大 κ → 不稳定 (如高阶向前差分 κ ~ 2^n)
    """
    return float(np.sum(np.abs(weights)))


# ---------------------------------------------------------------------------
# 稳定性扫描
# ---------------------------------------------------------------------------
class StabilityAnalyzer:
    """
    有限差分稳定性分析器。

    扫描步长 h, 计算:
    1. 相对误差 |D_h f - D_exact| / |D_exact|
    2. 算子范数 ||D_h||
    3. 舍入误差估计 ε_mach · ||D_h||

    诊断 "稳定平台" 区域: 误差随 h 变化最小的区间。
    """

    def __init__(
        self,
        fd_order: int = 4,
        deriv_order: int = 1,
    ):
        self.fd = FiniteDiff(m=deriv_order, fd_order=fd_order)
        self.m = deriv_order
        self.fd_order = fd_order
        self.results: Dict[str, List[float]] = {
            "h": [],
            "approx": [],
            "abs_error": [],
            "rel_error": [],
            "op_norm": [],
            "roundoff_bound": [],
        }

    def scan(
        self,
        f: Callable[[float], float],
        x: float,
        exact_deriv: float,
        h_values: Optional[np.ndarray] = None,
        eps_mach: float = 2.22e-16,
    ) -> Dict[str, np.ndarray]:
        """
        扫描步长 h, 收集稳定性诊断数据。

        Parameters
        ----------
        f : callable
        x : float
            求导点
        exact_deriv : float
            精确导数值 (用于计算误差)
        h_values : ndarray, optional
            步长扫描值 (默认对数均匀)
        eps_mach : float
            机器精度

        Returns
        -------
        dict
            各诊断量随 h 的变化
        """
        if h_values is None:
            h_values = np.logspace(-15, 0, 61)

        self.results = {k: [] for k in self.results}

        for h in h_values:
            approx = self.fd(f, x, h)
            abs_err = abs(approx - exact_deriv)
            rel_err = (
                abs_err / abs(exact_deriv)
                if abs(exact_deriv) > 1e-300
                else abs_err
            )
            # 算子范数
            op_norm = fd_operator_norm(self.fd.weights, h, self.m)
            # 舍入误差界
            roundoff = eps_mach * op_norm * max(abs(f(x)), 1.0)

            self.results["h"].append(h)
            self.results["approx"].append(approx)
            self.results["abs_error"].append(abs_err)
            self.results["rel_error"].append(rel_err)
            self.results["op_norm"].append(op_norm)
            self.results["roundoff_bound"].append(roundoff)

        # 转为 ndarray
        return {k: np.array(v) for k, v in self.results.items()}

    def find_stable_region(
        self, rel_error: np.ndarray, h_values: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        识别稳定平台区域: 相对误差最小的 decade。

        Returns
        -------
        (h_opt, min_rel_error, stable_width_decade)
        """
        # 在对数空间中寻找最小误差附近 ±1 decade 的平台
        idx_min = int(np.argmin(rel_error))
        h_opt = h_values[idx_min]
        min_err = rel_error[idx_min]

        # 向两侧扩展: 误差 < 10 × min_err 的区间
        log_h = np.log10(h_values)
        mask = rel_error < max(10.0 * min_err, 1e-14)
        if np.any(mask):
            log_h_stable = log_h[mask]
            width = float(log_h_stable[-1] - log_h_stable[0])
        else:
            width = 0.0

        return float(h_opt), float(min_err), width

    def estimate_optimal_h(
        self, f: Callable[[float], float], x: float, p: int = 4
    ) -> float:
        """
        理论最优步长:

            h* ≈ (ε_mach · m! / |f^{(m+p)}(x)|)^{1/(p+m)} · C

        简化估计 (不依赖高阶导数):
            h* ≈ ε_mach^{1/(p+m)}

        对 p=4, m=1: h* ≈ ε^{1/5} ≈ 6.5e-4
        """
        eps = 2.22e-16
        m = self.m
        h_opt = eps ** (1.0 / (p + m))
        return float(h_opt)


# ---------------------------------------------------------------------------
# 数值稳定性综合评分
# ---------------------------------------------------------------------------
def stability_score(
    rel_errors: np.ndarray, h_values: np.ndarray
) -> float:
    """
    稳定性综合评分 (0-100, 越高越稳定):

        score = 100 · (1 - min_rel_error / max_rel_error) · (stable_width / total_width)

    衡量:
    1. 最小误差的绝对水平
    2. 稳定平台的宽度 (对数尺度)
    """
    if len(rel_errors) < 2:
        return 0.0
    min_err = np.min(rel_errors[rel_errors > 0]) if np.any(rel_errors > 0) else 1e-16
    max_err = np.max(rel_errors)
    if max_err < 1e-300:
        return 100.0

    log_h = np.log10(h_values)
    total_width = log_h[-1] - log_h[0]
    threshold = max(10.0 * min_err, 1e-14)
    mask = rel_errors < threshold
    if np.any(mask):
        stable_width = log_h[mask][-1] - log_h[mask][0]
    else:
        stable_width = 0.0

    score = 100.0 * (1.0 - min_err / max_err) * (stable_width / total_width)
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    # Lebesgue 常数比较
    n = 10
    nodes_eq = equidistant_nodes(n)
    nodes_ch = chebyshev_nodes(n, kind=2)
    lam_eq = lebesgue_constant(nodes_eq)
    lam_ch = lebesgue_constant(nodes_ch)
    print(f"Lebesgue 常数 (n={n}):")
    print(f"  等距节点: Λ = {lam_eq:.2f}")
    print(f"  Chebyshev 节点: Λ = {lam_ch:.2f}")
    print(f"  比值: {lam_eq / lam_ch:.1f}")

    # 稳定性扫描
    f = lambda x: math.sin(x)
    x0 = 0.5
    exact = math.cos(x0)
    sa = StabilityAnalyzer(fd_order=4, deriv_order=1)
    results = sa.scan(f, x0, exact)
    h_opt, min_err, width = sa.find_stable_region(
        results["rel_error"], results["h"]
    )
    score = stability_score(results["rel_error"], results["h"])
    print(f"\n稳定性分析:")
    print(f"  最优 h = {h_opt:.2e}")
    print(f"  最小相对误差 = {min_err:.2e}")
    print(f"  稳定平台宽度 = {width:.1f} decades")
    print(f"  稳定性评分 = {score:.1f}/100")

    # 条件数
    fd4 = FiniteDiff(m=1, fd_order=4)
    kappa = fd_condition_number(fd4.weights, 1)
    print(f"\n4阶中心差分条件数: κ = {kappa:.4f}")
