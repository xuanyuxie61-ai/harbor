"""
pbpk_interpolation.py
基于种子项目 1213_test_interp_fun

实现 1D 插值测试函数库，用于构建 PBPK 中的非线性浓度-效应关系：
1. Runge 函数（等距插值发散）
2. Bernstein 例子 |x|（收敛缓慢）
3. 高振荡函数
4. 分段复合函数

在 PBPK 模型中用于：
- 药物浓度-效应（PD）关系的非线性插值
- 药时曲线的稀疏数据重构
- 剂量-反应曲线的分段建模
"""

import numpy as np
from typing import Callable, Tuple

# ---------------------------------------------------------------------------
# 经典插值测试函数
# ---------------------------------------------------------------------------

def runge_function(x: float) -> float:
    """
    Runge 函数：f(x) = 1 / (1 + x^2)。
    等距节点上的多项式插值在 |x| > ~0.726 时发散（Runge 现象）。
    在 PBPK 中用于测试浓度-效应外推的稳定性。
    """
    return 1.0 / (1.0 + x * x)


def bernstein_example(x: float) -> float:
    """
    Bernstein 例子：f(x) = |x|。
    在 [-1,1] 上等距节点插值仅在 x = -1, 0, 1 处收敛。
    用于测试 PD 关系中非光滑转折点的处理。
    """
    return abs(x)


def step_function(x: float, x0: float = 0.0) -> float:
    """
    阶跃函数：Heaviside step。
    在 PBPK 中模拟药物浓度达到阈值后的开关效应。
    """
    return 1.0 if x >= x0 else 0.0


def oscillatory_function(x: float) -> float:
    """
    高振荡函数：f(x) = sqrt(x(1-x)) * sin(2.1π / (x + 0.05))。
    在 x → 0 时无限振荡，测试插值方法对快速 PK 变化的适应性。
    """
    if x < 0.0 or x > 1.0:
        return 0.0
    val = np.sqrt(max(x * (1.0 - x), 0.0)) * np.sin(2.1 * np.pi / (x + 0.05))
    return val


def piecewise_composite(x: float) -> float:
    """
    分段复合函数（用于测试多相药代动力学）。
    在 x ∈ [0,5] 和 x ∈ [5,10] 上具有不同动力学特征。
    """
    if x < 0.0:
        return 0.0
    elif x <= 5.0:
        return max(np.sin(x) + np.sin(x * x), 0.0)
    else:
        return max(1.0 - abs(x - 5.0) / 5.0, 0.0)


# ---------------------------------------------------------------------------
# Lagrange 插值（等距 & Chebyshev）
# ---------------------------------------------------------------------------

def lagrange_interpolate(x_nodes: np.ndarray, y_nodes: np.ndarray,
                          x_eval: np.ndarray) -> np.ndarray:
    """
    Lagrange 多项式插值。
    警告：等距节点下高阶插值数值不稳定。
    """
    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have same length")
    n = len(x_nodes)
    result = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        Li = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i != j:
                diff = x_nodes[i] - x_nodes[j]
                if abs(diff) < 1e-15:
                    raise ValueError("Duplicate nodes")
                Li *= (x_eval - x_nodes[j]) / diff
        result += y_nodes[i] * Li
    return result


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Chebyshev 节点：x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)π/(2n))。
    最小化 Runge 现象，适合光滑函数插值。
    """
    if n < 1:
        raise ValueError("n must be positive")
    k = np.arange(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))
    return x


# ---------------------------------------------------------------------------
# 分段线性插值（鲁棒且常用）
# ---------------------------------------------------------------------------

def piecewise_linear_interpolate(x_nodes: np.ndarray, y_nodes: np.ndarray,
                                  x_eval: np.ndarray) -> np.ndarray:
    """
    分段线性插值。在 PBPK 中最常用（非光滑但鲁棒）。
    """
    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have same length")
    if len(x_nodes) < 2:
        raise ValueError("Need at least 2 nodes")
    sorted_idx = np.argsort(x_nodes)
    x_s = x_nodes[sorted_idx]
    y_s = y_nodes[sorted_idx]
    result = np.empty_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        if x <= x_s[0]:
            result[i] = y_s[0]
        elif x >= x_s[-1]:
            result[i] = y_s[-1]
        else:
            idx = np.searchsorted(x_s, x) - 1
            idx = max(0, min(idx, len(x_s) - 2))
            dx = x_s[idx + 1] - x_s[idx]
            if abs(dx) < 1e-15:
                result[i] = y_s[idx]
            else:
                t = (x - x_s[idx]) / dx
                result[i] = y_s[idx] * (1.0 - t) + y_s[idx + 1] * t
    return result


# ---------------------------------------------------------------------------
# PBPK 浓度-效应（PD）插值模型
# ---------------------------------------------------------------------------

def pd_effect_interpolate(concentration: float,
                           C_nodes: np.ndarray,
                           E_nodes: np.ndarray,
                           method: str = "linear") -> float:
    """
    根据离散的药效数据点 (C_i, E_i) 插值计算效应值 E(C)。
    支持 'linear' 和 'lagrange'（仅限低阶）。
    """
    if concentration < 0.0:
        raise ValueError("Concentration must be non-negative")
    if len(C_nodes) != len(E_nodes):
        raise ValueError("Node arrays must have same length")
    C = np.asarray(C_nodes, dtype=float)
    E = np.asarray(E_nodes, dtype=float)
    if method == "linear":
        val = piecewise_linear_interpolate(C, E, np.array([concentration]))
        return float(val[0])
    elif method == "lagrange":
        if len(C) > 10:
            raise ValueError("Lagrange interpolation unstable for >10 nodes")
        val = lagrange_interpolate(C, E, np.array([concentration]))
        return float(val[0])
    else:
        raise ValueError("method must be 'linear' or 'lagrange'")


def build_pd_curve_from_hill(C50: float, Emax: float, n_hill: float,
                              n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 Hill 方程生成理论 PD 曲线：
        E(C) = Emax * C^n / (C50^n + C^n)
    返回 (C_nodes, E_nodes) 用于插值。
    """
    if C50 <= 0 or Emax < 0 or n_hill <= 0:
        raise ValueError("Invalid Hill parameters")
    # 对数均匀分布的采样点
    C_nodes = np.logspace(-3, 3, n_points) * C50
    E_nodes = Emax * (C_nodes ** n_hill) / (C50 ** n_hill + C_nodes ** n_hill)
    return C_nodes, E_nodes


def pharmacodynamic_response(C_plasma: float, C50: float, Emax: float,
                              n_hill: float, baseline: float = 0.0) -> float:
    """
    计算药物的药效学响应（Emax 模型 + 基线效应）。
        E = E0 + Emax * C^n / (C50^n + C^n)
    """
    if C_plasma < 0 or C50 <= 0 or Emax < 0 or n_hill <= 0:
        raise ValueError("Invalid PD parameters")
    effect = baseline + Emax * (C_plasma ** n_hill) / (C50 ** n_hill + C_plasma ** n_hill)
    return effect


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x_test = np.linspace(-1, 1, 100)
    print(f"Runge at 0.5: {runge_function(0.5):.6f}")
    print(f"Bernstein at -0.3: {bernstein_example(-0.3):.6f}")
    print(f"Oscillatory at 0.1: {oscillatory_function(0.1):.6f}")
    # Chebyshev 插值测试
    nodes = chebyshev_nodes(10, -1, 1)
    vals = np.array([runge_function(xi) for xi in nodes])
    x_fine = np.linspace(-1, 1, 200)
    y_interp = lagrange_interpolate(nodes, vals, x_fine)
    y_exact = np.array([runge_function(xi) for xi in x_fine])
    print(f"Chebyshev Lagrange max error: {np.max(np.abs(y_interp - y_exact)):.4e}")
    # PD 曲线
    C_nodes, E_nodes = build_pd_curve_from_hill(1.0, 100.0, 2.0)
    eff = pd_effect_interpolate(1.5, C_nodes, E_nodes, method="linear")
    print(f"PD effect at 1.5xC50: {eff:.2f}")
    resp = pharmacodynamic_response(2.0, 1.0, 100.0, 2.0, 10.0)
    print(f"Pharmacodynamic response: {resp:.2f}")
