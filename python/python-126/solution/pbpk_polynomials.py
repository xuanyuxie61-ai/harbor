"""
pbpk_polynomials.py
基于种子项目 898_polynomials

实现多元多项式函数库，用于构建 PBPK 模型中的：
1. 药物-受体结合势能面（Rosenbrock、Himmelblau 等经典 landscape）
2. 代谢酶动力学的高阶多项式近似
3. 多靶点给药的复合目标函数

在 PBPK 模型中用于：
- 药物-受体结合的势能面建模（构象搜索）
- 多参数优化问题的测试 landscape
- 浓度-效应关系的非线性多项式拟合
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# 经典多元多项式测试函数（全局优化 landscape）
# ---------------------------------------------------------------------------

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock 函数（香蕉函数）：
        f(x) = Σ_{i=1}^{n-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    在 PBPK 中用于构建药物构象空间的势能面，全局极小在 x_i = 1。
    """
    if len(x) < 2:
        raise ValueError("Rosenbrock requires at least 2 dimensions")
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def himmelblau(x: np.ndarray) -> float:
    """
    Himmelblau 函数：
        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    4 个全局极小点，用于测试多稳态代谢通路的势能面。
    """
    if len(x) != 2:
        raise ValueError("Himmelblau requires exactly 2 dimensions")
    x, y = x[0], x[1]
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def camel_back(x: np.ndarray) -> float:
    """
    三峰骆驼函数（Three-hump camel）：
        f(x,y) = 2x^2 - 1.05x^4 + x^6/6 + xy + y^2
    用于模拟药物与受体结合的多个亚稳态构象。
    """
    if len(x) != 2:
        raise ValueError("Camel back requires exactly 2 dimensions")
    x, y = x[0], x[1]
    return (2.0 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6.0 + x * y + y ** 2)


def butchers_polynomial(x: np.ndarray) -> float:
    """
    Butcher 多项式（高维多项式 benchmark）。
    用于测试 PBPK 高维参数敏感性分析。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Butcher requires at least 2 dimensions")
    val = 0.0
    for i in range(n - 1):
        val += (x[i] + x[i + 1] - 1.0) ** 2
    return val


def cyclic_n_polynomial(x: np.ndarray) -> float:
    """
    Cyclic-n 多项式系统（代数几何经典问题）。
    在 PBPK 中用于代谢网络稳态方程的求根问题。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("cyclic-n requires at least 2 dimensions")
    # 简化为目标函数形式
    s1 = np.sum(x)
    s2 = np.sum(x ** 2)
    return (s1 - 1.0) ** 2 + (s2 - 1.0) ** 2


# ---------------------------------------------------------------------------
# PBPK 专用多项式模型
# ---------------------------------------------------------------------------

def binding_potential_surface(drug_conc: float, receptor_conc: float,
                               k_on: float, k_off: float,
                               cooperativity: int = 1) -> float:
    """
    药物-受体结合的势能面多项式模型。
    使用 Hill 方程的多项式展开近似：
        P(D,R) = k_on D R - k_off (1 - (D/K_d)^n / (1 + (D/K_d)^n))
    输入：
        drug_conc : 药物浓度 [M]
        receptor_conc : 受体浓度 [M]
        k_on, k_off : 结合/解离速率
        cooperativity : Hill 系数（协同性）
    返回：结合势能（kcal/mol 尺度归一化）
    """
    if drug_conc < 0 or receptor_conc < 0 or k_on < 0 or k_off < 0:
        raise ValueError("Concentrations and rates must be non-negative")
    K_d = k_off / max(k_on, 1e-20)
    ratio = drug_conc / max(K_d, 1e-20)
    # 多项式展开：使用 Taylor 展开到 5 阶
    if ratio < 1.0:
        hill = 0.0
        term = 1.0
        for n in range(1, 6):
            term *= -ratio ** cooperativity
            hill += term
        hill = -hill
    else:
        hill = 1.0 - 1.0 / (1.0 + ratio ** cooperativity)
    potential = k_on * drug_conc * receptor_conc - k_off * hill
    return potential


def enzyme_kinetics_polynomial_substrate(S: float, Vmax: float, Km: float,
                                          Ki: float = 1e10, I: float = 0.0) -> float:
    """
    竞争性抑制下的 Michaelis-Menten 速率的多项式近似：
        v = Vmax S / (Km (1 + I/Ki) + S)
    使用 Pade 近似转化为有理多项式形式。
    """
    if S < 0 or Vmax < 0 or Km <= 0 or Ki <= 0 or I < 0:
        raise ValueError("Invalid kinetic parameters")
    Km_app = Km * (1.0 + I / Ki)
    # [1/1] Pade 近似：v ≈ Vmax S / (Km_app + S)
    # 直接计算避免多项式展开的不稳定性
    v = Vmax * S / (Km_app + S)
    return v


def multi_target_objective(concentrations: np.ndarray,
                            target_eff: np.ndarray,
                            toxicity_weights: np.ndarray) -> float:
    """
    多靶点给药的复合目标函数（多项式形式）。
    目标：最大化疗效同时最小化毒性。
        J(C) = - Σ w_i log(1 + C_i/C50_i) + λ Σ toxicity_weights_j C_j^2
    输入：
        concentrations : 各 compartment 浓度向量
        target_eff : 靶点 C50 向量
        toxicity_weights : 毒性权重向量
    返回：标量目标函数值（越小越好）
    """
    C = np.asarray(concentrations, dtype=float)
    if np.any(C < 0):
        raise ValueError("Concentrations must be non-negative")
    if len(target_eff) != len(C) or len(toxicity_weights) != len(C):
        raise ValueError("Array lengths must match")
    efficacy = -np.sum(np.log1p(C / np.maximum(target_eff, 1e-20)))
    toxicity = np.sum(toxicity_weights * C ** 2)
    return efficacy + 0.1 * toxicity


# ---------------------------------------------------------------------------
# 高阶敏感性分析多项式
# ---------------------------------------------------------------------------

def sobol_g_function(x: np.ndarray, a: np.ndarray = None) -> float:
    """
    Sobol G-Function：全局敏感性分析的经典测试函数。
        f(x) = Π_{i=1}^d (|4x_i - 2| + a_i) / (1 + a_i)
    在 PBPK 中用于评估各生理参数对药物暴露量的影响。
    """
    x = np.asarray(x, dtype=float)
    d = len(x)
    if a is None:
        a = np.linspace(0.0, 9.0, d)
    if len(a) != d:
        raise ValueError("a must have same length as x")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("x must be in [0,1]")
    product = 1.0
    for i in range(d):
        product *= (abs(4.0 * x[i] - 2.0) + a[i]) / (1.0 + a[i])
    return product


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = np.array([1.0, 1.0, 1.0])
    print(f"Rosenbrock at [1,1,1]: {rosenbrock(x):.6f}")
    print(f"Himmelblau at [3,2]: {himmelblau(np.array([3.0, 2.0])):.6f}")
    print(f"Camel back at [0,0]: {camel_back(np.array([0.0, 0.0])):.6f}")
    p = binding_potential_surface(1e-6, 1e-9, 1e5, 1e-3, 2)
    print(f"Binding potential: {p:.6e}")
    v = enzyme_kinetics_polynomial_substrate(5.0, 10.0, 2.0, Ki=1.0, I=0.5)
    print(f"Enzyme velocity: {v:.6f}")
    obj = multi_target_objective(np.array([1.0, 2.0, 0.5]),
                                  np.array([1.0, 1.0, 1.0]),
                                  np.array([0.1, 0.2, 0.5]))
    print(f"Multi-target objective: {obj:.6f}")
    print(f"Sobol G-function: {sobol_g_function(np.array([0.5, 0.5, 0.5])):.6f}")
