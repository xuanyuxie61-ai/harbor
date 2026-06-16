"""
test_problems.py
================
标准优化测试问题集 —— 基准测试与算法比较

提供经典的无约束非线性优化测试函数, 含精确梯度与 Hessian.
用于验证优化器的正确性与效率.

函数列表:
  1. Sphere: f = Σ x_i²
  2. Rosenbrock: 香蕉函数
  3. Rastrigin: 多模态
  4. Ackley: 多模态
  5. Beale: 2D 测试
  6. Booth: 2D 测试
  7. Himmelblau: 多极小 2D
  8. Powell: 四维, 条件数差
  9. Trid: 三对角 Hessian
  10. Schwefel: 偏移多模态
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
import math


# ---------------------------------------------------------------------------
# 1. Sphere 函数
# ---------------------------------------------------------------------------

class SphereProblem:
    """Sphere 函数: f(x) = Σ x_i².
    全局极小: x* = 0, f(x*) = 0.
    Hessian: H = 2I, 条件数 κ = 1 (良态).
    最简单的凸二次函数.
    """
    name = "Sphere"
    dim_range = (1, float('inf'))
    x_opt = None  # 依赖维数

    def __init__(self, dim: int = 5):
        self.dim = dim
        self.x_opt = np.zeros(dim)
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * np.eye(self.dim)


# ---------------------------------------------------------------------------
# 2. Rosenbrock (已定义在 quantum_objective, 此处包装)
# ---------------------------------------------------------------------------

class RosenbrockProblem:
    """Rosenbrock 香蕉函数.
    f(x) = Σ [100(x_{i+1} - x_i²)² + (1-x_i)²]
    x* = (1,...,1), f(x*) = 0.
    窄弯曲山谷, 条件数差.
    """
    name = "Rosenbrock"

    def __init__(self, dim: int = 5):
        self.dim = dim
        self.x_opt = np.ones(dim)
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        n = len(x)
        val = 0.0
        for i in range(n - 1):
            val += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        return val

    def grad(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        g = np.zeros(n)
        for i in range(n - 1):
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
            g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        H = np.zeros((n, n))
        for i in range(n - 1):
            H[i, i] += -400.0 * (x[i + 1] - 3.0 * x[i] ** 2) + 2.0
            H[i, i + 1] += -400.0 * x[i]
            H[i + 1, i] += -400.0 * x[i]
            H[i + 1, i + 1] += 200.0
        return H


# ---------------------------------------------------------------------------
# 3. Rastrigin 问题
# ---------------------------------------------------------------------------

class RastriginProblem:
    """Rastrigin 函数.
    f(x) = 10d + Σ [x_i² - 10 cos(2πx_i)]
    x* = 0, f(x*) = 0. 大量局部极小.
    """
    name = "Rastrigin"

    def __init__(self, dim: int = 5):
        self.dim = dim
        self.x_opt = np.zeros(dim)
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        return 10.0 * self.dim + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


# ---------------------------------------------------------------------------
# 4. Beale 问题 (2D)
# ---------------------------------------------------------------------------

class BealeProblem:
    """Beale 函数 (2D).
    f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    x* = (3, 0.5), f(x*) = 0.
    """
    name = "Beale"

    def __init__(self):
        self.dim = 2
        self.x_opt = np.array([3.0, 0.5])
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        a, b = x[0], x[1]
        return ((1.5 - a + a * b) ** 2
                + (2.25 - a + a * b ** 2) ** 2
                + (2.625 - a + a * b ** 3) ** 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        a, b = x[0], x[1]
        t1 = 1.5 - a + a * b
        t2 = 2.25 - a + a * b ** 2
        t3 = 2.625 - a + a * b ** 3
        g0 = 2 * t1 * (b - 1) + 2 * t2 * (b ** 2 - 1) + 2 * t3 * (b ** 3 - 1)
        g1 = 2 * t1 * a + 2 * t2 * 2 * a * b + 2 * t3 * 3 * a * b ** 2
        return np.array([g0, g1])


# ---------------------------------------------------------------------------
# 5. Himmelblau 问题 (2D, 多极小)
# ---------------------------------------------------------------------------

class HimmelblauProblem:
    """Himmelblau 函数 (2D).
    f(x,y) = (x² + y - 11)² + (x + y² - 7)²
    四个全局极小, f(x*) = 0:
      (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    """
    name = "Himmelblau"

    def __init__(self):
        self.dim = 2
        self.x_opt = np.array([3.0, 2.0])
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        a, b = x[0], x[1]
        return (a ** 2 + b - 11) ** 2 + (a + b ** 2 - 7) ** 2

    def grad(self, x: np.ndarray) -> np.ndarray:
        a, b = x[0], x[1]
        g0 = 4 * a * (a ** 2 + b - 11) + 2 * (a + b ** 2 - 7)
        g1 = 2 * (a ** 2 + b - 11) + 4 * b * (a + b ** 2 - 7)
        return np.array([g0, g1])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        a, b = x[0], x[1]
        H00 = 4 * (a ** 2 + b - 11) + 8 * a ** 2 + 2
        H11 = 2 + 4 * (a + b ** 2 - 7) + 8 * b ** 2
        H01 = 4 * a + 4 * b
        return np.array([[H00, H01], [H01, H11]])


# ---------------------------------------------------------------------------
# 6. Powell 问题 (四维, 零残差)
# ---------------------------------------------------------------------------

class PowellProblem:
    """Powell 函数 (4D).
    f(x) = (x₁ + 10x₂)² + 5(x₃ - x₄)² + (x₂ - 2x₃)⁴ + 10(x₁ - x₄)⁴
    x* = (0,0,0,0), f(x*) = 0.
    在最优点 Hessian 奇异 (零特征值), 收敛慢.
    """
    name = "Powell"

    def __init__(self):
        self.dim = 4
        self.x_opt = np.zeros(4)
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        return ((x1 + 10 * x2) ** 2 + 5 * (x3 - x4) ** 2
                + (x2 - 2 * x3) ** 4 + 10 * (x1 - x4) ** 4)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x1, x2, x3, x4 = x
        g0 = 2 * (x1 + 10 * x2) + 40 * (x1 - x4) ** 3
        g1 = 20 * (x1 + 10 * x2) + 4 * (x2 - 2 * x3) ** 3
        g2 = 10 * (x3 - x4) - 8 * (x2 - 2 * x3) ** 3
        g3 = -10 * (x3 - x4) - 40 * (x1 - x4) ** 3
        return np.array([g0, g1, g2, g3])


# ---------------------------------------------------------------------------
# 7. Trid 问题
# ---------------------------------------------------------------------------

class TridProblem:
    """Trid 函数.
    f(x) = Σ(x_i - 1)² - Σ x_i x_{i-1}
    三对角 Hessian, 条件数 O(d²).
    精确极小: x_i = i(d+1-i), f(x*) = -d(d+1)(d-1)/6.
    """
    name = "Trid"

    def __init__(self, dim: int = 6):
        self.dim = dim
        self.x_opt = np.array([i * (dim + 1 - i) for i in range(1, dim + 1)], dtype=float)
        self.f_opt = -dim * (dim + 1) * (dim - 1) / 6.0

    def f(self, x: np.ndarray) -> float:
        n = len(x)
        val = np.sum((x - 1.0) ** 2)
        for i in range(1, n):
            val -= x[i] * x[i - 1]
        return val

    def grad(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        g = 2.0 * (x - 1.0)
        for i in range(1, n):
            g[i] -= x[i - 1]
            g[i - 1] -= x[i]
        return g

    def hessian(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        H = 2.0 * np.eye(n)
        for i in range(1, n):
            H[i, i - 1] = -1.0
            H[i - 1, i] = -1.0
        return H


# ---------------------------------------------------------------------------
# 8. Schwefel 问题
# ---------------------------------------------------------------------------

class SchwefelProblem:
    """Schwefel 函数.
    f(x) = 418.9829d - Σ x_i sin(√|x_i|)
    x_i ∈ [-500, 500], x* ≈ (420.9687,...), f(x*) ≈ 0.
    深谷结构, 次极小远离全局极小.
    """
    name = "Schwefel"

    def __init__(self, dim: int = 3):
        self.dim = dim
        self.x_opt = 420.9687 * np.ones(dim)
        self.f_opt = 0.0

    def f(self, x: np.ndarray) -> float:
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def grad(self, x: np.ndarray) -> np.ndarray:
        # 数值梯度
        eps = 1e-6
        g = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            g[i] = (self.f(x_plus) - self.f(x_minus)) / (2 * eps)
        return g


# ---------------------------------------------------------------------------
# 9. 测试问题注册
# ---------------------------------------------------------------------------

def get_test_problems() -> Dict[str, object]:
    """返回所有测试问题实例."""
    return {
        "sphere": SphereProblem(dim=5),
        "rosenbrock": RosenbrockProblem(dim=5),
        "rastrigin": RastriginProblem(dim=3),
        "beale": BealeProblem(),
        "himmelblau": HimmelblauProblem(),
        "powell": PowellProblem(),
        "trid": TridProblem(dim=6),
        "schwefel": SchwefelProblem(dim=2),
    }


def run_test_suite(optimizer_func, tol: float = 1e-8) -> Dict:
    """在全部测试问题上运行优化器.
    optimizer_func: Callable(f, grad_f, x0, tol, max_iter) -> Dict

    返回各问题的优化结果.
    """
    problems = get_test_problems()
    results = {}

    for name, prob in problems.items():
        x0 = np.ones(prob.dim) * 0.5  # 标准起始点
        # 对某些问题用更好的起始点
        if name == "rosenbrock":
            x0 = np.ones(prob.dim) * (-1.0)
        elif name == "beale":
            x0 = np.array([1.0, 1.0])
        elif name == "himmelblau":
            x0 = np.array([0.0, 0.0])
        elif name == "schwefel":
            x0 = 200.0 * np.ones(prob.dim)

        try:
            res = optimizer_func(prob.f, prob.grad, x0, tol=tol, max_iter=1000)
            error = abs(res["f_val"] - prob.f_opt)
            results[name] = {
                "f_val": res["f_val"],
                "f_opt": prob.f_opt,
                "error": error,
                "converged": res["converged"],
                "iterations": res["iterations"]
            }
        except Exception as e:
            results[name] = {
                "error": str(e),
                "converged": False
            }

    return results
