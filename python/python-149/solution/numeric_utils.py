"""
numeric_utils.py
数值工具集

融合种子项目:
  - 094_bisection: 二分法求根
  - 898_polynomials: 多项式基准测试函数（Rosenbrock等）
  - 824_octopus: 环境检测

科学背景:
  在最优控制中，经常需要求解以下数值问题:
  1. Bang-Bang控制切换时间: 找到 τ 使得 H(τ) = 0
  2. 验证优化算法在经典基准函数上的性能
  3. 检测计算环境以确保数值稳定性
"""

import numpy as np
import sys
from typing import Callable, Optional, Tuple


# ============================================================================
# 环境检测（融合octopus）
# ============================================================================

def check_environment() -> dict:
    """
    检测数值计算环境并返回兼容性报告。

    Returns
    -------
    report : dict
        包含Python版本、NumPy版本、浮点精度等信息
    """
    report = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "float_info": {
            "epsilon": np.finfo(float).eps,
            "max": np.finfo(float).max,
            "min": np.finfo(float).tiny,
        },
        "int_info": {
            "max": np.iinfo(int).max,
        },
    }
    # 检查BLAS/LAPACK后端
    config = np.__config__
    if hasattr(config, "show"):
        # 简单检测
        report["blas_available"] = True
    return report


def assert_numeric_stability(x: np.ndarray, name: str = "array") -> bool:
    """
    检查数值数组的稳定性：是否存在NaN、Inf或极端值。

    Returns
    -------
    stable : bool
    """
    if not np.all(np.isfinite(x)):
        return False
    max_abs = np.max(np.abs(x))
    if max_abs > 1e15:
        return False
    return True


# ============================================================================
# 二分法求根（融合bisection）
# ============================================================================

def bisection_find_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Tuple[float, float, int]:
    """
    二分法求函数零点。

    前提: f(a) 与 f(b) 异号。

    算法:
        c = (a+b)/2
        若 f(c)==0: 返回c
        否则: 根据 f(c) 的符号替换 a 或 b

    收敛性:
        |b_n - a_n| = (b_0 - a_0) / 2^n
        即每次迭代误差减半，线性收敛。

    Parameters
    ----------
    f : callable
    a, b : float
        有根区间端点
    tol : float
        区间长度容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    root_lower : float
        下界
    root_upper : float
        上界（满足 |upper-lower| < tol）
    it : int
        实际迭代次数
    """
    fa = f(a)
    fb = f(b)

    # 边界检查
    if fa == 0.0:
        return a, a, 0
    if fb == 0.0:
        return b, b, 0
    if fa * fb > 0:
        raise ValueError(f"区间[{a},{b}]不是变号区间: f(a)={fa}, f(b)={fb}")

    it = 0
    while abs(b - a) > tol:
        c = (a + b) / 2.0
        fc = f(c)
        it += 1

        if it > max_iter:
            break

        if fc == 0.0:
            a = c
            b = c
            break
        elif np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    return a, b, it


def find_switching_time(
    hamiltonian_fn: Callable[[float], float],
    t0: float,
    t1: float,
    tol: float = 1e-6,
) -> Optional[float]:
    """
    寻找Bang-Bang最优控制的切换时间 τ。

    对于Pontryagin极大值原理，最优控制满足:
        u*(t) = sign( H_u(t) )

    切换时间 τ 是 H_u(τ) = 0 的根。

    Parameters
    ----------
    hamiltonian_fn : callable
        H_u(t) 的函数
    t0, t1 : float
        搜索区间

    Returns
    -------
    tau : float or None
        切换时间估计，若无变号区间则返回None
    """
    try:
        a, b, it = bisection_find_root(hamiltonian_fn, t0, t1, tol=tol)
        return (a + b) / 2.0
    except ValueError:
        return None


# ============================================================================
# 多项式基准测试函数（融合polynomials/rosenbrock）
# ============================================================================

def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock函数（香蕉函数）:

        f(x) = Σ_{i=1}^{n-1} [ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    全局最小值在 x_i = 1 处，f(1,...,1) = 0。
    用于验证优化算法的收敛性能。
    """
    x = np.atleast_1d(x)
    n = len(x)
    if n < 2:
        return (1.0 - x[0]) ** 2
    val = 0.0
    for i in range(n - 1):
        val += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return val


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Rosenbrock函数的梯度。"""
    x = np.atleast_1d(x)
    n = len(x)
    grad = np.zeros(n)
    if n < 2:
        grad[0] = -2.0 * (1.0 - x[0])
        return grad
    for i in range(n - 1):
        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        grad[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return grad


def himmelblau_function(x: np.ndarray) -> float:
    """
    Himmelblau函数:

        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    有四个全局最小值，f=0。
    """
    x = np.atleast_1d(x)
    if len(x) < 2:
        return (x[0] ** 2 - 11.0) ** 2
    xx, yy = x[0], x[1]
    return (xx ** 2 + yy - 11.0) ** 2 + (xx + yy ** 2 - 7.0) ** 2


def benchmark_optimizer(
    optimizer_fn: Callable[[Callable, np.ndarray], Tuple[np.ndarray, int]],
    test_functions: Optional[list] = None,
    dims: list = [2, 4],
) -> dict:
    """
    使用基准函数验证优化器性能。

    Returns
    -------
    results : dict
        各测试函数的最终函数值与迭代次数
    """
    if test_functions is None:
        test_functions = [
            ("rosenbrock", rosenbrock_function),
            ("himmelblau", himmelblau_function),
        ]

    results = {}
    rng = np.random.default_rng(seed=42)

    for name, fn in test_functions:
        for dim in dims:
            if name == "himmelblau" and dim != 2:
                continue
            x0 = rng.uniform(-2, 2, dim)
            try:
                x_opt, n_iter = optimizer_fn(fn, x0)
                f_opt = fn(x_opt)
                results[f"{name}_d{dim}"] = {
                    "f_opt": float(f_opt),
                    "n_iter": n_iter,
                    "x_opt": x_opt.tolist(),
                }
            except Exception as e:
                results[f"{name}_d{dim}"] = {"error": str(e)}

    return results


# ============================================================================
# 数值稳定性工具
# ============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除零。"""
    if abs(b) < 1e-15:
        return default
    return a / b


def soft_clip(x: float, xmin: float, xmax: float, hardness: float = 10.0) -> float:
    """
    软裁剪函数（可微近似硬裁剪）:

        y = xmin + (xmax-xmin) * sigmoid(hardness * (x - (xmin+xmax)/2) / (xmax-xmin))

    在边界附近光滑过渡。
    """
    mid = (xmin + xmax) / 2.0
    scale = (xmax - xmin) / 2.0
    if scale < 1e-10:
        return mid
    z = hardness * (x - mid) / scale
    # 防止exp溢出
    if z > 50.0:
        return xmax
    if z < -50.0:
        return xmin
    sig = 1.0 / (1.0 + np.exp(-z))
    return xmin + (xmax - xmin) * sig
