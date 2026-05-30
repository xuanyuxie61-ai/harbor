
import numpy as np
import sys
from typing import Callable, Optional, Tuple






def check_environment() -> dict:
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

    config = np.__config__
    if hasattr(config, "show"):

        report["blas_available"] = True
    return report


def assert_numeric_stability(x: np.ndarray, name: str = "array") -> bool:
    if not np.all(np.isfinite(x)):
        return False
    max_abs = np.max(np.abs(x))
    if max_abs > 1e15:
        return False
    return True






def bisection_find_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Tuple[float, float, int]:
    fa = f(a)
    fb = f(b)


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
    try:
        a, b, it = bisection_find_root(hamiltonian_fn, t0, t1, tol=tol)
        return (a + b) / 2.0
    except ValueError:
        return None






def rosenbrock_function(x: np.ndarray) -> float:
    x = np.atleast_1d(x)
    n = len(x)
    if n < 2:
        return (1.0 - x[0]) ** 2
    val = 0.0
    for i in range(n - 1):
        val += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return val


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
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






def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-15:
        return default
    return a / b


def soft_clip(x: float, xmin: float, xmax: float, hardness: float = 10.0) -> float:
    mid = (xmin + xmax) / 2.0
    scale = (xmax - xmin) / 2.0
    if scale < 1e-10:
        return mid
    z = hardness * (x - mid) / scale

    if z > 50.0:
        return xmax
    if z < -50.0:
        return xmin
    sig = 1.0 / (1.0 + np.exp(-z))
    return xmin + (xmax - xmin) * sig
