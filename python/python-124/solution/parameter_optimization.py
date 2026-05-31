"""
parameter_optimization.py
骨重建参数反演优化模块

融合来源：
- 1220_test_nls: 非线性最小二乘测试问题集（More-Garbow-Hillstrom）
- 476_golden_section: 黄金分割搜索
- 835_opt_gradient_descent: 梯度下降优化

科学背景：
骨重建模型包含多个未知参数（k_form, k_res, U_ref），
需要通过临床测量数据（如DXA骨密度扫描）进行反演识别。

问题表述（参数识别）：
    min_{θ}  0.5 * Σ_{i=1}^M ||ρ_sim(x_i; θ) - ρ_meas(x_i)||²

其中 θ = [k_form, k_res, U_ref]^T 为待识别参数。

核心算法：
1. Levenberg-Marquardt 非线性最小二乘（基于 1220_test_nls 框架）
2. 黄金分割搜索（一维线搜索，来自 476_golden_section）
3. 梯度下降优化（来自 835_opt_gradient_descent）
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from scipy.optimize import least_squares


# ===================================================================
# 黄金分割搜索（来自 476_golden_section）
# ===================================================================
def golden_section_search(f: Callable[[float], float],
                          a: float, b: float,
                          max_iter: int = 100,
                          x_tol: float = 1e-7) -> Tuple[float, float, int, int]:
    """
    在区间 [a, b] 上寻找单峰函数的极小值点。

    黄金比例：g = (√5 - 1) / 2 ≈ 0.618033988749895

    算法：
        x1 = g*a + (1-g)*b
        x2 = (1-g)*a + g*b
        比较 f(x1) 和 f(x2)，缩小区间
        每轮迭代仅计算1次新函数值。

    Parameters
    ----------
    f : callable
        单峰目标函数
    a, b : float
        搜索区间，需满足 a < b
    max_iter : int
        最大迭代次数
    x_tol : float
        区间宽度容差

    Returns
    -------
    a, b : float
        最终包含极小值的区间
    it : int
        迭代次数
    nf : int
        函数求值次数
    """
    if a >= b:
        raise ValueError("Must have a < b")

    g = (np.sqrt(5.0) - 1.0) / 2.0  # 黄金比例
    x1 = g * a + (1.0 - g) * b
    x2 = (1.0 - g) * a + g * b
    f1 = f(x1)
    f2 = f(x2)
    nf = 2

    for it in range(max_iter):
        if abs(b - a) <= x_tol:
            return a, b, it, nf

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = g * a + (1.0 - g) * b
            f1 = f(x1)
            nf += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - g) * a + g * b
            f2 = f(x2)
            nf += 1

    return a, b, max_iter, nf


# ===================================================================
# 梯度下降优化（来自 835_opt_gradient_descent）
# ===================================================================
def gradient_descent(fp: Callable[[float], float], x0: float,
                     gamma: float = 0.01,
                     precision: float = 1e-7,
                     max_iter: int = 10000) -> Tuple[float, int]:
    """
    一维梯度下降法寻找局部最小值。

    迭代公式：
        x_{k+1} = x_k - γ * f'(x_k)

    Parameters
    ----------
    fp : callable
        导数函数 f'(x)
    x0 : float
        初始点
    gamma : float
        步长因子
    precision : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    x : float
        估计极小值点
    it : int
        迭代次数
    """
    x = x0
    for it in range(1, max_iter + 1):
        x_old = x
        grad = fp(x_old)
        if not np.isfinite(grad):
            raise RuntimeError(f"Non-finite gradient at x={x_old}")
        x = x_old - gamma * grad
        if abs(x - x_old) <= precision:
            return x, it
    return x, max_iter


# ===================================================================
# 非线性最小二乘参数识别（基于 1220_test_nls 框架）
# ===================================================================
class ParameterIdentification:
    """
    骨重建参数识别器。

    通过非线性最小二乘拟合，从测量数据反演模型参数。
    """

    def __init__(self, forward_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 measured_data: np.ndarray,
                 measurement_points: np.ndarray,
                 param_bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Parameters
        ----------
        forward_model : callable
            前向模型 f(params, x_points) -> predicted_values
        measured_data : np.ndarray
            测量数据
        measurement_points : np.ndarray
            测量位置
        param_bounds : list of tuple, optional
            参数边界 [(lb1, ub1), (lb2, ub2), ...]
        """
        self.forward_model = forward_model
        self.measured_data = np.asarray(measured_data)
        self.measurement_points = np.asarray(measurement_points)
        self.param_bounds = param_bounds
        self.n_params = len(param_bounds) if param_bounds else 3

    def residual(self, params: np.ndarray) -> np.ndarray:
        """
        计算残差向量 r_i = y_pred_i - y_meas_i。

        来自 1220_test_nls 的残差定义框架。
        """
        pred = self.forward_model(params, self.measurement_points)
        return pred - self.measured_data

    def objective(self, params: np.ndarray) -> float:
        """
        计算最小二乘目标函数值：
            Φ(θ) = 0.5 * ||r(θ)||²
        """
        r = self.residual(params)
        return 0.5 * np.dot(r, r)

    def jacobian_finite_difference(self, params: np.ndarray,
                                   h: float = 1e-7) -> np.ndarray:
        """
        有限差分计算 Jacobian 矩阵 J_{ij} = ∂r_i / ∂θ_j。
        """
        n = len(params)
        m = len(self.measured_data)
        J = np.zeros((m, n))
        r0 = self.residual(params)

        for j in range(n):
            params_plus = params.copy()
            params_plus[j] += h
            r_plus = self.residual(params_plus)
            J[:, j] = (r_plus - r0) / h

        return J

    def optimize(self, x0: np.ndarray, method: str = 'lm',
                 max_iter: int = 100) -> dict:
        """
        执行参数优化。

        Parameters
        ----------
        x0 : np.ndarray
            初始参数猜测
        method : str
            优化方法 ('lm' for Levenberg-Marquardt,
                     'trf' for Trust Region Reflective)
        max_iter : int
            最大迭代次数

        Returns
        -------
        dict
            优化结果
        """
        x0 = np.asarray(x0)
        if len(x0) != self.n_params:
            raise ValueError(f"Initial guess length {len(x0)} != n_params {self.n_params}")

        bounds = None
        if self.param_bounds is not None:
            lb = [b[0] for b in self.param_bounds]
            ub = [b[1] for b in self.param_bounds]
            bounds = (lb, ub)

        # 'lm' 方法不支持 bounds，若指定了 bounds 则改用 'trf'
        actual_method = method
        if bounds is not None and method == 'lm':
            actual_method = 'trf'

        result = least_squares(
            self.residual, x0, method=actual_method,
            max_nfev=max_iter * len(x0) * 10,
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            bounds=bounds
        )

        return {
            'params': result.x,
            'cost': result.cost,
            'nfev': result.nfev,
            'njev': result.njev,
            'success': result.success,
            'message': result.message,
            'jacobian': result.jac
        }

    def optimize_golden_section_1d(self, idx: int, fixed_params: np.ndarray,
                                   bracket: Tuple[float, float]) -> Tuple[float, float]:
        """
        对单个参数使用黄金分割搜索（一维优化）。

        Parameters
        ----------
        idx : int
            待优化参数的索引
        fixed_params : np.ndarray
            其他参数的固定值
        bracket : tuple
            搜索区间 (a, b)
        """
        def f_1d(x: float) -> float:
            p = fixed_params.copy()
            p[idx] = x
            return self.objective(p)

        a, b, it, nf = golden_section_search(f_1d, bracket[0], bracket[1])
        x_opt = (a + b) / 2.0
        f_opt = f_1d(x_opt)
        return x_opt, f_opt


# ===================================================================
# 骨重建前向模型（用于参数识别）
# ===================================================================
def bone_remodeling_forward_model(params: np.ndarray,
                                  x_points: np.ndarray,
                                  U_field: Optional[np.ndarray] = None,
                                  t_final: float = 365.0) -> np.ndarray:
    """
    简化的骨重建前向模型。

    假设稳态近似：
        ρ(x) ≈ ρ_max * (U(x) / (U(x) + U_ref))

    其中 params = [k_form, k_res, U_ref]
    """
    if len(params) < 3:
        raise ValueError("params must contain at least [k_form, k_res, U_ref]")

    k_form, k_res, U_ref = params[0], params[1], params[2]

    if U_ref <= 0:
        return np.full(len(x_points), 0.01)

    if U_field is None:
        # 模拟应变能场：中心高、边缘低
        U_field = 1.0 * np.exp(-0.01 * (x_points - np.mean(x_points)) ** 2)

    # 稳态密度（考虑形成/吸收平衡）
    ratio = k_form / max(k_res, 1e-10)
    rho = np.zeros(len(x_points))
    for i, U in enumerate(U_field):
        if U > U_ref:
            rho[i] = min(1.8, 0.5 + ratio * (U - U_ref))
        else:
            rho[i] = max(0.01, 0.5 - (U_ref - U) / U_ref)

    return rho
