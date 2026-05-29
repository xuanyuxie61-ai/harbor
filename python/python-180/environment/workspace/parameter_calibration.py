"""
parameter_calibration.py
无梯度参数标定与多项式基准测试

融合种子项目:
  - 907_praxis: Brent 主方向无梯度优化方法
  - 898_polynomials: Rosenbrock 等经典测试函数作为标定基准

科学背景:
  在 SPDE 参数反演中，观测数据 {Y_j}_{j=1}^M 与模型预测 U(theta) 之间的
  失配泛函 (misfit functional) 定义为:
      J(theta) = 0.5 * sum_{j=1}^M (U(x_j, t_j; theta) - Y_j)^2 / sigma_j^2
                 + 0.5 * alpha_reg ||theta - theta_prior||^2_{C_prior^{-1}}
  其中 theta = [epsilon, v, r, sigma_noise] 为待反演参数。

  该泛函关于 theta 的梯度通常难以解析求得（尤其当 SPDE 包含非线性反应项
  与乘性噪声时），因此无梯度优化方法具有显著优势。

  PRAXIS (Principal Axis) 方法:
      在每次迭代中，沿一组搜索方向 V = [v_1, ..., v_n] 进行一维最小化。
      通过 SVD 更新搜索方向以逼近 Hessian 的主轴:
          V_{new} = V * U * Sigma^{-1}
      其中 U, Sigma 来自当前近似二次型的 SVD 分解。
      收敛准则:
          ||x_{k+1} - x_k|| < tol + sqrt(eps) * ||x_k||

  测试函数 (用于验证优化器):
      Rosenbrock: f(x) = sum_{i=1}^{n-1} [100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
      全局最小值在 x_i = 1，f_min = 0。
"""

import numpy as np
from typing import Callable, Tuple, Optional
from numerical_utils import r8_hypot


class PraxisOptimizer:
    """
    简化版 PRAXIS 无梯度优化器，融合主方向搜索与 SVD 方向更新。
    """

    def __init__(self,
                 tol: float = 1e-8,
                 max_iter: int = 500,
                 h0: float = 1.0):
        self.tol = tol
        self.max_iter = max_iter
        self.h0 = h0

    def _line_minimize(self,
                       f: Callable[[np.ndarray], float],
                       x: np.ndarray,
                       d: np.ndarray,
                       f_x: float) -> Tuple[np.ndarray, float, float]:
        """
        沿方向 d 的一维二次插值最小化。
        采用黄金分割-抛物线混合搜索。
        """
        alpha = 0.0
        fa = f_x
        # 试探步长
        h = self.h0 / (np.linalg.norm(d) + 1e-12)
        fb = f(x + h * d)
        # 确定下降方向
        if fb > fa:
            h = -h
            fb = f(x + h * d)
            if fb > fa:
                return x, fa, 0.0

        # 抛物线拟合求极小点
        alpha_c = 2.0 * h
        fc = f(x + alpha_c * d)
        # 三点二次插值
        denom = 2.0 * ((fa - fc) / h - (fa - fb) / h)
        if abs(denom) < 1e-16:
            alpha_star = h if fb < fa else 0.0
        else:
            alpha_star = h - ((fa - fb) / h) / denom
            # 边界保护
            alpha_star = max(min(alpha_star, 2.0 * h), -abs(h))
        x_new = x + alpha_star * d
        f_new = f(x_new)
        if f_new < fa:
            return x_new, f_new, alpha_star
        elif fb < fa:
            return x + h * d, fb, h
        else:
            return x, fa, 0.0

    def minimize(self,
                 f: Callable[[np.ndarray], float],
                 x0: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        PRAXIS 主优化循环。

        返回:
            x_opt: 最优参数
            f_opt: 最优函数值
            n_iter: 实际迭代次数
        """
        x = x0.copy().astype(np.float64)
        n = len(x)
        if n < 1:
            raise ValueError("x0 must have length >= 1")

        # 初始搜索方向为单位矩阵
        V = np.eye(n, dtype=np.float64)
        fx = f(x)

        for iteration in range(self.max_iter):
            x_old = x.copy()
            # 沿每个主方向搜索
            for i in range(n):
                d = V[:, i]
                x, fx, _ = self._line_minimize(f, x, d, fx)

            # 计算新共轭方向
            s = x - x_old
            if np.linalg.norm(s) < self.tol:
                break

            # 归一化并替换最后一个方向
            s_norm = np.linalg.norm(s)
            if s_norm > 1e-14:
                d_new = s / s_norm
                x, fx, _ = self._line_minimize(f, x, d_new, fx)
                # 轮换方向
                V[:, 1:] = V[:, :-1]
                V[:, 0] = d_new

            # SVD 更新主方向 (每 5 步一次)
            if (iteration + 1) % 5 == 0 and n > 1:
                try:
                    U_svd, S_svd, _ = np.linalg.svd(V)
                    # 按奇异值降序重排
                    sort_idx = np.argsort(-S_svd)
                    V = U_svd[:, sort_idx]
                except np.linalg.LinAlgError:
                    pass

            if np.linalg.norm(x - x_old) < self.tol + np.sqrt(np.finfo(float).eps) * np.linalg.norm(x_old):
                break

        return x, fx, iteration + 1


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock 函数 (n>=2):
        f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    全局最小值: x_i = 1, f = 0。
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("Rosenbrock requires n >= 2")
    a = x[1:] - x[:-1] ** 2
    b = 1.0 - x[:-1]
    return float(np.sum(100.0 * a ** 2 + b ** 2))


def camel_back(x: np.ndarray) -> float:
    """
    Three-hump camel back function (2D):
        f(x,y) = 2x^2 - 1.05x^4 + x^6/6 + xy + y^2
    """
    if len(x) != 2:
        raise ValueError("Camel back is 2D only")
    xx, yy = x[0], x[1]
    return float(2.0 * xx ** 2 - 1.05 * xx ** 4 + xx ** 6 / 6.0 + xx * yy + yy ** 2)


class SPDEParameterCalibration:
    """
    SPDE 参数标定器：将模型输出与参考数据的差异作为优化目标。
    """

    def __init__(self,
                 reference_data: np.ndarray,
                 spatial_grid: np.ndarray,
                 solver_factory: Callable,
                 param_bounds: Optional[np.ndarray] = None):
        self.reference = reference_data
        self.spatial_grid = spatial_grid
        self.solver_factory = solver_factory
        self.param_bounds = param_bounds

    def misfit(self, theta: np.ndarray) -> float:
        """
        失配泛函。
        theta = [epsilon, v, r, sigma_noise]
        """
        theta = np.clip(theta, 1e-6, 10.0)  # 物理约束
        try:
            u_pred = self.solver_factory(theta)
            diff = u_pred - self.reference
            mse = 0.5 * np.mean(diff ** 2)
            # 正则化
            reg = 1e-4 * np.sum(theta ** 2)
            return float(mse + reg)
        except Exception:
            return 1e6

    def calibrate(self, theta0: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用 PRAXIS 标定参数。
        """
        opt = PraxisOptimizer(tol=1e-6, max_iter=100, h0=0.1)
        theta_opt, f_opt, _ = opt.minimize(self.misfit, theta0)
        if self.param_bounds is not None:
            theta_opt = np.clip(theta_opt, self.param_bounds[:, 0], self.param_bounds[:, 1])
        return theta_opt, f_opt
