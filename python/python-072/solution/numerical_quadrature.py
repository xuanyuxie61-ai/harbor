"""
numerical_quadrature.py
=======================
高精度数值积分与求积公式模块

融合种子项目：
- 464_gen_hermite_exactness: 广义 Gauss-Hermite 求积精确度检验
- 559_hypercube_integrals: 超立方体上的单项式积分
- 1207_test_int: 一维数值积分测试库（多种求积方法）

核心内容：
1. Gauss-Legendre 求积（有限区间）
2. Gauss-Hermite 求积（无限区间，权重 exp(-x²)）
3. 广义 Gauss-Hermite 求积（权重 |x|^α exp(-x²)）
4. 超立方体高维积分
5. Monte Carlo 积分
6. 复合 Simpson 求积
7. 求积规则精确度检验

Gauss-Hermite 求积公式：
    ∫_{-∞}^{+∞} |x|^α exp(-x²) f(x) dx ≈ Σ_i w_i f(x_i)

超立方体上的单项式积分：
    ∫_{[0,1]^m} ∏_{i=1}^m x_i^{e_i} dV = ∏_{i=1}^m 1/(e_i + 1)
"""

import numpy as np


class GaussQuadrature:
    """
    高斯型数值求积公式集合。
    """

    @staticmethod
    def gauss_legendre_3point():
        """
        3点 Gauss-Legendre 求积规则和节点。

        在 [-1, 1] 上精确积分 5 次多项式：
            ∫_{-1}^{1} f(x) dx ≈ Σ w_i f(x_i)

        Returns
        -------
        tuple
            (nodes, weights)
        """
        nodes = np.array([
            -0.7745966692414834,
             0.0,
             0.7745966692414834
        ])
        weights = np.array([
            0.5555555555555556,
            0.8888888888888889,
            0.5555555555555556
        ])
        return nodes, weights

    @staticmethod
    def gauss_legendre_5point():
        """
        5点 Gauss-Legendre 求积规则。
        精确积分 9 次多项式。
        """
        nodes = np.array([
            -0.9061798459386640,
            -0.5384693101056831,
             0.0,
             0.5384693101056831,
             0.9061798459386640
        ])
        weights = np.array([
            0.2369268850561891,
            0.4786286704993665,
            0.5688888888888889,
            0.4786286704993665,
            0.2369268850561891
        ])
        return nodes, weights

    @staticmethod
    def gauss_hermite_5point():
        """
        5点 Gauss-Hermite 求积规则。

        在 (-∞, +∞) 上权重为 exp(-x²)：
            ∫_{-∞}^{+∞} exp(-x²) f(x) dx ≈ Σ w_i f(x_i)

        精确积分 9 次 Hermite 多项式。

        Returns
        -------
        tuple
            (nodes, weights)
        """
        nodes = np.array([
            -2.0201828704560856,
            -0.9585724646138195,
             0.0,
             0.9585724646138195,
             2.0201828704560856
        ])
        weights = np.array([
            0.0199532420590459,
            0.3936193231522412,
            0.9453087204829419,
            0.3936193231522412,
            0.0199532420590459
        ])
        return nodes, weights

    @staticmethod
    def generalized_hermite_integral(expon, alpha):
        """
        计算广义 Hermite 积分的精确值：
            H(n, α) = ∫_{-∞}^{+∞} x^n |x|^α exp(-x²) dx

        当 n 为奇数时，积分为 0（被积函数为奇函数）。
        当 n 为偶数时：
            H(n, α) = Γ((α + n + 1) / 2)

        Parameters
        ----------
        expon : int
            单项式指数 n。
        alpha : float
            权重指数 α，必须 > -1。

        Returns
        -------
        float
            积分精确值。
        """
        if alpha <= -1.0:
            raise ValueError("alpha 必须大于 -1")

        if expon % 2 == 1:
            return 0.0

        a = alpha + expon
        if a <= -1.0:
            return -np.inf

        from scipy.special import gamma
        return gamma((a + 1.0) / 2.0)


class HypercubeIntegrals:
    """
    超立方体 [0,1]^m 上的积分计算。
    基于种子项目 559_hypercube_integrals。
    """

    @staticmethod
    def monomial_integral(m, exponents):
        """
        计算超立方体上的单项式积分精确值：
            I = ∫_{[0,1]^m} ∏_{i=1}^m x_i^{e_i} dV = ∏_{i=1}^m 1/(e_i + 1)

        Parameters
        ----------
        m : int
            空间维数。
        exponents : ndarray, shape (m,)
            各维度的指数，必须为非负整数。

        Returns
        -------
        float
            积分精确值。
        """
        exponents = np.asarray(exponents)
        if len(exponents) != m:
            raise ValueError("exponents 长度必须等于维数 m")
        if np.any(exponents < 0):
            raise ValueError("所有指数必须为非负")

        integral = 1.0
        for e in exponents:
            integral /= (e + 1.0)
        return integral

    @staticmethod
    def sample_hypercube(m, n):
        """
        在 [0,1]^m 上均匀随机采样 n 个点。

        Parameters
        ----------
        m : int
            维数。
        n : int
            采样点数。

        Returns
        -------
        ndarray, shape (m, n)
            采样点。
        """
        return np.random.rand(m, n)

    @staticmethod
    def monte_carlo_integral(func, m, n_samples, domain=(0.0, 1.0)):
        """
        Monte Carlo 积分估计。

        在 m 维超立方体 [a,b]^m 上积分：
            I ≈ (b-a)^m * (1/N) Σ f(x_i)

        Parameters
        ----------
        func : callable
            被积函数 func(x)，x 为 shape (m,) 的数组。
        m : int
            维数。
        n_samples : int
            采样点数。
        domain : tuple
            (a, b) 积分区间。

        Returns
        -------
        tuple
            (estimate, std_error) 积分估计值和标准误差。
        """
        a, b = domain
        samples = a + (b - a) * np.random.rand(m, n_samples)

        values = np.array([func(samples[:, i]) for i in range(n_samples)])

        volume = (b - a) ** m
        estimate = volume * np.mean(values)
        std_error = volume * np.std(values) / np.sqrt(n_samples)

        return estimate, std_error


class CompositeQuadrature:
    """
    复合求积公式，用于提高数值积分精度。
    基于种子项目 1207_test_int 中的多种求积方法。
    """

    @staticmethod
    def composite_simpson(f, a, b, n):
        """
        复合 Simpson 求积公式。

        将 [a,b] 分为 n 个等长子区间（n 为偶数），
        每个子区间上用 Simpson 规则：
            ∫_{x_i}^{x_{i+2}} f(x) dx ≈ (h/3)[f(x_i) + 4f(x_{i+1}) + f(x_{i+2})]

        总公式：
            I ≈ (h/3)[f_0 + 4Σf_{odd} + 2Σf_{even} + f_n]

        Parameters
        ----------
        f : callable
            被积函数。
        a, b : float
            积分区间。
        n : int
            子区间数（必须为偶数）。

        Returns
        -------
        float
            积分近似值。
        """
        if n % 2 != 0:
            n += 1  # 确保为偶数

        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])

        integral = y[0] + y[-1]
        integral += 4.0 * np.sum(y[1:-1:2])
        integral += 2.0 * np.sum(y[2:-1:2])
        integral *= h / 3.0

        return integral

    @staticmethod
    def composite_trapezoid(f, a, b, n):
        """
        复合梯形求积公式。

        I ≈ (h/2)[f_0 + 2Σ_{i=1}^{n-1} f_i + f_n]

        Parameters
        ----------
        f : callable
            被积函数。
        a, b : float
            积分区间。
        n : int
            子区间数。

        Returns
        -------
        float
            积分近似值。
        """
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])

        integral = 0.5 * (y[0] + y[-1]) + np.sum(y[1:-1])
        integral *= h

        return integral

    @staticmethod
    def gauss_legendre_composite(f, a, b, n_elements, order=3):
        """
        复合 Gauss-Legendre 求积。

        将 [a,b] 分为 n_elements 个子区间，每个子区间上用
        order 点 Gauss-Legendre 求积。

        Parameters
        ----------
        f : callable
            被积函数。
        a, b : float
            积分区间。
        n_elements : int
            子区间数。
        order : int
            每子区间求积点数（3 或 5）。

        Returns
        -------
        float
            积分近似值。
        """
        if order == 3:
            nodes, weights = GaussQuadrature.gauss_legendre_3point()
        elif order == 5:
            nodes, weights = GaussQuadrature.gauss_legendre_5point()
        else:
            raise ValueError("order 必须为 3 或 5")

        h = (b - a) / n_elements
        integral = 0.0

        for e in range(n_elements):
            x_left = a + e * h
            x_right = x_left + h
            # 坐标变换：ξ ∈ [-1,1] → x ∈ [x_left, x_right]
            for i in range(len(nodes)):
                x = 0.5 * (x_left + x_right) + 0.5 * h * nodes[i]
                integral += 0.5 * h * weights[i] * f(x)

        return integral


class QuadratureExactnessTest:
    """
    求积规则精确度检验。
    基于种子项目 464_gen_hermite_exactness 的思想。
    """

    def __init__(self, quad_nodes, quad_weights):
        """
        初始化求积规则。

        Parameters
        ----------
        quad_nodes : ndarray
            求积节点。
        quad_weights : ndarray
            求积权重。
        """
        self.nodes = np.asarray(quad_nodes)
        self.weights = np.asarray(quad_weights)

    def test_monomial_exactness(self, max_degree, integral_func):
        """
        检验求积规则对单项式 x^k（k = 0, ..., max_degree）的精确度。

        计算相对误差：
            err_k = |I_quad(x^k) - I_exact(x^k)| / |I_exact(x^k)|

        Parameters
        ----------
        max_degree : int
            最大检验次数。
        integral_func : callable
            精确积分函数 integral_func(degree) 返回 x^degree 的精确积分。

        Returns
        -------
        dict
            {degree: relative_error}
        """
        errors = {}
        for degree in range(max_degree + 1):
            # 求积近似值
            quad_value = np.sum(self.weights * (self.nodes ** degree))

            # 精确值
            exact_value = integral_func(degree)

            if exact_value == 0.0:
                rel_error = abs(quad_value)
            else:
                rel_error = abs((quad_value - exact_value) / exact_value)

            errors[degree] = rel_error

        return errors


def compute_phase_field_energy_integral(phi, epsilon, a_func, quadrature_order=5):
    """
    使用高斯求积计算相场能量泛函的积分：
        E[φ] = ∫ [ (ε²/2)|∇φ|² + W(φ) + a(x,y) φ² ] dΩ

    Parameters
    ----------
    phi : ndarray
        序参量场（这里简化为 1D 或 2D 离散场）。
    epsilon : float
        界面宽度。
    a_func : callable
        空间依赖的系数函数。
    quadrature_order : int
        求积阶数。

    Returns
    -------
    float
        能量泛函值。
    """
    # 这里采用离散求和近似
    nx, ny = phi.shape
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    # 计算梯度
    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
    grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)
    grad_sq = grad_x ** 2 + grad_y ** 2

    # 双阱势
    W = 0.25 * (phi ** 2 - 1.0) ** 2

    # 空间坐标
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 能量密度
    energy_density = 0.5 * epsilon ** 2 * grad_sq + W + a_func(X, Y) * phi ** 2

    # 数值积分（复合梯形）
    energy = np.sum(energy_density) * dx * dy

    return energy
