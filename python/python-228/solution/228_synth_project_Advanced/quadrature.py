"""
quadrature.py — 高精度求积规则
================================

融合种子项目:
  - 470_gl_fast_rule : 无迭代 Gauss-Legendre 节点计算
  - 678_line_fekete_rule : Fekete 点求积
  - 1319_triangle_symq_to_ref : 三角形对称求积

本模块实现簇射能量沉积积分所需的高精度求积方法:

1. Gauss-Legendre 求积 (任意阶):
   integral_{-1}^{1} f(x) dx ≈ sum_i w_i * f(x_i)
   其中 x_i 为 P_n(x) 的零点, w_i = 2/((1-x_i^2)*[P'_n(x_i)]^2)

2. Gauss-Chebyshev 求积 (带权重 1/sqrt(1-x^2)):
   integral_{-1}^{1} f(x)/sqrt(1-x^2) dx ≈ (pi/n) * sum_i f(x_i)
   x_i = cos((2i-1)*pi/(2n))

3. 三角形求积 (Xiao-Gimbutas 规则):
   integral_T f(x,y) dA = |T| * sum_i w_i * f(x_i, y_i)

4. 自适应 Simpson 求积 (误差控制):
   S = (h/6)*(f(a) + 4*f(m) + f(b))
   误差: |S(a,m) + S(m,b) - S(a,b)| / 15
"""

import math
from typing import List, Tuple, Callable, Optional


class GaussLegendreQuadrature:
    """Gauss-Legendre 求积器"""

    def __init__(self, n_points: int = 32):
        self.n = n_points
        self._nodes, self._weights = self._compute_nodes_weights()

    @property
    def nodes(self) -> List[float]:
        return self._nodes

    @property
    def weights(self) -> List[float]:
        return self._weights

    def integrate(
        self, func: Callable[[float], float],
        a: float = -1.0, b: float = 1.0,
    ) -> float:
        """
        在 [a,b] 上积分

        变换: x = (b-a)/2 * t + (a+b)/2
              dx = (b-a)/2 * dt
        integral_a^b f(x) dx = (b-a)/2 * sum_i w_i * f((b-a)/2 * t_i + (a+b)/2)
        """
        half_range = (b - a) / 2.0
        midpoint = (a + b) / 2.0
        result = 0.0
        for i in range(self.n):
            x = half_range * self._nodes[i] + midpoint
            result += self._weights[i] * func(x)
        return half_range * result

    def integrate_weighted(
        self, func: Callable[[float], float],
        weight_func: Callable[[float], float],
        a: float = -1.0, b: float = 1.0,
    ) -> float:
        """带权重函数的积分: integral f(x) * w(x) dx"""
        half_range = (b - a) / 2.0
        midpoint = (a + b) / 2.0
        result = 0.0
        for i in range(self.n):
            x = half_range * self._nodes[i] + midpoint
            result += self._weights[i] * func(x) * weight_func(x)
        return half_range * result

    def _compute_nodes_weights(self) -> Tuple[List[float], List[float]]:
        """
        计算 GL 节点和权重 (Newton 迭代 + 渐近初值)

        对于大 n, 使用渐近公式 (Bogaert 2014):
          x_k ≈ cos(theta_k)
          theta_k = (k + 3/4) * pi / (n + 1/2) * (1 + correction)

        Newton 迭代:
          x_{m+1} = x_m - P_n(x_m) / P'_n(x_m)
        """
        nodes = []
        weights = []

        for k in range(self.n):
            # 初始猜测 (Chebyshev + 修正)
            mu = (k + 0.75) / (self.n + 0.5)
            x = math.cos(math.pi * mu)

            # Newton 迭代
            for _ in range(50):
                p_prev = 1.0
                p_curr = x
                for j in range(2, self.n + 1):
                    p_next = ((2.0 * j - 1.0) * x * p_curr - (j - 1.0) * p_prev) / j
                    p_prev = p_curr
                    p_curr = p_next

                # P'_n(x) = n * (x * P_n - P_{n-1}) / (x^2 - 1)
                denom = x * x - 1.0
                if abs(denom) < 1e-30:
                    denom = 1e-30 if denom >= 0 else -1e-30
                dp = self.n * (x * p_curr - p_prev) / denom

                dx = -p_curr / (dp if abs(dp) > 1e-30 else 1e-30)
                x += dx
                if abs(dx) < 1e-15:
                    break

            # 最终 P_n 和 P'_n
            p_prev = 1.0
            p_curr = x
            for j in range(2, self.n + 1):
                p_next = ((2.0 * j - 1.0) * x * p_curr - (j - 1.0) * p_prev) / j
                p_prev = p_curr
                p_curr = p_next
            denom = x * x - 1.0
            if abs(denom) < 1e-30:
                denom = 1e-30
            dp = self.n * (x * p_curr - p_prev) / denom

            nodes.append(x)
            w = 2.0 / ((1.0 - x * x) * dp * dp) if abs(dp) > 1e-30 else 0.0
            weights.append(w)

        return nodes, weights


class GaussChebyshevQuadrature:
    """Gauss-Chebyshev 求积 (第一类)"""

    def __init__(self, n_points: int = 32):
        self.n = n_points

    def integrate(
        self, func: Callable[[float], float],
        a: float = -1.0, b: float = 1.0,
    ) -> float:
        """
        Gauss-Chebyshev 求积:
          integral_{-1}^{1} f(x) / sqrt(1-x^2) dx ≈ (pi/n) * sum_i f(x_i)
          x_i = cos((2i-1)*pi/(2n))
        """
        result = 0.0
        for i in range(self.n):
            x = math.cos((2.0 * i + 1.0) * math.pi / (2.0 * self.n))
            result += func(x)
        return math.pi / self.n * result


class TriangleQuadrature:
    """三角形域上的求积 (Xiao-Gimbutas 规则)"""

    # 低阶规则系数 (7点规则, 精度5)
    RULE_7_NODES = [
        (1.0 / 3.0, 1.0 / 3.0, 0.2250000000000),
        (0.05971587178977, 0.4701420641051, 0.1323941527885),
        (0.4701420641051, 0.05971587178977, 0.1323941527885),
        (0.4701420641051, 0.4701420641051, 0.1323941527885),
        (0.7974269853531, 0.1012865073235, 0.1259391805448),
        (0.1012865073235, 0.7974269853531, 0.1259391805448),
        (0.1012865073235, 0.1012865073235, 0.1259391805448),
    ]

    def integrate_on_reference(
        self, func: Callable[[float, float], float],
    ) -> float:
        """
        在参考三角形 (0,0)-(1,0)-(0,1) 上积分

        integral_T f(x,y) dA = sum_i w_i * f(x_i, y_i)
        (参考三角形面积 = 1/2)
        """
        result = 0.0
        for xi, eta, w in self.RULE_7_NODES:
            result += w * func(xi, eta)
        return result * 0.5  # 乘以参考三角形面积

    def integrate_on_physical(
        self, func: Callable[[float, float], float],
        vertices: List[Tuple[float, float]],
    ) -> float:
        """
        在物理三角形上积分

        映射: (x,y) = (1-xi-eta)*V0 + xi*V1 + eta*V2
        Jacobian: |J| = |det([V1-V0, V2-V0])|
        """
        if len(vertices) != 3:
            raise ValueError("需要3个顶点")
        v0, v1, v2 = vertices

        # Jacobian
        J = abs(
            (v1[0] - v0[0]) * (v2[1] - v0[1])
            - (v2[0] - v0[0]) * (v1[1] - v0[1])
        )

        def mapped_func(xi: float, eta: float) -> float:
            x = (1.0 - xi - eta) * v0[0] + xi * v1[0] + eta * v2[0]
            y = (1.0 - xi - eta) * v0[1] + xi * v1[1] + eta * v2[1]
            return func(x, y)

        return J * self.integrate_on_reference(mapped_func)


class AdaptiveSimpson:
    """自适应 Simpson 求积"""

    def integrate(
        self, func: Callable[[float], float],
        a: float, b: float,
        tol: float = 1e-10,
        max_depth: int = 50,
    ) -> Tuple[float, float]:
        """
        自适应 Simpson 求积

        S(a,b) = (h/6) * [f(a) + 4*f(m) + f(b)], h = b-a, m = (a+b)/2
        误差估计: |S(a,m) + S(m,b) - S(a,b)| / 15

        返回: (积分值, 误差估计)
        """
        fa = func(a)
        fb = func(b)
        m = (a + b) / 2.0
        fm = func(m)
        S_whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)

        result, error = self._adaptive_recursive(
            func, a, b, fa, fb, fm, S_whole, tol, max_depth,
        )
        return result, error

    def _adaptive_recursive(
        self, func, a, b, fa, fb, fm, S_whole, tol, depth,
    ) -> Tuple[float, float]:
        m = (a + b) / 2.0
        h = (b - a) / 2.0
        ml = (a + m) / 2.0
        mr = (m + b) / 2.0
        fml = func(ml)
        fmr = func(mr)

        S_left = h / 6.0 * (fa + 4.0 * fml + fm) / 2.0
        S_right = h / 6.0 * (fm + 4.0 * fmr + fb) / 2.0
        S_combined = S_left + S_right

        error_est = abs(S_combined - S_whole) / 15.0

        if depth <= 0 or error_est < tol:
            return S_combined + (S_combined - S_whole) / 15.0, error_est

        left_val, left_err = self._adaptive_recursive(
            func, a, m, fa, fm, fml, S_left, tol / 2.0, depth - 1,
        )
        right_val, right_err = self._adaptive_recursive(
            func, m, b, fm, fb, fmr, S_right, tol / 2.0, depth - 1,
        )
        return left_val + right_val, left_err + right_err


def energy_deposition_integral(
    profile_func: Callable[[float], float],
    depth_range: Tuple[float, float],
    n_quad: int = 64,
) -> Tuple[float, float]:
    """
    计算能量沉积积分 (主入口)

    E_dep = integral_{t1}^{t2} Gamma(t) dt

    使用 Gauss-Legendre 求积, 返回 (积分值, 误差)。
    """
    gl = GaussLegendreQuadrature(n_quad)
    a, b = depth_range

    # 自适应细化
    n_segments = 4
    total = 0.0
    seg_width = (b - a) / n_segments

    for i in range(n_segments):
        seg_a = a + i * seg_width
        seg_b = seg_a + seg_width
        total += gl.integrate(profile_func, seg_a, seg_b)

    # 误差估计 ( Richardson 外推)
    n_fine = n_quad * 2
    gl_fine = GaussLegendreQuadrature(n_fine)
    fine_total = gl_fine.integrate(profile_func, a, b)
    error_est = abs(fine_total - total) / 15.0

    return fine_total, error_est
