"""
随机域映射模块 (Stochastic Domain Mapping)
=============================================
将随机物理域映射到规则参考域, 实现随机几何的确定性处理。

核心思想:
  当物理域 Ω(ω) 本身是随机的 (如随机边界位置),
  需要将其映射到固定的参考域 Ω̂, 然后在参考域上求解 PDE。

映射:
  x = T(ξ̂; ξ(ω)): Ω̂ → Ω(ω)

  在参考域上的变换方程:
    -∇̂ · (κ̂(ξ̂; ξ) ∇̂û) = f̂(ξ̂; ξ)

  其中:
    κ̂ = κ J^{-T} J^{-1} |det J|  (变换后的扩散系数)
    f̂ = f |det J|                  (变换后的源项)
    J = ∂x/∂ξ̂                     (Jacobian 矩阵)

仿射映射 (1D):
  x = a(ω) + (b(ω) - a(ω)) ξ̂ / L̂

  其中 a(ω), b(ω) 是随机边界位置。
  J = (b(ω) - a(ω)) / L̂
  |det J| = (b(ω) - a(ω)) / L̂

多项式映射:
  x = Σ_{k=0}^{p} c_k(ω) ξ̂^k

  用于更复杂的域变形。
  系数 c_k(ω) 通过边界条件和内部锚点确定。

面积/体积修正:
  变换后的积分:
    ∫_{Ω(ω)} f(x) dx = ∫_{Ω̂} f(T(ξ̂)) |det J| dξ̂

  统计矩:
    E[u](x) = ∫ u(T(ξ̂; ξ), ξ) dμ(ξ)
    需要注意 x 本身也依赖于 ξ (域映射)

应用:
  - 随机边界问题 (腐蚀、磨损)
  - 流固耦合 (变形域)
  - 多孔介质 (随机孔隙几何)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class StochasticDomainMapper:
    """
    随机域映射器: 将随机物理域映射到参考域。

    支持:
    - 仿射映射 (1D 随机区间)
    - 多项式映射 (高阶变形)
    - 球面映射 (随机球域)
    """

    def __init__(
        self,
        reference_domain: Tuple[float, float] = (0.0, 1.0),
        mapping_type: str = 'affine'
    ):
        """
        参数:
            reference_domain: 参考域 (ξ̂_min, ξ̂_max)
            mapping_type: 'affine' | 'polynomial' | 'transcendental'
        """
        self.ref_domain = reference_domain
        self.mapping_type = mapping_type
        self.ref_length = reference_domain[1] - reference_domain[0]

    def affine_map_1d(
        self,
        xi_hat: np.ndarray,
        left_boundary: float,
        right_boundary: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D 仿射映射:
          x = a + (b - a) (ξ̂ - ξ̂_min) / (ξ̂_max - ξ̂_min)

        Jacobian:
          J = dx/dξ̂ = (b - a) / (ξ̂_max - ξ̂_min)

        参数:
            xi_hat: 参考域坐标, shape (N,)
            left_boundary: 随机左边界 a(ω)
            right_boundary: 随机右边界 b(ω)

        返回:
            x: 物理域坐标, shape (N,)
            jacobian: Jacobian 值, shape (N,) (常数)
        """
        a = left_boundary
        b = right_boundary
        L_hat = self.ref_length

        x = a + (b - a) * (xi_hat - self.ref_domain[0]) / L_hat
        J = (b - a) / L_hat  # 常数

        return x, np.full_like(xi_hat, J)

    def polynomial_map_1d(
        self,
        xi_hat: np.ndarray,
        coefficients: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        1D 多项式映射:
          x = Σ_{k=0}^{p} c_k ξ̂^k

        Jacobian:
          J = dx/dξ̂ = Σ_{k=1}^{p} k c_k ξ̂^{k-1}

        参数:
            xi_hat: 参考域坐标, shape (N,)
            coefficients: 多项式系数 [c_0, c_1, ..., c_p]

        返回:
            x: 物理域坐标, shape (N,)
            jacobian: Jacobian 值, shape (N,)
        """
        p = len(coefficients) - 1

        # x = c_0 + c_1 ξ̂ + c_2 ξ̂² + ...
        x = np.zeros_like(xi_hat)
        for k in range(p + 1):
            x += coefficients[k] * xi_hat ** k

        # J = c_1 + 2 c_2 ξ̂ + 3 c_3 ξ̂² + ...
        J = np.zeros_like(xi_hat)
        for k in range(1, p + 1):
            J += k * coefficients[k] * xi_hat ** (k - 1)

        return x, J

    def transcendental_map_1d(
        self,
        xi_hat: np.ndarray,
        amplitude: float = 0.1,
        frequency: float = 2.0 * np.pi,
        base_left: float = 0.0,
        base_right: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        超越函数映射 (用于光滑域变形):
          x = base_left + (base_right - base_left) ξ̂/L̂
              + amplitude × sin(frequency × π ξ̂/L̂)

        Jacobian:
          J = (base_right - base_left)/L̂
              + amplitude × frequency × π/L̂ × cos(frequency × π ξ̂/L̂)

        参数:
            xi_hat: 参考域坐标
            amplitude: 变形幅度
            frequency: 变形频率
            base_left, base_right: 基准边界

        返回:
            x, jacobian
        """
        L_hat = self.ref_length
        xi_norm = (xi_hat - self.ref_domain[0]) / L_hat

        base_length = base_right - base_left
        x = base_left + base_length * xi_norm + amplitude * np.sin(frequency * np.pi * xi_norm)

        J = (base_length / L_hat
             + amplitude * frequency * np.pi / L_hat
             * np.cos(frequency * np.pi * xi_norm))

        return x, J

    def transform_pde_coefficients(
        self,
        kappa_physical: np.ndarray,
        jacobian: np.ndarray,
        source_physical: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        变换 PDE 系数到参考域:

        原方程: -d/dx(κ du/dx) = f(x)
        变换后: -d/dξ̂(κ̂ dû/dξ̂) = f̂(ξ̂)

        其中:
          κ̂ = κ / |J|  (变换后扩散系数)
          f̂ = f |J|    (变换后源项)

        推导:
          d/dx = (1/J) d/dξ̂
          -d/dx(κ du/dx) = -(1/J) d/dξ̂(κ (1/J) dû/dξ̂)
          = -(1/J²) d/dξ̂(κ dû/dξ̂)  (当 J 为常数)
          乘以 |J|: -(1/|J|) d/dξ̂(κ dû/dξ̂) = f
          即: -d/dξ̂(κ/|J| dû/dξ̂) = f |J|

        参数:
            kappa_physical: 物理域扩散系数
            jacobian: Jacobian 值
            source_physical: 物理域源项

        返回:
            kappa_ref: 参考域扩散系数
            source_ref: 参考域源项
        """
        abs_J = np.abs(jacobian)
        abs_J = np.maximum(abs_J, 1e-15)  # 避免除零

        kappa_ref = kappa_physical / abs_J
        source_ref = source_physical * abs_J

        return kappa_ref, source_ref

    def generate_reference_grid(
        self, n_points: int = 101
    ) -> np.ndarray:
        """
        生成参考域上的均匀网格。
        """
        return np.linspace(self.ref_domain[0], self.ref_domain[1], n_points)

    def compute_mapping_distortion(
        self, jacobian: np.ndarray
    ) -> dict:
        """
        计算映射质量指标:

          - 最大/最小 Jacobian: 域变形程度
          - 条件数: max|J| / min|J|
          - 正则性: |J| > 0 处处成立 ⟹ 映射是微分同胚

        参数:
            jacobian: Jacobian 值

        返回:
            dict: 质量指标
        """
        abs_J = np.abs(jacobian)
        return {
            'min_jacobian': float(np.min(abs_J)),
            'max_jacobian': float(np.max(abs_J)),
            'mean_jacobian': float(np.mean(abs_J)),
            'condition_number': float(np.max(abs_J) / max(np.min(abs_J), 1e-30)),
            'is_diffeomorphism': bool(np.all(abs_J > 1e-15)),
        }
