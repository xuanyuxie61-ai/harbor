"""
quadrature_engine.py
================================================================================
高斯求积与蒙特卡洛积分引擎

融合原项目:
  - 947_quadmom   : 正交多项式与高斯求积（矩方法、Jacobi矩阵、Golub-Welsch）
  - 1312_triangle_monte_carlo : 三角形单元上的蒙特卡洛积分

核心科学内容:
  1. Legendre 高斯-勒让德求积规则（一维区间映射）
  2. 基于矩方法的自定义权函数高斯求积
  3. 三角形参考单元上的高斯求积（Dunavant规则）
  4. 三角形单元上的蒙特卡洛积分与方差估计
  5. 参考单元到物理单元的等参映射
================================================================================
"""

import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import eigh_tridiagonal


class GaussLegendreQuadrature:
    """
    高斯-勒让德求积规则。

    在标准区间 [-1, 1] 上，n 点 Gauss-Legendre 求积精确积分次数 ≤ 2n-1 的多项式。

        ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)

    节点 x_i 和权重 w_i 由 Legendre 多项式 P_n(x) 的零点确定。
    通过 Jacobi 矩阵的三对角特征值问题求解（Golub-Welsch 算法）。

    Legendre 多项式的三项递推关系:
        P_{-1}(x) = 0,  P_0(x) = 1
        (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)

    Jacobi 矩阵:
        J = diag(α_0, α_1, ..., α_{n-1}) + diag(β_1, β_2, ..., β_{n-1})^{(+1)} + diag(...)^{(-1)}

    其中 α_k = 0,  β_k = k / sqrt(4k^2 - 1).

    融合原项目 947_quadmom 的 sgqf / moment_method 核心思想。
    """

    def __init__(self, n_points: int):
        if n_points < 1:
            raise ValueError("求积点数必须 ≥ 1")
        self.n_points = n_points
        self._build_rule()

    def _build_rule(self):
        """通过 Jacobi 矩阵特征值问题构造求积规则."""
        n = self.n_points
        # Legendre 多项式的递推系数
        alpha = np.zeros(n)
        beta = np.zeros(n - 1)
        for k in range(1, n):
            beta[k - 1] = k / np.sqrt(4.0 * k * k - 1.0)
        # 求解对称三对角矩阵的特征值和特征向量
        # 使用 scipy 的 eigh_tridiagonal 以提高数值稳定性
        eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
        self.nodes = eigvals
        # 权重 = μ_0 * (第一个分量)^2, μ_0 = 2 (Legendre在[-1,1]上的0阶矩)
        self.weights = 2.0 * eigvecs[0, :] ** 2

    def integrate_1d(self, f, a: float = -1.0, b: float = 1.0) -> float:
        """
        在区间 [a, b] 上计算 ∫ f(x) dx.

        变量替换: x = (b+a)/2 + (b-a)/2 * t,  t ∈ [-1, 1]

            ∫_a^b f(x) dx = (b-a)/2 ∫_{-1}^{1} f((b+a)/2 + (b-a)/2 * t) dt
        """
        if a >= b:
            raise ValueError("积分上限必须大于下限")
        scale = 0.5 * (b - a)
        shift = 0.5 * (b + a)
        x_phys = shift + scale * self.nodes
        return scale * np.sum(self.weights * f(x_phys))

    def get_nodes_weights_1d(self, a: float = -1.0, b: float = 1.0):
        """返回映射到 [a,b] 上的节点和权重."""
        scale = 0.5 * (b - a)
        shift = 0.5 * (b + a)
        return shift + scale * self.nodes, scale * self.weights


class TriangleGaussianQuadrature:
    """
    三角形单元上的高斯求积规则（Dunavant规则，7点规则精确到5次多项式）。

    参考三角形 T_ref 的顶点为:
        v1 = (0, 0),  v2 = (1, 0),  v3 = (0, 1)

    面积坐标 (L1, L2, L3) 满足 L1 + L2 + L3 = 1, Li ≥ 0.

    7点 Dunavant 规则:
        - 重心点: (1/3, 1/3, 1/3), 权重 = 9/40
        - 3个边中点型: 轮换 (a, a, 1-2a), a = (6+√15)/21, 权重 = (155-√15)/1200
        - 3个内部点型: 轮换 (b, b, 1-2b), b = (6-√15)/21, 权重 = (155+√15)/1200

    物理三角形 T 上的积分:
        ∫∫_T f(x,y) dx dy = |det(J)| ∫∫_{T_ref} f(x(ξ,η), y(ξ,η)) dξ dη

    其中 J 为等参映射的 Jacobian 矩阵。
    """

    def __init__(self, order: int = 7):
        self.order = order
        self._build_dunavant_7()

    def _build_dunavant_7(self):
        """构造7点Dunavant规则."""
        # 面积坐标节点和权重
        a1 = (6.0 + np.sqrt(15.0)) / 21.0
        a2 = (6.0 - np.sqrt(15.0)) / 21.0
        w1 = (155.0 - np.sqrt(15.0)) / 1200.0
        w2 = (155.0 + np.sqrt(15.0)) / 1200.0
        w0 = 9.0 / 40.0

        self.ref_nodes = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [a1, a1, 1.0 - 2.0 * a1],
            [a1, 1.0 - 2.0 * a1, a1],
            [1.0 - 2.0 * a1, a1, a1],
            [a2, a2, 1.0 - 2.0 * a2],
            [a2, 1.0 - 2.0 * a2, a2],
            [1.0 - 2.0 * a2, a2, a2],
        ])
        self.ref_weights = np.array([w0, w1, w1, w1, w2, w2, w2])

    def integrate_triangle(self, f, vertices: np.ndarray) -> float:
        """
        在物理三角形上积分函数 f(x, y).

        参数:
            f : callable, 接受 (n, 2) 数组返回 (n,) 值
            vertices : (3, 2) 三角形顶点坐标
        """
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (3, 2):
            raise ValueError("顶点数组形状必须为 (3, 2)")

        # 等参映射: x = x1 + (x2-x1)*ξ + (x3-x1)*η
        x1, x2, x3 = vertices[0, 0], vertices[1, 0], vertices[2, 0]
        y1, y2, y3 = vertices[0, 1], vertices[1, 1], vertices[2, 1]
        jac = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        if abs(det_j) < 1.0e-14:
            raise ValueError("退化三角形: Jacobian行列式接近零")

        # 面积坐标 → 参考坐标 (ξ, η)
        # ξ = L2, η = L3, L1 = 1-ξ-η
        xi = self.ref_nodes[:, 1]
        eta = self.ref_nodes[:, 2]
        x_phys = x1 + jac[0, 0] * xi + jac[0, 1] * eta
        y_phys = y1 + jac[1, 0] * xi + jac[1, 1] * eta
        pts = np.column_stack((x_phys, y_phys))
        # det_j = 2 * Area_ref,  Dunavant权重已针对 Area_ref=0.5 归一化
        # 因此积分 = 0.5 * abs(det_j) * Σ w_i f_i = Area * Σ w_i f_i
        return 0.5 * abs(det_j) * np.sum(self.ref_weights * f(pts))

    def get_physical_nodes_weights(self, vertices: np.ndarray):
        """返回物理三角形上的求积节点和权重."""
        vertices = np.asarray(vertices, dtype=float)
        x1, x2, x3 = vertices[0, 0], vertices[1, 0], vertices[2, 0]
        y1, y2, y3 = vertices[0, 1], vertices[1, 1], vertices[2, 1]
        jac = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        det_j = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]

        xi = self.ref_nodes[:, 1]
        eta = self.ref_nodes[:, 2]
        x_phys = x1 + jac[0, 0] * xi + jac[0, 1] * eta
        y_phys = y1 + jac[1, 0] * xi + jac[1, 1] * eta
        pts = np.column_stack((x_phys, y_phys))
        w_phys = 0.5 * abs(det_j) * self.ref_weights
        return pts, w_phys


class TriangleMonteCarlo:
    """
    三角形单元上的蒙特卡洛积分，含方差估计与收敛性分析。

    融合原项目 1312_triangle_monte_carlo 的核心算法:
      - reference_to_physical_t3 : 参考单元到物理单元的映射
      - triangle_unit_sample_01  : 均匀采样（修正为更好的采样策略）
      - triangle_area            : 三角形面积

    蒙特卡洛估计:
        I ≈ |T| / N Σ_{i=1}^{N} f(x_i, y_i)

    标准误差:
        SE = |T| / N * sqrt( Σ (f_i - f̄)^2 / (N-1) )

    95% 置信区间:
        I ∈ [I_est - 1.96 SE, I_est + 1.96 SE]
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def triangle_area(vertices: np.ndarray) -> float:
        """计算三角形面积（鞋带公式）."""
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (3, 2):
            raise ValueError("顶点数组形状必须为 (3, 2)")
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        x3, y3 = vertices[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def _sample_unit_triangle(self, n: int) -> np.ndarray:
        """
        在单位三角形（顶点 (0,0), (1,0), (0,1)）内均匀采样。

        使用根号变换确保均匀分布:
            u1, u2 ~ U[0,1]
            ξ = 1 - sqrt(u1)
            η = sqrt(u1) * (1 - u2)
        """
        u1 = self.rng.random(n)
        u2 = self.rng.random(n)
        xi = 1.0 - np.sqrt(u1)
        eta = np.sqrt(u1) * (1.0 - u2)
        return np.column_stack((xi, eta))

    def _reference_to_physical(self, vertices: np.ndarray, ref_pts: np.ndarray) -> np.ndarray:
        """
        将参考三角形内的点映射到物理三角形。

        映射公式（线性等参）:
            x = x1 + (x2-x1)*ξ + (x3-x1)*η
            y = y1 + (y2-y1)*ξ + (y3-y1)*η
        """
        vertices = np.asarray(vertices, dtype=float)
        ref_pts = np.asarray(ref_pts, dtype=float)
        v1 = vertices[0]
        v2 = vertices[1]
        v3 = vertices[2]
        xi = ref_pts[:, 0]
        eta = ref_pts[:, 1]
        x = v1[0] + (v2[0] - v1[0]) * xi + (v3[0] - v1[0]) * eta
        y = v1[1] + (v2[1] - v1[1]) * xi + (v3[1] - v1[1]) * eta
        return np.column_stack((x, y))

    def integrate(self, f, vertices: np.ndarray, n_samples: int = 10000) -> dict:
        """
        在物理三角形上执行蒙特卡洛积分。

        返回:
            {
                'estimate': 积分估计值,
                'std_error': 标准误差,
                'ci_lower': 95% CI 下限,
                'ci_upper': 95% CI 上限,
                'area': 三角形面积
            }
        """
        area = self.triangle_area(vertices)
        if area < 1.0e-14:
            raise ValueError("退化三角形面积接近零")

        ref_pts = self._sample_unit_triangle(n_samples)
        phys_pts = self._reference_to_physical(vertices, ref_pts)
        values = f(phys_pts)
        values = np.asarray(values, dtype=float).flatten()

        mean_val = np.mean(values)
        estimate = area * mean_val

        if n_samples > 1:
            var_val = np.var(values, ddof=1)
            std_error = area * np.sqrt(var_val / n_samples)
        else:
            std_error = np.inf

        ci_lower = estimate - 1.96 * std_error
        ci_upper = estimate + 1.96 * std_error

        return {
            "estimate": float(estimate),
            "std_error": float(std_error),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "area": float(area),
        }


class MomentMethodQuadrature:
    """
    基于矩方法的自定义权函数高斯求积。

    融合原项目 947_quadmom 的 moment_method 核心算法。

    给定权函数 w(x) 的前 2n+1 个矩:
        m_k = ∫_a^b x^k w(x) dx,  k = 0, 1, ..., 2n

    构造 (n+1)×(n+1) Hankel 矩阵:
        H_{ij} = m_{i+j},  i,j = 0,...,n

    对 H 进行 Cholesky 分解 H = R^T R，得到 Jacobi 矩阵的递推系数:
        α_i = R_{i,i+1}/R_{i,i} - R_{i-1,i}/R_{i-1,i-1}   (R_{-1,*} = 0)
        β_i = R_{i+1,i+1} / R_{i,i}

    然后通过三对角矩阵的特征值问题求得求积节点和权重。

    应用场景: 电机绕组电流分布的非标准权函数积分。
    """

    def __init__(self, moments: np.ndarray, check_positive: bool = True):
        """
        参数:
            moments : 权函数的矩序列 m_0, m_1, ..., m_{2n}
        """
        moments = np.asarray(moments, dtype=float)
        if len(moments) % 2 == 0:
            raise ValueError("矩序列长度必须为奇数 (2n+1)")
        self.n = (len(moments) - 1) // 2
        self.moments = moments
        if check_positive:
            if moments[0] <= 0.0:
                raise ValueError("0阶矩必须为正")
        self._build_from_moments()

    def _build_from_moments(self):
        """由矩序列构造求积规则."""
        n = self.n
        # 构造 Hankel 矩阵
        H = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(n + 1):
                H[i, j] = self.moments[i + j]

        # Cholesky 分解
        try:
            R = np.linalg.cholesky(H).T  # 上三角
        except np.linalg.LinAlgError as exc:
            raise ValueError("Hankel矩阵不正定，矩序列可能不一致") from exc

        # 提取 Jacobi 矩阵系数
        alpha = np.zeros(n)
        alpha[0] = R[0, 1] / R[0, 0]
        for i in range(1, n):
            alpha[i] = R[i, i + 1] / R[i, i] - R[i - 1, i] / R[i - 1, i - 1]

        beta = np.zeros(n - 1)
        for i in range(n - 1):
            beta[i] = R[i + 1, i + 1] / R[i, i]

        # 求解特征值问题
        if n == 1:
            self.nodes = np.array([alpha[0]])
            self.weights = np.array([self.moments[0]])
        else:
            eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
            self.nodes = eigvals
            self.weights = self.moments[0] * eigvecs[0, :] ** 2

    def integrate(self, f) -> float:
        """计算 ∫ f(x) w(x) dx."""
        return np.sum(self.weights * f(self.nodes))
