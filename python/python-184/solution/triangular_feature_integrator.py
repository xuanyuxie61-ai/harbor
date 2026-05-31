"""
Symmetric Quadrature over Triangles for 2D Embedded Feature Integration
=======================================================================
源自种子项目 1316_triangle_symq_rule (Symmetric quadrature on reference triangle)。

在 time series 分析中，将三个连续时间窗口的特征向量映射为二维单纯形（三角形），
可以提取高阶交互特征。

参考三角形（标准单纯形）：
    T_ref = {(x,y) : x>=0, y>=0, x+y<=1}
    面积 = 1/2

对称求积规则：
    ∫_{T_ref} f(x,y) dxdy ≈ sum_k w_k f(x_k, y_k)
    节点 (x_k, y_k) 和权重 w_k 预计算至 50 阶精度。

本模块实现低阶对称求积（1 到 5 阶），用于：
- 2D 嵌入特征空间中的积分特征提取
- 三角区域上的概率密度积分
- 三变量交互效应的数值聚合

数学：
对于参考三角形，单项式积分有解析公式：
    ∫_{T_ref} x^m y^n dxdy = m! n! / (m+n+2)!

对于一般三角形顶点 (v1,v2,v3)，通过仿射变换：
    x = v1 + s (v2-v1) + t (v3-v1),  (s,t) ∈ T_ref
    dxdy = 2 |Area| ds dt
    ∫_{Δ} f(x,y) dxdy = 2 |Area| ∫_{T_ref} f(φ(s,t)) ds dt
"""

import numpy as np


class TriangularFeatureIntegrator:
    """
    三角形对称求积规则，用于 2D 嵌入特征积分。
    """

    def __init__(self, order: int = 3):
        if order < 1 or order > 7:
            raise ValueError("Order must be between 1 and 7.")
        self.order = order
        self.nodes, self.weights = self._get_rule(order)

    def _get_rule(self, p: int) -> tuple:
        """
        返回参考三角形上的求积节点（s,t 坐标）和权重。
        节点在重心坐标 (λ1,λ2,λ3) 下对称分布。
        """
        if p == 1:
            # 1 点规则，1 阶精度（重心）
            nodes = np.array([[1.0/3.0, 1.0/3.0]])
            weights = np.array([0.5])
        elif p == 2:
            # 3 点规则，2 阶精度
            nodes = np.array([
                [2.0/3.0, 1.0/6.0],
                [1.0/6.0, 2.0/3.0],
                [1.0/6.0, 1.0/6.0]
            ])
            weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        elif p == 3:
            # 4 点规则，3 阶精度
            a = 1.0 / 3.0
            b = 0.6
            c = 0.2
            w1 = -27.0 / 96.0
            w2 = 25.0 / 96.0
            nodes = np.array([
                [a, a],
                [b, c],
                [c, b],
                [c, c]
            ])
            weights = np.array([w1 * 0.5, w2 * 0.5, w2 * 0.5, w2 * 0.5])
        elif p == 4:
            # 6 点规则，4 阶精度（简化版）
            a1 = 0.108103018168070
            a2 = 0.445948490915965
            w1 = 0.223381589678011
            w2 = 0.109951743655322
            nodes = np.array([
                [a1, a1], [1.0 - 2.0*a1, a1], [a1, 1.0 - 2.0*a1],
                [a2, a2], [1.0 - 2.0*a2, a2], [a2, 1.0 - 2.0*a2]
            ])
            weights = np.array([w1, w1, w1, w2, w2, w2]) * 0.5
        elif p == 5:
            # 7 点规则，5 阶精度
            a = 1.0 / 3.0
            b = 0.797426985353087
            c = 0.101286507323456
            d = 0.059715871789770
            e = 0.470142064105115
            w1 = 0.225000000000000
            w2 = 0.125939180544827
            w3 = 0.132394152788506
            nodes = np.array([
                [a, a],
                [b, c], [c, b], [c, c],
                [d, e], [e, d], [e, e]
            ])
            weights = np.array([w1, w2, w2, w2, w3, w3, w3]) * 0.5
        else:
            # 默认用 3 阶
            return self._get_rule(3)
        return nodes, weights

    def integrate_reference(self, f: callable) -> float:
        """
        在参考三角形上积分 f(s,t)。
        """
        vals = np.array([f(s, t) for s, t in self.nodes])
        return float(np.dot(self.weights, vals))

    def integrate_triangle(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                           f: callable) -> float:
        """
        在一般三角形 Δ(v1,v2,v3) 上积分 f(x,y)。
        变换: x = v1 + s*(v2-v1) + t*(v3-v1)
        Jacobian = 2 * Area
        """
        area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
        if area < 1e-15:
            return 0.0

        def transform_f(s, t):
            x = v1 + s * (v2 - v1) + t * (v3 - v1)
            return f(x[0], x[1])

        return 2.0 * area * self.integrate_reference(transform_f)

    def extract_triangular_features(self, feature_triplets: np.ndarray) -> np.ndarray:
        """
        将时间序列的三元组特征映射为三角形积分特征。
        feature_triplets: shape (N, 3, 2) 或 (N, 3)
        若 shape 为 (N,3)，将映射到二维单纯形。
        """
        if feature_triplets.ndim == 2 and feature_triplets.shape[1] == 3:
            # 映射到 2D 单纯形：使用重心坐标展开
            N = feature_triplets.shape[0]
            triplets_2d = np.zeros((N, 3, 2))
            # 固定参考三角形顶点
            ref_v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]])
            for i in range(N):
                # 加权平均作为三角形顶点位置
                w = feature_triplets[i]
                w = np.maximum(w, 0.0)
                s = w.sum()
                if s > 1e-12:
                    w = w / s
                for j in range(3):
                    triplets_2d[i, j] = ref_v[j] * w[j]
            feature_triplets = triplets_2d

        if feature_triplets.ndim != 3 or feature_triplets.shape[1] != 3 or feature_triplets.shape[2] != 2:
            raise ValueError("feature_triplets must have shape (N,3,2) or (N,3)")

        N = feature_triplets.shape[0]
        features = np.zeros(N)
        for i in range(N):
            v1, v2, v3 = feature_triplets[i]
            # 特征：三角形上 ||x - centroid||^2 的积分
            centroid = (v1 + v2 + v3) / 3.0
            def g(x, y):
                return (x - centroid[0])**2 + (y - centroid[1])**2
            features[i] = self.integrate_triangle(v1, v2, v3, g)

        # 归一化
        if features.max() > 1e-12:
            features = features / features.max()
        return features

    def verify_monomial(self, m: int, n: int) -> dict:
        """
        验证单项式 x^m y^n 在参考三角形上的积分精度。
        解析值：m! n! / (m+n+2)!
        """
        import math
        exact = math.factorial(m) * math.factorial(n) / math.factorial(m + n + 2)
        numerical = self.integrate_reference(lambda s, t: s**m * t**n)
        return {
            "exact": exact,
            "numerical": numerical,
            "error": abs(numerical - exact),
            "order": self.order
        }
