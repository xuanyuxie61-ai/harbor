
import numpy as np


class TriangularFeatureIntegrator:

    def __init__(self, order: int = 3):
        if order < 1 or order > 7:
            raise ValueError("Order must be between 1 and 7.")
        self.order = order
        self.nodes, self.weights = self._get_rule(order)

    def _get_rule(self, p: int) -> tuple:
        if p == 1:

            nodes = np.array([[1.0/3.0, 1.0/3.0]])
            weights = np.array([0.5])
        elif p == 2:

            nodes = np.array([
                [2.0/3.0, 1.0/6.0],
                [1.0/6.0, 2.0/3.0],
                [1.0/6.0, 1.0/6.0]
            ])
            weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0])
        elif p == 3:

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

            return self._get_rule(3)
        return nodes, weights

    def integrate_reference(self, f: callable) -> float:
        vals = np.array([f(s, t) for s, t in self.nodes])
        return float(np.dot(self.weights, vals))

    def integrate_triangle(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                           f: callable) -> float:
        area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
        if area < 1e-15:
            return 0.0

        def transform_f(s, t):
            x = v1 + s * (v2 - v1) + t * (v3 - v1)
            return f(x[0], x[1])

        return 2.0 * area * self.integrate_reference(transform_f)

    def extract_triangular_features(self, feature_triplets: np.ndarray) -> np.ndarray:
        if feature_triplets.ndim == 2 and feature_triplets.shape[1] == 3:

            N = feature_triplets.shape[0]
            triplets_2d = np.zeros((N, 3, 2))

            ref_v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0)/2.0]])
            for i in range(N):

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

            centroid = (v1 + v2 + v3) / 3.0
            def g(x, y):
                return (x - centroid[0])**2 + (y - centroid[1])**2
            features[i] = self.integrate_triangle(v1, v2, v3, g)


        if features.max() > 1e-12:
            features = features / features.max()
        return features

    def verify_monomial(self, m: int, n: int) -> dict:
        import math
        exact = math.factorial(m) * math.factorial(n) / math.factorial(m + n + 2)
        numerical = self.integrate_reference(lambda s, t: s**m * t**n)
        return {
            "exact": exact,
            "numerical": numerical,
            "error": abs(numerical - exact),
            "order": self.order
        }
