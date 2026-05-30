
import numpy as np
from itertools import product


class SparseGridCollocation:

    def __init__(self, dim, level, rule='cc'):
        self.dim = dim
        self.level = level
        self.rule = rule
        self.points, self.weights = self._build_sparse_grid()
        self.n_points = len(self.weights)

    def _one_d_rule(self, level_k):
        if level_k == 0:
            return np.array([0.0]), np.array([2.0])

        if self.rule == 'cc':
            n = 2 ** level_k
            if n == 1:
                return np.array([0.0]), np.array([2.0])
            j = np.arange(n + 1)
            x = np.cos(j * np.pi / n)

            w = np.ones(n + 1)
            w[0] = 0.5
            w[-1] = 0.5

            w = w * 2.0 / n
            return x, w
        elif self.rule == 'gl':

            n = level_k + 1

            j = np.arange(1, n + 1)
            x = np.cos((2 * j - 1) * np.pi / (2 * n))
            w = np.ones(n) * 2.0 / n
            return x, w
        else:
            raise ValueError(f"未知求积规则: {self.rule}")

    def _build_sparse_grid(self):
        points_list = []
        weights_list = []


        rules_1d = {}
        for l in range(self.level + 1):
            rules_1d[l] = self._one_d_rule(l)


        def diff_rule(l):
            if l == 0:
                return rules_1d[0]
            x_h, w_h = rules_1d[l]
            x_l, w_l = rules_1d[l - 1]

            return x_h, w_h


        max_sum = self.level + self.dim - 1
        for multi_index in product(range(self.level + 1), repeat=self.dim):
            if sum(multi_index) > max_sum:
                continue


            coeff = self._smolyak_coefficient(multi_index, self.level, self.dim)
            if abs(coeff) < 1e-15:
                continue


            x_coords = [rules_1d[m][0] for m in multi_index]
            w_coords = [rules_1d[m][1] for m in multi_index]

            for pt_tuple in product(*x_coords):
                w = coeff
                for wi, idx in zip(w_coords, pt_tuple):

                    pos = np.argmin(np.abs(wi[0] - idx))
                    w *= wi[1][pos]
                points_list.append(np.array(pt_tuple))
                weights_list.append(w)

        if len(points_list) == 0:
            return np.zeros((1, self.dim)), np.array([1.0])

        points = np.array(points_list)
        weights = np.array(weights_list)


        unique_pts = []
        unique_w = []
        tol = 1e-10
        for i, pt in enumerate(points):
            found = False
            for j, upt in enumerate(unique_pts):
                if np.linalg.norm(pt - upt) < tol:
                    unique_w[j] += weights[i]
                    found = True
                    break
            if not found:
                unique_pts.append(pt)
                unique_w.append(weights[i])

        return np.array(unique_pts), np.array(unique_w)

    def _smolyak_coefficient(self, multi_index, L, D):
        s = sum(multi_index)
        if s > L + D - 1:
            return 0.0
        k = L + D - s - 1
        if k < 0 or k > D - 1:
            return 0.0

        comb = np.math.comb(D - 1, k)
        return (-1) ** k * comb

    def integrate(self, func):
        values = func(self.points)

        result = np.zeros_like(values[0])
        for i in range(self.n_points):
            result += self.weights[i] * values[i]
        return result

    def compute_statistics(self, func):
        values = func(self.points)
        mean = np.zeros_like(values[0])
        mean_sq = np.zeros_like(values[0])
        for i in range(self.n_points):
            mean += self.weights[i] * values[i]
            mean_sq += self.weights[i] * values[i] ** 2
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0.0)
        return mean, variance


class StochasticAlphaDynamo:

    def __init__(self, base_alpha, sigma_alpha, spatial_modes,
                 dim_random=3, sg_level=3):
        self.base_alpha = base_alpha
        self.sigma_alpha = sigma_alpha
        self.spatial_modes = spatial_modes[:dim_random]
        self.dim_random = dim_random
        self.sg = SparseGridCollocation(dim_random, sg_level, rule='cc')

    def alpha_realization(self, xi, nodes, theta, phi):
        alpha = self.base_alpha.copy()
        for i, (l, m) in enumerate(self.spatial_modes):

            from special_functions import associated_legendre
            P_lm = associated_legendre(l, abs(m), np.cos(theta))
            mode = P_lm * np.cos(m * phi)
            alpha += self.sigma_alpha * xi[i] * mode
        return alpha

    def estimate_reversal_probability(self, dipole_trajectories):
        n_samples, n_times = dipole_trajectories.shape
        sign0 = np.sign(dipole_trajectories[:, 0])
        reversals = np.zeros(n_times)
        for t in range(1, n_times):
            sign_t = np.sign(dipole_trajectories[:, t])
            reversals[t] = np.mean(sign_t * sign0 < 0)
        return reversals
