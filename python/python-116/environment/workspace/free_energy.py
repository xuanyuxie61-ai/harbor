
import numpy as np


class SparseGridIntegration:

    def __init__(self, dim_num, level_max):
        if dim_num < 1:
            raise ValueError("dim_num 必须至少为 1。")
        if level_max < 0:
            raise ValueError("level_max 必须非负。")
        self.dim_num = dim_num
        self.level_max = level_max
        self.points = []
        self.weights = []
        self._build_sparse_grid()

    def _index_to_level(self, index):
        return index

    def _univariate_nodes_weights(self, level):
        if level == 0:
            return np.array([0.0]), np.array([2.0])
        n = (1 << level) + 1
        j = np.arange(n)
        x = np.cos(j * np.pi / (n - 1))

        w = np.ones(n)
        w[0] = 0.5
        w[-1] = 0.5
        w = w * (2.0 / (n - 1))
        return x, w

    def _build_sparse_grid(self):
        dim = self.dim_num
        L = self.level_max


        indices = []

        def gen_indices(pos, current, current_sum):
            if pos == dim:
                if current_sum <= L + dim - 1:
                    indices.append(current.copy())
                return
            for idx in range(L + 1):
                if current_sum + idx > L + dim - 1:
                    break
                current.append(idx)
                gen_indices(pos + 1, current, current_sum + idx)
                current.pop()

        gen_indices(0, [], 0)


        point_dict = {}
        for idx in indices:

            nodes_list = []
            weights_list = []
            for d in range(dim):
                x_d, w_d = self._univariate_nodes_weights(idx[d])
                nodes_list.append(x_d)
                weights_list.append(w_d)


            coef = self._compute_coefficient_naive(idx, indices)
            if coef == 0:
                continue


            grids = np.meshgrid(*nodes_list, indexing='ij')
            w_grids = np.meshgrid(*weights_list, indexing='ij')

            flat_pts = np.stack([g.ravel() for g in grids], axis=1)
            flat_w = np.prod(np.stack([g.ravel() for g in w_grids], axis=0), axis=0)

            for pt, w in zip(flat_pts, flat_w):
                key = tuple(np.round(pt, 12))
                if key not in point_dict:
                    point_dict[key] = 0.0
                point_dict[key] += coef * w

        self.points = np.array([np.array(k) for k in point_dict.keys()])
        self.weights = np.array(list(point_dict.values()))

    def _compute_coefficient_naive(self, idx, all_indices):
        idx = np.asarray(idx)
        dim = len(idx)
        coef = 0

        for mask in range(1 << dim):
            j = idx.copy()
            diff_sum = 0
            valid = True
            for d in range(dim):
                if mask & (1 << d):
                    j[d] += 1
                    diff_sum += 1

                if j[d] > self.level_max + 1:
                    valid = False
                    break
            if not valid:
                continue

            if np.sum(j) <= self.level_max + dim - 1:
                coef += ((-1) ** diff_sum)
        return coef

    def integrate(self, func):
        total = 0.0
        for pt, w in zip(self.points, self.weights):
            total += w * func(pt)
        return total

    def partition_function(self, energy_func, beta):
        def integrand(x):
            return np.exp(-beta * energy_func(x))
        return self.integrate(integrand)

    def free_energy(self, energy_func, beta):
        Z = self.partition_function(energy_func, beta)
        if Z <= 0 or not np.isfinite(Z):
            return np.inf
        return -np.log(Z) / beta

    def expectation(self, observable_func, energy_func, beta):
        def num(x):
            return observable_func(x) * np.exp(-beta * energy_func(x))

        Z = self.partition_function(energy_func, beta)
        if Z <= 0:
            return 0.0
        return self.integrate(num) / Z


class FreeEnergyCalculator:

    @staticmethod
    def maier_saupe_free_energy(S, T, Tc, J_coupling=2.5):
        if T <= 0:
            raise ValueError("温度必须为正。")
        kb = 0.008314
        if abs(T - Tc) < 1e-6:
            T = Tc + 1e-6


        a = J_coupling
        b = J_coupling * 0.5
        c = J_coupling * 0.1
        tau = (T - Tc) / Tc
        f = a * tau * S ** 2 + b * S ** 4 + c * S ** 6
        return f

    @staticmethod
    def landau_expansion_coefficients(T, Tc, J=2.5):


        raise NotImplementedError("landau_expansion_coefficients 方法需要补全")

    @staticmethod
    def transition_temperature_estimate(J=2.5, S_init=0.8):
        kb = 0.008314
        Tc = 0.220 * J / kb
        return Tc
