
import numpy as np
from itertools import combinations_with_replacement


class UncertaintyQuantify:

    def __init__(self, dim_num=4, level_max=3):
        self.dim_num = dim_num
        self.level_max = level_max




    @staticmethod
    def hermite_gauss_rule(order):
        from numpy.polynomial.hermite import hermgauss
        x, w = hermgauss(order)
        return x, w

    def level_to_order_open(self, level):
        return 2 * level + 1




    def sparse_grid_hermite(self):
        dim = self.dim_num
        level_max = self.level_max
        level_min = max(0, level_max + 1 - dim)

        grid_points = []
        grid_weights = []

        for level in range(level_min, level_max + 1):


            for comp in self._comp_next(level, dim):
                level_1d = np.array(comp, dtype=np.int32)
                order_1d = np.array([self.level_to_order_open(l) for l in level_1d])


                x_1d_list = []
                w_1d_list = []
                for d_idx in range(dim):
                    x_d, w_d = self.hermite_gauss_rule(order_1d[d_idx])
                    x_1d_list.append(x_d)
                    w_1d_list.append(w_d)


                import itertools
                for indices in itertools.product(*[range(len(xd)) for xd in x_1d_list]):
                    point = np.array([x_1d_list[d][indices[d]] for d in range(dim)])
                    weight = 1.0
                    for d in range(dim):
                        weight *= w_1d_list[d][indices[d]]


                    coeff = (-1) ** (level_max - level)

                    from math import comb
                    coeff *= comb(dim - 1, level_max - level)
                    weight *= coeff

                    grid_points.append(point)
                    grid_weights.append(weight)

        if len(grid_points) == 0:

            return np.zeros((1, dim)), np.ones(1)

        points = np.array(grid_points)
        weights = np.array(grid_weights)


        points_unique = []
        weights_unique = []
        tol = 1e-10
        for i in range(len(points)):
            found = False
            for j in range(len(points_unique)):
                if np.linalg.norm(points[i] - points_unique[j]) < tol:
                    weights_unique[j] += weights[i]
                    found = True
                    break
            if not found:
                points_unique.append(points[i])
                weights_unique.append(weights[i])

        return np.array(points_unique), np.array(weights_unique)

    def _comp_next(self, n, k):
        if k == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for tail in self._comp_next(n - i, k - 1):
                yield (i,) + tail




    def propagate_moments(self, model_func, sigma_params):
        points, weights = self.sparse_grid_hermite()
        n_points = len(points)



        values = []
        for i in range(n_points):
            val = model_func(points[i])
            values.append(val)
        values = np.array(values)


        w_norm = weights / np.sum(weights)

        mean = np.sum(w_norm * values)
        variance = np.sum(w_norm * (values - mean) ** 2)
        std = np.sqrt(variance)

        if std > 1e-15:
            skewness = np.sum(w_norm * (values - mean) ** 3) / std ** 3
            kurtosis = np.sum(w_norm * (values - mean) ** 4) / std ** 4
        else:
            skewness = 0.0
            kurtosis = 3.0

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'points': points,
            'weights': w_norm,
            'values': values,
        }

    def phase_sensitivity_analysis(self, base_params, sigma_params,
                                    phase_model):
        dim = self.dim_num

        def wrapped_model(xi):
            params = base_params + sigma_params * xi
            return phase_model(params)

        stats = self.propagate_moments(wrapped_model, sigma_params)



        sobol_first = np.zeros(dim)
        total_var = stats['variance']

        if total_var > 1e-15:
            for i in range(dim):

                points, weights = self.sparse_grid_hermite()

                unique_xi = np.unique(np.round(points[:, i], decimals=8))
                conditional_means = []
                conditional_weights = []
                for ux in unique_xi:
                    mask = np.abs(points[:, i] - ux) < 1e-7
                    w_sum = np.sum(weights[mask])
                    if np.sum(mask) > 0 and w_sum > 1e-15:
                        wm = weights[mask] / w_sum
                        cm = np.sum(wm * stats['values'][mask])
                        conditional_means.append(cm)
                        conditional_weights.append(w_sum)
                if len(conditional_means) > 1:
                    cw = np.array(conditional_weights)
                    cw = cw / np.sum(cw)
                    cm_arr = np.array(conditional_means)
                    var_cond = np.sum(cw * (cm_arr - np.mean(cm_arr)) ** 2)
                    sobol_first[i] = var_cond / total_var

        stats['sobol_first'] = sobol_first
        return stats


def demo():
    uq = UncertaintyQuantify(dim_num=3, level_max=4)


    k0 = 2.0 * np.pi / 1.55e-6
    n_si = 3.48

    def phase_model(params):
        h0 = 0.6e-6
        w0 = 0.3e-6
        sigma_h = 0.02e-6
        sigma_w = 0.01e-6
        sigma_x = 0.005e-6
        h = h0 + sigma_h * params[0]
        w = w0 + sigma_w * params[1]

        n_eff = 1.0 + (n_si - 1.0) * (w / 0.5e-6) ** 0.7
        phi = k0 * (n_eff - 1.0) * h

        phi += k0 * sigma_x * params[2] * 0.1
        return phi

    base = np.array([0.6e-6, 0.3e-6, 0.0])
    sigma = np.array([0.02e-6, 0.01e-6, 0.005e-6])

    stats = uq.phase_sensitivity_analysis(base, sigma, phase_model)
    print("[uncertainty_quantify] 相位响应统计矩:")
    print(f"  均值 μ = {stats['mean']:.4f} rad = {np.degrees(stats['mean']):.2f}°")
    print(f"  方差 σ² = {stats['variance']:.4e}")
    print(f"  标准差 σ = {stats['std']:.4f} rad = {np.degrees(stats['std']):.2f}°")
    print(f"  偏度 S = {stats['skewness']:.4f}")
    print(f"  峰度 K = {stats['kurtosis']:.4f}")
    print(f"  Sobol 一阶指数: h={stats['sobol_first'][0]:.3f}, "
          f"w={stats['sobol_first'][1]:.3f}, x={stats['sobol_first'][2]:.3f}")
    return stats


if __name__ == "__main__":
    demo()
