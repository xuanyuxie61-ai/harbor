
import numpy as np


class MetasurfaceCVT:

    def __init__(self, region=(-5.0e-6, 5.0e-6, -5.0e-6, 5.0e-6)):
        self.xmin, self.xmax, self.ymin, self.ymax = region
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

    def density_function(self, x, y, target_phase_func=None):
        if target_phase_func is None:

            f = 20.0e-6
            k0 = 2.0 * np.pi / 1.55e-6
            r2 = x ** 2 + y ** 2

            grad_mag = k0 * np.sqrt(r2) / np.sqrt(r2 + f ** 2)
        else:

            dx = self.Lx * 1e-4
            dy = self.Ly * 1e-4
            dxph = (target_phase_func(x + dx, y) - target_phase_func(x - dx, y)) / (2 * dx)
            dyph = (target_phase_func(x, y + dy) - target_phase_func(x, y - dy)) / (2 * dy)
            grad_mag = np.sqrt(dxph ** 2 + dyph ** 2)

        alpha = 0.5e-6
        rho = 1.0 + alpha * grad_mag

        edge_factor = 1.0 + 0.3 * (
            np.exp(-((x - self.xmin) / (0.3e-6)) ** 2) +
            np.exp(-((x - self.xmax) / (0.3e-6)) ** 2) +
            np.exp(-((y - self.ymin) / (0.3e-6)) ** 2) +
            np.exp(-((y - self.ymax) / (0.3e-6)) ** 2)
        )
        return rho * edge_factor

    def sample_density(self, n, target_phase_func=None):
        samples = np.zeros((n, 2), dtype=np.float64)
        count = 0
        max_rho = 5.0
        while count < n:

            x_cand = np.random.uniform(self.xmin, self.xmax, size=n * 2)
            y_cand = np.random.uniform(self.ymin, self.ymax, size=n * 2)
            rho_cand = self.density_function(x_cand, y_cand, target_phase_func)
            u = np.random.uniform(0, max_rho, size=n * 2)
            mask = u <= rho_cand
            valid_x = x_cand[mask]
            valid_y = y_cand[mask]
            n_valid = len(valid_x)
            take = min(n_valid, n - count)
            samples[count:count + take, 0] = valid_x[:take]
            samples[count:count + take, 1] = valid_y[:take]
            count += take
        return samples

    def voronoi_centroid(self, generator, samples):
        n_gen = generator.shape[0]



        dx = samples[:, 0][:, None] - generator[:, 0][None, :]
        dy = samples[:, 1][:, None] - generator[:, 1][None, :]
        dist2 = dx ** 2 + dy ** 2
        nearest = np.argmin(dist2, axis=1)

        new_generator = np.zeros_like(generator)
        counts = np.zeros(n_gen, dtype=np.int32)
        for j in range(n_gen):
            mask = nearest == j
            counts[j] = np.sum(mask)
            if counts[j] > 0:
                new_generator[j] = np.mean(samples[mask], axis=0)
            else:

                new_generator[j] = generator[j]
        return new_generator, counts

    def compute_cvt(self, n_generators, n_samples_per_iter=5000,
                    max_iter=50, tol=1.0e-8,
                    target_phase_func=None):

        generators = self.sample_density(n_generators, target_phase_func)
        energy_history = []

        for it in range(max_iter):

            samples = self.sample_density(n_samples_per_iter, target_phase_func)

            new_generators, counts = self.voronoi_centroid(generators, samples)


            dx = samples[:, 0][:, None] - generators[:, 0][None, :]
            dy = samples[:, 1][:, None] - generators[:, 1][None, :]
            dist2 = dx ** 2 + dy ** 2
            nearest = np.argmin(dist2, axis=1)
            min_dist2 = np.min(dist2, axis=1)
            rho_s = self.density_function(samples[:, 0], samples[:, 1], target_phase_func)
            energy = np.mean(rho_s * min_dist2)
            energy_history.append(energy)

            shift = np.max(np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)))
            generators = new_generators

            if shift < tol * max(self.Lx, self.Ly):
                print(f"[metasurface_grid] CVT 收敛于迭代 {it}, shift={shift:.3e}")
                break

        return generators, energy_history

    def assign_pillar_parameters(self, generators, phase_func,
                                  height_range=(0.3e-6, 1.2e-6),
                                  width_range=(0.15e-6, 0.5e-6)):
        n = generators.shape[0]
        x = generators[:, 0]
        y = generators[:, 1]
        target_phases = phase_func(x, y)


        target_phases = np.mod(target_phases, 2.0 * np.pi)



        h_min, h_max = height_range
        w_min, w_max = width_range



        beta = 0.7



        k0 = 2.0 * np.pi / 1.55e-6
        n_si = 3.48
        n_air = 1.0

        heights = np.zeros(n, dtype=np.float64)
        widths = np.zeros(n, dtype=np.float64)

        for i in range(n):
            phi = target_phases[i]

            w_try = w_max
            n_eff = n_air + (n_si - n_air) * (w_try / w_max) ** beta
            h_needed = phi / (k0 * (n_eff - n_air))
            if h_min <= h_needed <= h_max:
                heights[i] = h_needed
                widths[i] = w_try
            elif h_needed < h_min:

                heights[i] = h_min
                n_eff_needed = phi / (k0 * h_min) + n_air
                if n_eff_needed > n_si:
                    n_eff_needed = n_si
                if n_eff_needed < n_air:
                    n_eff_needed = n_air
                w_needed = w_max * ((n_eff_needed - n_air) / (n_si - n_air)) ** (1.0 / beta)
                widths[i] = np.clip(w_needed, w_min, w_max)
            else:

                heights[i] = h_max
                n_eff_needed = phi / (k0 * h_max) + n_air
                if n_eff_needed > n_si:
                    n_eff_needed = n_si
                w_needed = w_max * ((n_eff_needed - n_air) / (n_si - n_air)) ** (1.0 / beta)
                widths[i] = np.clip(w_needed, w_min, w_max)

        return heights, widths

    def compute_voronoi_areas(self, generators, n_samples=200000):
        x = np.random.uniform(self.xmin, self.xmax, size=n_samples)
        y = np.random.uniform(self.ymin, self.ymax, size=n_samples)
        dx = x[:, None] - generators[:, 0][None, :]
        dy = y[:, None] - generators[:, 1][None, :]
        dist2 = dx ** 2 + dy ** 2
        nearest = np.argmin(dist2, axis=1)
        n_gen = generators.shape[0]
        areas = np.zeros(n_gen, dtype=np.float64)
        total_area = self.Lx * self.Ly
        for j in range(n_gen):
            areas[j] = total_area * np.sum(nearest == j) / n_samples
        return areas


def demo():
    grid = MetasurfaceCVT(region=(-5.0e-6, 5.0e-6, -5.0e-6, 5.0e-6))


    k0 = 2.0 * np.pi / 1.55e-6
    f = 20.0e-6

    def phase_func(x, y):
        return -k0 * (np.sqrt(x ** 2 + y ** 2 + f ** 2) - f)

    generators, energy = grid.compute_cvt(
        n_generators=200,
        n_samples_per_iter=8000,
        max_iter=30,
        target_phase_func=phase_func
    )
    print(f"[metasurface_grid] CVT 能量最终值: {energy[-1]:.6e}")

    heights, widths = grid.assign_pillar_parameters(generators, phase_func)
    areas = grid.compute_voronoi_areas(generators)
    print(f"[metasurface_grid] 平均纳米柱高度: {np.mean(heights):.3e} m, "
          f"平均宽度: {np.mean(widths):.3e} m")
    print(f"[metasurface_grid] 平均单元面积: {np.mean(areas):.3e} m², "
          f"填充因子≈{np.mean(widths**2 / areas)*100:.1f}%")
    return generators, heights, widths, areas


if __name__ == "__main__":
    demo()
