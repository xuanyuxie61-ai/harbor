
import numpy as np


class CVT1DPeriodic:

    def __init__(self, num_generators, domain_length=1.0,
                 density_func=None, it_max=100, tol=1e-6, sample_num=50000):
        if num_generators < 2:
            raise ValueError("生成元数量至少为 2。")
        self.g_num = num_generators
        self.L = domain_length
        self.it_max = it_max
        self.tol = tol
        self.sample_num = sample_num

        if density_func is None:
            self.density = lambda x: np.ones_like(x)
        else:
            self.density = density_func


        self.generators = np.linspace(0.0, self.L, num_generators, endpoint=False)
        self.energy_history = []
        self.motion_history = []

    def _periodic_distance(self, x, g):
        d1 = np.abs(x - g)
        d2 = np.abs(x - g + self.L)
        d3 = np.abs(x - g - self.L)
        return np.minimum(np.minimum(d1, d2), d3)

    def _find_nearest_periodic(self, samples):

        s = samples[:, np.newaxis]
        g = self.generators[np.newaxis, :]

        d1 = np.abs(s - g)
        d2 = np.abs(s - g + self.L)
        d3 = np.abs(s - g - self.L)
        d = np.minimum(np.minimum(d1, d2), d3)

        nearest = np.argmin(d, axis=1)
        min_d = np.min(d, axis=1)
        return nearest, min_d

    def iterate(self):
        for it in range(self.it_max):

            samples = np.random.rand(self.sample_num) * self.L
            sa = samples - self.L
            sb = samples + self.L


            nearest_s, dist_s = self._find_nearest_periodic(samples)
            nearest_a, dist_a = self._find_nearest_periodic(sa)
            nearest_b, dist_b = self._find_nearest_periodic(sb)


            g_new = np.zeros(self.g_num)
            w_new = np.zeros(self.g_num)
            energy = 0.0

            for idx in range(self.sample_num):

                d_list = [dist_s[idx], dist_a[idx], dist_b[idx]]
                k_list = [nearest_s[idx], nearest_a[idx], nearest_b[idx]]
                best = int(np.argmin(d_list))
                k = k_list[best]
                d = d_list[best]


                if best == 1:
                    s_mapped = sa[idx]
                elif best == 2:
                    s_mapped = sb[idx]
                else:
                    s_mapped = samples[idx]

                rho_val = self.density(s_mapped)
                g_new[k] += s_mapped * rho_val
                w_new[k] += rho_val
                energy += d * d * rho_val


            mask = w_new > 0
            g_new[mask] /= w_new[mask]
            g_new[~mask] = self.generators[~mask]


            g_new = np.mod(g_new, self.L)
            g_new = np.sort(g_new)


            motion = 0.0
            for k in range(self.g_num):
                diff = np.abs(g_new[k] - self.generators[k])
                diff = min(diff, self.L - diff)
                motion += diff * diff
            motion = np.sqrt(motion / self.g_num)

            self.generators = g_new.copy()
            self.energy_history.append(energy / self.sample_num)
            self.motion_history.append(motion)

            if motion < self.tol:
                break

        return self.generators.copy()


class CVT2DReflect:

    def __init__(self, num_generators, bounds, density_func=None,
                 it_max=50, tol=1e-5, sample_num=20000):
        self.g_num = num_generators
        self.bounds = bounds
        self.it_max = it_max
        self.tol = tol
        self.sample_num = sample_num
        self.xmin, self.xmax = bounds[0]
        self.ymin, self.ymax = bounds[1]

        if density_func is None:
            self.density = lambda x, y: np.ones_like(x)
        else:
            self.density = density_func


        self.generators = np.column_stack((
            np.random.rand(num_generators) * (self.xmax - self.xmin) + self.xmin,
            np.random.rand(num_generators) * (self.ymax - self.ymin) + self.ymin
        ))
        self.energy_history = []

    def _reflect_sample(self, sample):
        x, y = sample
        rx, ry = x, y
        inside = True
        if x < self.xmin:
            rx = 2.0 * self.xmin - x
            inside = False
        elif x > self.xmax:
            rx = 2.0 * self.xmax - x
            inside = False
        if y < self.ymin:
            ry = 2.0 * self.ymin - y
            inside = False
        elif y > self.ymax:
            ry = 2.0 * self.ymax - y
            inside = False

        if inside:
            return None
        if self.xmin <= rx <= self.xmax and self.ymin <= ry <= self.ymax:
            return np.array([rx, ry])
        return None

    def iterate(self):
        for it in range(self.it_max):
            samples = np.column_stack((
                np.random.rand(self.sample_num) * (self.xmax - self.xmin) + self.xmin,
                np.random.rand(self.sample_num) * (self.ymax - self.ymin) + self.ymin
            ))

            g_new = np.zeros((self.g_num, 2))
            w_new = np.zeros(self.g_num)
            energy = 0.0

            for s in samples:

                dists = np.sum((self.generators - s) ** 2, axis=1)
                nearest = int(np.argmin(dists))
                dmin = dists[nearest]


                reflected = self._reflect_sample(s)
                if reflected is not None:

                    pass

                rho_val = self.density(s[0], s[1])
                g_new[nearest] += s * rho_val
                w_new[nearest] += rho_val
                energy += dmin * rho_val


            mask = w_new > 0
            g_new[mask] /= w_new[mask][:, np.newaxis]
            g_new[~mask] = self.generators[~mask]


            g_new[:, 0] = np.clip(g_new[:, 0], self.xmin, self.xmax)
            g_new[:, 1] = np.clip(g_new[:, 1], self.ymin, self.ymax)

            motion = np.sqrt(np.mean(np.sum((g_new - self.generators) ** 2, axis=1)))
            self.generators = g_new.copy()
            self.energy_history.append(energy / self.sample_num)

            if motion < self.tol:
                break

        return self.generators.copy()


def wake_density_function(X, Y, cylinder_x, cylinder_y, r_cyl,
                          wake_length_factor=5.0, wake_width_factor=1.5,
                          base_density=1.0, peak_density=10.0):
    lw = wake_length_factor * r_cyl
    sx = lw / 2.0
    sy = wake_width_factor * r_cyl / 2.0

    dx = X - cylinder_x - lw
    dy = Y - cylinder_y

    rho = base_density + (peak_density - base_density) * np.exp(
        -0.5 * (dx / sx) ** 2 - 0.5 * (dy / sy) ** 2
    )


    dist_to_cyl = np.sqrt((X - cylinder_x) ** 2 + (Y - cylinder_y) ** 2)
    wall_enhance = 1.0 + 2.0 * np.exp(-dist_to_cyl / r_cyl)
    rho *= wall_enhance

    return np.clip(rho, base_density, peak_density * 3.0)
