
import numpy as np





def trigcardinal(xi, xdj, nd, h):
    xi = np.asarray(xi, dtype=float)
    diff = np.pi * (xi - xdj) / h

    if nd % 2 == 1:
        denom = nd * np.sin(diff / nd)
    else:
        denom = nd * np.tan(diff / nd)
    tau = np.sin(diff) / (denom + 1e-300)

    tau[np.isclose(xi, xdj, atol=1e-12)] = 1.0
    return tau


def trig_interpolant(xd, yd, xi):
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    nd = xd.size
    if nd < 2:
        raise ValueError("trig_interpolant: 至少需要 2 个数据点")
    h = xd[1] - xd[0]
    if abs(h) < 1e-15:
        raise ValueError("trig_interpolant: 节点间隔为零")

    scalar_input = np.isscalar(xi)
    xi_arr = np.atleast_1d(np.asarray(xi, dtype=float))
    yi = np.zeros_like(xi_arr)
    for j in range(nd):
        yi += yd[j] * trigcardinal(xi_arr, xd[j], nd, h)
    return float(yi[0]) if scalar_input else yi





def cvt_circle_nonuniform(n_generators: int,
                          density_func,
                          n_samples: int = None,
                          n_iterations: int = 20,
                          domain_radius: float = 1.0):
    n_generators = max(1, int(n_generators))
    if n_samples is None:
        n_samples = 5000 * n_generators
    n_samples = max(n_generators * 10, int(n_samples))


    rng = np.random.default_rng(seed=42)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_generators)
    r = domain_radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_generators))
    generators = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    for it in range(n_iterations):

        samples = np.zeros((n_samples, 2), dtype=float)
        accepted = 0
        max_density = 0.0

        for g in generators:
            max_density = max(max_density, density_func(g))
        max_density = max(max_density, 1.0) * 1.5

        while accepted < n_samples:
            batch = min(n_samples - accepted, 2000)
            cand_theta = rng.uniform(0.0, 2.0 * np.pi, size=batch)
            cand_r = domain_radius * np.sqrt(rng.uniform(0.0, 1.0, size=batch))
            cand = np.column_stack([cand_r * np.cos(cand_theta),
                                    cand_r * np.sin(cand_theta)])
            rho_vals = np.array([density_func(pt) for pt in cand])
            mask = rng.uniform(0.0, max_density, size=batch) <= rho_vals
            n_acc = int(np.sum(mask))
            end = min(accepted + n_acc, n_samples)
            samples[accepted:end] = cand[mask][:end - accepted]
            accepted = end



        diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)


        new_generators = np.zeros_like(generators)
        for i in range(n_generators):
            mask = nearest == i
            if np.sum(mask) == 0:
                new_generators[i] = generators[i]
            else:
                pts = samples[mask]
                rhos = np.array([density_func(pt) for pt in pts])
                weights = rhos + 1e-15
                new_generators[i] = np.average(pts, axis=0, weights=weights)


        norms = np.linalg.norm(new_generators, axis=1)
        scale = np.where(norms > domain_radius, domain_radius / (norms + 1e-15), 1.0)
        new_generators *= scale[:, np.newaxis]
        generators = new_generators

    return generators


class AdaptiveChemotaxisSampler:

    def __init__(self, concentration_field_func, domain=((-1, 1), (-1, 1))):
        self.c_func = concentration_field_func
        self.domain = domain

    def density_at(self, pt):
        c = self.c_func(pt)
        return max(0.0, c)

    def sample_adaptive(self, n_points: int = 30, n_iter: int = 15):

        xmid = 0.5 * (self.domain[0][0] + self.domain[0][1])
        ymid = 0.5 * (self.domain[1][0] + self.domain[1][1])
        rx = 0.5 * (self.domain[0][1] - self.domain[0][0])
        ry = 0.5 * (self.domain[1][1] - self.domain[1][0])

        def circle_density(q):
            x = xmid + rx * q[0]
            y = ymid + ry * q[1]
            return self.density_at(np.array([x, y]))

        gens = cvt_circle_nonuniform(n_points, circle_density,
                                      n_iterations=n_iter, domain_radius=1.0)

        points = np.zeros_like(gens)
        points[:, 0] = xmid + rx * gens[:, 0]
        points[:, 1] = ymid + ry * gens[:, 1]
        return points

    def interpolate_periodic_signal(self, t_values, signal_values, t_query):
        return trig_interpolant(t_values, signal_values, t_query)
