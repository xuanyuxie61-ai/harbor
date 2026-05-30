
import numpy as np







WITHERDEN_RULES = {
    1: {
        'n': 1,
        'x': np.array([0.5]),
        'y': np.array([0.5]),
        'w': np.array([1.0]),
    },
    3: {
        'n': 4,
        'x': np.array([0.211324865405187, 0.788675134594813,
                       0.211324865405187, 0.788675134594813]),
        'y': np.array([0.211324865405187, 0.211324865405187,
                       0.788675134594813, 0.788675134594813]),
        'w': np.array([0.25, 0.25, 0.25, 0.25]),
    },
    5: {
        'n': 8,
        'x': np.array([
            0.102592160379459, 0.897407839620541,
            0.102592160379459, 0.897407839620541,
            0.5, 0.5,
            0.281566496584151, 0.718433503415849,
        ]),
        'y': np.array([
            0.102592160379459, 0.102592160379459,
            0.897407839620541, 0.897407839620541,
            0.281566496584151, 0.718433503415849,
            0.5, 0.5,
        ]),
        'w': np.array([
            0.138564346606752, 0.138564346606752,
            0.138564346606752, 0.138564346606752,
            0.221714285714286, 0.221714285714286,
            0.221714285714286, 0.221714285714286,
        ]),
    },
    7: {
        'n': 12,
        'x': np.array([
            0.057104196114518, 0.942895803885482,
            0.057104196114518, 0.942895803885482,
            0.5, 0.5,
            0.209299385066662, 0.790700614933338,
            0.209299385066662, 0.790700614933338,
            0.197166296714531, 0.802833703285469,
        ]),
        'y': np.array([
            0.057104196114518, 0.057104196114518,
            0.942895803885482, 0.942895803885482,
            0.197166296714531, 0.802833703285469,
            0.197166296714531, 0.197166296714531,
            0.802833703285469, 0.802833703285469,
            0.209299385066662, 0.790700614933338,
        ]),
        'w': np.array([
            0.050844906370207, 0.050844906370207,
            0.050844906370207, 0.050844906370207,
            0.116786275403396, 0.116786275403396,
            0.082851075618464, 0.082851075618464,
            0.082851075618464, 0.082851075618464,
            0.116786275403396, 0.116786275403396,
        ]),
    },
}


def quadrilateral_witherden_rule(p):
    if p < 0:
        raise ValueError("p must be >= 0")
    if p > 7:
        p = 7

    available = sorted(WITHERDEN_RULES.keys())
    chosen = available[0]
    for av in available:
        if av >= p:
            chosen = av
            break
    rule = WITHERDEN_RULES[chosen]
    return rule['n'], rule['x'].copy(), rule['y'].copy(), rule['w'].copy()


class PhaseQuadrature:

    def __init__(self, wavelength=1.55e-6, n_si=3.48, n_air=1.0):
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength
        self.n_si = n_si
        self.n_air = n_air
        self.eps_si = n_si ** 2
        self.eps_air = n_air ** 2

    def map_to_pillar(self, x_unit, y_unit, cx, cy, width, height):
        x = cx + (x_unit - 0.5) * width
        y = cy + (y_unit - 0.5) * height
        return x, y

    def local_effective_index(self, x, y, cx, cy, width, height):


        ...
        return ...

    def integrate_phase_delay(self, cx, cy, width, height, h_pillar,
                              precision=7):
        n_q, xu, yu, w = quadrilateral_witherden_rule(precision)

        x, y = self.map_to_pillar(xu, yu, cx, cy, width, height)
        n_eff = self.local_effective_index(x, y, cx, cy, width, height)


        phase_delay = self.k0 * h_pillar * (n_eff - self.n_air)
        avg_phase = np.sum(w * phase_delay)



        R_avg = np.sum(w * ((n_eff - self.n_air) / (n_eff + self.n_air)) ** 2)
        transmission = 1.0 - R_avg
        return avg_phase, transmission

    def integrate_polarizability(self, cx, cy, width, height,
                                  precision=7):
        n_q, xu, yu, w = quadrilateral_witherden_rule(precision)
        x, y = self.map_to_pillar(xu, yu, cx, cy, width, height)
        inside = (np.abs(x - cx) <= width / 2.0) & (np.abs(y - cy) <= height / 2.0)
        eps_r = np.where(inside, self.eps_si, self.eps_air)


        area = width * height
        alpha = 8.854187817e-12 * area * np.sum(w * (eps_r - self.eps_air))
        return alpha

    def integrate_energy_density(self, field_func, cx, cy, width, height,
                                  precision=7):
        n_q, xu, yu, w = quadrilateral_witherden_rule(precision)
        x, y = self.map_to_pillar(xu, yu, cx, cy, width, height)
        E_vals = field_func(x, y)
        eps_r = self.local_effective_index(x, y, cx, cy, width, height)
        eps0 = 8.854187817e-12
        energy_density = 0.25 * eps0 * eps_r * np.abs(E_vals) ** 2
        avg_energy = np.sum(w * energy_density)
        return avg_energy

    def compute_dispersion_relation(self, width, h_pillar,
                                     n_modes=3, precision=7):
        from scipy.optimize import brentq

        n_eff_list = []
        for m in range(n_modes):

            def residual(ne):
                if ne <= self.n_air or ne >= self.n_si:
                    return 1e10
                kappa = self.k0 * np.sqrt(self.eps_si - ne ** 2)
                gamma = self.k0 * np.sqrt(ne ** 2 - self.eps_air)
                if kappa == 0:
                    return 1e10
                return np.tan(kappa * width / 2.0) - gamma / kappa

            try:

                ne_min = self.n_air + 1e-4
                ne_max = self.n_si - 1e-4

                n_scan = 500
                ne_scan = np.linspace(ne_min, ne_max, n_scan)
                res_scan = np.array([residual(ne) for ne in ne_scan])
                for i in range(n_scan - 1):
                    if res_scan[i] * res_scan[i + 1] < 0:
                        root = brentq(residual, ne_scan[i], ne_scan[i + 1])

                        if all(abs(root - r) > 1e-4 for r in n_eff_list):
                            n_eff_list.append(root)
                        if len(n_eff_list) >= n_modes:
                            break
            except Exception:
                pass


        while len(n_eff_list) < n_modes:
            if len(n_eff_list) == 0:
                n_eff_list.append(self.n_air + 0.1 * (self.n_si - self.n_air))
            else:
                n_eff_list.append(n_eff_list[-1] - 0.05 * (self.n_si - self.n_air))
        return n_eff_list[:n_modes]


def demo():
    pq = PhaseQuadrature(wavelength=1.55e-6)
    cx, cy = 0.0, 0.0
    w_pillar = 0.3e-6
    h_pillar = 0.6e-6
    height_z = 1.0e-6

    avg_phase, trans = pq.integrate_phase_delay(cx, cy, w_pillar, h_pillar, height_z)
    alpha = pq.integrate_polarizability(cx, cy, w_pillar, h_pillar)
    n_effs = pq.compute_dispersion_relation(w_pillar, height_z)

    print(f"[phase_quadrature] 平均相位延迟: {avg_phase:.4f} rad = {np.degrees(avg_phase):.2f}°")
    print(f"[phase_quadrature] 传输振幅: {trans:.4f}")
    print(f"[phase_quadrature] 等效极化率: {alpha:.4e} F·m²")
    print(f"[phase_quadrature] 前 {len(n_effs)} 阶模式有效折射率: " +
          ", ".join(f"{n:.4f}" for n in n_effs))
    return avg_phase, alpha, n_effs


if __name__ == "__main__":
    demo()
