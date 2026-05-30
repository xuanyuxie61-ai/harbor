
import numpy as np


class EthierNavierStokes:

    def __init__(self, a=np.pi / 4.0, d=np.pi / 2.0):
        self.a = a
        self.d = d

    def evaluate(self, x, y, z, t):
        a = self.a
        d = self.d

        ex = np.exp(a * x)
        ey = np.exp(a * y)
        ez = np.exp(a * z)
        e2t = np.exp(-d * d * t)

        exy = np.exp(a * (x + y))
        eyz = np.exp(a * (y + z))
        ezx = np.exp(a * (z + x))

        sxy = np.sin(a * x + d * y)
        syz = np.sin(a * y + d * z)
        szx = np.sin(a * z + d * x)

        cxy = np.cos(a * x + d * y)
        cyz = np.cos(a * y + d * z)
        czx = np.cos(a * z + d * x)

        u = -a * (ex * syz + ez * cxy) * e2t
        v = -a * (ey * szx + ex * cyz) * e2t
        w = -a * (ez * sxy + ey * czx) * e2t
        p = 0.5 * a * a * e2t * e2t * (
            ex * ex + 2.0 * sxy * czx * eyz
            + ey * ey + 2.0 * syz * cxy * ezx
            + ez * ez + 2.0 * szx * cyz * exy
        )
        return u, v, w, p

    def vorticity(self, x, y, z, t):
        eps = 1e-5
        u_y = (self.evaluate(x, y + eps, z, t)[0] - self.evaluate(x, y - eps, z, t)[0]) / (2 * eps)
        u_z = (self.evaluate(x, y, z + eps, t)[0] - self.evaluate(x, y, z - eps, t)[0]) / (2 * eps)
        v_x = (self.evaluate(x + eps, y, z, t)[1] - self.evaluate(x - eps, y, z, t)[1]) / (2 * eps)
        v_z = (self.evaluate(x, y, z + eps, t)[1] - self.evaluate(x, y, z - eps, t)[1]) / (2 * eps)
        w_x = (self.evaluate(x + eps, y, z, t)[2] - self.evaluate(x - eps, y, z, t)[2]) / (2 * eps)
        w_y = (self.evaluate(x, y + eps, z, t)[2] - self.evaluate(x, y - eps, z, t)[2]) / (2 * eps)

        omega_x = w_y - v_z
        omega_y = u_z - w_x
        omega_z = v_x - u_y
        return omega_x, omega_y, omega_z


class KeastTetrahedronRule:


    _RULES = {
        1: {
            'points': np.array([[0.25, 0.25, 0.25]]),
            'weights': np.array([1.0 / 6.0])
        },
        4: {
            'points': np.array([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660],
            ]),
            'weights': np.array([0.25, 0.25, 0.25, 0.25]) / 6.0
        }
    }

    def __init__(self, rule_id=4):
        if rule_id not in self._RULES:
            raise ValueError(f"Rule {rule_id} not available. Use 1 or 4.")
        self.rule_id = rule_id
        data = self._RULES[rule_id]
        self.points_ref = data['points']
        self.weights = data['weights']
        self.Nq = len(self.weights)

    def integrate(self, func, vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (4, 3):
            raise ValueError("vertices must be (4,3).")

        v0 = vertices[0]
        J = np.column_stack([
            vertices[1] - v0,
            vertices[2] - v0,
            vertices[3] - v0
        ])
        detJ = np.linalg.det(J)
        volume = abs(detJ) / 6.0
        if volume < 1e-14:
            raise ValueError("Degenerate tetrahedron (zero volume).")

        integral = 0.0
        for q in range(self.Nq):
            xi = self.points_ref[q]

            x_phys = v0 + J @ xi
            val = func(x_phys[0], x_phys[1], x_phys[2])
            integral += self.weights[q] * val

        integral *= abs(detJ)
        return integral


class NeuralVolumeIntegral:

    def __init__(self, keast_rule=None):
        if keast_rule is None:
            keast_rule = KeastTetrahedronRule(rule_id=4)
        self.keast = keast_rule

    def ionic_charge_density(self, x, y, z, V_membrane=-65.0, Na_out=145.0, Na_in=15.0,
                              K_out=5.0, K_in=140.0, T=310.0):


        F_faraday = 96485.0
        thickness = 5e-9

        rho = F_faraday * 1e3 * (V_membrane + 65.0) / thickness

        modulation = 1.0 + 0.1 * np.sin(x) * np.cos(y) * np.sin(z)
        return rho * modulation

    def integrate_region(self, tetrahedra, V_membrane=-65.0):
        total = 0.0
        for verts in tetrahedra:
            val = self.keast.integrate(
                lambda x, y, z: self.ionic_charge_density(x, y, z, V_membrane),
                verts
            )
            total += val
        return total


def demo_navier_stokes():
    ns = EthierNavierStokes()
    x, y, z, t = 0.5, 0.5, 0.5, 0.05
    u, v, w, p = ns.evaluate(x, y, z, t)
    return u, v, w, p


def demo_tetrahedron_integral():
    keast = KeastTetrahedronRule(rule_id=4)

    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    result = keast.integrate(lambda x, y, z: x + y + z, verts)

    return result


def demo_volume_integral():
    vol = NeuralVolumeIntegral()

    tet1 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    tet2 = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ])
    total = vol.integrate_region([tet1, tet2], V_membrane=-50.0)
    return total
