
import numpy as np


class TrigammaFunction:

    def __init__(self):
        self.a = 0.0001
        self.b = 5.0
        self.b2 = 0.1666666667
        self.b4 = -0.03333333333
        self.b6 = 0.02380952381
        self.b8 = -0.03333333333

    def evaluate(self, x):
        if x <= 0.0:
            return 0.0, 1

        z = x
        value = 0.0


        if x <= self.a:
            return 1.0 / (x * x), 0


        while z < self.b:
            value += 1.0 / (z * z)
            z += 1.0


        y = 1.0 / (z * z)
        value += 0.5 * y + (1.0 + y * (self.b2 + y * (self.b4
                            + y * (self.b6 + y * self.b8)))) / z

        return value, 0

    def fermi_dirac_integral_derivative(self, eta, nu=0):

        if eta > 5.0 and nu > 0:
            dF = nu * (eta ** (nu - 1.0) / nu
                       + (np.pi ** 2 / 6.0) * (nu - 1.0) * eta ** (nu - 3.0))
            return dF


        t_vals = np.linspace(0.0, 50.0, 5000)
        dt = t_vals[1] - t_vals[0]
        integrand = t_vals ** nu / ((np.exp(t_vals - eta) + 1.0)
                                    * (np.exp(eta - t_vals) + 1.0))
        dF = np.sum(integrand) * dt
        return dF


class CarlsonEllipticIntegrals:

    def __init__(self, errtol=1e-6):
        self.errtol = errtol
        self.lolim = 6e-51
        self.uplim = 1e48

    def rf(self, x, y, z):
        if (x < 0.0 or y < 0.0 or z < 0.0 or
                x + y < self.lolim or x + z < self.lolim or y + z < self.lolim or
                x > self.uplim or y > self.uplim or z > self.uplim):
            return 0.0, 1

        ierr = 0
        xn, yn, zn = x, y, z
        sigma = 0.0
        power4 = 1.0

        while True:
            mu = (xn + yn + zn) / 3.0
            xndev = (mu - xn) / mu
            yndev = (mu - yn) / mu
            zndev = (mu - zn) / mu
            epslon = max(abs(xndev), abs(yndev), abs(zndev))

            if epslon < self.errtol:
                c1 = 1.0 / 24.0
                c2 = 3.0 / 44.0
                c3 = 1.0 / 14.0
                ea = xndev * yndev
                eb = zndev * zndev
                ec = ea - eb
                ed = ea - 6.0 * eb
                ef = ed + ec + ec
                s1 = ed * (-c1 + 0.25 * c3 * ed - 1.5 * c2 * zndev * ef)
                s2 = zndev * (c2 * ef + zndev * (-c3 * ec + zndev * c2 * ea))
                value = (1.0 + s1 + s2) / np.sqrt(mu)
                return value, ierr

            xnroot = np.sqrt(xn)
            ynroot = np.sqrt(yn)
            znroot = np.sqrt(zn)
            lamda = xnroot * (ynroot + znroot) + ynroot * znroot
            sigma += power4 / (znroot * (zn + lamda))
            power4 *= 0.25
            xn = (xn + lamda) * 0.25
            yn = (yn + lamda) * 0.25
            zn = (zn + lamda) * 0.25

    def rd(self, x, y, z):
        if (x < 0.0 or y < 0.0 or x + y < self.lolim or z < self.lolim or
                x > self.uplim or y > self.uplim or z > self.uplim):
            return 0.0, 1

        ierr = 0
        xn, yn, zn = x, y, z
        sigma = 0.0
        power4 = 1.0

        while True:
            mu = (xn + yn + 3.0 * zn) * 0.2
            xndev = (mu - xn) / mu
            yndev = (mu - yn) / mu
            zndev = (mu - zn) / mu
            epslon = max(abs(xndev), abs(yndev), abs(zndev))

            if epslon < self.errtol:
                c1 = 3.0 / 14.0
                c2 = 1.0 / 6.0
                c3 = 9.0 / 22.0
                c4 = 3.0 / 26.0
                ea = xndev * yndev
                eb = zndev * zndev
                ec = ea - eb
                ed = ea - 6.0 * eb
                ef = ed + ec + ec
                s1 = ed * (-c1 + 0.25 * c3 * ed - 1.5 * c4 * zndev * ef)
                s2 = zndev * (c2 * ef + zndev * (-c3 * ec + zndev * c4 * ea))
                value = 3.0 * sigma + power4 * (1.0 + s1 + s2) / (mu * np.sqrt(mu))
                return value, ierr

            xnroot = np.sqrt(xn)
            ynroot = np.sqrt(yn)
            znroot = np.sqrt(zn)
            lamda = xnroot * (ynroot + znroot) + ynroot * znroot
            sigma += power4 / (znroot * (zn + lamda))
            power4 *= 0.25
            xn = (xn + lamda) * 0.25
            yn = (yn + lamda) * 0.25
            zn = (zn + lamda) * 0.25

    def ellipsoid_surface_area(self, a, b, c):
        a, b, c = abs(a), abs(b), abs(c)

        axes = sorted([a, b, c], reverse=True)
        a, b, c = axes[0], axes[1], axes[2]

        if a < 1e-30:
            return 0.0

        phi = np.arccos(min(1.0, c / a))
        sin_phi = np.sin(phi)

        if abs(a ** 2 - c ** 2) < 1e-30:
            m = 1.0
        else:
            m = (a ** 2 * (b ** 2 - c ** 2)) / (b ** 2 * (a ** 2 - c ** 2))




        cp2 = (np.cos(phi)) ** 2
        sp2 = (np.sin(phi)) ** 2

        rf_val, _ = self.rf(cp2, 1.0 - m * sp2, 1.0)
        rd_val, _ = self.rd(cp2, 1.0 - m * sp2, 1.0)

        elliptic_f = sin_phi * rf_val
        elliptic_e = sin_phi * rf_val - (m / 3.0) * (sin_phi ** 3) * rd_val

        if abs(sin_phi) < 1e-30:
            temp2 = 1.0
        else:
            temp = elliptic_e * sp2 + elliptic_f * cp2
            temp2 = temp / sin_phi

        area = 2.0 * np.pi * (c ** 2 + a * b * temp2)
        return area


class Interpolation2D:

    def __init__(self, points, values):
        self.points = np.asarray(points)
        self.values = np.asarray(values)
        if self.points.shape[0] != self.values.shape[0]:
            raise ValueError("points and values must have the same length.")
        if self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2).")

    def inverse_distance_weighting(self, x, y, power=2, n_neighbors=8):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        result = np.zeros_like(x, dtype=float)

        for j in range(x.size):
            dx = self.points[:, 0] - x.flat[j]
            dy = self.points[:, 1] - y.flat[j]
            dist = np.sqrt(dx ** 2 + dy ** 2)


            exact = dist < 1e-12
            if np.any(exact):
                result.flat[j] = self.values[np.argmax(exact)]
                continue


            idx = np.argsort(dist)[:n_neighbors]
            w = 1.0 / (dist[idx] ** power)
            result.flat[j] = np.sum(w * self.values[idx]) / np.sum(w)

        return result

    def radial_basis_function(self, x, y, epsilon=1.0):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)


        N = self.points.shape[0]
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                r = np.linalg.norm(self.points[i] - self.points[j])
                dist_matrix[i, j] = np.exp(-(epsilon * r) ** 2)


        dist_matrix += 1e-10 * np.eye(N)
        c = np.linalg.solve(dist_matrix, self.values)

        result = np.zeros_like(x, dtype=float)
        for j in range(x.size):
            r = np.sqrt((self.points[:, 0] - x.flat[j]) ** 2
                        + (self.points[:, 1] - y.flat[j]) ** 2)
            phi = np.exp(-(epsilon * r) ** 2)
            result.flat[j] = np.sum(c * phi)

        return result


def polygamma_identity_check():
    tg = TrigammaFunction()
    val_half, _ = tg.evaluate(0.5)
    val_one, _ = tg.evaluate(1.0)
    print(f"Trigamma(0.5) = {val_half:.10f}, expected = {np.pi**2/2:.10f}")
    print(f"Trigamma(1.0) = {val_one:.10f}, expected = {np.pi**2/6:.10f}")
