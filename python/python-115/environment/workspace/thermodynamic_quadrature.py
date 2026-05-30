
import numpy as np
from math import comb


class IMTQLX:

    @staticmethod
    def diagonalize(d, e, z, max_iter=30):
        d = np.asarray(d, dtype=float).copy()
        e = np.asarray(e, dtype=float).copy()
        z = np.asarray(z, dtype=float).copy()
        n = len(d)

        if n == 1:
            return d, z

        e[n - 1] = 0.0
        prec = np.finfo(float).eps

        for l in range(n):
            j = 0
            while True:
                for m in range(l, n):
                    if m == n - 1:
                        break
                    if abs(e[m]) <= prec * (abs(d[m]) + abs(d[m + 1])):
                        break

                p = d[l]
                if m == l:
                    break

                if j == max_iter:
                    raise RuntimeError("IMTQLX: 迭代次数超过上限")
                j += 1

                g = (d[l + 1] - p) / (2.0 * e[l])
                r = np.sqrt(g * g + 1.0)
                g = d[m] - p + e[l] / (g + np.sign(g) * abs(r))
                s = 1.0
                c = 1.0
                p = 0.0
                mml = m - l

                for ii in range(1, mml + 1):
                    i = m - ii
                    f = s * e[i]
                    b = c * e[i]

                    if abs(f) >= abs(g):
                        c = g / f
                        r = np.sqrt(c * c + 1.0)
                        e[i + 1] = f * r
                        s = 1.0 / r
                        c *= s
                    else:
                        s = f / g
                        r = np.sqrt(s * s + 1.0)
                        e[i + 1] = g * r
                        c = 1.0 / r
                        s *= c

                    g = d[i + 1] - p
                    r = (d[i] - g) * s + 2.0 * c * b
                    p = s * r
                    d[i + 1] = g + p
                    g = c * r - b
                    f = z[i + 1]
                    z[i + 1] = s * z[i] + c * f
                    z[i] = c * z[i] - s * f

                d[l] -= p
                e[l] = g
                e[m] = 0.0


        for ii in range(1, n):
            i = ii - 1
            k = i
            p = d[i]
            for j in range(ii, n):
                if d[j] < p:
                    k = j
                    p = d[j]
            if k != i:
                d[k] = d[i]
                d[i] = p
                p = z[i]
                z[i] = z[k]
                z[k] = p

        return d, z


class GaussLaguerreQuadrature:

    def __init__(self, order, alpha_param, a=0.0, b=1.0):
        if alpha_param <= -1.0:
            raise ValueError("alpha 必须 > -1")
        if b <= 0.0:
            raise ValueError("b 必须 > 0")

        self.order = order
        self.alpha = alpha_param
        self.a = a
        self.b = b
        self.x = None
        self.w = None
        self._compute_rule()

    def _compute_rule(self):
        m = self.order
        aj = np.zeros(m, dtype=float)
        bj = np.zeros(m, dtype=float)


        for i in range(m):
            aj[i] = 2.0 * i + 1.0 + self.alpha
            bj[i] = (i + 1) * (i + 1 + self.alpha)


        from math import gamma
        zemu = gamma(self.alpha + 1.0)


        z = np.zeros(m, dtype=float)
        z[0] = np.sqrt(zemu)
        d, w = IMTQLX.diagonalize(aj, np.sqrt(bj), z)


        w = w ** 2






        self.x = self.a + d / self.b
        self.w = w / (self.b ** (self.alpha + 1.0))

    def integrate(self, func):
        return np.sum(self.w * func(self.x))


class PolygonMoments:

    @staticmethod
    def r8_mop(i):
        return 1.0 if i % 2 == 0 else -1.0

    @classmethod
    def moment_unnormalized(cls, n, x, y, p, q):
        nu_pq = 0.0
        xj = x[n - 1]
        yj = y[n - 1]

        for i in range(n):
            xi = x[i]
            yi = y[i]
            s_pq = 0.0
            for k in range(p + 1):
                for l in range(q + 1):
                    s_pq += (comb(k + l, l) * comb(p + q - k - l, q - l) *
                             xi ** k * xj ** (p - k) * yi ** l * yj ** (q - l))
            nu_pq += (xj * yi - xi * yj) * s_pq
            xj = xi
            yj = yi

        denom = (p + q + 2) * (p + q + 1) * comb(p + q, p)
        return nu_pq / denom

    @classmethod
    def moment_normalized(cls, n, x, y, p, q):
        nu_pq = cls.moment_unnormalized(n, x, y, p, q)
        nu_00 = cls.moment_unnormalized(n, x, y, 0, 0)
        if abs(nu_00) < 1e-15:
            raise ValueError("多边形面积为零")
        return nu_pq / nu_00

    @classmethod
    def moment_central(cls, n, x, y, p, q):
        alpha_10 = cls.moment_normalized(n, x, y, 1, 0)
        alpha_01 = cls.moment_normalized(n, x, y, 0, 1)

        mu_pq = 0.0
        for i in range(p + 1):
            for j in range(q + 1):
                alpha_ij = cls.moment_normalized(n, x, y, i, j)
                mu_pq += (cls.r8_mop(p + q - i - j) * comb(p, i) * comb(q, j) *
                          alpha_10 ** (p - i) * alpha_01 ** (q - j) * alpha_ij)
        return mu_pq


class HexagonMoments:

    def __init__(self):
        a = np.sqrt(3.0) / 2.0
        self.n = 6
        self.x = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5], dtype=float)
        self.y = np.array([0.0, a, a, 0.0, -a, -a], dtype=float)

    def integral_monomial(self, p, q):
        if p % 2 == 1 or q % 2 == 1:
            return 0.0
        return PolygonMoments.moment_unnormalized(self.n, self.x, self.y, p, q)


class ThermodynamicIntegration:

    def __init__(self, n_lambda=20, temperature=300.0):
        self.n_lambda = n_lambda
        self.T = temperature
        self.kB = 0.0019872041
        self.beta = 1.0 / (self.kB * temperature)

    def free_energy_barrier(self, energy_profile, xi_values):



        raise NotImplementedError("Hole_1: 请实现 free_energy_barrier 方法")

    def entropic_contribution(self, energy_profile, xi_values):
        free_energy, dG, dG_rev = self.free_energy_barrier(energy_profile, xi_values)
        dE = np.max(energy_profile) - np.min(energy_profile)
        return -self.T * (dG - dE) / self.T
