
import math
import numpy as np
from stats_utils import setup_discrete_histogram


class BiomassPSD:

    def __init__(self, d_min=0.1e-3, d_max=50.0e-3):
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.hist_x = None
        self.hist_y = None

    def rosin_rammler_cdf(self, d, d_50, n):
        if d_50 <= 0.0 or n <= 0.0:
            return 0.0
        d = np.asarray(d, dtype=float)
        d = np.clip(d, 0.0, None)
        return 1.0 - np.exp(-(d / d_50) ** n)

    def rosin_rammler_pdf(self, d, d_50, n):
        if d_50 <= 0.0 or n <= 0.0:
            return 0.0
        d = np.asarray(d, dtype=float)
        d = np.clip(d, 1.0e-15, None)
        return (n / d_50) * (d / d_50) ** (n - 1.0) * np.exp(-(d / d_50) ** n)

    def lognormal_pdf(self, d, mu, sigma):
        if sigma <= 0.0:
            return 0.0
        d = np.asarray(d, dtype=float)
        d = np.clip(d, 1.0e-15, None)
        return (1.0 / (d * sigma * math.sqrt(2.0 * math.pi))) * \
               np.exp(-(np.log(d) - mu) ** 2 / (2.0 * sigma ** 2))

    def sauter_mean_diameter(self, d_50, n, distribution='rosin-rammler',
                             num_points=200):
        d = np.linspace(self.d_min, self.d_max, num_points)
        if distribution == 'rosin-rammler':
            pdf = self.rosin_rammler_pdf(d, d_50, n)
        elif distribution == 'lognormal':
            pdf = self.lognormal_pdf(d, d_50, n)
        else:
            return d_50


        d3 = d ** 3 * pdf
        d2 = d ** 2 * pdf
        num = np.trapz(d3, d)
        den = np.trapz(d2, d)
        if abs(den) > 1.0e-15:
            return num / den
        return d_50

    def specific_surface_area(self, d_32, particle_density=700.0):
        if d_32 <= 0.0 or particle_density <= 0.0:
            return 0.0
        return 6.0 / (particle_density * d_32)

    def build_histogram(self, samples):
        self.hist_x, self.hist_y = setup_discrete_histogram(
            samples, self.d_min, self.d_max
        )
        return self.hist_x, self.hist_y

    def mean_diameter_from_histogram(self):
        if self.hist_x is None or self.hist_y is None:
            return 0.0
        x = self.hist_x
        y = self.hist_y

        integrand = x * y
        return np.trapz(integrand, x)

    def biot_number(self, d_32, h_conv, k_char):
        if k_char <= 0.0:
            return 0.0
        L_c = d_32 / 6.0
        return h_conv * L_c / k_char

    def thiele_modulus(self, d_32, rate_const, D_eff):
        if D_eff <= 0.0:
            return 0.0
        return (d_32 / 2.0) * math.sqrt(rate_const / D_eff)

    def effectiveness_factor(self, phi):
        if phi <= 1.0e-6:
            return 1.0
        return 3.0 / phi * (1.0 / math.tanh(phi) - 1.0 / phi)
