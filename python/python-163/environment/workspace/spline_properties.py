
import numpy as np


class CubicSplineInterpolator:

    def __init__(self, xdata, ydata):
        xdata = np.asarray(xdata, dtype=np.float64)
        ydata = np.asarray(ydata, dtype=np.float64)
        if xdata.ndim != 1 or ydata.ndim != 1:
            raise ValueError("xdata and ydata must be 1-D arrays.")
        if xdata.size != ydata.size:
            raise ValueError("xdata and ydata must have the same length.")
        if xdata.size < 3:
            raise ValueError("At least 3 points are required for not-a-knot spline.")
        if not np.all(np.diff(xdata) > 0):
            raise ValueError("xdata must be strictly increasing.")

        self.x = xdata.copy()
        self.y = ydata.copy()
        self.n = self.x.size - 1
        self._compute_coefficients()

    def _compute_coefficients(self):
        n = self.n
        x = self.x
        y = self.y
        h = np.diff(x)


        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)



        A[0, 0] = h[1]
        A[0, 1] = -(h[0] + h[1])
        A[0, 2] = h[0]
        b[0] = 0.0


        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i]
                          - (y[i] - y[i - 1]) / h[i - 1])


        A[n, n - 2] = h[n - 1]
        A[n, n - 1] = -(h[n - 2] + h[n - 1])
        A[n, n] = h[n - 2]
        b[n] = 0.0


        m = np.linalg.solve(A, b)
        self.m = m


        self.a_coef = y[:-1]
        self.b_coef = (y[1:] - y[:-1]) / h - h * (2.0 * m[:-1] + m[1:]) / 6.0
        self.c_coef = m[:-1] / 2.0
        self.d_coef = (m[1:] - m[:-1]) / (6.0 * h)
        self.h = h

    def evaluate(self, xq):
        xq = np.asarray(xq, dtype=np.float64)
        scalar_input = (xq.ndim == 0)
        xq = np.atleast_1d(xq)


        xq_clipped = np.clip(xq, self.x[0], self.x[-1])


        idx = np.searchsorted(self.x, xq_clipped, side='right') - 1
        idx = np.clip(idx, 0, self.n - 1)

        dx = xq_clipped - self.x[idx]
        yq = (self.a_coef[idx]
              + self.b_coef[idx] * dx
              + self.c_coef[idx] * dx ** 2
              + self.d_coef[idx] * dx ** 3)

        if scalar_input:
            return float(yq[0])
        return yq

    def derivative(self, xq, order=1):
        xq = np.asarray(xq, dtype=np.float64)
        scalar_input = (xq.ndim == 0)
        xq = np.atleast_1d(xq)
        xq_clipped = np.clip(xq, self.x[0], self.x[-1])
        idx = np.searchsorted(self.x, xq_clipped, side='right') - 1
        idx = np.clip(idx, 0, self.n - 1)
        dx = xq_clipped - self.x[idx]

        if order == 1:
            dyq = (self.b_coef[idx]
                   + 2.0 * self.c_coef[idx] * dx
                   + 3.0 * self.d_coef[idx] * dx ** 2)
        elif order == 2:
            dyq = (2.0 * self.c_coef[idx]
                   + 6.0 * self.d_coef[idx] * dx)
        else:
            raise ValueError("Only first and second derivatives supported.")

        if scalar_input:
            return float(dyq[0])
        return dyq


def build_temperature_spline_property(T_data, prop_data, prop_name="property"):
    if len(T_data) < 3:
        raise ValueError(f"Need at least 3 data points for {prop_name} spline.")
    if np.any(T_data <= 0):
        raise ValueError(f"Temperature data must be positive for {prop_name}.")
    return CubicSplineInterpolator(T_data, prop_data)


def default_rock_thermal_conductivity_spline():
    T = np.array([273.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15], dtype=np.float64)
    lam = np.array([2.8, 2.65, 2.5, 2.35, 2.2, 2.05, 1.9], dtype=np.float64)
    return build_temperature_spline_property(T, lam, "rock_thermal_conductivity")


def default_fluid_viscosity_spline():
    T = np.array([273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15,
                  343.15, 353.15, 363.15, 373.15], dtype=np.float64)
    mu = np.array([1.79e-3, 1.31e-3, 1.00e-3, 7.98e-4, 6.53e-4,
                   5.47e-4, 4.67e-4, 4.04e-4, 3.55e-4, 3.15e-4, 2.82e-4],
                  dtype=np.float64)
    return build_temperature_spline_property(T, mu, "fluid_viscosity")


def default_porosity_stress_spline():
    sigma_mpa = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0], dtype=np.float64)
    phi = 0.20 * np.exp(-0.02 * sigma_mpa)
    return build_temperature_spline_property(sigma_mpa, phi, "porosity_vs_stress")
