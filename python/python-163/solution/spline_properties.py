"""
spline_properties.py
====================
Cubic spline interpolation for temperature- and pressure-dependent
material properties in the geothermal THM reservoir model.

Incorporates algorithms from:
  - 595_interp_spline_data: cubic spline interpolation with not-a-knot end conditions

Mathematical formulation:
Given a set of data points {(x_i, y_i)}_{i=0}^{n}, the cubic spline
S(x) is a piecewise cubic polynomial on each subinterval [x_i, x_{i+1}]:

  S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3

with the following conditions:
  1. Interpolation: S_i(x_i) = y_i, S_i(x_{i+1}) = y_{i+1}
  2. Continuity: S_i(x_{i+1}) = S_{i+1}(x_{i+1})
  3. First-derivative continuity: S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})
  4. Second-derivative continuity: S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})

For not-a-knot end conditions:
  S''' is continuous at x_1 and x_{n-1}, i.e.,
  d_0 = d_1  and  d_{n-2} = d_{n-1}

This leads to a tridiagonal system for the second derivatives m_i = S''(x_i):
  h_{i-1} m_{i-1} + 2(h_{i-1} + h_i) m_i + h_i m_{i+1}
  = 6[(y_{i+1} - y_i)/h_i - (y_i - y_{i-1})/h_{i-1}]

where h_i = x_{i+1} - x_i.
"""

import numpy as np


class CubicSplineInterpolator:
    """
    Cubic spline interpolator with not-a-knot end conditions.
    """

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
        """Compute spline coefficients via the not-a-knot formulation."""
        n = self.n
        x = self.x
        y = self.y
        h = np.diff(x)

        # Build tridiagonal system for second derivatives m
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)

        # Not-a-knot conditions at i=1 and i=n-1
        # At i=1: h_1^2 m_0 - (h_0^2 + h_1^2) m_1 + h_0^2 m_2 = 0
        A[0, 0] = h[1]
        A[0, 1] = -(h[0] + h[1])
        A[0, 2] = h[0]
        b[0] = 0.0

        # Interior points: standard continuity equation
        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i]
                          - (y[i] - y[i - 1]) / h[i - 1])

        # At i=n-1: h_{n-1}^2 m_{n-2} - (h_{n-2}^2 + h_{n-1}^2) m_{n-1} + h_{n-2}^2 m_n = 0
        A[n, n - 2] = h[n - 1]
        A[n, n - 1] = -(h[n - 2] + h[n - 1])
        A[n, n] = h[n - 2]
        b[n] = 0.0

        # Solve for second derivatives
        m = np.linalg.solve(A, b)
        self.m = m

        # Compute cubic coefficients a_i, b_i, c_i, d_i for each interval
        self.a_coef = y[:-1]
        self.b_coef = (y[1:] - y[:-1]) / h - h * (2.0 * m[:-1] + m[1:]) / 6.0
        self.c_coef = m[:-1] / 2.0
        self.d_coef = (m[1:] - m[:-1]) / (6.0 * h)
        self.h = h

    def evaluate(self, xq):
        """
        Evaluate spline at query points xq.

        Parameters
        ----------
        xq : float or np.ndarray
            Query points.

        Returns
        -------
        yq : float or np.ndarray
            Interpolated values.
        """
        xq = np.asarray(xq, dtype=np.float64)
        scalar_input = (xq.ndim == 0)
        xq = np.atleast_1d(xq)

        # Clip to valid domain for robustness
        xq_clipped = np.clip(xq, self.x[0], self.x[-1])

        # Find interval indices
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
        """
        Evaluate derivative of spline at query points.

        Parameters
        ----------
        xq : float or np.ndarray
            Query points.
        order : int
            Derivative order (1 or 2).

        Returns
        -------
        dyq : float or np.ndarray
            Derivative values.
        """
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
    """
    Build a cubic spline interpolator for a temperature-dependent property.

    Parameters
    ----------
    T_data : np.ndarray
        Temperature values (K).
    prop_data : np.ndarray
        Property values.
    prop_name : str
        Name of the property for error messages.

    Returns
    -------
    CubicSplineInterpolator
    """
    if len(T_data) < 3:
        raise ValueError(f"Need at least 3 data points for {prop_name} spline.")
    if np.any(T_data <= 0):
        raise ValueError(f"Temperature data must be positive for {prop_name}.")
    return CubicSplineInterpolator(T_data, prop_data)


def default_rock_thermal_conductivity_spline():
    """
    Default spline for rock thermal conductivity vs temperature (W/(m·K)).
    Data for granite.
    """
    T = np.array([273.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15], dtype=np.float64)
    lam = np.array([2.8, 2.65, 2.5, 2.35, 2.2, 2.05, 1.9], dtype=np.float64)
    return build_temperature_spline_property(T, lam, "rock_thermal_conductivity")


def default_fluid_viscosity_spline():
    """
    Default spline for water viscosity vs temperature (Pa·s).
    """
    T = np.array([273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15,
                  343.15, 353.15, 363.15, 373.15], dtype=np.float64)
    mu = np.array([1.79e-3, 1.31e-3, 1.00e-3, 7.98e-4, 6.53e-4,
                   5.47e-4, 4.67e-4, 4.04e-4, 3.55e-4, 3.15e-4, 2.82e-4],
                  dtype=np.float64)
    return build_temperature_spline_property(T, mu, "fluid_viscosity")


def default_porosity_stress_spline():
    """
    Spline for porosity change with effective stress (dimensionless).
    \phi = \phi_0 \exp(-c_\phi \sigma')
    where \sigma' is effective stress in MPa.
    """
    sigma_mpa = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0], dtype=np.float64)
    phi = 0.20 * np.exp(-0.02 * sigma_mpa)
    return build_temperature_spline_property(sigma_mpa, phi, "porosity_vs_stress")
