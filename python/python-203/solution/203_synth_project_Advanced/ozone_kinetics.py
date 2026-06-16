"""
ozone_kinetics.py -- Uncertain Atmospheric Ozone Chemistry ODE System
======================================================================
Implements a 4-species stiff chemical kinetics ODE for atmospheric
ozone-photochemistry, with uncertain rate coefficients for UQ propagation.

Species: y = [O, NO, NO2, O3]
Reactions:
    NO2 + h*nu -> O + NO     (photolysis, rate k1(t))
    O + O2 + M -> O3 + M     (rate k2)
    NO + O3 -> NO2 + O2      (rate k3)

ODE system:
    dy1/dt = k1*y3 - k2*y1
    dy2/dt = k1*y3 - k3*y2*y4 + sigma2
    dy3/dt = -k1*y3 + k3*y2*y4
    dy4/dt = k2*y1 - k3*y2*y4

Conservation law: y1 + y3 + y4 = const (total odd oxygen)

Seed references:
  - 842_ozone2_ode: ozone2_deriv, ozone2_k1, ozone2_parameters, ozone2_conserved
  - 100_blood_pressure_ode: periodic forcing, parameter management
"""
import numpy as np
from typing import Tuple


class OzoneChemistry:
    """
    Atmospheric ozone photochemistry system with uncertain parameters.

    The photolysis rate k1(t) follows a diurnal cycle:
        k1(t) = k1_max * max(0, sin(pi*(hour-4)/16))^0.2  for 4 < hour < 20
        k1(t) = 0                                          otherwise

    Uncertain parameters (for UQ):
        k2, k3: reaction rate coefficients
        sigma2: emission source term
        initial concentrations y0
    """

    def __init__(self, k2: float = 1.7e-2, k3: float = 3.5e-2,
                 sigma2: float = 5.0, k1_max: float = 1e-5,
                 y0: np.ndarray = None):
        """
        Parameters
        ----------
        k2 : float
            Rate of O + O2 + M -> O3 + M.
        k3 : float
            Rate of NO + O3 -> NO2 + O2.
        sigma2 : float
            NO emission rate.
        k1_max : float
            Maximum photolysis rate.
        y0 : ndarray or None
            Initial concentrations [O, NO, NO2, O3].
            Default: [1e-16, 1e-16, 1e-4, 1e-6].
        """
        self.k2 = k2
        self.k3 = k3
        self.sigma2 = sigma2
        self.k1_max = k1_max
        self.y0 = y0 if y0 is not None else np.array([1e-16, 1e-16, 1e-4, 1e-6])
        self.species_names = ['O', 'NO', 'NO2', 'O3']

    def photolysis_rate(self, t: float) -> float:
        """
        Time-dependent photolysis rate k1(t).

        Models the diurnal cycle of solar radiation driving NO2 photolysis.
        Daytime (hours 4-20): k1 = k1_max * exp(7 * sin(pi*(t_bar-4)/16)^0.2)
        Nighttime: k1 ~ 0 (exponentially small)

        Parameters
        ----------
        t : float
            Time in seconds (converted to hours via t / 3600).

        Returns
        -------
        k1 : float
            Photolysis rate (1/s).
        """
        hour = (t / 3600.0) % 24.0  # hour of day
        if 4.0 < hour < 20.0:
            t_bar = hour
            sin_val = np.sin(np.pi * (t_bar - 4.0) / 16.0)
            if sin_val > 0:
                k1 = self.k1_max * np.exp(7.0 * sin_val ** 0.2)
            else:
                k1 = 0.0
        else:
            # Nighttime: exponentially small rate
            if hour <= 4.0:
                k1 = self.k1_max * 1e-10 * np.exp(-(4.0 - hour) * 2)
            else:
                k1 = self.k1_max * 1e-10 * np.exp(-(hour - 20.0) * 2)
        return k1

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute the ODE right-hand side dy/dt = f(t, y).

        The chemical kinetics ODE system:
            dy1/dt = k1*y3 - k2*y1                        (atomic oxygen)
            dy2/dt = k1*y3 - k3*y2*y4 + sigma2            (nitric oxide)
            dy3/dt = -k1*y3 + k3*y2*y4                    (nitrogen dioxide)
            dy4/dt = k2*y1 - k3*y2*y4                     (ozone)

        Physical constraints: all y_i >= 0 (enforced by exponential integrator).
        """
        k1 = self.photolysis_rate(t)
        y1, y2, y3, y4 = y[0], y[1], y[2], y[3]

        # Ensure non-negativity (clamp small negative values)
        y1 = max(y1, 0.0)
        y2 = max(y2, 0.0)
        y3 = max(y3, 0.0)
        y4 = max(y4, 0.0)

        dydt = np.zeros(4)
        dydt[0] = k1 * y3 - self.k2 * y1
        dydt[1] = k1 * y3 - self.k3 * y2 * y4 + self.sigma2
        dydt[2] = -k1 * y3 + self.k3 * y2 * y4
        dydt[3] = self.k2 * y1 - self.k3 * y2 * y4

        return dydt

    def conserved_quantity(self, y: np.ndarray) -> float:
        """
        Compute the conserved quantity (total odd oxygen):
            h = y1 + y3 + y4 = [O] + [NO2] + [O3]
        This should remain constant for sigma2 = 0.
        """
        return y[0] + y[2] + y[3]

    def jacobian(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Analytical Jacobian df/dy for implicit time integration.

        J = [[-k2,   0,    k1,    0   ],
             [ 0,   -k3*y4,  k1,  -k3*y2],
             [ 0,    k3*y4, -k1,   k3*y2],
             [ k2,   0,    0,   -k3*y2]]
        """
        k1 = self.photolysis_rate(t)
        J = np.zeros((4, 4))
        J[0, 0] = -self.k2
        J[0, 2] = k1
        J[1, 1] = -self.k3 * y[3]
        J[1, 2] = k1
        J[1, 3] = -self.k3 * y[1]
        J[2, 1] = self.k3 * y[3]
        J[2, 2] = -k1
        J[2, 3] = self.k3 * y[1]
        J[3, 0] = self.k2
        J[3, 3] = -self.k3 * y[1]
        return J

    def run_reference(self, t_end: float = 86400.0, n_steps: int = 1000
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ozone chemistry ODE with a simple implicit Euler scheme.

        Returns (t_array, y_array) where y_array has shape (n_steps+1, 4).
        """
        dt = t_end / n_steps
        t_array = np.linspace(0, t_end, n_steps + 1)
        y_array = np.zeros((n_steps + 1, 4))
        y_array[0] = self.y0.copy()

        y = self.y0.copy()
        for step in range(n_steps):
            t = t_array[step]
            # Implicit Euler: y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
            # Solve via Newton iteration
            t_new = t_array[step + 1]
            y_new = y.copy()
            for _ in range(20):
                F = y_new - y - dt * self.deriv(t_new, y_new)
                J = np.eye(4) - dt * self.jacobian(t_new, y_new)
                try:
                    delta = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    delta = -0.1 * F
                y_new += delta
                if np.linalg.norm(delta) < 1e-12 * (1 + np.linalg.norm(y_new)):
                    break
            y_new = np.maximum(y_new, 0.0)  # Enforce non-negativity
            y = y_new
            y_array[step + 1] = y.copy()

        return t_array, y_array

    def run_uncertain(self, param_perturbations: dict,
                      t_end: float = 86400.0, n_steps: int = 500
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the ozone chemistry with perturbed parameters for UQ sensitivity.

        Parameters
        ----------
        param_perturbations : dict
            Keys: 'k2', 'k3', 'sigma2', 'y0'. Values: multiplicative factors.
        """
        k2_p = self.k2 * param_perturbations.get('k2', 1.0)
        k3_p = self.k3 * param_perturbations.get('k3', 1.0)
        sigma2_p = self.sigma2 * param_perturbations.get('sigma2', 1.0)
        y0_p = self.y0 * param_perturbations.get('y0_scale', 1.0)

        chem = OzoneChemistry(k2=k2_p, k3=k3_p, sigma2=sigma2_p,
                              k1_max=self.k1_max, y0=y0_p)
        return chem.run_reference(t_end, n_steps)


def ozone_coupling_term(wave_velocity: float, base_rate: float = 1e-3) -> float:
    """
    Compute the coupling term between seismic wave velocity and ozone
    chemistry. Models the effect of ground vibration on atmospheric mixing:

        coupling = base_rate * exp(-c_ref / c(x))
    where c_ref is a reference velocity.

    This represents the (fictional but physically motivated) coupling between
    seismic wave propagation and atmospheric chemistry through vertical mixing.
    """
    c_ref = 3000.0  # reference velocity (m/s)
    return base_rate * np.exp(-c_ref / max(wave_velocity, 1.0))
