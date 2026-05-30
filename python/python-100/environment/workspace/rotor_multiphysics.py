
import numpy as np
from scipy.integrate import solve_ivp


class RotorDynamics:

    def __init__(
        self,
        J: float = 0.01,
        B_d: float = 0.001,
        tau_load: float = 5.0,
        n_slots: int = 6,
        cogging_amplitude: float = 0.5,
    ):
        self.J = float(J)
        self.B_d = float(B_d)
        self.tau_load = float(tau_load)
        self.n_slots = int(n_slots)
        self.cogging_amp = float(cogging_amplitude)

    def cogging_torque(self, theta: float) -> float:
        return self.cogging_amp * (
            np.sin(self.n_slots * theta)
            + 0.3 * np.sin(2 * self.n_slots * theta)
        )

    def rotor_ode(self, t: float, y: np.ndarray, tau_em_func) -> np.ndarray:
        theta, omega = y
        tau_em = tau_em_func(t, y)
        tau_cog = self.cogging_torque(theta)
        dtheta = omega
        domega = (tau_em - self.tau_load - self.B_d * omega - tau_cog) / self.J
        return np.array([dtheta, domega])

    def simulate(
        self,
        tau_em_func,
        y0: np.ndarray = None,
        t_span: tuple = (0.0, 1.0),
        t_eval: np.ndarray = None,
    ) -> dict:
        if y0 is None:
            y0 = np.array([0.0, 0.0])

        sol = solve_ivp(
            lambda t, y: self.rotor_ode(t, y, tau_em_func),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
            max_step=0.01,
        )

        if not sol.success:
            raise RuntimeError("转子动力学ODE求解失败")

        return {
            "t": sol.t,
            "theta": sol.y[0],
            "omega": sol.y[1],
            "sol": sol,
        }


class EccentricVibration:

    def __init__(
        self,
        m_r: float = 2.0,
        k_r: float = 1.0e6,
        c_r: float = 100.0,
        r0: float = 0.0,
        s_nl: float = 1.0e8,
        epsilon: float = 0.1e-3,
    ):
        self.m_r = float(m_r)
        self.k_r = float(k_r)
        self.c_r = float(c_r)
        self.r0 = float(r0)
        self.s_nl = float(s_nl)
        self.epsilon = float(epsilon)
        self.omega_n = np.sqrt(k_r / m_r)

    def vibration_ode(self, t: float, y: np.ndarray, theta_func, omega_func) -> np.ndarray:
        u, v, p, q = y
        theta = theta_func(t)
        omega = omega_func(t)


        r = np.sqrt(u * u + v * v)
        r_safe = max(r, 1.0e-12)


        nl_force_u = self.s_nl * (r - self.r0) ** 3 * (u / r_safe)
        nl_force_v = self.s_nl * (r - self.r0) ** 3 * (v / r_safe)


        unbalance_u = self.epsilon * omega * omega * np.cos(theta)
        unbalance_v = self.epsilon * omega * omega * np.sin(theta)

        dp = (
            -(self.k_r / self.m_r) * u
            - (self.c_r / self.m_r) * p
            + unbalance_u
            + nl_force_u
        )
        dq = (
            -(self.k_r / self.m_r) * v
            - (self.c_r / self.m_r) * q
            + unbalance_v
            + nl_force_v
        )

        return np.array([p, q, dp, dq])

    def simulate(
        self,
        theta_func,
        omega_func,
        y0: np.ndarray = None,
        t_span: tuple = (0.0, 0.5),
        t_eval: np.ndarray = None,
    ) -> dict:
        if y0 is None:
            y0 = np.array([self.epsilon, 0.0, 0.0, 0.0])

        sol = solve_ivp(
            lambda t, y: self.vibration_ode(t, y, theta_func, omega_func),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
            max_step=0.005,
        )

        if not sol.success:
            raise RuntimeError("偏心振动ODE求解失败")

        u = sol.y[0]
        v = sol.y[1]
        displacement = np.sqrt(u * u + v * v)

        return {
            "t": sol.t,
            "u": u,
            "v": v,
            "displacement": displacement,
            "sol": sol,
        }


class GyroscopicEffects:

    def __init__(
        self,
        A1: float = 0.005,
        A2: float = 0.005,
        A3: float = 0.015,
        m_unbalance: float = 0.01,
    ):
        self.A1 = float(A1)
        self.A2 = float(A2)
        self.A3 = float(A3)
        self.m_unb = float(m_unbalance)

    def gyro_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        psi, theta, phi, omega1, omega2, omega3 = y


        M1 = -self.m_unb * self.A1 * np.sin(theta) * np.cos(phi)
        M2 = self.m_unb * self.A2 * np.sin(theta) * np.sin(phi)
        M3 = 0.0


        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)


        sin_theta_safe = sin_theta if abs(sin_theta) > 1.0e-10 else 1.0e-10 * np.sign(sin_theta + 1.0e-20)

        dpsi = (omega1 * sin_phi + omega2 * cos_phi) / sin_theta_safe
        dtheta = omega1 * cos_phi - omega2 * sin_phi
        dphi = omega3 - cos_theta * dpsi

        domega1 = ((self.A2 - self.A3) * omega2 * omega3 + M1) / self.A1
        domega2 = ((self.A3 - self.A1) * omega3 * omega1 + M2) / self.A2
        domega3 = ((self.A1 - self.A2) * omega1 * omega2 + M3) / self.A3

        return np.array([dpsi, dtheta, dphi, domega1, domega2, domega3])

    def simulate(
        self,
        y0: np.ndarray = None,
        t_span: tuple = (0.0, 0.5),
        t_eval: np.ndarray = None,
    ) -> dict:
        if y0 is None:
            y0 = np.array([0.25, 0.4, 0.1, 100.0, 50.0, 314.0])

        sol = solve_ivp(
            self.gyro_ode,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            dense_output=True,
            max_step=0.005,
        )

        if not sol.success:
            raise RuntimeError("陀螺效应ODE求解失败")

        return {
            "t": sol.t,
            "psi": sol.y[0],
            "theta": sol.y[1],
            "phi": sol.y[2],
            "omega1": sol.y[3],
            "omega2": sol.y[4],
            "omega3": sol.y[5],
            "sol": sol,
        }


class NonlinearPeriodEstimator:

    ALPHA = 2.338107
    B0 = 0.1723
    D = 0.4889

    @classmethod
    def urabe_period(cls, mu: float) -> float:
        if mu <= 0.0:

            return 2.0 * np.pi

        term1 = (3.0 - 2.0 * np.log(2.0)) * mu
        term2 = 3.0 * cls.ALPHA / (mu ** (1.0 / 3.0))
        term3 = -(1.0 / 3.0) * np.log(mu) / mu
        term4 = (3.0 * np.log(2.0) - np.log(3.0) - 1.5 + cls.B0 - 2.0 * cls.D) / mu

        return term1 + term2 + term3 + term4

    @classmethod
    def estimate_motor_fault_period(
        cls, eccentricity: float, nominal_airgap: float, omega_0: float = 1.0
    ) -> float:
        if nominal_airgap <= 1.0e-12:
            raise ValueError("气隙必须为正")
        mu_eff = eccentricity / nominal_airgap
        T_dimless = cls.urabe_period(mu_eff)

        return T_dimless / omega_0

    @classmethod
    def cartwright_period(cls, mu: float) -> float:
        if mu <= 0.0:
            return 2.0 * np.pi
        return (3.0 - 2.0 * np.log(2.0)) * mu + 2.0 * np.pi / (mu ** (1.0 / 3.0))

    @classmethod
    def grimshaw_period(cls, mu: float) -> float:
        if mu <= 0.0:
            return 2.0 * np.pi
        alpha = 2.338
        return (3.0 - 2.0 * np.log(2.0)) * mu + 2.0 * alpha / (mu ** (1.0 / 3.0))
