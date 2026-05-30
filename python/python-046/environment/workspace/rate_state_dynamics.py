
import numpy as np
from scipy.integrate import solve_ivp
from utils import check_finite, clip_to_range


class RateStateFriction:

    def __init__(self, a, b, Dc, sigma_n, mu0, V0, k, V_pl,
                 radiation_damping=False):
        self.a = a
        self.b = b
        self.Dc = Dc
        self.sigma_n = sigma_n
        self.mu0 = mu0
        self.V0 = V0
        self.k = k
        self.V_pl = V_pl
        self.radiation_damping = radiation_damping

        self.eta = 3e9 / (2.0 * 3000.0) if radiation_damping else 0.0

    def friction_coefficient(self, V, theta):
        V_safe = clip_to_range(V, 1e-18, 1e2)
        theta_safe = clip_to_range(theta, 1e-18, 1e10)
        mu = (self.mu0 +
              self.a * np.log(V_safe / self.V0) +
              self.b * np.log(self.V0 * theta_safe / self.Dc))
        return mu

    def shear_stress(self, V, theta):
        return self.sigma_n * self.friction_coefficient(V, theta)

    def dstate_dt(self, V, theta):
        return 1.0 - V * theta / self.Dc

    def derivatives(self, t, y):
        slip, V, theta, tau = y
        V = clip_to_range(V, 1e-20, 1e2)
        theta = clip_to_range(theta, 1e-20, 1e10)

        dtheta = self.dstate_dt(V, theta)


        dtau_load = self.k * (self.V_pl - V)


        dmu_dV = self.a / V
        dmu_dtheta = self.b / theta


        damping = self.sigma_n * dmu_dV + self.eta
        if damping < 1e-20:
            damping = 1e-20

        dV = (dtau_load - self.sigma_n * dmu_dtheta * dtheta) / damping
        dV = clip_to_range(dV, -1e6, 1e6)

        dstate = np.array([V, dV, dtheta, dtau_load])
        return dstate

    def solve_ode(self, t_span, y0, t_eval=None, method='RK45'):
        y0 = np.asarray(y0, dtype=float)
        if y0[1] <= 0 or y0[2] <= 0:
            raise ValueError("rate_state_dynamics: V and theta must be positive")

        sol = solve_ivp(
            fun=self.derivatives,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        if not sol.success:
            raise RuntimeError("RateStateFriction ODE integration failed")
        return sol

    def steady_state_solution(self, V_ss):
        theta_ss = self.Dc / V_ss
        tau_ss = self.shear_stress(V_ss, theta_ss)
        return theta_ss, tau_ss


class MultiSegmentRateState:

    def __init__(self, segments_params):
        self.n_segments = len(segments_params)
        self.segments = [RateStateFriction(**p) for p in segments_params]

    def derivatives_coupled(self, t, y):
        n = self.n_segments
        dydt = np.zeros(4 * n)

        coupling = 1e8
        V_avg = np.mean(y[1::4])

        for i in range(n):
            seg = self.segments[i]
            slip_i = y[4 * i]
            V_i = y[4 * i + 1]
            theta_i = y[4 * i + 2]
            tau_i = y[4 * i + 3]

            V_i = clip_to_range(V_i, 1e-20, 1e2)
            theta_i = clip_to_range(theta_i, 1e-20, 1e10)

            dtheta = seg.dstate_dt(V_i, theta_i)
            dtau_load = seg.k * (seg.V_pl - V_i) + coupling * (V_avg - V_i)

            dmu_dV = seg.a / V_i
            dmu_dtheta = seg.b / theta_i
            damping = seg.sigma_n * dmu_dV + seg.eta
            if damping < 1e-20:
                damping = 1e-20
            dV = (dtau_load - seg.sigma_n * dmu_dtheta * dtheta) / damping
            dV = clip_to_range(dV, -1e6, 1e6)

            dydt[4 * i] = V_i
            dydt[4 * i + 1] = dV
            dydt[4 * i + 2] = dtheta
            dydt[4 * i + 3] = dtau_load

        return dydt

    def solve_coupled(self, t_span, y0_list, t_eval=None, method='RK45'):
        y0 = np.concatenate(y0_list)
        sol = solve_ivp(
            fun=self.derivatives_coupled,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-7,
            atol=1e-9
        )
        if not sol.success:
            raise RuntimeError("MultiSegmentRateState ODE integration failed")
        return sol
