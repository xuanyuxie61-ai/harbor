
import numpy as np


class StructureDynamics:

    def __init__(self, mass, damping, stiffness, mass_ratio,
                 u_inf, diameter, fn, nonlinear_params=None,
                 time_integrator='cn_rk2'):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.mass_ratio = mass_ratio
        self.u_inf = u_inf
        self.diameter = diameter
        self.fn = fn
        self.omega_n = 2.0 * np.pi * fn
        self.time_integrator = time_integrator


        self.zeta = damping / (2.0 * np.sqrt(mass * stiffness))
        self.reduced_mass = mass_ratio


        self.state = np.zeros(4)
        self.prev_state = np.zeros(4)
        self.prev_force = np.zeros(2)


        self.has_nonlinear = False
        self.k_nl = 0.0
        self.bw_params = None
        self.bw_state = None

        if nonlinear_params is not None:
            self.has_nonlinear = True
            self.k_nl = nonlinear_params.get('k_nl', 0.0)
            if 'bw_params' in nonlinear_params:
                self.bw_params = nonlinear_params['bw_params']
                self.bw_state = np.zeros(2)


        self.dt = None

    def set_time_step(self, dt):
        if dt <= 0:
            raise ValueError("dt 必须为正。")
        self.dt = dt

    def _linear_force(self, disp, vel):
        return -self.stiffness * disp - self.damping * vel

    def _nonlinear_force(self, disp, vel):
        f_nl = -self.k_nl * (disp ** 3)
        if self.bw_params is not None and self.bw_state is not None:
            alpha_bw = self.bw_params.get('alpha', 0.5)
            k_bw = self.bw_params.get('k', self.stiffness)
            f_bw = (1.0 - alpha_bw) * k_bw * self.bw_state
            f_nl += -f_bw
        return f_nl

    def _bouc_wen_derivative(self, vel, z_bw):
        if self.bw_params is None:
            return np.zeros(2)
        gamma_bw = self.bw_params.get('gamma', 0.5)
        beta_bw = self.bw_params.get('beta', 0.5)
        n_bw = self.bw_params.get('n', 1.0)

        dz = vel.copy()
        abs_vel = np.abs(vel)
        abs_z = np.abs(z_bw)
        abs_z_pow = np.where(abs_z < 1e-15, 1e-15, abs_z) ** n_bw

        dz -= gamma_bw * abs_vel * z_bw * abs_z_pow / np.where(abs_z < 1e-15, 1e-15, abs_z)
        dz -= beta_bw * vel * abs_z_pow
        return dz

    def _rhs(self, state, force_ext):
        disp = state[0:2]
        vel = state[2:4]

        f_lin = self._linear_force(disp, vel)
        f_nl = self._nonlinear_force(disp, vel)
        acc = (f_lin + f_nl + force_ext) / self.mass

        rhs = np.zeros_like(state)
        rhs[0:2] = vel
        rhs[2:4] = acc
        return rhs

    def step_cn_rk2(self, force_ext):
        if self.dt is None:
            raise RuntimeError("必须先调用 set_time_step(dt)。")

        dt = self.dt
        state_n = self.state.copy()
        disp_n = state_n[0:2]
        vel_n = state_n[2:4]







        m = self.mass
        c = self.damping
        k = self.stiffness


        f_lin_n = self._linear_force(disp_n, vel_n)
        f_nl_n = self._nonlinear_force(disp_n, vel_n)
        acc_n = (f_lin_n + f_nl_n + force_ext) / m

        disp_p = disp_n + dt * vel_n
        vel_p = vel_n + dt * acc_n


        f_nl_p = self._nonlinear_force(disp_p, vel_p)
        f_nl_avg = 0.5 * (f_nl_n + f_nl_p)




























        raise NotImplementedError("Hole 3: 隐式梯形法矩阵求解尚未实现")


        if self.bw_state is not None:
            vel_avg = 0.5 * (vel_n + new_state[2:4])
            dz = self._bouc_wen_derivative(vel_avg, self.bw_state)
            self.bw_state += dt * dz

        self.prev_state = state_n.copy()
        self.prev_force = force_ext.copy()
        self.state = new_state

    def step_rk4(self, force_ext):
        if self.dt is None:
            raise RuntimeError("必须先调用 set_time_step(dt)。")

        dt = self.dt
        y = self.state.copy()

        k1 = self._rhs(y, force_ext)
        k2 = self._rhs(y + 0.5 * dt * k1, force_ext)
        k3 = self._rhs(y + 0.5 * dt * k2, force_ext)
        k4 = self._rhs(y + dt * k3, force_ext)

        self.prev_state = y.copy()
        self.prev_force = force_ext.copy()
        self.state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if self.bw_state is not None:
            vel = self.state[2:4]
            dz = self._bouc_wen_derivative(vel, self.bw_state)
            self.bw_state += dt * dz

    def step(self, force_ext):
        if self.time_integrator == 'cn_rk2':
            self.step_cn_rk2(force_ext)
        elif self.time_integrator == 'rk4':
            self.step_rk4(force_ext)
        else:
            raise ValueError(f"未知积分器: {self.time_integrator}")

    def get_displacement(self):
        return self.state[0:2].copy()

    def get_velocity(self):
        return self.state[2:4].copy()

    def get_acceleration(self, force_ext):
        disp = self.state[0:2]
        vel = self.state[2:4]
        f_lin = self._linear_force(disp, vel)
        f_nl = self._nonlinear_force(disp, vel)
        return (f_lin + f_nl + force_ext) / self.mass

    def get_amplitude_envelope(self, history_disp, window=50):
        N = len(history_disp)
        amp = np.zeros_like(history_disp)
        half = window // 2
        for i in range(N):
            i0 = max(0, i - half)
            i1 = min(N, i + half)
            local_max = np.max(history_disp[i0:i1], axis=0)
            local_min = np.min(history_disp[i0:i1], axis=0)
            amp[i] = 0.5 * (local_max - local_min)
        return amp
