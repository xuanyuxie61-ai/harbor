
import numpy as np




class PIController:

    def __init__(self, Kp=0.5, Ki=0.1, dt=1e-3, integral_limit=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.dt = dt
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.error_history = []

    def reset(self):
        self.integral = 0.0
        self.error_history = []

    def update(self, error):
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        self.error_history.append(error)
        u = self.Kp * error + self.Ki * self.integral
        return u




class BandwidthLimitedActuator:

    def __init__(self, bandwidth_hz, dt):
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive.")
        self.tau = 1.0 / (2.0 * np.pi * bandwidth_hz)
        self.dt = dt
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def step(self, input_cmd):
        alpha = self.dt / (self.tau + self.dt)
        self.state = self.state + alpha * (input_cmd - self.state)
        return self.state




class HHControlCircuit:

    def __init__(self, C=1.0, E_K=-74.7, E_Na=54.2, G_K=12.0, G_Na=30.0,
                 dt=1e-5):
        self.C = C
        self.E_K = E_K
        self.E_Na = E_Na
        self.G_K = G_K
        self.G_Na = G_Na
        self.dt = dt

        self.V = -65.0
        self.n = 0.3177
        self.m = 0.0529
        self.h = 0.5961

    def reset(self):
        self.V = -65.0
        self.n = 0.3177
        self.m = 0.0529
        self.h = 0.5961

    def alpha_n(self, V):
        if abs(V - 10.0) < 1e-6:
            return 0.1
        return 0.01 * (10.0 - V) / (np.exp((10.0 - V) / 10.0) - 1.0)

    def beta_n(self, V):
        return 0.125 * np.exp(-V / 80.0)

    def alpha_m(self, V):
        if abs(V - 25.0) < 1e-6:
            return 1.0
        return 0.1 * (25.0 - V) / (np.exp((25.0 - V) / 10.0) - 1.0)

    def beta_m(self, V):
        return 4.0 * np.exp(-V / 18.0)

    def alpha_h(self, V):
        return 0.07 * np.exp(-V / 20.0)

    def beta_h(self, V):
        return 1.0 / (np.exp((30.0 - V) / 10.0) + 1.0)

    def step(self, I_ext):
        I_K = self.n ** 4 * self.G_K * (self.V - self.E_K)
        I_Na = self.m ** 3 * self.G_Na * self.h * (self.V - self.E_Na)

        dV = (I_ext - I_K - I_Na) / self.C
        dn = self.alpha_n(self.V) * (1.0 - self.n) - self.beta_n(self.V) * self.n
        dm = self.alpha_m(self.V) * (1.0 - self.m) - self.beta_m(self.V) * self.m
        dh = self.alpha_h(self.V) * (1.0 - self.h) - self.beta_h(self.V) * self.h

        self.V += dV * self.dt
        self.n += dn * self.dt
        self.n = np.clip(self.n, 0.0, 1.0)
        self.m += dm * self.dt
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h += dh * self.dt
        self.h = np.clip(self.h, 0.0, 1.0)

        return self.V




class ClosedLoopAO:

    def __init__(self, controller, actuator, n_modes, dt=1e-3):
        self.controller = controller
        self.actuator = actuator
        self.n_modes = n_modes
        self.dt = dt
        self.control_history = []
        self.residual_history = []
        self.strehl_history = []

    def reset(self):
        self.controller.reset()
        self.actuator.reset()
        self.control_history = []
        self.residual_history = []
        self.strehl_history = []

    def step(self, residual_error, strehl=None):
        u_raw = self.controller.update(residual_error)
        u_filtered = self.actuator.step(u_raw)
        self.control_history.append(u_filtered)
        self.residual_history.append(residual_error)
        if strehl is not None:
            self.strehl_history.append(strehl)
        return u_filtered


def parameter_sweep_optimization(Kp_grid, Ki_grid, bandwidth_grid,
                                  simulation_func, dt=1e-3, n_steps=500):
    if len(Kp_grid) == 0 or len(Ki_grid) == 0 or len(bandwidth_grid) == 0:
        raise ValueError("Parameter grids must be non-empty.")

    n_kp = len(Kp_grid)
    n_ki = len(Ki_grid)
    n_bw = len(bandwidth_grid)

    performance = np.zeros((n_kp, n_ki, n_bw), dtype=np.float64)
    best_strehl = -1.0
    best_params = (Kp_grid[0], Ki_grid[0], bandwidth_grid[0])

    for i, Kp in enumerate(Kp_grid):
        for j, Ki in enumerate(Ki_grid):
            for k, bw in enumerate(bandwidth_grid):
                try:
                    strehl = simulation_func(Kp, Ki, bw, dt, n_steps)
                    if not np.isfinite(strehl):
                        strehl = 0.0
                except Exception:
                    strehl = 0.0
                performance[i, j, k] = strehl
                if strehl > best_strehl:
                    best_strehl = strehl
                    best_params = (Kp, Ki, bw)

    return best_params, best_strehl, performance


def simulate_modal_control_loop(n_modes, turb_covariance, Kp, Ki, bandwidth_hz,
                                dt=1e-3, n_steps=1000, noise_std=0.01):
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    tau = 0.01
    decay = np.exp(-dt / tau)
    diffusion = np.sqrt(1.0 - decay ** 2)


    controllers = [PIController(Kp=Kp, Ki=Ki, dt=dt, integral_limit=1.0) for _ in range(n_modes)]
    actuators = [BandwidthLimitedActuator(bandwidth_hz, dt) for _ in range(n_modes)]


    a_turb = np.random.multivariate_normal(np.zeros(n_modes), turb_covariance)
    a_corr = np.zeros(n_modes)

    residual_history = []
    strehl_history = []

    for step in range(n_steps):

        a_turb = decay * a_turb + diffusion * np.random.multivariate_normal(np.zeros(n_modes), turb_covariance)


        a_meas = a_turb - a_corr + np.random.normal(0, noise_std, n_modes)


        da_corr = np.zeros(n_modes)
        for m in range(n_modes):
            u = controllers[m].update(a_meas[m])
            u_filt = actuators[m].step(u)
            da_corr[m] = u_filt * dt

        a_corr = a_corr + da_corr
        a_corr = np.clip(a_corr, -5.0, 5.0)

        residual = a_turb - a_corr
        residual_history.append(np.linalg.norm(residual))


        sigma_phi_sq = np.sum(residual ** 2)
        strehl = np.exp(-min(sigma_phi_sq, 50.0))
        strehl_history.append(strehl)

    final_strehl = strehl_history[-1] if len(strehl_history) > 0 else 0.0
    mean_strehl = np.mean(strehl_history) if len(strehl_history) > 0 else 0.0

    if mean_strehl < 1e-6 and len(strehl_history) > 100:
        mean_strehl = np.mean(strehl_history[-100:])
    return final_strehl, mean_strehl, np.array(residual_history), np.array(strehl_history)
