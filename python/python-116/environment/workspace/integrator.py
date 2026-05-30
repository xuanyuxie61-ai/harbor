
import numpy as np


class MDIntegrator:

    def __init__(self, system, friction_gamma=0.5, seed=None):
        if system is None:
            raise ValueError("system 不能为 None。")
        if friction_gamma < 0:
            raise ValueError("摩擦系数必须非负。")

        self.sys = system
        self.gamma = friction_gamma
        self.rng = np.random.default_rng(seed)

    def step(self):
        dt = self.sys.dt
        m = self.sys.mass
        nx, ny = self.sys.nx, self.sys.ny


        self.sys.compute_forces()


        T_local = self.sys.temperature_field
        sigma = np.sqrt(2.0 * self.gamma * self.sys.kb * T_local / dt)
        noise_theta = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma
        noise_phi = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma




        damping = 1.0 - 0.5 * self.gamma * dt / m
        damping = max(damping, 0.0)

        self.sys.omega_theta = (damping * self.sys.omega_theta +
                                0.5 * dt * (self.sys.torque_theta + noise_theta) / m)
        self.sys.omega_phi = (damping * self.sys.omega_phi +
                              0.5 * dt * (self.sys.torque_phi + noise_phi) / m)


        self.sys.theta = self.sys.theta + dt * self.sys.omega_theta
        self.sys.phi = self.sys.phi + dt * self.sys.omega_phi


        self.sys.theta = np.mod(self.sys.theta, 2.0 * np.pi)
        self.sys.phi = np.mod(self.sys.phi, 2.0 * np.pi)


        self.sys.area = self.sys.area + dt * self.sys.force_area / m
        self.sys.area = np.clip(self.sys.area, 0.1 * self.sys.area0, 5.0 * self.sys.area0)


        self.sys.compute_forces()


        noise_theta2 = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma
        noise_phi2 = self.rng.normal(0.0, 1.0, (nx, ny)) * sigma

        self.sys.omega_theta = (damping * self.sys.omega_theta +
                                0.5 * dt * (self.sys.torque_theta + noise_theta2) / m)
        self.sys.omega_phi = (damping * self.sys.omega_phi +
                              0.5 * dt * (self.sys.torque_phi + noise_phi2) / m)

    def run_equilibration(self, n_steps=2000):
        energy_trace = []
        s2_trace = []
        for step in range(n_steps):
            self.step()
            if step % 10 == 0:
                e = self.sys.compute_total_energy()
                s2 = self.sys.global_order_parameter()
                energy_trace.append(e)
                s2_trace.append(s2)
        return np.array(energy_trace), np.array(s2_trace)


class IntegratorStability:

    def __init__(self, integrator_type='verlet'):
        if integrator_type not in ('verlet', 'euler', 'rk4'):
            raise ValueError("integrator_type 必须是 'verlet', 'euler' 或 'rk4'")
        self.integrator_type = integrator_type

    def amplification_factor(self, z):
        z = np.asarray(z, dtype=complex)
        if self.integrator_type == 'euler':

            return 1.0 + z
        elif self.integrator_type == 'rk4':

            return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
        elif self.integrator_type == 'verlet':



            disc = z**2 + 4.0
            sqrt_disc = np.sqrt(disc)
            lambda1 = 1.0 + z**2 / 2.0 + 0.5 * z * sqrt_disc
            lambda2 = 1.0 + z**2 / 2.0 - 0.5 * z * sqrt_disc
            return np.maximum(np.abs(lambda1), np.abs(lambda2))
        else:
            raise NotImplementedError

    def stability_region_mask(self, xlim=(-3.0, 3.0), ylim=(-3.0, 3.0),
                               npts=401):
        x = np.linspace(xlim[0], xlim[1], npts)
        y = np.linspace(ylim[0], ylim[1], npts)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        Rval = self.amplification_factor(Z)
        mask = Rval <= 1.0
        return X, Y, mask

    def check_system_stability(self, omega_max=2.5, gamma=0.5, dt=None):
        if dt is None:
            dt = 0.002


        if self.integrator_type == 'verlet':
            stable = omega_max * dt < 2.0
            disc = gamma**2 - 4.0 * omega_max**2
            if disc >= 0:
                lambda1 = (-gamma + np.sqrt(disc)) / 2.0
                lambda2 = (-gamma - np.sqrt(disc)) / 2.0
            else:
                lambda1 = (-gamma + 1j * np.sqrt(-disc)) / 2.0
                lambda2 = (-gamma - 1j * np.sqrt(-disc)) / 2.0
            z_points = [dt * lambda1, dt * lambda2]
            return stable, z_points

        disc = gamma**2 - 4.0 * omega_max**2
        if disc >= 0:
            lambda1 = (-gamma + np.sqrt(disc)) / 2.0
            lambda2 = (-gamma - np.sqrt(disc)) / 2.0
        else:
            lambda1 = (-gamma + 1j * np.sqrt(-disc)) / 2.0
            lambda2 = (-gamma - 1j * np.sqrt(-disc)) / 2.0

        z_points = [dt * lambda1, dt * lambda2]
        stable = all(np.abs(self.amplification_factor(z)) <= 1.0 + 1e-10
                     for z in z_points)
        return stable, z_points
