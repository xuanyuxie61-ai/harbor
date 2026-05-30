
import numpy as np




def gaussian_influence_function(x, y, x0, y0, sigma):
    dx = x - x0
    dy = y - y0
    return np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))


def thin_plate_spline_influence(r, r0, c):
    if np.any(r < 0):
        r = np.clip(r, 0, None)
    phi = np.zeros_like(r, dtype=np.float64)
    mask = r > 1e-10
    phi[mask] = c * (r[mask] ** 2) * np.log(r[mask] / max(r0, 1e-10))
    return phi




def magic4_matrix(n):
    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4.")
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            k1 = i * n + j + 1
            m1 = abs(i - j) % 4
            m2 = (i + j + 1) % 4
            if m1 == 0 or m2 == 0:
                M[i, j] = n * n + 1 - k1
            else:
                M[i, j] = k1
    return M


def generate_actuator_layout(n_actuators, aperture_radius=1.0, use_magic_square=False):
    if n_actuators < 1:
        raise ValueError("n_actuators must be >= 1.")


    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_actuators)
    r = aperture_radius * np.sqrt(indices / (n_actuators - 0.5)) if n_actuators > 1 else np.array([0.0])
    theta = indices * golden_angle

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if use_magic_square and n_actuators >= 16:

        magic = magic4_matrix(4)
        magic_norm = (magic - np.mean(magic)) / (np.max(magic) - np.min(magic) + 1e-10)
        for idx in range(min(16, n_actuators)):
            i = idx // 4
            j = idx % 4
            perturb = magic_norm[i, j] * 0.05 * aperture_radius
            x[idx] += perturb * np.cos(theta[idx])
            y[idx] += perturb * np.sin(theta[idx])

    return x, y




class DeformableMirror:

    def __init__(self, n_actuators, grid_size, aperture_radius=1.0,
                 influence_sigma=None, use_magic_square_layout=False):
        self.n_actuators = n_actuators
        self.grid_size = grid_size
        self.aperture_radius = aperture_radius

        if influence_sigma is None:
            influence_sigma = 0.15 * aperture_radius

        self.influence_sigma = influence_sigma
        self.act_x, self.act_y = generate_actuator_layout(
            n_actuators, aperture_radius, use_magic_square_layout
        )


        x = np.linspace(-aperture_radius, aperture_radius, grid_size)
        y = np.linspace(-aperture_radius, aperture_radius, grid_size)
        X, Y = np.meshgrid(x, y)

        self.influence_matrix = np.zeros((grid_size, grid_size, n_actuators), dtype=np.float64)
        for k in range(n_actuators):
            self.influence_matrix[:, :, k] = gaussian_influence_function(
                X, Y, self.act_x[k], self.act_y[k], influence_sigma
            )


        self.influence_flat = self.influence_matrix.reshape(grid_size * grid_size, n_actuators)

    def compute_surface(self, voltages):
        if len(voltages) != self.n_actuators:
            raise ValueError("voltages length must equal n_actuators.")
        surface = np.tensordot(self.influence_matrix, voltages, axes=([2], [0]))
        return surface

    def compute_surface_flat(self, voltages):
        return self.influence_flat @ voltages

    def voltage_to_zernike_response(self, zernike_basis_flat, mask):
        n_modes = zernike_basis_flat.shape[1]
        R = np.zeros((n_modes, self.n_actuators), dtype=np.float64)
        mask_vec = mask.ravel()

        for k in range(self.n_actuators):
            Ik = self.influence_flat[:, k]
            for j in range(n_modes):
                Zj = zernike_basis_flat[:, j]
                num = np.sum(Zj[mask_vec] * Ik[mask_vec])
                den = np.sum(Zj[mask_vec] ** 2)
                if den > 1e-30:
                    R[j, k] = num / den

        return R




class FastSteeringMirrorDynamics:

    def __init__(self, g=9.81, m1=0.01, m2=0.01, l1=0.05, l2=0.05,
                 damping1=0.5, damping2=0.5, coupling=0.1):
        self.g = g
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.damping1 = damping1
        self.damping2 = damping2
        self.coupling = coupling

    def derivatives(self, state, t, control_torque1, control_torque2):
        theta1, omega1, theta2, omega2 = state

        delta = theta1 - theta2
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)
        denom1 = 2.0 * self.l1 * (self.m1 + self.m2 - self.m2 * cos_delta ** 2)
        denom2 = self.l2 * (self.m1 + self.m2 - self.m2 * cos_delta ** 2)

        if abs(denom1) < 1e-10:
            denom1 = 1e-10
        if abs(denom2) < 1e-10:
            denom2 = 1e-10

        alpha1 = (-self.g * (2.0 * self.m1 + self.m2) * np.sin(theta1)
                  - self.m2 * self.g * np.sin(theta1 - 2.0 * theta2)
                  - 2.0 * sin_delta * self.m2 * (
                      omega2 ** 2 * self.l2 + omega1 ** 2 * self.l1 * cos_delta)
                  + control_torque1 - self.damping1 * omega1
                  + self.coupling * (theta2 - theta1)) / denom1

        alpha2 = ((self.m1 + self.m2) * (
                      self.l1 * omega1 ** 2 + self.g * np.cos(theta1)) * sin_delta
                  + self.m2 * self.l2 * omega2 ** 2 * sin_delta * cos_delta
                  + control_torque2 - self.damping2 * omega2
                  + self.coupling * (theta1 - theta2)) / denom2

        return np.array([omega1, alpha1, omega2, alpha2], dtype=np.float64)

    def step_rk4(self, state, dt, t, torque1, torque2):
        k1 = dt * self.derivatives(state, t, torque1, torque2)
        k2 = dt * self.derivatives(state + 0.5 * k1, t + 0.5 * dt, torque1, torque2)
        k3 = dt * self.derivatives(state + 0.5 * k2, t + 0.5 * dt, torque1, torque2)
        k4 = dt * self.derivatives(state + k3, t + dt, torque1, torque2)
        return state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def simulate_response(self, control_sequence_x, control_sequence_y, dt=1e-4):
        n_steps = len(control_sequence_x)
        if len(control_sequence_y) != n_steps:
            raise ValueError("control sequences must have same length.")

        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        history = np.zeros((n_steps, 4), dtype=np.float64)

        for i in range(n_steps):
            state = self.step_rk4(state, dt, i * dt,
                                  control_sequence_x[i], control_sequence_y[i])
            history[i, :] = state

        return history[:, 0], history[:, 2]
