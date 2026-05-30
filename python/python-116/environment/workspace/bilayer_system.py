
import numpy as np


class LipidBilayerSystem:

    def __init__(self, nx=24, ny=24, k_boltzmann=1.380649e-23,
                 j_coupling=2.5, epsilon_nn=1.0, kappa_a=25.0,
                 area0=0.64, dt_md=0.002, mass=1.0):
        if nx < 4 or ny < 4:
            raise ValueError("格点尺寸 nx, ny 必须至少为 4 以保证边界处理有效。")
        if j_coupling <= 0 or epsilon_nn <= 0 or kappa_a <= 0:
            raise ValueError("耦合常数与模量必须为正。")
        if area0 <= 0:
            raise ValueError("平衡面积必须为正。")
        if dt_md <= 0:
            raise ValueError("时间步长必须为正。")

        self.nx = nx
        self.ny = ny
        self.n_lipids = nx * ny
        self.kb = k_boltzmann
        self.J = j_coupling
        self.eps_nn = epsilon_nn
        self.kappa_a = kappa_a
        self.area0 = area0
        self.dt = dt_md
        self.mass = mass



        self.theta = np.full((nx, ny), 0.1) + 0.05 * np.random.randn(nx, ny)
        self.phi = 2.0 * np.pi * np.random.rand(nx, ny)


        self.omega_theta = np.zeros((nx, ny))
        self.omega_phi = np.zeros((nx, ny))


        self.area = np.full((nx, ny), area0)


        self.temperature_field = np.ones((nx, ny)) * 300.0


        self.torque_theta = np.zeros((nx, ny))
        self.torque_phi = np.zeros((nx, ny))
        self.force_area = np.zeros((nx, ny))

    def _p2_legendre(self, cos_theta):
        return 0.5 * (3.0 * cos_theta ** 2 - 1.0)

    def _safe_modulo(self, idx, max_idx):
        return idx % max_idx

    def compute_local_order_parameter(self):
        nx, ny = self.nx, self.ny
        s2 = np.zeros((nx, ny))
        cos_t = np.cos(self.theta)
        p2 = self._p2_legendre(cos_t)

        for i in range(nx):
            for j in range(ny):
                s = 0.0
                count = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ii = self._safe_modulo(i + di, nx)
                    jj = self._safe_modulo(j + dj, ny)
                    s += p2[ii, jj]
                    count += 1
                s2[i, j] = s / count if count > 0 else 0.0
        return s2

    def compute_total_energy(self):
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        p2 = self._p2_legendre(cos_t)
        s2 = self.compute_local_order_parameter()


        e_orient = -self.J * np.sum(p2 * s2)


        e_nn = 0.0
        for i in range(self.nx):
            for j in range(self.ny):
                n_i = np.array([
                    sin_t[i, j] * np.cos(self.phi[i, j]),
                    sin_t[i, j] * np.sin(self.phi[i, j]),
                    cos_t[i, j]
                ])
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ii = self._safe_modulo(i + di, self.nx)
                    jj = self._safe_modulo(j + dj, self.ny)
                    n_j = np.array([
                        sin_t[ii, jj] * np.cos(self.phi[ii, jj]),
                        sin_t[ii, jj] * np.sin(self.phi[ii, jj]),
                        cos_t[ii, jj]
                    ])
                    dot = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
                    e_nn -= self.eps_nn * dot ** 2
        e_nn *= 0.5


        dA = self.area - self.area0
        e_compress = 0.5 * self.kappa_a * np.sum(dA ** 2 / self.area0)


        ratio = self.area / self.area0
        ratio = np.where(ratio > 1e-12, ratio, 1e-12)
        e_entropy = np.sum(self.kb * self.temperature_field * np.log(ratio))

        return e_orient + e_nn + e_compress + e_entropy

    def compute_forces(self):
        nx, ny = self.nx, self.ny
        self.torque_theta = np.zeros((nx, ny))
        self.torque_phi = np.zeros((nx, ny))
        self.force_area = np.zeros((nx, ny))

        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        s2 = self.compute_local_order_parameter()




        dp2_dcos = 3.0 * cos_t
        dcos_dtheta = -sin_t
        self.torque_theta += -self.J * dp2_dcos * dcos_dtheta * s2


        eps = 1e-6
        for i in range(nx):
            for j in range(ny):
                e0 = self._local_nn_energy(i, j)
                self.theta[i, j] += eps
                e1 = self._local_nn_energy(i, j)
                self.theta[i, j] -= eps
                self.torque_theta[i, j] += -(e1 - e0) / eps

                e0 = self._local_nn_energy(i, j)
                self.phi[i, j] += eps
                e1 = self._local_nn_energy(i, j)
                self.phi[i, j] -= eps
                self.torque_phi[i, j] += -(e1 - e0) / eps


        dA = self.area - self.area0
        self.force_area = -(self.kappa_a * dA / self.area0 +
                            self.kb * self.temperature_field / self.area)

    def _local_nn_energy(self, i, j):
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        n_i = np.array([
            sin_t[i, j] * np.cos(self.phi[i, j]),
            sin_t[i, j] * np.sin(self.phi[i, j]),
            cos_t[i, j]
        ])
        e = 0.0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii = self._safe_modulo(i + di, self.nx)
            jj = self._safe_modulo(j + dj, self.ny)
            n_j = np.array([
                sin_t[ii, jj] * np.cos(self.phi[ii, jj]),
                sin_t[ii, jj] * np.sin(self.phi[ii, jj]),
                cos_t[ii, jj]
            ])
            dot = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
            e -= self.eps_nn * dot ** 2
        return e

    def thermalize_temperature_field(self, boundary_temp_high=350.0,
                                      boundary_temp_low=250.0,
                                      epsilon_conv=1e-4,
                                      max_iter=5000):
        nx, ny = self.nx, self.ny
        T = self.temperature_field.copy()


        T[:, 0] = boundary_temp_low
        T[:, -1] = boundary_temp_low
        T[0, :] = boundary_temp_high
        T[-1, :] = boundary_temp_high

        diff = epsilon_conv + 1.0
        iteration = 0
        while diff > epsilon_conv and iteration < max_iter:
            T_old = T.copy()

            T[1:nx-1, 1:ny-1] = 0.25 * (
                T_old[0:nx-2, 1:ny-1] +
                T_old[2:nx, 1:ny-1] +
                T_old[1:nx-1, 0:ny-2] +
                T_old[1:nx-1, 2:ny]
            )

            T[:, 0] = boundary_temp_low
            T[:, -1] = boundary_temp_low
            T[0, :] = boundary_temp_high
            T[-1, :] = boundary_temp_high

            diff = np.max(np.abs(T - T_old))
            iteration += 1

        self.temperature_field = T
        return iteration, diff

    def global_order_parameter(self):
        cos_t = np.cos(self.theta)
        p2 = self._p2_legendre(cos_t)
        return float(np.mean(p2))

    def get_positions(self):
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y
