# -*- coding: utf-8 -*-

import numpy as np
from parameters import get_parameters


class ImpurityTransport:

    def __init__(self, params=None):
        if params is None:
            params = get_parameters()
        self.params = params
        self.n0 = params.get('n_0')
        self.Te = params.get('T_e')
        self.lambda_D = params.debye_length()
        self.c_s = params.ion_sound_speed()


        self.D_bohm = self.Te / (16.0 * params.get('B_t')) if params.get('B_t') > 0 else 1.0
        self.gamma_coll = self.c_s / self.lambda_D if self.lambda_D > 0 else 1.0e6

    def compute_diffusion_coefficient(self, model='bohm'):
        if model == 'bohm':
            B = self.params.get('B_t')
            if B <= 0:
                B = 1.0
            return self.Te / (16.0 * B)
        elif model == 'classical':

            e_charge = 1.602176634e-19
            epsilon_0 = 8.854187817e-12
            ln_lambda = 15.0
            m_p = 1.67262192369e-27
            mi_kg = self.params.get('m_i') * m_p
            nu_ii = (self.n0 * e_charge**4 * ln_lambda /
                     (12.0 * np.pi**1.5 * epsilon_0**2 * np.sqrt(mi_kg) *
                      (self.Te * e_charge)**1.5))
            D_cl = self.Te * e_charge / (mi_kg * max(nu_ii, 1.0e-10))
            return D_cl
        elif model == 'neo':

            q = 3.0
            m_p = 1.67262192369e-27
            mi_kg = self.params.get('m_i') * m_p
            e_charge = 1.602176634e-19
            B = max(self.params.get('B_t'), 0.1)
            rho_i = np.sqrt(mi_kg * self.Te * e_charge) / (e_charge * B)
            nu_i = self.gamma_coll
            D_neo = q**2 * rho_i**2 * nu_i
            return D_neo
        else:
            return self.D_bohm

    def ifs_transport_map(self, x, mode='diffusion'):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n = x.shape[0]
        x_new = np.zeros_like(x)


        if mode == 'diffusion':
            A0 = np.array([[0.0, 0.0], [0.0, 0.5]])
            b0 = np.array([0.5, 0.0])
            A1 = np.array([[0.1, 0.0], [0.0, 0.1]])
            b1 = np.array([0.45, 0.15])
            A2 = np.array([[0.42, -0.42], [0.42, 0.42]])
            b2 = np.array([0.29, -0.01])
            A3 = np.array([[0.42, 0.42], [-0.42, 0.42]])
            b3 = np.array([0.29, 0.41])
            probs = [0.25, 0.25, 0.25, 0.25]
        elif mode == 'sheath_drift':

            A0 = np.array([[0.8, 0.0], [0.0, 0.3]])
            b0 = np.array([0.15, 0.0])
            A1 = np.array([[0.2, 0.0], [0.0, 0.2]])
            b1 = np.array([0.1, 0.1])
            A2 = np.array([[0.5, 0.1], [0.0, 0.5]])
            b2 = np.array([0.2, 0.0])
            A3 = np.array([[0.3, 0.0], [0.0, 0.3]])
            b3 = np.array([0.1, 0.2])
            probs = [0.5, 0.2, 0.2, 0.1]
        else:
            A0, b0 = np.eye(2), np.zeros(2)
            A1, b1 = np.eye(2) * 0.5, np.zeros(2)
            A2, b2 = np.eye(2) * 0.3, np.zeros(2)
            A3, b3 = np.eye(2) * 0.2, np.zeros(2)
            probs = [0.25, 0.25, 0.25, 0.25]


        cumprobs = np.cumsum(probs)

        for i in range(n):
            r = np.random.rand()
            if r < cumprobs[0]:
                x_new[i] = A0.dot(x[i]) + b0
            elif r < cumprobs[1]:
                x_new[i] = A1.dot(x[i]) + b1
            elif r < cumprobs[2]:
                x_new[i] = A2.dot(x[i]) + b2
            else:
                x_new[i] = A3.dot(x[i]) + b3

        return x_new

    def simulate_langevin_trajectory(self, x0, v0, dt, n_steps, D=None):
        if D is None:
            D = self.compute_diffusion_coefficient()

        traj = np.zeros((n_steps + 1, 3))
        vel = np.zeros((n_steps + 1, 3))
        traj[0] = x0
        vel[0] = v0

        gamma = self.gamma_coll
        sqrt_term = np.sqrt(2.0 * D * gamma * dt)

        for n in range(n_steps):

            noise = np.random.randn(3)
            vel[n+1] = vel[n] - gamma * vel[n] * dt + sqrt_term * noise


            traj[n+1] = traj[n] + vel[n+1] * dt


            if traj[n+1, 0] < 0:
                traj[n+1, 0] = -traj[n+1, 0]
                vel[n+1, 0] = -vel[n+1, 0]

        return traj, vel

    def simulate_ensemble(self, n_particles, n_steps, dt, D=None):
        if D is None:
            D = self.compute_diffusion_coefficient()

        final_positions = np.zeros((n_particles, 3))
        displacements = np.zeros(n_particles)

        lambda_D = self.lambda_D
        v_th = self.params.ion_thermal_velocity()

        for p in range(n_particles):

            x0 = np.array([np.random.exponential(lambda_D),
                           np.random.normal(0, lambda_D),
                           np.random.normal(0, lambda_D)])
            v0 = np.random.randn(3) * v_th

            traj, _ = self.simulate_langevin_trajectory(x0, v0, dt, n_steps, D)
            final_positions[p] = traj[-1]
            displacements[p] = np.linalg.norm(traj[-1] - traj[0])

        stats = {
            'mean_displacement': np.mean(displacements),
            'std_displacement': np.std(displacements),
            'mean_final_x': np.mean(final_positions[:, 0]),
            'mean_final_y': np.mean(final_positions[:, 1]),
            'mean_final_z': np.mean(final_positions[:, 2]),
            'return_fraction': np.sum(final_positions[:, 0] < lambda_D) / n_particles,
        }

        return final_positions, stats

    def compute_deposition_profile(self, n_particles=1000, n_steps=500, dt=1.0e-9):
        final_positions, stats = self.simulate_ensemble(n_particles, n_steps, dt)


        x_bins = np.linspace(0, 5 * self.lambda_D, 51)
        counts, edges = np.histogram(final_positions[:, 0], bins=x_bins)
        dx = edges[1] - edges[0]
        density = counts / (n_particles * dx)

        return edges[:-1], density, stats


def demo_transport():
    transport = ImpurityTransport()


    D_bohm = transport.compute_diffusion_coefficient('bohm')
    D_classical = transport.compute_diffusion_coefficient('classical')
    D_neo = transport.compute_diffusion_coefficient('neo')

    print("扩散系数对比:")
    print(f"  Bohm扩散:       D = {D_bohm:.3e} m^2/s")
    print(f"  经典碰撞扩散:   D = {D_classical:.3e} m^2/s")
    print(f"  新经典扩散:     D = {D_neo:.3e} m^2/s")


    print("\n侵蚀杂质系综输运模拟 (100粒子)...")
    final_pos, stats = transport.simulate_ensemble(n_particles=100, n_steps=200, dt=1.0e-10)

    print("统计结果:")
    for k, v in stats.items():
        print(f"  {k}: {v:.3e}")


    x = np.random.rand(2)
    print(f"\nIFS混沌映射测试: x0 = {x}")
    for _ in range(5):
        x = transport.ifs_transport_map(x, mode='sheath_drift')
    print(f"  5步后: x = {x.flatten()}")

    return transport


if __name__ == "__main__":
    demo_transport()
