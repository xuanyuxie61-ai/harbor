# -*- coding: utf-8 -*-

import numpy as np


class PolygonRoughness:

    def __init__(self, n_vertices=100):
        if n_vertices < 4:
            raise ValueError("n_vertices 必须至少为 4")
        self.n = n_vertices
        self.x = np.linspace(0.0, 1.0, n_vertices)
        self.h = np.zeros(n_vertices)

    def initialize_random_roughness(self, amplitude=1.0e-6, n_modes=10, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.h = np.zeros(self.n)
        for k in range(1, n_modes + 1):
            A_k = amplitude / k
            phi_k = np.random.uniform(0.0, 2.0 * np.pi)
            self.h += A_k * np.sin(2.0 * np.pi * k * self.x + phi_k)


        max_h = np.max(np.abs(self.h))
        if max_h > 1.0e-20:
            self.h = self.h / max_h * amplitude

        return self

    def averaging_step(self, alpha=0.5):
        if alpha < 0.0 or alpha > 1.0:
            alpha = max(0.0, min(1.0, alpha))


        h2 = np.zeros(self.n)
        for i in range(self.n):
            ip1 = (i + 1) % self.n
            h2[i] = (1.0 - alpha) * self.h[i] + alpha * 0.5 * (self.h[i] + self.h[ip1])


        h2 = h2 - np.mean(h2)


        max_h = np.max(np.abs(h2))
        if max_h > 1.0e-20:
            h2 = h2 / max_h * np.max(np.abs(self.h))

        self.h = h2
        return self

    def apply_erosion(self, erosion_rate, dt, material_removal_func=None):
        if erosion_rate < 0:
            erosion_rate = 0.0
        if dt < 0:
            dt = 0.0


        curvature = np.zeros(self.n)
        dx = self.x[1] - self.x[0]
        if dx > 1.0e-20:
            for i in range(1, self.n - 1):
                curvature[i] = (self.h[i+1] - 2*self.h[i] + self.h[i-1]) / (dx*dx)

            curvature[0] = (self.h[1] - 2*self.h[0] + self.h[-1]) / (dx*dx)
            curvature[-1] = (self.h[0] - 2*self.h[-1] + self.h[-2]) / (dx*dx)


        if material_removal_func is not None:
            dh = material_removal_func(self.x, self.h, curvature) * dt
        else:
            enhancement = 1.0 + 0.5 * np.abs(curvature) / (np.max(np.abs(curvature)) + 1.0e-20)
            dh = -erosion_rate * enhancement * dt

        self.h += dh
        return self

    def apply_redeposition(self, redeposition_rate, dt, stochastic=True):
        if redeposition_rate < 0:
            redeposition_rate = 0.0
        if dt < 0:
            dt = 0.0

        dh = redeposition_rate * dt * np.ones(self.n)

        if stochastic:
            noise_amplitude = 0.3 * redeposition_rate * dt
            dh += noise_amplitude * (2.0 * np.random.rand(self.n) - 1.0)

        self.h += dh
        return self

    def compute_roughness_parameters(self):
        h_centered = self.h - np.mean(self.h)
        n = float(self.n)

        Ra = np.mean(np.abs(h_centered))
        Rq = np.sqrt(np.mean(h_centered**2))
        Rz = np.max(self.h) - np.min(self.h)

        if Rq > 1.0e-30:
            skewness = np.mean(h_centered**3) / (Rq**3)
            kurtosis = np.mean(h_centered**4) / (Rq**4)
        else:
            skewness = 0.0
            kurtosis = 3.0

        return {
            'Ra': Ra,
            'Rq': Rq,
            'Rz': Rz,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }

    def evolve_surface(self, n_steps, erosion_rate, redeposition_rate, dt,
                       alpha_smooth=0.3, stochastic=True):
        history = []
        for step in range(n_steps):
            self.averaging_step(alpha=alpha_smooth)
            self.apply_erosion(erosion_rate, dt)
            self.apply_redeposition(redeposition_rate, dt, stochastic=stochastic)

            stats = self.compute_roughness_parameters()
            stats['step'] = step
            history.append(stats)

        return history


def demo_roughness():
    surface = PolygonRoughness(n_vertices=128)
    surface.initialize_random_roughness(amplitude=1.0e-6, n_modes=15, seed=42)

    init_stats = surface.compute_roughness_parameters()
    print("初始粗糙度参数:")
    for k, v in init_stats.items():
        print(f"  {k}: {v:.3e}")


    history = surface.evolve_surface(
        n_steps=50,
        erosion_rate=1.0e-9,
        redeposition_rate=3.0e-10,
        dt=1.0e-3,
        alpha_smooth=0.2,
        stochastic=True
    )

    final_stats = surface.compute_roughness_parameters()
    print("\n最终粗糙度参数:")
    for k, v in final_stats.items():
        print(f"  {k}: {v:.3e}")

    return surface, history


if __name__ == "__main__":
    demo_roughness()
