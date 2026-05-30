# -*- coding: utf-8 -*-

import numpy as np


class DefectGenerator:

    def __init__(self, geometry, seed: int = 42):
        self.geom = geometry
        self.rng = np.random.default_rng(seed)

    def single_mode_defect(self, m: int, n: int, amplitude: float,
                           phase: float = 0.0) -> callable:
        R, L = self.geom.R, self.geom.L

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            return amplitude * np.sin(m * np.pi * x / L) * np.cos(n * theta + phase)
        return defect_func

    def double_c_defect(self, n1: int, n2: int, amplitude: float) -> callable:
        R, L = self.geom.R, self.geom.L
        m1 = max(1, n1 // 2)
        m2 = max(1, n2 // 2)

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            w1 = np.sin(m1 * np.pi * x / L) * np.cos(n1 * theta)
            w2 = np.sin(m2 * np.pi * L / L) * np.cos(n2 * theta)
            return amplitude * (w1 - 0.7 * w2)
        return defect_func

    def monte_carlo_multi_mode(self, n_modes: int = 10,
                               amplitude_ratio: float = 0.01,
                               distribution: str = "gaussian") -> callable:
        t = self.geom.t
        L = self.geom.L
        max_amp = amplitude_ratio * t

        modes = []
        for k in range(n_modes):
            m = self.rng.integers(1, 6)
            n = self.rng.integers(1, 12)
            phase = self.rng.uniform(0.0, 2.0 * np.pi)

            decay = 1.0 / np.sqrt(m ** 2 + n ** 2 + 1.0)
            if distribution == "gaussian":
                a = self.rng.normal(0.0, max_amp * decay)
            else:
                a = self.rng.uniform(-max_amp * decay, max_amp * decay)
            modes.append((m, n, a, phase))

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            w = np.zeros_like(theta, dtype=float)
            for m, n, a, ph in modes:
                w += a * np.sin(m * np.pi * x / L) * np.cos(n * theta + ph)
            return w
        return defect_func

    def ball_volume_defect(self, n_samples: int = 1000,
                           amplitude_ratio: float = 0.01) -> callable:
        t = self.geom.t
        max_amp = amplitude_ratio * t
        L = self.geom.L


        samples = []
        while len(samples) < n_samples:
            xyz = self.rng.normal(size=3)
            r = np.linalg.norm(xyz)
            if r > 1e-10 and r <= 1.0:
                samples.append(xyz / r)

        modes = []
        n_modes = min(n_samples, 20)
        for s in samples[:n_modes]:
            m = 1 + int(5.0 * abs(s[0]))
            n = 1 + int(10.0 * abs(s[1]))
            a = s[2] * max_amp / (m + n)
            ph = self.rng.uniform(0.0, 2.0 * np.pi)
            modes.append((m, n, a, ph))

        def defect_func(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            w = np.zeros_like(theta, dtype=float)
            for m, n, a, ph in modes:
                w += a * np.sin(m * np.pi * x / L) * np.cos(n * theta + ph)
            return w
        return defect_func

    def apply_defect_to_mesh(self, mesh, defect_func: callable) -> np.ndarray:
        nodes = mesh.nodes.copy()
        R = mesh.geom.R
        theta = np.arctan2(nodes[:, 1], nodes[:, 0])
        x = nodes[:, 2]
        w_bar = defect_func(theta, x)

        normal = mesh.geom.surface_normal(theta)
        new_nodes = nodes + normal * w_bar[:, np.newaxis]
        return new_nodes

    def defect_statistics(self, mesh, defect_func: callable) -> dict:
        theta = np.arctan2(mesh.nodes[:, 1], mesh.nodes[:, 0])
        x = mesh.nodes[:, 2]
        w = defect_func(theta, x)
        t = self.geom.t
        stats = {
            'max_defect': float(np.max(np.abs(w))),
            'rms_defect': float(np.sqrt(np.mean(w ** 2))),
            'defect_to_thickness': float(np.max(np.abs(w)) / t),
            'mean_defect': float(np.mean(w))
        }
        return stats
