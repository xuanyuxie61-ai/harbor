# -*- coding: utf-8 -*-

import numpy as np


class RandLC:

    A = 1220703125.0
    R23 = 2.0**(-23)
    R46 = 2.0**(-46)
    T23 = 2.0**23
    T46 = 2.0**46

    def __init__(self, seed=314159265.0):
        self.x = float(seed)
        if self.x == 0.0:
            self.x = 314159265.0
        if self.x < 0.0:
            self.x = -self.x

    def _break_x(self, x):
        t1 = self.R23 * x
        x1 = int(t1)
        x2 = x - self.T23 * x1
        return x1, x2

    def next(self):
        x1, x2 = self._break_x(self.x)

        t1 = self.A_x1 * x2 + self.A_x2 * x1
        t2 = int(self.R23 * t1)
        z = t1 - self.T23 * t2

        t3 = self.T23 * z + self.A_x2 * x2
        t4 = int(self.R46 * t3)
        self.x = t3 - self.T46 * t4

        u = self.R46 * self.x
        return u

    @property
    def A_x1(self):
        t1 = self.R23 * self.A
        return int(t1)

    @property
    def A_x2(self):
        t1 = self.R23 * self.A
        return self.A - self.T23 * int(t1)

    def jump(self, k):
        if k < 0:
            raise ValueError("k 必须 >= 0")
        if k == 0:
            return self.R46 * self.x


        b = self.A
        b1 = int(self.R23 * b)
        b2 = b - self.T23 * b1
        x = self.x


        m = 1
        twom = 2
        while twom <= k:
            twom *= 2
            m += 1

        kk = k
        for _ in range(m):
            j = kk // 2


            if 2 * j != kk:
                x1, x2 = self._break_x(x)
                t1 = b1 * x2 + b2 * x1
                t2 = int(self.R23 * t1)
                z = t1 - self.T23 * t2
                t3 = self.T23 * z + b2 * x2
                t4 = int(self.R46 * t3)
                x = t3 - self.T46 * t4


            x1, x2 = self._break_x(b)
            t1 = b1 * x2 + b2 * x1
            t2 = int(self.R23 * t1)
            z = t1 - self.T23 * t2
            t3 = self.T23 * z + b2 * x2
            t4 = int(self.R46 * t3)
            b = t3 - self.T46 * t4

            b1 = int(self.R23 * b)
            b2 = b - self.T23 * b1

            kk = j

        self.x = x
        return self.R46 * x

    def generate_sequence(self, n):
        return np.array([self.next() for _ in range(n)])


class MonteCarloSampler:

    def __init__(self, seed=314159265):
        self.rng = RandLC(seed)
        self.np_rng = np.random.default_rng(seed % (2**32))

    def sample_hypersphere_positive(self, m):
        if m < 1:
            raise ValueError("维度 m 必须 >= 1")

        x = self.np_rng.standard_normal(m)
        norm = np.linalg.norm(x)
        if norm < 1.0e-20:
            x = np.ones(m) / np.sqrt(m)
        else:
            x = np.abs(x) / norm

        return x

    def sample_hypersphere_distance_stats(self, m, n_samples):
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            p = self.sample_hypersphere_positive(m)
            q = self.sample_hypersphere_positive(m)
            distances[i] = np.linalg.norm(p - q)

        mu = np.mean(distances)
        if n_samples > 1:
            var = np.sum((distances - mu)**2) / (n_samples - 1)
        else:
            var = 0.0

        return mu, var

    def sample_maxwellian_velocity(self, v_th, n_samples=1):
        if v_th <= 0:
            raise ValueError("v_th 必须为正")

        v = np.zeros((n_samples, 3))
        for i in range(n_samples):
            for dim in range(3):
                u1 = max(self.rng.next(), 1.0e-30)
                u2 = self.rng.next()
                v[i, dim] = v_th * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

        return v

    def sample_birth_location_on_surface(self, triangles, areas, n_samples=1):
        if len(areas) == 0:
            raise ValueError("areas 为空")

        total_area = np.sum(areas)
        if total_area < 1.0e-30:
            raise ValueError("总面积为零")

        probs = areas / total_area
        tri_indices = self.np_rng.choice(len(areas), size=n_samples, p=probs)


        bary_coords = np.zeros((n_samples, 3))
        for i in range(n_samples):
            u = self.rng.next()
            v = self.rng.next()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            bary_coords[i] = [1.0 - u - v, u, v]

        return tri_indices, bary_coords

    def sample_collision_parameter(self, cross_section, n_density, path_length):
        if cross_section <= 0 or n_density <= 0:
            return False, np.inf

        mean_free_path = 1.0 / (n_density * cross_section)
        if mean_free_path <= 0:
            return False, np.inf

        prob = 1.0 - np.exp(-path_length / mean_free_path)
        collided = self.rng.next() < prob

        return collided, mean_free_path


def demo_mc():
    mc = MonteCarloSampler(seed=42)


    x = mc.sample_hypersphere_positive(5)
    print(f"5维正超球面采样: 模长 = {np.linalg.norm(x):.6f}, 最小分量 = {np.min(x):.6f}")


    mu, var = mc.sample_hypersphere_distance_stats(3, 1000)
    print(f"3维正超球面距离统计: mu={mu:.4f}, var={var:.6f}")


    v_th = 1.0e5
    v = mc.sample_maxwellian_velocity(v_th, 1000)
    v_rms = np.sqrt(np.mean(v**2))
    print(f"Maxwellian采样: v_th={v_th:.3e}, 实测rms={v_rms:.3e}")


    collided, mfp = mc.sample_collision_parameter(1.0e-19, 1.0e19, 0.01)
    print(f"碰撞采样: 碰撞={collided}, 平均自由程={mfp:.3e} m")

    return mc


if __name__ == "__main__":
    demo_mc()
