# -*- coding: utf-8 -*-
"""
monte_carlo_sampler.py
蒙特卡洛随机采样模块
基于种子项目 567_hypersphere_positive_distance (超球面采样) 和 1005_randlc (线性同余随机数) 重构

本模块实现面向等离子体-壁相互作用模拟的蒙特卡洛采样器，包括:
    - 离子速度空间采样（超球面均匀分布）
    - 碰撞参数随机采样
    - 并行可重复的伪随机数序列（randlc）
"""

import numpy as np


class RandLC:
    """
    NAS Parallel Benchmark 线性同余随机数生成器（基于 randlc.m / randlc_jump.m）
    
    递推公式:
        X_{k+1} = a * X_k  mod 2^46
        u_k = X_k / 2^46
    
    其中 a = 5^13 = 1220703125。
    
    该生成器具有良好的统计性质，且支持直接跳跃到第 k 个元素，
    非常适合大规模并行蒙特卡洛模拟。
    """

    A = 1220703125.0
    R23 = 2.0**(-23)
    R46 = 2.0**(-46)
    T23 = 2.0**23
    T46 = 2.0**46

    def __init__(self, seed=314159265.0):
        """
        Parameters:
            seed: 初始种子（奇数）
        """
        self.x = float(seed)
        if self.x == 0.0:
            self.x = 314159265.0
        if self.x < 0.0:
            self.x = -self.x

    def _break_x(self, x):
        """将 X 分解为 2^23 * X1 + X2"""
        t1 = self.R23 * x
        x1 = int(t1)
        x2 = x - self.T23 * x1
        return x1, x2

    def next(self):
        """
        生成下一个随机数 [0, 1)
        
        Returns:
            u: 均匀随机数
        """
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
        """A 的高23位"""
        t1 = self.R23 * self.A
        return int(t1)

    @property
    def A_x2(self):
        """A 的低23位"""
        t1 = self.R23 * self.A
        return self.A - self.T23 * int(t1)

    def jump(self, k):
        """
        直接计算第 k 个随机数（基于 randlc_jump.m 的二进制幂算法）
        
        X_k = A^k * X_0 mod 2^46
        
        调用后生成器状态更新为 X_k。
        
        Parameters:
            k: 跳跃步数
        
        Returns:
            u_k: 第 k 个随机数
        """
        if k < 0:
            raise ValueError("k 必须 >= 0")
        if k == 0:
            return self.R46 * self.x

        # 二进制幂算法
        b = self.A
        b1 = int(self.R23 * b)
        b2 = b - self.T23 * b1
        x = self.x

        # 找到 m 使得 k < 2^m
        m = 1
        twom = 2
        while twom <= k:
            twom *= 2
            m += 1

        kk = k
        for _ in range(m):
            j = kk // 2

            # 若 k 为奇数，更新 X
            if 2 * j != kk:
                x1, x2 = self._break_x(x)
                t1 = b1 * x2 + b2 * x1
                t2 = int(self.R23 * t1)
                z = t1 - self.T23 * t2
                t3 = self.T23 * z + b2 * x2
                t4 = int(self.R46 * t3)
                x = t3 - self.T46 * t4

            # 更新 A -> A^2 mod 2^46
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
        """生成 n 个随机数序列"""
        return np.array([self.next() for _ in range(n)])


class MonteCarloSampler:
    """
    蒙特卡洛采样器（面向等离子体物理）
    """

    def __init__(self, seed=314159265):
        self.rng = RandLC(seed)
        self.np_rng = np.random.default_rng(seed % (2**32))

    def sample_hypersphere_positive(self, m):
        """
        在单位正超球面上采样一个点（基于 hypersphere_positive_sample.m）
        
        算法:
            1. 生成 m 维标准正态随机向量
            2. 归一化到单位长度
            3. 取绝对值确保所有分量非负
        
        应用于: 离子速度方向采样（假设速度分量均为正）
        
        Parameters:
            m: 维度
        
        Returns:
            x: (m,) 单位正超球面上的点
        """
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
        """
        估计单位正超球面上两点距离的统计量（基于 hypersphere_positive_distance_stats.m）
        
        应用于: 侵蚀原子之间距离分布的统计建模
        
        Parameters:
            m:         空间维度
            n_samples: 采样数
        
        Returns:
            mu:  平均距离
            var: 距离方差
        """
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
        """
        采样 Maxwellian 速度分布
        
        概率密度:
            f(v) = (m/(2*pi*k_B*T))^{3/2} * exp(-m*v^2/(2*k_B*T))
        
        通过 Box-Muller 变换生成:
            v_i = v_th * sqrt(-2*ln(u1)) * cos(2*pi*u2)
        
        Parameters:
            v_th:      热速度 [m/s]
            n_samples: 采样数
        
        Returns:
            v: (n_samples, 3) 速度向量
        """
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
        """
        按面积加权在表面三角网格上随机采样诞生位置
        
        应用于: 侵蚀原子的表面诞生位置
        
        Parameters:
            triangles: (n_tri, 3) 三角形顶点索引
            areas:     (n_tri,) 三角形面积
            n_samples: 采样数
        
        Returns:
            tri_indices: 选中的三角形索引
            bary_coords: 重心坐标 (n_samples, 3)
        """
        if len(areas) == 0:
            raise ValueError("areas 为空")

        total_area = np.sum(areas)
        if total_area < 1.0e-30:
            raise ValueError("总面积为零")

        probs = areas / total_area
        tri_indices = self.np_rng.choice(len(areas), size=n_samples, p=probs)

        # 在三角形内均匀采样（重心坐标）
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
        """
        采样碰撞参数（平均自由程模型）
        
        碰撞概率:
            P = 1 - exp(-n * sigma * L)
        
        Parameters:
            cross_section: 碰撞截面 [m^2]
            n_density:     靶密度 [m^-3]
            path_length:   路径长度 [m]
        
        Returns:
            collided: bool，是否发生碰撞
            mean_free_path: 平均自由程 [m]
        """
        if cross_section <= 0 or n_density <= 0:
            return False, np.inf

        mean_free_path = 1.0 / (n_density * cross_section)
        if mean_free_path <= 0:
            return False, np.inf

        prob = 1.0 - np.exp(-path_length / mean_free_path)
        collided = self.rng.next() < prob

        return collided, mean_free_path


def demo_mc():
    """演示蒙特卡洛采样"""
    mc = MonteCarloSampler(seed=42)

    # 超球面采样
    x = mc.sample_hypersphere_positive(5)
    print(f"5维正超球面采样: 模长 = {np.linalg.norm(x):.6f}, 最小分量 = {np.min(x):.6f}")

    # 距离统计
    mu, var = mc.sample_hypersphere_distance_stats(3, 1000)
    print(f"3维正超球面距离统计: mu={mu:.4f}, var={var:.6f}")

    # Maxwellian速度
    v_th = 1.0e5
    v = mc.sample_maxwellian_velocity(v_th, 1000)
    v_rms = np.sqrt(np.mean(v**2))
    print(f"Maxwellian采样: v_th={v_th:.3e}, 实测rms={v_rms:.3e}")

    # 碰撞采样
    collided, mfp = mc.sample_collision_parameter(1.0e-19, 1.0e19, 0.01)
    print(f"碰撞采样: 碰撞={collided}, 平均自由程={mfp:.3e} m")

    return mc


if __name__ == "__main__":
    demo_mc()
