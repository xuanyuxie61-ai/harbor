"""
noise_model.py — 探测器噪声模型与击中模拟
==========================================

融合种子项目:
    [581_image_noise]: 椒盐噪声 + 均匀噪声模型
    → 映射为: 探测器死道 (椒盐) + 电子学噪声 (均匀)

物理模型:
    探测器噪声来源:
    1. 死道/热道 (椒盐噪声): 以概率 p_dead 将通道输出设为 0 或最大值
    2. 电子学噪声 (均匀噪声): 以概率 p_noise 替换为均匀随机值
    3. 高斯测量误差: σ_r, σ_z 由探测器空间分辨率决定

    击中效率:
        ε_hit = 1 - p_dead - p_noise
    假阳性率 (鬼击中):
        r_fake = f(occupancy, threshold)
"""

import math
import random
from typing import List, Tuple, Optional


class DetectorNoiseModel:
    """
    探测器噪声模型

    基于 [581_image_noise] 的两种噪声模型:

    [椒盐噪声 → 死道/热道]
    原算法: 对每个像素，以 level/2 概率设为 0 (黑点)，
            以 level/2 概率设为 255 (白点)
    映射:  对每个读出通道，以 p_dead/2 概率标记为死道 (无信号)，
           以 p_dead/2 概率标记为热道 (持续触发)

    [均匀噪声 → 电子学噪声]
    原算法: 以 level 概率替换为 randi([0, 255])
    映射:  以 p_noise 概率将测量值替换为均匀随机偏移

    Parameters
    ----------
    dead_channel_prob : float
        死道概率 (椒盐噪声级别)
    hot_channel_prob : float
        热道概率
    noise_prob : float
        均匀噪声概率
    noise_amplitude_r : float
        径向噪声幅度 [mm]
    noise_amplitude_z : float
        纵向噪声幅度 [mm]
    hit_efficiency : float
        探测效率 (0-1)
    fake_hit_rate : float
        假阳性率 (鬼击中/事件)
    seed : int or None
        随机种子
    """

    def __init__(self, dead_channel_prob=0.001, hot_channel_prob=0.0005,
                 noise_prob=0.01, noise_amplitude_r=0.05,
                 noise_amplitude_z=0.1, hit_efficiency=0.97,
                 fake_hit_rate=0.5, seed=None):
        self.p_dead = dead_channel_prob
        self.p_hot = hot_channel_prob
        self.p_noise = noise_prob
        self.noise_r = noise_amplitude_r
        self.noise_z = noise_amplitude_z
        self.epsilon = hit_efficiency
        self.fake_rate = fake_hit_rate

        # 初始化随机状态
        self._rng = random.Random(seed)
        self._gauss_cache = None

    def _gauss(self, sigma=1.0):
        """生成高斯随机数 (Box-Muller 变换)"""
        u1 = self._rng.random()
        u2 = self._rng.random()
        while u1 < 1e-15:
            u1 = self._rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return z * sigma

    # ============================================================
    # [581_image_noise] 椒盐噪声 → 死道/热道模拟
    # ============================================================
    def apply_salt_and_pepper(self, hit_value, channel_range=(0.0, 1.0)):
        """
        基于 [581] 的椒盐噪声模型

        原算法 (gray_salt_and_pepper):
            for each pixel:
                r = rand()
                if r < level/2: pixel = 0      # 椒 (黑色)
                elif r < level: pixel = 255     # 盐 (白色)

        映射到探测器:
            for each channel:
                r = rand()
                if r < p_dead/2: channel = dead    (无信号输出)
                elif r < p_dead: channel = hot     (持续触发)

        Parameters
        ----------
        hit_value : float
            原始击中信号值
        channel_range : tuple
            信号范围 (min, max)

        Returns
        -------
        tuple : (modified_value, channel_status)
            channel_status: 'normal', 'dead', 'hot'
        """
        r = self._rng.random()
        level = self.p_dead  # 总噪声级别

        if r < level / 2.0:
            # 椒: 死道 (信号归零)
            return channel_range[0], 'dead'
        elif r < level:
            # 盐: 热道 (信号饱和)
            return channel_range[1], 'hot'
        else:
            return hit_value, 'normal'

    # ============================================================
    # [581_image_noise] 均匀噪声 → 电子学噪声
    # ============================================================
    def apply_uniform_noise(self, measurement, sigma_r, sigma_z):
        """
        基于 [581] 的均匀噪声模型

        原算法 (gray_uniform_noise):
            for each pixel:
                if rand() < level:
                    pixel = randi([0, 255])

        映射到探测器:
            for each hit measurement:
                if rand() < p_noise:
                    r_meas += uniform(-noise_r, +noise_r)
                    z_meas += uniform(-noise_z, +noise_z)

        Parameters
        ----------
        measurement : tuple
            (r_true, z_true) 真实击中位置 [mm]
        sigma_r, sigma_z : float
            探测器分辨率 [mm]

        Returns
        -------
        tuple : (r_meas, z_meas, is_noisy)
        """
        r_true, z_true = measurement

        r_noise = self._rng.random()
        if r_noise < self.p_noise:
            # 添加均匀噪声 (与 [581] 一致)
            dr = self._rng.uniform(-self.noise_r, self.noise_r)
            dz = self._rng.uniform(-self.noise_z, self.noise_z)
            return (r_true + dr, z_true + dz, True)

        return (r_true, z_true, False)

    # ============================================================
    # 综合击中模拟
    # ============================================================
    def simulate_hit(self, true_r, true_z, sigma_r, sigma_z):
        """
        模拟完整的探测器击中过程

        流程:
        1. 检查探测效率 (是否丢失击中)
        2. 添加高斯测量误差
        3. 应用椒盐噪声 (死道/热道)
        4. 应用均匀噪声 (电子学噪声)

        Parameters
        ----------
        true_r, true_z : float
            真实击中位置 [mm]
        sigma_r, sigma_z : float
            探测器空间分辨率 [mm]

        Returns
        -------
        dict or None
            击中信息字典，如果击中丢失返回 None
            {
                'r_meas': float,    # 测量径向位置
                'z_meas': float,    # 测量纵向位置
                'r_true': float,    # 真实径向位置
                'z_true': float,    # 真实纵向位置
                'is_fake': bool,    # 是否为假击中
                'is_noisy': bool,   # 是否含噪声
                'channel_status': str,  # 通道状态
            }
        """
        # Step 1: 探测效率检查
        if self._rng.random() > self.epsilon:
            return None  # 击中丢失

        # Step 2: 高斯测量误差
        r_meas = true_r + self._gauss(sigma_r)
        z_meas = true_z + self._gauss(sigma_z)

        # Step 3: 椒盐噪声
        r_val, status_r = self.apply_salt_and_pepper(r_meas)
        if status_r == 'dead':
            return None  # 死道导致击中丢失
        z_val, status_z = self.apply_salt_and_pepper(z_meas)
        if status_z == 'dead':
            return None

        # Step 4: 均匀噪声
        (r_final, z_final, is_noisy) = self.apply_uniform_noise(
            (r_val, z_val), sigma_r, sigma_z)

        channel_status = 'normal'
        if status_r == 'hot' or status_z == 'hot':
            channel_status = 'hot'

        return {
            'r_meas': r_final,
            'z_meas': z_final,
            'r_true': true_r,
            'z_true': true_z,
            'is_fake': False,
            'is_noisy': is_noisy or (channel_status != 'normal'),
            'channel_status': channel_status,
            'residual_r': r_final - true_r,
            'residual_z': z_final - true_z,
        }

    def generate_fake_hits(self, layer_radius, half_length, n_fake):
        """
        生成假阳性击中 (鬼击中/组合背景)

        Parameters
        ----------
        layer_radius : float
            层半径 [mm]
        half_length : float
            层半长 [mm]
        n_fake : int
            假击中数量

        Returns
        -------
        list of dict : 假击中列表
        """
        fake_hits = []
        for _ in range(n_fake):
            phi = self._rng.uniform(0, 2.0 * math.pi)
            z = self._rng.uniform(-half_length, half_length)
            r_meas = layer_radius + self._gauss(0.1)

            fake_hits.append({
                'r_meas': r_meas,
                'z_meas': z,
                'phi': phi,
                'r_true': layer_radius,
                'z_true': z,
                'is_fake': True,
                'is_noisy': True,
                'channel_status': 'normal',
                'residual_r': r_meas - layer_radius,
                'residual_z': 0.0,
            })

        return fake_hits

    def summary(self):
        """噪声模型摘要"""
        return (f"NoiseModel: p_dead={self.p_dead:.4f}, p_hot={self.p_hot:.4f}, "
                f"p_noise={self.p_noise:.3f}, ε={self.epsilon:.3f}, "
                f"fake_rate={self.fake_rate:.1f}/event")
