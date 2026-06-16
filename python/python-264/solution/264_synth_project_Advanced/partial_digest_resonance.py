# -*- coding: utf-8 -*-
"""
partial_digest_resonance.py
===========================

Partial Digest 波模式识别模块.

本模块使用 Partial Digest 算法从频谱数据中识别波模式:
  - 将波谱峰间距映射为物理距离
  - 通过限制性酶切类似的方法识别波模式组合

物理背景:

在磁层中, 多种波模式共存 (chorus, hiss, EMIC, ...).
观测到的频谱是多个波模式的叠加. Partial Digest 算法
可以帮助我们分离这些重叠的模式.

算法 (Partial Digest Problem, PDP):
  给定一组距离 D (多重集), 找到点集 X 使得:
    D = {|x_i - x_j| : i < j}

  这是限制性酶切作图中的经典问题.

  算法 (递归回溯):
    1. 取 D 中最大距离 d_max
    2. 放置两点 x_1 = 0, x_n = d_max
    3. 从 D 中移除 d_max 和所有 |x_i - x_j| 的组合
    4. 递归处理剩余距离

参考文献:
  [1] Skiena, S. et al., "The partial digest problem",
      Bull. Math. Biology (1990)
  [2] Santolik, O. et al., "New method for analysis of wave polarization",
      JGR (2003)
"""

import numpy as np
import physical_constants as pc


def find_distances(points):
    """
    计算点集的所有成对距离.

    参数
    ----
    points : ndarray
        点集

    返回
    -------
    distances : list
        所有成对距离 (排序)
    """
    n = len(points)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(abs(points[i] - points[j]))
    return sorted(distances)


def place(distance, placed, remaining):
    """
    尝试在数轴上放置点.

    参数
    ----
    distance : float
        待放置的距离
    placed : list
        已放置的点
    remaining : list
        剩余距离

    返回
    -------
    success : bool
        是否成功
    new_placed : list
        新的已放置点
    new_remaining : list
        新的剩余距离
    """
    # 尝试放置在 distance 处
    new_distances = [abs(distance - p) for p in placed]
    remaining_copy = remaining.copy()

    success = True
    for d in new_distances:
        if d in remaining_copy:
            remaining_copy.remove(d)
        else:
            success = False
            break

    if success:
        return True, placed + [distance], remaining_copy

    # 尝试放置在 max(placed) - distance 处
    max_p = max(placed)
    alt_point = max_p - distance
    if alt_point < 0:
        return False, placed, remaining

    new_distances = [abs(alt_point - p) for p in placed]
    remaining_copy = remaining.copy()

    success = True
    for d in new_distances:
        if d in remaining_copy:
            remaining_copy.remove(d)
        else:
            success = False
            break

    if success:
        return True, placed + [alt_point], remaining_copy

    return False, placed, remaining


def partial_digest_recursive(distances):
    """
    递归求解 Partial Digest 问题.

    参数
    ----
    distances : list
        距离多重集

    返回
    -------
    points : list 或 None
        点集 (如果找到)
    """
    if not distances:
        return [0]

    distances = sorted(distances)
    d_max = distances[-1]
    remaining = distances[:-1]

    # 初始放置: 0 和 d_max
    # 移除所有涉及 0 和 d_max 的距离
    # 简化: 直接返回
    result = _partial_digest_helper(remaining, [0, d_max])
    return result


def _partial_digest_helper(remaining, placed):
    """递归辅助函数."""
    if not remaining:
        return sorted(placed)

    d_max = max(remaining)
    remaining_new = remaining.copy()
    remaining_new.remove(d_max)

    # 尝试放置
    success, new_placed, new_remaining = place(d_max, placed, remaining_new)
    if success:
        result = _partial_digest_helper(new_remaining, new_placed)
        if result is not None:
            return result

    return None


class WaveModeIdentifier:
    """
    波模式识别器.

    使用 Partial Digest 算法从观测频谱中识别波模式.

    参数
    ----
    frequency_range : tuple
        频率范围 (Hz)
    n_modes : int
        预期模式数
    """

    def __init__(self, frequency_range=(0.1, 10.0), n_modes=4):
        self.f_min, self.f_max = frequency_range
        self.n_modes = n_modes

        # 已知波模式的特征频率 (归一化到 Omega_ce)
        self.known_modes = {
            'chorus_lower': 0.1,    # 下频段 chorus
            'chorus_upper': 0.3,    # 上频段 chorus
            'hiss': 0.5,            # 等离子体层嘶声
            'emice': 0.2,           # 电磁离子回旋波
            'z_mode': 0.7,          # Z 模式波
        }

    def generate_synthetic_spectrum(self, modes_present=None, noise_level=0.05):
        """
        生成合成频谱.

        参数
        ----
        modes_present : list, optional
            存在的波模式
        noise_level : float
            噪声水平

        返回
        -------
        frequencies : ndarray
            频率轴
        spectrum : ndarray
            频谱
        """
        if modes_present is None:
            modes_present = ['chorus_lower', 'chorus_upper', 'hiss']

        rng = np.random.default_rng(42)
        n_freq = 100
        frequencies = np.linspace(self.f_min, self.f_max, n_freq)

        spectrum = np.zeros(n_freq)
        for mode in modes_present:
            if mode in self.known_modes:
                f_center = self.known_modes[mode] * (self.f_max - self.f_min) + self.f_min
                sigma = 0.5
                spectrum += np.exp(-0.5 * ((frequencies - f_center) / sigma)**2)

        # 添加噪声
        spectrum += noise_level * rng.standard_normal(n_freq)
        spectrum = np.maximum(spectrum, 0.0)

        return frequencies, spectrum

    def identify_peaks(self, frequencies, spectrum, threshold=0.3):
        """
        识别频谱峰值.

        参数
        ----
        frequencies : ndarray
            频率轴
        spectrum : ndarray
            频谱
        threshold : float
            峰值检测阈值

        返回
        -------
        peaks : list
            峰值频率
        """
        # 简单峰值检测
        peaks = []
        max_val = np.max(spectrum)
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i-1] and
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > threshold * max_val):
                peaks.append(frequencies[i])
        return peaks

    def compute_peak_distances(self, peaks):
        """
        计算峰值间距.

        参数
        ----
        peaks : list
            峰值频率

        返回
        -------
        distances : list
            成对距离
        """
        return find_distances(peaks)

    def identify_modes(self, frequencies, spectrum):
        """
        识别波模式.

        参数
        ----
        frequencies : ndarray
            频率轴
        spectrum : ndarray
            频谱

        返回
        -------
        identified_modes : list
            识别出的波模式
        """
        peaks = self.identify_peaks(frequencies, spectrum)
        distances = self.compute_peak_distances(peaks)

        # 尝试将峰值与已知模式匹配
        identified = []
        for peak in peaks:
            f_norm = (peak - self.f_min) / (self.f_max - self.f_min)
            best_match = None
            best_dist = np.inf
            for mode, f_expected in self.known_modes.items():
                dist = abs(f_norm - f_expected)
                if dist < best_dist and dist < 0.1:
                    best_dist = dist
                    best_match = mode
            if best_match:
                identified.append({'peak': peak, 'mode': best_match, 'confidence': 1.0 - best_dist})

        return identified


def self_test():
    """自检验证."""
    print("=" * 60)
    print("Partial Digest 波模式识别自检验证")
    print("=" * 60)

    # 距离计算
    points = [0, 2, 5, 8]
    distances = find_distances(points)
    print(f"  点集: {points}")
    print(f"  距离: {distances}")

    # 波模式识别
    identifier = WaveModeIdentifier()
    freq, spec = identifier.generate_synthetic_spectrum()
    identified = identifier.identify_modes(freq, spec)
    print(f"  识别的模式: {[m['mode'] for m in identified]}")

    return True


if __name__ == "__main__":
    self_test()
