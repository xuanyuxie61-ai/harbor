# -*- coding: utf-8 -*-
"""
wavelet_analysis.py
鞘层波动小波分析模块
基于种子项目 1403_wavelet (Daubechies小波变换) 重构

本模块使用Daubechies小波对等离子体鞘层中的电势/密度波动进行多尺度分解，
识别不同频率成分的湍流结构与输运事件。
"""

import numpy as np


class WaveletAnalysis:
    """
    Daubechies 小波分析器
    
    基于 daub10_transform.m / daub_coefficients.m 的核心算法:
        - 使用 Daubechies 小波系数进行多级分解
        - 支持多种阶数（2, 4, 6, 8, 10, ..., 20）
    
    应用于:
        - 鞘层电势波动的多尺度分析
        - 识别_blob_输运事件的特征尺度
        - 湍流功率谱计算
    """

    # Daubechies 小波系数（基于 daub_coefficients.m，已乘以 sqrt(2)）
    DAUB_COEFFS = {
        2: np.array([1.0, 1.0]) / np.sqrt(2.0),
        4: np.array([0.4829629131445341, 0.8365163037378079,
                     0.2241438680420133, -0.1294095225512603]),
        6: np.array([0.3326705529500826, 0.8068915093110925,
                     0.4598775021184915, -0.1350110200102545,
                     -0.08544127388202666, 0.03522629188570953]),
        8: np.array([0.2303778133088965, 0.7148465705529156,
                     0.6308807679298589, -0.02798376941685985,
                     -0.1870348117190930, 0.03084138183556076,
                     0.03288301166688519, -0.01059740178506903]),
        10: np.array([0.1601023979741929, 0.6038292697971895,
                      0.7243085284377726, 0.1384281459013203,
                      -0.2422948870663823, -0.0322448695846381,
                      0.0775714938400459, -0.0062414902127983,
                      -0.0125807519990820, 0.0033357252854738]),
        12: np.array([0.1115407433501094, 0.4946238903984530,
                      0.7511339080210953, 0.3152503517091976,
                      -0.2262646939654398, -0.1297668675672619,
                      0.09750160558732304, 0.02752286553030572,
                      -0.03158203931748602, 0.0005538422011614961,
                      0.00477257510945510, -0.001073301085308479]),
    }

    def __init__(self, order=10):
        """
        Parameters:
            order: Daubechies小波阶数 (2, 4, 6, 8, 10, 12)
        """
        if order not in self.DAUB_COEFFS:
            raise ValueError(f"不支持的小波阶数 {order}，支持的阶数: {list(self.DAUB_COEFFS.keys())}")
        self.order = order
        self.c = self.DAUB_COEFFS[order]
        self.p = order - 1

    def _i4_wrap(self, ival, ilo, ihi):
        """循环索引包装（基于 i4_wrap.m）"""
        if ival < ilo:
            wide = ihi - ilo + 1
            return ihi - ((ilo - ival - 1) % wide)
        elif ival > ihi:
            wide = ihi - ilo + 1
            return ilo + ((ival - ilo) % wide)
        else:
            return ival

    def transform(self, signal):
        """
        离散小波变换（基于 daub10_transform.m）
        
        对信号进行多级分解，返回近似系数和细节系数。
        
        Parameters:
            signal: 输入信号，长度必须是2的幂且 >= 4
        
        Returns:
            coeffs: 小波系数数组（低频在前，高频在后）
        """
        x = np.asarray(signal, dtype=float).copy()
        n = len(x)

        if n < 4:
            raise ValueError("信号长度必须至少为 4")
        if (n & (n - 1)) != 0:
            # 补零到2的幂
            new_n = 1
            while new_n < n:
                new_n *= 2
            x = np.resize(x, new_n)
            x[n:] = 0.0
            n = new_n

        y = x.copy()
        m = n
        q = (self.p - 1) // 2

        while m >= 4:
            z = np.zeros(m)
            i = 0
            for j in range(0, m, 2):
                mh = m // 2
                for k in range(0, self.p, 2):
                    j0 = self._i4_wrap(j + k, 0, m - 1)
                    j1 = self._i4_wrap(j + k + 1, 0, m - 1)
                    if i < mh:
                        z[i] += self.c[k] * y[j0] + self.c[k+1] * y[j1]
                        z[i + mh] += (self.c[self.p - k] * y[j0] -
                                      self.c[self.p - k - 1] * y[j1])
                i += 1

            y[:m] = z
            m //= 2

        return y

    def inverse_transform(self, coeffs):
        """
        逆离散小波变换（基于 daub10_transform_inverse.m）
        
        从小波系数重建原始信号。
        """
        y = np.asarray(coeffs, dtype=float).copy()
        n = len(y)

        if n < 4 or (n & (n - 1)) != 0:
            raise ValueError("系数长度必须是2的幂且 >= 4")

        m = 4
        while m <= n:
            z = np.zeros(m)
            i = 0
            for j in range(0, m, 2):
                mh = m // 2
                for k in range(0, self.p, 2):
                    j0 = self._i4_wrap(j + k, 0, m - 1)
                    j1 = self._i4_wrap(j + k + 1, 0, m - 1)
                    if i < mh:
                        z[j0] += (self.c[k] * y[i] +
                                  self.c[self.p - k] * y[i + mh])
                        z[j1] += (self.c[k + 1] * y[i] -
                                  self.c[self.p - k - 1] * y[i + mh])
                i += 1

            y[:m] = z
            m *= 2

        return y

    def decompose_levels(self, signal):
        """
        多级分解，返回各层系数
        
        Returns:
            levels: dict {level: coefficients}
        """
        x = np.asarray(signal, dtype=float).copy()
        n = len(x)

        # 补零到2的幂
        if (n & (n - 1)) != 0:
            new_n = 1
            while new_n < n:
                new_n *= 2
            x = np.resize(x, new_n)
            x[n:] = 0.0
            n = new_n

        levels = {}
        y = x.copy()
        m = n
        level = 0
        q = (self.p - 1) // 2

        while m >= 4:
            z = np.zeros(m)
            i = 0
            for j in range(0, m, 2):
                mh = m // 2
                for k in range(0, self.p, 2):
                    j0 = self._i4_wrap(j + k, 0, m - 1)
                    j1 = self._i4_wrap(j + k + 1, 0, m - 1)
                    if i < mh:
                        z[i] += self.c[k] * y[j0] + self.c[k+1] * y[j1]
                        z[i + mh] += (self.c[self.p - k] * y[j0] -
                                      self.c[self.p - k - 1] * y[j1])
                i += 1

            levels[level] = z[mh:].copy()  # 细节系数
            y[:mh] = z[:mh]  # 保留近似系数
            m //= 2
            level += 1

        levels['approximation'] = y[:m].copy()
        return levels

    def power_spectrum(self, signal, sample_rate=1.0):
        """
        计算小波功率谱
        
        返回各尺度的能量分布:
            P_j = sum_k |d_{j,k}|^2
        
        Parameters:
            signal:      输入信号
            sample_rate: 采样率 [Hz]
        
        Returns:
            scales:      特征尺度
            power:       各尺度功率
            frequencies: 中心频率
        """
        levels = self.decompose_levels(signal)
        n_levels = len(levels) - 1

        scales = []
        power = []
        frequencies = []

        for level in range(n_levels):
            detail = levels[level]
            scales.append(2**level)
            power.append(np.sum(detail**2))
            # 中心频率近似: f_j = sample_rate / (2^{j+1})
            frequencies.append(sample_rate / (2**(level + 1)))

        return np.array(scales), np.array(power), np.array(frequencies)

    def denoise(self, signal, threshold_ratio=0.1):
        """
        小波阈值去噪
        
        软阈值:
            d' = sign(d) * max(|d| - threshold, 0)
        """
        coeffs = self.transform(signal)
        threshold = threshold_ratio * np.max(np.abs(coeffs))

        coeffs_denoised = np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
        return self.inverse_transform(coeffs_denoised)


def demo_wavelet():
    """演示小波分析"""
    # 合成测试信号：低频基频 + 中频湍流 + 高频噪声
    t = np.linspace(0, 1, 1024)
    f_slow = 5.0   # 慢变鞘层振荡
    f_turb = 50.0  # 湍流频率
    signal = (np.sin(2*np.pi*f_slow*t) +
              0.3 * np.sin(2*np.pi*f_turb*t) *
              (1.0 + 0.5*np.sin(2*np.pi*2*t)) +  # 振幅调制
              0.1 * np.random.randn(len(t)))

    wv = WaveletAnalysis(order=10)

    # 多级分解
    levels = wv.decompose_levels(signal)
    print("小波多级分解:")
    for key in sorted([k for k in levels.keys() if isinstance(k, int)]):
        detail = levels[key]
        print(f"  Level {key}: 细节系数能量 = {np.sum(detail**2):.4e}")

    # 功率谱
    scales, power, freqs = wv.power_spectrum(signal, sample_rate=1024.0)
    print("\n功率谱峰值频率:")
    if len(freqs) > 0:
        peak_idx = np.argmax(power)
        print(f"  主导频率 = {freqs[peak_idx]:.1f} Hz")

    # 去噪
    denoised = wv.denoise(signal, threshold_ratio=0.15)
    noise_reduction = 1.0 - np.std(denoised - np.sin(2*np.pi*f_slow*t)) / np.std(signal)
    print(f"\n去噪效果: 噪声降低比例 = {noise_reduction*100:.1f}%")

    return wv


if __name__ == "__main__":
    demo_wavelet()
