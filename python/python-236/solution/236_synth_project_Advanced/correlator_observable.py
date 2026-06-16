"""
correlator_observable.py — 强子关联函数构造与物理观测量
========================================================
融合种子项目:
  [1086_Cuuung_LiH_Clifford_Reproduction] : 变分法 → 多算符基组优化
  [925_pwl_approx_1d] : 分段线性 → 源-汇插值
  [305_dist_plot]  : 距离函数 → 空间关联

物理背景:
  强子两体关联函数 (介子):
  C(t) = sum_{x} < O_H(t, x) O_H^dag(0, 0) >
       = sum_n |<0|O_H|n>|^2 exp(-E_n * t)
  其中 O_H 为强子内插场算符. 对介子:
  O_pi^+(x) = bar{d}(x) gamma_5 u(x)   (pi 介子)
  O_rho^+_k(x) = bar{d}(x) gamma_k u(x)  (rho 介子)

  传播子: S_q(x,y) = <q(x) bar{q}(y)> = D_q^{-1}(x,y)
  Wick 缩并: C(t) = -sum_x Tr[gamma_5 S_u(x,0) gamma_5 S_d(0,x)]  (pi)

核心公式:
  有效质量:  m_eff(t) = arccosh((C(t-1)+C(t+1))/(2*C(t)))
  变分法:  求解广义本征值问题 C^{(ij)}(t) v_j = lambda(t) C^{(ij)}(t_0) v_j
  光谱权重: Z_n = |<0|O|n>|^2 = 指数衰减振幅
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from lattice_geometry import LatticeGeometry


class HadronCorrelator:
    """强子两体关联函数.

    参数
    ----
    geo : LatticeGeometry
    hadron_type : str
        'pion' (赝标量), 'rho' (矢量), 'nucleon' (核子).
    """

    def __init__(self, geo: LatticeGeometry,
                 hadron_type: str = 'pion'):
        self.geo = geo
        self.hadron_type = hadron_type
        self.Lt = geo.Lt
        self.Ls = geo.Ls

        # 关联函数数据: C[t] for t = 0, ..., Lt-1
        self.C = np.zeros(self.Lt, dtype=np.float64)
        self.C_errors = np.zeros(self.Lt, dtype=np.float64)

        # 多算符基组 (源自 [1086] 的变分法思想)
        self.n_operators = 1
        self.C_matrix = None  # shape: (n_op, n_op, Lt)

    def generate_synthetic_correlator(self, masses: List[float],
                                      amplitudes: List[float],
                                      noise_level: float = 0.01,
                                      seed: int = 236):
        """生成合成关联函数数据 (用于可复现实验).

        C(t) = sum_n A_n * [exp(-m_n * t) + exp(-m_n * (Lt - t))]
               (周期性边界条件)

        参数
        ----
        masses : list of float
            态的质量 [m_0, m_1, ...].
        amplitudes : list of float
            对应振幅 [A_0, A_1, ...].
        noise_level : float
            高斯噪声相对水平.
        seed : int
            随机种子 (保证可重复).
        """
        if len(masses) != len(amplitudes):
            raise ValueError("质量和振幅数组长度必须一致")
        if len(masses) == 0:
            raise ValueError("至少需要一个态")
        for m in masses:
            if m <= 0:
                raise ValueError(f"质量 m={m} 必须为正")

        rng = np.random.default_rng(seed)
        for t in range(self.Lt):
            val = 0.0
            for m, A in zip(masses, amplitudes):
                # 周期性边界: C(t) = A*(exp(-m*t) + exp(-m*(Lt-t)))
                val += A * (np.exp(-m * t) + np.exp(-m * (self.Lt - t)))
            # 添加统计噪声
            noise = noise_level * abs(val) * rng.standard_normal()
            self.C[t] = val + noise
            self.C_errors[t] = max(noise_level * abs(val), 1e-15)

        self._true_masses = masses
        self._true_amplitudes = amplitudes

    def generate_multi_op_correlator(self, n_ops: int = 3,
                                     masses: Optional[List[float]] = None,
                                     overlap_matrix: Optional[np.ndarray] = None,
                                     noise_level: float = 0.01,
                                     seed: int = 236):
        """生成多算符基组的关联矩阵 (源自 [1086] 的变分法).

        C^{(ij)}(t) = sum_n Z_n^{(i)} Z_n^{(j)*} exp(-E_n t)

        其中 Z_n^{(i)} = <0|O_i|n> 为第 i 个算符到第 n 个态的重叠.

        参数
        ----
        n_ops : int
            算符基组大小.
        masses : list of float, optional
            态的质量.
        overlap_matrix : ndarray, optional
            重叠矩阵 Z[i, n], shape (n_ops, n_states).
        """
        n_states = max(n_ops, 2)
        if masses is None:
            masses = [0.3 + 0.2 * n for n in range(n_states)]

        self.n_operators = n_ops
        self.C_matrix = np.zeros((n_ops, n_ops, self.Lt))

        if overlap_matrix is None:
            # 默认重叠: 对角占优
            rng_gen = np.random.default_rng(seed)
            Z = rng_gen.standard_normal((n_ops, n_states)) * 0.3
            for i in range(min(n_ops, n_states)):
                Z[i, i] = 1.0 + 0.5 * rng_gen.standard_normal()
        else:
            Z = overlap_matrix

        rng = np.random.default_rng(seed + 1)
        for i in range(n_ops):
            for j in range(n_ops):
                for t in range(self.Lt):
                    val = 0.0
                    for n in range(n_states):
                        val += (Z[i, n] * Z[j, n]
                                * (np.exp(-masses[n] * t)
                                   + np.exp(-masses[n] * (self.Lt - t))))
                    noise = noise_level * abs(val) * rng.standard_normal()
                    self.C_matrix[i, j, t] = val + noise

        self._true_masses = masses

    # ------------------------------------------------------------------
    # 有效质量 (核心观测物理量)
    # ------------------------------------------------------------------
    def effective_mass(self, method: str = 'cosh') -> Tuple[np.ndarray, np.ndarray]:
        """计算有效质量 m_eff(t).

        方法 'cosh' (默认, 利用 PBC):
          C(t) ~ A * cosh(m*(t - Lt/2))
          m_eff = arccosh((C(t-1)+C(t+1))/(2*C(t)))

        方法 'log':
          m_eff(t) = ln(C(t)/C(t+1))

        方法 'second_diff':
          m_eff = arccosh(1 + Delta2 C / (2*C))
          其中 Delta2 C(t) = C(t+1) - 2*C(t) + C(t-1)

        返回
        ----
        t_arr : ndarray
        m_eff : ndarray (含 NaN 在边界)
        """
        t_arr = np.arange(self.Lt, dtype=np.float64)
        m_eff = np.full(self.Lt, np.nan)

        if method == 'cosh':
            for t in range(1, self.Lt - 1):
                denom = 2.0 * self.C[t]
                if abs(denom) < 1e-300:
                    continue
                arg = (self.C[t - 1] + self.C[t + 1]) / denom
                # arccosh 定义域: arg >= 1
                if arg >= 1.0:
                    m_eff[t] = np.arccosh(arg)
                else:
                    m_eff[t] = np.nan

        elif method == 'log':
            for t in range(self.Lt - 1):
                if self.C[t + 1] > 0 and self.C[t] > 0:
                    m_eff[t] = np.log(self.C[t] / self.C[t + 1])

        elif method == 'second_diff':
            for t in range(1, self.Lt - 1):
                if abs(self.C[t]) < 1e-300:
                    continue
                d2 = (self.C[t + 1] - 2.0 * self.C[t] + self.C[t - 1])
                arg = 1.0 + d2 / (2.0 * self.C[t])
                if arg >= 1.0:
                    m_eff[t] = np.arccosh(arg)
        else:
            raise ValueError(f"未知有效质量方法: {method}")

        return t_arr, m_eff

    # ------------------------------------------------------------------
    # 空间关联函数 (源自 [305_dist_plot] 的距离场)
    # ------------------------------------------------------------------
    def spatial_correlator(self, source: Optional[np.ndarray] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """计算空间关联函数 C(r) = sum_{|x|=r} C(t_fixed, x).

        源自 [305_dist_plot]: 使用格点距离函数将空间点按
        距源的距离分桶, 然后对每个壳层求和.

        参数
        ----
        source : ndarray, optional
            源点坐标 (默认为原点).

        返回
        ----
        r_values : ndarray
            距离值.
        C_r : ndarray
            空间关联函数值.
        """
        if source is None:
            source = np.zeros(self.geo.ndim, dtype=np.int64)

        coords = self.geo.all_coords()
        distances = np.array([self.geo.pbc_distance(source, c)
                              for c in coords])

        # 距离分桶
        r_max = min(self.Ls / 2.0, 3.0)
        n_bins = min(int(r_max * self.Ls), 20)
        r_edges = np.linspace(0, r_max, n_bins + 1)
        C_r = np.zeros(n_bins)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

        for b in range(n_bins):
            mask = (distances >= r_edges[b]) & (distances < r_edges[b + 1])
            count = np.sum(mask)
            if count > 0:
                C_r[b] = np.sum(distances[mask]) / count  # 简化

        return r_centers, C_r

    # ------------------------------------------------------------------
    # 变分法基组分析 (源自 [1086] 的变分优化)
    # ------------------------------------------------------------------
    def variational_analysis(self, t_0: int = 2
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """求解广义本征值问题 (GEVP) 提取能谱.

        C(t) v_n = lambda_n(t, t_0) C(t_0) v_n

        其中 lambda_n(t, t_0) ~ exp(-E_n * (t - t_0))
        有效质量: E_n(t) = -1/dt * ln(lambda_n(t+dt, t_0) / lambda_n(t, t_0))

        参数
        ----
        t_0 : int
            参考时间片.

        返回
        ----
        eigenvalues : ndarray, shape (n_ops,)
            广义本征值.
        energies : ndarray, shape (n_ops,)
            提取的能量.
        """
        if self.C_matrix is None:
            raise ValueError("需要先调用 generate_multi_op_correlator")

        n_ops = self.n_operators
        if t_0 >= self.Lt or t_0 < 1:
            raise ValueError(f"t_0={t_0} 超出范围 [1, {self.Lt})")

        # C(t_0+1) 和 C(t_0) 矩阵
        C_t1 = self.C_matrix[:, :, t_0 + 1] if t_0 + 1 < self.Lt else self.C_matrix[:, :, t_0]
        C_t0 = self.C_matrix[:, :, t_0]

        # 对称化
        C_t0 = 0.5 * (C_t0 + C_t0.T)
        C_t1 = 0.5 * (C_t1 + C_t1.T)

        # 正则化 C_t0 (加小量保证正定)
        reg = max(abs(np.trace(C_t0)) * 1e-10, 1e-15) * np.eye(n_ops)
        C_t0_reg = C_t0 + reg

        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(C_t1, C_t0_reg)
        except Exception:
            # 退化情况: 使用 numpy
            eigenvalues = np.linalg.eigvalsh(
                np.linalg.solve(C_t0_reg, C_t1))
            eigenvectors = None

        # 提取能量
        energies = np.full(n_ops, np.nan)
        for n in range(n_ops):
            if eigenvalues[n] > 0:
                energies[n] = -np.log(abs(eigenvalues[n]))

        return eigenvalues, energies

    def __repr__(self) -> str:
        return (f"HadronCorrelator(type={self.hadron_type}, "
                f"Lt={self.Lt}, n_ops={self.n_operators})")
