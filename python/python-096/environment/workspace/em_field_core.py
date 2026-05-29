"""
em_field_core.py
================
电磁场核心计算模块：天线阵列辐射方向图与近场耦合

核心科学公式：
  1. 阵列方向图（Array Factor）：
       AF(\theta, \phi) = \sum_{n=0}^{N-1} w_n e^{j k_0 (x_n u + y_n v + z_n w)}
     其中 u = \sin\theta \cos\phi, v = \sin\theta \sin\phi, w = \cos\theta

  2. 单元方向图（Element Pattern）：
       对于半波偶极子：E_e(\theta) = \cos(\pi/2 \cos\theta) / \sin\theta

  3. 总方向图：
       F(\theta, \phi) = E_e(\theta) \cdot AF(\theta, \phi)

  4. 近场耦合（Mutual Impedance）：
       使用修正的感应电动势法（EMF），互阻抗近似为：
       Z_{ij} \approx 30 [Ci(k_0 d) - j Si(k_0 d)]
       其中 Ci, Si 为余弦/正弦积分。

  5. 波束赋形权重优化目标（凸松弛形式）：
       \min_{\mathbf{w}} \|\mathbf{w}\|_2^2 + \lambda \|\mathbf{A}_{SL}\mathbf{w} - \mathbf{b}_{SL}\|_2^2
       s.t. \mathbf{a}^H(\theta_0) \mathbf{w} = 1

物理常数：
  c = 2.99792458e8 m/s（光速）
  \mu_0 = 4\pi \times 10^{-7} H/m
  \varepsilon_0 = 8.854187817e-12 F/m
"""

import numpy as np
from typing import Tuple, Optional
from numerical_utils import safe_inverse_sqrt, rotation_matrix_y, rotation_matrix_z


C_LIGHT = 2.99792458e8
MU_0 = 4.0 * np.pi * 1e-7
EPS_0 = 8.854187817e-12
ETA_0 = np.sqrt(MU_0 / EPS_0)  # 自由空间波阻抗 ~377 Ohm


class ArrayFactorCalculator:
    """
    天线阵列方向图计算器。

    支持平面阵列、共形阵列（球面）以及带有随机位置扰动的实际阵列。
    """

    def __init__(self, element_positions: np.ndarray,
                 frequency_hz: float = 3.0e9,
                 element_weights: Optional[np.ndarray] = None):
        """
        参数：
            element_positions: (N, 3) 单元位置（米）
            frequency_hz: 工作频率（Hz）
            element_weights: (N,) 复数权重，None 时为均匀加权
        """
        self.positions = np.asarray(element_positions, dtype=float)
        self.n_elements = self.positions.shape[0]
        self.frequency = frequency_hz
        self.wavelength = C_LIGHT / frequency_hz
        self.k0 = 2.0 * np.pi / self.wavelength
        if element_weights is None:
            self.weights = np.ones(self.n_elements, dtype=complex) / self.n_elements
        else:
            self.weights = np.asarray(element_weights, dtype=complex)
            if self.weights.size != self.n_elements:
                raise ValueError("权重数量必须与单元数一致")

    def _direction_cosines(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算方向余弦 u, v, w。"""
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        w = np.cos(theta)
        return u, v, w

    def compute_array_factor(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        计算阵列方向图 AF(\theta, \phi)。

        公式：
          AF = \sum_{n=0}^{N-1} w_n \exp(j k_0 (x_n u + y_n v + z_n w))
        """
        # TODO: Hole 1 - 请根据阵列方向图的物理公式实现此处代码
        # 提示：需要计算方向余弦 u, v, w，然后利用波数 k0 和单元位置 positions
        # 计算相位项 phase，最后对加权指数项求和得到阵列因子 AF
        raise NotImplementedError("Hole 1: 阵列方向图因子 compute_array_factor 待实现")

    def compute_element_pattern_dipole(self, theta: np.ndarray) -> np.ndarray:
        """
        半波偶极子单元方向图。

        公式：
          E_e(\theta) = \cos(\frac{\pi}{2} \cos\theta) / \sin\theta

        边界处理：当 \theta -> 0 或 \pi 时，使用泰勒展开：
          E_e(\theta) \approx (\pi/4) (\theta - \pi/2)^2 在 \theta=\pi/2 附近
          实际上在 \theta=0 时极限为 0，我们用平滑截断处理。
        """
        theta = np.asarray(theta, dtype=float)
        sin_t = np.sin(theta)
        sin_t_safe = np.where(np.abs(sin_t) > 1e-6, sin_t, 1e-6)
        pat = np.cos(0.5 * np.pi * np.cos(theta)) / sin_t_safe
        # 在端点附近平滑
        pat = np.where(np.abs(sin_t) > 1e-6, pat, 0.0)
        return pat

    def compute_total_pattern(self, theta: np.ndarray, phi: np.ndarray,
                              normalize: bool = True) -> np.ndarray:
        """
        计算总方向图 = 单元方向图 * 阵列因子。

        参数：
            theta: 俯仰角（rad）
            phi:   方位角（rad）
            normalize: 是否归一化到主瓣峰值 0 dB
        """
        af = self.compute_array_factor(theta, phi)
        ep = self.compute_element_pattern_dipole(theta)
        pattern = np.abs(af * ep)
        if normalize and np.max(pattern) > 0:
            pattern /= np.max(pattern)
        return pattern

    def compute_total_pattern_db(self, theta: np.ndarray, phi: np.ndarray,
                                 normalize: bool = True,
                                 floor_db: float = -80.0) -> np.ndarray:
        """计算总方向图（dB）。"""
        pat = self.compute_total_pattern(theta, phi, normalize)
        pat_db = 20.0 * np.log10(np.maximum(pat, 10.0 ** (floor_db / 20.0)))
        return pat_db

    def apply_steering(self, theta_s: float, phi_s: float = 0.0):
        """
        应用波束指向 (theta_s, phi_s) 的相移权重。

        公式：
          w_n = \exp(-j k_0 (x_n u_s + y_n v_s + z_n w_s))
          其中 (u_s, v_s, w_s) = (\sin\theta_s \cos\phi_s, \sin\theta_s \sin\phi_s, \cos\theta_s)
        """
        u_s = np.sin(theta_s) * np.cos(phi_s)
        v_s = np.sin(theta_s) * np.sin(phi_s)
        w_s = np.cos(theta_s)
        phase = -self.k0 * (self.positions[:, 0] * u_s
                          + self.positions[:, 1] * v_s
                          + self.positions[:, 2] * w_s)
        self.weights = np.exp(1j * phase)
        # 归一化
        self.weights /= np.linalg.norm(self.weights)

    def apply_chebyshev_weights(self, sidelobe_db: float = -30.0):
        """
        应用 Chebyshev 加权以控制旁瓣电平。

        数学背景（Dolph-Chebyshev）：
          对于均匀线阵，Chebyshev 多项式 T_{N-1}(x) 的等旁瓣特性
          可通过多项式求根映射到阵列权重。

          这里使用简化版：通过迭代调整权重的幅度来逼近等旁瓣分布。
        """
        # 简化实现：使用余弦幅度锥削逼近
        n = self.n_elements
        idx = np.arange(n)
        alpha = np.arccosh(10.0 ** (-sidelobe_db / 20.0))
        # Chebyshev 近似幅度分布
        x = (idx - (n - 1) / 2.0) / ((n - 1) / 2.0)
        amplitude = np.cosh(alpha * np.sqrt(np.maximum(1.0 - x ** 2, 0.0)))
        amplitude = np.maximum(amplitude, 1e-6)
        self.weights = amplitude * np.exp(1j * np.angle(self.weights))
        self.weights /= np.linalg.norm(self.weights)

    def directivity(self, theta_grid: int = 181, phi_grid: int = 360) -> float:
        """
        计算阵列方向性系数 D。

        公式：
          D = \frac{4\pi |F_{max}|^2}{\int_0^{2\pi} \int_0^{\pi} |F(\theta,\phi)|^2 \sin\theta d\theta d\phi}

        使用高斯-勒让德求积（简化版：梯形法则）。
        """
        theta = np.linspace(0.0, np.pi, theta_grid)
        phi = np.linspace(0.0, 2.0 * np.pi, phi_grid)
        theta_m, phi_m = np.meshgrid(theta, phi, indexing='ij')
        pat = self.compute_total_pattern(theta_m.ravel(), phi_m.ravel(), normalize=False)
        pat_sq = np.abs(pat) ** 2
        # 数值积分（梯形法则）
        dtheta = np.pi / (theta_grid - 1)
        dphi = 2.0 * np.pi / (phi_grid - 1)
        integrand = pat_sq.reshape(theta_grid, phi_grid) * np.sin(theta_m)
        total = np.sum(integrand) * dtheta * dphi
        p_max = np.max(pat_sq)
        D = 4.0 * np.pi * p_max / max(total, 1e-18)
        return float(D)


class MutualCouplingMatrix:
    """
    天线单元互耦矩阵计算。

    物理模型：
      对于半波偶极子，互阻抗的感应电动势法近似：

        Z_{ij} = R_{ij} + j X_{ij}

      其中：
        R_{ij} = 30 [ 2 Ci(k_0 d) - Ci(k_0(\sqrt{d^2+l^2}+l)) - Ci(k_0(\sqrt{d^2+l^2}-l)) ]
        X_{ij} = -30 [ 2 Si(k_0 d) - Si(k_0(\sqrt{d^2+l^2}+l)) - Si(k_0(\sqrt{d^2+l^2}-l)) ]

      l = \lambda/2 为偶极子半长，d 为单元间距。

      自阻抗（孤立半波偶极子）：
        Z_{11} = 73.1 + j 42.5 \, \Omega
    """

    def __init__(self, element_positions: np.ndarray, frequency_hz: float = 3.0e9):
        self.positions = np.asarray(element_positions, dtype=float)
        self.n_elements = self.positions.shape[0]
        self.frequency = frequency_hz
        self.wavelength = C_LIGHT / frequency_hz
        self.k0 = 2.0 * np.pi / self.wavelength
        self.half_length = self.wavelength / 4.0  # 半波偶极子半长

    def _ci_si_approx(self, x: float) -> Tuple[float, float]:
        """
        余弦积分 Ci(x) 和正弦积分 Si(x) 的近似。

        对于小 x：
          Ci(x) \approx \gamma + \ln(x) + \sum_{n=1}^{\infty} (-1)^n x^{2n} / (2n \cdot (2n)!)
          Si(x) \approx \sum_{n=0}^{\infty} (-1)^n x^{2n+1} / ((2n+1) \cdot (2n+1)!)

        对于大 x：
          Ci(x) \approx \sin(x)/x - \cos(x)/x^2
          Si(x) \approx \pi/2 - \cos(x)/x - \sin(x)/x^2
        """
        gamma = 0.5772156649015329
        x = float(x)
        if x < 1e-12:
            return -1e6, 0.0  # Ci(0) -> -inf
        if x < 2.0:
            # 级数展开
            ci = gamma + np.log(x)
            si = 0.0
            term_ci = 1.0
            term_si = x
            for n in range(1, 15):
                term_ci *= -x * x / ((2 * n - 1) * (2 * n))
                term_si *= -x * x / ((2 * n) * (2 * n + 1))
                ci += term_ci / (2 * n)
                si += term_si / (2 * n + 1)
            return ci, si
        else:
            # 渐近展开
            ci = np.sin(x) / x - np.cos(x) / (x * x)
            si = 0.5 * np.pi - np.cos(x) / x - np.sin(x) / (x * x)
            return ci, si

    def compute_mutual_impedance(self, i: int, j: int) -> complex:
        """计算第 i 和第 j 个单元之间的互阻抗。"""
        if i == j:
            return 73.1 + 42.5j
        d_vec = self.positions[i, :] - self.positions[j, :]
        d = np.linalg.norm(d_vec)
        d = max(d, 1e-6)
        l = self.half_length
        k0 = self.k0

        ci_d, si_d = self._ci_si_approx(k0 * d)
        arg1 = k0 * (np.sqrt(d ** 2 + l ** 2) + l)
        arg2 = k0 * (np.sqrt(d ** 2 + l ** 2) - l)
        ci1, si1 = self._ci_si_approx(arg1)
        ci2, si2 = self._ci_si_approx(arg2)

        R = 30.0 * (2.0 * ci_d - ci1 - ci2)
        X = -30.0 * (2.0 * si_d - si1 - si2)
        return R + 1j * X

    def build_impedance_matrix(self) -> np.ndarray:
        """构建完整的 N x N 阻抗矩阵。"""
        Z = np.zeros((self.n_elements, self.n_elements), dtype=complex)
        for i in range(self.n_elements):
            for j in range(i, self.n_elements):
                z_ij = self.compute_mutual_impedance(i, j)
                Z[i, j] = z_ij
                if i != j:
                    Z[j, i] = z_ij
        return Z

    def active_reflection_coefficient(self, port_idx: int,
                                      port_voltages: np.ndarray) -> complex:
        """
        计算第 port_idx 个端口的有效反射系数。

        公式：
          \Gamma_{active} = \frac{Z_{in} - Z_0}{Z_{in} + Z_0}
          Z_{in} = Z_{ii} + \sum_{j \neq i} \frac{I_j}{I_i} Z_{ij}
        """
        Z = self.build_impedance_matrix()
        I = np.asarray(port_voltages, dtype=complex)
        I_i = I[port_idx]
        if abs(I_i) < 1e-18:
            return 0.0
        Z_in = Z[port_idx, port_idx]
        for j in range(self.n_elements):
            if j != port_idx:
                Z_in += (I[j] / I_i) * Z[port_idx, j]
        Z0 = 50.0
        gamma = (Z_in - Z0) / (Z_in + Z0)
        return gamma


def near_field_e_field(positions: np.ndarray,
                       currents: np.ndarray,
                       observation_points: np.ndarray,
                       frequency_hz: float = 3.0e9) -> np.ndarray:
    """
    计算近区电场（基于自由空间格林函数）。

    公式：
      \mathbf{E}(\mathbf{r}) = -j \omega \mu_0 \sum_n I_n G(\mathbf{r}, \mathbf{r}_n)

      G(\mathbf{r}, \mathbf{r}') = \frac{e^{-j k_0 R}}{4\pi R}
      R = |\mathbf{r} - \mathbf{r}'|

      近场修正包含 1/R, 1/R^2, 1/R^3 项：
      \mathbf{E} = \frac{\eta_0 k_0}{4\pi} \sum_n I_n \mathbf{l}_n
        \left[ \frac{j}{k_0 R} + \frac{1}{(k_0 R)^2} - \frac{j}{(k_0 R)^3} \right] e^{-j k_0 R} \sin\theta_n
    """
    k0 = 2.0 * np.pi * frequency_hz / C_LIGHT
    eta0 = ETA_0
    N_obs = observation_points.shape[0]
    E = np.zeros((N_obs, 3), dtype=complex)

    for n in range(positions.shape[0]):
        R_vec = observation_points - positions[n, :]
        R = np.linalg.norm(R_vec, axis=1)
        R_safe = np.maximum(R, 1e-6)
        kR = k0 * R_safe
        # 远场近似为主 + 近场修正项
        phase = np.exp(-1j * kR) / (4.0 * np.pi * R_safe)
        # 标量势贡献（简化）
        E_scalar = -1j * eta0 * k0 * currents[n] * phase
        # 分配到各观测点（假设单元为 z 向偶极子）
        # 投影到径向
        cos_theta = R_vec[:, 2] / R_safe
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0))
        E_amp = E_scalar * sin_theta
        # 方向：垂直于 R 且在 z-R 平面内
        # 简化：只保留 z 分量（对于 z 向偶极子）
        E[:, 2] += E_amp

    return E
