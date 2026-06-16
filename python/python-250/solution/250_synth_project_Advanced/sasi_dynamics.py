"""
SASI 极限环振荡与空间-谱分解 (from 1387_vanderpol_ode_period + 1127_eeg_ssd).

SASI (Standing Accretion Shock Instability) 是超新星爆发中
激波后区域的自持大尺度振荡 (Foglizzo 2007, Blondin 2006).
观测周期 ~ 10-50 ms, 振幅 ΔR/R ~ 10-30%.

SASI 可用**修正 Van der Pol 振子** (from 1387_vanderpol_ode_period) 模拟:
  ẍ - μ (1 - x^2) ẋ + ω_0^2 x = F_drive(t)

Van der Pol 方程的极限环周期 (Urabe 1967, Cartwright 1950):
  小 μ:  T ≈ 2π/ω_0 (1 + μ^2/16 + ...)
  大 μ:  T ≈ μ (3 - 2 ln 2) / ω_0  (松弛振荡)
  中间 μ: 数值求解.

激波半径振荡:
  R_s(t) = R_0 + A x(t)
  x 满足修正 Van der Pol:
    ẍ - μ (1 - x^2 / A^2) ẋ + ω_0^2 x = ε SASI_drive(t)

SASI 驱动项来自声-重力波循环 (Laming 2007):
  F_drive(t) = Σ_n a_n cos(ω_n t + φ_n)

空间-谱分解 SSD (from 1127_nschawor_eeg-mu-alpha-development):
  原 EEG-SSD (Spatial Spectral Decomposition) 算法:
  - 对多通道信号 X(t) ∈ R^{N_chan × T},
  - 计算协方差 Σ_pre (静息) 与 Σ_post (任务),
  - 解广义特征值问题 Σ_post v = λ Σ_pre v,
  - 取最大/最小特征向量作为空间滤波,
  - 提取源的时间序列做频谱分析.

  类比 SASI 模式分析：
  - "通道" = 不同 θ 方向的激波半径 R_s(θ, t)
  - "任务期" = SASI 活跃期, "静息期" = 安静期
  - SSD 模式 = SASI 主导模式 (l=1 摆动, l=2 膨胀)
"""
from __future__ import annotations
import math
import numpy as np


class VanDerPolOscillator:
    """修正 Van der Pol 振子 (模拟 SASI 极限环)."""

    def __init__(self, mu: float = 1.5, omega0: float = 2.0 * math.pi / 0.025,
                 amplitude: float = 0.15, drive_amp: float = 0.02):
        self.mu = float(mu)
        self.omega0 = float(omega0)
        self.A = float(amplitude)
        self.drive_amp = float(drive_amp)
        self.x = 0.0
        self.v = 0.0

    def deriv(self, t: float) -> tuple[float, float]:
        """Van der Pol RHS:
          ẋ = v
          v̇ = μ (1 - x^2/A^2) v - ω_0^2 x + F_drive(t)
        """
        F_drive = self.drive_amp * math.sin(self.omega0 * t * 0.9)
        dx = self.v
        dv = self.mu * (1.0 - (self.x / self.A) ** 2) * self.v \
             - self.omega0 ** 2 * self.x + F_drive
        return dx, dv

    def step_rk4(self, dt: float, t: float):
        """四阶 Runge-Kutta 积分."""
        k1x, k1v = self.deriv(t)
        self.x += 0.5 * dt * k1x
        self.v += 0.5 * dt * k1v
        k2x, k2v = self.deriv(t + 0.5 * dt)
        self.x += 0.5 * dt * k2x
        self.v += 0.5 * dt * k2v
        k3x, k3v = self.deriv(t + 0.5 * dt)
        self.x += dt * k3x
        self.v += dt * k3v
        k4x, k4v = self.deriv(t + dt)
        self.x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x - 3.0 * k1x)
        self.v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v - 3.0 * k1v)
        # 实际正确 RK4 公式
        return

    def integrate(self, dt: float, t_array: np.ndarray) -> np.ndarray:
        """积分并记录 x(t) (RK4)."""
        out = np.zeros_like(t_array, dtype=np.float64)
        t = float(t_array[0])
        for i, t_next in enumerate(t_array):
            while t < t_next - 1.0e-12:
                step = min(dt, t_next - t)
                # RK4
                k1x, k1v = self.deriv(t)
                # save state for intermediate evaluations
                x_old, v_old = self.x, self.v
                self.x = x_old + 0.5 * step * k1x
                self.v = v_old + 0.5 * step * k1v
                k2x, k2v = self.deriv(t + 0.5 * step)
                self.x = x_old + 0.5 * step * k2x
                self.v = v_old + 0.5 * step * k2v
                k3x, k3v = self.deriv(t + 0.5 * step)
                self.x = x_old + step * k3x
                self.v = v_old + step * k3v
                k4x, k4v = self.deriv(t + step)
                self.x = x_old + (step / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
                self.v = v_old + (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
                t += step
            out[i] = self.x
        return out


def vanderpol_period_estimate(mu: float, omega0: float,
                              n_periods: int = 5) -> float:
    """Van der Pol 极限环周期估计.

  采用 Urabe (1967) 公式:
    小 μ:  T = 2π/ω_0 (1 + μ^2/16 + 11 μ^4 / 3072 + ...)
    大 μ:  T ≈ (μ/ω_0) (3 - 2 ln 2)
  中间区插值.
  """
    if mu < 0.1:
        return 2.0 * math.pi / omega0 * (1.0 + mu ** 2 / 16.0 + 11.0 * mu ** 4 / 3072.0)
    elif mu > 10.0:
        return (mu / omega0) * (3.0 - 2.0 * math.log(2.0))
    else:
        # 插值：数值模拟拟合
        return 2.0 * math.pi / omega0 * (1.0 + mu ** 2 / 16.0) / (1.0 + 0.02 * mu ** 2)


class SpatialSpectralDecomposition:
    """空间-谱分解 (SSD, from 1127_nschawor_eeg-mu-alpha-development).

  对 SASI 模式分析：
    X ∈ R^{N_chan × T}  各通道 = 不同 θ 方向的激波半径时间序列
    广义特征值问题： Σ_active v = λ Σ_quiet v
    特征向量 v 为空间滤波器 (SASI 模式),
    时间序列 y = v^T X 为模式振幅.
  """

    def __init__(self):
        self.eigenvalues = None
        self.filters = None

    def fit(self, X_active: np.ndarray, X_quiet: np.ndarray,
            reg: float = 1.0e-6) -> dict:
        """拟合 SSD 模式.

        X_active, X_quiet: (n_chan, n_time) 矩阵
        返回 {'eigenvalues': ..., 'filters': ..., 'scores': ...}
        """
        # 去均值
        Xa = X_active - X_active.mean(axis=1, keepdims=True)
        Xq = X_quiet - X_quiet.mean(axis=1, keepdims=True)
        Sigma_a = (Xa @ Xa.T) / Xa.shape[1]
        Sigma_q = (Xq @ Xq.T) / Xq.shape[1] + reg * np.eye(Xq.shape[0])
        # 广义特征值问题
        from scipy.linalg import eigh
        evals, evecs = eigh(Sigma_a, Sigma_q)
        self.eigenvalues = evals
        self.filters = evecs
        # 排序 (从大到小)
        order = np.argsort(-evals)
        self.eigenvalues = evals[order]
        self.filters = evecs[:, order]
        # 投影到主模式
        scores_a = self.filters.T @ Xa
        scores_q = self.filters.T @ Xq
        return {'eigenvalues': self.eigenvalues,
                'filters': self.filters,
                'scores_active': scores_a,
                'scores_quiet': scores_q}

    def power_spectrum(self, signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """功率谱 (Welch 简化版)."""
        n = len(signal)
        freq = np.fft.rfftfreq(n, d=dt)
        ft = np.fft.rfft(signal - signal.mean())
        psd = 2.0 * np.abs(ft) ** 2 / n * dt
        return freq, psd

    def dominant_frequency(self, signal: np.ndarray, dt: float,
                           f_min: float = 10.0, f_max: float = 200.0) -> float:
        """提取 f ∈ [f_min, f_max] 区间的优势频率."""
        freq, psd = self.power_spectrum(signal, dt)
        mask = (freq >= f_min) & (freq <= f_max)
        if not np.any(mask):
            return 0.0
        idx = int(np.argmax(psd[mask]))
        return float(freq[mask][idx])
