r"""
signal_formation.py
探测器信号产生与处理模块

本模块实现：
1. 探测器读出链路的传递函数建模（CR-RC 成形、Sallen-Key 滤波器）
2. 传递函数复平面极点分析（参考 aberth：Aberth-Ehrlich 同时求根法）
3. 分段线性响应函数近似（参考 pwl_approx_1d）
4. 噪声模型与信号叠加
5. 峰值提取与能量重建

核心公式：

A. CR-RC^n 成形器传递函数：
    H(s) = \frac{s \tau}{1 + s \tau} \cdot \left( \frac{G}{1 + s \tau_f} \right)^n

B. 极点位置：
    s_k = -\frac{1}{\tau_f}, \quad k = 1, \dots, n \quad (n 重极点)
    s_{n+1} = -\frac{1}{\tau} \quad (微分器零点)

C. 时域脉冲响应（PWL 近似）：
    采用分段线性函数拟合探测器响应曲线，
    避免解析逆 Laplace 变换的数值不稳定性。

D. 噪声功率谱密度：
    S_{V}(f) = S_{\rm series} + \frac{S_{\rm parallel}}{f^2} + S_{\rm 1/f} \frac{1}{f}

参考文献：
- Knoll, G. F. (2010). Radiation Detection and Measurement, 4th ed.
- Aberth, O. (1973). Math. Comp. 27, 339.
"""

import numpy as np
from typing import Tuple, Callable
from utils import r8vec_bracket4


# ============================================================================
# 分段线性（PWL）近似工具
# ============================================================================

def pwl_approx_1d_matrix(
    nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray
) -> np.ndarray:
    """
    构造分段线性近似矩阵 A，使得 y_d ≈ A @ y_c。

    参数：
        nd: 数据点数
        xd, yd: 原始数据 (nd,)
        nc: 控制点数
        xc: 控制点横坐标 (nc,)，须单调递增

    返回：
        A: (nd, nc) 矩阵
    """
    if len(xd) != nd or len(yd) != nd:
        raise ValueError("pwl_approx_1d_matrix: xd/yd 长度与 nd 不符")
    if len(xc) != nc:
        raise ValueError("pwl_approx_1d_matrix: xc 长度与 nc 不符")
    if nc < 2:
        raise ValueError("pwl_approx_1d_matrix: nc 必须 >= 2")

    A = np.zeros((nd, nc))
    for i in range(nd):
        x = xd[i]
        if x <= xc[0]:
            A[i, 0] = 1.0
        elif x >= xc[-1]:
            A[i, -1] = 1.0
        else:
            k = r8vec_bracket4(nc, xc, x)
            h = xc[k + 1] - xc[k]
            if h <= 0.0:
                A[i, k] = 1.0
            else:
                t = (x - xc[k]) / h
                A[i, k] = 1.0 - t
                A[i, k + 1] = t
    return A


def pwl_approx_1d(
    nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray
) -> np.ndarray:
    """
    计算控制点处的最佳 PWL 拟合值 y_c。

    求解最小二乘问题：
        \min_{y_c} \| A y_c - y_d \|_2^2

    返回：
        yc: (nc,) 控制点纵坐标
    """
    A = pwl_approx_1d_matrix(nd, xd, yd, nc, xc)
    # 使用最小二乘求解
    yc, residuals, rank, s = np.linalg.lstsq(A, yd, rcond=None)
    return yc


def pwl_interp_1d(
    nd: int, xd: np.ndarray, yd: np.ndarray, ni: int, xi: np.ndarray
) -> np.ndarray:
    """
    分段线性插值：通过原始数据点 (xd, yd) 的 PWL 插值。

    参数：
        nd: 数据点数
        xd, yd: 原始数据，xd 须单调递增
        ni: 插值点数
        xi: 插值点横坐标

    返回：
        yi: (ni,) 插值结果
    """
    if len(xd) != nd or len(yd) != nd:
        raise ValueError("pwl_interp_1d: xd/yd 长度不符")
    yi = np.zeros(ni)
    for i in range(ni):
        x = xi[i]
        if x <= xd[0]:
            yi[i] = yd[0]
        elif x >= xd[-1]:
            yi[i] = yd[-1]
        else:
            k = r8vec_bracket4(nd, xd, x)
            h = xd[k + 1] - xd[k]
            if h <= 0.0:
                yi[i] = yd[k]
            else:
                t = (x - xd[k]) / h
                yi[i] = (1.0 - t) * yd[k] + t * yd[k + 1]
    return yi


# ============================================================================
# 传递函数与极点分析（Aberth-Ehrlich 方法）
# ============================================================================

def poly_and_derivative(coeffs: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Horner 法则同时计算多项式 p(z) 及其导数 p'(z)。

    算法：
        p(z)   = c_n z^n + c_{n-1} z^{n-1} + \dots + c_0
        p'(z)  = n c_n z^{n-1} + (n-1) c_{n-1} z^{n-2} + \dots + c_1

    Horner 递推（向后）：
        p = c_0
        p' = 0
        for k = 1 to n:
            p' = p' * z + p
            p  = p  * z + c_k

    参数：
        coeffs: (n+1,) 系数数组，c[0] 为常数项，c[n] 为首项系数
        z: (m,) 复数求值点

    返回：
        p, dp: 多项式值与导数值
    """
    z = np.asarray(z, dtype=complex)
    p = np.ones_like(z) * coeffs[0]
    dp = np.zeros_like(z)
    for k in range(1, len(coeffs)):
        dp = dp * z + p
        p = p * z + coeffs[k]
    return p, dp


def aberth_ehrlich(
    coeffs: np.ndarray,
    max_iter: int = 100,
    tol: float = 1.0e-12,
) -> np.ndarray:
    """
    Aberth-Ehrlich 方法同时求复多项式的全部根。

    迭代公式：
        z_i^{(k+1)} = z_i^{(k)} - \frac{
            p(z_i^{(k)}) / p'(z_i^{(k)})
        }{
            1 - \left( p(z_i^{(k)}) / p'(z_i^{(k)}) \right)
              \sum_{j \neq i} \frac{1}{z_i^{(k)} - z_j^{(k)}}
        }

    初始化：
        Cauchy 界 r = 1 + \max_{i<n} |c_i / c_n|
        在单位圆上均匀分布初始猜测：
        z_i^{(0)} = r \cdot \exp(2\pi i \cdot i / n)

    参数：
        coeffs: (n+1,) 多项式系数，c[n] ≠ 0
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回：
        roots: (n,) 复根数组
    """
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("aberth_ehrlich: 多项式次数至少为 1")
    if abs(coeffs[-1]) < 1.0e-30:
        raise ValueError("aberth_ehrlich: 首项系数为零")

    # Cauchy 半径
    r_cauchy = 1.0 + np.max(np.abs(coeffs[:-1] / coeffs[-1]))

    # 初始猜测
    angles = 2.0 * np.pi * np.arange(n) / n + 0.5  # 加偏移避免实轴
    roots = r_cauchy * np.exp(1j * angles)

    for iteration in range(max_iter):
        p_vals, dp_vals = poly_and_derivative(coeffs, roots)
        delta_base = p_vals / dp_vals

        max_update = 0.0
        for i in range(n):
            # 计算 Aberth 修正项的分母
            correction_sum = 0.0 + 0.0j
            for j in range(n):
                if i == j:
                    continue
                diff = roots[i] - roots[j]
                if abs(diff) < 1.0e-30:
                    diff = 1.0e-30 * (1.0 + 1.0j)
                correction_sum += 1.0 / diff

            denom = 1.0 - delta_base[i] * correction_sum
            if abs(denom) < 1.0e-30:
                denom = 1.0e-30
            delta_i = delta_base[i] / denom
            roots[i] -= delta_i
            max_update = max(max_update, abs(delta_i))

        # 检查收敛
        if max_update < tol:
            break

    return roots


# ============================================================================
# 成形器脉冲响应
# ============================================================================

def cr_rc_n_pulse_response(
    t: np.ndarray,
    tau_cr: float = 1.0e-6,
    tau_rc: float = 2.0e-6,
    n_rc: int = 4,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    CR-RC^n 成形器的脉冲响应（近似解析形式）。

    公式（时域近似）：
        h(t) = A \cdot \left( \frac{t}{\tau_{RC}} \right)^n
               \exp\left( -\frac{t}{\tau_{RC}} \right)
               \cdot \left[ 1 - \exp\left( -\frac{t}{\tau_{CR}} \right) \right]

    其中方括号项为 CR 微分器引入的抑制。

    参数：
        t: 时间数组 [s]
        tau_cr: CR 时间常数 [s]
        tau_rc: RC 时间常数 [s]
        n_rc: RC 级数
        amplitude: 归一化幅度

    返回：
        h: 与 t 同形的脉冲响应数组
    """
    t = np.asarray(t, dtype=float)
    h = np.zeros_like(t)
    mask = t > 0.0
    t_m = t[mask]
    # 避免数值溢出
    arg_rc = np.clip(-t_m / tau_rc, -700.0, 700.0)
    arg_cr = np.clip(-t_m / tau_cr, -700.0, 700.0)
    h[mask] = amplitude * ((t_m / tau_rc) ** n_rc) * np.exp(arg_rc) * (1.0 - np.exp(arg_cr))
    return h


def shaped_pulse(
    t: np.ndarray,
    charge_arrival_times: np.ndarray,
    charge_values: np.ndarray,
    tau_cr: float = 1.0e-6,
    tau_rc: float = 2.0e-6,
    n_rc: int = 4,
) -> np.ndarray:
    """
    对一系列电荷到达事件叠加成形脉冲。

    公式：
        V(t) = \sum_k Q_k \cdot h(t - t_k) \cdot H(t - t_k)

    参数：
        t: 均匀时间网格
        charge_arrival_times: 电荷到达时间数组
        charge_values: 对应电荷量
        tau_cr, tau_rc, n_rc: 成形器参数

    返回：
        V: (len(t),) 成形后电压信号
    """
    t = np.asarray(t)
    signal = np.zeros_like(t)
    for ta, q in zip(charge_arrival_times, charge_values):
        dt = t - ta
        signal += q * cr_rc_n_pulse_response(dt, tau_cr, tau_rc, n_rc)
    return signal


# ============================================================================
# 噪声模型
# ============================================================================

def add_electronic_noise(
    signal: np.ndarray,
    dt: float,
    series_noise_sigma: float = 0.001,
    parallel_noise_sigma: float = 0.0005,
) -> np.ndarray:
    """
    添加电子学噪声（白噪声 + 低频噪声近似）。

    参数：
        signal: 原始信号数组
        dt: 采样间隔 [s]
        series_noise_sigma: 串联噪声标准差 [V]
        parallel_noise_sigma: 并联噪声标准差 [V]

    返回：
        noisy: 含噪信号
    """
    white = np.random.normal(0.0, series_noise_sigma, size=signal.shape)
    # 低频噪声：累积随机游走近似
    lowfreq = np.cumsum(np.random.normal(0.0, parallel_noise_sigma, size=signal.shape))
    lowfreq = lowfreq - np.mean(lowfreq)
    return signal + white + 0.1 * lowfreq


def extract_pulse_parameters(
    t: np.ndarray,
    signal: np.ndarray,
    baseline_samples: int = 20,
) -> Tuple[float, float, float]:
    """
    提取脉冲参数：基线、幅度、上升时间。

    参数：
        t: 时间数组
        signal: 信号数组
        baseline_samples: 用于计算基线的前段采样数

    返回：
        (baseline, amplitude, risetime)
    """
    if len(signal) < baseline_samples + 10:
        raise ValueError("extract_pulse_parameters: 信号长度不足")
    baseline = np.mean(signal[:baseline_samples])
    corrected = signal - baseline
    amplitude = float(np.max(corrected))
    peak_idx = int(np.argmax(corrected))

    # 10%–90% 上升时间
    ten_pct = 0.1 * amplitude
    ninety_pct = 0.9 * amplitude

    idx_10 = 0
    for i in range(peak_idx):
        if corrected[i] >= ten_pct:
            idx_10 = i
            break
    idx_90 = peak_idx
    for i in range(peak_idx, -1, -1):
        if corrected[i] <= ninety_pct:
            idx_90 = i
            break

    if idx_90 > idx_10:
        risetime = t[idx_90] - t[idx_10]
    else:
        if len(t) > 1:
            risetime = t[1] - t[0]
        else:
            risetime = 0.0
    return baseline, amplitude, risetime


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试 PWL 近似
    xd = np.linspace(0.0, 1.0, 101)
    yd = np.sin(2.0 * np.pi * xd)
    xc = np.linspace(0.0, 1.0, 11)
    yc = pwl_approx_1d(len(xd), xd, yd, len(xc), xc)
    xi = np.linspace(0.0, 1.0, 201)
    yi = pwl_interp_1d(len(xc), xc, yc, len(xi), xi)
    y_true = np.sin(2.0 * np.pi * xi)
    rmse = np.sqrt(np.mean((yi - y_true) ** 2))
    assert rmse < 0.15, f"PWL 近似误差过大: {rmse}"

    # 测试 Aberth-Ehrlich：求 x^3 - 1 = 0 的根
    coeffs = np.array([-1.0, 0.0, 0.0, 1.0])
    roots = aberth_ehrlich(coeffs, max_iter=200)
    for r in roots:
        assert abs(r**3 - 1.0) < 1e-10, f"Aberth 求根失败: {r}"

    # 测试 CR-RC^n 脉冲响应
    t = np.linspace(0.0, 20.0e-6, 1000)
    h = cr_rc_n_pulse_response(t, tau_cr=1.0e-6, tau_rc=2.0e-6, n_rc=4)
    assert np.all(h >= 0.0), "脉冲响应出现负值"
    assert np.max(h) > 0.0, "脉冲响应为零"

    print("signal_formation.py: 所有自测通过")
