
import numpy as np
from typing import Tuple, Callable
from utils import r8vec_bracket4






def pwl_approx_1d_matrix(
    nd: int, xd: np.ndarray, yd: np.ndarray, nc: int, xc: np.ndarray
) -> np.ndarray:
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
    A = pwl_approx_1d_matrix(nd, xd, yd, nc, xc)

    yc, residuals, rank, s = np.linalg.lstsq(A, yd, rcond=None)
    return yc


def pwl_interp_1d(
    nd: int, xd: np.ndarray, yd: np.ndarray, ni: int, xi: np.ndarray
) -> np.ndarray:
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






def poly_and_derivative(coeffs: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    n = len(coeffs) - 1
    if n < 1:
        raise ValueError("aberth_ehrlich: 多项式次数至少为 1")
    if abs(coeffs[-1]) < 1.0e-30:
        raise ValueError("aberth_ehrlich: 首项系数为零")


    r_cauchy = 1.0 + np.max(np.abs(coeffs[:-1] / coeffs[-1]))


    angles = 2.0 * np.pi * np.arange(n) / n + 0.5
    roots = r_cauchy * np.exp(1j * angles)

    for iteration in range(max_iter):
        p_vals, dp_vals = poly_and_derivative(coeffs, roots)
        delta_base = p_vals / dp_vals

        max_update = 0.0
        for i in range(n):

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


        if max_update < tol:
            break

    return roots






def cr_rc_n_pulse_response(
    t: np.ndarray,
    tau_cr: float = 1.0e-6,
    tau_rc: float = 2.0e-6,
    n_rc: int = 4,
    amplitude: float = 1.0,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    h = np.zeros_like(t)
    mask = t > 0.0
    t_m = t[mask]

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
    t = np.asarray(t)
    signal = np.zeros_like(t)
    for ta, q in zip(charge_arrival_times, charge_values):
        dt = t - ta
        signal += q * cr_rc_n_pulse_response(dt, tau_cr, tau_rc, n_rc)
    return signal






def add_electronic_noise(
    signal: np.ndarray,
    dt: float,
    series_noise_sigma: float = 0.001,
    parallel_noise_sigma: float = 0.0005,
) -> np.ndarray:
    white = np.random.normal(0.0, series_noise_sigma, size=signal.shape)

    lowfreq = np.cumsum(np.random.normal(0.0, parallel_noise_sigma, size=signal.shape))
    lowfreq = lowfreq - np.mean(lowfreq)
    return signal + white + 0.1 * lowfreq


def extract_pulse_parameters(
    t: np.ndarray,
    signal: np.ndarray,
    baseline_samples: int = 20,
) -> Tuple[float, float, float]:
    if len(signal) < baseline_samples + 10:
        raise ValueError("extract_pulse_parameters: 信号长度不足")
    baseline = np.mean(signal[:baseline_samples])
    corrected = signal - baseline
    amplitude = float(np.max(corrected))
    peak_idx = int(np.argmax(corrected))


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






if __name__ == "__main__":

    xd = np.linspace(0.0, 1.0, 101)
    yd = np.sin(2.0 * np.pi * xd)
    xc = np.linspace(0.0, 1.0, 11)
    yc = pwl_approx_1d(len(xd), xd, yd, len(xc), xc)
    xi = np.linspace(0.0, 1.0, 201)
    yi = pwl_interp_1d(len(xc), xc, yc, len(xi), xi)
    y_true = np.sin(2.0 * np.pi * xi)
    rmse = np.sqrt(np.mean((yi - y_true) ** 2))
    assert rmse < 0.15, f"PWL 近似误差过大: {rmse}"


    coeffs = np.array([-1.0, 0.0, 0.0, 1.0])
    roots = aberth_ehrlich(coeffs, max_iter=200)
    for r in roots:
        assert abs(r**3 - 1.0) < 1e-10, f"Aberth 求根失败: {r}"


    t = np.linspace(0.0, 20.0e-6, 1000)
    h = cr_rc_n_pulse_response(t, tau_cr=1.0e-6, tau_rc=2.0e-6, n_rc=4)
    assert np.all(h >= 0.0), "脉冲响应出现负值"
    assert np.max(h) > 0.0, "脉冲响应为零"

    print("signal_formation.py: 所有自测通过")
