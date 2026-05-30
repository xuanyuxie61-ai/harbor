
import numpy as np


def complex_logarithm_toms243(z: complex) -> complex:
    a = float(z.real)
    b = float(z.imag)

    if abs(a) < 1e-300 and abs(b) < 1e-300:
        return complex(np.nan, np.nan)

    e = a / 2.0
    f = b / 2.0

    if abs(e) < 0.5 and abs(f) < 0.5:
        c = abs(2.0 * a) + abs(2.0 * b)
        if c < 1e-300:
            return complex(-700.0, 0.0)
        d = 8.0 * (a / c) * a + 8.0 * (b / c) * b
        c_val = 0.5 * (np.log(c) + np.log(d)) - np.log(np.sqrt(8.0))
    else:
        c = abs(e / 2.0) + abs(f / 2.0)
        if c < 1e-300:
            return complex(-700.0, 0.0)
        d = 0.5 * (e / c) * e + 0.5 * (f / c) * f
        c_val = 0.5 * (np.log(c) + np.log(d)) + np.log(np.sqrt(8.0))


    if a != 0.0 and abs(f) <= abs(e):
        if np.sign(a) >= 0:
            d_val = np.arctan(b / a)
        elif np.sign(b) >= 0:
            d_val = np.arctan(b / a) + np.pi
        else:
            d_val = np.arctan(b / a) - np.pi
    else:
        if b > 0:
            d_val = -np.arctan(a / b) + np.pi / 2.0
        elif b < 0:
            d_val = -np.arctan(a / b) - np.pi / 2.0
        else:
            d_val = 0.0 if a > 0 else np.pi

    return complex(c_val, d_val)


def shepard_interp_1d(xd: np.ndarray, yd: np.ndarray, p: float, xi: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    nd = len(xd)
    ni = len(xi)
    yi = np.zeros(ni, dtype=np.float64)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd) / nd
        else:
            w = np.abs(xi[i] - xd)
            z = np.where(w < 1e-15)[0]
            if len(z) > 0:
                w = np.zeros(nd)
                w[z[0]] = 1.0
            else:
                w = 1.0 / (w ** p)
                s = np.sum(w)
                if s > 0:
                    w = w / s
                else:
                    w = np.ones(nd) / nd
        yi[i] = np.dot(w, yd)

    return yi


class SonarSignalProcessor:

    def __init__(self, fs: float = 48000.0, f0: float = 12000.0, bandwidth: float = 4000.0):
        self.fs = float(fs)
        self.f0 = float(f0)
        self.bandwidth = float(bandwidth)

    def generate_chirp_pulse(self, duration: float = 0.01) -> tuple:
        T = duration
        t = np.arange(0.0, T, 1.0 / self.fs)
        f_start = self.f0 - self.bandwidth / 2.0
        k = self.bandwidth / T
        phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t ** 2)
        s = np.exp(1j * phase)
        return t, s

    def matched_filter(self, received: np.ndarray, template: np.ndarray) -> np.ndarray:
        n_recv = len(received)
        n_temp = len(template)
        n_fft = 1
        while n_fft < n_recv + n_temp - 1:
            n_fft *= 2

        R = np.fft.fft(received, n_fft)
        S = np.fft.fft(template, n_fft)
        Y = R * np.conj(S)
        y = np.fft.ifft(Y)
        return y[:n_recv]

    def compute_envelope(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        n_fft = 1
        while n_fft < n:
            n_fft *= 2
        S = np.fft.fft(signal, n_fft)

        h = np.zeros(n_fft)
        h[0] = 1.0
        h[1:n_fft // 2] = 2.0
        if n_fft % 2 == 0:
            h[n_fft // 2] = 1.0
        S_h = S * h
        s_analytic = np.fft.ifft(S_h)
        envelope = np.abs(s_analytic[:n])
        return envelope

    def detect_peak_time(
        self,
        t: np.ndarray,
        envelope: np.ndarray,
        threshold_ratio: float = 0.3
    ) -> float:
        threshold = threshold_ratio * np.max(envelope)
        peaks = np.where(envelope > threshold)[0]
        if len(peaks) == 0:
            return -1.0

        peak_idx = peaks[np.argmax(envelope[peaks])]

        half_win = min(5, peak_idx, len(t) - peak_idx - 1)
        if half_win < 2:
            return float(t[peak_idx])

        local_idx = np.arange(peak_idx - half_win, peak_idx + half_win + 1)
        local_t = t[local_idx]
        local_env = envelope[local_idx]


        ti_fine = np.linspace(local_t[0], local_t[-1], 200)
        env_fine = shepard_interp_1d(local_t, local_env, p=2.0, xi=ti_fine)
        max_fine_idx = np.argmax(env_fine)
        return float(ti_fine[max_fine_idx])

    def compute_log_spectrum(self, signal: np.ndarray) -> tuple:
        n = len(signal)
        n_fft = 1
        while n_fft < n:
            n_fft *= 2
        S = np.fft.fft(signal, n_fft)

        log_spectrum = np.array([complex_logarithm_toms243(z) for z in S], dtype=complex)
        freqs = np.fft.fftfreq(n_fft, d=1.0 / self.fs)
        return freqs, log_spectrum

    def process_single_ping(
        self,
        received_signal: np.ndarray,
        pulse_duration: float = 0.01,
        noise_std: float = 0.01
    ) -> dict:

        _, template = self.generate_chirp_pulse(pulse_duration)


        mf_output = self.matched_filter(received_signal, template)


        envelope = self.compute_envelope(mf_output)


        t = np.arange(len(received_signal)) / self.fs


        peak_time = self.detect_peak_time(t, envelope)


        freqs, log_spec = self.compute_log_spectrum(received_signal)


        signal_power = np.mean(np.abs(mf_output) ** 2)
        noise_power = noise_std ** 2
        snr_db = 10.0 * np.log10(signal_power / (noise_power + 1e-15) + 1e-15)

        return {
            't': t,
            'envelope': envelope,
            'peak_time': peak_time,
            'freqs': freqs,
            'log_spectrum': log_spec,
            'snr_db': float(snr_db),
        }
