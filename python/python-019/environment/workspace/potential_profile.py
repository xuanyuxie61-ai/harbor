
import numpy as np


def complex_poschl_teller(x, V0=1.0, W0=0.3, alpha=1.0):
    c = np.cosh(alpha * x)
    s = np.sinh(alpha * x)
    V = -V0 / (c ** 2) + 1j * W0 * s / (c ** 2)
    return V


def nonhermitian_kronig_penney(x, V1=1.0, V2=0.3, a=1.0):
    k = 2.0 * np.pi / a
    V = V1 * np.cos(k * x) + 1j * V2 * np.sin(k * x)
    return V


def double_well_nonhermitian(x, V0=1.0, gamma=0.2, a=2.0, b=0.5):
    V_real = V0 * ((x ** 2 / a ** 2 - 1.0) ** 2)
    V_imag = V0 * gamma * (x / b)
    return V_real + 1j * V_imag


def profile_based_potential(x, profile_points, scale=1.0, imaginary_ratio=0.2):
    pos = profile_points[:, 0]
    amp = profile_points[:, 1]
    V_real = np.interp(x, pos, amp, left=amp[0], right=amp[-1])

    grad = np.gradient(amp, pos)
    V_imag = np.interp(x, pos, grad, left=grad[0], right=grad[-1])
    return scale * (V_real + 1j * imaginary_ratio * V_imag)


def get_default_profile():
    raw = np.array([
        [2.0, 0.0], [3.0, -1.0], [4.0, -0.5], [5.0, 0.0],
        [6.0, 1.0], [7.0, 2.0], [7.5, 5.0], [8.0, 7.5],
        [9.0, 8.0], [10.0, 8.0], [11.0, 8.2], [11.5, 9.0],
        [12.0, 8.2], [12.5, 6.5], [12.9, 8.9], [13.0, 9.0],
        [14.0, 9.0], [14.7, 10.0], [14.8, 10.5], [14.9, 11.0],
        [15.0, 11.4], [15.3, 11.6], [15.6, 11.5], [16.0, 11.5],
        [17.0, 11.0], [18.0, 10.4], [19.0, 10.0], [20.0, 9.7],
        [21.0, 10.0], [22.0, 10.5], [23.0, 10.7], [24.0, 11.2],
        [25.0, 10.9], [26.0, 10.4], [27.0, 10.2], [28.0, 10.0],
        [29.0, 11.2], [30.0, 11.3], [31.0, 10.2], [32.0, 9.0],
        [33.0, 7.0], [34.0, 5.0], [35.0, 3.0], [36.0, 0.0]
    ])

    raw[:, 0] = (raw[:, 0] - raw[:, 0].min()) / (raw[:, 0].max() - raw[:, 0].min())
    raw[:, 1] = (raw[:, 1] - raw[:, 1].min()) / (raw[:, 1].max() - raw[:, 1].min()) - 0.5
    return raw
