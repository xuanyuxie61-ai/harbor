"""
potential_profile.py
--------------------
Spatial potential profiles for 1D non-Hermitian waveguide and
Schrödinger operators.

Adapted from seed project 920_profile_data (facial profile coordinates).

Scientific Background
=====================
In non-Hermitian photonic waveguides, the complex refractive-index
profile n(x) = n_r(x) + i n_i(x) acts as a spatial potential:

    V(x) = V_0 [ n_r(x) - n_{bg} ] + i V_0 n_i(x),

where n_r controls the real confining potential and n_i controls
gain/loss. A common model is the complex Pöschl-Teller potential:

    V(x) = -V_0 / cosh^2(α x) + i W_0 sinh(α x) / cosh^2(α x).

This potential is reflectionless for certain parameter ratios and
supports exceptional points where bound states coalesce with the
continuum threshold.

We also implement a periodic Kronig-Penney-like non-Hermitian potential:

    V(x) = V_1 cos(2π x / a) + i V_2 sin(2π x / a),

which is the continuous-space analog of the non-Hermitian SSH model.
"""

import numpy as np


def complex_poschl_teller(x, V0=1.0, W0=0.3, alpha=1.0):
    """
    Complex Pöschl-Teller potential.

    V(x) = -V0 / cosh^2(α x) + i W0 sinh(α x) / cosh^2(α x)

    Parameters
    ----------
    x : float or ndarray
    V0, W0, alpha : float

    Returns
    -------
    V : complex or ndarray, dtype=complex
    """
    c = np.cosh(alpha * x)
    s = np.sinh(alpha * x)
    V = -V0 / (c ** 2) + 1j * W0 * s / (c ** 2)
    return V


def nonhermitian_kronig_penney(x, V1=1.0, V2=0.3, a=1.0):
    """
    Periodic non-Hermitian Kronig-Penney potential.

    V(x) = V1 cos(2π x / a) + i V2 sin(2π x / a)

    Parameters
    ----------
    x : float or ndarray
    V1, V2, a : float

    Returns
    -------
    V : complex or ndarray, dtype=complex
    """
    k = 2.0 * np.pi / a
    V = V1 * np.cos(k * x) + 1j * V2 * np.sin(k * x)
    return V


def double_well_nonhermitian(x, V0=1.0, gamma=0.2, a=2.0, b=0.5):
    """
    Double-well potential with balanced gain and loss.

    V(x) = V0 [ (x^2 / a^2 - 1)^2 + i γ (x / b) ]

    The real part creates a symmetric double well, while the imaginary
    part introduces asymmetric gain/loss that can be tuned to achieve
    PT-symmetry breaking at an exceptional point.

    Parameters
    ----------
    x : float or ndarray
    V0, gamma, a, b : float

    Returns
    -------
    V : complex or ndarray, dtype=complex
    """
    V_real = V0 * ((x ** 2 / a ** 2 - 1.0) ** 2)
    V_imag = V0 * gamma * (x / b)
    return V_real + 1j * V_imag


def profile_based_potential(x, profile_points, scale=1.0, imaginary_ratio=0.2):
    """
    Generate a non-Hermitian potential by interpolating a 1D spatial
    profile (analogous to the facial profile data) and adding an
    imaginary component proportional to the gradient.

    The real part is the interpolated profile; the imaginary part is
    proportional to the derivative, modeling a directional gain/loss
    profile.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    profile_points : ndarray, shape (M, 2)
        (position, amplitude) pairs defining the profile.
    scale : float
        Overall amplitude scale.
    imaginary_ratio : float
        Ratio of imaginary to real amplitude.

    Returns
    -------
    V : ndarray, dtype=complex
    """
    pos = profile_points[:, 0]
    amp = profile_points[:, 1]
    V_real = np.interp(x, pos, amp, left=amp[0], right=amp[-1])
    # Approximate derivative via finite differences
    grad = np.gradient(amp, pos)
    V_imag = np.interp(x, pos, grad, left=grad[0], right=grad[-1])
    return scale * (V_real + 1j * imaginary_ratio * V_imag)


def get_default_profile():
    """
    Return a default 1D spatial profile analogous to the seed project
    profile_data, but normalized and suitable for potential applications.
    """
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
    # Center and normalize
    raw[:, 0] = (raw[:, 0] - raw[:, 0].min()) / (raw[:, 0].max() - raw[:, 0].min())
    raw[:, 1] = (raw[:, 1] - raw[:, 1].min()) / (raw[:, 1].max() - raw[:, 1].min()) - 0.5
    return raw
