
import numpy as np


def rf_elliptic_integral(x, y, z, errtol=1.0e-3):
    lolim = 3.0e-78
    uplim = 1.0e+75

    if (x < 0.0 or y < 0.0 or z < 0.0 or
        x + y < lolim or x + z < lolim or y + z < lolim or
        uplim <= x or uplim <= y or uplim <= z):
        return 0.0, 1

    xn = float(x)
    yn = float(y)
    zn = float(z)

    while True:
        mu = (xn + yn + zn) / 3.0
        xndev = 2.0 - (mu + xn) / mu
        yndev = 2.0 - (mu + yn) / mu
        zndev = 2.0 - (mu + zn) / mu
        epslon = max(abs(xndev), max(abs(yndev), abs(zndev)))

        if epslon < errtol:
            c1 = 1.0 / 24.0
            c2 = 3.0 / 44.0
            c3 = 1.0 / 14.0
            e2 = xndev * yndev - zndev * zndev
            e3 = xndev * yndev * zndev
            s = 1.0 + (c1 * e2 - 0.1 - c2 * e3) * e2 + c3 * e3
            value = s / np.sqrt(mu)
            return value, 0

        xnroot = np.sqrt(xn)
        ynroot = np.sqrt(yn)
        znroot = np.sqrt(zn)
        lamda = xnroot * (ynroot + znroot) + ynroot * znroot
        xn = (xn + lamda) * 0.25
        yn = (yn + lamda) * 0.25
        zn = (zn + lamda) * 0.25


def flame_curvature_elliptic(a, b, c_axis, s_param):

    a = max(a, 1.0e-12)
    b = max(b, 1.0e-12)
    c_axis = max(c_axis, 1.0e-12)


    x_rf = (b / a) ** 2
    y_rf = (c_axis / a) ** 2
    z_rf = 1.0

    rf_val, ierr = rf_elliptic_integral(x_rf, y_rf, z_rf, errtol=1.0e-3)
    if ierr != 0:
        rf_val = 1.0


    curvature = 1.0 / rf_val * (1.0 / a + 1.0 / b) / 2.0


    area_element = 4.0 * np.pi * a * b * rf_val

    return curvature, area_element


def markstein_length(Le, alpha_diff=2.0e-5, S_L=0.4):
    delta_L = alpha_diff / S_L if S_L > 0 else 1.0e-3

    if abs(Le - 1.0) < 1.0e-6:
        L_M = delta_L
    else:

        if Le <= 0.0:
            Le = 1.0e-6
        L_M = delta_L * (Le * np.log(1.0 / Le) / (Le - 1.0))

    return L_M


def curved_flame_speed(S_L, curvature, Le, alpha_diff=2.0e-5):
    L_M = markstein_length(Le, alpha_diff, S_L)


    correction = 1.0 - L_M * curvature
    correction = np.clip(correction, 0.3, 3.0)

    S_n = S_L * correction
    return S_n
