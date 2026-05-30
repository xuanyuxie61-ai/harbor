
import numpy as np
from typing import Tuple





def carlson_rf(x: float, y: float, z: float, tol: float = 1e-12) -> float:
    if x < 0.0 or y < 0.0 or z < 0.0:
        raise ValueError("RF arguments must be non-negative")
    if x + y == 0.0 or x + z == 0.0 or y + z == 0.0:
        raise ValueError("At most one of RF arguments may be zero")

    xn, yn, zn = float(x), float(y), float(z)
    a0 = (xn + yn + zn) / 3.0
    dx, dy, dz = a0 - xn, a0 - yn, a0 - zn
    e2 = dx * dy + dy * dz + dz * dx
    e3 = dx * dy * dz

    result = (1.0 - e2 / (10.0 * a0 * a0)
              + e3 / (14.0 * a0 * a0 * a0)
              + e2 * e2 / (24.0 * a0 ** 4)) / np.sqrt(a0)
    return result


def carlson_rd(x: float, y: float, z: float, tol: float = 1e-12) -> float:
    if x < 0.0 or y < 0.0 or z <= 0.0:
        raise ValueError("RD arguments must be non-negative and z>0")
    xn, yn, zn = float(x), float(y), float(z)
    s = 0.0
    fac = 1.0
    for _ in range(100):
        if max(abs(xn - yn), abs(xn - zn), abs(yn - zn)) < tol * (abs(xn) + abs(yn) + abs(zn)):
            break
        sx = np.sqrt(xn)
        sy = np.sqrt(yn)
        sz = np.sqrt(zn)
        lm = sx * sy + sx * sz + sy * sz
        s += fac / (sz * (zn + lm))
        fac /= 4.0
        xn = (xn + lm) / 4.0
        yn = (yn + lm) / 4.0
        zn = (zn + lm) / 4.0
    else:
        raise RuntimeError("RD iteration did not converge")
    a0 = (xn + yn + 3.0 * zn) / 5.0
    dx, dy, dz = a0 - xn, a0 - yn, a0 - zn
    e2 = dx * dy + 3.0 * dz * dz + 2.0 * dz * (dx + dy)
    e3 = 3.0 * dx * dy * dz + 2.0 * dz * dz * (dx + dy) + dz ** 3
    e4 = dx * dy * dz * dz
    e5 = dx * dy * dz ** 3
    result = (3.0 * s + fac * (1.0
                                - 3.0 * e2 / (14.0 * a0 * a0)
                                + e3 / (6.0 * a0 ** 3)
                                + 9.0 * e2 * e2 / (88.0 * a0 ** 4)
                                - 3.0 * e4 / (22.0 * a0 ** 4)
                                - 9.0 * e2 * e3 / (52.0 * a0 ** 5)
                                + 3.0 * e5 / (26.0 * a0 ** 5))) / (a0 * np.sqrt(a0))
    return result


def carlson_rc(x: float, y: float, tol: float = 1e-12) -> float:
    if x < 0.0 or y <= 0.0:
        raise ValueError("RC requires x>=0 and y>0")
    if x == 0.0:
        return np.pi / (2.0 * np.sqrt(y))
    xn, yn = float(x), float(y)
    for _ in range(100):
        if abs(xn - yn) < tol * (abs(xn) + abs(yn)):
            break
        sx = np.sqrt(xn)
        sy = np.sqrt(yn)
        lm = 2.0 * sx * sy + yn
        xn = (xn + lm) / 4.0
        yn = (yn + lm) / 4.0
    else:
        raise RuntimeError("RC iteration did not converge")
    return 1.0 / np.sqrt(yn)






def jacobi_sncndn(u: float, m: float) -> Tuple[float, float, float]:
    if not (0.0 <= m <= 1.0):
        raise ValueError("Jacobi parameter m must be in [0,1]")
    if m == 0.0:
        return np.sin(u), np.cos(u), 1.0
    if m == 1.0:
        return np.tanh(u), 1.0 / np.cosh(u), 1.0 / np.cosh(u)

    a = [1.0]
    b = [np.sqrt(1.0 - m)]
    c = [np.sqrt(m)]

    for _ in range(16):
        if abs(c[-1]) < 1e-15:
            break
        a_next = (a[-1] + b[-1]) / 2.0
        b_next = np.sqrt(a[-1] * b[-1])
        c_next = (a[-1] - b[-1]) / 2.0
        a.append(a_next)
        b.append(b_next)
        c.append(c_next)

    n = len(a) - 1
    phi = (2.0 ** n) * a[-1] * u

    for i in range(n, 0, -1):
        phi = (phi + np.arcsin(c[i] / a[i] * np.sin(phi))) / 2.0

    sn = np.sin(phi)
    cn = np.cos(phi)
    dn = np.sqrt(1.0 - m * sn * sn)
    return sn, cn, dn






def gauss_agm(a: float, b: float, tol: float = 1e-14) -> Tuple[float, int]:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("AGM requires positive arguments")
    a_n, b_n = float(a), float(b)
    count = 0
    for _ in range(100):
        if abs(a_n - b_n) < tol * (abs(a_n) + abs(b_n)):
            break
        a_next = (a_n + b_n) / 2.0
        b_next = np.sqrt(a_n * b_n)
        a_n, b_n = a_next, b_next
        count += 1
    return a_n, count






def jacobi_theta(x: float, q: float, which: int = 1) -> float:
    if not (0.0 <= q < 1.0):
        raise ValueError("Jacobi theta requires 0 <= q < 1")
    if which not in (1, 2, 3, 4):
        raise ValueError("which must be 1,2,3,4")



    eps = 1e-15
    max_iter = 1000
    result = 0.0
    if which == 1:
        for n in range(max_iter):
            term = 2.0 * ((-1) ** n) * (q ** ((n + 0.5) ** 2)) * np.sin((2.0 * n + 1.0) * x)
            result += term
            if abs(term) < eps:
                break
    elif which == 2:
        for n in range(max_iter):
            term = 2.0 * (q ** ((n + 0.5) ** 2)) * np.cos((2.0 * n + 1.0) * x)
            result += term
            if abs(term) < eps:
                break
    elif which == 3:
        result = 1.0
        for n in range(1, max_iter):
            term = 2.0 * (q ** (n * n)) * np.cos(2.0 * n * x)
            result += term
            if abs(term) < eps:
                break
    else:
        result = 1.0
        for n in range(1, max_iter):
            term = 2.0 * ((-1) ** n) * (q ** (n * n)) * np.cos(2.0 * n * x)
            result += term
            if abs(term) < eps:
                break
    return result






def hyper_2f1(a: float, b: float, c: float, z: float, max_iter: int = 10000) -> float:
    if abs(z) >= 1.0:


        if z >= 1.0:
            raise ValueError("Hypergeometric series diverges for z>=1")


        z_new = z / (z - 1.0)
        factor = (1.0 - z) ** (-a)
        return factor * hyper_2f1(a, c - b, c, z_new, max_iter)

    if c == 0.0 or c == -1.0 or c == -2.0:
        raise ValueError("c must not be zero or negative integer")

    result = 1.0
    term = 1.0
    for n in range(1, max_iter):
        term *= (a + n - 1.0) * (b + n - 1.0) * z / ((c + n - 1.0) * n)
        result += term
        if abs(term) < 1e-15 * abs(result):
            break
    else:
        raise RuntimeError("Hypergeometric series did not converge")
    return result


def drug_protein_binding_fraction(K_a: float, C_p: float, n_sites: int = 1) -> float:
    if K_a < 0.0 or C_p < 0.0 or n_sites < 1:
        raise ValueError("Invalid binding parameters")
    if C_p == 0.0:
        return 1.0
    z = -K_a * C_p

    if abs(z) >= 1.0:



        if abs(z) > 100.0:
            return 1.0 / (1.0 + K_a * C_p / n_sites)
        z_new = z / (z - 1.0)
        factor = (1.0 - z) ** (-1.0)
        hf = factor * hyper_2f1(1.0, (n_sites + 1.0) - n_sites, n_sites + 1.0, z_new)
    else:
        hf = hyper_2f1(1.0, float(n_sites), float(n_sites + 1), z)
    if hf <= 0.0:
        raise RuntimeError("Hypergeometric computation produced non-positive result")
    f_u = 1.0 / hf
    return max(0.0, min(1.0, f_u))






def effective_diffusion_coefficient(D_parallel: float, D_perpendicular: float,
                                     theta: float = 0.0) -> float:
    if D_parallel <= 0.0 or D_perpendicular <= 0.0:
        raise ValueError("Diffusion coefficients must be positive")

    D11 = D_parallel * np.cos(theta) ** 2 + D_perpendicular * np.sin(theta) ** 2
    D22 = D_parallel * np.sin(theta) ** 2 + D_perpendicular * np.cos(theta) ** 2
    inv_sqrt_D11 = 1.0 / np.sqrt(D11)
    inv_sqrt_D22 = 1.0 / np.sqrt(D22)
    agm_val, _ = gauss_agm(inv_sqrt_D11, inv_sqrt_D22)
    D_eff = np.pi / (4.0 * agm_val)
    return D_eff






if __name__ == "__main__":

    rf_val = carlson_rf(1.0, 2.0, 0.0)
    print(f"RF(1,2,0) = {rf_val:.10f}")

    sn, cn, dn = jacobi_sncndn(0.5, 0.5)
    print(f"sn(0.5|0.5)={sn:.10f}, cn={cn:.10f}, dn={dn:.10f}")

    agm_val, _ = gauss_agm(1.0, np.sqrt(2.0))
    print(f"AGM(1,sqrt(2)) = {agm_val:.10f}")

    hf = hyper_2f1(1.0, 2.0, 3.0, 0.5)
    print(f"2F1(1,2;3;0.5) = {hf:.10f}")

    fu = drug_protein_binding_fraction(1e5, 1e-6)
    print(f"Free fraction at K_a=1e5, C_p=1e-6: {fu:.6f}")

    D_eff = effective_diffusion_coefficient(1e-9, 1e-10, np.pi / 4.0)
    print(f"Effective diffusion coefficient: {D_eff:.3e} m^2/s")
