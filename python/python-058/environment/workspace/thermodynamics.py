
import numpy as np
from typing import Tuple, Callable


_Rd = 287.05
_Rv = 461.51
_cp = 1004.0
_Lv = 2.501e6
_g  = 9.80665
_p0 = 100000.0
_T0 = 273.15
_e0 = 611.2
_epsilon = _Rd / _Rv






def lambert_w(x: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    if x < -1.0 / np.e:
        return np.nan
    if x == 0.0:
        return 0.0
    if x < -0.35:

        p = np.sqrt(2.0 * (np.e * x + 1.0))
        w = -1.0 + p - p**2 / 3.0 + 11.0 * p**3 / 72.0
    elif x < 0.5:
        w = x * (1.0 + x * (1.0 + x * (3.0 / 2.0))) / (1.0 + x * (5.0 / 2.0))
    elif x < 2.0:

        w = 0.5 + 0.5 * x / (x + np.e)
    elif x < 10.0:
        lx = np.log(x)
        if lx > 0.01:
            w = lx - np.log(lx) + np.log(lx) / lx
        else:
            w = lx
    else:
        L1 = np.log(x)
        L2 = np.log(L1)
        w = L1 - L2 + L2 / L1 + L2 * (L2 - 2.0) / (2.0 * L1**2)

    for _ in range(max_iter):
        ew = np.exp(w)
        we = w * ew - x
        w1 = w + 1.0
        dw = we / (ew * w1 - (w + 2.0) * we / (2.0 * w1))
        w -= dw
        if abs(dw) < tol * (abs(w) + 1.0):
            break
    return w


def log_gamma(x: float) -> float:
    if x <= 0.0:
        return np.nan
    if x < 7.0:

        f = 1.0
        y = x
        while y < 7.0:
            f *= y
            y += 1.0
        return log_gamma(y) - np.log(f)

    z = 1.0 / x**2
    s = (1.0 / 12.0 - z * (1.0 / 360.0 - z * (1.0 / 1260.0 - z * (1.0 / 1680.0)))) / x
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi) + s






def saturation_vapor_pressure(temperature: float) -> float:
    if temperature <= 0.0:
        return 0.0
    return _e0 * np.exp((_Lv / _Rv) * (1.0 / _T0 - 1.0 / temperature))


def saturation_vapor_pressure_lambert(temperature: float) -> float:
    a = _Lv / (_Rv * _T0)
    b = _Lv / _Rv
    if temperature <= 0.0:
        return 0.0
    arg = (a - b / temperature) * np.exp(a - b / temperature)
    if arg < -1.0 / np.e:
        return 0.0
    w = lambert_w(arg)
    return _e0 * w / (a - b / temperature) if abs(a - b / temperature) > 1e-12 else saturation_vapor_pressure(temperature)


def dewpoint_from_vapor_pressure(e: float) -> float:
    if e <= 0.0:
        return _T0 - 50.0
    if e >= saturation_vapor_pressure(350.0):
        return 350.0
    a = _Lv / (_Rv * _T0)
    b = _Lv / _Rv

    Td_approx = b / (a - np.log(e / _e0))

    for _ in range(5):
        f = saturation_vapor_pressure(Td_approx) - e
        df = _Lv / (_Rv * Td_approx**2) * saturation_vapor_pressure(Td_approx)
        if abs(df) < 1e-20:
            break
        dT = f / df
        Td_approx -= dT
        if abs(dT) < 1e-4:
            break
    return max(150.0, Td_approx)






def potential_temperature(temperature: float, pressure: float) -> float:
    if pressure <= 0.0:
        return temperature
    return temperature * (_p0 / pressure)**(_Rd / _cp)


def virtual_temperature(temperature, qv):
    qv_safe = np.asarray(qv)
    qv_safe = np.where(qv_safe > 0.0, qv_safe, 0.0)
    return np.asarray(temperature) * (1.0 + 0.608 * qv_safe)


def mixing_ratio_to_specific_humidity(w: float) -> float:
    return w / (1.0 + w)


def specific_humidity_from_t_rh(temperature: float, rh: float, pressure: float) -> float:
    es = saturation_vapor_pressure(temperature)
    q = _epsilon * es / (pressure - (1.0 - _epsilon) * es) * rh
    return max(0.0, min(q, 0.05))






def brent_zero(f: Callable[[float], float], a: float, b: float,
               tol: float = 1e-8, max_iter: int = 100) -> float:
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:

        for scale in [2.0, 5.0, 10.0, 50.0, 100.0]:
            mid = (a + b) / 2.0
            new_a = mid - scale * (b - a)
            new_b = mid + scale * (b - a)
            fa_new = f(new_a)
            fb_new = f(new_b)
            if fa_new * fb_new <= 0.0:
                a, b = new_a, new_b
                fa, fb = fa_new, fb_new
                break
        else:
            return np.nan

    c, fc = a, fa
    s = b
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        tol_act = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or fb == 0.0:
            return b
        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < (min1 if min1 < min2 else min2):
                d = e = p / q
            else:
                d = e = m
        a = b
        fa = fb
        if abs(d) > tol_act:
            b += d
        else:
            b += tol_act if m > 0.0 else -tol_act
        fb = f(b)
    return b






def fixed_point_iteration(g: Callable[[float], float], x0: float,
                          tol: float = 1e-10, max_iter: int = 100) -> float:
    x = x0
    for k in range(max_iter):
        try:
            x_new = g(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            return x
        if not np.isfinite(x_new):

            x_new = x * 0.5 + x0 * 0.5
        if abs(x_new - x) < tol * (abs(x) + 1.0):
            return x_new

        if k >= 2:
            x_new = x_new - (x_new - x)**2 / (x_new - 2.0 * x + x_prev2 + 1e-30)
        x_prev2 = x
        x = x_new
    return x


def saturation_adjustment(temperature: float, q_total: float, pressure: float,
                          dt: float = 1.0) -> Tuple[float, float]:
    def qsat(Tk):
        es = saturation_vapor_pressure(Tk)
        return _epsilon * es / (pressure - (1.0 - _epsilon) * es)

    qv = min(q_total, qsat(temperature))

    def g(Tk):
        qs = qsat(Tk)
        if qs >= q_total:
            return temperature
        return temperature + (_Lv / _cp) * (qs - q_total)

    if qsat(temperature) >= q_total:
        return temperature, q_total

    T_new = fixed_point_iteration(g, temperature, tol=1e-6, max_iter=50)
    qv_new = min(qsat(T_new), q_total)

    T_new = max(150.0, min(T_new, 350.0))
    qv_new = max(0.0, min(qv_new, q_total))
    return T_new, qv_new






def lifting_condensation_level(p_sfc: float, T_sfc: float, qv_sfc: float) -> Tuple[float, float]:
    Td = dewpoint_from_vapor_pressure(
        qv_sfc * p_sfc / (_epsilon + (1.0 - _epsilon) * qv_sfc)
    )

    TL = 1.0 / (1.0 / (Td - 56.0) + np.log(T_sfc / Td) / 800.0) + 56.0
    p_lcl = p_sfc * (TL / T_sfc)**(1.0 / (_Rd / _cp))
    return max(10000.0, p_lcl), max(200.0, TL)


def parcel_temperature(pressure: float, p_sfc: float, T_sfc: float, qv_sfc: float) -> float:
    p_lcl, T_lcl = lifting_condensation_level(p_sfc, T_sfc, qv_sfc)
    if pressure >= p_lcl:

        return T_sfc * (pressure / p_sfc)**(_Rd / _cp)

    T = T_lcl
    p = p_lcl
    dp = -5000.0
    while p > pressure and p > 10000.0:
        p_next = max(pressure, p + dp)
        es = saturation_vapor_pressure(T)
        qs = _epsilon * es / (p - (1.0 - _epsilon) * es)
        dT_dp = (1.0 / p) * (_Rd * T + _Lv * qs) / (_cp + (_Lv**2) * qs * _epsilon / (_Rv * T**2))
        T += dT_dp * (p_next - p)
        p = p_next
    return max(150.0, T)


def compute_cape_cin(pressure_levels: np.ndarray, T_env: np.ndarray,
                     qv_env: np.ndarray, p_sfc: float, T_sfc: float,
                     qv_sfc: float) -> Tuple[float, float, float, float, float]:
    n = len(pressure_levels)

    Tv_env = virtual_temperature(T_env, qv_env)

    T_parcel = np.array([parcel_temperature(p, p_sfc, T_sfc, qv_sfc) for p in pressure_levels])

    qv_parcel = np.zeros(n)
    p_lcl, _ = lifting_condensation_level(p_sfc, T_sfc, qv_sfc)
    for i, p in enumerate(pressure_levels):
        if p >= p_lcl:
            qv_parcel[i] = qv_sfc
        else:
            qv_parcel[i] = _epsilon * saturation_vapor_pressure(T_parcel[i]) / (
                p - (1.0 - _epsilon) * saturation_vapor_pressure(T_parcel[i])
            )
    Tv_parcel = virtual_temperature(T_parcel, qv_parcel)












    raise NotImplementedError("HOLE 1: compute_cape_cin 的 buoyancy/CAPE/CIN 计算尚未实现")

