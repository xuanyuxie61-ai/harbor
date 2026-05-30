
import numpy as np
from math import factorial




R_UNIVERSAL = 8.314462618
GAMMA_AIR = 1.4
GAMMA_COMBUSTION = 1.2
CP_COMBUSTION = 1800.0
RHO_OX = 1141.0
RHO_FUEL = 807.0
T_ADIABATIC = 3600.0
PRESSURE_CHAMBER = 7.0e6
SOUND_SPEED = 1200.0
DYNAMIC_VISCOSITY = 8.5e-5
LATENT_HEAT = 2.13e6
DIFFUSIVITY_THERMAL = 1.2e-4
PRE_EXPONENTIAL = 1.8e10
ACTIVATION_ENERGY = 1.26e5
STOICHIOMETRIC_RATIO = 2.56










def cordic_angles_table():
    angles = np.zeros(60)
    for k in range(60):
        angles[k] = np.arctan(2.0 ** (-k))
    return angles


def cordic_kprod_table():
    kprod = np.zeros(33)
    k_running = 1.0
    for k in range(33):
        k_running *= 1.0 / np.sqrt(1.0 + (2.0 ** (-2 * k)))
        kprod[k] = k_running
    return kprod


_CORDIC_ANGLES = cordic_angles_table()
_CORDIC_KPROD = cordic_kprod_table()


def cordic_cos_sin(beta: float, n_iter: int = 40) -> tuple:
    if not np.isfinite(beta):
        return np.nan, np.nan
    

    theta = beta % (2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    elif theta < -np.pi:
        theta += 2.0 * np.pi
    

    sign_factor = 1.0
    if theta < -0.5 * np.pi:
        theta += np.pi
        sign_factor = -1.0
    elif theta > 0.5 * np.pi:
        theta -= np.pi
        sign_factor = -1.0
    
    v = np.array([1.0, 0.0])
    poweroftwo = 1.0
    
    n_iter = max(1, min(n_iter, 60))
    
    for j in range(n_iter):
        sigma = 1.0 if theta >= 0.0 else -1.0
        factor = sigma * poweroftwo
        

        v_new = np.array([
            v[0] - factor * v[1],
            factor * v[0] + v[1]
        ])
        v = v_new
        

        angle = _CORDIC_ANGLES[j] if j < 60 else _CORDIC_ANGLES[59] / (2.0 ** (j - 59))
        theta -= sigma * angle
        poweroftwo *= 0.5
    

    if n_iter > 0:
        idx = min(n_iter - 1, len(_CORDIC_KPROD) - 1)
        v = v * _CORDIC_KPROD[idx]
    
    v = sign_factor * v
    return float(v[0]), float(v[1])


def cordic_arctan2(y: float, x: float, n_iter: int = 40) -> float:
    if x == 0.0 and y == 0.0:
        return 0.0
    

    angle = 0.0
    poweroftwo = 1.0
    
    n_iter = max(1, min(n_iter, 60))
    
    for j in range(n_iter):
        if y > 0:
            sigma = 1.0
        else:
            sigma = -1.0
        
        factor = sigma * poweroftwo
        x_new = x + factor * y
        y_new = y - factor * x
        x, y = x_new, y_new
        
        angle += sigma * _CORDIC_ANGLES[j]
        poweroftwo *= 0.5
    
    return float(angle)








def gamma_function_half_integer(n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive for gamma_function_half_integer")
    
    if n % 2 == 0:
        k = n // 2
        return float(factorial(k - 1))
    else:
        k = n // 2

        result = factorial(2 * k)
        result = result / (4.0 ** k * factorial(k))
        result = result * np.sqrt(np.pi)
        return float(result)


def circle_monomial_integral(e1: int, e2: int) -> float:
    if e1 < 0 or e2 < 0:
        raise ValueError("Exponents must be nonnegative")
    
    if (e1 % 2 == 1) or (e2 % 2 == 1):
        return 0.0
    

    if e1 > 100 or e2 > 100:

        log_I = np.log(2.0) + \
                _log_gamma((e1 + 1) * 0.5) + \
                _log_gamma((e2 + 1) * 0.5) - \
                _log_gamma((e1 + e2 + 2) * 0.5)
        return float(np.exp(log_I))
    
    integral = 2.0
    integral *= gamma_function_half_integer(e1 + 1)
    integral *= gamma_function_half_integer(e2 + 1)
    integral /= gamma_function_half_integer(e1 + e2 + 2)
    return float(integral)


def _log_gamma(z: float) -> float:
    if z <= 0:
        return np.inf

    p = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7]
    x = z
    y = z
    if y < 0.5:
        return np.log(np.pi) - np.log(np.sin(np.pi * y)) - _log_gamma(1.0 - y)
    y -= 1.0
    a = 0.99999999999980993
    for i, pi in enumerate(p):
        a += pi / (y + i + 1)
    t = y + len(p) - 0.5
    return 0.5 * np.log(2.0 * np.pi) + np.log(a) + (y + 0.5) * np.log(t) - t






def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-300:
        return default
    return a / b


def robust_sqrt(x: float) -> float:
    if x < 0:
        if x > -1e-14:
            return 0.0
        return np.nan
    return np.sqrt(x)


def check_finite_array(arr: np.ndarray, name: str = "array") -> None:
    if not np.all(np.isfinite(arr)):
        bad_idx = np.where(~np.isfinite(arr))[0]
        raise ValueError(f"{name} contains non-finite values at indices: {bad_idx[:10]}")






def combustion_temperature(pressure: float, mixture_ratio: float) -> float:
    if pressure <= 0:
        raise ValueError("Pressure must be positive")
    if mixture_ratio <= 0:
        raise ValueError("Mixture ratio must be positive")
    
    p_ratio = pressure / PRESSURE_CHAMBER
    r_dev = (mixture_ratio - STOICHIOMETRIC_RATIO) / STOICHIOMETRIC_RATIO
    
    T_c = T_ADIABATIC * (p_ratio ** 0.03) * np.exp(-0.05 * r_dev ** 2)
    return float(T_c)


def specific_impulse_ideal(pressure: float, expansion_ratio: float, gamma: float = GAMMA_COMBUSTION) -> float:
    g_0 = 9.80665
    R_specific = R_UNIVERSAL / 0.022
    T_c = combustion_temperature(pressure, STOICHIOMETRIC_RATIO)
    


    def area_ratio_func(Ma):
        if Ma <= 0:
            return 1e300
        term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * Ma ** 2)
        return (1.0 / Ma) * (term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
    

    Ma_low, Ma_high = 1.01, 20.0
    for _ in range(100):
        Ma_mid = 0.5 * (Ma_low + Ma_high)
        f_mid = area_ratio_func(Ma_mid)
        if f_mid > expansion_ratio:
            Ma_high = Ma_mid
        else:
            Ma_low = Ma_mid
        if Ma_high - Ma_low < 1e-8:
            break
    Ma_e = 0.5 * (Ma_low + Ma_high)
    

    p_ratio = (1.0 + (gamma - 1.0) / 2.0 * Ma_e ** 2) ** (-gamma / (gamma - 1.0))
    

    term = 1.0 - p_ratio ** ((gamma - 1.0) / gamma)
    v_e = np.sqrt(2.0 * gamma / (gamma - 1.0) * R_specific * T_c * term)
    I_sp = v_e / g_0
    
    return float(I_sp)


if __name__ == "__main__":

    c, s = cordic_cos_sin(np.pi / 3.0)
    print(f"CORDIC cos(π/3)={c:.15f}, sin(π/3)={s:.15f}")
    print(f"NumPy  cos(π/3)={np.cos(np.pi/3):.15f}, sin(π/3)={np.sin(np.pi/3):.15f}")
    
    I = circle_monomial_integral(2, 2)
    print(f"Circle integral x^2·y^2 = {I:.10f}")
    
    print(f"Specific impulse = {specific_impulse_ideal(7e6, 20.0):.2f} s")
