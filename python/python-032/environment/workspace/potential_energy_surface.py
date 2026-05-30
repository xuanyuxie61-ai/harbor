
import numpy as np
from typing import Callable, Tuple, Optional


LDM_A_VOLUME = 15.4941
LDM_A_SURFACE = 17.9439
LDM_A_COULOMB = 0.7053
LDM_A_ASYMMETRY = 23.2


def surface_area_ratio(beta2: float) -> float:

    b2 = beta2
    b2sq = b2 * b2
    return 1.0 + 0.4 * b2sq - (38.0 / 105.0) * b2sq * b2 + 0.2 * b2sq * b2sq


def coulomb_shape_factor(beta2: float) -> float:
    b2 = beta2
    b2sq = b2 * b2
    return 1.0 - 0.2 * b2sq - (4.0 / 105.0) * b2sq * b2 + 0.05 * b2sq * b2sq


def liquid_drop_energy(mass_number: int, charge_number: int, beta2: float) -> float:
    if mass_number <= 0 or charge_number <= 0 or charge_number > mass_number:
        raise ValueError("Invalid nuclear parameters")
    A = float(mass_number)
    Z = float(charge_number)
    N = A - Z
    
    Bs = surface_area_ratio(beta2)
    Bc = coulomb_shape_factor(beta2)
    
    E_vol = -LDM_A_VOLUME * A
    E_surf = LDM_A_SURFACE * (A ** (2.0 / 3.0)) * Bs
    E_coul = LDM_A_COULOMB * (Z ** 2) / (A ** (1.0 / 3.0)) * Bc
    E_asym = LDM_A_ASYMMETRY * (N - Z) ** 2 / A
    
    return E_vol + E_surf + E_coul + E_asym


def shell_correction_energy(beta2: float, beta3: float, mass_number: int) -> float:

    A_shell = 5.0 * np.exp(-mass_number / 200.0)
    

    omega2 = 8.0
    phase2 = omega2 * beta2
    

    A3 = 0.3 * A_shell
    omega3 = 6.0
    phase3 = omega3 * beta3
    

    damping = np.exp(-0.5 * (beta2 ** 2 + beta3 ** 2))
    
    delta_E = A_shell * np.cos(phase2) * damping + A3 * np.cos(phase3) * damping
    return delta_E


def pairing_correction_energy(delta: float, delta_0: float) -> float:
    if delta < 0:
        delta = 0.0
    if delta_0 <= 0:
        delta_0 = 1.0
    return -(delta ** 2) / delta_0


def fission_barrier_height(mass_number: int, charge_number: int) -> float:
    A = float(mass_number)
    Z = float(charge_number)
    x_fissility = (LDM_A_COULOMB * Z ** 2 / A ** (1.0 / 3.0)) / (
        2.0 * LDM_A_SURFACE * A ** (2.0 / 3.0)
    )
    x_fissility = np.clip(x_fissility, 0.0, 1.0)
    E_surf_0 = LDM_A_SURFACE * A ** (2.0 / 3.0)
    if x_fissility >= 1.0:
        return 0.0
    barrier = E_surf_0 * (1.0 - x_fissility) ** 3 / (1.0 + x_fissility)
    return barrier


def potential_energy(
    q: np.ndarray,
    mass_number: int,
    charge_number: int,
    delta_0: float = 1.5,
) -> float:
    if len(q) < 5:
        raise ValueError("q must contain at least 5 elements")
    beta2, beta3, beta4, beta5, delta_val = q[0], q[1], q[2], q[3], q[4]
    

    E_gs = liquid_drop_energy(mass_number, charge_number, 0.0)
    

    E_ldm = liquid_drop_energy(mass_number, charge_number, beta2)
    

    E_higher = 20.0 * (beta4 ** 2 + beta5 ** 2) * (mass_number ** (2.0 / 3.0))
    

    E_shell = shell_correction_energy(beta2, beta3, mass_number)
    

    E_pair = pairing_correction_energy(delta_val, delta_0)
    

    V = (E_ldm - E_gs) + E_higher + E_shell + E_pair
    return float(V)






def potential_energy_1d(beta2: float, mass_number: int, charge_number: int) -> float:
    q = np.array([beta2, 0.0, 0.0, 0.0, 0.0])
    return potential_energy(q, mass_number, charge_number)


def zero_laguerre(
    f: Callable[[float], float],
    x0: float,
    degree: int = 4,
    abserr: float = 1e-10,
    kmax: int = 100,
) -> Tuple[float, int, int]:
    if degree < 2:
        degree = 2
    x = float(x0)
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1)
    
    h = 1e-6
    
    while True:
        fx = f(x)
        if abs(fx) <= abserr:
            break
        

        fp = (f(x + h) - f(x - h)) / (2.0 * h)
        fpp = (f(x + h) - 2.0 * fx + f(x - h)) / (h * h)
        
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k
        
        z = fp ** 2 - (beta + 1.0) * fx * fpp
        z = max(z, 0.0)
        bot = beta * fp + np.sqrt(z)
        
        if abs(bot) < 1e-15:
            ierror = 3
            return x, ierror, k
        
        dx = -(beta + 1.0) * fx / bot
        x = x + dx
        

        if not np.isfinite(x):
            ierror = 4
            return x, ierror, k
    
    return x, ierror, k


def zero_muller(
    f: Callable[[complex], complex],
    x1: complex,
    x2: complex,
    x3: complex,
    fatol: float = 1e-12,
    xatol: float = 1e-12,
    xrtol: float = 1e-12,
    itmax: int = 100,
) -> Tuple[complex, complex]:
    xnew = complex(x1)
    xmid = complex(x2)
    xold = complex(x3)
    
    fxnew = f(xnew)
    fxmid = f(xmid)
    fxold = f(xold)
    
    if abs(fxnew) < fatol:
        return xnew, fxnew
    
    iterate = 0
    while True:

        if abs(fxmid) <= abs(fxnew):
            xnew, xmid = xmid, xnew
            fxnew, fxmid = fxmid, fxnew
        
        xlast = xnew
        iterate += 1
        if iterate > itmax:
            break
        

        a_num = (xmid - xnew) * (fxold - fxnew) - (xold - xnew) * (fxmid - fxnew)
        b_num = (xold - xnew) ** 2 * (fxmid - fxnew) - (xmid - xnew) ** 2 * (fxold - fxnew)
        c_num = (xold - xnew) * (xmid - xnew) * (xold - xmid) * fxnew
        
        denom = (xold - xnew) * (xmid - xnew) * (xold - xmid)
        if abs(denom) < 1e-20:
            break
        
        a = a_num / denom
        b = b_num / denom

        

        a_coef = a_num
        b_coef = b_num
        c_coef = c_num
        
        discrm = b_coef ** 2 - 4.0 * a_coef * c_coef
        
        if abs(a_coef) < 1e-20:
            break
        
        sqrt_disc = np.sqrt(discrm)
        xplus = xnew + (-b_coef + sqrt_disc) / (2.0 * a_coef)
        xminus = xnew + (-b_coef - sqrt_disc) / (2.0 * a_coef)
        
        fplus = f(xplus)
        fminus = f(xminus)
        
        if abs(fminus) < abs(fplus):
            xnew = xminus
            fxnew = fminus
        else:
            xnew = xplus
            fxnew = fplus
        
        fxold = fxmid
        fxmid = fxnew
        xold = xmid
        xmid = xlast
        

        x_inc = xnew - xmid
        x_ave = abs(xnew + xmid + xold) / 3.0
        if abs(x_inc) <= xatol:
            break
        if abs(x_inc) <= xrtol * x_ave:
            break
        if abs(fxnew) <= fatol:
            break
    
    return xnew, fxnew


def find_saddle_point_1d(
    mass_number: int,
    charge_number: int,
    beta2_min: float = -0.3,
    beta2_max: float = 2.0,
) -> Tuple[float, float]:
    n_scan = 200
    beta2_grid = np.linspace(beta2_min, beta2_max, n_scan)
    V_grid = np.array([potential_energy_1d(b, mass_number, charge_number) for b in beta2_grid])
    

    dV = np.diff(V_grid)
    sign_change = np.where((dV[:-1] > 0) & (dV[1:] < 0))[0]
    
    if len(sign_change) == 0:

        idx_max = np.argmax(V_grid)
        return float(beta2_grid[idx_max]), float(V_grid[idx_max])
    

    idx = sign_change[0]
    x0 = float(beta2_grid[idx + 1])
    
    dV_func = lambda b: (potential_energy_1d(b + 1e-5, mass_number, charge_number) -
                         potential_energy_1d(b - 1e-5, mass_number, charge_number)) / (2e-5)
    
    beta_saddle, ierr, _ = zero_laguerre(dV_func, x0, degree=6, abserr=1e-8, kmax=200)
    
    if ierr != 0:

        beta_saddle = x0
    
    V_saddle = potential_energy_1d(beta_saddle, mass_number, charge_number)
    return float(beta_saddle), float(V_saddle)


def find_scission_point_1d(
    mass_number: int,
    charge_number: int,
    beta2_saddle: float,
    beta2_max: float = 3.0,
) -> Tuple[float, float]:
    n_scan = 300
    beta2_grid = np.linspace(beta2_saddle, beta2_max, n_scan)
    V_grid = np.array([potential_energy_1d(b, mass_number, charge_number) for b in beta2_grid])
    
    dV = np.gradient(V_grid, beta2_grid)
    

    threshold = 0.5
    candidates = np.where((np.abs(dV) < threshold) & (beta2_grid > beta2_saddle + 0.2))[0]
    
    if len(candidates) == 0:
        idx = len(beta2_grid) - 1
    else:
        idx = candidates[0]
    
    return float(beta2_grid[idx]), float(V_grid[idx])
