"""
thermophysical_props.py
======================
Thermophysical property correlations for pulverized coal combustion.

Incorporates Newton divided-difference interpolation (from interp_equal)
for high-accuracy evaluation of temperature-dependent properties:
- Specific heat capacity c_p(T) [J/(kg·K)]
- Thermal conductivity k(T) [W/(m·K)]
- Dynamic viscosity mu(T) [Pa·s]
- Mass diffusivity D(T) [m^2/s]

All properties use piecewise polynomial or spline-like interpolation
over tabulated data points with rigorous extrapolation guards.

Mathematical basis:
    Newton interpolation of order n-1 through n tabulated points:
        P(x) = d[0] + (x-x0)*(d[1] + (x-x1)*(d[2] + ...))
    where d[j] are divided differences:
        d[j] = f[x0, x1, ..., xj]

Divided differences recurrence:
    f[xi] = yi
    f[xi, ..., xj] = (f[x_{i+1},...,x_j] - f[x_i,...,x_{j-1}]) / (x_j - x_i)
"""

import numpy as np
from utils import safe_exp


# ======================================================================
# Tabulated data for major species (polynomial coefficients from
# NASA polynomials, valid 300-5000 K)
# ======================================================================

class ThermoTable:
    """Container for tabulated thermophysical data."""
    
    def __init__(self, T_points: np.ndarray, values: np.ndarray, name: str = ""):
        self.T = np.asarray(T_points, dtype=float)
        self.v = np.asarray(values, dtype=float)
        self.name = name
        self._divdif = None  # lazy compute
        self._sorted = False
        self._ensure_sorted()
    
    def _ensure_sorted(self):
        """Ensure T is strictly increasing."""
        if not self._sorted:
            idx = np.argsort(self.T)
            self.T = self.T[idx]
            self.v = self.v[idx]
            # Remove duplicates
            mask = np.concatenate(([True], np.diff(self.T) > 1e-12))
            self.T = self.T[mask]
            self.v = self.v[mask]
            self._sorted = True
    
    def _compute_divdif(self):
        """Compute divided difference table (upper triangular)."""
        n = len(self.T)
        if n == 0:
            self._divdif = np.array([])
            return
        d = np.zeros((n, n))
        d[:, 0] = self.v
        for j in range(1, n):
            for i in range(n - j):
                denom = self.T[i + j] - self.T[i]
                if abs(denom) < 1e-300:
                    denom = 1e-300
                d[i, j] = (d[i + 1, j - 1] - d[i, j - 1]) / denom
        self._divdif = d
    
    def eval_newton(self, T_query: float, order: int = -1) -> float:
        """
        Evaluate Newton interpolation polynomial at T_query.
        If order < 0, use all available points (degree n-1).
        Otherwise use order+1 nearest points for local interpolation.
        """
        if len(self.T) == 0:
            return 0.0
        if len(self.T) == 1:
            return self.v[0]
        
        # Find nearest interval
        if T_query <= self.T[0]:
            # Extrapolation guard: linear from first two points
            if len(self.T) >= 2:
                slope = (self.v[1] - self.v[0]) / (self.T[1] - self.T[0])
                return self.v[0] + slope * (T_query - self.T[0])
            return self.v[0]
        if T_query >= self.T[-1]:
            # Extrapolation guard: linear from last two points
            if len(self.T) >= 2:
                slope = (self.v[-1] - self.v[-2]) / (self.T[-1] - self.T[-2])
                return self.v[-1] + slope * (T_query - self.T[-1])
            return self.v[-1]
        
        # Select local stencil
        if order < 0:
            order = len(self.T) - 1
        else:
            order = min(order, len(self.T) - 1)
        
        npts = order + 1
        # Find center index
        idx = np.searchsorted(self.T, T_query)
        left = max(0, idx - npts // 2)
        right = min(len(self.T), left + npts)
        left = max(0, right - npts)
        
        Tloc = self.T[left:right]
        vloc = self.v[left:right]
        
        # Compute divided differences for local points
        m = len(Tloc)
        d = np.zeros(m)
        d[:] = vloc
        for j in range(1, m):
            for i in range(m - 1, j - 1, -1):
                denom = Tloc[i] - Tloc[i - j]
                if abs(denom) < 1e-300:
                    denom = 1e-300
                d[i] = (d[i] - d[i - 1]) / denom
        
        # Horner evaluation
        result = d[m - 1]
        for i in range(m - 2, -1, -1):
            result = result * (T_query - Tloc[i]) + d[i]
        return result
    
    def eval(self, T_query: float) -> float:
        """Default: cubic interpolation (order=3) for smoothness."""
        return self.eval_newton(T_query, order=3)


# ======================================================================
# Build tabulated thermophysical data for combustion gases
# Data from NIST Chemistry WebBook and combustion handbooks
# ======================================================================

def _build_cp_table_n2() -> ThermoTable:
    """Specific heat of N2 [J/(kg·K)] vs T [K]."""
    T = np.array([300, 400, 500, 600, 700, 800, 900, 1000,
                  1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    cp = np.array([1040.7, 1041.3, 1043.5, 1051.1, 1063.7, 1079.7, 1097.6,
                   1116.2, 1151.4, 1182.3, 1208.4, 1230.0, 1247.7, 1285.7,
                   1307.7, 1321.0, 1329.4, 1334.9, 1338.7])
    return ThermoTable(T, cp, "N2_cp")


def _build_cp_table_o2() -> ThermoTable:
    """Specific heat of O2 [J/(kg·K)] vs T [K]."""
    T = np.array([300, 400, 500, 600, 700, 800, 900, 1000,
                  1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    cp = np.array([920.0, 947.0, 978.0, 1007.0, 1033.0, 1056.0, 1076.0,
                   1093.0, 1122.0, 1145.0, 1163.0, 1177.0, 1189.0, 1212.0,
                   1227.0, 1238.0, 1246.0, 1252.0, 1257.0])
    return ThermoTable(T, cp, "O2_cp")


def _build_cp_table_co2() -> ThermoTable:
    """Specific heat of CO2 [J/(kg·K)] vs T [K]."""
    T = np.array([300, 400, 500, 600, 700, 800, 900, 1000,
                  1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    cp = np.array([845.0, 939.0, 1027.0, 1105.0, 1173.0, 1232.0, 1283.0,
                   1327.0, 1397.0, 1452.0, 1495.0, 1529.0, 1555.0, 1602.0,
                   1634.0, 1657.0, 1674.0, 1687.0, 1697.0])
    return ThermoTable(T, cp, "CO2_cp")


def _build_k_table_mix() -> ThermoTable:
    """Thermal conductivity of flue gas mixture [W/(m·K)] vs T [K].
    Approximated for typical combustion products (N2, CO2, H2O)."""
    T = np.array([300, 400, 500, 600, 700, 800, 900, 1000,
                  1200, 1400, 1600, 1800, 2000, 2500])
    k = np.array([0.026, 0.034, 0.042, 0.049, 0.056, 0.062, 0.068,
                  0.074, 0.084, 0.094, 0.103, 0.111, 0.119, 0.137])
    return ThermoTable(T, k, "mix_k")


def _build_mu_table_mix() -> ThermoTable:
    """Dynamic viscosity of flue gas mixture [Pa·s] vs T [K]."""
    T = np.array([300, 400, 500, 600, 700, 800, 900, 1000,
                  1200, 1400, 1600, 1800, 2000, 2500])
    mu = np.array([1.78e-5, 2.18e-5, 2.54e-5, 2.86e-5, 3.16e-5, 3.44e-5,
                   3.70e-5, 3.94e-5, 4.38e-5, 4.78e-5, 5.14e-5, 5.48e-5,
                   5.80e-5, 6.52e-5])
    return ThermoTable(T, mu, "mix_mu")


def _build_diffusivity_table() -> ThermoTable:
    """Binary diffusivity D_NO_N2 [m^2/s] vs T [K] at 1 atm.
    Using Chapman-Enskog theory: D_AB ~ T^{1.75} / P.
    """
    T = np.array([300, 400, 500, 600, 700, 800, 900, 1000,
                  1200, 1400, 1600, 1800, 2000, 2500])
    D = 1.6e-5 * (T / 300.0) ** 1.75 / 1.01325e5 * 1.01325e5
    return ThermoTable(T, D, "D_NO_N2")


# Global tables
_TABLE_N2_CP = _build_cp_table_n2()
_TABLE_O2_CP = _build_cp_table_o2()
_TABLE_CO2_CP = _build_cp_table_co2()
_TABLE_K_MIX = _build_k_table_mix()
_TABLE_MU_MIX = _build_mu_table_mix()
_TABLE_D_MIX = _build_diffusivity_table()


def cp_mixture(T: float, Y: dict) -> float:
    """
    Mixture specific heat [J/(kg·K)] by mass-fraction weighting:
        c_p,mix = sum_i (Y_i * c_p,i(T))
    
    Args:
        T: temperature [K]
        Y: dict of mass fractions, keys like 'N2', 'O2', 'CO2', 'H2O'
    
    Returns:
        c_p,mix [J/(kg·K)]
    """
    cp = 0.0
    weights = {
        'N2': _TABLE_N2_CP.eval(T),
        'O2': _TABLE_O2_CP.eval(T),
        'CO2': _TABLE_CO2_CP.eval(T),
    }
    # Default other species to N2 value
    for sp, yi in Y.items():
        if sp in weights:
            cp += yi * weights[sp]
        else:
            cp += yi * _TABLE_N2_CP.eval(T)
    return max(cp, 100.0)


def thermal_conductivity(T: float) -> float:
    """Thermal conductivity of combustion gas mixture [W/(m·K)]."""
    return max(_TABLE_K_MIX.eval(T), 1e-6)


def dynamic_viscosity(T: float) -> float:
    """Dynamic viscosity [Pa·s]."""
    return max(_TABLE_MU_MIX.eval(T), 1e-10)


def mass_diffusivity_NO(T: float, P: float = 101325.0) -> float:
    """
    Mass diffusivity of NO in N2 [m^2/s] at temperature T [K] and pressure P [Pa].
    Corrected from 1 atm reference using:
        D(T, P) = D_ref(T) * (P_ref / P)
    """
    D_ref = max(_TABLE_D_MIX.eval(T), 1e-12)
    return D_ref * (101325.0 / max(P, 1.0))


def mixture_density(T: float, P: float = 101325.0, MW_mix: float = 0.029) -> float:
    """
    Ideal gas mixture density [kg/m^3]:
        rho = P * MW_mix / (R * T)
    """
    R = 8.314462618
    if T <= 0.0:
        T = 300.0
    return P * MW_mix / (R * T)


def prandtl_number(T: float, cp: float = None) -> float:
    """
    Prandtl number:
        Pr = mu * c_p / k
    """
    mu = dynamic_viscosity(T)
    k = thermal_conductivity(T)
    if cp is None:
        cp = _TABLE_N2_CP.eval(T)
    if k < 1e-30:
        k = 1e-30
    return mu * cp / k


def lewis_number(T: float, P: float = 101325.0, cp: float = None) -> float:
    """
    Lewis number:
        Le = k / (rho * c_p * D) = alpha / D
    where alpha = k / (rho * c_p) is thermal diffusivity.
    """
    k = thermal_conductivity(T)
    rho = mixture_density(T, P)
    if cp is None:
        cp = _TABLE_N2_CP.eval(T)
    D = mass_diffusivity_NO(T, P)
    if rho * cp * D < 1e-300:
        return 1.0
    return k / (rho * cp * D)


def schmidt_number(T: float, P: float = 101325.0) -> float:
    """
    Schmidt number:
        Sc = mu / (rho * D)
    """
    mu = dynamic_viscosity(T)
    rho = mixture_density(T, P)
    D = mass_diffusivity_NO(T, P)
    if rho * D < 1e-300:
        return 1.0
    return mu / (rho * D)
