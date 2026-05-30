
import numpy as np
from typing import Tuple





def rosenbrock(x: np.ndarray) -> float:
    if len(x) < 2:
        raise ValueError("Rosenbrock requires at least 2 dimensions")
    x = np.asarray(x, dtype=float)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def himmelblau(x: np.ndarray) -> float:
    if len(x) != 2:
        raise ValueError("Himmelblau requires exactly 2 dimensions")
    x, y = x[0], x[1]
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def camel_back(x: np.ndarray) -> float:
    if len(x) != 2:
        raise ValueError("Camel back requires exactly 2 dimensions")
    x, y = x[0], x[1]
    return (2.0 * x ** 2 - 1.05 * x ** 4 + x ** 6 / 6.0 + x * y + y ** 2)


def butchers_polynomial(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Butcher requires at least 2 dimensions")
    val = 0.0
    for i in range(n - 1):
        val += (x[i] + x[i + 1] - 1.0) ** 2
    return val


def cyclic_n_polynomial(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("cyclic-n requires at least 2 dimensions")

    s1 = np.sum(x)
    s2 = np.sum(x ** 2)
    return (s1 - 1.0) ** 2 + (s2 - 1.0) ** 2






def binding_potential_surface(drug_conc: float, receptor_conc: float,
                               k_on: float, k_off: float,
                               cooperativity: int = 1) -> float:
    if drug_conc < 0 or receptor_conc < 0 or k_on < 0 or k_off < 0:
        raise ValueError("Concentrations and rates must be non-negative")
    K_d = k_off / max(k_on, 1e-20)
    ratio = drug_conc / max(K_d, 1e-20)

    if ratio < 1.0:
        hill = 0.0
        term = 1.0
        for n in range(1, 6):
            term *= -ratio ** cooperativity
            hill += term
        hill = -hill
    else:
        hill = 1.0 - 1.0 / (1.0 + ratio ** cooperativity)
    potential = k_on * drug_conc * receptor_conc - k_off * hill
    return potential


def enzyme_kinetics_polynomial_substrate(S: float, Vmax: float, Km: float,
                                          Ki: float = 1e10, I: float = 0.0) -> float:
    if S < 0 or Vmax < 0 or Km <= 0 or Ki <= 0 or I < 0:
        raise ValueError("Invalid kinetic parameters")
    Km_app = Km * (1.0 + I / Ki)


    v = Vmax * S / (Km_app + S)
    return v


def multi_target_objective(concentrations: np.ndarray,
                            target_eff: np.ndarray,
                            toxicity_weights: np.ndarray) -> float:
    C = np.asarray(concentrations, dtype=float)
    if np.any(C < 0):
        raise ValueError("Concentrations must be non-negative")
    if len(target_eff) != len(C) or len(toxicity_weights) != len(C):
        raise ValueError("Array lengths must match")
    efficacy = -np.sum(np.log1p(C / np.maximum(target_eff, 1e-20)))
    toxicity = np.sum(toxicity_weights * C ** 2)
    return efficacy + 0.1 * toxicity






def sobol_g_function(x: np.ndarray, a: np.ndarray = None) -> float:
    x = np.asarray(x, dtype=float)
    d = len(x)
    if a is None:
        a = np.linspace(0.0, 9.0, d)
    if len(a) != d:
        raise ValueError("a must have same length as x")
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("x must be in [0,1]")
    product = 1.0
    for i in range(d):
        product *= (abs(4.0 * x[i] - 2.0) + a[i]) / (1.0 + a[i])
    return product






if __name__ == "__main__":
    x = np.array([1.0, 1.0, 1.0])
    print(f"Rosenbrock at [1,1,1]: {rosenbrock(x):.6f}")
    print(f"Himmelblau at [3,2]: {himmelblau(np.array([3.0, 2.0])):.6f}")
    print(f"Camel back at [0,0]: {camel_back(np.array([0.0, 0.0])):.6f}")
    p = binding_potential_surface(1e-6, 1e-9, 1e5, 1e-3, 2)
    print(f"Binding potential: {p:.6e}")
    v = enzyme_kinetics_polynomial_substrate(5.0, 10.0, 2.0, Ki=1.0, I=0.5)
    print(f"Enzyme velocity: {v:.6f}")
    obj = multi_target_objective(np.array([1.0, 2.0, 0.5]),
                                  np.array([1.0, 1.0, 1.0]),
                                  np.array([0.1, 0.2, 0.5]))
    print(f"Multi-target objective: {obj:.6f}")
    print(f"Sobol G-function: {sobol_g_function(np.array([0.5, 0.5, 0.5])):.6f}")
