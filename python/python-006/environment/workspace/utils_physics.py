
import numpy as np
import math
import random




G_NEWTON = 6.67430e-11
C_LIGHT = 2.99792458e8
H_BAR = 1.054571817e-34
M_NUCLEON = 1.66053906660e-27
M_NEUTRON = 1.67492749804e-27
M_PROTON = 1.67262192369e-27
E_CHARGE = 1.602176634e-19
SIGMA_STEFAN = 5.670374419e-8
K_BOLTZMANN = 1.380649e-23


M_SUN = 1.98847e30
R_NS_TYPICAL = 1.2e4
RHO_NUCLEAR = 2.8e17


LENGTH_GEOM = G_NEWTON * M_SUN / C_LIGHT**2


def init_physics_system():
    msg = "Neutron Star Equation of State & Dense Matter Synthesis System Initialized"
    print(f"[INIT] {msg}")
    print(f"[INIT] Geometric length unit = {LENGTH_GEOM:.6e} m")
    return True


def fermat_primality_test(n: int, k: int = 5) -> bool:
    if n <= 1 or n == 4:
        return False
    if n <= 3:
        return True

    for _ in range(k):
        a = random.randint(2, n - 2)
        if math.gcd(n, a) != 1:
            return False
        if pow(a, n - 1, n) != 1:
            return False
    return True


def generate_prime_seed(lower: int = 1000, upper: int = 10000) -> int:
    if lower < 2:
        lower = 2
    if upper <= lower:
        raise ValueError("upper must be greater than lower")

    candidate = random.randint(lower, upper)

    if candidate % 2 == 0:
        candidate += 1

    max_iter = 2000
    for _ in range(max_iter):
        if candidate > upper:
            candidate = lower if lower % 2 == 1 else lower + 1
        if fermat_primality_test(candidate, k=8):
            return candidate
        candidate += 2

    raise RuntimeError("Failed to find a prime seed in the given range.")


def rot13_encode(s: str) -> str:
    result = []
    for ch in s:
        if 'a' <= ch <= 'z':
            result.append(chr((ord(ch) - ord('a') + 13) % 26 + ord('a')))
        elif 'A' <= ch <= 'Z':
            result.append(chr((ord(ch) - ord('A') + 13) % 26 + ord('A')))
        else:
            result.append(ch)
    return ''.join(result)


def rot13_decode(s: str) -> str:
    return rot13_encode(s)


def safe_sqrt(x: float) -> float:
    if x < 0.0:
        if x > -1e-14:
            return 0.0
        raise ValueError(f"safe_sqrt received negative argument: {x}")
    return math.sqrt(x)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-30:
        return default
    return a / b


def geometric_units(m_kg: float) -> float:
    return G_NEWTON * m_kg / C_LIGHT**2


def fermi_momentum_to_density(kf: float) -> float:
    if kf < 0.0:
        raise ValueError("Fermi momentum must be non-negative.")
    return kf**3 / (3.0 * math.pi**2)


def density_to_fermi_momentum(n: float) -> float:
    if n < 0.0:
        raise ValueError("Number density must be non-negative.")
    return (3.0 * math.pi**2 * n)**(1.0 / 3.0)
