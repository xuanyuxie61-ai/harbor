
import numpy as np






def pi_spigot(n_digits: int) -> str:
    if n_digits < 1:
        return "3."
    pi_val = 0.0

    n_terms = max(n_digits * 5, 50)
    for k in range(n_terms):
        coeff = 1.0 / (16.0 ** k)
        pi_val += coeff * (4.0 / (8.0 * k + 1.0)
                           - 2.0 / (8.0 * k + 4.0)
                           - 1.0 / (8.0 * k + 5.0)
                           - 1.0 / (8.0 * k + 6.0))
    fmt = f"{{:.{n_digits}f}}"
    return fmt.format(pi_val)


def pi_high_precision() -> float:
    pi_str = pi_spigot(15)
    return float(pi_str)






def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(np.sqrt(n)) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True


def prime_sieve(limit: int) -> np.ndarray:
    if limit < 2:
        return np.array([], dtype=int)
    is_prime_arr = np.ones(limit + 1, dtype=bool)
    is_prime_arr[0:2] = False
    for p in range(2, int(np.sqrt(limit)) + 1):
        if is_prime_arr[p]:
            is_prime_arr[p * p:limit + 1:p] = False
    return np.nonzero(is_prime_arr)[0]


def bifurcation_prime_level(level: int) -> bool:
    return is_prime(level)






class FEMMesh:
    def __init__(self, nodes: np.ndarray = None, elements: np.ndarray = None):
        self.nodes = nodes if nodes is not None else np.zeros((0, 2))
        self.elements = elements if elements is not None else np.zeros((0, 3), dtype=int)

    def node_count(self) -> int:
        return self.nodes.shape[0]

    def element_count(self) -> int:
        return self.elements.shape[0]

    def element_area(self, elem_idx: int) -> float:
        idx = self.elements[elem_idx]
        p1, p2, p3 = self.nodes[idx[0]], self.nodes[idx[1]], self.nodes[idx[2]]
        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) +
                         p2[0] * (p3[1] - p1[1]) +
                         p3[0] * (p1[1] - p2[1]))
        return area

    def total_area(self) -> float:
        return sum(self.element_area(i) for i in range(self.element_count()))

    def scale_to_area(self, target_area: float):
        current = self.total_area()
        if current <= 0:
            return
        s = np.sqrt(target_area / current)
        self.nodes *= s






def circular_cross_section_area(radius: float) -> float:
    PI = pi_high_precision()
    return PI * radius * radius


def circular_cross_section_perimeter(radius: float) -> float:
    PI = pi_high_precision()
    return 2.0 * PI * radius


def womersley_number(radius: float, kinematic_viscosity: float,
                     angular_frequency: float) -> float:
    if kinematic_viscosity <= 0 or radius <= 0 or angular_frequency <= 0:
        raise ValueError("Womersley number parameters must be positive.")
    return radius * np.sqrt(angular_frequency / kinematic_viscosity)


def reynolds_number(mean_velocity: float, diameter: float,
                    kinematic_viscosity: float) -> float:
    if kinematic_viscosity <= 0 or diameter <= 0:
        raise ValueError("Reynolds number parameters must be positive.")
    return mean_velocity * diameter / kinematic_viscosity


def murray_law_radius(r_parent: float, n_children: int,
                      bifurcation_angle_deg: float = 60.0) -> float:
    if r_parent <= 0 or n_children < 1:
        raise ValueError("Invalid Murray law parameters.")
    if not (0 < bifurcation_angle_deg < 180):
        raise ValueError("Bifurcation angle must be in (0, 180) degrees.")
    return r_parent / (n_children ** (1.0 / 3.0))






def safe_sqrt(x: float) -> float:
    if x < 0:
        if x > -1e-12:
            return 0.0
        raise ValueError(f"safe_sqrt received negative value: {x}")
    return np.sqrt(x)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-14:
        return default
    return a / b
