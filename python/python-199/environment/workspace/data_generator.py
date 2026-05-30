
import math
import random
from typing import List, Tuple


def circle_monte_carlo_integrand(theta: float, e1: int, e2: int) -> float:
    return (math.cos(theta) ** e1) * (math.sin(theta) ** e2)


def exact_circle_integral(e1: int, e2: int) -> float:
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0


    def gamma_half(k: int) -> float:



        n = int(k - 0.5)
        if n < 0:
            return math.sqrt(math.pi)
        result = math.sqrt(math.pi)
        for i in range(1, n + 1):
            result *= (i - 0.5)
        return result

    k1 = (e1 + 1) / 2.0
    k2 = (e2 + 1) / 2.0
    k12 = (e1 + e2 + 2) / 2.0
    return 2.0 * gamma_half(k1) * gamma_half(k2) / gamma_half(k12)


def generate_monte_carlo_keys(n: int, seed: int = 42) -> List[float]:
    random.seed(seed)
    keys = []
    for _ in range(n):
        theta = random.uniform(0.0, 2.0 * math.pi)
        val = circle_monte_carlo_integrand(theta, 2, 2)

        noise = random.gauss(0.0, 0.001)
        keys.append(max(val + noise, 0.0))
    return keys


class SIRDataFlow:

    def __init__(self, alpha: float, beta: float, gamma: float, N: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.N = N

    def rhs(self, t: float, y: List[float]) -> List[float]:



        raise NotImplementedError("Hole_1: 请实现 SIR ODE 的 rhs 定义")

    def simulate_euler(self, S0: float, I0: float, R0: float,
                       t_end: float, n_steps: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        h = t_end / n_steps
        t_vals = [i * h for i in range(n_steps + 1)]
        S, I, R = S0, I0, R0
        S_vals, I_vals, R_vals = [S0], [I0], [R0]
        for i in range(n_steps):
            t = t_vals[i]
            dS, dI, dR = self.rhs(t, [S, I, R])
            S += h * dS
            I += h * dI
            R += h * dR

            S = max(S, 0.0)
            I = max(I, 0.0)
            R = max(R, 0.0)
            total = S + I + R
            if total > 1e-15 and abs(total - self.N) > 1e-6:
                scale = self.N / total
                S *= scale
                I *= scale
                R *= scale
            S_vals.append(S)
            I_vals.append(I)
            R_vals.append(R)
        return t_vals, S_vals, I_vals, R_vals

    def basic_reproduction_number(self) -> float:
        if self.beta < 1e-15:
            return float('inf')
        return self.alpha / self.beta


def logistic_chaotic_sequence(n: int, r: float, x0: float = 0.5) -> List[float]:
    seq = [x0]
    x = x0
    for _ in range(n - 1):
        x = r * x * (1.0 - x)

        if x <= 0.0:
            x = 1e-12
        elif x >= 1.0:
            x = 1.0 - 1e-12
        seq.append(x)
    return seq


def generate_heterogeneous_dataset(
    total_records: int,
    memory_limit: int,
    seed: int = 199
) -> List[Tuple[float, ...]]:
    random.seed(seed)


    sir = SIRDataFlow(alpha=0.3, beta=0.1, gamma=0.05, N=1000.0)
    _, S_vals, I_vals, _ = sir.simulate_euler(
        S0=990.0, I0=10.0, R0=0.0, t_end=100.0, n_steps=total_records
    )


    logistic_seq = logistic_chaotic_sequence(total_records, r=3.9, x0=0.314159)


    mc_keys = generate_monte_carlo_keys(total_records, seed=seed + 1)


    omega = 2.0 * math.pi / 50.0
    energy_base = 100.0

    dataset = []
    for i in range(total_records):

        timestamp = 1000.0 * i / total_records + 50.0 * math.sin(0.1 * i) + 10.0 * I_vals[i] / sir.N


        px = mc_keys[i] * math.cos(2.0 * math.pi * logistic_seq[i])
        py = mc_keys[i] * math.sin(2.0 * math.pi * logistic_seq[i])
        pz = logistic_seq[i] * 100.0


        energy = energy_base + 30.0 * math.sin(omega * i) + 5.0 * random.gauss(0.0, 1.0)
        energy = max(energy, 0.0)


        composite_key = timestamp + 0.01 * energy + 0.0001 * random.gauss(0.0, 1.0)

        record = (composite_key, timestamp, px, py, pz, energy, logistic_seq[i])
        dataset.append(record)

    return dataset
