
import numpy as np
from typing import Tuple, List, Optional


class GrayCodeEncoder:

    @staticmethod
    def binary_to_gray(n: int) -> int:
        return n ^ (n >> 1)

    @staticmethod
    def gray_to_binary(g: int) -> int:
        mask = g >> 1
        while mask != 0:
            g ^= mask
            mask >>= 1
        return g

    @staticmethod
    def hamming_distance(a: int, b: int, num_bits: int = 8) -> int:
        diff = a ^ b
        count = 0
        for _ in range(num_bits):
            count += diff & 1
            diff >>= 1
        return count

    @staticmethod
    def generate_gray_sequence(n_bits: int) -> List[int]:
        return [GrayCodeEncoder.binary_to_gray(i)
                for i in range(1 << n_bits)]


class TopologicalParityCheck:

    def __init__(self, num_majorana: int = 4):
        if num_majorana % 2 != 0 or num_majorana < 4:
            raise ValueError("马约拉纳数目必须为≥4的偶数")
        self.N = num_majorana
        self.num_qubits = num_majorana // 2 - 1

    def compute_stabilizer_eigenvalues(self,
                                        state_vector: np.ndarray) -> np.ndarray:
        n_pairs = self.N // 2
        eigenvalues = np.zeros(n_pairs)


        for j in range(n_pairs):
            idx1 = 2 * j
            idx2 = 2 * j + 1

            if idx2 < len(state_vector):
                ev = np.real(state_vector[idx1] * np.conj(state_vector[idx2]))
                eigenvalues[j] = np.sign(ev) if abs(ev) > 1e-10 else 0.0

        return eigenvalues

    def parity_check_matrix(self) -> np.ndarray:
        n_pairs = self.N // 2

        H = np.ones((1, n_pairs))
        return H

    def syndrome(self, measured_parity: np.ndarray) -> int:
        H = self.parity_check_matrix()
        syn = int(np.dot(H[0], measured_parity) % 2)
        return syn

    def upc_style_check_digit(self, data_digits: np.ndarray) -> int:
        if len(data_digits) < 1:
            return 0

        odd_sum = np.sum(data_digits[0::2])
        even_sum = np.sum(data_digits[1::2])


        check = (odd_sum + even_sum) % 2
        return int(check)


class BlochSphereFidelity:

    @staticmethod
    def sample_unit_sphere(n_samples: int,
                            rng_seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.RandomState(rng_seed)
        points = np.zeros((n_samples, 3))

        for i in range(n_samples):
            while True:
                x, y = rng.uniform(-1.0, 1.0, 2)
                if x * x + y * y < 1.0:
                    break
            z = np.sqrt(max(1.0 - x * x - y * y, 0.0))
            if rng.rand() < 0.5:
                z = -z
            points[i] = [x, y, z]

        return points

    @staticmethod
    def state_to_bloch_vector(state: np.ndarray) -> np.ndarray:
        if len(state) != 2:
            raise ValueError("仅适用于单量子比特")

        a, b = state[0], state[1]
        norm = np.sqrt(np.abs(a) ** 2 + np.abs(b) ** 2)
        if norm < 1e-15:
            return np.array([0.0, 0.0, 1.0])
        a /= norm
        b /= norm

        rx = 2.0 * np.real(np.conj(a) * b)
        ry = 2.0 * np.imag(np.conj(a) * b)
        rz = np.abs(a) ** 2 - np.abs(b) ** 2

        vec = np.array([rx, ry, rz])

        vnorm = np.linalg.norm(vec)
        if vnorm > 1e-15:
            vec /= vnorm
        return vec

    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        overlap = np.vdot(state1, state2)
        return float(np.abs(overlap) ** 2)

    @staticmethod
    def bloch_distance(state1: np.ndarray,
                        state2: np.ndarray) -> float:
        F = BlochSphereFidelity.fidelity(state1, state2)

        F = np.clip(F, 0.0, 1.0)
        r1 = BlochSphereFidelity.state_to_bloch_vector(state1)
        r2 = BlochSphereFidelity.state_to_bloch_vector(state2)
        dot = np.clip(np.dot(r1, r2), -1.0, 1.0)
        return float(np.arccos(dot))

    @staticmethod
    def average_fidelity_statistics(n_samples: int = 1000,
                                     rng_seed: Optional[int] = None
                                     ) -> Tuple[float, float]:
        rng = np.random.RandomState(rng_seed)
        fidelities = []

        for _ in range(n_samples):

            theta = np.arccos(2.0 * rng.rand() - 1.0)
            phi = 2.0 * np.pi * rng.rand()
            psi1 = np.array([np.cos(theta / 2.0),
                             np.exp(1j * phi) * np.sin(theta / 2.0)])

            theta2 = np.arccos(2.0 * rng.rand() - 1.0)
            phi2 = 2.0 * np.pi * rng.rand()
            psi2 = np.array([np.cos(theta2 / 2.0),
                             np.exp(1j * phi2) * np.sin(theta2 / 2.0)])

            fidelities.append(BlochSphereFidelity.fidelity(psi1, psi2))

        f_arr = np.array(fidelities)
        return float(np.mean(f_arr)), float(np.var(f_arr))


class MajoranaQuantumCode:

    def __init__(self, num_majorana: int = 6):
        self.N = num_majorana
        self.encoder = GrayCodeEncoder()
        self.parity = TopologicalParityCheck(num_majorana)
        self.fidelity = BlochSphereFidelity()

    def encode_logical_state(self, logical_state: int) -> np.ndarray:
        gray = self.encoder.binary_to_gray(logical_state)
        num_pairs = self.N // 2


        config = np.zeros(num_pairs)
        for i in range(min(num_pairs, 32)):
            config[i] = (gray >> i) & 1

        return config

    def decode_to_logical(self, parity_config: np.ndarray) -> int:
        gray = 0
        for i, p in enumerate(parity_config):
            if int(round(p)) & 1:
                gray |= (1 << i)
        return self.encoder.gray_to_binary(gray)

    def error_detection_rate(self, error_probability: float,
                              num_trials: int = 1000) -> float:
        rng = np.random.RandomState(42)
        detected = 0
        num_pairs = self.N // 2

        for _ in range(num_trials):

            config = np.zeros(num_pairs)

            for i in range(num_pairs):
                if rng.rand() < error_probability:
                    config[i] = 1 - config[i]

            syndrome = self.parity.syndrome(config)
            if syndrome != 0:
                detected += 1

        return detected / num_trials


def demo():
    encoder = GrayCodeEncoder()
    gray_seq = encoder.generate_gray_sequence(3)
    print("3-bit Gray sequence:", gray_seq)


    for i in range(len(gray_seq) - 1):
        d = encoder.hamming_distance(gray_seq[i], gray_seq[i + 1], 3)
        print(f"  G({i}) vs G({i+1}): Hamming distance = {d}")


    parity = TopologicalParityCheck(num_majorana=4)
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    ev = parity.compute_stabilizer_eigenvalues(state)
    print("Stabilizer eigenvalues:", ev)


    psi0 = np.array([1.0, 0.0])
    psi1 = np.array([0.0, 1.0])
    F = BlochSphereFidelity.fidelity(psi0, psi1)
    d = BlochSphereFidelity.bloch_distance(psi0, psi1)
    print(f"Fidelity |0><1|: {F:.4f}, Bloch distance: {d:.4f}")

    mu, var = BlochSphereFidelity.average_fidelity_statistics(
        n_samples=500, rng_seed=42)
    print(f"Random state average fidelity: {mu:.4f} ± {np.sqrt(var):.4f}")


    code = MajoranaQuantumCode(num_majorana=6)
    for logical in range(4):
        config = code.encode_logical_state(logical)
        decoded = code.decode_to_logical(config)
        print(f"Logical {logical} -> config {config} -> decoded {decoded}")

    det_rate = code.error_detection_rate(error_probability=0.1, num_trials=500)
    print(f"Error detection rate at p=0.1: {det_rate:.4f}")


if __name__ == "__main__":
    demo()
