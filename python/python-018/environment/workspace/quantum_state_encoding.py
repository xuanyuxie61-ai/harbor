"""
quantum_state_encoding.py

基于种子项目 485_gray_code_display（Gray码与Hamming距离）、
1375_upc（校验位计算）和 1115_sphere_distance（球面距离统计），
实现拓扑量子计算中的量子态编码、纠错校验与保真度分析。

物理模型：
    1) Gray码编码：
        在拓扑量子计算中，由2N个马约拉纳费米子可以编码
        N-1个拓扑量子比特。每个量子比特的状态由一对
        马约拉纳算符的占据数决定：
            |0_L> = (1 + iγ_{2j-1}γ_{2j})/√2 |vac>
            |1_L> = (1 - iγ_{2j-1}γ_{2j})/√2 |vac>

        Gray码用于量子态的经典表示，确保相邻量子态的
        Hamming距离最小化，降低比特翻转错误的影响。

    2) 校验编码：
        类比UPC校验码，为拓扑量子比特引入奇偶校验：
            对于4个马约拉纳（2个量子比特），
            总费米宇称必须守恒：
                P = i^N γ_1 γ_2 ... γ_{2N} = ±1

    3) 布洛赫球距离：
        量子态的保真度与布洛赫球上的测地距离相关：
            F = |<ψ_1|ψ_2>|^2 = cos^2(θ/2)
            d = arccos(2F - 1)
        其中θ为布洛赫矢量夹角。
"""

import numpy as np
from typing import Tuple, List, Optional


class GrayCodeEncoder:
    """
    Gray码编码器（基于485_gray_code_display）。

    Gray码性质：相邻整数的二进制表示仅有一位不同。
    映射公式：
        G(n) = n ^ (n >> 1)
        n = 二进制反射Gray码
    """

    @staticmethod
    def binary_to_gray(n: int) -> int:
        """
        二进制到Gray码转换。
        """
        return n ^ (n >> 1)

    @staticmethod
    def gray_to_binary(g: int) -> int:
        """
        Gray码到二进制转换。
        """
        mask = g >> 1
        while mask != 0:
            g ^= mask
            mask >>= 1
        return g

    @staticmethod
    def hamming_distance(a: int, b: int, num_bits: int = 8) -> int:
        """
        计算两个整数的Hamming距离。
        """
        diff = a ^ b
        count = 0
        for _ in range(num_bits):
            count += diff & 1
            diff >>= 1
        return count

    @staticmethod
    def generate_gray_sequence(n_bits: int) -> List[int]:
        """
        生成n_bits位的Gray码序列。
        """
        return [GrayCodeEncoder.binary_to_gray(i)
                for i in range(1 << n_bits)]


class TopologicalParityCheck:
    """
    拓扑量子比特的奇偶校验（基于1375_upc校验思想）。

    对于由4个马约拉纳构成的单拓扑量子比特，
    两个可能的基态对应不同的联合费米宇称：
        |+> = (|00> + |11>)/√2   (偶宇称)
        |-> = (|01> + |10>)/√2   (奇宇称)

    校验通过测量所有马约拉纳对的联合宇称实现。
    """

    def __init__(self, num_majorana: int = 4):
        if num_majorana % 2 != 0 or num_majorana < 4:
            raise ValueError("马约拉纳数目必须为≥4的偶数")
        self.N = num_majorana
        self.num_qubits = num_majorana // 2 - 1

    def compute_stabilizer_eigenvalues(self,
                                        state_vector: np.ndarray) -> np.ndarray:
        """
        计算稳定子（stabilizer）的本征值。

        对于Kitaev链中的马约拉ana对，稳定子算符为：
            S_j = i γ_{2j-1} γ_{2j},  j=1,...,N/2

        每个稳定子满足 S_j^2 = 1，本征值为 ±1。
        """
        n_pairs = self.N // 2
        eigenvalues = np.zeros(n_pairs)

        # 简化：假设state_vector为配对占据数表示
        for j in range(n_pairs):
            idx1 = 2 * j
            idx2 = 2 * j + 1
            # 模拟稳定子期望值
            if idx2 < len(state_vector):
                ev = np.real(state_vector[idx1] * np.conj(state_vector[idx2]))
                eigenvalues[j] = np.sign(ev) if abs(ev) > 1e-10 else 0.0

        return eigenvalues

    def parity_check_matrix(self) -> np.ndarray:
        """
        构建奇偶校验矩阵H。

        行对应校验方程，列对应马约拉纳对。
        对于N个马约拉纳，总宇称约束为：
            Σ_j P_j = 0 (mod 2)
        其中P_j为第j对的宇称。
        """
        n_pairs = self.N // 2
        # 简化为单校验：所有对宇称之和为偶数
        H = np.ones((1, n_pairs))
        return H

    def syndrome(self, measured_parity: np.ndarray) -> int:
        """
        计算错误综合征。

        syndrome = H · measured_parity (mod 2)
        syndrome ≠ 0 表示存在错误。
        """
        H = self.parity_check_matrix()
        syn = int(np.dot(H[0], measured_parity) % 2)
        return syn

    def upc_style_check_digit(self, data_digits: np.ndarray) -> int:
        """
        基于UPC校验思想的校验位计算。

        原始UPC算法：
            d = (3*Σ_odd + Σ_even) mod 10
            check = (10 - d) mod 10

        修改为模2运算（适用于量子比特）。
        """
        if len(data_digits) < 1:
            return 0

        odd_sum = np.sum(data_digits[0::2])
        even_sum = np.sum(data_digits[1::2])

        # 模2版本
        check = (odd_sum + even_sum) % 2
        return int(check)


class BlochSphereFidelity:
    """
    布洛赫球上的量子态距离与保真度（基于1115_sphere_distance）。
    """

    @staticmethod
    def sample_unit_sphere(n_samples: int,
                            rng_seed: Optional[int] = None) -> np.ndarray:
        """
        在单位球面上均匀采样点。

        使用Marsaglia方法：
            1) 在单位圆盘内采样(x,y)
            2) z = ±sqrt(1 - x^2 - y^2)
        """
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
        """
        将单量子比特态转换为布洛赫矢量。

        对于 |ψ> = cos(θ/2)|0> + e^{iφ}sin(θ/2)|1>：
            r_x = sin(θ)cos(φ)
            r_y = sin(θ)sin(φ)
            r_z = cos(θ)
        """
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
        # 归一化到单位球
        vnorm = np.linalg.norm(vec)
        if vnorm > 1e-15:
            vec /= vnorm
        return vec

    @staticmethod
    def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
        """
        计算两个量子态的保真度。

        F = |<ψ_1|ψ_2>|^2
        """
        overlap = np.vdot(state1, state2)
        return float(np.abs(overlap) ** 2)

    @staticmethod
    def bloch_distance(state1: np.ndarray,
                        state2: np.ndarray) -> float:
        """
        计算布洛赫球上的测地距离。

        d = arccos(2F - 1) = arccos(<r_1, r_2>)
        """
        F = BlochSphereFidelity.fidelity(state1, state2)
        # 数值稳定性
        F = np.clip(F, 0.0, 1.0)
        r1 = BlochSphereFidelity.state_to_bloch_vector(state1)
        r2 = BlochSphereFidelity.state_to_bloch_vector(state2)
        dot = np.clip(np.dot(r1, r2), -1.0, 1.0)
        return float(np.arccos(dot))

    @staticmethod
    def average_fidelity_statistics(n_samples: int = 1000,
                                     rng_seed: Optional[int] = None
                                     ) -> Tuple[float, float]:
        """
        随机量子态对的平均保真度统计。
        """
        rng = np.random.RandomState(rng_seed)
        fidelities = []

        for _ in range(n_samples):
            # 随机单量子比特态
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
    """
    基于马约拉纳的拓扑量子编码方案。
    """

    def __init__(self, num_majorana: int = 6):
        self.N = num_majorana
        self.encoder = GrayCodeEncoder()
        self.parity = TopologicalParityCheck(num_majorana)
        self.fidelity = BlochSphereFidelity()

    def encode_logical_state(self, logical_state: int) -> np.ndarray:
        """
        将逻辑态编码为马约拉纳算符的联合宇称配置。

        对于N个马约拉纳，使用Gray码表示逻辑态，
        确保相邻逻辑态仅有一对马约拉纳的宇称不同。
        """
        gray = self.encoder.binary_to_gray(logical_state)
        num_pairs = self.N // 2

        # 将gray码的每一位映射到一对马约拉纳的宇称
        config = np.zeros(num_pairs)
        for i in range(min(num_pairs, 32)):
            config[i] = (gray >> i) & 1

        return config

    def decode_to_logical(self, parity_config: np.ndarray) -> int:
        """
        从宇称配置解码回逻辑态。
        """
        gray = 0
        for i, p in enumerate(parity_config):
            if int(round(p)) & 1:
                gray |= (1 << i)
        return self.encoder.gray_to_binary(gray)

    def error_detection_rate(self, error_probability: float,
                              num_trials: int = 1000) -> float:
        """
        估计错误检测率。

        模拟随机单比特翻转错误，计算能被稳定子检测到的比例。
        """
        rng = np.random.RandomState(42)
        detected = 0
        num_pairs = self.N // 2

        for _ in range(num_trials):
            # 初始正确配置
            config = np.zeros(num_pairs)
            # 引入随机错误
            for i in range(num_pairs):
                if rng.rand() < error_probability:
                    config[i] = 1 - config[i]

            syndrome = self.parity.syndrome(config)
            if syndrome != 0:
                detected += 1

        return detected / num_trials


def demo():
    """演示量子态编码。"""
    encoder = GrayCodeEncoder()
    gray_seq = encoder.generate_gray_sequence(3)
    print("3-bit Gray sequence:", gray_seq)

    # Hamming距离检查
    for i in range(len(gray_seq) - 1):
        d = encoder.hamming_distance(gray_seq[i], gray_seq[i + 1], 3)
        print(f"  G({i}) vs G({i+1}): Hamming distance = {d}")

    # 拓扑校验
    parity = TopologicalParityCheck(num_majorana=4)
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    ev = parity.compute_stabilizer_eigenvalues(state)
    print("Stabilizer eigenvalues:", ev)

    # 保真度
    psi0 = np.array([1.0, 0.0])
    psi1 = np.array([0.0, 1.0])
    F = BlochSphereFidelity.fidelity(psi0, psi1)
    d = BlochSphereFidelity.bloch_distance(psi0, psi1)
    print(f"Fidelity |0><1|: {F:.4f}, Bloch distance: {d:.4f}")

    mu, var = BlochSphereFidelity.average_fidelity_statistics(
        n_samples=500, rng_seed=42)
    print(f"Random state average fidelity: {mu:.4f} ± {np.sqrt(var):.4f}")

    # 编码
    code = MajoranaQuantumCode(num_majorana=6)
    for logical in range(4):
        config = code.encode_logical_state(logical)
        decoded = code.decode_to_logical(config)
        print(f"Logical {logical} -> config {config} -> decoded {decoded}")

    det_rate = code.error_detection_rate(error_probability=0.1, num_trials=500)
    print(f"Error detection rate at p=0.1: {det_rate:.4f}")


if __name__ == "__main__":
    demo()
