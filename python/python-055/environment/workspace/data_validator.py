"""
data_validator.py
基于种子项目 1393_vin 的校验和算法与 1426_xyzl_display 的数据结构思想，
构建多波束声纳数据包完整性校验与质量控制模块。

科学背景：在海洋声纳探测中，原始数据包在传输与存储过程中可能因电磁干扰、
机械振动或水声信道多径效应而产生比特错误。本模块采用加权校验和与
一致性边界检测，确保用于海底地形反演的测深数据具有足够的可信度。
"""

import numpy as np


class SonarDataValidator:
    """
    多波束声纳数据包验证器。

    对测深数据包实施多层校验：
    1. 加权模校验和（Weighted Checksum）——源自 VIN 校验思想；
    2. 物理边界一致性检测——深度、角度、声速必须在合理海洋物理范围内；
    3. 时间序列连续性检测——避免时间戳跳变导致的反演伪影。
    """

    # 海洋物理边界常量
    MIN_DEPTH = 10.0          # 米，浅水最小可信深度
    MAX_DEPTH = 11000.0       # 米，马里亚纳海沟最大深度
    MIN_SOUND_SPEED = 1400.0  # m/s，极地海水声速下限
    MAX_SOUND_SPEED = 1600.0  # m/s，热带高盐海水声速上限
    MIN_ANGLE = -75.0         # 度，多波束最外侧开角下限
    MAX_ANGLE = 75.0          # 度，多波束最外侧开角上限
    MAX_TIMESTAMP_JUMP = 10.0 # 秒，相邻数据包最大允许时间间隔

    # 校验和权重向量（质数权重，降低碰撞概率）
    _WEIGHTS = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71
    ], dtype=np.float64)

    def __init__(self):
        self._last_timestamp = None
        self._error_log = []

    @staticmethod
    def compute_checksum(data_bytes: bytes) -> int:
        """
        计算字节流的加权模 89 校验和。

        公式:
            S = (Σ_{i=1}^{n} w_{i mod 20} · b_i) mod 89
        其中 w_j 为质数权重向量，b_i 为第 i 个字节的整数值。

        参数:
            data_bytes: 输入字节序列
        返回:
            校验和整数值 (0 <= S < 89)
        """
        if not isinstance(data_bytes, bytes):
            raise TypeError("输入必须是 bytes 类型")
        if len(data_bytes) == 0:
            return 0

        total = 0.0
        weights = SonarDataValidator._WEIGHTS
        nw = len(weights)
        for i, byte in enumerate(data_bytes):
            # byte 在 Python3 中已经是 int (0-255)
            total += weights[i % nw] * float(byte)
        checksum = int(total) % 89
        return checksum

    def validate_packet(self, packet: dict) -> bool:
        """
        验证单个声纳数据包的完整性与物理合理性。

        参数:
            packet: 字典，必须包含以下键：
                - 'timestamp': 时间戳（秒，Unix epoch）
                - 'depth': 测深值（米）
                - 'angle': 波束入射角（度）
                - 'sound_speed': 当地声速（m/s）
                - 'checksum': 数据校验和（整数）
                - 'payload': 原始字节载荷（bytes）
        返回:
            True 如果全部校验通过，否则 False
        """
        if not isinstance(packet, dict):
            self._error_log.append("数据包必须为字典类型")
            return False

        required_keys = {'timestamp', 'depth', 'angle', 'sound_speed', 'checksum', 'payload'}
        if not required_keys.issubset(packet.keys()):
            missing = required_keys - set(packet.keys())
            self._error_log.append(f"数据包缺失字段: {missing}")
            return False

        # 1. 校验和验证
        expected = self.compute_checksum(packet['payload'])
        if expected != packet['checksum']:
            self._error_log.append(
                f"校验和不匹配: 期望 {expected}, 实际 {packet['checksum']}"
            )
            return False

        # 2. 物理边界检测
        depth = float(packet['depth'])
        angle = float(packet['angle'])
        ss = float(packet['sound_speed'])

        if not (self.MIN_DEPTH <= depth <= self.MAX_DEPTH):
            self._error_log.append(
                f"深度越界: {depth} m (允许范围 [{self.MIN_DEPTH}, {self.MAX_DEPTH}])"
            )
            return False

        if not (self.MIN_ANGLE <= angle <= self.MAX_ANGLE):
            self._error_log.append(
                f"角度越界: {angle}° (允许范围 [{self.MIN_ANGLE}, {self.MAX_ANGLE}])"
            )
            return False

        if not (self.MIN_SOUND_SPEED <= ss <= self.MAX_SOUND_SPEED):
            self._error_log.append(
                f"声速越界: {ss} m/s (允许范围 [{self.MIN_SOUND_SPEED}, {self.MAX_SOUND_SPEED}])"
            )
            return False

        # 3. 时间连续性检测
        ts = float(packet['timestamp'])
        if self._last_timestamp is not None:
            dt = abs(ts - self._last_timestamp)
            if dt > self.MAX_TIMESTAMP_JUMP:
                self._error_log.append(
                    f"时间戳跳变过大: Δt = {dt:.2f} s > {self.MAX_TIMESTAMP_JUMP} s"
                )
                return False
        self._last_timestamp = ts

        return True

    def validate_batch(self, packets: list) -> np.ndarray:
        """
        批量验证数据包，返回布尔掩码。

        参数:
            packets: 数据包字典列表
        返回:
            一维布尔数组，True 表示对应数据包有效
        """
        mask = np.zeros(len(packets), dtype=bool)
        for i, pkt in enumerate(packets):
            mask[i] = self.validate_packet(pkt)
        return mask

    def get_error_log(self) -> list:
        """返回错误日志列表。"""
        return self._error_log.copy()

    def reset(self):
        """重置验证器状态。"""
        self._last_timestamp = None
        self._error_log.clear()


def generate_test_packets(n: int = 20, seed: int = 55) -> list:
    """
    生成用于测试的模拟声纳数据包。

    参数:
        n: 数据包数量
        seed: 随机种子
    返回:
        数据包字典列表
    """
    rng = np.random.default_rng(seed)
    packets = []
    base_time = 1700000000.0

    for i in range(n):
        # 模拟真实海洋剖面：深度 100-4000m，角度 -60~60°，声速 1450-1550 m/s
        depth = rng.uniform(100.0, 4000.0)
        angle = rng.uniform(-60.0, 60.0)
        ss = 1480.0 + 20.0 * np.sin(depth / 1000.0) + rng.normal(0.0, 2.0)
        ss = float(np.clip(ss, 1450.0, 1550.0))
        timestamp = base_time + i * 0.5 + rng.normal(0.0, 0.05)

        # 构造原始载荷字节
        payload_str = f"{timestamp:.6f},{depth:.3f},{angle:.4f},{ss:.2f}"
        payload = payload_str.encode('utf-8')
        checksum = SonarDataValidator.compute_checksum(payload)

        packet = {
            'timestamp': timestamp,
            'depth': depth,
            'angle': angle,
            'sound_speed': ss,
            'checksum': checksum,
            'payload': payload,
        }
        packets.append(packet)

    return packets
