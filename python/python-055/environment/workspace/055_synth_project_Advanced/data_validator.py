
import numpy as np


class SonarDataValidator:


    MIN_DEPTH = 10.0
    MAX_DEPTH = 11000.0
    MIN_SOUND_SPEED = 1400.0
    MAX_SOUND_SPEED = 1600.0
    MIN_ANGLE = -75.0
    MAX_ANGLE = 75.0
    MAX_TIMESTAMP_JUMP = 10.0


    _WEIGHTS = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71
    ], dtype=np.float64)

    def __init__(self):
        self._last_timestamp = None
        self._error_log = []

    @staticmethod
    def compute_checksum(data_bytes: bytes) -> int:
        if not isinstance(data_bytes, bytes):
            raise TypeError("输入必须是 bytes 类型")
        if len(data_bytes) == 0:
            return 0

        total = 0.0
        weights = SonarDataValidator._WEIGHTS
        nw = len(weights)
        for i, byte in enumerate(data_bytes):

            total += weights[i % nw] * float(byte)
        checksum = int(total) % 89
        return checksum

    def validate_packet(self, packet: dict) -> bool:
        if not isinstance(packet, dict):
            self._error_log.append("数据包必须为字典类型")
            return False

        required_keys = {'timestamp', 'depth', 'angle', 'sound_speed', 'checksum', 'payload'}
        if not required_keys.issubset(packet.keys()):
            missing = required_keys - set(packet.keys())
            self._error_log.append(f"数据包缺失字段: {missing}")
            return False


        expected = self.compute_checksum(packet['payload'])
        if expected != packet['checksum']:
            self._error_log.append(
                f"校验和不匹配: 期望 {expected}, 实际 {packet['checksum']}"
            )
            return False


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
        mask = np.zeros(len(packets), dtype=bool)
        for i, pkt in enumerate(packets):
            mask[i] = self.validate_packet(pkt)
        return mask

    def get_error_log(self) -> list:
        return self._error_log.copy()

    def reset(self):
        self._last_timestamp = None
        self._error_log.clear()


def generate_test_packets(n: int = 20, seed: int = 55) -> list:
    rng = np.random.default_rng(seed)
    packets = []
    base_time = 1700000000.0

    for i in range(n):

        depth = rng.uniform(100.0, 4000.0)
        angle = rng.uniform(-60.0, 60.0)
        ss = 1480.0 + 20.0 * np.sin(depth / 1000.0) + rng.normal(0.0, 2.0)
        ss = float(np.clip(ss, 1450.0, 1550.0))
        timestamp = base_time + i * 0.5 + rng.normal(0.0, 0.05)


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
