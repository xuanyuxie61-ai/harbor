"""
luhn_checksum_adapter.py
========================
数值校验和适配器（基于 Luhn 算法）。

融合种子项目：
  - 704_luhn : Luhn 校验和算法

科学应用：
  在分布式声学传感器网络和高性能并行计算中，数据完整性验证至关重要。
  本模块将 Luhn 算法的核心思想（交替加权求和与模运算）改造为：
  - 浮点数组校验和：验证大规模数值模拟结果在传输/存储中未被损坏
  - 网格拓扑一致性校验：验证三角形/六边形网格索引的数值一致性
  - 传感器数据链完整性验证

  Luhn 原始算法：
  1. 从右往左，偶数位数字翻倍，若结果 >=10 则减去 9
  2. 所有数字求和，对 10 取模，校验位使总和 ≡ 0 (mod 10)

  科学适配（浮点版本）：
  1. 将浮点数序列量化为整数序列（基于缩放因子）
  2. 应用 Luhn 加权模式
  3. 模运算验证一致性
"""

import numpy as np


def float_to_digits(value, scale=1e9, max_digits=15):
    """
    将浮点数转换为整数数字序列。

    Parameters
    ----------
    value : float
        输入浮点数。
    scale : float
        缩放因子。
    max_digits : int
        最大数字位数。

    Returns
    -------
    list of int
        数字序列。
    """
    if not np.isfinite(value):
        return [0]
    # 缩放并取整
    scaled = int(np.round(np.abs(value) * scale))
    if scaled == 0:
        return [0]
    digits = []
    while scaled > 0 and len(digits) < max_digits:
        digits.append(scaled % 10)
        scaled //= 10
    return digits[::-1]  # 从高到低


def luhn_checksum_digits(digits):
    """
    Luhn 校验和计算（数字列表版本）。

    原始逻辑来自 704_luhn/luhn_checksum.m。

    Parameters
    ----------
    digits : list or np.ndarray
        数字序列。

    Returns
    -------
    int
        校验和 (0-9)。
    """
    digits = np.asarray(digits, dtype=int)
    n = len(digits)
    if n == 0:
        return 0

    # 从末尾开始，每隔一个数字翻倍
    total = np.sum(digits[n % 2::2])  # 奇数位（从末尾数）
    for i in range(n % 2 == 0, n, 2):
        d2 = 2 * digits[i]
        total += d2 // 10 + d2 % 10

    return int(total % 10)


def compute_array_luhn(array, scale=1e9):
    """
    计算浮点数组的 Luhn 风格校验和。

    Parameters
    ----------
    array : np.ndarray
        输入数组。
    scale : float
        量化缩放因子。

    Returns
    -------
    int
        校验和 (0-9)。
    str
        校验位（为使总和为0 mod 10 需要追加的数字）。
    """
    array = np.asarray(array, dtype=float)
    flat = array.ravel()

    all_digits = []
    for val in flat:
        all_digits.extend(float_to_digits(val, scale))

    if len(all_digits) == 0:
        return 0, '0'

    checksum = luhn_checksum_digits(all_digits)
    # 计算使总和 ≡ 0 (mod 10) 的校验位
    check_digit = (10 - checksum) % 10
    return checksum, str(check_digit)


def verify_array_integrity(array, expected_check_digit, scale=1e9):
    """
    验证数组完整性。

    Parameters
    ----------
    array : np.ndarray
        待验证数组。
    expected_check_digit : str or int
        期望的校验位。
    scale : float
        缩放因子。

    Returns
    -------
    bool
        验证是否通过。
    """
    checksum, check_digit = compute_array_luhn(array, scale)
    return str(check_digit) == str(expected_check_digit)


def mesh_topology_checksum(triangles, nodes):
    """
    计算网格拓扑的校验和。

    将三角形索引和节点坐标编码为校验和，
    用于检测网格生成/传输过程中的索引错位。

    Parameters
    ----------
    triangles : np.ndarray, shape (M, 3)
        三角形索引。
    nodes : np.ndarray, shape (N, 2) or (N, 3)
        节点坐标。

    Returns
    -------
    int
        校验和。
    str
        校验位。
    """
    triangles = np.asarray(triangles, dtype=int)
    nodes = np.asarray(nodes, dtype=float)

    # 编码拓扑信息为校验和
    all_digits = []
    for tri in triangles.ravel():
        all_digits.extend(float_to_digits(float(tri), scale=1.0))
    for coord in nodes.ravel():
        all_digits.extend(float_to_digits(coord, scale=1e6))

    if len(all_digits) == 0:
        return 0, '0'

    checksum = luhn_checksum_digits(all_digits)
    check_digit = (10 - checksum) % 10
    return checksum, str(check_digit)


class NumericalIntegrityChecker:
    """
    数值模拟结果完整性检查器。
    """

    def __init__(self, scale=1e9):
        self.scale = float(scale)
        self._checkpoints = {}

    def checkpoint(self, name, array):
        """
        为数组建立校验检查点。

        Parameters
        ----------
        name : str
            检查点名称。
        array : np.ndarray
            数据数组。
        """
        checksum, check_digit = compute_array_luhn(array, self.scale)
        self._checkpoints[name] = {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'checksum': checksum,
            'check_digit': check_digit,
            'mean': float(np.mean(array)) if array.size > 0 else 0.0,
            'std': float(np.std(array)) if array.size > 0 else 0.0
        }

    def verify(self, name, array):
        """
        验证当前数组是否与检查点一致。

        Parameters
        ----------
        name : str
            检查点名称。
        array : np.ndarray
            当前数据。

        Returns
        -------
        bool
            是否通过验证。
        dict
            验证详情。
        """
        if name not in self._checkpoints:
            return False, {'error': 'Checkpoint not found'}

        cp = self._checkpoints[name]
        details = {
            'shape_match': array.shape == cp['shape'],
            'checksum_match': False,
            'statistical_match': False
        }

        _, check_digit = compute_array_luhn(array, self.scale)
        details['checksum_match'] = (check_digit == cp['check_digit'])

        if array.size > 0:
            mean_diff = abs(float(np.mean(array)) - cp['mean'])
            std_diff = abs(float(np.std(array)) - cp['std'])
            details['statistical_match'] = (mean_diff < 1e-6 * max(abs(cp['mean']), 1.0) and
                                            std_diff < 1e-6 * max(cp['std'], 1.0))

        passed = details['shape_match'] and details['checksum_match']
        return passed, details

    def summary(self):
        """
        返回所有检查点摘要。
        """
        return self._checkpoints.copy()
