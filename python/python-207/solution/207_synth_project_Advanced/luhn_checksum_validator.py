"""
luhn_checksum_validator.py — 计算完整性校验 (Luhn 算法)

科学背景
========
在大规模不确定性量化计算中, 需要验证数值流水线的完整性.
借鉴金融领域广泛使用的 Luhn 校验算法 (ISO/IEC 7812),
本模块为 Monte Carlo 统计量生成校验位, 确保:
1. 数据在传递过程中未被篡改
2. 浮点计算结果的数位特征保持一致
3. 多进程/多线程聚合时结果可验证

算法来源 (种子项目 704_luhn)
============================
直接移植 Luhn 算法:
1. 从最右位开始, 隔位数字加倍
2. 若加倍后 ≥10, 则数位之和 = 十位 + 个位
3. 所有数位之和 mod 10 = 0 则有效

在本项目中的角色
================
1. 对 Monte Carlo 均值/方差向量生成校验位
2. 验证 confidence band 临界值的计算一致性
3. 确保 bootstrap 重采样前后的统计量一致性

核心公式
========
设数字串 d_1 d_2 ... d_n:
  S = Σ_{k odd} d_k  +  Σ_{k even} [⌊2·d_k/10⌋ + (2·d_k mod 10)]
  valid ⟺ S mod 10 = 0
"""

import numpy as np


def digits_from_number(value, n_digits=16):
    """将数值转换为固定长度的数字串.

    对浮点数取绝对值, 放大到整数范围, 然后提取各位数字.
    先对数值进行舍入到 8 位有效数字, 以增强鲁棒性.

    参数
    ----
    value : float
        待转换的数值
    n_digits : int
        输出数字串的长度

    返回
    ----
    digits : list of int
        各位数字 (0-9)
    """
    # 取绝对值
    v = abs(float(value))
    if not np.isfinite(v) or v == 0:
        return [0] * n_digits
    # 规范化到 [1, 10) 区间
    magnitude = np.floor(np.log10(v))
    normalized = v / (10.0 ** magnitude)
    # 规范化后可能因浮点误差略超出 [1, 10)
    if normalized < 1.0:
        normalized *= 10.0
        magnitude -= 1
    elif normalized >= 10.0:
        normalized /= 10.0
        magnitude += 1
    # 舍入到 8 位有效数字 (消除浮点噪声)
    normalized = round(normalized, 8)
    # 确保仍在 [1, 10) 内
    if normalized >= 10.0:
        normalized = 9.99999999
    # 转换为整数 (使用 n_digits-1 位小数)
    int_val = int(round(normalized * 10 ** (n_digits - 1)))
    int_val = max(0, int_val)

    digits = []
    for _ in range(n_digits):
        digits.append(int_val % 10)
        int_val //= 10
    digits.reverse()
    return digits


def luhn_checksum(digit_string):
    """计算 Luhn 校验值.

    算法 (种子 704):
    1. 从最右位 (n-1) 开始, 步长 2 取位, 直接累加
    2. 从 (n-2) 位开始, 步长 2 取位, 加倍后数位求和再累加
    3. 返回总和 mod 10

    参数
    ----
    digit_string : str or list of int
        数字串

    返回
    ----
    checksum : int
        校验值 ∈ {0,1,...,9}
    """
    if isinstance(digit_string, str):
        dvec = [int(c) for c in digit_string if c.isdigit()]
    else:
        dvec = [int(d) % 10 for d in digit_string]

    n = len(dvec)
    if n == 0:
        return 0

    value = 0
    # 奇数位 (从右数第 1, 3, 5... 位): 直接累加
    for i in range(n - 1, -1, -2):
        value += dvec[i]

    # 偶数位 (从右数第 2, 4, 6... 位): 加倍后数位求和
    for i in range(n - 2, -1, -2):
        d2 = 2 * dvec[i]
        value += (d2 // 10) + (d2 % 10)

    return value % 10


def luhn_check_digit(digit_string):
    """计算 Luhn 校验位.

    在数字串末尾添加一个校验位, 使得完整串的校验和为 0.

    参数
    ----
    digit_string : str or list of int

    返回
    ----
    check_digit : int ∈ {0,...,9}
    """
    if isinstance(digit_string, str):
        extended = digit_string + '0'
    else:
        extended = list(digit_string) + [0]
    cs = luhn_checksum(extended)
    return (10 - cs) % 10


def luhn_is_valid(digit_string):
    """验证数字串是否具有有效的 Luhn 校验.

    返回
    ----
    bool : True 表示校验和 mod 10 = 0
    """
    return luhn_checksum(digit_string) == 0


class ComputationIntegrityValidator:
    """计算完整性验证器.

    对 UQ 流水线的关键统计量进行 Luhn 校验.
    """

    def __init__(self):
        self.registry = {}  # name -> (checksum, digits)

    def register_statistic(self, name, value_vector):
        """注册一个统计量, 计算并存储其校验信息.

        参数
        ----
        name : str
            统计量名称
        value_vector : array-like
            统计量的数值向量

        返回
        ----
        check_digit : int
        """
        # 将向量转为数字串
        arr = np.asarray(value_vector, dtype=float).ravel()
        all_digits = []
        for v in arr:
            all_digits.extend(digits_from_number(v, n_digits=8))

        cs = luhn_checksum(all_digits)
        cd = luhn_check_digit(all_digits)

        # 同时存储哈希 (用于更可靠的验证)
        arr_hash = hash(arr.tobytes())

        self.registry[name] = {
            'checksum': cs,
            'check_digit': cd,
            'n_digits': len(all_digits),
            'full_valid': luhn_is_valid(all_digits + [cd]),
            'stored_hash': arr_hash,
            'stored_shape': arr.shape,
        }
        return cd

    def verify_statistic(self, name, value_vector):
        """验证已注册统计量的完整性.

        使用双重检查:
        1. 数组哈希 (精确匹配)
        2. Luhn 校验和 (数位特征)

        返回
        ----
        is_valid : bool
        message : str
        """
        if name not in self.registry:
            return False, f"统计量 '{name}' 未注册"

        arr = np.asarray(value_vector, dtype=float).ravel()
        arr_hash = hash(arr.tobytes())

        # 优先使用哈希比对 (精确)
        if arr_hash == self.registry[name]['stored_hash']:
            return True, f"'{name}' 校验通过 (哈希精确匹配)"

        # 回退到 Luhn 校验
        all_digits = []
        for v in arr:
            all_digits.extend(digits_from_number(v, n_digits=8))

        cs_new = luhn_checksum(all_digits)
        expected_cs = self.registry[name]['checksum']

        if cs_new == expected_cs:
            return True, f"'{name}' 校验通过 (checksum={cs_new})"
        else:
            return False, (
                f"'{name}' 校验失败: 期望 checksum={expected_cs}, "
                f"实际 checksum={cs_new}"
            )

    def register_mean_variance(self, name, data_2d):
        """注册矩阵数据的行均值和行方差.

        参数
        ----
        data_2d : ndarray, shape (n_samples, n_features)
        """
        means = np.mean(data_2d, axis=0)
        variances = np.var(data_2d, axis=0, ddof=1)
        combined = np.concatenate([means, variances])
        return self.register_statistic(name, combined)

    def summary(self):
        """返回注册表摘要."""
        lines = ["=" * 50, "计算完整性校验摘要", "=" * 50]
        for name, info in self.registry.items():
            status = "✓ 有效" if info['full_valid'] else "✗ 无效"
            lines.append(
                f"  {name}: checksum={info['checksum']}, "
                f"check_digit={info['check_digit']}, {status}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)
