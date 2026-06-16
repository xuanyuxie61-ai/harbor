"""
numerical_core.py - 数值完整性校验工具

本模块提供仿真过程中数据完整性验证的基础工具。

包含三个核心功能:

1. VIN 校验和算法 (源自 ISO 3779 标准):
   用于验证仿真参数向量的完整性。将参数编码为类似 VIN 的
   标识符，通过加权校验和检测参数是否被意外修改。

   校验和公式:
       S = sum(s_i * w_i) mod 11
   其中 w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2] 为权重向量,
   s_i 为参数的数值编码。

2. Atbash 替换密码 (源自古代替换加密):
   用于参数标签的对称编码/解码。在数据存储时对
   参数名称进行编码，防止配置文件被人工误修改。

   映射规则: A<->Z, B<->Y, ..., a<->z

3. 数值范围检查与边界验证:
   提供数组边界检查、NaN/Inf 检测、物理量合理性验证。

这些工具虽然简单，但在大规模仿真中对保证数值结果的
可追溯性和可复现性至关重要。
"""

import numpy as np


# ============================================================
# VIN 校验和算法 (ISO 3779)
# ============================================================

# VIN 字符到数值的映射表
# 数字 0-9 直接映射, 字母按 ISO 3779 标准转译
_VIN_TRANSLATION = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5,
    'F': 6, 'G': 7, 'H': 8,             'J': 1,
    'K': 2, 'L': 3, 'M': 4, 'N': 5,     'P': 6,
    'R': 9, 'S': 2, 'T': 3, 'U': 4, 'V': 5,
    'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
}

# ISO 3779 权重向量 (17 位 VIN)
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def _char_to_vin_num(ch):
    """将单个字符转换为 VIN 数值。

    映射规则:
        '0'-'9' -> 0-9
        'A'-'Z' -> ISO 3779 转译值 (I, O, Q 不使用)
        其他字符 -> 0

    Parameters
    ----------
    ch : str
        单个字符

    Returns
    -------
    int
        对应的 VIN 数值
    """
    ch = ch.upper()
    return _VIN_TRANSLATION.get(ch, 0)


def vin_checksum(parameter_string):
    """计算参数字符串的 VIN 校验和。

    校验和算法:
        1. 将参数字符串截取/填充到 17 个字符
        2. 将每个字符转换为数值 s_i
        3. 计算加权和: S = sum(s_i * w_i)
        4. 校验位: check = S mod 11
        5. 如果 check == 10, 用 'X' 表示

    此方法可用于仿真参数的快速完整性检查。

    Parameters
    ----------
    parameter_string : str
        待校验的参数标识字符串

    Returns
    -------
    checksum : int
        校验和值 (0-10)
    check_char : str
        校验位字符 ('0'-'9' 或 'X')
    """
    # 将字符串标准化为 17 位
    pstr = str(parameter_string).upper()
    if len(pstr) < 17:
        pstr = pstr + '0' * (17 - len(pstr))
    elif len(pstr) > 17:
        pstr = pstr[:17]

    # 计算加权和
    weighted_sum = 0
    for i, ch in enumerate(pstr):
        val = _char_to_vin_num(ch)
        weighted_sum += val * _VIN_WEIGHTS[i]

    checksum = weighted_sum % 11
    check_char = 'X' if checksum == 10 else str(checksum)

    return checksum, check_char


def validate_parameter_integrity(param_name, expected_checksum):
    """验证参数名的校验和是否与期望值匹配。

    Parameters
    ----------
    param_name : str
        参数名称
    expected_checksum : int
        期望的校验和值

    Returns
    -------
    valid : bool
        校验是否通过
    """
    actual, _ = vin_checksum(param_name)
    return actual == expected_checksum


# ============================================================
# Atbash 替换密码
# ============================================================

def atbash_encode(text):
    """Atbash 替换加密。

    Atbash 密码是一种古老的替换密码，其映射规则为字母表的
    首尾对称交换:
        A <-> Z,  B <-> Y,  C <-> X,  ..., M <-> N
        a <-> z,  b <-> y,  c <-> x,  ..., m <-> n

    数学表达:
        对于字母字符 c:
            f(c) = ord('A') + ord('Z') - ord(c)   (大写)
            f(c) = ord('a') + ord('z') - ord(c)   (小写)

    非字母字符保持不变。

    在仿真中用于参数标签的对称编码，编码和解码使用同一函数。

    Parameters
    ----------
    text : str
        输入文本

    Returns
    -------
    encoded : str
        编码/解码后的文本 (Atbash 是对合函数)
    """
    result = []
    small_a, small_z = ord('a'), ord('z')
    big_a, big_z = ord('A'), ord('Z')

    for ch in text:
        ival = ord(ch)
        if small_a <= ival <= small_z:
            jval = small_a + small_z - ival
            result.append(chr(jval))
        elif big_a <= ival <= big_z:
            jval = big_a + big_z - ival
            result.append(chr(jval))
        else:
            result.append(ch)

    return ''.join(result)


def atbash_decode(text):
    """Atbash 解码 (与编码相同, 因为 Atbash 是对合函数)。

    由于 Atbash 变换满足 f(f(x)) = x, 解码就是再次编码。

    Parameters
    ----------
    text : str
        编码的文本

    Returns
    -------
    decoded : str
        解码后的文本
    """
    return atbash_encode(text)


# ============================================================
# 数值范围检查与边界验证
# ============================================================

def check_array_finite(arr, name="array"):
    """检查数组是否全部为有限值 (无 NaN 或 Inf)。

    在时间步进仿真中，数值不稳定性可能导致解发散为 NaN 或 Inf。
    此函数用于在每步之后检测这种情况。

    Parameters
    ----------
    arr : ndarray
        待检查的数组
    name : str
        数组名称 (用于错误消息)

    Returns
    -------
    is_finite : bool
        是否全部有限
    info : str
        诊断信息
    """
    arr = np.asarray(arr)
    nan_count = np.count_nonzero(np.isnan(arr))
    inf_count = np.count_nonzero(np.isinf(arr))

    if nan_count == 0 and inf_count == 0:
        return True, f"{name}: 全部有限, shape={arr.shape}"
    else:
        info = f"{name}: 检测到 {nan_count} 个 NaN, {inf_count} 个 Inf, shape={arr.shape}"
        return False, info


def check_physical_bounds(value, lower, upper, name="quantity"):
    """检查物理量是否在合理范围内。

    Parameters
    ----------
    value : float or ndarray
        物理量的值
    lower : float
        下界
    upper : float
        上界
    name : str
        物理量名称

    Returns
    -------
    in_bounds : bool
        是否在范围内
    info : str
        诊断信息
    """
    value = np.asarray(value)
    violations_low = np.count_nonzero(value < lower)
    violations_high = np.count_nonzero(value > upper)

    if violations_low == 0 and violations_high == 0:
        return True, f"{name}: 范围 [{value.min():.6e}, {value.max():.6e}] 在 [{lower}, {upper}] 内"
    else:
        info = f"{name}: {violations_low} 个低于下界 {lower}, {violations_high} 个超过上界 {upper}"
        return False, info


def check_conservation(quantity, name="quantity", tol=1.0e-6):
    """检查守恒量的相对变化。

    在哈密顿系统中，能量、动量等守恒量应在数值误差范围内保持不变。
    此函数计算守恒量的相对漂移。

    相对漂移定义:
        delta_Q / Q_0 = |Q(t) - Q(0)| / |Q(0)|

    Parameters
    ----------
    quantity : ndarray
        守恒量的时间序列
    name : str
        守恒量名称
    tol : float
        容许的相对漂移

    Returns
    -------
    conserved : bool
        是否满足守恒
    relative_drift : float
        最大相对漂移
    info : str
        诊断信息
    """
    quantity = np.asarray(quantity)
    q0 = abs(quantity[0])
    if q0 < 1.0e-30:
        q0 = 1.0e-30

    drifts = np.abs(quantity - quantity[0]) / q0
    max_drift = np.max(drifts)

    if max_drift < tol:
        return True, max_drift, f"{name}: 守恒, 最大漂移 = {max_drift:.2e} < {tol:.2e}"
    else:
        return False, max_drift, f"{name}: 不守恒, 最大漂移 = {max_drift:.2e} > {tol:.2e}"


def compute_simulation_fingerprint(config):
    """生成仿真参数指纹 (用于可复现性追踪)。

    将仿真配置的关键参数编码为唯一标识符，通过 VIN 校验和
    和 Atbash 编码确保参数未被意外修改。

    Parameters
    ----------
    config : SimulationConfig
        仿真配置对象

    Returns
    -------
    fingerprint : dict
        包含编码参数名、校验和、编码标签的字典
    """
    params = {
        'Nx': str(config.N_x),
        'Nv': str(config.N_v),
        'a0': f"{config.a0:.4f}",
        'omega0': f"{config.omega_0:.4f}",
        'fdord': str(config.fd_order),
        'dgord': str(config.dg_order),
    }

    fingerprint = {}
    for key, val in params.items():
        encoded_key = atbash_encode(key)
        checksum, check_char = vin_checksum(val)
        fingerprint[key] = {
            'encoded_label': encoded_key,
            'value': val,
            'checksum': checksum,
            'check_char': check_char,
        }

    return fingerprint
