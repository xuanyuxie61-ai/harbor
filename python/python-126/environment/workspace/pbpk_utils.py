"""
pbpk_utils.py
基于种子项目 704_luhn + 116_box_plot

实现数据完整性校验与离散化统计工具：
1. Luhn 校验算法（用于药物批号/患者 ID 校验）
2. 浓度分箱统计（binning statistics）
3. 数值鲁棒性工具（边界检查、安全除法、log-exp 稳定化）
4. 单位转换与生理常数

在 PBPK 模型中用于：
- 临床试验数据的完整性校验
- 药物浓度分布的统计分析
- 数值计算中的边界保护与单位一致性
"""

import numpy as np
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Luhn 校验（基于 704_luhn）
# ---------------------------------------------------------------------------

def luhn_checksum(digits: List[int]) -> int:
    """
    计算 Luhn 校验和。
    算法：
        1. 从右起，每隔一位乘以 2
        2. 若乘积 >= 10，则减去 9（即数字和）
        3. 求所有数字之和
        4. checksum = (10 - sum % 10) % 10
    """
    if not digits:
        return 0
    total = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d >= 10:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def luhn_is_valid(digits: List[int]) -> bool:
    """验证数字序列是否通过 Luhn 校验。"""
    if len(digits) < 2:
        return False
    check = luhn_checksum(digits[:-1])
    return check == digits[-1]


def luhn_check_digit_calculate(digits: List[int]) -> int:
    """计算应追加的 Luhn 校验位。"""
    return luhn_checksum(digits)


def validate_patient_id(patient_id_str: str) -> bool:
    """
    校验患者 ID 字符串（去除非数字字符后执行 Luhn 校验）。
    """
    digits = [int(c) for c in patient_id_str if c.isdigit()]
    return luhn_is_valid(digits)


# ---------------------------------------------------------------------------
# 浓度分箱统计（基于 116_box_plot 的离散化思想）
# ---------------------------------------------------------------------------

def concentration_binning(concentrations: np.ndarray,
                           bin_edges: np.ndarray = None,
                           n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将药物浓度数据分箱统计。
    返回：
        counts : 每箱计数
        bin_edges : 分箱边界
        bin_centers : 箱中心
    """
    C = np.asarray(concentrations, dtype=float)
    if len(C) == 0:
        raise ValueError("concentrations array is empty")
    if bin_edges is None:
        cmin, cmax = np.min(C), np.max(C)
        if cmin == cmax:
            cmax = cmin + 1.0
        bin_edges = np.linspace(cmin, cmax, n_bins + 1)
    counts, edges = np.histogram(C, bins=bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers


def summary_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    计算鲁棒的描述统计量（用于药代动力学参数报告）。
    """
    x = np.asarray(data, dtype=float)
    if len(x) == 0:
        raise ValueError("data is empty")
    q25, q50, q75 = np.percentile(x, [25.0, 50.0, 75.0])
    iqr = q75 - q25
    return {
        "n": float(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "cv": float(np.std(x, ddof=1) / max(abs(np.mean(x)), 1e-20)),
    }


# ---------------------------------------------------------------------------
# 数值鲁棒性工具
# ---------------------------------------------------------------------------

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零。"""
    if abs(b) < 1e-300:
        return default
    return a / b


def safe_log(x: float, min_val: float = 1e-300) -> float:
    """安全对数，防止 log(0) 或 log(负数)。"""
    return np.log(max(x, min_val))


def safe_exp(x: float, max_val: float = 700.0) -> float:
    """安全指数，防止 exp 溢出。"""
    if x > max_val:
        return np.exp(max_val)
    if x < -max_val:
        return 0.0
    return np.exp(x)


def softplus(x: float) -> float:
    """Softplus 函数：log(1+exp(x))，数值稳定的 ReLU 平滑版本。"""
    if x > 20.0:
        return x
    return np.log1p(np.exp(x))


def clip_concentration(C: float, C_min: float = 1e-12, C_max: float = 1e6) -> float:
    """将药物浓度裁剪到物理合理范围。"""
    return max(C_min, min(C, C_max))


# ---------------------------------------------------------------------------
# 单位转换与生理常数
# ---------------------------------------------------------------------------

def ml_to_L(x: float) -> float:
    return x / 1000.0


def L_to_ml(x: float) -> float:
    return x * 1000.0


def mg_to_ng(x: float) -> float:
    return x * 1e6


def ng_to_mg(x: float) -> float:
    return x / 1e6


def min_to_hr(x: float) -> float:
    return x / 60.0


def hr_to_min(x: float) -> float:
    return x * 60.0


PHYSIOLOGICAL_CONSTANTS = {
    "BLOOD_DENSITY": 1.055,           # kg/L
    "PLASMA_VOLUME_70KG": 3.0,        # L
    "CARDIAC_OUTPUT_REST": 5.0,       # L/min
    "HEPATIC_BLOOD_FLOW": 1.5,        # L/min
    "RENAL_BLOOD_FLOW": 1.2,          # L/min
    "GFR_STANDARD": 0.125,            # L/min (≈ 125 mL/min)
    "BODY_WEIGHT_STANDARD": 70.0,     # kg
    "AVOGADRO": 6.02214076e23,        # 1/mol
    "BOLTZMANN": 1.380649e-23,        # J/K
    "GAS_CONSTANT": 8.314462618,      # J/(mol·K)
    "TEMP_BODY": 310.15,              # K (37°C)
}


def scale_by_body_weight(value_70kg: float, actual_bw: float) -> float:
    """按体重缩放生理参数（异速生长缩放，指数 0.75）。"""
    if actual_bw <= 0:
        raise ValueError("Body weight must be positive")
    # TODO: Hole 2 - 实现基于异速生长定律(allometric scaling)的体重缩放公式
    # 标准 70kg 成年人的生理参数需要按体重进行幂律缩放
    raise NotImplementedError("Hole 2: Allometric scaling formula not implemented")


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    digits = [4, 5, 3, 2, 0, 1, 9, 4, 9, 6, 2]
    check = luhn_check_digit_calculate(digits)
    print(f"Luhn check digit: {check}")
    valid = luhn_is_valid(digits + [check])
    print(f"Luhn valid: {valid}")
    data = np.random.lognormal(0, 1, 1000)
    counts, edges, centers = concentration_binning(data, n_bins=10)
    print(f"Bin counts: {counts}")
    stats = summary_statistics(data)
    print(f"Summary: mean={stats['mean']:.3f}, median={stats['median']:.3f}, cv={stats['cv']:.3f}")
    print(f"Safe divide 5/0: {safe_divide(5.0, 0.0, default=999.0)}")
    print(f"Softplus(5): {softplus(5.0):.4f}")
    print(f"Scale Q by BW 50kg: {scale_by_body_weight(5.0, 50.0):.3f}")
