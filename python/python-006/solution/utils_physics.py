"""
utils_physics.py
物理常数、基础工具与辅助算法模块

原项目映射:
- 912_prime_fermat  -> 费马素性检验，用于生成伪随机物理初始条件的种子
- 1045_rot13        -> 字符串旋转编码，用于结果摘要的混淆输出
- 513_hello         -> 系统初始化问候
"""

import numpy as np
import math
import random

# =============================================================================
# 天体物理基本常数（国际单位制，部分以 cgs 形式给出以方便核物理计算）
# =============================================================================
G_NEWTON = 6.67430e-11          # m^3 kg^-1 s^-2
C_LIGHT = 2.99792458e8          # m/s
H_BAR = 1.054571817e-34         # J*s
M_NUCLEON = 1.66053906660e-27   # kg (原子质量单位)
M_NEUTRON = 1.67492749804e-27   # kg
M_PROTON = 1.67262192369e-27    # kg
E_CHARGE = 1.602176634e-19      # C
SIGMA_STEFAN = 5.670374419e-8   # W m^-2 K^-4
K_BOLTZMANN = 1.380649e-23      # J/K

# 中子星特征尺度
M_SUN = 1.98847e30              # kg
R_NS_TYPICAL = 1.2e4            # m (12 km)
RHO_NUCLEAR = 2.8e17            # kg/m^3 (核饱和密度)

# 几何化单位制转换因子
LENGTH_GEOM = G_NEWTON * M_SUN / C_LIGHT**2   # ~1.477 km (GM_sun/c^2)


def init_physics_system():
    """
    初始化物理计算系统，输出诊断信息。
    源自 513_hello 的初始化思想。
    """
    msg = "Neutron Star Equation of State & Dense Matter Synthesis System Initialized"
    print(f"[INIT] {msg}")
    print(f"[INIT] Geometric length unit = {LENGTH_GEOM:.6e} m")
    return True


def fermat_primality_test(n: int, k: int = 5) -> bool:
    """
    费马素性检验 (Fermat primality test)。
    源自 912_prime_fermat 的核心算法。

    在致密物质计算中，该检验用于生成高质量的伪随机种子，
    以初始化不同密度网格上的Monte Carlo采样。

    算法基于费马小定理:
        若 p 为素数，且 gcd(a, p) = 1，则 a^(p-1) ≡ 1 (mod p)

    Parameters
    ----------
    n : int
        待检验的正整数。
    k : int
        迭代次数，增加 k 可降低伪素数通过的概率。

    Returns
    -------
    bool
        True 表示 n 通过了费马检验（大概率是素数）。
        False 表示 n 确定是合数。
    """
    if n <= 1 or n == 4:
        return False
    if n <= 3:
        return True

    for _ in range(k):
        a = random.randint(2, n - 2)
        if math.gcd(n, a) != 1:
            return False
        if pow(a, n - 1, n) != 1:
            return False
    return True


def generate_prime_seed(lower: int = 1000, upper: int = 10000) -> int:
    """
    在指定区间内搜索一个通过费马检验的素数，用作数值模拟的随机种子。
    """
    if lower < 2:
        lower = 2
    if upper <= lower:
        raise ValueError("upper must be greater than lower")

    candidate = random.randint(lower, upper)
    # 若候选数为偶数，先加1变为奇数
    if candidate % 2 == 0:
        candidate += 1

    max_iter = 2000
    for _ in range(max_iter):
        if candidate > upper:
            candidate = lower if lower % 2 == 1 else lower + 1
        if fermat_primality_test(candidate, k=8):
            return candidate
        candidate += 2

    raise RuntimeError("Failed to find a prime seed in the given range.")


def rot13_encode(s: str) -> str:
    """
    ROT13 字符串编码，源自 1045_rot13。
    在本项目中用于对数值摘要进行轻度混淆输出（非安全加密，仅演示）。
    """
    result = []
    for ch in s:
        if 'a' <= ch <= 'z':
            result.append(chr((ord(ch) - ord('a') + 13) % 26 + ord('a')))
        elif 'A' <= ch <= 'Z':
            result.append(chr((ord(ch) - ord('A') + 13) % 26 + ord('A')))
        else:
            result.append(ch)
    return ''.join(result)


def rot13_decode(s: str) -> str:
    """ROT13 是自逆的，两次应用即恢复原文。"""
    return rot13_encode(s)


def safe_sqrt(x: float) -> float:
    """带边界保护的平方根，避免负值导致的 nan。"""
    if x < 0.0:
        if x > -1e-14:
            return 0.0
        raise ValueError(f"safe_sqrt received negative argument: {x}")
    return math.sqrt(x)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """带边界保护的除法，避免除以零。"""
    if abs(b) < 1e-30:
        return default
    return a / b


def geometric_units(m_kg: float) -> float:
    """
    将质量（kg）转换为几何单位制下的长度（m），即 GM/c^2。
    """
    return G_NEWTON * m_kg / C_LIGHT**2


def fermi_momentum_to_density(kf: float) -> float:
    """
    由费米动量 kf (1/fm) 计算数密度 n (fm^-3)。
    公式: n = kf^3 / (3 * pi^2)
    """
    if kf < 0.0:
        raise ValueError("Fermi momentum must be non-negative.")
    return kf**3 / (3.0 * math.pi**2)


def density_to_fermi_momentum(n: float) -> float:
    """
    由数密度 n (fm^-3) 计算费米动量 kf (1/fm)。
    公式: kf = (3 * pi^2 * n)^(1/3)
    """
    if n < 0.0:
        raise ValueError("Number density must be non-negative.")
    return (3.0 * math.pi**2 * n)**(1.0 / 3.0)
