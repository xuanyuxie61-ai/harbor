"""
geometry_utils.py
血管几何建模与网格格式工具

融合来源:
- 868_pi_spigot: 高精度圆周率计算（Spigot算法），用于血管圆形截面精确几何
- 915_prime_plot: 素性试除法，用于血管分叉级别的素数标记与离散化索引
- 381_fem_to_triangle: FEM节点-单元数据结构（R8MAT/I4MAT映射思想）

科学背景:
在计算流体力学中，动脉血管通常被建模为圆管或锥形管。
血管横截面的几何参数（面积 A = πR²，周长 C = 2πR）依赖于高精度的π值。
主动脉弓存在多级分叉（第1级：升主动脉→主动脉弓；第2级：头臂干；第3级：左颈总；
第5级：更细分支...），素数标记用于唯一标识关键分叉节点。
"""

import numpy as np


# ======================================================================
# 来自 868_pi_spigot 的高精度π计算
# ======================================================================

def pi_spigot(n_digits: int) -> str:
    """
    Bailey–Borwein–Plouffe (BBP) 类Spigot算法计算π。

    数学原理:
    BBP公式（1995）允许直接计算π的任意位十六进制数字：
        π = Σ_{k=0}^{∞} (1/16^k) · [ 4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6) ]

    该公式属于Spigot算法家族，通过级数求和逐位提取数字。
    我们用它来计算π的双精度浮点近似值，再格式化为字符串。

    参数:
        n_digits: 需要保留的π小数位数

    返回:
        π的字符串表示（含小数点）
    """
    if n_digits < 1:
        return "3."
    pi_val = 0.0
    # 取足够多的级数项以保证精度
    n_terms = max(n_digits * 5, 50)
    for k in range(n_terms):
        coeff = 1.0 / (16.0 ** k)
        pi_val += coeff * (4.0 / (8.0 * k + 1.0)
                           - 2.0 / (8.0 * k + 4.0)
                           - 1.0 / (8.0 * k + 5.0)
                           - 1.0 / (8.0 * k + 6.0))
    fmt = f"{{:.{n_digits}f}}"
    return fmt.format(pi_val)


def pi_high_precision() -> float:
    """
    使用Spigot算法计算π到15位小数，返回双精度浮点数。
    该精度足以满足血管几何计算需求。
    """
    pi_str = pi_spigot(15)
    return float(pi_str)


# ======================================================================
# 来自 915_prime_plot 的素数工具（用于血管分叉级别标记）
# ======================================================================

def is_prime(n: int) -> bool:
    """
    试除法判定素数。

    数学原理:
    若整数 n > 1 没有除1和自身外的正因子，则n为素数。
    只需检验到 √n 即可，因为若 n = ab，则 min(a,b) ≤ √n。

    参数:
        n: 待检测整数

    返回:
        True 若n为素数，否则False
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(np.sqrt(n)) + 1
    for d in range(3, limit, 2):
        if n % d == 0:
            return False
    return True


def prime_sieve(limit: int) -> np.ndarray:
    """
    埃拉托斯特尼筛法生成不超过limit的所有素数。

    参数:
        limit: 上限

    返回:
        素数数组
    """
    if limit < 2:
        return np.array([], dtype=int)
    is_prime_arr = np.ones(limit + 1, dtype=bool)
    is_prime_arr[0:2] = False
    for p in range(2, int(np.sqrt(limit)) + 1):
        if is_prime_arr[p]:
            is_prime_arr[p * p:limit + 1:p] = False
    return np.nonzero(is_prime_arr)[0]


def bifurcation_prime_level(level: int) -> bool:
    """
    判断血管分叉级别是否为素数级别。
    在动脉树中，素数级别的分叉往往对应血流动力学关键节点
    （如第2级→左颈总，第3级→左锁骨下，第5级→重要侧支）。
    """
    return is_prime(level)


# ======================================================================
# 来自 381_fem_to_triangle 的FEM数据结构
# ======================================================================

class FEMMesh:
    """
    有限元三角形网格数据结构。
    存储节点坐标和三角形单元连接关系。
    """
    def __init__(self, nodes: np.ndarray = None, elements: np.ndarray = None):
        """
        参数:
            nodes: (N_nodes, 2) 节点坐标数组
            elements: (N_elements, 3) 三角形单元节点索引（0-based）
        """
        self.nodes = nodes if nodes is not None else np.zeros((0, 2))
        self.elements = elements if elements is not None else np.zeros((0, 3), dtype=int)

    def node_count(self) -> int:
        return self.nodes.shape[0]

    def element_count(self) -> int:
        return self.elements.shape[0]

    def element_area(self, elem_idx: int) -> float:
        """
        计算指定三角形单元的有向面积。
        对于节点 p1=(x1,y1), p2=(x2,y2), p3=(x3,y3):
            A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        """
        idx = self.elements[elem_idx]
        p1, p2, p3 = self.nodes[idx[0]], self.nodes[idx[1]], self.nodes[idx[2]]
        area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) +
                         p2[0] * (p3[1] - p1[1]) +
                         p3[0] * (p1[1] - p2[1]))
        return area

    def total_area(self) -> float:
        """计算网格覆盖的总面积。"""
        return sum(self.element_area(i) for i in range(self.element_count()))

    def scale_to_area(self, target_area: float):
        """
        等比例缩放网格使其总面积等于target_area。
        缩放因子 s = sqrt(target_area / current_area)。
        """
        current = self.total_area()
        if current <= 0:
            return
        s = np.sqrt(target_area / current)
        self.nodes *= s


# ======================================================================
# 血管几何参数计算
# ======================================================================

def circular_cross_section_area(radius: float) -> float:
    """
    计算圆形血管横截面积: A = π R²
    """
    PI = pi_high_precision()
    return PI * radius * radius


def circular_cross_section_perimeter(radius: float) -> float:
    """
    计算圆形血管横截面周长: C = 2π R
    """
    PI = pi_high_precision()
    return 2.0 * PI * radius


def womersley_number(radius: float, kinematic_viscosity: float,
                     angular_frequency: float) -> float:
    """
    计算Womersley数（无量纲脉动流参数）。

    定义:
        α = R * sqrt(ω / ν)

    其中:
        R: 血管半径 [m]
        ω: 角频率 [rad/s], ω = 2πf, f为心率
        ν: 运动粘度 [m²/s]

    物理意义:
        α << 1: 准稳态流，惯性效应可忽略
        α ~ 1: 过渡区
        α >> 1: 惯性主导，速度剖面呈活塞状
    """
    if kinematic_viscosity <= 0 or radius <= 0 or angular_frequency <= 0:
        raise ValueError("Womersley number parameters must be positive.")
    return radius * np.sqrt(angular_frequency / kinematic_viscosity)


def reynolds_number(mean_velocity: float, diameter: float,
                    kinematic_viscosity: float) -> float:
    """
    计算雷诺数。

    定义:
        Re = U * D / ν

    物理意义:
        Re < 2300: 层流
        Re > 4000: 湍流
        动脉血流通常 Re ∈ [500, 2000]（层流或过渡流）
    """
    if kinematic_viscosity <= 0 or diameter <= 0:
        raise ValueError("Reynolds number parameters must be positive.")
    return mean_velocity * diameter / kinematic_viscosity


def murray_law_radius(r_parent: float, n_children: int,
                      bifurcation_angle_deg: float = 60.0) -> float:
    """
    Murray定律：最优血管分叉满足 r_p³ = Σ r_c³。
    对于对称二分叉:
        r_child = r_parent / 2^(1/3) ≈ 0.794 * r_parent

    参数:
        r_parent: 母管半径 [m]
        n_children: 子管数量
        bifurcation_angle_deg: 分叉角度 [度]（用于边界检查）

    返回:
        子管半径 [m]
    """
    if r_parent <= 0 or n_children < 1:
        raise ValueError("Invalid Murray law parameters.")
    if not (0 < bifurcation_angle_deg < 180):
        raise ValueError("Bifurcation angle must be in (0, 180) degrees.")
    return r_parent / (n_children ** (1.0 / 3.0))


# ======================================================================
# 边界检查与数值鲁棒性工具
# ======================================================================

def safe_sqrt(x: float) -> float:
    """安全的平方根，对负数返回0并发出警告。"""
    if x < 0:
        if x > -1e-12:
            return 0.0
        raise ValueError(f"safe_sqrt received negative value: {x}")
    return np.sqrt(x)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，b接近0时返回default。"""
    if abs(b) < 1e-14:
        return default
    return a / b
