"""
lattice_utils.py — 格点工具与时序分析 (PROJECT_232)

融合种子项目:
  - 1412_weekday_zeller: Zeller 同余公式 (weekday_gregorian, weekday_julian)
  - 492_gridlines: 结构化网格 (grid_rectangular, grid_triangular)

核心物理:
  格点 QCD (Lattice QCD) 是计算强子散射振幅的第一性原理方法:
    - 欧氏时空被离散化为四维格点
    - 夸克场定义在格点上，胶子场定义在链接上
    - 路径积分通过 Monte Carlo 方法计算

  Lüscher 方法:
    在有限体积 V = L³ 中，两强子能级 E_n(L) 与无限体积散射相移的关系:
      tan δ_l(k) = π^{3/2} q / Z_{00}(1; q²)
    其中 q = kL/(2π), Z_{00} 为广义 zeta 函数。

  本模块实现:
    1. 有限体积格点生成
    2. Lüscher 公式的数值实现
    3. 计算时序分析 (融合 Zeller 公式思想)
"""
import numpy as np
from constants import PI, TWOPI, EPS_MACH, MASS_PI_PLUS


# ===================================================================
# 格点生成 (融合 492_gridlines)
# ===================================================================

def momentum_lattice(L, p_max, m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    有限体积中的动量格点

    在边长为 L 的立方体中，周期性边界条件给出离散动量:
      p⃗ = (2π/L) n⃗,  n⃗ ∈ Z³

    总动量 P⃗ = p⃗_1 + p⃗_2 = (2π/L) d⃗,  d⃗ ∈ Z³

    对于 P⃗ = 0 (静止系):
      p⃗_1 = -p⃗_2 = (2π/L) n⃗

    Parameters
    ----------
    L : float
        格点空间尺寸 [GeV⁻¹]
    p_max : float
        最大动量截断 [GeV]
    m_a, m_b : float
        粒子质量

    Returns
    -------
    momenta : ndarray, shape (N, 3)
        允许的三动量
    energies : ndarray, shape (N,)
        对应的两粒子能量 E = √(p²+m_a²) + √(p²+m_b²)
    n_vectors : ndarray, shape (N, 3)
        整数向量 n⃗
    """
    n_max = int(np.ceil(p_max * L / TWOPI)) + 1
    momenta = []
    energies = []
    n_vectors = []

    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            for nz in range(-n_max, n_max + 1):
                p_vec = TWOPI / L * np.array([nx, ny, nz], dtype=np.float64)
                p_mag = np.linalg.norm(p_vec)
                if p_mag > p_max:
                    continue
                E = np.sqrt(p_mag ** 2 + m_a ** 2) + np.sqrt(p_mag ** 2 + m_b ** 2)
                momenta.append(p_vec)
                energies.append(E)
                n_vectors.append([nx, ny, nz])

    if len(momenta) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3), dtype=int)

    # 按能量排序
    energies = np.array(energies)
    idx = np.argsort(energies)
    return np.array(momenta)[idx], energies[idx], np.array(n_vectors)[idx]


def zeta_function_Z00(s, q_squared, n_max=50):
    """
    Lüscher zeta 函数 Z_{00}(s; q²)

    Z_{00}(s; q²) = Σ_{n⃗ ∈ Z³} 1/(|n⃗|² - q²)^s

    对于 s=1 (Lüscher 公式中使用的):
      Z_{00}(1; q²) = Σ_{n⃗} 1/(|n⃗|² - q²)

    此级数条件收敛，需使用解析延拓或指数正则化:
      Z_{00}(1; q²) = lim_{α→0} Σ_{n⃗} e^{-α|n⃗|²} / (|n⃗|² - q²)

    物理意义: 将有限体积能级与无限体积相移联系起来。

    Parameters
    ----------
    s : float
        zeta 函数参数 (通常为 1)
    q_squared : float
        无量纲动量平方 q² = (kL/(2π))²
    n_max : int
        求和截断

    Returns
    -------
    float
        Z_{00}(s; q²)
    """
    result = 0.0
    alpha_reg = 0.01  # 正则化参数

    for nx in range(-n_max, n_max + 1):
        for ny in range(-n_max, n_max + 1):
            for nz in range(-n_max, n_max + 1):
                n_sq = nx ** 2 + ny ** 2 + nz ** 2
                denom = n_sq - q_squared
                if abs(denom) < EPS_MACH:
                    continue
                # 指数正则化
                reg_factor = np.exp(-alpha_reg * n_sq)
                result += reg_factor / (denom ** s)

    return result


def luscher_phase_shift(E_n, L, m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS,
                         l_quantum=0):
    """
    Lüscher 公式: 从有限体积能级提取散射相移

    对于总动量为零的参考系:
      tan δ_l(k) = (π^{3/2} q) / Z_{00}(1; q²)

    其中:
      q = kL/(2π)
      k = 质心动量 (由 E_n 和 L 确定)

    E_n = 2√(k² + m²)  (对于等质量粒子)
    → k = √((E_n/2)² - m²)

    Parameters
    ----------
    E_n : float
        有限体积两粒子能级 [GeV]
    L : float
        格点空间尺寸 [GeV⁻¹]
    m_a, m_b : float
        粒子质量 [GeV]
    l_quantum : int
        角动量量子数

    Returns
    -------
    delta_l : float
        散射相移 [弧度]
    k_cm : float
        质心动量 [GeV]
    """
    if E_n < m_a + m_b:
        return 0.0, 0.0

    # 质心动量 (等质量简化)
    half_E = E_n / 2.0
    k_sq = half_E ** 2 - m_a ** 2
    if k_sq < 0:
        return 0.0, 0.0
    k_cm = np.sqrt(k_sq)

    q = k_cm * L / TWOPI
    q_sq = q ** 2

    # Z_{00}(1; q²)
    Z00 = zeta_function_Z00(1, q_sq)

    # tan δ = π^{3/2} q / Z_{00}
    if abs(Z00) < EPS_MACH:
        delta_l = PI / 2.0
    else:
        tan_delta = (PI ** 1.5) * q / Z00
        delta_l = np.arctan(tan_delta)

    return delta_l, k_cm


def luscher_analysis(energy_levels, L, m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    从多个有限体积能级提取相移曲线

    Parameters
    ----------
    energy_levels : ndarray
        有限体积能级列表
    L : float
        格点尺寸
    m_a, m_b : float
        粒子质量

    Returns
    -------
    k_values : ndarray
        质心动量
    delta_values : ndarray
        散射相移
    """
    k_values = []
    delta_values = []

    for E_n in energy_levels:
        delta, k = luscher_phase_shift(E_n, L, m_a, m_b)
        if k > EPS_MACH:
            k_values.append(k)
            delta_values.append(delta)

    return np.array(k_values), np.array(delta_values)


# ===================================================================
# 计算时序分析 (融合 1412_weekday_zeller)
# ===================================================================

def zeller_congruence(year, month, day, calendar='gregorian'):
    """
    Zeller 同余公式计算星期几

    融合 1412_weekday_zeller/weekday_gregorian.m 的算法。

    Gregorian 日历:
      h = (q + ⌊13(m+1)/5⌋ + K + ⌊K/4⌋ + ⌊J/4⌋ - 2J) mod 7

    其中:
      q = 日期 (1-31)
      m = 月份 (3=三月, ..., 14=二月)
      K = year mod 100
      J = ⌊year/100⌋

    物理应用: 在格点 QCD Monte Carlo 计算中，
    用于标记配置的时间戳和分析计算进度的周期性。

    Parameters
    ----------
    year : int
        年份
    month : int
        月份 (1-12)
    day : int
        日期 (1-31)
    calendar : str
        'gregorian' 或 'julian'

    Returns
    -------
    int
        星期几 (0=周六, 1=周日, ..., 6=周五)
    """
    q = day
    m = month
    y = year

    # 一月和二月视为上一年的 13、14 月
    if m <= 2:
        m += 12
        y -= 1

    K = y % 100
    J = y // 100

    if calendar == 'gregorian':
        h = (q + (13 * (m + 1)) // 5 + K + K // 4 + J // 4 - 2 * J) % 7
    elif calendar == 'julian':
        h = (q + (13 * (m + 1)) // 5 + K + K // 4 + 5 - J) % 7
    else:
        raise ValueError(f"Unknown calendar: {calendar}")

    return int(h)


def weekday_name(day_index):
    """
    将 Zeller 输出转换为星期名称

    Parameters
    ----------
    day_index : int
        Zeller 输出 (0=Saturday, ..., 6=Friday)

    Returns
    -------
    str
        星期名称
    """
    names = ['Saturday', 'Sunday', 'Monday', 'Tuesday',
             'Wednesday', 'Thursday', 'Friday']
    if 0 <= day_index <= 6:
        return names[day_index]
    return 'Unknown'


def computational_schedule_marker(config_index, total_configs,
                                    base_year=2024, base_month=1, base_day=1):
    """
    为格点 QCD 配置生成时序标记

    模拟 Monte Carlo 计算中配置的"生成日期"，
    用于分析计算进度的周期性模式。

    Parameters
    ----------
    config_index : int
        配置编号
    total_configs : int
        总配置数
    base_year, base_month, base_day : int
        基准日期

    Returns
    -------
    year, month, day : int
        标记日期
    weekday : str
        星期名称
    """
    # 每天生成约 100 个配置
    days_offset = config_index // 100
    total_days = total_configs // 100

    # 简单日期推进 (不考虑月份长度)
    day = base_day + (config_index % 100)
    month = base_month + (config_index // 100) % 12
    year = base_year + (config_index // 1200)

    # 简化处理
    if day > 28:
        day = ((day - 1) % 28) + 1
    if month > 12:
        month = ((month - 1) % 12) + 1

    wdx = zeller_congruence(year, month, day)
    return year, month, day, weekday_name(wdx)


# ===================================================================
# 矩形与三角格点 (融合 492_gridlines)
# ===================================================================

def rectangular_grid_2d(x_min, x_max, y_min, y_max, nx, ny):
    """
    二维矩形格点

    来自 492_gridlines/grid_rectangular.m

    物理应用: 格点 QCD 中的二维截面 (如 xy 平面)。

    Parameters
    ----------
    x_min, x_max : float
        x 范围
    y_min, y_max : float
        y 范围
    nx, ny : int
        格点数

    Returns
    -------
    x_grid, y_grid : ndarray
        二维网格坐标
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    return np.meshgrid(x, y, indexing='ij')


def triangular_grid_2d(x_min, x_max, y_min, y_max, n_points):
    """
    二维三角格点 (FCC 截面)

    来自 492_gridlines/grid_triangular.m

    三角格点比矩形格点具有更好的旋转对称性，
    在格点场论中用于改进离散化误差。

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        范围
    n_points : int
        约略格点数

    Returns
    -------
    points : ndarray, shape (N, 2)
        格点坐标
    """
    nx = int(np.sqrt(n_points))
    ny = int(np.sqrt(n_points))
    dx = (x_max - x_min) / max(nx, 1)
    dy = (y_max - y_min) / max(ny, 1)

    points = []
    for i in range(nx):
        for j in range(ny):
            x = x_min + i * dx
            y = y_min + j * dy
            # 偶数行偏移
            if i % 2 == 1:
                y += dy / 2.0
            if x_min <= x <= x_max and y_min <= y <= y_max:
                points.append([x, y])

    return np.array(points) if points else np.zeros((0, 2))
