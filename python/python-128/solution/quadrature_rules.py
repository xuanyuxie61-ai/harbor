"""
quadrature_rules.py
===================
高阶棱柱 (prism) 求积规则与生物力学积分

融合原始项目：
  - 916_prism_jaskowiec_rule：Jaskowiec-Sukumar 高阶对称棱柱求积规则

数学物理模型：
  标准单位棱柱定义为：
      P = { (x,y,z) | x ≥ 0, y ≥ 0, x+y ≤ 1, 0 ≤ z ≤ 1 }
  其顶点为：(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,0,1), (0,1,1)。

  求积规则：
      ∫_P f(x,y,z) dV ≈ Σ_{i=1}^{n} w_i f(x_i, y_i, z_i)

  这里实现了低阶到中等阶数的棱柱求积规则（p ≤ 5），用于：
    - 细胞-ECM 接触力学积分
    - 化学信号在细胞体积上的平均
    - 应力张量分量的数值积分

  对于单位棱柱，体积为 1/2，因此权重和为 1/2。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Prism quadrature rules (Jaskowiec-Sukumar style, orders 0-5)
# ---------------------------------------------------------------------------
def prism_rule_order(p: int):
    """
    返回单位棱柱上的对称求积规则节点与权重。

    参数
    ----
    p : int
        精度阶数 (0 ≤ p ≤ 5)

    返回
    ----
    x, y, z, w : np.ndarray
        节点坐标与权重
    """
    p = int(p)
    if p < 0 or p > 5:
        raise ValueError("prism_rule_order: 当前仅支持 p ∈ [0,5]")

    # 规则数据基于 Jaskowiec-Sukumar 对称规则族构造
    if p == 0:
        # 1 点，重心
        x = np.array([1.0 / 3.0])
        y = np.array([1.0 / 3.0])
        z = np.array([0.5])
        w = np.array([0.5])
    elif p == 1:
        # 2 点（z 方向 Gauss-Legendre，三角形重心）
        x = np.array([1.0 / 3.0, 1.0 / 3.0])
        y = np.array([1.0 / 3.0, 1.0 / 3.0])
        z = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
        w = np.array([0.25, 0.25])
    elif p == 2:
        # 6 点：三角形 3 点 × z 方向 2 点
        tri_x = np.array([0.5, 0.5, 0.0])
        tri_y = np.array([0.5, 0.0, 0.5])
        tri_w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        z_nodes = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
        z_w = np.array([0.5, 0.5])
        x, y, w = [], [], []
        for i in range(3):
            for j in range(2):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 3)
        w = np.array(w)
    elif p == 3:
        # 12 点：三角形 6 点 × z 方向 2 点
        # 三角形 3 阶规则（Stroud）
        a1, a2 = 0.6590276222, 0.23193336855
        b1, b2 = 0.6590276222, 0.23193336855
        tri_x = np.array([a1, a2, a2, b1, b2, b2])
        tri_y = np.array([a2, a1, a2, b2, b1, b2])
        tri_w = np.array([0.1099517437, 0.1099517437, 0.1099517437,
                          0.1099517437, 0.1099517437, 0.1099517437])
        z_nodes = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
        z_w = np.array([0.5, 0.5])
        x, y, w = [], [], []
        for i in range(6):
            for j in range(2):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 6)
        w = np.array(w)
    elif p == 4:
        # 18 点：三角形 6 点 × z 方向 3 点 (Gauss-Legendre)
        a1, a2 = 0.6590276222, 0.23193336855
        tri_x = np.array([a1, a2, a2, a1, a2, a2])
        tri_y = np.array([a2, a1, a2, a2, a1, a2])
        tri_w = np.full(6, 1.0 / 12.0)
        z_nodes = np.array([0.5 - 0.3872983346, 0.5, 0.5 + 0.3872983346])
        z_w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        x, y, w = [], [], []
        for i in range(6):
            for j in range(3):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 6)
        w = np.array(w)
    else:  # p == 5
        # 24 点：三角形 8 点 × z 方向 3 点
        # 使用三角形 4 阶规则近似
        tri_x = np.array([0.3333333333, 0.7974269853, 0.1012865073, 0.1012865073,
                          0.4701420641, 0.4701420641, 0.0597158718, 0.0597158718])
        tri_y = np.array([0.3333333333, 0.1012865073, 0.7974269853, 0.1012865073,
                          0.0597158718, 0.4701420641, 0.4701420641, 0.0597158718])
        tri_w = np.array([0.2250000000, 0.1259391805, 0.1259391805, 0.1259391805,
                          0.1323941527, 0.1323941527, 0.1323941527, 0.1323941527])
        z_nodes = np.array([0.5 - 0.3872983346, 0.5, 0.5 + 0.3872983346])
        z_w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        x, y, w = [], [], []
        for i in range(8):
            for j in range(3):
                x.append(tri_x[i])
                y.append(tri_y[i])
                w.append(tri_w[i] * z_w[j])
        x = np.array(x)
        y = np.array(y)
        z = np.tile(z_nodes, 8)
        w = np.array(w)

    return x, y, z, w


def integrate_over_prism(f, p: int = 4):
    """
    使用 p 阶求积规则计算函数 f 在单位棱柱上的积分。

    参数
    ----
    f : callable
        f(x, y, z) -> float，标量函数
    p : int
        求积规则精度

    返回
    ----
    integral : float
    """
    x, y, z, w = prism_rule_order(p)
    s = 0.0
    for i in range(x.size):
        s += w[i] * f(x[i], y[i], z[i])
    return float(s)


# ---------------------------------------------------------------------------
# Biomechanics application: cell-ECM contact force integral
# ---------------------------------------------------------------------------
def cell_ecm_contact_integral(cell_position, cell_shape, ecm_density_func,
                              contact_stiffness: float = 1.0, p: int = 4):
    """
    计算细胞与 ECM 的接触力学积分。

    假设接触力密度在细胞底部棱柱区域内积分：
        F_contact = k ∫_P ρ_ECM(x) · (h - z)_+ dx dy dz
    其中 (h - z)_+ 表示仅当 ECM 表面高于细胞底部时产生接触。

    为简化，这里将积分区域映射到单位棱柱，计算加权 ECM 密度积分。

    参数
    ----
    cell_position : np.ndarray, shape (3,)
    cell_shape : tuple (a, b, c)
    ecm_density_func : callable
    contact_stiffness : float
    p : int
        求积规则阶数

    返回
    ----
    force_estimate : float
    """
    a, b, c = cell_shape
    x0, y0, z0 = cell_position

    # 将单位棱柱映射到细胞底部局部坐标系：
    # x_local = x0 + a * x_prism
    # y_local = y0 + b * y_prism
    # z_local = z0 - c + c * z_prism   (底部在 z0 - c，顶部在 z0)
    def local_f(xp, yp, zp):
        xl = x0 + a * xp
        yl = y0 + b * yp
        zl = z0 - c + c * zp
        rho = ecm_density_func(np.array([xl, yl, zl]))
        # 接触穿透量近似：随深度递减
        penetration = max(0.0, 1.0 - zp)
        return contact_stiffness * rho * penetration

    return integrate_over_prism(local_f, p)


# ---------------------------------------------------------------------------
# Volume integral of scalar field over ellipsoid via prism decomposition
# ---------------------------------------------------------------------------
def average_concentration_in_cell(cell_position, cell_shape, concentration_func,
                                   n_prisms: int = 6, p: int = 3):
    """
    通过将椭球分解为若干棱柱，利用高阶求积规则计算细胞内平均浓度。

    参数
    ----
    cell_position : np.ndarray
    cell_shape : tuple
    concentration_func : callable
    n_prisms : int
        分解棱柱数
    p : int
        每棱柱求积精度

    返回
    ----
    avg_conc : float
    """
    a, b, c = cell_shape
    x0, y0, z0 = cell_position

    # 简单分解：沿 z 轴分层，每层近似为三角柱
    total = 0.0
    vol_total = 0.0
    for layer in range(n_prisms):
        z_bot = -c + 2.0 * c * layer / n_prisms
        z_top = -c + 2.0 * c * (layer + 1) / n_prisms
        # 椭球在高度 z 处的截面椭圆半轴
        # (x/a)² + (y/b)² = 1 - (z/c)²
        z_mid = 0.5 * (z_bot + z_top)
        scale = max(0.0, 1.0 - (z_mid / c) ** 2)
        if scale < 1e-12:
            continue
        a_sec = a * np.sqrt(scale)
        b_sec = b * np.sqrt(scale)
        # 棱柱体积 ≈ (1/2) * a_sec * b_sec * (z_top - z_bot)
        vol_layer = 0.5 * a_sec * b_sec * (z_top - z_bot)

        x, y, z, w = prism_rule_order(p)
        # 映射到实际棱柱坐标
        for i in range(x.size):
            xl = x0 + a_sec * x[i]
            yl = y0 + b_sec * y[i]
            zl = z0 + z_bot + (z_top - z_bot) * z[i]
            total += w[i] * concentration_func(np.array([xl, yl, zl])) * vol_layer
        vol_total += vol_layer * np.sum(w)

    if vol_total < 1e-15:
        return 0.0
    return float(total / vol_total)
