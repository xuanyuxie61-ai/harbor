# -*- coding: utf-8 -*-
"""
crystal_lattice.py
==================
晶体格点结构与Burgers矢量拓扑模块

本模块融合以下种子项目算法:
- 170_chinese_remainder_theorem: 中国剩余定理 → 多滑移系索引映射
- 104_boundary_locus: 边界轨迹法 → 晶界取向空间参数化

核心物理:
---------
对于FCC晶体，滑移系为 {111}<110>，共12个独立滑移系。
每个滑移系由滑移面法线 n̂ 和Burgers矢量方向 b̂ 定义。

滑移面: (111), (1̄11), (11̄1), (111̄) — 4个 {111} 面
滑移方向: [1̄10], [101̄], [011̄] — 每个面3个 <110> 方向

Schmid因子:
m = cos(φ) cos(λ) = (n̂·σ̂)·(b̂·σ̂)

其中 φ 是载荷轴与滑移面法线的夹角，λ 是载荷轴与Burgers矢量的夹角。

中国剩余定理用于多滑移系的唯一索引:
给定滑移系编号 i ∈ {0,...,11}，映射到 (滑移面索引 p, 滑移方向索引 d):
i ≡ p (mod 4), i ≡ d (mod 3)
使用CRT保证映射的双射性。
"""

import math
from physical_constants import PI, SQRT2, SQRT3, DEFAULT_MATERIAL


# ============================================================================
# 中国剩余定理 (CRT) — 来自 170_chinese_remainder_theorem
# 用于多滑移系唯一索引映射
# ============================================================================

def extended_gcd(a, b):
    """
    扩展欧几里得算法

    求解: a*x + b*y = gcd(a, b)

    Args:
        a, b: 正整数

    Returns:
        tuple: (gcd, x, y) 使得 a*x + b*y = gcd
    """
    if a == 0:
        return b, 0, 1
    g, x1, y1 = extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def chinese_remainder_theorem(residues, moduli):
    """
    中国剩余定理求解同余方程组

    x ≡ r₁ (mod m₁)
    x ≡ r₂ (mod m₂)
    ...
    x ≡ rₖ (mod mₖ)

    当 m₁, m₂, ..., mₖ 两两互素时，存在唯一解 x mod M
    其中 M = m₁ * m₂ * ... * mₖ

    在位错理论中应用:
    - m₁ = 4 (FCC的{111}滑移面数)
    - m₂ = 3 (每个面的<110>滑移方向数)
    - 滑移系编号 i 唯一确定 (面, 方向) 对

    Args:
        residues: 余数列表 [r₁, r₂, ..., rₖ]
        moduli: 模数列表 [m₁, m₂, ..., mₖ]

    Returns:
        int: 唯一解 x mod M

    Raises:
        ValueError: 当模数不两两互素时
    """
    if len(residues) != len(moduli):
        raise ValueError("余数和模数列表长度必须相等")

    # 检查两两互素
    n = len(moduli)
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            if g != 1:
                raise ValueError(f"模数 {moduli[i]} 和 {moduli[j]} 不互素, gcd={g}")

    # 计算 M = ∏ mᵢ
    M = 1
    for m in moduli:
        M *= m

    # CRT求解: x = Σ rᵢ * Mᵢ * yᵢ (mod M)
    # 其中 Mᵢ = M/mᵢ, yᵢ = Mᵢ^{-1} mod mᵢ
    x = 0
    for r, m in zip(residues, moduli):
        M_i = M // m
        _, y_i, _ = extended_gcd(M_i, m)
        y_i = y_i % m  # 确保正数
        x += r * M_i * y_i

    return x % M


def slip_system_index(plane_idx, direction_idx, n_planes=4, n_dirs=3):
    """
    使用CRT将滑移系(面,方向)编码为唯一整数索引

    对于FCC {111}<110>:
    - 4个滑移面 × 3个方向 = 12个滑移系
    - CRT保证: i ≡ plane_idx (mod 4), i ≡ direction_idx (mod 3)

    Args:
        plane_idx: 滑移面索引 (0-3)
        direction_idx: 滑移方向索引 (0-2)
        n_planes: 滑移面数量 (默认4)
        n_dirs: 滑移方向数量 (默认3)

    Returns:
        int: 唯一滑移系索引 (0-11)
    """
    return chinese_remainder_theorem([plane_idx, direction_idx], [n_planes, n_dirs])


def decode_slip_system(index, n_planes=4, n_dirs=3):
    """
    从唯一索引解码滑移系

    逆映射: i → (i mod 4, i mod 3)

    Args:
        index: 滑移系索引 (0-11)
        n_planes: 滑移面数量
        n_dirs: 滑移方向数量

    Returns:
        tuple: (plane_idx, direction_idx)
    """
    plane_idx = index % n_planes
    direction_idx = index % n_dirs
    return plane_idx, direction_idx


# ============================================================================
# FCC滑移系几何 (Miller指数)
# ============================================================================

# FCC 4个 {111} 滑移面 (法线方向)
FCC_SLIP_PLANES = [
    (1, 1, 1),    # (111)
    (-1, 1, 1),   # (1̄11)
    (1, -1, 1),   # (11̄1)
    (1, 1, -1),   # (111̄)
]

# FCC 每个面的3个 <110> Burgers矢量方向
# 每个面内的3个方向必须满足: b̂ · n̂ = 0 (在滑移面内)
FCC_SLIP_DIRECTIONS = [
    # 面 (111): b ⊥ (111)
    [(-1, 1, 0), (0, -1, 1), (1, 0, -1)],
    # 面 (1̄11): b ⊥ (1̄11)
    [(1, 1, 0), (0, -1, 1), (-1, 0, -1)],
    # 面 (11̄1): b ⊥ (11̄1)
    [(1, 1, 0), (0, 1, 1), (-1, 0, -1)],
    # 面 (111̄): b ⊥ (111̄)
    [(1, -1, 0), (0, 1, 1), (1, 0, 1)],
]

# BCC 12个 <111> 滑移方向 × {110}/{112}/{123} 滑移面
BCC_SLIP_DIRECTIONS = [
    (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
]

BCC_SLIP_PLANES_110 = [
    (0, 1, -1), (1, 0, -1), (1, -1, 0),
    (0, 1, 1), (1, 0, 1), (1, 1, 0),
]


def normalize_miller(indices):
    """
    将Miller指数归一化为单位矢量

    n̂ = (h, k, l) / sqrt(h² + k² + l²)

    Args:
        indices: Miller指数元组 (h, k, l)

    Returns:
        tuple: 归一化方向余弦 (nx, ny, nz)
    """
    h, k, l = indices
    norm = math.sqrt(h**2 + k**2 + l**2)
    if norm < 1e-15:
        raise ValueError(f"零矢量不可归一化: ({h},{k},{l})")
    return (h / norm, k / norm, l / norm)


def dot_product_miller(v1, v2):
    """
    两个Miller指数矢量的点积 (使用方向余弦)

    v1 · v2 = |v1| |v2| cos(θ)

    Args:
        v1, v2: Miller指数元组

    Returns:
        float: 归一化点积 = cos(θ)
    """
    n1 = normalize_miller(v1)
    n2 = normalize_miller(v2)
    return sum(a * b for a, b in zip(n1, n2))


def cross_product_miller(v1, v2):
    """
    两个Miller指数矢量的叉积

    结果可能不是最小整数Miller指数，需要化简

    Args:
        v1, v2: Miller指数元组

    Returns:
        tuple: 叉积矢量 (非归一化)
    """
    h1, k1, l1 = v1
    h2, k2, l2 = v2
    return (
        k1 * l2 - l1 * k2,
        l1 * h2 - h1 * l2,
        h1 * k2 - k1 * h2,
    )


class SlipSystemDatabase:
    """
    滑移系数据库

    存储晶体的所有滑移系几何信息，提供Schmid因子计算、
    滑移系交互矩阵等功能。

    Attributes:
        material: 材料参数对象
        crystal_type: 晶体类型 ('FCC' 或 'BCC')
        n_systems: 滑移系总数
        plane_normals: 滑移面法线列表
        burgers_dirs: Burgers矢量方向列表
        burgers_magnitudes: 每个滑移系的Burgers矢量模
    """

    def __init__(self, material=None, crystal_type='FCC'):
        """
        初始化滑移系数据库

        Args:
            material: 材料参数对象 (默认使用Al)
            crystal_type: 晶体类型 ('FCC' 或 'BCC')
        """
        self.material = material or DEFAULT_MATERIAL
        self.crystal_type = crystal_type

        self.plane_normals = []
        self.burgers_dirs = []
        self.burgers_magnitudes = []
        self.schmid_tensors = []
        self.interaction_matrix = None

        if crystal_type == 'FCC':
            self._build_fcc_systems()
        elif crystal_type == 'BCC':
            self._build_bcc_systems()
        else:
            raise ValueError(f"不支持的晶体类型: {crystal_type}")

        self.n_systems = len(self.plane_normals)
        self._build_interaction_matrix()

    def _build_fcc_systems(self):
        """构建FCC {111}<110> 12个滑移系"""
        for p_idx, plane in enumerate(FCC_SLIP_PLANES):
            n_hat = normalize_miller(plane)

            for d_idx, b_dir in enumerate(FCC_SLIP_DIRECTIONS[p_idx]):
                b_hat = normalize_miller(b_dir)
                # 每个滑移系都有自己的法线记录 (虽然同面相同)
                self.plane_normals.append(n_hat)
                self.burgers_dirs.append(b_hat)

                # Burgers矢量大小: |b| = a/2 * <110> = a/√2
                b_mag = self.material.b_magnitude
                self.burgers_magnitudes.append(b_mag)

                # Schmid张量: P_sym = (1/2)(n̂ ⊗ b̂ + b̂ ⊗ n̂)
                P = self._schmid_tensor(n_hat, b_hat)
                self.schmid_tensors.append(P)

    def _build_bcc_systems(self):
        """构建BCC {110}<111> 滑移系 (简化: 12个主要系)"""
        count = 0
        for b_dir in BCC_SLIP_DIRECTIONS:
            for plane in BCC_SLIP_PLANES_110:
                # 检查 b·n = 0 (方向在面内)
                dot = sum(a * b for a, b in zip(b_dir, plane))
                if abs(dot) < 1e-10:
                    n_hat = normalize_miller(plane)
                    b_hat = normalize_miller(b_dir)
                    self.plane_normals.append(n_hat)
                    self.burgers_dirs.append(b_hat)

                    # BCC: |b| = a√3/2 <111>
                    b_mag = self.material.b_magnitude
                    self.burgers_magnitudes.append(b_mag)

                    P = self._schmid_tensor(n_hat, b_hat)
                    self.schmid_tensors.append(P)
                    count += 1

    def _schmid_tensor(self, n_hat, b_hat):
        """
        计算Schmid张量 (对称化)

        P^sym_{ij} = (1/2)(n_i b_j + b_j n_i)

        用于计算分解剪应力:
        τ = σ : P^sym = Σᵢⱼ σᵢⱼ P^symᵢⱼ

        Args:
            n_hat: 滑移面法线 (3-tuple)
            b_hat: Burgers矢量方向 (3-tuple)

        Returns:
            list: 3×3 Schmid张量矩阵
        """
        P = [[0.0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                P[i][j] = 0.5 * (n_hat[i] * b_hat[j] + b_hat[j] * n_hat[i])
        return P

    def _build_interaction_matrix(self):
        """
        构建滑移系交互矩阵 (位错反应强度)

        基于Franciosi等 (1980) 的位错交互分类:
        - 共面系 (coplanar): a₁ (最强)
        - 共Burgers矢量 (colinear): a₂
        - 交叉滑移 (cross-slip): a₃
        - 无交互: 0

        交互强度参数 (对FCC Al):
        a₁ = 0.10 (共面交截)
        a₂ = 0.15 (共线交截)
        a₃ = 0.08 (交叉交互)
        """
        n = self.n_systems
        self.interaction_matrix = [[0.0]*n for _ in range(n)]

        # 交互系数
        a_coplanar = 0.10
        a_colinear = 0.15
        a_cross = 0.08

        for alpha in range(n):
            for beta in range(n):
                if alpha == beta:
                    self.interaction_matrix[alpha][beta] = 1.0  # 自硬化
                    continue

                n_a = self.plane_normals[alpha]
                n_b = self.plane_normals[beta]
                b_a = self.burgers_dirs[alpha]
                b_b = self.burgers_dirs[beta]

                # 判断交互类型
                # 共面: 同一滑移面
                dot_n = sum(n_a[k] * n_b[k] for k in range(3))
                # 共线: 同一Burgers矢量方向
                dot_b = sum(b_a[k] * b_b[k] for k in range(3))

                if abs(abs(dot_n) - 1.0) < 1e-10:
                    # 共面系
                    self.interaction_matrix[alpha][beta] = a_coplanar
                elif abs(abs(dot_b) - 1.0) < 1e-10:
                    # 共Burgers矢量 (colinear junction)
                    self.interaction_matrix[alpha][beta] = a_colinear
                else:
                    # 一般交叉交互
                    self.interaction_matrix[alpha][beta] = a_cross

    def schmid_factor(self, load_direction, slip_system_idx):
        """
        计算给定载荷方向的Schmid因子

        m = cos(φ) cos(λ) = (n̂ · l̂)(b̂ · l̂)

        其中 l̂ 是载荷轴方向

        最大Schmid因子 = 0.5 (当 φ = λ = 45°)

        Args:
            load_direction: 载荷方向 Miller指数
            slip_system_idx: 滑移系索引

        Returns:
            float: Schmid因子 (0 ≤ m ≤ 0.5)
        """
        l_hat = normalize_miller(load_direction)
        n_hat = self.plane_normals[slip_system_idx]
        b_hat = self.burgers_dirs[slip_system_idx]

        cos_phi = sum(n_hat[k] * l_hat[k] for k in range(3))
        cos_lambda = sum(b_hat[k] * l_hat[k] for k in range(3))

        return abs(cos_phi * cos_lambda)

    def max_schmid_factor(self, load_direction):
        """
        查找所有滑移系中的最大Schmid因子

        对应最先激活的滑移系 ( Schmid定律)

        Args:
            load_direction: 载荷方向

        Returns:
            tuple: (max_m, active_system_idx)
        """
        max_m = 0.0
        active_idx = 0
        for alpha in range(self.n_systems):
            m = self.schmid_factor(load_direction, alpha)
            if m > max_m:
                max_m = m
                active_idx = alpha
        return max_m, active_idx

    def resolved_shear_stress(self, stress_tensor, slip_system_idx):
        """
        计算分解剪应力 (RSS)

        τ_α = σ : P^sym_α = Σᵢⱼ σᵢⱼ P^sym_α,ᵢⱼ

        Args:
            stress_tensor: 3×3应力张量
            slip_system_idx: 滑移系索引

        Returns:
            float: 分解剪应力 (Pa)
        """
        P = self.schmid_tensors[slip_system_idx]
        tau = 0.0
        for i in range(3):
            for j in range(3):
                tau += stress_tensor[i][j] * P[i][j]
        return tau


# ============================================================================
# 边界轨迹法 — 来自 104_boundary_locus
# 用于参数化晶界取向空间
# ============================================================================

class GrainBoundaryLocus:
    """
    晶界取向空间参数化

    使用边界轨迹法遍历晶界取向差空间:
    - 取向差角 θ ∈ [0, π/4] (对于立方晶体)
    - 旋转轴 [uvw] 在标准立体角三角形内

    晶界能量: γ(θ) = γ_max * θ * (A - ln(θ))  (Read-Shockley模型)

    其中 A = ln(θ_max) + 1, θ_max ≈ 15° (小角晶界极限)
    """

    def __init__(self, n_theta=36, n_axis=10):
        """
        初始化晶界取向空间

        Args:
            n_theta: 取向差角离散点数
            n_axis: 旋转轴离散点数
        """
        self.n_theta = n_theta
        self.n_axis = n_axis
        self.theta_max = 15.0 * PI / 180.0  # 15° → rad
        self.gamma_max = DEFAULT_MATERIAL.mu * DEFAULT_MATERIAL.b_magnitude / (4.0 * PI * SQRT2)

        self.boundary_points = []
        self._generate_locus()

    def _generate_locus(self):
        """生成边界轨迹上的取向差点"""
        # 立方晶体的标准立体角三角形顶点:
        # [100] → [110] → [111]
        vertices = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 1, 1),
        ]

        for i in range(self.n_theta + 1):
            theta = self.theta_max * i / self.n_theta
            for j in range(self.n_axis):
                # 在三角形内插值旋转轴
                t = j / max(self.n_axis - 1, 1)
                # 从 [100] 到 [110] 的线性插值 (简化)
                axis = (
                    vertices[0][0] + t * (vertices[1][0] - vertices[0][0]),
                    vertices[0][1] + t * (vertices[1][1] - vertices[0][1]),
                    vertices[0][2] + t * (vertices[1][2] - vertices[0][2]),
                )
                axis_norm = normalize_miller(axis)
                self.boundary_points.append({
                    'theta': theta,
                    'axis': axis_norm,
                    'energy': self.read_shockley_energy(theta),
                })

    def read_shockley_energy(self, theta):
        """
        Read-Shockley晶界能量模型

        γ(θ) = γ_max * (θ/θ_max) * (1 - ln(θ/θ_max))  for θ ≤ θ_max
        γ(θ) = γ_max                                      for θ > θ_max

        这是大角/小角晶界能量的经典插值

        Args:
            theta: 取向差角 (rad)

        Returns:
            float: 晶界能量 (J/m²)
        """
        if theta < 1e-15:
            return 0.0
        if theta >= self.theta_max:
            return self.gamma_max

        ratio = theta / self.theta_max
        energy = self.gamma_max * ratio * (1.0 - math.log(ratio))
        return max(0.0, energy)

    def frank_bilby_dislocation_content(self, theta, axis, plane_normal):
        """
        Frank-Bilby方程计算晶界位错含量

        B = (I - R) · p

        其中 R 是取向差旋转矩阵，p 是晶界面上的矢量

        对于小角晶界 (θ << 1):
        |B| ≈ θ * |p| * sin(χ)

        其中 χ 是旋转轴与晶界面的夹角

        位错间距: D = b / (2 sin(θ/2)) ≈ b / θ (小角近似)

        Args:
            theta: 取向差角 (rad)
            axis: 旋转轴 (单位矢量)
            plane_normal: 晶界面法线 (单位矢量)

        Returns:
            dict: 晶界位错含量信息
        """
        # 旋转矩阵 (Rodrigues公式)
        # R = I cos θ + (1-cos θ) k⊗k + sin θ [k]×
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        kx, ky, kz = axis

        R = [[0.0]*3 for _ in range(3)]
        R[0][0] = cos_t + kx*kx*(1-cos_t)
        R[0][1] = kx*ky*(1-cos_t) - kz*sin_t
        R[0][2] = kx*kz*(1-cos_t) + ky*sin_t
        R[1][0] = ky*kx*(1-cos_t) + kz*sin_t
        R[1][1] = cos_t + ky*ky*(1-cos_t)
        R[1][2] = ky*kz*(1-cos_t) - kx*sin_t
        R[2][0] = kz*kx*(1-cos_t) - ky*sin_t
        R[2][1] = kz*ky*(1-cos_t) + kx*sin_t
        R[2][2] = cos_t + kz*kz*(1-cos_t)

        # B = (I - R) · p (取 p 为面内某方向)
        # 简化: 取 p ⊥ axis, p ⊥ plane_normal
        p = cross_product_miller(
            tuple(axis), tuple(plane_normal)
        )
        p_norm = math.sqrt(sum(x**2 for x in p))
        if p_norm < 1e-15:
            # p 平行于 axis 或 plane_normal，取替代方向
            p = (1.0, 0.0, 0.0)
            p_norm = 1.0

        p_hat = tuple(x / p_norm for x in p)

        # B = p - R·p
        Rp = [sum(R[i][j] * p_hat[j] for j in range(3)) for i in range(3)]
        B = [p_hat[i] - Rp[i] for i in range(3)]
        B_mag = math.sqrt(sum(x**2 for x in B))

        # 位错间距
        b = DEFAULT_MATERIAL.b_magnitude
        if B_mag > 1e-15:
            D_spacing = b / B_mag
        else:
            D_spacing = float('inf')

        return {
            'burgers_content': B,
            'content_magnitude': B_mag,
            'dislocation_spacing': D_spacing,
            'rotation_matrix': R,
        }

    def get_all_boundary_energies(self):
        """获取所有边界轨迹点的能量"""
        return [pt['energy'] for pt in self.boundary_points]


if __name__ == '__main__':
    print("=" * 70)
    print("晶体格点与滑移系分析")
    print("=" * 70)

    # 测试CRT索引
    print("\n中国剩余定理 - 滑移系索引:")
    for p in range(4):
        for d in range(3):
            idx = slip_system_index(p, d)
            p_dec, d_dec = decode_slip_system(idx)
            print(f"  面{p}, 方向{d} → 索引{idx} → 解码({p_dec},{d_dec})")

    # 测试滑移系数据库
    print("\nFCC滑移系 (Al):")
    db = SlipSystemDatabase(crystal_type='FCC')
    print(f"  滑移系数: {db.n_systems}")
    for i in range(db.n_systems):
        n = db.plane_normals[i]
        b = db.burgers_dirs[i]
        print(f"  系{i}: n=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f}) "
              f"b=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f})")

    # 测试Schmid因子
    load = (0, 0, 1)
    max_m, active = db.max_schmid_factor(load)
    print(f"\n沿[001]加载: 最大Schmid因子 = {max_m:.4f}, 激活系 = {active}")

    # 测试晶界轨迹
    print("\nRead-Shockley晶界能量:")
    gb = GrainBoundaryLocus(n_theta=5, n_axis=3)
    for theta_deg in [1, 5, 10, 15]:
        theta_rad = theta_deg * PI / 180.0
        e = gb.read_shockley_energy(theta_rad)
        print(f"  θ = {theta_deg}°: γ = {e:.4f} J/m²")
