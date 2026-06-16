"""
crystal_descriptor.py
=====================
晶体结构编码与描述符生成模块。

科学背景:
  材料基因组计划 (MGI) 的核心是将晶体结构转化为可计算的数值描述符。
  本模块实现:
    1. 晶体几何编码 (源自 1266_ansysresearch_geometry_encoding)
       - 将晶格参数 (a,b,c,alpha,beta,gamma) 编码为特征向量
       - 使用 Coulomb matrix 和 Sine matrix 表示
    2. Wyckoff 位置对称性编码
       - 空间群信息压缩为低维特征
    3. 化学组成描述符
       - 元素属性加权平均

晶体学基础:
  三斜晶系: a!=b!=c, alpha!=beta!=gamma!=90
  立方晶系: a=b=c, alpha=beta=gamma=90 (如 LLZO 的 Ia-3d 空间群)
  六方晶系: a=b!=c, alpha=beta=90, gamma=120

晶格体积 (三斜通式):
  V = a*b*c * sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma)
       + 2*cos(alpha)*cos(beta)*cos(gamma))

倒格矢:
  b1 = 2*pi * (a2 x a3) / V, 循环置换得 b2, b3

Brillouin 区体积:
  V_BZ = (2*pi)^3 / V

Coulomb matrix 元素:
  M_ij = 0.5 * Z_i * Z_j^2.4 / |R_i - R_j|  (i != j)
  M_ii = 0.5 * Z_i^2.4

Sine matrix (周期性):
  S_ij = 0.5 * Z_i * Z_j / sin^2(pi * (R_i - R_j) / a)  (i != j)
"""

import numpy as np
from material_constants import LLZO_LATTICE_PARAM, SMALL_NUMBER


# 常见元素原子序数 (LLZO 相关)
ATOMIC_NUMBERS = {
    'H': 1, 'Li': 3, 'La': 57, 'Zr': 40, 'O': 8,
    'Al': 13, 'Ga': 31, 'Ta': 73, 'Nb': 41, 'Y': 39,
}

# 元素原子质量 [amu]
ATOMIC_MASSES = {
    'Li': 6.94, 'La': 138.91, 'Zr': 91.22, 'O': 16.00,
    'Al': 26.98, 'Ga': 69.72, 'Ta': 180.95, 'Nb': 92.91,
}

# 元素电负性 (Pauling scale)
ELECTRONEGATIVITY = {
    'Li': 0.98, 'La': 1.10, 'Zr': 1.33, 'O': 3.44,
    'Al': 1.61, 'Ga': 1.81, 'Ta': 1.50, 'Nb': 1.60,
}

# 元素离子半径 [Angstrom]
IONIC_RADII = {
    'Li': 0.76, 'La': 1.03, 'Zr': 0.72, 'O': 1.40,
    'Al': 0.54, 'Ga': 0.62, 'Ta': 0.64, 'Nb': 0.64,
}


def lattice_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    三斜晶胞体积:
      V = abc*sqrt(1 - ca^2 - cb^2 - cg^2 + 2*ca*cb*cg)
    其中 ca=cos(alpha), cb=cos(beta), cg=cos(gamma)
    """
    ca = np.cos(np.radians(alpha_deg))
    cb = np.cos(np.radians(beta_deg))
    cg = np.cos(np.radians(gamma_deg))
    arg = 1.0 - ca**2 - cb**2 - cg**2 + 2.0*ca*cb*cg
    arg = max(arg, SMALL_NUMBER)
    return a * b * c * np.sqrt(arg)


def metric_tensor(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    度量张量 G_ij = a_i . a_j:

        | a^2       ab*cos(gamma)  ac*cos(beta)  |
    G = | ab*cos(gamma)  b^2       bc*cos(alpha)|
        | ac*cos(beta)   bc*cos(alpha)  c^2      |

    其行列式 det(G) = V^2
    """
    ca = np.cos(np.radians(alpha_deg))
    cb = np.cos(np.radians(beta_deg))
    cg = np.cos(np.radians(gamma_deg))
    G = np.array([
        [a*a,     a*b*cg,  a*c*cb],
        [a*b*cg,  b*b,     b*c*ca],
        [a*c*cb,  b*c*ca,  c*c  ],
    ])
    return G


def reciprocal_lattice(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    计算倒格矢 b1, b2, b3:
      b1 = 2*pi*(a2 x a3)/V, b2 = 2*pi*(a3 x a1)/V, b3 = 2*pi*(a1 x a2)/V

    实空间基矢 (以 a1 沿 x 轴为约定):
      a1 = (a, 0, 0)
      a2 = (b*cos(gamma), b*sin(gamma), 0)
      a3 = (c*cos(beta), c*cos(alpha*)*sin(gamma), c*cos(alpha*)*cos(gamma)...)

    简化形式: 直接通过 G^{-1} 得到倒格矢的度量张量 G* = (2*pi)^2 * G^{-1}
    """
    G = metric_tensor(a, b, c, alpha_deg, beta_deg, gamma_deg)
    G_inv = np.linalg.inv(G)
    G_star = (2.0 * np.pi)**2 * G_inv
    return G_star


def coulomb_matrix(positions, atomic_numbers, max_atoms=20):
    """
    Coulomb matrix 表示 (Rupp et al. 2012):

    M_ij = 0.5 * Z_i * Z_j^2.4 / |R_i - R_j|  (i != j)
    M_ii = 0.5 * Z_i^2.4

    返回排序后的特征值作为描述符 (排列不变性)
    """
    n = min(len(positions), max_atoms)
    M = np.zeros((max_atoms, max_atoms))
    for i in range(n):
        Zi = atomic_numbers[i]
        M[i, i] = 0.5 * Zi**2.4
        for j in range(i + 1, n):
            Zj = atomic_numbers[j]
            rij = np.linalg.norm(positions[i] - positions[j])
            rij = max(rij, SMALL_NUMBER)
            M[i, j] = 0.5 * Zi * Zj**2.4 / rij
            M[j, i] = M[i, j]
    eigvals = np.linalg.eigvalsh(M)
    return np.sort(eigvals)[::-1]


def sine_matrix(positions, atomic_numbers, lattice_vectors, max_atoms=20):
    """
    Sine matrix (Brocker et al. 2016) 包含周期性边界条件:

    S_ij = 0.5 * Z_i * Z_j / ||sin(pi * P_ij * L^{-1})||^2

    其中 P_ij = R_i - R_j 为笛卡尔坐标差, L 为晶格矩阵

    此描述符对平移和周期镜像具有不变性
    """
    n = min(len(positions), max_atoms)
    S = np.zeros((max_atoms, max_atoms))
    Linv = np.linalg.inv(lattice_vectors)
    for i in range(n):
        Zi = atomic_numbers[i]
        S[i, i] = 0.5 * Zi**2.4
        for j in range(i + 1, n):
            Zj = atomic_numbers[j]
            dR = positions[i] - positions[j]
            frac = dR @ Linv  # 分数坐标差
            sin_term = np.sin(np.pi * frac)
            norm_sin = np.linalg.norm(sin_term)
            norm_sin = max(norm_sin, SMALL_NUMBER)
            S[i, j] = 0.5 * Zi * Zj / (norm_sin**2)
            S[j, i] = S[i, j]
    eigvals = np.linalg.eigvalsh(S)
    return np.sort(eigvals)[::-1]


def composition_descriptor(formula_dict):
    """
    组成描述符: 加权平均元素属性

    对于化合物 A_x B_y C_z:
      f_avg = (x*f_A + y*f_B + z*f_C) / (x+y+z)
      f_var = (x*(f_A-f_avg)^2 + y*(f_B-f_avg)^2 + z*(f_C-f_avg)^2) / (x+y+z)

    返回8维特征向量:
      [avg_mass, var_mass, avg_en, var_en, avg_r, var_r, avg_Z, packing_fraction]
    """
    total_atoms = sum(formula_dict.values())
    if total_atoms == 0:
        return np.zeros(8)

    masses, ens, radii, Zs = [], [], [], []
    for elem, count in formula_dict.items():
        for _ in range(count):
            masses.append(ATOMIC_MASSES.get(elem, 0.0))
            ens.append(ELECTRONEGATIVITY.get(elem, 0.0))
            radii.append(IONIC_RADII.get(elem, 0.0))
            Zs.append(ATOMIC_NUMBERS.get(elem, 1))

    masses = np.array(masses)
    ens = np.array(ens)
    radii = np.array(radii)
    Zs = np.array(Zs, dtype=np.float64)

    avg_mass = np.mean(masses)
    var_mass = np.var(masses)
    avg_en = np.mean(ens)
    var_en = np.var(ens)
    avg_r = np.mean(radii)
    var_r = np.var(radii)
    avg_Z = np.mean(Zs)

    # 堆积分数估计 (硬球近似)
    V_atom_total = np.sum((4.0/3.0) * np.pi * (radii * 1e-10)**3)
    packing_fraction = V_atom_total * 1e30  # 相对量

    return np.array([
        avg_mass, var_mass, avg_en, var_en,
        avg_r, var_r, avg_Z, packing_fraction,
    ])


def llzo_unit_cell(a_lat=None):
    """
    构建 LLZO 立方晶胞 (Ia-3d, #230) 的简化原子位置。

    立方晶系: a=b=c, alpha=beta=gamma=90
    晶胞含 8 个 Li7La3Zr2O12 公式单元, 共 480 个原子

    这里返回简化的代表性 Wyckoff 位置子集用于描述符计算
    """
    if a_lat is None:
        a_lat = LLZO_LATTICE_PARAM

    # 简化: La (24c), Zr (16a), O (96h), Li (48g) + (24d)
    positions = []
    atomic_numbers = []

    # La at 24c Wyckoff position (简化为3个代表位)
    la_frac = np.array([
        [0.125, 0.0, 0.25],
        [0.0, 0.25, 0.125],
        [0.25, 0.125, 0.0],
    ])
    for pos in la_frac:
        positions.append(pos * a_lat)
        atomic_numbers.append(ATOMIC_NUMBERS['La'])

    # Zr at 16a
    zr_frac = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
    ])
    for pos in zr_frac:
        positions.append(pos * a_lat)
        atomic_numbers.append(ATOMIC_NUMBERS['Zr'])

    # O at 96h (代表性位置)
    o_frac = np.array([
        [0.25, 0.1, 0.15],
        [0.15, 0.25, 0.1],
        [0.1, 0.15, 0.25],
        [0.75, 0.9, 0.85],
    ])
    for pos in o_frac:
        positions.append(pos * a_lat)
        atomic_numbers.append(ATOMIC_NUMBERS['O'])

    # Li at 48g (迁移活性位点)
    li_frac = np.array([
        [0.5, 0.25, 0.1],
        [0.1, 0.5, 0.25],
        [0.25, 0.1, 0.5],
    ])
    for pos in li_frac:
        positions.append(pos * a_lat)
        atomic_numbers.append(ATOMIC_NUMBERS['Li'])

    return np.array(positions), np.array(atomic_numbers), a_lat


def band_structure_descriptor(k_points, energy_bands):
    """
    从能带结构提取描述符:

    带隙 E_g = min(E_CB) - max(E_VB)
    有效质量 m* = hbar^2 / (d^2E/dk^2)
    带宽度 W = max(E) - min(E) 对每个带

    参数:
        k_points: [N_k] 波矢点 [1/A]
        energy_bands: [N_bands, N_k] 能量 [eV]
    """
    n_bands, n_k = energy_bands.shape
    descriptors = []

    # 带隙估计 (假设绝缘体, 最高占据带和最低未占据带之间)
    mid_band = n_bands // 2
    if mid_band > 0 and mid_band < n_bands:
        ev_max = np.max(energy_bands[:mid_band, :])
        ec_min = np.min(energy_bands[mid_band:, :])
        band_gap = ec_min - ev_max
    else:
        band_gap = 0.0
    descriptors.append(band_gap)

    # 带宽统计
    bandwidths = np.max(energy_bands, axis=1) - np.min(energy_bands, axis=1)
    descriptors.extend([np.mean(bandwidths), np.max(bandwidths), np.min(bandwidths)])

    # 有效质量估计 (在 Gamma 点附近)
    if n_k >= 3:
        dk = k_points[1] - k_points[0]
        d2E_dk2 = (energy_bands[:, 2:] - 2*energy_bands[:, 1:-1] + energy_bands[:, :-2]) / (dk**2)
        # m* = hbar^2 / d2E, 转换单位
        hbar_eV = 6.582119569e-16  # eV*s
        m_eff_vals = []
        for b in range(n_bands):
            curv = np.abs(np.mean(d2E_dk2[b, :]))
            curv = max(curv, SMALL_NUMBER)
            m_eff = (hbar_eV**2) / curv  # 约化质量单位
            m_eff_vals.append(m_eff)
        descriptors.extend([np.mean(m_eff_vals), np.min(m_eff_vals)])
    else:
        descriptors.extend([0.0, 0.0])

    return np.array(descriptors)
