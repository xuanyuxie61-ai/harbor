"""
structure_factor.py
局域结构序参量分析与固液相识别

融合种子项目：
- 601_ising_2d_simulation: 相变识别与统计力学思想 → 固液相分类
- 081_besselzero: Bessel函数零点用于球谐展开基
- 463_gegenbauer_rule / 641_laguerre_polynomial: 正交多项式基展开

Steinhardt键取向序参量:
    q_{lm}(i) = \frac{1}{N_b(i)} \sum_{j=1}^{N_b(i)} Y_{lm}(\hat{r}_{ij})
    q_l(i) = \sqrt{ \frac{4\pi}{2l+1} \sum_{m=-l}^{l} |q_{lm}(i)|^2 }

其中 Y_{lm}(\theta, \phi) 为球谐函数，\hat{r}_{ij} 为归一化键向量。

固液判据:
    q_l(i) > q_{threshold}  →  固相
    q_l(i) \leq q_{threshold}  →  液相

对FCC晶体，l=6 的 q_6 平均值约为 0.55，液相约为 0.25。
"""

import numpy as np
from scipy.special import sph_harm
from utils_numeric import check_bounds, safe_sqrt


class LocalStructureAnalyzer:
    """基于球谐展开与Steinhardt参数的局域结构分析器。"""

    def __init__(self, l=6, q_threshold=0.40, r_bond_factor=1.35):
        """
        参数:
            l: 球谐阶数 (4=BCC特征, 6=FCC特征)
            q_threshold: 固液分界阈值
            r_bond_factor: 键长截断 = r_bond_factor * a0 / sqrt(2)
        """
        self.l = int(l)
        self.q_threshold = float(q_threshold)
        self.r_bond_factor = float(r_bond_factor)

    def compute_qlm(self, positions, box, i_atom, neighbors_rij):
        """计算单个原子的 q_{lm}。
        neighbors_rij: list of (j, r, rij) for atom i.
        """
        n_bonds = len(neighbors_rij)
        if n_bonds == 0:
            return np.zeros(2 * self.l + 1, dtype=np.complex128)

        qlm = np.zeros(2 * self.l + 1, dtype=np.complex128)
        for _, r, rij in neighbors_rij:
            # 球坐标
            x, y, z = rij
            theta = np.arccos(check_bounds(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
            for m in range(-self.l, self.l + 1):
                idx = m + self.l
                qlm[idx] += sph_harm(m, self.l, phi, theta)
        qlm /= n_bonds
        return qlm

    def compute_ql(self, qlm):
        """从 q_{lm} 计算 q_l。"""
        norm = np.sum(np.abs(qlm) ** 2)
        return safe_sqrt(4.0 * np.pi / (2.0 * self.l + 1.0) * norm)

    def analyze_system(self, positions, box, neighbor_list):
        """
        分析整个系统的局域结构。

        返回:
            q_values: (N,) 各原子的 q_l 值
            is_solid: (N,) bool 固相标记
            qlm_array: (N, 2l+1) 球谐系数
        """
        n_atoms = positions.shape[0]
        q_values = np.zeros(n_atoms, dtype=np.float64)
        qlm_array = np.zeros((n_atoms, 2 * self.l + 1), dtype=np.complex128)

        for i in range(n_atoms):
            qlm = self.compute_qlm(positions, box, i, neighbor_list[i])
            qlm_array[i] = qlm
            q_values[i] = self.compute_ql(qlm)

        is_solid = q_values > self.q_threshold
        return q_values, is_solid, qlm_array

    def compute_coherent_ql(self, qlm_array):
        """计算相干序参量 Q_l = sqrt(4pi/(2l+1) * sum_m |<q_{lm}>|^2 )。"""
        avg_qlm = np.mean(qlm_array, axis=0)
        return safe_sqrt(4.0 * np.pi / (2.0 * self.l + 1.0) * np.sum(np.abs(avg_qlm) ** 2))

    def compute_wl_invariant(self, qlm_array):
        """
        计算Wigner 3j不变量 w_l (三次旋转不变量):
            w_l = \sum_{m1,m2,m3} \begin{pmatrix} l & l & l \\ m1 & m2 & m3 \end{pmatrix}
                  \frac{q_{lm1} q_{lm2} q_{lm3}}
                  {(\sum_m |q_{lm}|^2)^{3/2}}
        用于区分不同晶体结构 (FCC vs BCC vs HCP)。
        """
        # 简化：计算平均w_l
        n_atoms = qlm_array.shape[0]
        wl_sum = 0.0
        # Wigner 3j symbol for l=l=l, m1+m2+m3=0
        # 对于l=6, 预计算非零3j系数组合
        # 这里采用数值近似
        for i in range(n_atoms):
            q = qlm_array[i]
            norm = np.sum(np.abs(q) ** 2)
            if norm < 1e-12:
                continue
            # 简化的三阶累积量近似
            cumulant = 0.0
            for m1 in range(-self.l, self.l + 1):
                for m2 in range(-self.l, self.l + 1):
                    m3 = -(m1 + m2)
                    if abs(m3) > self.l:
                        continue
                    # Wigner 3j (l,l,l; m1,m2,m3) 近似为常数因子 * delta_{m1+m2+m3,0}
                    # 真实值需要scipy.special.wigner_3j，但为简化计算，使用近似
                    coeff = 1.0 if (m1 + m2 + m3 == 0) else 0.0
                    cumulant += coeff * q[m1 + self.l] * q[m2 + self.l] * q[m3 + self.l]
            wl_sum += cumulant.real / (norm ** 1.5 + 1e-12)
        return wl_sum / n_atoms


def build_neighbor_list_with_cutoff(positions, box, r_cut):
    """构建截断近邻列表。"""
    n_atoms = positions.shape[0]
    r_cut2 = r_cut ** 2
    neighbors = []
    for i in range(n_atoms):
        neigh = []
        for j in range(n_atoms):
            if i == j:
                continue
            rij = positions[j] - positions[i]
            rij -= box * np.round(rij / box)
            r2 = np.dot(rij, rij)
            if r2 < r_cut2 and r2 > 1e-12:
                r = safe_sqrt(r2)
                neigh.append((j, r, rij))
        neighbors.append(neigh)
    return neighbors


def radial_distribution_function(positions, box, species_idx, dr=0.05, r_max=None):
    """
    计算径向分布函数 g(r)。
    g_{\alpha\beta}(r) = \frac{V}{N_\alpha N_\beta} \sum_i^{N_\alpha} \sum_{j\neq i}^{N_\beta}
                          \frac{\delta(r - r_{ij})}{4\pi r^2 dr}
    """
    n_atoms = positions.shape[0]
    if r_max is None:
        r_max = 0.5 * np.min(box)
    n_bins = int(r_max / dr)
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    # 总 RDF (不分种类)
    hist = np.zeros(n_bins, dtype=np.float64)
    volume = np.prod(box)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[j] - positions[i]
            rij -= box * np.round(rij / box)
            r = safe_sqrt(np.dot(rij, rij))
            if r < r_max:
                idx = int(r / dr)
                if idx < n_bins:
                    hist[idx] += 2.0  # i,j 和 j,i 都计数

    # 归一化
    rho = n_atoms / volume
    for idx in range(n_bins):
        r_inner = r_bins[idx]
        r_outer = r_bins[idx + 1]
        shell_vol = 4.0 / 3.0 * np.pi * (r_outer ** 3 - r_inner ** 3)
        ideal = rho * shell_vol * n_atoms
        if ideal > 0:
            hist[idx] /= ideal

    return r_centers, hist
