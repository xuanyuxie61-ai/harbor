"""
eam_potential.py
嵌入原子法(EAM)势能与力场计算模块

融合种子项目：
- 463_gegenbauer_rule: 正交多项式基函数用于势函数拟合
- 641_laguerre_polynomial / 466_gen_laguerre_exactness: 径向函数展开基

EAM总能量:
    E_{total} = \sum_i F_i(\bar{\rho}_i) + \frac{1}{2}\sum_{i \neq j} \phi_{ij}(r_{ij})
其中:
    \bar{\rho}_i = \sum_{j \neq i} \rho_j(r_{ij})  (嵌入密度)
    F_i(\rho) = A_i \sqrt{\rho} + B_i \rho^2  (Morse-type嵌入函数)
    \phi_{ij}(r) = D_{ij} [ e^{-2\alpha_{ij}(r-r_0)} - 2e^{-\alpha_{ij}(r-r_0)} ]  (Morse对势)
    \rho_j(r) = f_j \left( \frac{r}{r_{cut}} \right)^6 e^{-\beta_j (r - r_{cut})}  (电子密度函数)
"""

import numpy as np
from utils_numeric import check_bounds, safe_sqrt


class EAMPotential:
    """二元合金EAM势参数化与力场计算。"""

    def __init__(self, species_a="A", species_b="B",
                 D_aa=0.50, D_bb=0.35, D_ab=0.42,
                 alpha_aa=1.60, alpha_bb=1.45, alpha_ab=1.52,
                 r0_aa=2.56, r0_bb=2.78, r0_ab=2.67,
                 A_a=0.80, A_b=0.65, B_a=-0.02, B_b=-0.015,
                 f_a=1.0, f_b=0.85, beta_a=3.0, beta_b=2.8,
                 r_cut=5.5, mass_a=58.69, mass_b=63.55):
        """
        初始化二元合金EAM势参数。
        默认参数近似于Ni-Cu体系(原子量相近，晶格常数匹配)。
        """
        self.species = [species_a, species_b]
        self.r_cut = float(r_cut)
        self.r_cut2 = self.r_cut ** 2
        self.mass = np.array([mass_a, mass_b], dtype=np.float64)

        # Morse对势参数矩阵 D[si, sj]
        self.D_mat = np.array([[D_aa, D_ab], [D_ab, D_bb]], dtype=np.float64)
        self.alpha_mat = np.array([[alpha_aa, alpha_ab], [alpha_ab, alpha_bb]], dtype=np.float64)
        self.r0_mat = np.array([[r0_aa, r0_ab], [r0_ab, r0_bb]], dtype=np.float64)

        # 嵌入函数参数 F(rho) = A*sqrt(rho) + B*rho^2
        self.A_emb = np.array([A_a, A_b], dtype=np.float64)
        self.B_emb = np.array([B_a, B_b], dtype=np.float64)

        # 电子密度函数参数 rho(r) = f * (r/r_cut)^6 * exp(-beta*(r - r_cut)) for r < r_cut
        self.f_rho = np.array([f_a, f_b], dtype=np.float64)
        self.beta_rho = np.array([beta_a, beta_b], dtype=np.float64)

        # 预计算势能与力的截断平滑函数系数
        self._compute_cutoff_smoothing()

    def _compute_cutoff_smoothing(self):
        """构造截断平滑函数 s(r) = (1 - r/r_cut)^3 * H(r_cut - r)，
        保证势能在截断处一阶导数连续。"""
        pass  # 在计算中直接应用

    def pair_potential(self, r, si, sj):
        """Morse对势 phi_{si,sj}(r)，含截断平滑。"""
        r = check_bounds(r, 1e-8, self.r_cut, name="pair_distance")
        D = self.D_mat[si, sj]
        alpha = self.alpha_mat[si, sj]
        r0 = self.r0_mat[si, sj]
        x = np.exp(-alpha * (r - r0))
        phi = D * (x * x - 2.0 * x)
        # 截断平滑: 当 r -> r_cut 时势能为0
        s = (1.0 - r / self.r_cut) ** 3
        s = np.where(r < self.r_cut, s, 0.0)
        return phi * s

    def pair_force_magnitude(self, r, si, sj):
        """对势产生的径向力大小 f = -dphi/dr (正值为排斥)。"""
        r = check_bounds(r, 1e-8, self.r_cut, name="pair_distance")
        D = self.D_mat[si, sj]
        alpha = self.alpha_mat[si, sj]
        r0 = self.r0_mat[si, sj]
        x = np.exp(-alpha * (r - r0))
        dphi_dr = D * (-2.0 * alpha * x * x + 2.0 * alpha * x)
        # 截断平滑函数的导数贡献
        s = (1.0 - r / self.r_cut) ** 3
        ds_dr = -3.0 * (1.0 - r / self.r_cut) ** 2 / self.r_cut
        phi = D * (x * x - 2.0 * x)
        mask = r < self.r_cut
        total = np.where(mask, dphi_dr * s + phi * ds_dr, 0.0)
        return -total  # 返回 -dphi/dr

    def electron_density(self, r, sj):
        """原子j在距离r处产生的电子密度 rho_j(r)。"""
        r = check_bounds(r, 1e-8, self.r_cut, name="rho_distance")
        f = self.f_rho[sj]
        beta = self.beta_rho[sj]
        val = f * (r / self.r_cut) ** 6 * np.exp(-beta * (r - self.r_cut))
        mask = r < self.r_cut
        return np.where(mask, val, 0.0)

    def embedding_energy(self, rho_bar, si):
        """嵌入能 F_{si}(\bar{\rho})。"""
        rho_bar = np.maximum(rho_bar, 1e-12)
        return self.A_emb[si] * safe_sqrt(rho_bar) + self.B_emb[si] * rho_bar ** 2

    def embedding_derivative(self, rho_bar, si):
        """dF/d\bar{\rho}。"""
        rho_bar = np.maximum(rho_bar, 1e-12)
        return 0.5 * self.A_emb[si] / safe_sqrt(rho_bar) + 2.0 * self.B_emb[si] * rho_bar

    def compute_forces_and_energies(self, positions, species_idx, box):
        """
        计算系统总能量与原子受力。

        参数:
            positions: (N, 3) 原子位置
            species_idx: (N,) 原子种类索引 (0或1)
            box: (3,) 模拟盒子边长
        返回:
            total_energy: 总势能
            forces: (N, 3) 原子受力
            virial: 维里应力标量
        """
        n_atoms = positions.shape[0]
        forces = np.zeros_like(positions)
        energies = np.zeros(n_atoms, dtype=np.float64)

        # 构建近邻列表 (简单O(N^2)版本，带截断)
        rho_bar = np.zeros(n_atoms, dtype=np.float64)

        # 第一步：计算电子密度
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                rij = positions[j] - positions[i]
                rij -= box * np.round(rij / box)  # 最小镜像约定
                r2 = np.dot(rij, rij)
                if r2 < self.r_cut2:
                    r = safe_sqrt(r2)
                    sj = species_idx[j]
                    si = species_idx[i]
                    rho_bar[i] += self.electron_density(r, sj)
                    rho_bar[j] += self.electron_density(r, si)

        # 嵌入能贡献
        for i in range(n_atoms):
            si = species_idx[i]
            energies[i] += self.embedding_energy(rho_bar[i], si)

        # 第二步：计算对势与总力
        virial = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                rij = positions[j] - positions[i]
                rij -= box * np.round(rij / box)
                r2 = np.dot(rij, rij)
                if r2 < self.r_cut2:
                    r = safe_sqrt(r2)
                    si = species_idx[i]
                    sj = species_idx[j]
                    # 对势
                    phi = self.pair_potential(r, si, sj)
                    energies[i] += 0.5 * phi
                    energies[j] += 0.5 * phi

                    # 对势力
                    f_pair = self.pair_force_magnitude(r, si, sj)
                    f_vec = f_pair * rij / r
                    forces[i] += f_vec
                    forces[j] -= f_vec
                    virial += np.dot(rij, f_vec)

                    # TODO: Hole 1 — 实现EAM嵌入力计算
                    # 需要计算 dF/drho * drho/dr 的链式导数，并组装为矢量力
                    # 提示: 使用 self.embedding_derivative, self.f_rho, self.beta_rho
                    pass

        total_energy = np.sum(energies)
        return total_energy, forces, virial


def eam_parameterized_alloy(species_pair=("Ni", "Cu")):
    """返回参数化的Ni-Cu或Fe-Ni EAM势实例。"""
    if species_pair == ("Ni", "Cu"):
        return EAMPotential(
            species_a="Ni", species_b="Cu",
            D_aa=0.4207, D_bb=0.3429, D_ab=0.3818,
            alpha_aa=1.5590, alpha_bb=1.3885, alpha_ab=1.4738,
            r0_aa=2.7800, r0_bb=2.5560, r0_ab=2.6180,
            A_a=1.100, A_b=0.900, B_a=-0.018, B_b=-0.012,
            f_a=1.00, f_b=0.82, beta_a=2.80, beta_b=2.60,
            r_cut=5.50, mass_a=58.69, mass_b=63.55
        )
    else:
        return EAMPotential()
