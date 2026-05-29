"""
lattice_builder.py
晶体晶格构建与初始化模块：BCC/FCC晶格 + 液相随机结构 + 固液界面构型

融合种子项目：
- 1233_tet_mesh_l2q: 四面体网格拓扑升阶思想 → 晶格近邻关系构建
- 1373_uniform: 伪随机数生成 → 初始速度、液相原子位置扰动
"""

import numpy as np
from utils_numeric import RandomState


class LatticeBuilder:
    """构建合金晶体/液体/界面初始构型。"""

    def __init__(self, lattice_type="fcc", a0=3.52, rng_seed=None):
        """
        参数:
            lattice_type: "fcc", "bcc", "hcp"
            a0: 晶格常数 (Angstrom)
        """
        self.lattice_type = lattice_type.lower()
        self.a0 = float(a0)
        self.rng = RandomState(seed=rng_seed)

    def build_fcc_block(self, nx, ny, nz):
        """构建FCC晶格块，返回基元位置 (nx*ny*nz*4, 3)。
        FCC基元坐标 (以晶格常数a0为单位):
            (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        """
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]) * self.a0

        positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    origin = np.array([ix, iy, iz]) * self.a0
                    for b in basis:
                        positions.append(origin + b)
        return np.array(positions, dtype=np.float64)

    def build_bcc_block(self, nx, ny, nz):
        """构建BCC晶格块。"""
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]) * self.a0
        positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    origin = np.array([ix, iy, iz]) * self.a0
                    for b in basis:
                        positions.append(origin + b)
        return np.array(positions, dtype=np.float64)

    def build_liquid_block(self, n_atoms, box):
        """在盒子内生成液相近似结构：随机位置但保证最小距离排斥。
        采用随机投放+拒绝采样法，融合1373_uniform。"""
        positions = np.zeros((n_atoms, 3), dtype=np.float64)
        min_dist2 = (0.7 * self.a0) ** 2
        max_trials = 1000
        for i in range(n_atoms):
            for trial in range(max_trials):
                pos = self.rng.uniform_ab(3, 0.0, 1.0) * box
                if i == 0:
                    positions[i] = pos
                    break
                diff = positions[:i] - pos
                diff -= box * np.round(diff / box)
                d2 = np.sum(diff ** 2, axis=1)
                if np.min(d2) > min_dist2:
                    positions[i] = pos
                    break
            else:
                # 若拒绝采样失败，采用简单均匀分布
                positions[i] = self.rng.uniform_ab(3, 0.0, 1.0) * box
        return positions

    def build_solid_liquid_interface(self, n_solid_x=4, n_solid_y=4, n_solid_z=4,
                                      n_liquid_z=4, y_span=None, z_span=None,
                                      concentration_b=0.15, randomize_solute=True):
        """
        构建固液双相界面初始构型。
        固体区域: z < 0 (或前半部分)
        液体区域: z > 0 (或后半部分)
        界面位于 z = 0 附近。
        
        返回:
            positions: (N, 3)
            species_idx: (N,)  0=溶剂A, 1=溶质B
            box: (3,)
            is_solid: (N,) bool 标记固相原子
        """
        if self.lattice_type == "fcc":
            solid_pos = self.build_fcc_block(n_solid_x, n_solid_y, n_solid_z)
        elif self.lattice_type == "bcc":
            solid_pos = self.build_bcc_block(n_solid_x, n_solid_y, n_solid_z)
        else:
            solid_pos = self.build_fcc_block(n_solid_x, n_solid_y, n_solid_z)

        # 盒子尺寸
        box_x = n_solid_x * self.a0
        if y_span is None:
            box_y = n_solid_y * self.a0
        else:
            box_y = y_span
        if z_span is None:
            box_z = (n_solid_z + n_liquid_z) * self.a0
        else:
            box_z = z_span
        box = np.array([box_x, box_y, box_z], dtype=np.float64)

        # 液体区域原子数 ≈ 固体区域原子数 * (液体密度/固体密度)
        # 简单假设密度相近
        n_solid = solid_pos.shape[0]
        n_liquid = int(n_solid * (n_liquid_z / n_solid_z))
        liquid_pos = self.build_liquid_block(n_liquid, box)
        # 将液体原子平移到固体上方
        liquid_pos[:, 2] += n_solid_z * self.a0

        positions = np.vstack([solid_pos, liquid_pos])
        n_total = positions.shape[0]

        # 标记固相
        is_solid = np.zeros(n_total, dtype=bool)
        is_solid[:n_solid] = True

        # 分配溶质B (species_idx=1)
        species_idx = np.zeros(n_total, dtype=np.int32)
        if randomize_solute:
            # 固相中溶质浓度
            n_solute_solid = int(concentration_b * n_solid)
            solid_indices = np.arange(n_solid)
            rng = np.random.default_rng(42)
            solute_solid = rng.choice(solid_indices, size=n_solute_solid, replace=False)
            species_idx[solute_solid] = 1

            # 液相中溶质浓度 (通常较高)
            liquid_conc = min(concentration_b * 1.5, 0.5)
            n_solute_liquid = int(liquid_conc * n_liquid)
            liquid_indices = np.arange(n_solid, n_total)
            solute_liquid = rng.choice(liquid_indices, size=n_solute_liquid, replace=False)
            species_idx[solute_liquid] = 1
        else:
            n_solute = int(concentration_b * n_total)
            rng = np.random.default_rng(42)
            solute_idx = rng.choice(np.arange(n_total), size=n_solute, replace=False)
            species_idx[solute_idx] = 1

        return positions, species_idx, box, is_solid

    def build_neighbor_list(self, positions, box, r_cut):
        """构建近邻列表，融合tet_mesh_l2q的拓扑关系思想。
        返回 list of (neighbor_indices, distances, distance_vectors)。"""
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
                if r2 < r_cut2:
                    neigh.append((j, np.sqrt(r2), rij))
            neighbors.append(neigh)
        return neighbors

    def apply_thermal_displacement(self, positions, T, masses_amu, species_idx):
        """根据Einstein模型施加热位移: u_rms = sqrt(3 k_B T / k) ~ 0.05 A @ 300K。"""
        kb = 8.617333e-5  # eV/K
        # 简谐近似: k ~ 20 eV/A^2
        k_spring = 20.0
        u_rms = np.sqrt(3.0 * kb * T / k_spring)
        n_atoms = positions.shape[0]
        disp = self.rng.maxwell_boltzmann(n_atoms, T=0.01, m=1.0) * u_rms / 0.1
        positions_new = positions + disp
        return positions_new
