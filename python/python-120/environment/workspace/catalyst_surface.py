"""
catalyst_surface.py
催化剂表面晶格结构与吸附位点管理模块

整合原项目:
  - 148_cellular_automaton: 元胞自动机规则用于表面位点占据态演化
  - 249_cvt_3d_lumping: 三维 Centroidal Voronoi Tessellation 用于吸附原子最优采样
  - 1175_subpak: 多维网格生成

科学背景: Pt(111) 表面催化 CO 氧化反应
"""

import numpy as np
from typing import List, Tuple, Optional
from utils import (
    PT_LATTICE_CONSTANT, grid_uniform_nd, safe_divide,
    BOLTZMANN_KB, ELEMENTARY_CHARGE
)


class Pt111Surface:
    """
    Pt(111) 面心立方晶体表面模型
    
    Pt(111) 表面具有六方对称性，晶格常数 a = 3.924 Å
    表面最近邻原子间距: d_nn = a / sqrt(2) ≈ 2.775 Å
    """

    def __init__(self, nx: int = 6, ny: int = 6, n_layers: int = 3):
        if nx < 2 or ny < 2 or n_layers < 1:
            raise ValueError("nx, ny >= 2 且 n_layers >= 1")
        self.nx = nx
        self.ny = ny
        self.n_layers = n_layers
        self.a = PT_LATTICE_CONSTANT
        self.d_nn = self.a / np.sqrt(2.0)
        self.atoms = self._build_fcc_111()
        self.n_atoms = self.atoms.shape[0]
        # 表面位点类型: 0=top, 1=bridge, 2=fcc-hollow, 3=hcp-hollow
        self.sites = self._build_adsorption_sites()
        self.n_sites = self.sites.shape[0]
        # 元胞自动机状态: 0=空, 1=CO, 2=O
        self.site_occupancy = np.zeros(self.n_sites, dtype=int)
        # 位点占据能 (eV)
        self.site_energies = self._compute_site_energies()

    def _build_fcc_111(self) -> np.ndarray:
        """
        构建 FCC(111) 表面原子坐标
        
        (111) 面内基矢:
            a1 = (d_nn, 0, 0)
            a2 = (d_nn/2, d_nn*sqrt(3)/2, 0)
        
        层间堆垛: ABC 序列，层间距 d_z = a / sqrt(3)
        """
        d = self.d_nn
        dz = self.a / np.sqrt(3.0)
        atoms = []
        for layer in range(self.n_layers):
            z = -layer * dz
            offset = (layer % 3) * d / 3.0
            for i in range(self.nx):
                for j in range(self.ny):
                    x = i * d + (j % 2) * d * 0.5 + offset
                    y = j * d * np.sqrt(3.0) / 2.0
                    atoms.append([x, y, z])
        return np.array(atoms, dtype=float)

    def _build_adsorption_sites(self) -> np.ndarray:
        """
        构建高对称吸附位点坐标
        
        位点类型:
          - top: Pt 原子正上方
          - bridge: 两个 Pt 原子之间中点
          - fcc hollow: 三个 Pt 原子围成的空穴 (第二层无原子)
          - hcp hollow: 三个 Pt 原子围成的空穴 (第二层有原子)
        """
        d = self.d_nn
        sites = []
        # 仅在最顶层 (z=0) 构建吸附位点
        z_top = 1.5e-10  # 约 1.5 Å 上方
        for i in range(self.nx):
            for j in range(self.ny):
                x0 = i * d + (j % 2) * d * 0.5
                y0 = j * d * np.sqrt(3.0) / 2.0
                # top 位点
                sites.append([x0, y0, z_top, 0])
                # bridge 位点 (x + d/2)
                sites.append([x0 + d * 0.5, y0, z_top, 1])
                # fcc hollow
                sites.append([x0 + d * 0.5, y0 + d * np.sqrt(3.0) / 6.0, z_top, 2])
                # hcp hollow
                sites.append([x0 + d * 0.5, y0 - d * np.sqrt(3.0) / 6.0, z_top, 3])
        return np.array(sites, dtype=float)

    def _compute_site_energies(self) -> np.ndarray:
        """
        计算各吸附位点的相对能量
        
        CO on Pt(111):
          - top: E ≈ -1.3 eV (最稳定)
          - bridge: E ≈ -1.0 eV
          - fcc hollow: E ≈ -0.8 eV
          - hcp hollow: E ≈ -0.7 eV
        
        O on Pt(111):
          - fcc hollow: E ≈ -2.0 eV (最稳定)
          - hcp hollow: E ≈ -1.8 eV
          - bridge: E ≈ -1.2 eV
          - top: E ≈ -0.5 eV
        """
        energies = np.zeros(self.n_sites)
        site_types = self.sites[:, 3].astype(int)
        # 以 CO 吸附能为基准
        co_energy = {0: -1.3, 1: -1.0, 2: -0.8, 3: -0.7}
        for i, st in enumerate(site_types):
            energies[i] = co_energy.get(st, -0.5)
        return energies

    def get_surface_atoms(self) -> np.ndarray:
        """返回表面层原子坐标"""
        return self.atoms[self.atoms[:, 2] == 0.0]

    def find_nearest_site(self, pos: np.ndarray) -> int:
        """找到离给定位置最近的吸附位点索引"""
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,):
            raise ValueError("pos 必须为 3D 坐标")
        diffs = self.sites[:, :3] - pos
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        return int(np.argmin(dists))

    def update_occupancy_ca(self, rule: int = 30, steps: int = 1):
        """
        使用一维元胞自动机规则演化表面位点占据态
        
        整合原项目 148_cellular_automaton:
        将表面位点按线性排序，应用规则 30 演化吸附/脱附过程
        
        规则 30:
          111 -> 0, 110 -> 0, 101 -> 0, 100 -> 1
          011 -> 1, 010 -> 1, 001 -> 1, 000 -> 0
        
        物理意义: 相邻位点的占据状态影响中心位点的吸附概率
        """
        if steps < 0:
            raise ValueError("steps >= 0")
        occ = self.site_occupancy.copy()
        n = self.n_sites
        for _ in range(steps):
            new_occ = np.zeros(n, dtype=int)
            for j in range(1, n - 1):
                left = occ[j - 1]
                center = occ[j]
                right = occ[j + 1]
                # 规则 30: 将三态映射到二进制
                pattern = (left << 2) | (center << 1) | right
                # 规则 30 输出表 (从 111=7 到 000=0): 0,0,0,1,1,1,1,0
                rule_table = [0, 1, 1, 1, 1, 0, 0, 0]
                new_occ[j] = rule_table[pattern]
            occ = new_occ
        self.site_occupancy = occ

    def cvt_optimize_sites(self, n_generators: int = 8,
                           it_num: int = 20,
                           density_exp: float = 5.0) -> np.ndarray:
        """
        三维 Centroidal Voronoi Tessellation (CVT) 优化吸附位点分布
        
        整合原项目 249_cvt_3d_lumping:
        在表面附近三维空间中使用 Lloyd 算法迭代优化采样点分布
        
        算法:
          1. 随机初始化生成器 g_i ∈ [-L, L]^3
          2. 在区域内均匀采样点 s_j
          3. 计算密度 ρ(s_j) = μ(s_j)^density_exp
          4. 对每个生成器，计算其 Voronoi 区域内采样点的质量加权质心
          5. 将生成器移至质心位置
          6. 重复 2-5 直至收敛
        
        能量泛函:
          E = Σ_j ρ(s_j) * ||s_j - g_{k(j)}||^2
        
        其中 k(j) = argmin_i ||s_j - g_i||
        """
        if n_generators < 4:
            raise ValueError("n_generators >= 4")
        if it_num < 1:
            raise ValueError("it_num >= 1")

        # 初始化生成器: 在表面附近随机分布
        L = self.nx * self.d_nn * 0.5
        rng = np.random.default_rng(seed=42)
        g = rng.uniform(-L, L, size=(n_generators, 3))
        g[:, 2] = rng.uniform(0.3e-10, L, n_generators)  # 严格限制在表面上方

        # 均匀采样点 (避免边界)
        s_num_1d = 20
        eps_margin = 1e-12
        s_1d = np.linspace(-L + eps_margin, L - eps_margin, s_num_1d)
        sx, sy, sz = np.meshgrid(s_1d, s_1d, s_1d, indexing='ij')
        s = np.vstack([sx.ravel(), sy.ravel(), sz.ravel()]).T
        n_samples = s.shape[0]

        # 密度函数: 在表面附近密度更高 (模拟吸附原子分布)
        def mu_density(xyz):
            dist_to_surface = np.abs(xyz[:, 2])
            mu = 1.0 / (dist_to_surface + 0.5e-10)
            return np.clip(mu, 0.0, 10.0)

        mu_vals = mu_density(s)
        rho_vals = mu_vals ** density_exp

        for it in range(it_num):
            # 最近生成器分配 (Voronoi 区域)
            k_idx = np.zeros(n_samples, dtype=int)
            for j in range(n_samples):
                dists = np.sum((s[j] - g) ** 2, axis=1)
                k_idx[j] = int(np.argmin(dists))

            # 质量加权质心更新
            g_new = np.zeros_like(g)
            for i in range(n_generators):
                mask = k_idx == i
                if np.any(mask):
                    mass = np.sum(rho_vals[mask])
                    if mass > 1e-300:
                        g_new[i] = np.sum(rho_vals[mask][:, None] * s[mask], axis=0) / mass
                    else:
                        g_new[i] = g[i]
                else:
                    g_new[i] = g[i]
            g = g_new
            # 确保生成器始终在表面上方 (z > 0.2 Å)
            g[:, 2] = np.maximum(g[:, 2], 0.2e-10)

        return g

    def surface_coverage(self, species: int = 1) -> float:
        """
        计算表面覆盖率 θ
        
        公式:
            θ = N_adsorbed / N_total_sites
        """
        if self.n_sites == 0:
            return 0.0
        count = np.sum(self.site_occupancy == species)
        return float(count) / self.n_sites

    def compute_lateral_interaction(self, interaction_strength: float = 0.1) -> np.ndarray:
        """
        计算吸附物种之间的横向相互作用能
        
        公式 (平均场近似):
            E_int(i) = Σ_j J * θ_j / r_ij
        
        其中 J 为相互作用强度 (eV)，θ_j 为位点 j 的占据态
        """
        n = self.n_sites
        e_int = np.zeros(n)
        pos = self.sites[:, :3]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r_ij = np.linalg.norm(pos[i] - pos[j])
                if r_ij > 1e-12:
                    e_int[i] += interaction_strength * self.site_occupancy[j] / r_ij
        return e_int

    def dump_site_info(self):
        """输出表面位点信息摘要"""
        print("=" * 60)
        print("Pt(111) 表面结构参数")
        print("=" * 60)
        print(f"  晶格常数 a          = {self.a * 1e10:.4f} Å")
        print(f"  最近邻间距 d_nn     = {self.d_nn * 1e10:.4f} Å")
        print(f"  表面原子数          = {self.n_atoms}")
        print(f"  吸附位点数          = {self.n_sites}")
        print(f"  CO 覆盖率           = {self.surface_coverage(species=1):.4f}")
        print(f"  O 覆盖率            = {self.surface_coverage(species=2):.4f}")
        print("=" * 60)
