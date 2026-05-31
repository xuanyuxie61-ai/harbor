"""
nuclear_data_io.py
===================
核数据结构 I/O 与数据处理模块

基于种子项目 447_freefem_msh_io 的网格数据结构读写思想
以及 118_brc_naive 的大规模数据聚合思想，
本模块实现核质量表、核素数据库的读取、写入与聚合处理。

功能:
1. 核素质量表读取与插值
2. 球形壳层网格的构造与 I/O (用于输运计算)
3. 核数据聚合统计 (类似 EXFOR 数据处理)
4. 分离能、Q 值计算

核心公式
--------
质量过剩 (Mass Excess):
    ΔM = M_atom - A * u

结合能 (Semi-empirical mass formula):
    B = a_v A - a_s A^{2/3} - a_c Z(Z-1)/A^{1/3} - a_a (A-2Z)²/A + δ(A,Z)

中子分离能:
    S_n = [M(A-1,Z) + m_n - M(A,Z)] c²

质子分离能:
    S_p = [M(A-1,Z-1) + m_p - M(A,Z)] c²
"""

import numpy as np


# 液滴模型参数 (MeV)
LIQUID_DROP_PARAMS = {
    'volume': 15.75,
    'surface': 17.8,
    'coulomb': 0.711,
    'asymmetry': 23.7,
    'pairing': 11.18,
}


class Nuclide:
    """
    单个核素的数据容器。
    """

    def __init__(self, Z, A, mass_excess=None):
        self.Z = int(Z)
        self.A = int(A)
        self.N = self.A - self.Z
        self.mass_excess = mass_excess  # MeV

    def binding_energy(self):
        """使用半经验质量公式计算结合能。"""
        a_v = LIQUID_DROP_PARAMS['volume']
        a_s = LIQUID_DROP_PARAMS['surface']
        a_c = LIQUID_DROP_PARAMS['coulomb']
        a_a = LIQUID_DROP_PARAMS['asymmetry']
        a_p = LIQUID_DROP_PARAMS['pairing']

        # 体积项
        B_vol = a_v * self.A
        # 表面项
        B_sur = -a_s * (self.A ** (2.0 / 3.0))
        # 库仑项
        B_cou = -a_c * self.Z * (self.Z - 1.0) / (self.A ** (1.0 / 3.0))
        # 不对称项
        B_asym = -a_a * ((self.A - 2.0 * self.Z) ** 2) / self.A
        # 对能项
        delta = 0.0
        if self.Z % 2 == 0 and self.N % 2 == 0:
            delta = a_p / np.sqrt(self.A)
        elif self.Z % 2 == 1 and self.N % 2 == 1:
            delta = -a_p / np.sqrt(self.A)

        return B_vol + B_sur + B_cou + B_asym + delta

    def neutron_separation_energy(self):
        """中子分离能 S_n。"""
        if self.A <= 1:
            return 0.0
        # 简化计算：使用液滴模型差分
        this_BE = self.binding_energy()
        # 近似：A-1 同位素的结合能
        prev = Nuclide(self.Z, self.A - 1)
        prev_BE = prev.binding_energy()
        return this_BE - prev_BE

    def proton_separation_energy(self):
        """质子分离能 S_p。"""
        if self.Z <= 1 or self.A <= 1:
            return 0.0
        this_BE = self.binding_energy()
        prev = Nuclide(self.Z - 1, self.A - 1)
        prev_BE = prev.binding_energy()
        return this_BE - prev_BE

    def __repr__(self):
        return f"Nuclide(Z={self.Z}, A={self.A}, BE={self.binding_energy():.3f} MeV)"


class NuclearDataAggregator:
    """
    核数据聚合器，参照 118_brc_naive 的数据聚合思想。
    对多个核素数据进行统计聚合。
    """

    def __init__(self):
        self.nuclides = {}

    def add(self, nuclide):
        """添加核素。"""
        key = (nuclide.Z, nuclide.A)
        self.nuclides[key] = nuclide

    def aggregate_by_Z(self):
        """按原子序数 Z 聚合统计。"""
        stats = {}
        for (Z, A), nuc in self.nuclides.items():
            if Z not in stats:
                stats[Z] = {'count': 0, 'A_list': [], 'BE_list': []}
            stats[Z]['count'] += 1
            stats[Z]['A_list'].append(A)
            stats[Z]['BE_list'].append(nuc.binding_energy())

        # 计算每 Z 的平均值
        for Z in stats:
            stats[Z]['A_mean'] = np.mean(stats[Z]['A_list'])
            stats[Z]['BE_mean'] = np.mean(stats[Z]['BE_list'])
            stats[Z]['BE_max'] = np.max(stats[Z]['BE_list'])
            stats[Z]['BE_min'] = np.min(stats[Z]['BE_list'])
        return stats

    def get_mass_table_array(self):
        """导出质量表为数组。"""
        data = []
        for (Z, A), nuc in self.nuclides.items():
            data.append([Z, A, nuc.N, nuc.binding_energy(),
                         nuc.neutron_separation_energy(),
                         nuc.proton_separation_energy()])
        return np.array(data)


class SphericalShellMesh:
    """
    球形壳层网格，参照 447_freefem_msh_io 的网格数据结构思想。

    构造同心球壳网格用于核输运/光学模型径向计算。
    """

    def __init__(self, R_max=15.0, n_r=100, n_theta=30, n_phi=60):
        self.R_max = R_max
        self.n_r = n_r
        self.n_theta = n_theta
        self.n_phi = n_phi

        # 径向节点 (对数-线性混合分布)
        # 内区密集，外区稀疏
        t = np.linspace(0.0, 1.0, n_r)
        self.r_nodes = R_max * (t ** 1.5)  # 幂律分布
        self.r_nodes[0] = 1e-6  # 避免零点

        # 角向节点
        self.theta_nodes = np.linspace(0.0, np.pi, n_theta)
        self.phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi)

        self.n_vertices = n_r * n_theta * n_phi
        self.n_elements = (n_r - 1) * (n_theta - 1) * (n_phi - 1)

    def get_vertex_coordinates(self):
        """获取所有顶点坐标。"""
        r = self.r_nodes
        theta = self.theta_nodes
        phi = self.phi_nodes
        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)
        return X.flatten(), Y.flatten(), Z.flatten()

    def write_mesh_file(self, filename):
        """将网格写入简化格式文件。"""
        X, Y, Z = self.get_vertex_coordinates()
        with open(filename, 'w') as f:
            f.write(f"# SphericalShellMesh: {self.n_r}x{self.n_theta}x{self.n_phi}\n")
            f.write(f"{self.n_vertices} {self.n_elements} 0\n")
            for i in range(self.n_vertices):
                f.write(f"{i+1} {X[i]:.8e} {Y[i]:.8e} {Z[i]:.8e} 0\n")

    def read_mesh_file(self, filename):
        """读取网格文件。"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        # 解析头部
        header = lines[1].strip().split()
        n_v = int(header[0])
        coords = np.zeros((n_v, 3))
        for i in range(n_v):
            parts = lines[3 + i].strip().split()
            coords[i, :] = [float(parts[1]), float(parts[2]), float(parts[3])]
        return coords


def generate_nuclear_mass_table(Z_range, A_range_func):
    """
    生成指定范围内的核素质量表。

    Parameters
    ----------
    Z_range : range
        原子序数范围。
    A_range_func : callable
        接受 Z 返回 A 范围的函数。

    Returns
    -------
    aggregator : NuclearDataAggregator
        聚合后的核数据。
    """
    agg = NuclearDataAggregator()
    for Z in Z_range:
        A_min, A_max = A_range_func(Z)
        for A in range(A_min, A_max + 1):
            if A >= Z:
                nuc = Nuclide(Z, A)
                agg.add(nuc)
    return agg


def compute_q_value_reaction(Z_target, A_target, Z_proj, A_proj, Z_out, A_out):
    """
    计算核反应的 Q 值。

    Q = [M_target + M_proj - M_residual - M_out] c²
      ≈ BE_residual + BE_out - BE_target - BE_proj

    Parameters
    ----------
    Z_target, A_target : int
        靶核。
    Z_proj, A_proj : int
        入射粒子。
    Z_out, A_out : int
        出射粒子。

    Returns
    -------
    Q : float
        Q 值 (MeV)。
    """
    target = Nuclide(Z_target, A_target)
    proj = Nuclide(Z_proj, A_proj)
    Z_res = Z_target + Z_proj - Z_out
    A_res = A_target + A_proj - A_out
    if Z_res < 0 or A_res < Z_res:
        return -999.0
    residual = Nuclide(Z_res, A_res)
    outgoing = Nuclide(Z_out, A_out)

    Q = (residual.binding_energy() + outgoing.binding_energy()
         - target.binding_energy() - proj.binding_energy())
    return Q


if __name__ == "__main__":
    # 自检
    nuc = Nuclide(26, 56)
    print(nuc)
    print(f"S_n = {nuc.neutron_separation_energy():.3f} MeV")
    print(f"S_p = {nuc.proton_separation_energy():.3f} MeV")

    agg = generate_nuclear_mass_table(range(20, 30), lambda Z: (Z + 20, Z + 40))
    stats = agg.aggregate_by_Z()
    print(f"Z=26 同位素数目: {stats[26]['count']}")

    mesh = SphericalShellMesh(R_max=10.0, n_r=20, n_theta=10, n_phi=20)
    print(f"网格顶点数: {mesh.n_vertices}")

    Q = compute_q_value_reaction(26, 56, 0, 1, 0, 1)
    print(f"n + 56Fe -> n + 56Fe 的 Q 值 ≈ {Q:.3f} MeV")
