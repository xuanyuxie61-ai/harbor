"""
polymer_chain.py
聚合物链构建模块

融合原项目:
- 1008_random_walk_1d_simulation: 1D 随机游走思想扩展至 3D 自回避随机游走
- 330_ellipse_grid: 椭圆截面内网格点生成，用于链的横截面单体分布

功能:
1. 构建粗粒化聚合物链（ bead-spring 模型）
2. 3D 自回避随机游走（SARW）生成初始构象
3. 椭圆截面网格用于环状或椭圆形链的截面单体定位
"""

import numpy as np
from typing import Tuple, Optional
from numeric_utils import seeded_random, safe_divide


class PolymerChain:
    """
    粗粒化聚合物链模型。
    
    物理模型:
        采用 bead-spring 模型，每条链由 N_beads 个单体（bead）组成，
        相邻单体通过弹簧势连接。链构象由 3D 自回避随机游走初始化。
    
    数学描述:
        端到端向量 R_e 满足高斯链统计:
            <R_e^2> = N_b b^2
        其中 N_b 为库恩段数，b 为库恩长度。
        
        回转半径:
            R_g^2 = (1/N) Σ_i |r_i - r_cm|^2
    """
    
    def __init__(
        self,
        n_chains: int = 10,
        beads_per_chain: int = 50,
        bond_length: float = 1.0,
        box: np.ndarray = None,
        random_seed: int = 42,
    ):
        """
        初始化聚合物系统。
        
        参数:
            n_chains: 链的数量
            beads_per_chain: 每条链的单体数
            bond_length: 键长（库恩长度的粗粒化表示）
            box: 模拟盒子尺寸 (3,)
            random_seed: 随机种子
        """
        if n_chains < 1:
            raise ValueError("n_chains 必须 >= 1")
        if beads_per_chain < 2:
            raise ValueError("beads_per_chain 必须 >= 2")
        if bond_length <= 0:
            raise ValueError("bond_length 必须 > 0")
        
        self.n_chains = n_chains
        self.beads_per_chain = beads_per_chain
        self.bond_length = bond_length
        self.n_total = n_chains * beads_per_chain
        
        if box is None:
            # 根据链尺寸和熔融态数密度估算盒子大小
            # 单体体积 ~ (4/3)π σ^3，目标体积分数 ~ 0.6-0.7
            bead_volume = (4.0 / 3.0) * np.pi * (bond_length ** 3)
            total_bead_volume = self.n_total * bead_volume
            target_fraction = 0.65
            box_volume = total_bead_volume / target_fraction
            est_size = box_volume ** (1.0 / 3.0)
            self.box = np.array([est_size, est_size, est_size])
        else:
            self.box = np.array(box, dtype=float)
            if np.any(self.box <= 0):
                raise ValueError("box 尺寸必须 > 0")
        
        self.dim = 3
        self.positions = np.zeros((self.n_total, self.dim))
        self.velocities = np.zeros((self.n_total, self.dim))
        self.forces = np.zeros((self.n_total, self.dim))
        self.masses = np.ones(self.n_total)
        
        # 记录每条链的起始索引
        self.chain_starts = np.arange(0, self.n_total, self.beads_per_chain)
        
        # 初始化构象
        self._initialize_conformation(random_seed)
        
        # 初始化速度（Maxwell-Boltzmann 分布在 thermostat 中完成）
        self._initialize_velocities(temperature=1.0, random_seed=random_seed + 1)
    
    def _initialize_conformation(self, seed: int):
        """
        使用 3D 自回避随机游走（SARW）初始化链构象。
        
        算法:
            1. 对每条链，从盒子中心附近开始
            2. 每一步尝试 6 个方向（±x, ±y, ±z）
            3. 选择不与已有单体太近的方向（自回避）
            4. 应用周期性边界条件
        """
        rng = np.random.RandomState(seed)
        
        # 方向向量: 6 个晶格方向
        directions = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        
        min_distance = 0.7 * self.bond_length  # 自回避最小距离
        
        for c in range(self.n_chains):
            start_idx = self.chain_starts[c]
            
            # 链的起始位置在盒子内随机放置
            origin = rng.rand(self.dim) * self.box
            self.positions[start_idx] = origin
            
            for bead in range(1, self.beads_per_chain):
                idx = start_idx + bead
                prev_pos = self.positions[idx - 1].copy()
                
                # 尝试所有方向，选择不冲突的
                valid_directions = []
                for d in directions:
                    trial_pos = prev_pos + self.bond_length * d
                    # 周期性边界
                    trial_pos = trial_pos % self.box
                    
                    # 检查自回避: 与所有已有单体的距离
                    conflict = False
                    for existing in range(idx):
                        dr = trial_pos - self.positions[existing]
                        dr = dr - self.box * np.rint(dr / self.box)
                        dist = np.linalg.norm(dr)
                        if dist < min_distance:
                            conflict = True
                            break
                    
                    if not conflict:
                        valid_directions.append(d)
                
                if len(valid_directions) > 0:
                    chosen = valid_directions[rng.randint(len(valid_directions))]
                    self.positions[idx] = (prev_pos + self.bond_length * chosen) % self.box
                else:
                    # 自回避失败，退化为普通随机游走（允许轻微重叠）
                    theta = rng.uniform(0, 2 * np.pi)
                    phi = rng.uniform(0, np.pi)
                    dx = self.bond_length * np.sin(phi) * np.cos(theta)
                    dy = self.bond_length * np.sin(phi) * np.sin(theta)
                    dz = self.bond_length * np.cos(phi)
                    self.positions[idx] = (prev_pos + np.array([dx, dy, dz])) % self.box
    
    def _initialize_velocities(self, temperature: float, random_seed: int):
        """
        根据 Maxwell-Boltzmann 分布初始化速度。
        
        公式:
            P(v) ∝ exp(-mv^2 / (2k_B T))
            <v_i^2> = k_B T / m
        
        参数:
            temperature: 约化温度（k_B = 1）
            random_seed: 随机种子
        """
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")
        
        rng = np.random.RandomState(random_seed)
        sigma = np.sqrt(temperature / self.masses)
        
        for d in range(self.dim):
            self.velocities[:, d] = rng.normal(0.0, sigma, self.n_total)
        
        # 减去质心速度，避免整体漂移
        v_cm = np.mean(self.velocities, axis=0)
        self.velocities -= v_cm
    
    def get_chain_positions(self, chain_id: int) -> np.ndarray:
        """
        获取指定链的所有单体位置。
        
        参数:
            chain_id: 链索引
        
        返回:
            (beads_per_chain, dim) 位置数组
        """
        if chain_id < 0 or chain_id >= self.n_chains:
            raise IndexError("chain_id 越界")
        start = self.chain_starts[chain_id]
        end = start + self.beads_per_chain
        return self.positions[start:end, :].copy()
    
    def radius_of_gyration(self, chain_id: Optional[int] = None) -> float:
        """
        计算回转半径 R_g。
        
        公式:
            R_g = sqrt( (1/N) Σ_i |r_i - r_cm|^2 )
        
        参数:
            chain_id: 若指定则计算单条链，否则计算整个系统
        
        返回:
            回转半径
        """
        if chain_id is None:
            pos = self.positions
        else:
            pos = self.get_chain_positions(chain_id)
        
        cm = np.mean(pos, axis=0)
        rg_sq = np.mean(np.sum((pos - cm) ** 2, axis=1))
        return float(np.sqrt(max(rg_sq, 0.0)))
    
    def end_to_end_distance(self, chain_id: int) -> float:
        """
        计算端到端距离。
        
        参数:
            chain_id: 链索引
        
        返回:
            端到端距离
        """
        pos = self.get_chain_positions(chain_id)
        dr = pos[-1] - pos[0]
        # 最小像约定
        dr = dr - self.box * np.rint(dr / self.box)
        return float(np.linalg.norm(dr))
    
    def apply_pbc(self):
        """
        对所有单体应用周期性边界条件。
        """
        self.positions = self.positions % self.box
    
    def kinetic_energy(self) -> float:
        """
        计算系统总动能。
        
        公式:
            E_k = 0.5 Σ_i m_i v_i^2
        """
        return float(0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities ** 2))
    
    def instantaneous_temperature(self) -> float:
        """
        计算瞬时温度。
        
        公式:
            T = (2 E_k) / (N_dof * k_B)
            N_dof = 3N - 3（减去质心平动自由度）
        
        TODO: 请根据统计力学温度-动能关系实现此函数。
        注意: 此处的自由度计算必须与 thermostat.py 中的 BerendsenThermostat.apply() 保持一致。
        """
        # HOLE 1 START
        raise NotImplementedError("Hole 1: 请实现瞬时温度计算，确保与 thermostat.py 的自由度定义一致")
        # HOLE 1 END


def generate_ellipse_cross_section(
    n_points: int,
    semi_axes: Tuple[float, float],
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    在椭圆截面内生成网格点，模拟聚合物链的横截面单体分布。
    
    融合原项目 330_ellipse_grid:
        椭圆方程: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
    
    参数:
        n_points: 短轴方向的分割数
        semi_axes: (a, b) 半轴长度
        center: 椭圆中心
    
    返回:
        (N, 2) 网格点数组
    """
    if n_points < 1:
        raise ValueError("n_points 必须 >= 1")
    a, b = semi_axes
    if a <= 0 or b <= 0:
        raise ValueError("半轴长度必须 > 0")
    
    cx, cy = center
    
    # 确定步长，基于较短轴
    if a < b:
        h = 2.0 * a / (2.0 * n_points + 1.0)
        ni = n_points
        nj = int(np.ceil(b / a) * n_points)
    else:
        h = 2.0 * b / (2.0 * n_points + 1.0)
        nj = n_points
        ni = int(np.ceil(a / b) * n_points)
    
    points = []
    
    for j in range(nj + 1):
        i = 0
        x = cx
        y = cy + j * h
        # 检查是否在椭圆内
        if ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1.0 + 1e-12:
            points.append([x, y])
        if j > 0:
            y_mirror = 2 * cy - y
            if ((x - cx) / a) ** 2 + ((y_mirror - cy) / b) ** 2 <= 1.0 + 1e-12:
                points.append([x, y_mirror])
        
        while True:
            i += 1
            x = cx + i * h
            ellipse_val = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2
            if ellipse_val > 1.0:
                break
            
            # 四个对称点
            points.append([x, y])
            points.append([2 * cx - x, y])
            if j > 0:
                points.append([x, 2 * cy - y])
                points.append([2 * cx - x, 2 * cy - y])
    
    arr = np.array(points)
    if arr.size == 0:
        return np.zeros((0, 2))
    return arr
