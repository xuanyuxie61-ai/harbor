"""
microstructure_model.py
骨小梁微观结构模型模块

融合来源：
- 864_pentominoes: 0/1 矩阵表示几何形状

科学背景：
松质骨（trabecular bone）由相互连接的骨小梁构成，形成多孔结构。
骨小梁的取向、厚度和连通性决定了松质骨的宏观力学性能。
本项目使用类似 pentomino 的 0/1 矩阵来编码骨小梁微观结构，
并基于 Mori-Tanaka 均匀化理论或孔隙率幂律模型计算有效弹性模量。
"""

import numpy as np
from typing import Dict, Tuple, List


# ===================================================================
# Pentomino-like 微观结构定义（来自 864_pentominoes 的 pentomino_matrix）
# ===================================================================
PENTOMINO_SHAPES: Dict[str, np.ndarray] = {
    'F': np.array([[0, 1, 1],
                   [1, 1, 0],
                   [0, 1, 0]], dtype=int),
    'I': np.array([[1, 1, 1, 1, 1]], dtype=int),
    'L': np.array([[0, 0, 0, 1],
                   [1, 1, 1, 1]], dtype=int),
    'N': np.array([[1, 1, 0, 0],
                   [0, 1, 1, 1]], dtype=int),
    'P': np.array([[1, 1],
                   [1, 1],
                   [1, 0]], dtype=int),
    'T': np.array([[1, 1, 1],
                   [0, 1, 0],
                   [0, 1, 0]], dtype=int),
    'U': np.array([[1, 0, 1],
                   [1, 1, 1]], dtype=int),
    'V': np.array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1]], dtype=int),
    'W': np.array([[1, 0, 0],
                   [1, 1, 0],
                   [0, 1, 1]], dtype=int),
    'X': np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=int),
    'Y': np.array([[0, 0, 1, 0],
                   [1, 1, 1, 1]], dtype=int),
    'Z': np.array([[1, 1, 0],
                   [0, 1, 0],
                   [0, 1, 1]], dtype=int),
}

PENTOMINO_NAMES = list(PENTOMINO_SHAPES.keys())


def get_pentomino_matrix(name: str) -> np.ndarray:
    """
    获取指定 pentomino 的 0/1 矩阵表示。

    Parameters
    ----------
    name : str
        'F', 'I', 'L', 'N', 'P', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'

    Returns
    -------
    np.ndarray
        0/1 矩阵，1 表示骨组织，0 表示骨髓腔
    """
    key = name.upper()
    if key not in PENTOMINO_SHAPES:
        raise ValueError(f"Unknown pentomino name '{name}'. Valid: {PENTOMINO_NAMES}")
    return PENTOMINO_SHAPES[key].copy()


def rotate_matrix_90(mat: np.ndarray, k: int = 1) -> np.ndarray:
    """
    将矩阵逆时针旋转 90*k 度。
    """
    return np.rot90(mat, k=k)


def flip_matrix(mat: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    沿指定轴翻转矩阵。
    """
    return np.flip(mat, axis=axis)


class TrabecularMicrostructure:
    """
    骨小梁微观结构模型。

    使用 pentomino-like 的 0/1 矩阵拼接构建骨小梁代表性体积单元(RVE)，
    计算孔隙率、比表面积及有效弹性模量。
    """

    def __init__(self, grid_size: int = 15, pattern_seed: int = 42):
        """
        Parameters
        ----------
        grid_size : int
            RVE 网格尺寸（必须为大于5的正整数）
        pattern_seed : int
            随机种子
        """
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        self.grid_size = grid_size
        self.pattern_seed = pattern_seed
        self.rng = np.random.default_rng(pattern_seed)
        self.rve_grid = self._build_rve()
        self.porosity = self._compute_porosity()
        self.specific_surface = self._compute_specific_surface()
        self.effective_modulus = self._compute_effective_young_modulus()

    def _build_rve(self) -> np.ndarray:
        """
        构建代表性体积单元(RVE)。

        策略：在 grid_size x grid_size 网格中放置 pentomino 形状的骨小梁。
        使用一种类似拼图的方式放置多个 pentomino，未覆盖区域为骨髓腔(0)。
        """
        n = self.grid_size
        grid = np.zeros((n, n), dtype=int)

        # 选择一种骨小梁取向模式（模拟不同力学加载方向下的骨小梁取向）
        names = ['I', 'L', 'N', 'T', 'X', 'V', 'W']
        self.rng.shuffle(names)

        placed = 0
        for name in names:
            p = get_pentomino_matrix(name)
            # 随机旋转和翻转
            p = rotate_matrix_90(p, k=self.rng.integers(0, 4))
            if self.rng.random() > 0.5:
                p = flip_matrix(p, axis=self.rng.integers(0, 2))

            ph, pw = p.shape
            # 尝试放置
            max_attempts = 50
            for _ in range(max_attempts):
                i = self.rng.integers(0, n - ph + 1)
                j = self.rng.integers(0, n - pw + 1)
                # 检查重叠：只允许部分重叠以模拟真实骨小梁的连通性
                region = grid[i:i+ph, j:j+pw]
                overlap = np.sum((region == 1) & (p == 1))
                if overlap <= 1:  # 允许最多1格重叠（连通性）
                    region[p == 1] = 1
                    placed += 1
                    break

        # 确保至少有一定骨体积分数
        if np.mean(grid) < 0.05:
            # 填充中心区域
            c = n // 2
            grid[c-1:c+2, c-1:c+2] = 1

        return grid

    def _compute_porosity(self) -> float:
        """
        计算孔隙率 phi = V_void / V_total。
        """
        total = self.rve_grid.size
        solid = np.sum(self.rve_grid)
        void = total - solid
        phi = void / total
        return float(phi)

    def _compute_specific_surface(self) -> float:
        """
        计算比表面积 S_V = (固体-孔隙界面面积) / (总体积)。

        在二维离散模型中，通过计算 0-1 边界边数来估算。
        """
        grid = self.rve_grid
        n = grid.shape[0]
        interface_edges = 0

        # 水平边
        for i in range(n):
            for j in range(n - 1):
                if grid[i, j] != grid[i, j + 1]:
                    interface_edges += 1

        # 垂直边
        for i in range(n - 1):
            for j in range(n):
                if grid[i, j] != grid[i + 1, j]:
                    interface_edges += 1

        # 归一化：每单位面积内的界面长度
        sv = interface_edges / (n * n)
        return float(sv)

    def _compute_effective_young_modulus(self) -> float:
        """
        计算骨小梁有效弹性模量 E_eff。

        采用孔隙率幂律模型（Carter & Hayes, 1977; Gibson & Ashby, 1988）：

            E_eff / E_bone = C * (rho_relative)^n

        其中：
            rho_relative = 1 - phi  （相对密度）
            C = 1.0  （经验常数）
            n = 2.0  （对于开放细胞泡沫结构，n ≈ 2）

        致密骨（皮质骨）的弹性模量 E_bone ≈ 17 GPa。
        """
        E_bone = 17.0e3  # MPa = 17 GPa
        rho_rel = 1.0 - self.porosity
        if rho_rel <= 0:
            return 0.0
        C = 1.0
        n_exp = 2.0
        E_eff = E_bone * C * (rho_rel ** n_exp)
        return float(E_eff)

    def get_local_density(self, x: float, y: float) -> float:
        """
        在 RVE 局部坐标 (x, y) ∈ [0,1]² 上采样局部骨密度。

        Returns
        -------
        float
            0 或 1（离散模型），表示该位置是否为骨组织
        """
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("Coordinates must be in [0,1]^2")
        n = self.grid_size
        i = min(int(y * n), n - 1)
        j = min(int(x * n), n - 1)
        return float(self.rve_grid[i, j])

    def generate_microstructure_report(self) -> Dict[str, float]:
        """
        生成微观结构分析报告。
        """
        return {
            "grid_size": self.grid_size,
            "porosity": self.porosity,
            "relative_density": 1.0 - self.porosity,
            "specific_surface": self.specific_surface,
            "effective_young_modulus_MPa": self.effective_modulus,
            "effective_young_modulus_GPa": self.effective_modulus / 1000.0,
        }


def build_trabecular_field(nx: int, ny: int, cortical_mask: np.ndarray,
                           seed_offset: int = 0) -> np.ndarray:
    """
    在整个骨骼截面上构建骨密度场。

    皮质骨区域密度 = 1.8 g/cm³（致密骨）
    松质骨区域密度 = 基于微观结构模型计算的有效密度

    Parameters
    ----------
    nx, ny : int
        场网格尺寸
    cortical_mask : np.ndarray, shape (nx*ny,)
        布尔数组，True 表示皮质骨节点
    seed_offset : int
        随机种子偏移

    Returns
    -------
    np.ndarray, shape (nx*ny,)
        每个节点的骨密度 (g/cm³)
    """
    density = np.zeros(nx * ny)
    rho_cortical = 1.8   # g/cm³
    rho_marrow = 0.001   # g/cm³ (忽略)

    # 为每个松质骨区域分配一个微观结构模型
    # 简化：整个松质骨区域使用统一的平均有效密度
    micro = TrabecularMicrostructure(grid_size=15, pattern_seed=42 + seed_offset)
    rho_trabecular = (1.0 - micro.porosity) * rho_cortical

    density[cortical_mask] = rho_cortical
    density[~cortical_mask] = max(rho_trabecular, 0.05)

    return density
