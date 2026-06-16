"""
velocity_grid.py — 速度空间网格构造与 Voronoi 自适应划分
========================================================

基于 Voronoi 图的自适应速度空间网格.

种子项目映射:
  - 1398_voronoi_plot: Voronoi 最近邻划分算法
  - 178_circle_distance: 速度空间距离分布统计
  - 928_pwl_interp_2d_scattered: 分段线性插值 (在 pwl_velocity.py 中)

物理背景:
  在 Fokker-Planck 方程的数值求解中, 速度空间的网格设计直接影响
  精度和效率. 各向同性分布函数 f(v,t) 定义在 [0, v_max] 上,
  但在速度空间中粒子数密度集中在热速度附近, 因此需要自适应加密.

核心算法:
  1. 均匀对数网格 / 自适应网格 (基于 Voronoi 生成元)
  2. Voronoi 胞腔诊断 (胞腔体积, 最近邻距离分布)
  3. 圆上距离统计 (散射角分布表征)
"""

import math
import numpy as np

from physical_constants import PI, FOUR_PI


# ===========================================================================
#  §1  均匀网格
# ===========================================================================
def make_uniform_grid(v_min, v_max, N):
    """构造均匀速度网格.

    Parameters
    ----------
    v_min : float  最小速度 (> 0 避免奇点)
    v_max : float  最大速度
    N : int  网格点数

    Returns
    -------
    v : ndarray(N)  节点位置
    dv : ndarray(N)  局部网格间距
    """
    if v_min <= 0:
        raise ValueError("v_min 必须 > 0 以避免坐标奇异性")
    v = np.linspace(v_min, v_max, N)
    dv = np.full(N, (v_max - v_min) / (N - 1))
    return v, dv


def make_log_grid(v_min, v_max, N):
    """对数拉伸网格: 在低速度区域加密.

    v_i = v_min * (v_max/v_min)^{i/(N-1)}

    适用于需要高精度解析热核区的碰撞问题.
    """
    if v_min <= 0:
        raise ValueError("v_min 必须 > 0")
    ratio = v_max / v_min
    i_idx = np.arange(N)
    v = v_min * ratio**(i_idx / (N - 1))
    dv = np.zeros(N)
    for i in range(N):
        if i == 0:
            dv[i] = v[1] - v[0]
        elif i == N - 1:
            dv[i] = v[-1] - v[-2]
        else:
            dv[i] = 0.5 * (v[i+1] - v[i-1])
    return v, dv


# ===========================================================================
#  §2  Voronoi 自适应网格  (源自 1398_voronoi_plot)
# ===========================================================================
class VoronoiVelocityMesh:
    """基于 Voronoi 图的一维自适应速度网格.

    核心思想 (映射自 voronoi_plot):
      原项目: 在 2D 区域中, 对每个像素找最近生成元, 着色.
      本项目: 在 1D 速度空间 [v_min, v_max] 中放置 N_gen 个生成元,
              每个速度点分配给最近生成元, 形成 Voronoi 胞腔.
              生成元的位置由误差指示子自适应调整.

    映射:
      像素 → 速度空间测试点
      生成元 → 网格节点
      Lp 距离 → 加权距离 (考虑分布函数梯度)
      着色 → 误差分配到最近节点

    Attributes
    ----------
    generators : ndarray  生成元位置 (网格节点)
    cells : list of tuples  每个生成元的 Voronoi 胞腔 [left, right]
    weights : ndarray  每个胞腔的权重 (体积 × 误差指示)
    """

    def __init__(self, v_min, v_max, N_gen, f_dist=None):
        """
        Parameters
        ----------
        v_min, v_max : float  速度范围
        N_gen : int  生成元数量
        f_dist : callable or None  分布函数 (用于自适应加密)
        """
        self.v_min = v_min
        self.v_max = v_max
        self.N_gen = N_gen
        self.f_dist = f_dist

        # 初始化: 均匀分布生成元
        self.generators = np.linspace(v_min, v_max, N_gen)
        self._update_cells()

        # 如果有分布函数, 进行自适应调整
        if f_dist is not None:
            self._adapt_to_distribution(f_dist)

    def _update_cells(self):
        """计算每个生成元的 Voronoi 胞腔边界."""
        gen = self.generators
        N = len(gen)
        self.cells = []
        for i in range(N):
            if i == 0:
                left = self.v_min
            else:
                left = 0.5 * (gen[i-1] + gen[i])
            if i == N - 1:
                right = self.v_max
            else:
                right = 0.5 * (gen[i] + gen[i+1])
            self.cells.append((left, right))
        # 计算胞腔体积
        self.cell_volumes = np.array([r - l for l, r in self.cells])

    def _adapt_to_distribution(self, f_dist, n_iter=5):
        """基于分布函数梯度自适应调整生成元位置.

        原理: 梯度大的区域需要更密的网格. 使用 Monge-Kantorovich
        最优传输的思想, 将生成元重新分布使得每个胞腔中的
        ∫ |∇f| dv 近似相等.

        算法: 迭代梯度下降, 目标函数 = Σ_i (E_i - E_avg)²
        其中 E_i = ∫_{cell_i} |f'(v)| dv
        """
        for iteration in range(n_iter):
            # 计算每个胞腔的误差指示
            errors = np.zeros(self.N_gen)
            for i in range(self.N_gen):
                left, right = self.cells[i]
                v_cell = np.linspace(left, right, max(20, 5))
                dv_cell = np.diff(v_cell)
                f_vals = np.array([f_dist(vv) for vv in v_cell])
                # 误差 = ∫ |f'(v)| dv ≈ Σ |Δf|
                errors[i] = np.sum(np.abs(np.diff(f_vals)))

            # 目标: 均分误差
            total_error = np.sum(errors) + 1e-30
            target = total_error / self.N_gen

            # 调整生成元: 误差大的胞腔扩大, 小的缩小
            adjustment = np.zeros(self.N_gen)
            for i in range(self.N_gen):
                ratio = errors[i] / target
                # 限制调整幅度
                adjustment[i] = 0.1 * (ratio - 1.0) * self.cell_volumes[i]

            # 应用调整 (只移动内部生成元)
            new_gen = self.generators.copy()
            for i in range(1, self.N_gen - 1):
                delta = 0.5 * (adjustment[i-1] - adjustment[i])
                new_gen[i] += delta
                # 保持顺序
                new_gen[i] = np.clip(new_gen[i],
                                     new_gen[i-1] + 1e-10,
                                     new_gen[i+1] - 1e-10)
            self.generators = new_gen
            self._update_cells()

    def find_nearest(self, v):
        """对给定速度 v, 找最近的生成元 (Voronoi 归属).

        映射自 voronoi_plot 中的 "对每个像素找最近生成元".
        """
        v = np.asarray(v)
        distances = np.abs(self.generators[np.newaxis, :] - v[:, np.newaxis])
        return np.argmin(distances, axis=1)

    def interpolate(self, f_at_gen, v_query):
        """在 Voronoi 网格上分段常数插值.

        Parameters
        ----------
        f_at_gen : ndarray  各生成元处的函数值
        v_query : ndarray  查询点

        Returns
        -------
        f_interp : ndarray  插值结果
        """
        indices = self.find_nearest(v_query)
        return f_at_gen[indices]

    def get_fine_grid(self, points_per_cell=10):
        """获取每个胞腔内的精细网格 (用于诊断)."""
        v_fine = []
        for left, right in self.cells:
            v_fine.append(np.linspace(left, right, points_per_cell))
        return np.concatenate(v_fine)


# ===========================================================================
#  §3  圆上距离统计  (源自 178_circle_distance)
# ===========================================================================
class VelocityDistanceStatistics:
    """速度空间距离分布统计.

    映射自 circle_distance:
      原项目: 单位圆上随机点对距离的概率密度
              pdf(d) = (1/π) / √(1 - d²/(4r²))
      本项目: 速度空间中网格点间距离的统计分布.
              用于表征网格分辨率是否充分解析了碰撞散射角分布.

    物理联系:
      在 Fokker-Planck 碰撞理论中, 散射角 θ 的分布满足:
        dσ/dΩ ∝ 1/sin⁴(θ/2)  (Rutherford 散射)
      小角散射占主导, 速度空间中近邻点的距离分布决定了
      数值分辨率是否足够解析小角累积效应.
    """

    def __init__(self, velocity_grid):
        self.v_grid = velocity_grid
        self.N = len(velocity_grid)
        # 计算所有点对距离
        self.distances = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                self.distances.append(abs(velocity_grid[i] - velocity_grid[j]))
        self.distances = np.array(self.distances)

    def empirical_pdf(self, n_bins=50):
        """经验概率密度函数."""
        hist, edges = np.histogram(self.distances, bins=n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, hist

    def mean_distance(self):
        """平均距离."""
        return np.mean(self.distances)

    def min_distance(self):
        """最小距离 (决定了最高分辨率)."""
        return np.min(self.distances)

    def circle_pdf_analog(self, d, r_eff=None):
        """类比圆上距离 PDF:

        pdf(d) = (1/π) / √(1 - d²/(4 r_eff²))

        其中 r_eff 是有效 "半径" = 速度范围/2.
        """
        if r_eff is None:
            r_eff = (self.v_grid[-1] - self.v_grid[0]) / 2.0
        d = np.asarray(d)
        ratio = d / (2.0 * r_eff)
        ratio = np.clip(ratio, 0, 0.9999)
        pdf = (1.0 / PI) / np.sqrt(1.0 - ratio**2)
        return pdf

    def resolution_diagnostic(self):
        """网格分辨率诊断.

        返回一个字典, 包含:
        - mean_d: 平均距离
        - min_d: 最小距离
        - max_d: 最大距离
        - ratio: max/min (网格拉伸比)
        - effective_N: 有效独立分辨率点数
        """
        d_min = self.min_distance()
        d_max = np.max(self.distances)
        d_mean = self.mean_distance()
        d_std = np.std(self.distances)
        return {
            "mean_distance": d_mean,
            "min_distance": d_min,
            "max_distance": d_max,
            "std_distance": d_std,
            "stretch_ratio": d_max / max(d_min, 1e-30),
            "effective_resolution": (self.v_grid[-1] - self.v_grid[0]) / max(d_mean, 1e-30),
        }


# ===========================================================================
#  §4  综合网格构造函数
# ===========================================================================
def create_velocity_grid(v_min, v_max, N, grid_type="uniform",
                         f_dist=None, adapt_iter=5):
    """统一接口: 创建速度空间网格.

    Parameters
    ----------
    v_min, v_max : float  速度范围
    N : int  网格点数
    grid_type : str  "uniform", "log", "voronoi"
    f_dist : callable or None  分布函数 (仅 voronoi 使用)
    adapt_iter : int  自适应迭代次数

    Returns
    -------
    grid_dict : dict  包含 v, dv, voronoi_mesh, distance_stats
    """
    result = {}

    if grid_type == "uniform":
        v, dv = make_uniform_grid(v_min, v_max, N)
        result["v"] = v
        result["dv"] = dv
        result["voronoi_mesh"] = None

    elif grid_type == "log":
        v, dv = make_log_grid(v_min, v_max, N)
        result["v"] = v
        result["dv"] = dv
        result["voronoi_mesh"] = None

    elif grid_type == "voronoi":
        vmesh = VoronoiVelocityMesh(v_min, v_max, N, f_dist=f_dist)
        result["v"] = vmesh.generators
        result["dv"] = vmesh.cell_volumes
        result["voronoi_mesh"] = vmesh

    else:
        raise ValueError(f"未知网格类型: {grid_type}")

    # 距离统计
    result["distance_stats"] = VelocityDistanceStatistics(result["v"])

    return result
