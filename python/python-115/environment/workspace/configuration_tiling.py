"""
configuration_tiling.py
构型空间铺砌与采样模块

核心功能：
- 基于多格拼板（Pentomino）思想的构型空间离散化
- 构型空间区域的旋转、反射对称操作
- 构型空间体积的粗粒化采样
- 反应路径的网格化表示

科学背景：
在酶催化过渡态搜索中，构型空间是 3N-6 维的。直接采样不可行。
本模块借鉴 Pentomino 拼板问题的离散化思想：
    - 将高维构型空间投影到关键反应坐标上
    - 用多格拼板（polyomino）覆盖可及区域
    - 每个拼板单元代表一个构型族

构型空间的可及区域可表示为：
    Ω = {q ∈ R^{3N} | V(q) < V_cutoff 且 |q_i - q_j| > σ_{ij}}

在二维反应坐标投影 (ξ_1, ξ_2) 中，可及区域常呈多边形：
    - 反应物盆地：低能量多边形区域
    - 产物盆地：另一低能量区域
    - 过渡态：两盆地间的鞍点通道

Pentomino（五格拼板）的 12 种形状对应于 12 种基本构型变换模式：
    F, I, L, N, P, T, U, V, W, X, Y, Z

在分子对称性中，这些对应于点群操作的不同组合。
"""

import numpy as np


class PentominoShapes:
    """
    12 种标准 Pentomino 形状定义
    每种形状为 5 个单位方格的连通组合
    """

    SHAPES = {
        'F': np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]]),
        'I': np.array([[1, 1, 1, 1, 1]]),
        'L': np.array([[0, 0, 0, 1], [1, 1, 1, 1]]),
        'N': np.array([[1, 1, 0, 0], [0, 1, 1, 1]]),
        'P': np.array([[1, 1], [1, 1], [1, 0]]),
        'T': np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
        'U': np.array([[1, 0, 1], [1, 1, 1]]),
        'V': np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]]),
        'W': np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
        'X': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
        'Y': np.array([[0, 0, 1, 0], [1, 1, 1, 1]]),
        'Z': np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
    }

    @classmethod
    def get_shape(cls, name):
        name = name.upper()
        if name not in cls.SHAPES:
            raise ValueError(f"未知 Pentomino 形状: {name}")
        return cls.SHAPES[name].copy()

    @classmethod
    def all_shapes(cls):
        return {k: v.copy() for k, v in cls.SHAPES.items()}


class ConfigurationTiling:
    """
    构型空间铺砌采样器
    """

    def __init__(self, xi_range, eta_range, n_xi=20, n_eta=20):
        """
        参数：
            xi_range: (xi_min, xi_max) 反应坐标范围
            eta_range: (eta_min, eta_max) 正交坐标范围
            n_xi, n_eta: 网格数
        """
        self.xi_min, self.xi_max = xi_range
        self.eta_min, self.eta_max = eta_range
        self.n_xi = n_xi
        self.n_eta = n_eta
        self.dxi = (self.xi_max - self.xi_min) / n_xi
        self.deta = (self.eta_max - self.eta_min) / n_eta

    def grid_to_physical(self, i, j):
        """网格索引到物理坐标"""
        xi = self.xi_min + (i + 0.5) * self.dxi
        eta = self.eta_min + (j + 0.5) * self.deta
        return xi, eta

    def physical_to_grid(self, xi, eta):
        """物理坐标到网格索引"""
        i = int((xi - self.xi_min) / self.dxi)
        j = int((eta - self.eta_min) / self.deta)
        i = max(0, min(i, self.n_xi - 1))
        j = max(0, min(j, self.n_eta - 1))
        return i, j

    def tile_coverage(self, energy_func, energy_cutoff):
        """
        用多格拼板覆盖低能构型区域

        算法：
            1. 计算每个网格点的能量
            2. 标记能量 < cutoff 的网格为可及
            3. 用 Pentomino 形状尽可能覆盖可及区域
            4. 返回覆盖率和采样点坐标
        """
        accessible = np.zeros((self.n_xi, self.n_eta), dtype=bool)
        energy_grid = np.zeros((self.n_xi, self.n_eta), dtype=float)

        for i in range(self.n_xi):
            for j in range(self.n_eta):
                xi, eta = self.grid_to_physical(i, j)
                e = energy_func(xi, eta)
                energy_grid[i, j] = e
                if e < energy_cutoff:
                    accessible[i, j] = True

        # 用 T-形拼板（代表过渡态通道）进行覆盖采样
        samples = []
        coverage = accessible.copy()

        # 简单贪婪覆盖：在可及区域内放置 T-形拼板
        t_shape = PentominoShapes.get_shape('T')
        sh, sw = t_shape.shape

        for i in range(self.n_xi - sw + 1):
            for j in range(self.n_eta - sh + 1):
                if np.all(coverage[i:i + sw, j:j + sh]):
                    # 记录拼板中心采样点
                    cx = i + sw // 2
                    cy = j + sh // 2
                    xi_s, eta_s = self.grid_to_physical(cx, cy)
                    samples.append((xi_s, eta_s, energy_grid[cx, cy]))
                    coverage[i:i + sw, j:j + sh] = False

        coverage_ratio = np.sum(accessible) / (self.n_xi * self.n_eta)
        return coverage_ratio, samples, energy_grid

    def rotate_tile(self, tile, k):
        """
        将拼板逆时针旋转 k*90 度

        旋转矩阵：
            R(θ) = [cosθ  -sinθ]
                   [sinθ   cosθ]
            θ = k*π/2
        """
        return np.rot90(tile, k=k)

    def reflect_tile(self, tile):
        """沿 x 轴反射拼板"""
        return np.fliplr(tile)


class ConfigurationSpaceSampler:
    """
    构型空间系统采样器
    基于多格拼板思想的粗粒化采样
    """

    def __init__(self, n_atoms, temperature=300.0):
        self.n_atoms = n_atoms
        self.kB = 0.0019872041
        self.T = temperature
        self.beta = 1.0 / (self.kB * temperature)

    def metropolis_sampling(self, energy_func, x0, n_steps=1000, step_size=0.1):
        """
        Metropolis Monte Carlo 采样

        接受准则：
            min(1, exp(-βΔE))
        """
        x = np.asarray(x0, dtype=float).copy()
        e_curr = energy_func(x)
        samples = [x.copy()]
        energies = [e_curr]
        n_accept = 0

        for _ in range(n_steps):
            x_trial = x + np.random.randn(len(x)) * step_size
            e_trial = energy_func(x_trial)
            delta_e = e_trial - e_curr

            if delta_e < 0 or np.random.rand() < np.exp(-self.beta * delta_e):
                x = x_trial
                e_curr = e_trial
                n_accept += 1

            samples.append(x.copy())
            energies.append(e_curr)

        acceptance_ratio = n_accept / n_steps
        return np.array(samples), np.array(energies), acceptance_ratio

    def reaction_path_sampling(self, energy_func, x_reactant, x_product, n_images=20):
        """
        线性插值生成初始反应路径图像

        路径参数化：
            x(λ) = (1-λ) x_R + λ x_P,   λ ∈ [0,1]
        """
        x_R = np.asarray(x_reactant, dtype=float)
        x_P = np.asarray(x_product, dtype=float)
        lambdas = np.linspace(0, 1, n_images)
        path = []
        energies = []

        for lam in lambdas:
            x_img = (1.0 - lam) * x_R + lam * x_P
            path.append(x_img)
            energies.append(energy_func(x_img))

        return np.array(path), np.array(energies), lambdas
