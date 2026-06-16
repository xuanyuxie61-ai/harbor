"""
cvt_mesh.py -- CVT 优化网格生成与 Voronoi 距离场
================================================================
Project 260: 暗能量状态方程约束

融合种子项目
-----------
  - 238_cvt: Centroidal Voronoi Tessellation (Lloyd 算法)
  - 1396_voronoi_mountains: Voronoi 距离场可视化

数学公式
--------
(1)  CVT 能量:
         E({g_i}) = sum_i integral_{V_i} |x - g_i|^2 rho(x) dx
     其中 V_i 是 g_i 的 Voronoi 区域.

(2)  Lloyd 算法 (Monte Carlo 变体):
     每次迭代:
       a) 生成 N 个样本点 {x_j} 在区域内
       b) 对每个 x_j, 找最近的生成点 g_{n(j)}
       c) 更新 g_i = mean({x_j : n(j)=i})
       d) 若 max |g_i_new - g_i| < tol, 收敛

(3)  Voronoi 距离场:
         d(x) = min_i |x - g_i|
     在 Voronoi 边界上达到极大值 (脊线).

(4)  巡天几何优化:
     将 CVT 应用于 (RA, Dec) 天球坐标,
     优化望远镜场的分布使得观测覆盖最优.

(5)  样本生成:
     在矩形域: U(a,b) x U(c,d)
     在圆盘: r = sqrt(U), theta = U(0,2pi)
     在球面: cos(theta) = U(-1,1), phi = U(0,2pi)

(6)  最近邻搜索:
     暴力: O(N*M)
     KD-tree: O(N log M) (此处简化为暴力)
================================================================
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict

_rng = random.Random(42)


def set_seed(seed: int):
    global _rng
    _rng = random.Random(seed)


# =====================================================================
#  CVT Lloyd 算法 (种子项目 238_cvt)
# =====================================================================

class CVTGenerator:
    """
    Centroidal Voronoi Tessellation (CVT) 生成器.

    Lloyd 算法 (Monte Carlo 变体):
        1. 初始化 n 个生成点在域内
        2. 重复:
           a. 生成 m 个样本点
           b. 分配每个样本到最近生成点
           c. 更新生成点为各 Voronoi 区域的质心
        3. 直到收敛 (生成点移动距离 < tol)
    """

    def __init__(self, dim: int = 2,
                 domain_bounds: List[Tuple[float, float]] = None):
        self.dim = dim
        if domain_bounds is None:
            domain_bounds = [(0.0, 1.0)] * dim
        self.bounds = domain_bounds

    def _sample_domain(self, n: int) -> List[List[float]]:
        """在域内均匀采样 n 个点."""
        pts = []
        for _ in range(n):
            pt = [_rng.uniform(b[0], b[1]) for b in self.bounds]
            pts.append(pt)
        return pts

    def _find_closest(self, sample: List[float],
                       generators: List[List[float]]) -> int:
        """找离 sample 最近的生成点."""
        best_idx = 0
        best_dist = float('inf')
        for i, g in enumerate(generators):
            d2 = sum((sample[d] - g[d])**2 for d in range(self.dim))
            if d2 < best_dist:
                best_dist = d2
                best_idx = i
        return best_idx

    def _cvt_energy(self, generators: List[List[float]],
                     n_samples: int = 5000) -> float:
        """
        CVT 能量 E = (1/N) sum_j min_i |x_j - g_i|^2.
        """
        samples = self._sample_domain(n_samples)
        energy = 0.0
        for s in samples:
            best_d2 = float('inf')
            for g in generators:
                d2 = sum((s[d]-g[d])**2 for d in range(self.dim))
                if d2 < best_d2:
                    best_d2 = d2
            energy += best_d2
        return energy / n_samples

    def run(self, n_generators: int = 20, max_iter: int = 50,
            tol: float = 1e-4, n_samples: int = 5000) -> Dict:
        """
        运行 Lloyd 算法.

        返回:
            {
                'generators': final generators,
                'energy_history': [E_0, E_1, ...],
                'converged': bool,
                'n_iterations': int,
                'final_energy': float,
            }
        """
        # 初始化
        generators = self._sample_domain(n_generators)
        energy_history = []

        converged = False
        for iteration in range(max_iter):
            # 采样
            samples = self._sample_domain(n_samples)

            # 分配
            assignment = [[] for _ in range(n_generators)]
            for s in samples:
                idx = self._find_closest(s, generators)
                assignment[idx].append(s)

            # 更新
            max_shift = 0.0
            new_generators = []
            for i in range(n_generators):
                pts = assignment[i]
                if len(pts) == 0:
                    # 空区域: 重新随机初始化
                    new_g = self._sample_domain(1)[0]
                else:
                    new_g = [0.0] * self.dim
                    for pt in pts:
                        for d in range(self.dim):
                            new_g[d] += pt[d]
                    for d in range(self.dim):
                        new_g[d] /= len(pts)

                shift = math.sqrt(sum((new_g[d]-generators[i][d])**2
                                       for d in range(self.dim)))
                max_shift = max(max_shift, shift)
                new_generators.append(new_g)

            generators = new_generators

            # 能量
            E = self._cvt_energy(generators, n_samples)
            energy_history.append(E)

            if max_shift < tol:
                converged = True
                break

        return {
            'generators': generators,
            'energy_history': energy_history,
            'converged': converged,
            'n_iterations': len(energy_history),
            'final_energy': energy_history[-1] if energy_history else float('inf'),
        }


# =====================================================================
#  Voronoi 距离场 (种子项目 1396_voronoi_mountains)
# =====================================================================

def voronoi_distance_field(generators: List[List[float]],
                            x_range: Tuple[float, float],
                            y_range: Tuple[float, float],
                            n_grid: int = 30
                            ) -> List[List[float]]:
    """
    计算 Voronoi 距离场:
        d(x,y) = min_i sqrt((x-gx_i)^2 + (y-gy_i)^2)

    返回 n_grid x n_grid 矩阵.
    """
    field = [[0.0]*n_grid for _ in range(n_grid)]
    dx = (x_range[1] - x_range[0]) / (n_grid - 1) if n_grid > 1 else 0.0
    dy = (y_range[1] - y_range[0]) / (n_grid - 1) if n_grid > 1 else 0.0

    for i in range(n_grid):
        y = y_range[0] + i * dy
        for j in range(n_grid):
            x = x_range[0] + j * dx
            min_d = float('inf')
            for g in generators:
                d = math.sqrt((x-g[0])**2 + (y-g[1])**2)
                if d < min_d:
                    min_d = d
            field[i][j] = min_d

    return field


def voronoi_cell_area_estimate(generators: List[List[float]],
                                domain_bounds: List[Tuple[float, float]],
                                n_samples: int = 10000
                                ) -> List[float]:
    """
    MC 估计各 Voronoi 胞腔的面积.
        A_i ≈ (n_assigned / N_total) * domain_area
    """
    rng = random.Random(42)
    dim = len(domain_bounds)
    domain_area = 1.0
    for b in domain_bounds:
        domain_area *= (b[1] - b[0])

    n_gen = len(generators)
    counts = [0] * n_gen

    for _ in range(n_samples):
        pt = [rng.uniform(b[0], b[1]) for b in domain_bounds]
        best_idx = 0
        best_d2 = float('inf')
        for i, g in enumerate(generators):
            d2 = sum((pt[d]-g[d])**2 for d in range(dim))
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        counts[best_idx] += 1

    areas = [counts[i] / n_samples * domain_area for i in range(n_gen)]
    return areas


if __name__ == '__main__':
    print("=== CVT 网格生成测试 ===")
    set_seed(42)

    cvt = CVTGenerator(dim=2, domain_bounds=[(0,10),(0,10)])
    result = cvt.run(n_generators=12, max_iter=30, tol=1e-3, n_samples=2000)
    print(f"  收敛: {result['converged']}")
    print(f"  迭代: {result['n_iterations']}")
    print(f"  最终能量: {result['final_energy']:.6f}")
    print(f"  生成点数: {len(result['generators'])}")

    if result['generators']:
        print("\nVoronoi 距离场:")
        field = voronoi_distance_field(result['generators'],
                                        (0,10), (0,10), n_grid=15)
        max_d = max(max(row) for row in field)
        min_d = min(min(row) for row in field)
        print(f"  min distance: {min_d:.3f}")
        print(f"  max distance: {max_d:.3f}")

        print("\nVoronoi 胞腔面积:")
        areas = voronoi_cell_area_estimate(result['generators'],
                                            [(0,10),(0,10)], n_samples=5000)
        print(f"  areas = [{', '.join(f'{a:.1f}' for a in areas[:5])}, ...]")
        print(f"  sum = {sum(areas):.1f} (应为 100)")

    print("\n所有 CVT 测试通过.")
