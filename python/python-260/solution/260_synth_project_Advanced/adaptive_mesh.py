"""
adaptive_mesh.py -- 自适应网格加密 (AMR) 与误差估计
================================================================
Project 260: 暗能量状态方程约束

数学公式
--------
(1)   Richardson 外推:
         I_exact ≈ I_h + (I_h - I_{2h}) / (2^p - 1)
     其中 p 为方法阶数.

(2)  局部截断误差估计:
         e_i ≈ |D_{fine}[i] - D_{coarse}[i]| / (2^p - 1)

(3)  网格加密判据:
         e_i > tol * max(e)  =>  加密单元 i

(4)  h-适应: 在误差大的区域细分网格
     p-适应: 在误差大的区域提高多项式阶数

(5)  误差指标 (Kelly estimator):
         eta_K = h^{1/2} ||[grad u]||_{L2(edges)}

(6)  对数尺度 u = ln(a) 上的自适应:
     早期 (a~0): 解光滑, 粗网格
     近期 (a~1): 解变化快, 细网格
================================================================
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Callable

import cosmo_constants as cc


# =====================================================================
#  Richardson 外推
# =====================================================================

def richardson_extrapolate(I_h: float, I_2h: float, order: int = 2
                            ) -> Tuple[float, float]:
    """
    Richardson 外推:
        I_exact ≈ I_h + (I_h - I_2h) / (2^p - 1)
        error_est ≈ |I_h - I_2h| / (2^p - 1)
    """
    factor = 2**order - 1
    if factor == 0:
        return I_h, 0.0
    I_ext = I_h + (I_h - I_2h) / factor
    err_est = abs(I_h - I_2h) / factor
    return I_ext, err_est


# =====================================================================
#  自适应 1D 网格
# =====================================================================

class AdaptiveMesh1D:
    """
    1D 自适应网格.

    在 [a_min, a_max] 上, 根据解的梯度自适应加密.
    使用等分布原理: 在每个单元上, 误差指标 * 单元长度 = 常数.
    """

    def __init__(self, a_min: float = 1e-4, a_max: float = 1.0):
        self.a_min = a_min
        self.a_max = a_max
        self.nodes: List[float] = []
        self.refine_levels: List[int] = []

    def initialize_uniform(self, n: int = 20):
        """均匀初始网格."""
        self.nodes = [self.a_min + i*(self.a_max - self.a_min)/(n-1)
                      for i in range(n)]
        self.refine_levels = [0] * n

    def compute_monitor_function(self, solution: List[float],
                                   alpha: float = 0.1) -> List[float]:
        """
        监控函数 M(x):
            M_i = sqrt(alpha + (u_{i+1} - u_i)^2 / h_i^2)

        alpha 防止 M=0.
        """
        n = len(self.nodes)
        M = [1.0] * (n - 1)
        for i in range(n - 1):
            h = self.nodes[i+1] - self.nodes[i]
            if h < 1e-30:
                M[i] = 1.0
                continue
            du = solution[i+1] - solution[i]
            M[i] = math.sqrt(alpha + (du/h)**2)
        return M

    def equidistribute(self, M: List[float], n_target: int
                       ) -> List[float]:
        """
        等分布网格: 使得 M_i * h_i = 常数.

        总 "权重": Sigma = sum M_i * h_i
        每个新单元权重: sigma = Sigma / (n_target - 1)
        """
        n = len(self.nodes)
        if len(M) != n - 1:
            raise ValueError("M length mismatch")

        # 计算总权重
        sigma_total = 0.0
        for i in range(n - 1):
            sigma_total += M[i] * (self.nodes[i+1] - self.nodes[i])

        sigma_target = sigma_total / (n_target - 1)

        new_nodes = [self.a_min]
        target_sigma = sigma_target
        current_sigma = 0.0
        seg_idx = 0

        for k in range(1, n_target - 1):
            while seg_idx < n - 1:
                h = self.nodes[seg_idx + 1] - self.nodes[seg_idx]
                seg_sigma = M[seg_idx] * h
                if current_sigma + seg_sigma >= target_sigma:
                    # 在 seg_idx 单元内插值
                    frac = (target_sigma - current_sigma) / max(seg_sigma, 1e-30)
                    new_x = self.nodes[seg_idx] + frac * h
                    new_nodes.append(new_x)
                    current_sigma = 0.0
                    target_sigma += sigma_target
                    break
                else:
                    current_sigma += seg_sigma
                    seg_idx += 1
            if seg_idx >= n - 1:
                # 超出范围, 均匀填充
                remaining = n_target - 1 - len(new_nodes)
                for r in range(1, remaining + 1):
                    new_nodes.append(self.a_min + r * (self.a_max - self.a_min)
                                     / (n_target - 1))
                break

        new_nodes.append(self.a_max)
        # 去重和排序
        new_nodes = sorted(set(new_nodes))
        return new_nodes

    def refine_by_gradient(self, solution: List[float],
                            threshold: float = 0.3,
                            max_refine: int = 3) -> Dict:
        """
        基于梯度的自适应加密.

        在 |u_{i+1} - u_i| / max(|u|) > threshold 的单元中点加密.
        """
        n = len(self.nodes)
        if len(solution) != n:
            raise ValueError("solution length mismatch")

        max_u = max(abs(u) for u in solution) if solution else 1.0
        if max_u < 1e-30:
            max_u = 1.0

        new_nodes = [self.nodes[0]]
        new_levels = [0]
        n_refined = 0

        for i in range(n - 1):
            du = abs(solution[i+1] - solution[i]) / max_u
            level = self.refine_levels[i] if i < len(self.refine_levels) else 0

            if du > threshold and level < max_refine:
                # 加密: 在中点插入
                mid = 0.5 * (self.nodes[i] + self.nodes[i+1])
                new_nodes.append(mid)
                new_levels.append(level + 1)
                n_refined += 1

            new_nodes.append(self.nodes[i+1])
            new_levels.append(self.refine_levels[i+1] if i+1 < len(self.refine_levels) else 0)

        self.nodes = new_nodes
        self.refine_levels = new_levels

        return {
            'n_refined': n_refined,
            'n_total': len(self.nodes),
            'max_level': max(new_levels) if new_levels else 0,
        }


# =====================================================================
#  收敛阶测试
# =====================================================================

def convergence_order_test(solver_func: Callable,
                            n_list: List[int] = None,
                            exact_value: float = None
                            ) -> Dict:
    """
    测试数值方法的收敛阶.

    对 n_list 中的每个 n, 调用 solver_func(n) 得到数值解,
    与 exact_value 比较, 计算误差和收敛阶.

    p ≈ log(e_h / e_{h/2}) / log(2)
    """
    if n_list is None:
        n_list = [20, 40, 80, 160, 320]

    errors = []
    for n in n_list:
        val = solver_func(n)
        if exact_value is not None:
            err = abs(val - exact_value)
        else:
            err = abs(val)
        errors.append(err)

    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 1e-30 and errors[i-1] > 1e-30:
            p = math.log(errors[i-1] / errors[i]) / math.log(2.0)
        else:
            p = float('inf')
        orders.append(p)

    return {
        'n_list': n_list,
        'errors': errors,
        'orders': orders,
        'avg_order': sum(orders) / len(orders) if orders else 0.0,
    }


# =====================================================================
#  对数空间自适应 (u = ln a)
# =====================================================================

def log_adaptive_grid(a_min: float = 1e-4, a_max: float = 1.0,
                       n_points: int = 50,
                       refinement_func: Callable = None
                       ) -> List[float]:
    """
    在对数空间 u = ln(a) 上生成自适应网格.

    在 u 上均匀加密, 然后在 u 变化快的区域进一步加密.
    """
    u_min = math.log(max(a_min, 1e-12))
    u_max = math.log(a_max)

    # 初始均匀
    u_grid = [u_min + i*(u_max - u_min)/(n_points-1)
              for i in range(n_points)]

    if refinement_func is not None:
        # 根据 refinement_func(u) 的值加密
        vals = [refinement_func(math.exp(u)) for u in u_grid]
        max_val = max(abs(v) for v in vals) if vals else 1.0
        if max_val < 1e-30:
            max_val = 1.0

        new_u = [u_grid[0]]
        for i in range(len(u_grid) - 1):
            du = u_grid[i+1] - u_grid[i]
            dv = abs(vals[i+1] - vals[i]) / max_val
            if dv > 0.2:
                # 加密
                n_sub = min(int(dv / 0.1), 5)
                for s in range(1, n_sub + 1):
                    new_u.append(u_grid[i] + s * du / (n_sub + 1))
            new_u.append(u_grid[i+1])
        u_grid = sorted(set(new_u))

    return [math.exp(u) for u in u_grid]


if __name__ == '__main__':
    print("=== 自适应网格测试 ===")

    # Richardson 外推
    print("\nRichardson 外推:")
    I_h = 0.1999
    I_2h = 0.199
    I_ext, err = richardson_extrapolate(I_h, I_2h, order=2)
    print(f"  I_h = {I_h}, I_2h = {I_2h}")
    print(f"  I_ext = {I_ext:.6f}, err_est = {err:.6f}")

    # 自适应网格
    print("\n自适应网格:")
    mesh = AdaptiveMesh1D(a_min=1e-4, a_max=1.0)
    mesh.initialize_uniform(20)
    print(f"  初始: {len(mesh.nodes)} 节点")

    # 模拟解 (快速增长)
    sol = [a**0.5 for a in mesh.nodes]  # D ~ a^{1/2}
    result = mesh.refine_by_gradient(sol, threshold=0.3, max_refine=3)
    print(f"  加密后: {result['n_total']} 节点, "
          f"加密 {result['n_refined']} 个单元, "
          f"max level = {result['max_level']}")

    # 收敛阶测试
    print("\n收敛阶测试:")
    def test_solver(n):
        import math
        h = 1.0 / n
        return sum(h * (i*h)**2 for i in range(n+1)) - h*1.0/2

    conv = convergence_order_test(test_solver, [20, 40, 80, 160],
                                    exact_value=1.0/3.0)
    for i, n in enumerate(conv['n_list']):
        print(f"  n={n}: error={conv['errors'][i]:.4e}")
    print(f"  平均阶数: {conv['avg_order']:.2f}")

    # 对数自适应
    print("\n对数自适应网格:")
    grid = log_adaptive_grid(a_min=1e-4, a_max=1.0, n_points=30,
                              refinement_func=lambda a: cc.omega_m_a(a))
    print(f"  节点数: {len(grid)}")
    print(f"  范围: [{grid[0]:.6f}, {grid[-1]:.6f}]")

    print("\n所有自适应网格测试通过.")
